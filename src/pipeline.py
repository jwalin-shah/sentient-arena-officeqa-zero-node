from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any

from src.backends import run_openhands_backend, run_openhands_grounding_judge, run_openhands_parse
from src.corpus import build_retrieval_query, load_transformed_txt_context, parse_source_files
from src.evaluator import compare_numeric, normalize_numeric_prediction, parse_numeric
from src.failure import classify_failure
from src.stages.calculate import calculate_answer
from src.stages.extract import extract_values
from src.stages.finalize import finalize_answer
from src.stages.parse import parse_question
from src.stages.retrieve import select_evidence
from src.stages.verify import evaluate_grounding, verify_result
from src.types import (
    AttemptResult,
    Calculation,
    EvidenceRow,
    ExampleResult,
    Extraction,
    FinalDecision,
    ParsedQuestion,
    StageOutcome,
    Verification,
)


def _strict_numeric_gate(
    *,
    strict_eval_mode: bool,
    numeric_correct: bool,
    parse_valid: bool,
    solve_valid: bool,
    row_presence_gate: bool,
    row_grounding_gate: bool,
    verification: Verification,
) -> bool:
    if not strict_eval_mode:
        return bool(numeric_correct)
    return bool(
        numeric_correct
        and parse_valid
        and solve_valid
        and row_presence_gate
        and row_grounding_gate
        and verification.evidence_sufficiency_check
    )


def _failed_families(*, parse_valid: bool, solve_valid: bool, numeric_correct: bool, grounding_checks: list[dict[str, Any]], verification: Verification) -> set[str]:
    failures: set[str] = set()
    if not parse_valid:
        failures.add("parse")
    if not solve_valid:
        failures.add("solve")
    grounded_rows_ok = bool(grounding_checks) and all(c.get("matched") for c in grounding_checks)
    if not grounded_rows_ok or not verification.evidence_sufficiency_check:
        failures.add("evidence")
    if not numeric_correct:
        failures.add("arithmetic")
    return failures


def _build_repair_brief(*, attempt_idx: int, schema_validation: dict[str, Any], grounding_checks: list[dict[str, Any]], eval_details: dict[str, Any], scores: dict[str, Any], final_answer: Any, expected_answer: Any) -> dict[str, Any]:
    failing_rows = [c for c in grounding_checks if not c.get("matched")]
    next_iteration_instructions: list[str] = []
    parse_errors = schema_validation.get("parse_errors", []) or []
    solve_errors = schema_validation.get("solve_errors", []) or []
    if parse_errors:
        next_iteration_instructions.append(
            "Return parse JSON with exactly required keys and valid canonical operation."
        )
    if solve_errors:
        next_iteration_instructions.append(
            "Return solve JSON with required evidence_rows and all required row fields, including matched_snippet."
        )
    if failing_rows:
        next_iteration_instructions.append(
            "Use only metadata.source_files and copy row_label, column_label, and raw_value as literal strings from provided corpus chunks."
        )
    rel = eval_details.get("relative_error")
    if isinstance(rel, (int, float)):
        if rel > 0.01:
            next_iteration_instructions.append(
                "Recompute arithmetic from extracted values and output final_answer as a numeric string."
            )
    if not next_iteration_instructions:
        next_iteration_instructions.append("Keep the same strategy and preserve grounded evidence rows.")

    return {
        "attempt": attempt_idx,
        "schema_failures": {
            "parse_errors": parse_errors,
            "solve_errors": solve_errors,
        },
        "grounding_failures": [
            {
                "row_index": r.get("row_index"),
                "source_file": r.get("source_file"),
                "reason_codes": r.get("reason_codes", []),
            }
            for r in failing_rows
        ],
        "numeric_mismatch": {
            "predicted": final_answer,
            "expected": expected_answer,
            "predicted_numeric": eval_details.get("predicted_numeric"),
            "expected_numeric": eval_details.get("expected_numeric"),
            "relative_error": eval_details.get("relative_error"),
        },
        "gate_state": {
            "numeric_correct": bool(scores.get("numeric_correct", False)),
            "strict_grounded_correct": bool(scores.get("strict_grounded_correct", False)),
        },
        "next_iteration_instructions": next_iteration_instructions,
        "next_iteration_prompt": " | ".join(next_iteration_instructions),
    }


def _extract_int_note(prefix: str, notes: list[str]) -> int:
    for n in notes:
        if isinstance(n, str) and n.startswith(prefix):
            try:
                return int(n.split("=", 1)[1].strip())
            except Exception:
                return 0
    return 0


def _expected_evidence_row_range(
    *,
    parsed: ParsedQuestion,
    quality_cfg: dict[str, Any],
) -> tuple[int, int]:
    op = (parsed.operation or "lookup").strip().lower()
    op_required = (quality_cfg.get("required_rows_by_operation", {}) or {})
    if isinstance(op_required, dict) and op in op_required:
        required_rows = int(op_required.get(op, 1))
    elif op == "lookup":
        required_rows = 1
    elif op in {"difference", "ratio"}:
        required_rows = 2
    elif op == "average":
        required_rows = 2
    else:
        required_rows = 1

    hard_cap = int(quality_cfg.get("max_evidence_rows", max(4, required_rows)))
    min_rows = max(1, required_rows)
    max_rows = max(min_rows, hard_cap)
    max_rows = max(1, min(max_rows, hard_cap))
    min_rows = max(1, min(min_rows, max_rows))
    return min_rows, max_rows


def _filter_evidence_rows_for_quality(
    *,
    rows: list[EvidenceRow],
    raw_question: str,
    metadata: dict[str, Any],
    parsed: ParsedQuestion,
    corpus_chunks: list[dict[str, str]],
    max_rows: int,
    min_rows: int,
    min_signal_score: int,
    require_full_grounding: bool,
    llm_grounded_by_index: dict[int, bool] | None = None,
) -> tuple[list[EvidenceRow], list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], [], {"input_count": 0, "dedup_count": 0, "kept_count": 0, "dropped_count": 0}

    deduped: list[EvidenceRow] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for r in rows:
        key = (
            str(r.source_file).strip(),
            str(r.table_or_section).strip(),
            str(r.row_label).strip(),
            str(r.column_label).strip(),
            str(r.raw_value).strip(),
            str(r.unit).strip(),
            str(r.matched_snippet).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    checks = evaluate_grounding(
        raw_question=raw_question,
        metadata=metadata,
        parsed=parsed,
        evidence_rows=deduped,
        corpus_chunks=corpus_chunks,
    )

    scored: list[tuple[int, int, int, EvidenceRow, dict[str, Any]]] = []
    for idx, (row, chk) in enumerate(zip(deduped, checks)):
        llm_grounded = bool((llm_grounded_by_index or {}).get(idx, False))
        score = 0
        for k in [
            "source_file_allowed",
            "raw_value_found",
            "row_label_found",
            "column_label_found",
            "matched_snippet_found",
            "year_aligned",
            "topic_aligned",
        ]:
            if chk.get(k):
                score += 1
        if llm_grounded:
            score += 2
        scored.append((score, -idx, idx, row, chk))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    kept_rows: list[EvidenceRow] = []
    kept_checks: list[dict[str, Any]] = []
    row_diagnostics: list[dict[str, Any]] = []
    for score, _, row_index, row, chk in scored:
        llm_grounded = bool((llm_grounded_by_index or {}).get(row_index, False))
        if len(kept_rows) >= max(1, max_rows):
            row_diagnostics.append(
                {
                    "kept": False,
                    "drop_reason": "over_max_rows",
                    "score": score,
                    "llm_grounded": llm_grounded,
                    "reason_codes": chk.get("reason_codes", []),
                    "row": row.__dict__,
                }
            )
            continue
        if require_full_grounding:
            keep = bool(chk.get("matched")) or llm_grounded
        else:
            keep = bool(chk.get("matched")) or llm_grounded or score >= min_signal_score
        if keep:
            kept_rows.append(row)
            kept_checks.append(chk)
            row_diagnostics.append(
                {
                    "kept": True,
                    "drop_reason": "",
                    "score": score,
                    "llm_grounded": llm_grounded,
                    "reason_codes": chk.get("reason_codes", []),
                    "row": row.__dict__,
                }
            )
        else:
            row_diagnostics.append(
                {
                    "kept": False,
                    "drop_reason": "insufficient_grounding_signal",
                    "score": score,
                    "llm_grounded": llm_grounded,
                    "reason_codes": chk.get("reason_codes", []),
                    "row": row.__dict__,
                }
            )

    if len(kept_rows) < max(1, min_rows):
        row_diagnostics = [
            {
                **d,
                "kept": False,
                "drop_reason": ("below_min_rows_after_filter" if d.get("kept") else d.get("drop_reason")),
            }
            for d in row_diagnostics
        ]
        kept_rows = []
        kept_checks = []

    stats = {
        "input_count": len(rows),
        "dedup_count": len(deduped),
        "kept_count": len(kept_rows),
        "dropped_count": len(deduped) - len(kept_rows),
        "min_rows": min_rows,
        "max_rows": max_rows,
        "min_signal_score": min_signal_score,
        "require_full_grounding": require_full_grounding,
        "row_diagnostics": row_diagnostics,
    }
    return kept_rows, kept_checks, stats


def process_example(example: dict[str, Any], config: dict[str, Any]) -> ExampleResult:
    qid = str(example.get("question_id", "unknown"))
    raw_question = str(example.get("question", ""))
    expected_answer = example.get("expected_answer")
    metadata = example.get("metadata") or {}

    backend_name = str((config.get("backend", {}) or {}).get("name", "local_stub"))
    parse_mode = str((config.get("pipeline", {}) or {}).get("parse_mode", "rule_based"))
    strict_eval_mode = bool((config.get("pipeline", {}) or {}).get("strict_eval_mode", False))
    max_attempts = int((config.get("pipeline", {}) or {}).get("max_attempts", 3))
    stop_on_strict_pass = bool((config.get("pipeline", {}) or {}).get("stop_on_strict_pass", True))
    retry_failed_stages_only = bool((config.get("pipeline", {}) or {}).get("retry_failed_stages_only", True))

    decision_cfg = config.get("decision", {}) or {}
    final_gate = str(decision_cfg.get("final_gate", "strict_plus_numeric"))

    corpus_cfg = config.get("corpus", {}) or {}
    corpus_format = str(corpus_cfg.get("format", "transformed_txt"))
    corpus_root = Path(corpus_cfg.get("root_dir", "external/officeqa/treasury_bulletins_parsed/transformed")).resolve()
    source_files = parse_source_files(metadata.get("source_files", ""))
    if corpus_format != "transformed_txt":
        corpus_ctx = load_transformed_txt_context(source_files=[], corpus_root=corpus_root)
    else:
        corpus_ctx = load_transformed_txt_context(
            source_files=source_files,
            corpus_root=corpus_root,
            max_chunks_per_file=int(corpus_cfg.get("max_chunks_per_file", 2)),
            max_chars=int(corpus_cfg.get("max_chars_per_chunk", 5000)),
            query=raw_question,
            max_files_from_year_fallback=int(corpus_cfg.get("max_files_from_year_fallback", 2)),
            max_total_chunks=int(corpus_cfg.get("max_total_chunks", 0)),
        )

    parsed = ParsedQuestion()
    evidence_candidates = []
    chosen_evidence = None
    extraction = Extraction()
    calculation = Calculation()
    verification = Verification()
    evidence_rows: list[EvidenceRow] = []
    final_answer: Any = None

    parse_valid = False
    solve_valid = False
    parse_errors: list[str] = []
    solve_errors: list[str] = []

    attempts: list[AttemptResult] = []
    snapshots: list[dict[str, Any]] = []
    previous_failed_families = {"parse", "solve", "evidence", "arithmetic"}
    repair_brief: dict[str, Any] | None = None

    for attempt_idx in range(1, max_attempts + 1):
        attempt_notes: list[str] = []
        model_trace: dict[str, Any] = {
            "backend_request": {},
            "backend_response_raw": "",
            "backend_response_structured": {},
            "stage_trace": {
                "attempt_index": attempt_idx,
                "parsed_question": {},
                "chosen_evidence": {},
                "extraction": {},
                "calculation": {},
                "verification": {},
                "grounding_checks": [],
                "corpus": {
                    "requested_files": corpus_ctx.requested_files,
                    "loaded_files": corpus_ctx.loaded_files,
                    "missing_files": corpus_ctx.missing_files,
                    "corpus_root": str(corpus_root),
                },
            },
        }
        stage_outcomes: dict[str, StageOutcome] = {
            "parse": StageOutcome(),
            "retrieve": StageOutcome(),
            "extract": StageOutcome(),
            "calculate": StageOutcome(),
            "solve": StageOutcome(),
            "grounding_precheck": StageOutcome(),
            "verify": StageOutcome(),
            "decision": StageOutcome(),
        }
        row_presence_gate = False
        row_grounding_gate = False
        expected_min_rows = 1
        expected_max_rows = 4

        rerun_parse = attempt_idx == 1 or (not retry_failed_stages_only) or ("parse" in previous_failed_families)
        rerun_support = attempt_idx == 1 or (not retry_failed_stages_only) or rerun_parse
        rerun_solve = (
            attempt_idx == 1
            or (not retry_failed_stages_only)
            or rerun_parse
            or ("solve" in previous_failed_families)
            or ("evidence" in previous_failed_families)
            or ("arithmetic" in previous_failed_families)
        )

        parse_errors = [] if rerun_parse else parse_errors
        solve_errors = [] if rerun_solve else solve_errors

        if corpus_ctx.missing_files and strict_eval_mode:
            attempt_notes.append("strict_missing_source_file=true")
            solve_errors.append("missing_source_file")

        if rerun_parse:
            parse_t0 = time.perf_counter()
            if parse_mode == "llm_structured" and backend_name == "openhands_sdk":
                llm_parsed, parse_trace, parse_notes = run_openhands_parse(
                    raw_question=raw_question,
                    metadata=metadata,
                    corpus_chunks=corpus_ctx.chunks,
                    config=config,
                    project_root=Path(__file__).resolve().parents[1],
                    repair_brief=repair_brief,
                    attempt_index=attempt_idx,
                )
                attempt_notes.extend(parse_notes)
                model_trace["backend_request"] = parse_trace.get("backend_request", {})
                model_trace["backend_response_raw"] = parse_trace.get("backend_response_raw", "")
                model_trace["backend_response_structured"] = parse_trace.get("backend_response_structured", {})
                model_trace["stage_trace"]["parse_backend_request"] = parse_trace.get("backend_request", {})
                model_trace["stage_trace"]["parse_backend_response_raw"] = parse_trace.get("backend_response_raw", "")
                model_trace["stage_trace"]["parse_backend_response_structured"] = parse_trace.get("backend_response_structured", {})
                if llm_parsed is not None:
                    parsed = llm_parsed
                    parse_valid = True
                    stage_outcomes["parse"] = StageOutcome(
                        status="pass",
                        artifacts=parsed.__dict__,
                        timing_ms=(time.perf_counter() - parse_t0) * 1000.0,
                    )
                else:
                    parse_valid = False
                    parse_errors.append("invalid_structured_parse")
                    if strict_eval_mode:
                        attempt_notes.append("strict_parse_failed=true")
                        parsed = ParsedQuestion()
                    else:
                        attempt_notes.append("parse_fallback_used=true")
                        parsed = parse_question(raw_question)
                        parse_valid = True
                    stage_outcomes["parse"] = StageOutcome(
                        status="fail",
                        errors=parse_errors.copy(),
                        artifacts=parsed.__dict__,
                        timing_ms=(time.perf_counter() - parse_t0) * 1000.0,
                    )
            else:
                parsed = parse_question(raw_question)
                parse_valid = True
                stage_outcomes["parse"] = StageOutcome(
                    status="pass",
                    artifacts=parsed.__dict__,
                    timing_ms=(time.perf_counter() - parse_t0) * 1000.0,
                )
        else:
            stage_outcomes["parse"] = StageOutcome(status="skip", artifacts=parsed.__dict__)

        model_trace["stage_trace"]["parsed_question"] = parsed.__dict__

        # Retrieval is guided by parsed fields so the model sees targeted evidence
        # instead of generic high-volume context.
        if corpus_format == "transformed_txt" and (rerun_support or rerun_solve):
            retrieval_query = build_retrieval_query(
                raw_question=raw_question,
                parsed_question=(parsed.__dict__ if parse_valid else None),
            )
            corpus_ctx = load_transformed_txt_context(
                source_files=source_files,
                corpus_root=corpus_root,
                max_chunks_per_file=int(corpus_cfg.get("max_chunks_per_file", 2)),
                max_chars=int(corpus_cfg.get("max_chars_per_chunk", 5000)),
                query=retrieval_query,
                max_files_from_year_fallback=int(corpus_cfg.get("max_files_from_year_fallback", 2)),
                max_total_chunks=int(corpus_cfg.get("max_total_chunks", 0)),
            )
            model_trace["stage_trace"]["corpus"] = {
                "requested_files": corpus_ctx.requested_files,
                "loaded_files": corpus_ctx.loaded_files,
                "missing_files": corpus_ctx.missing_files,
                "corpus_root": str(corpus_root),
                "retrieval_query": retrieval_query,
                "chunk_count": len(corpus_ctx.chunks),
            }

        if rerun_support:
            retrieve_t0 = time.perf_counter()
            evidence_candidates, chosen_evidence = select_evidence(qid, parsed, metadata)
            model_trace["stage_trace"]["chosen_evidence"] = {} if chosen_evidence is None else chosen_evidence.__dict__
            chunks_by_file: dict[str, int] = {}
            for c in corpus_ctx.chunks:
                sf = str(c.get("source_file", ""))
                chunks_by_file[sf] = chunks_by_file.get(sf, 0) + 1
            max_chunks_per_file = int(corpus_cfg.get("max_chunks_per_file", 2))
            max_total_chunks = int(corpus_cfg.get("max_total_chunks", 0))
            chunk_budget_respected = all(v <= max_chunks_per_file for v in chunks_by_file.values())
            if max_total_chunks > 0:
                chunk_budget_respected = chunk_budget_respected and (len(corpus_ctx.chunks) <= max_total_chunks)
            retrieval_query_present = bool((model_trace.get("stage_trace", {}).get("corpus", {}) or {}).get("retrieval_query"))
            hint_file_resolved = len(corpus_ctx.missing_files) == 0
            chunk_count_nonzero = len(corpus_ctx.chunks) > 0
            retrieval_errors: list[str] = []
            if not hint_file_resolved:
                retrieval_errors.append("hint_file_not_resolved")
            if not chunk_count_nonzero:
                retrieval_errors.append("chunk_count_zero")
            if not chunk_budget_respected:
                retrieval_errors.append("chunk_budget_exceeded")
            if not retrieval_query_present:
                retrieval_errors.append("retrieval_query_missing")
            stage_outcomes["retrieve"] = StageOutcome(
                status="pass" if chosen_evidence is not None and not retrieval_errors else "fail",
                errors=([] if chosen_evidence is not None else ["no_evidence_candidate"]) + retrieval_errors,
                artifacts={
                    "candidate_count": len(evidence_candidates),
                    "hint_file_resolved": hint_file_resolved,
                    "chunk_count_nonzero": chunk_count_nonzero,
                    "chunk_budget_respected": chunk_budget_respected,
                    "retrieval_query_present": retrieval_query_present,
                    "chunk_count": len(corpus_ctx.chunks),
                },
                timing_ms=(time.perf_counter() - retrieve_t0) * 1000.0,
            )

            extract_t0 = time.perf_counter()
            extraction, used_fallback = extract_values(qid, parsed, chosen_evidence, metadata)
            if strict_eval_mode and used_fallback:
                extraction = Extraction()
                attempt_notes.append("strict_extraction_fallback_blocked=true")
                solve_errors.append("extraction_fallback_blocked")
            elif used_fallback:
                attempt_notes.append("extraction_fallback_used=true")
            model_trace["stage_trace"]["extraction"] = {
                "used_fallback": used_fallback,
                "extracted_values": extraction.__dict__,
            }
            stage_outcomes["extract"] = StageOutcome(
                status="pass" if extraction.raw_value is not None else "fail",
                errors=[] if extraction.raw_value is not None else ["missing_raw_value"],
                artifacts=extraction.__dict__,
                timing_ms=(time.perf_counter() - extract_t0) * 1000.0,
            )

            calc_t0 = time.perf_counter()
            calculation = calculate_answer(parsed, extraction)
            model_trace["stage_trace"]["calculation"] = calculation.__dict__
            stage_outcomes["calculate"] = StageOutcome(
                status="pass" if calculation.result is not None else "fail",
                errors=[] if calculation.result is not None else ["missing_calculation_result"],
                artifacts=calculation.__dict__,
                timing_ms=(time.perf_counter() - calc_t0) * 1000.0,
            )

            final_answer = finalize_answer(calculation, extraction)
        else:
            stage_outcomes["retrieve"] = StageOutcome(status="skip")
            stage_outcomes["extract"] = StageOutcome(status="skip", artifacts=extraction.__dict__)
            stage_outcomes["calculate"] = StageOutcome(status="skip", artifacts=calculation.__dict__)

        if backend_name == "openhands_sdk" and rerun_solve:
            solve_t0 = time.perf_counter()
            quality_cfg = (config.get("evidence_quality", {}) or {})
            expected_min_rows, expected_max_rows = _expected_evidence_row_range(
                parsed=parsed,
                quality_cfg=quality_cfg,
            )
            llm_answer, backend_trace, backend_notes = run_openhands_backend(
                raw_question=raw_question,
                parsed_question=parsed.__dict__,
                metadata=metadata,
                corpus_chunks=corpus_ctx.chunks,
                config=config,
                project_root=Path(__file__).resolve().parents[1],
                repair_brief=repair_brief,
                attempt_index=attempt_idx,
                expected_rows_hint={
                    "min_rows": expected_min_rows,
                    "max_rows": expected_max_rows,
                },
            )
            attempt_notes.extend(backend_notes)
            if backend_trace:
                model_trace["backend_request"] = backend_trace.get("backend_request", model_trace["backend_request"])
                model_trace["backend_response_raw"] = backend_trace.get("backend_response_raw", model_trace["backend_response_raw"])
                model_trace["backend_response_structured"] = backend_trace.get(
                    "backend_response_structured", model_trace["backend_response_structured"]
                )
                model_trace["stage_trace"]["solve_backend_request"] = backend_trace.get("backend_request", {})
                model_trace["stage_trace"]["solve_backend_response_raw"] = backend_trace.get("backend_response_raw", "")
                model_trace["stage_trace"]["solve_backend_response_structured"] = backend_trace.get("backend_response_structured", {})
            if llm_answer is not None:
                final_answer = llm_answer

            evidence_rows = []
            solve_valid = False
            structured = model_trace.get("backend_response_structured") or {}
            if isinstance(structured, dict):
                raw_rows = structured.get("evidence_rows")
                if isinstance(raw_rows, list):
                    for row in raw_rows:
                        if isinstance(row, dict):
                            evidence_rows.append(
                                EvidenceRow(
                                    source_file=str(row.get("source_file", "")).strip(),
                                    table_or_section=str(row.get("table_or_section", "")).strip(),
                                    row_label=str(row.get("row_label", "")).strip(),
                                    column_label=str(row.get("column_label", "")).strip(),
                                    raw_value=row.get("raw_value"),
                                    unit=str(row.get("unit", "unknown")).strip(),
                                    matched_snippet=str(row.get("matched_snippet", "")).strip(),
                                )
                            )
                llm_grounded_by_index: dict[int, bool] = {}
                checks_cfg = (config.get("checks", {}) or {})
                use_llm_grounding_judge = bool(checks_cfg.get("use_llm_grounding_judge", False))
                llm_grounding_override = bool(checks_cfg.get("llm_grounding_override", True))
                if (
                    use_llm_grounding_judge
                    and backend_name == "openhands_sdk"
                    and evidence_rows
                ):
                    judge_rows = [er.__dict__.copy() for er in evidence_rows]
                    judge_obj, judge_trace, judge_notes = run_openhands_grounding_judge(
                        raw_question=raw_question,
                        parsed_question=parsed.__dict__,
                        metadata=metadata,
                        evidence_rows=judge_rows,
                        corpus_chunks=corpus_ctx.chunks,
                        config=config,
                        project_root=Path(__file__).resolve().parents[1],
                        attempt_index=attempt_idx,
                    )
                    attempt_notes.extend(judge_notes)
                    model_trace["stage_trace"]["llm_grounding_judge"] = {
                        "backend_request": judge_trace.get("backend_request", {}),
                        "backend_response_raw": judge_trace.get("backend_response_raw", ""),
                        "backend_response_structured": judge_trace.get("backend_response_structured", {}),
                    }
                    if isinstance(judge_obj, dict):
                        for rc in (judge_obj.get("row_checks") or []):
                            try:
                                ridx = int(rc.get("row_index"))
                            except Exception:
                                continue
                            llm_grounded_by_index[ridx] = bool(rc.get("grounded", False))
                        attempt_notes.append(f"llm_grounding_judge_rows={len(llm_grounded_by_index)}")
                min_signal_score = int(quality_cfg.get("min_signal_score", 5))
                require_full_grounding = bool(quality_cfg.get("require_full_grounding", True))
                filtered_rows, filtered_checks, quality_stats = _filter_evidence_rows_for_quality(
                    rows=evidence_rows,
                    raw_question=raw_question,
                    metadata=metadata,
                    parsed=parsed,
                    corpus_chunks=corpus_ctx.chunks,
                    max_rows=expected_max_rows,
                    min_rows=expected_min_rows,
                    min_signal_score=min_signal_score,
                    require_full_grounding=require_full_grounding,
                    llm_grounded_by_index=(llm_grounded_by_index if llm_grounding_override else None),
                )
                model_trace["stage_trace"]["solve_row_quality"] = {
                    **quality_stats,
                    "kept_reason_codes": [c.get("reason_codes", []) for c in filtered_checks],
                    "llm_grounding_override": llm_grounding_override,
                }
                if quality_stats.get("dropped_count", 0) > 0:
                    attempt_notes.append(f"evidence_rows_dropped={quality_stats.get('dropped_count')}")
                if quality_stats.get("input_count", 0) > expected_max_rows:
                    attempt_notes.append("evidence_row_cap_applied=true")
                attempt_notes.append(f"expected_evidence_rows_min={expected_min_rows}")
                attempt_notes.append(f"expected_evidence_rows_max={expected_max_rows}")
                evidence_rows = filtered_rows
                missing_row_fields: list[str] = []
                for ridx, er in enumerate(evidence_rows):
                    if not er.source_file:
                        missing_row_fields.append(f"row{ridx}:source_file")
                    if not er.table_or_section:
                        missing_row_fields.append(f"row{ridx}:table_or_section")
                    if not er.row_label:
                        missing_row_fields.append(f"row{ridx}:row_label")
                    if not er.column_label:
                        missing_row_fields.append(f"row{ridx}:column_label")
                    if er.raw_value is None or str(er.raw_value).strip() == "":
                        missing_row_fields.append(f"row{ridx}:raw_value")
                    if not er.unit:
                        missing_row_fields.append(f"row{ridx}:unit")
                    if not er.matched_snippet:
                        missing_row_fields.append(f"row{ridx}:matched_snippet")

                row_presence_gate = len(evidence_rows) >= expected_min_rows and not missing_row_fields
                kept_diags = [d for d in (quality_stats.get("row_diagnostics") or []) if d.get("kept", False)]
                row_grounding_gate = bool(kept_diags) and all(
                    bool(d.get("llm_grounded", False)) or bool((d.get("reason_codes") or []) == [])
                    for d in kept_diags
                )

                if row_presence_gate:
                    first = evidence_rows[0]
                    extraction = Extraction(
                        document=first.source_file,
                        table_title=first.table_or_section or extraction.table_title,
                        row_label=first.row_label or extraction.row_label,
                        column_label=first.column_label or extraction.column_label,
                        raw_value=first.raw_value,
                        unit=first.unit,
                    )
                    solve_valid = True
                else:
                    if len(evidence_rows) < expected_min_rows:
                        solve_errors.append("missing_evidence_rows")
                    else:
                        solve_errors.append("missing_evidence_row_fields")
                        solve_errors.extend(missing_row_fields)
                    if strict_eval_mode:
                        attempt_notes.append("strict_missing_evidence_rows=true")
                if not row_grounding_gate:
                    solve_errors.append("row_grounding_gate_failed")
                if not row_presence_gate:
                    solve_errors.append("row_presence_gate_failed")

            if parse_numeric(final_answer) is None:
                solve_errors.append("non_numeric_final_answer")
                if strict_eval_mode:
                    attempt_notes.append("strict_non_numeric_final_answer=true")

            stage_outcomes["solve"] = StageOutcome(
                status="pass" if solve_valid else "fail",
                errors=solve_errors.copy(),
                artifacts={
                    "final_answer": final_answer,
                    "evidence_row_count": len(evidence_rows),
                    "row_presence_gate": row_presence_gate,
                    "row_grounding_gate": row_grounding_gate,
                    "required_min_rows": expected_min_rows,
                    "allowed_max_rows": expected_max_rows,
                },
                timing_ms=(time.perf_counter() - solve_t0) * 1000.0,
            )
        else:
            stage_outcomes["solve"] = StageOutcome(status="skip", artifacts={"final_answer": final_answer})

        if strict_eval_mode and (not evidence_rows):
            extraction = Extraction()

        grounding_t0 = time.perf_counter()
        grounding_checks = evaluate_grounding(
            raw_question=raw_question,
            metadata=metadata,
            parsed=parsed,
            evidence_rows=evidence_rows,
            corpus_chunks=corpus_ctx.chunks,
        )
        model_trace["stage_trace"]["grounding_checks"] = grounding_checks
        if not rerun_solve:
            row_presence_gate = len(evidence_rows) >= expected_min_rows
            row_grounding_gate = bool(grounding_checks) and all(c.get("matched") for c in grounding_checks)
        stage_outcomes["grounding_precheck"] = StageOutcome(
            status="pass" if row_grounding_gate else "fail",
            errors=[] if row_grounding_gate else ["grounding_precheck_failed"],
            artifacts={
                "row_count": len(grounding_checks),
                "row_presence_gate": row_presence_gate,
                "row_grounding_gate": row_grounding_gate,
            },
            timing_ms=(time.perf_counter() - grounding_t0) * 1000.0,
        )

        verify_t0 = time.perf_counter()
        verification = verify_result(
            parsed,
            chosen_evidence,
            extraction,
            calculation,
            raw_question,
            metadata,
            evidence_rows,
            corpus_ctx.chunks,
        )
        # Combined strict evidence gate: deterministic checks plus optional LLM override.
        verification.evidence_sufficiency_check = bool(
            row_presence_gate and row_grounding_gate and chosen_evidence is not None and extraction.raw_value is not None
        )
        model_trace["stage_trace"]["verification_post_backend"] = verification.__dict__
        if not verification.evidence_sufficiency_check:
            attempt_notes.append("evidence_alignment=false")
        stage_outcomes["verify"] = StageOutcome(
            status="pass" if verification.evidence_sufficiency_check else "fail",
            errors=[] if verification.evidence_sufficiency_check else ["evidence_not_sufficient"],
            artifacts=verification.__dict__,
            timing_ms=(time.perf_counter() - verify_t0) * 1000.0,
        )

        evidence_row_dicts = [er.__dict__ for er in evidence_rows]
        normalized_pred, norm_details = normalize_numeric_prediction(
            predicted=final_answer,
            raw_question=raw_question,
            evidence_rows=evidence_row_dicts,
        )
        model_trace["stage_trace"]["normalization"] = norm_details
        tolerance = float((config.get("decision", {}) or {}).get("numeric_tolerance", config.get("evaluation", {}).get("numeric_tolerance", 0.01)))
        numeric_correct, eval_details = compare_numeric(normalized_pred, expected_answer, tolerance=tolerance)

        attempt_notes.append(f"eval_relative_error={eval_details.get('relative_error')}")
        attempt_notes.append(f"predicted_parse_source={eval_details.get('predicted_parse_source')}")
        attempt_notes.append(f"expected_parse_source={eval_details.get('expected_parse_source')}")
        attempt_notes.append(f"predicted_numeric={eval_details.get('predicted_numeric')}")
        attempt_notes.append(f"expected_numeric={eval_details.get('expected_numeric')}")
        attempt_notes.append(f"normalization_applied={norm_details.get('applied')}")
        attempt_notes.append(f"normalization_reason={norm_details.get('reason')}")

        strict_grounded_correct = _strict_numeric_gate(
            strict_eval_mode=strict_eval_mode,
            numeric_correct=numeric_correct,
            parse_valid=parse_valid,
            solve_valid=solve_valid,
            row_presence_gate=row_presence_gate,
            row_grounding_gate=row_grounding_gate,
            verification=verification,
        )

        schema_validation = {
            "parse_valid": parse_valid,
            "solve_valid": solve_valid,
            "parse_errors": parse_errors.copy(),
            "solve_errors": solve_errors.copy(),
        }
        scores = {
            "numeric_correct": numeric_correct,
            "strict_grounded_correct": strict_grounded_correct,
            "relative_error": eval_details.get("relative_error"),
        }

        gate_ok = strict_grounded_correct if final_gate == "strict_plus_numeric" else numeric_correct
        decision_t0 = time.perf_counter()
        stage_outcomes["decision"] = StageOutcome(
            status="pass" if gate_ok else "fail",
            errors=[] if gate_ok else ["final_gate_not_met"],
            artifacts={"final_gate": final_gate, "numeric_correct": numeric_correct, "strict_grounded_correct": strict_grounded_correct},
            timing_ms=(time.perf_counter() - decision_t0) * 1000.0,
        )

        failed_families = _failed_families(
            parse_valid=parse_valid,
            solve_valid=solve_valid,
            numeric_correct=numeric_correct,
            grounding_checks=grounding_checks,
            verification=verification,
        )
        stop_reason = ""
        if gate_ok and stop_on_strict_pass:
            stop_reason = "strict_pass"
        elif attempt_idx >= max_attempts:
            stop_reason = "max_attempts_reached"

        repair_brief = _build_repair_brief(
            attempt_idx=attempt_idx,
            schema_validation=schema_validation,
            grounding_checks=grounding_checks,
            eval_details=eval_details,
            scores=scores,
            final_answer=final_answer,
            expected_answer=expected_answer,
        )

        parse_prompt_chars_attempt = _extract_int_note("parse_prompt_char_count=", attempt_notes)
        solve_prompt_chars_attempt = _extract_int_note("solve_prompt_char_count=", attempt_notes)
        solve_prompt_chunks_attempt = _extract_int_note("solve_prompt_chunk_count=", attempt_notes)
        judge_prompt_chars_attempt = _extract_int_note("grounding_judge_prompt_char_count=", attempt_notes)
        judge_prompt_chunks_attempt = _extract_int_note("grounding_judge_prompt_chunk_count=", attempt_notes)
        parse_calls_attempt = 1 if model_trace.get("stage_trace", {}).get("parse_backend_request") else 0
        solve_calls_attempt = 1 if model_trace.get("stage_trace", {}).get("solve_backend_request") else 0
        grounding_judge_calls_attempt = 1 if (model_trace.get("stage_trace", {}).get("llm_grounding_judge", {}) or {}).get("backend_request") else 0
        attempt_budget = {
            "llm_calls": parse_calls_attempt + solve_calls_attempt + grounding_judge_calls_attempt,
            "parse_calls": parse_calls_attempt,
            "solve_calls": solve_calls_attempt,
            "grounding_judge_calls": grounding_judge_calls_attempt,
            "prompt_chars": parse_prompt_chars_attempt + solve_prompt_chars_attempt + judge_prompt_chars_attempt,
            "chunks_sent": solve_prompt_chunks_attempt + judge_prompt_chunks_attempt,
        }
        model_trace["stage_trace"]["attempt_budget"] = attempt_budget

        attempt_result = AttemptResult(
            attempt_index=attempt_idx,
            stage_outcomes=stage_outcomes,
            scores=scores,
            schema_validation=schema_validation,
            grounding_checks=grounding_checks,
            model_trace=model_trace,
            attempt_budget=attempt_budget,
            notes=attempt_notes,
            stop_reason=stop_reason,
            repair_brief=repair_brief,
        )
        attempts.append(attempt_result)
        snapshots.append(
            {
                "parsed_question": copy.deepcopy(parsed),
                "evidence_candidates": copy.deepcopy(evidence_candidates),
                "chosen_evidence": copy.deepcopy(chosen_evidence),
                "extracted_values": copy.deepcopy(extraction),
                "evidence_rows": copy.deepcopy(evidence_rows),
                "calculation": copy.deepcopy(calculation),
                "verification": copy.deepcopy(verification),
                "final_answer": final_answer,
                "scores": scores,
                "schema_validation": schema_validation,
                "grounding_checks": copy.deepcopy(grounding_checks),
                "model_trace": copy.deepcopy(model_trace),
                "notes": attempt_notes.copy(),
            }
        )

        previous_failed_families = failed_families
        if stop_reason:
            break

    strict_candidates = [a for a in attempts if (a.scores or {}).get("strict_grounded_correct")]
    accepted = len(strict_candidates) > 0

    def _rel(a: AttemptResult) -> float:
        v = (a.scores or {}).get("relative_error")
        return float(v) if isinstance(v, (int, float)) else 1e12

    best_numeric = min(attempts, key=_rel) if attempts else None
    best_strict = strict_candidates[0] if strict_candidates else None

    if accepted and best_strict is not None:
        selected_attempt = best_strict.attempt_index
        final_reason = "accepted_final"
    elif best_numeric is not None:
        selected_attempt = best_numeric.attempt_index
        final_reason = "rejected_after_max_attempts"
    else:
        selected_attempt = 1
        final_reason = "rejected_after_max_attempts"

    selected_idx = max(0, min(selected_attempt - 1, len(snapshots) - 1))
    selected = snapshots[selected_idx] if snapshots else {
        "parsed_question": ParsedQuestion(),
        "evidence_candidates": [],
        "chosen_evidence": None,
        "extracted_values": Extraction(),
        "evidence_rows": [],
        "calculation": Calculation(),
        "verification": Verification(),
        "final_answer": None,
        "scores": {"numeric_correct": False, "strict_grounded_correct": False, "relative_error": None},
        "schema_validation": {"parse_valid": False, "solve_valid": False, "parse_errors": ["no_attempts"], "solve_errors": ["no_attempts"]},
        "grounding_checks": [],
        "model_trace": {},
        "notes": ["no_attempts_executed"],
    }

    final_decision = FinalDecision(
        accepted=accepted,
        reason=final_reason,
        selected_attempt=selected_attempt,
        final_candidate=selected.get("final_answer"),
        final_reason=final_reason,
        gate_results={
            "numeric_correct": bool((selected.get("scores") or {}).get("numeric_correct", False)),
            "strict_grounded_correct": bool((selected.get("scores") or {}).get("strict_grounded_correct", False)),
        },
        best_numeric_candidate=None if best_numeric is None else snapshots[best_numeric.attempt_index - 1].get("final_answer"),
        best_numeric_attempt=0 if best_numeric is None else best_numeric.attempt_index,
        best_strict_candidate=None if best_strict is None else snapshots[best_strict.attempt_index - 1].get("final_answer"),
        best_strict_attempt=0 if best_strict is None else best_strict.attempt_index,
        selection_policy=final_gate,
    )

    parse_calls = 0
    solve_calls = 0
    grounding_judge_calls = 0
    parse_prompt_chars = 0
    solve_prompt_chars = 0
    grounding_judge_prompt_chars = 0
    solve_prompt_chunks = 0
    grounding_judge_prompt_chunks = 0
    for a in attempts:
        mt = a.model_trace or {}
        st = mt.get("stage_trace") or {}
        if st.get("parse_backend_request"):
            parse_calls += 1
        if st.get("solve_backend_request"):
            solve_calls += 1
        if (st.get("llm_grounding_judge") or {}).get("backend_request"):
            grounding_judge_calls += 1
        parse_prompt_chars += _extract_int_note("parse_prompt_char_count=", a.notes or [])
        solve_prompt_chars += _extract_int_note("solve_prompt_char_count=", a.notes or [])
        grounding_judge_prompt_chars += _extract_int_note("grounding_judge_prompt_char_count=", a.notes or [])
        solve_prompt_chunks += _extract_int_note("solve_prompt_chunk_count=", a.notes or [])
        grounding_judge_prompt_chunks += _extract_int_note("grounding_judge_prompt_chunk_count=", a.notes or [])

    call_budget = {
        "attempts_executed": len(attempts),
        "parse_calls": parse_calls,
        "solve_calls": solve_calls,
        "grounding_judge_calls": grounding_judge_calls,
        "total_llm_calls": parse_calls + solve_calls + grounding_judge_calls,
        "parse_prompt_chars_total": parse_prompt_chars,
        "solve_prompt_chars_total": solve_prompt_chars,
        "grounding_judge_prompt_chars_total": grounding_judge_prompt_chars,
        "solve_prompt_chunks_total": solve_prompt_chunks,
        "grounding_judge_prompt_chunks_total": grounding_judge_prompt_chunks,
    }

    result = ExampleResult(
        question_id=qid,
        raw_question=raw_question,
        parsed_question=selected.get("parsed_question"),
        evidence_candidates=selected.get("evidence_candidates"),
        chosen_evidence=selected.get("chosen_evidence"),
        extracted_values=selected.get("extracted_values"),
        evidence_rows=selected.get("evidence_rows"),
        calculation=selected.get("calculation"),
        verification=selected.get("verification"),
        final_answer=selected.get("final_answer"),
        expected_answer=expected_answer,
        is_correct=bool((selected.get("scores") or {}).get("numeric_correct", False)),
        failure_type=None,
        backend_name=backend_name,
        model_trace=selected.get("model_trace") or {},
        scores=selected.get("scores") or {},
        call_budget=call_budget,
        grounding_checks=selected.get("grounding_checks") or [],
        schema_validation=selected.get("schema_validation") or {},
        attempts=attempts,
        final_decision=final_decision,
        notes=selected.get("notes") or [],
        metadata=metadata,
    )
    result.failure_type = classify_failure(result)
    return result
