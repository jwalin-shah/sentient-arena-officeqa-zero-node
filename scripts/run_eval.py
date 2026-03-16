#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import sys
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import process_example
from src.trace_review import build_trace_report, render_trace_report_md


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sanitize_tag(tag: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", tag).strip("-") or "run"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def _last_solve_artifacts(record: dict[str, Any]) -> dict[str, Any]:
        attempts = record.get("attempts") or []
        if not attempts:
            return {}
        stage_outcomes = (attempts[-1] or {}).get("stage_outcomes") or {}
        solve = stage_outcomes.get("solve") or {}
        return (solve.get("artifacts") or {})

    total = len(results)
    numeric_correct = sum(
        1
        for r in results
        if (r.get("scores") or {}).get("numeric_correct", r.get("is_correct", False))
    )
    strict_correct = sum(
        1
        for r in results
        if (r.get("scores") or {}).get("strict_grounded_correct", r.get("is_correct", False))
    )
    numeric_accuracy = (numeric_correct / total) if total else 0.0
    strict_accuracy = (strict_correct / total) if total else 0.0

    failures = Counter((r.get("failure_type") or "none") for r in results)
    accepted_final = sum(
        1 for r in results if ((r.get("final_decision") or {}).get("accepted") is True)
    )
    stop_reason_counts = Counter()
    for r in results:
        attempts = r.get("attempts") or []
        if attempts:
            last = attempts[-1] or {}
            reason = str(last.get("stop_reason", "")).strip() or "unknown"
        else:
            reason = str((r.get("final_decision") or {}).get("reason", "unknown"))
        stop_reason_counts[reason] += 1

    strict_pass_by_attempt = Counter()
    attempt_seen = Counter()
    for r in results:
        for a in (r.get("attempts") or []):
            idx = int(a.get("attempt_index", 0) or 0)
            if idx <= 0:
                continue
            attempt_seen[idx] += 1
            if (a.get("scores") or {}).get("strict_grounded_correct", False):
                strict_pass_by_attempt[idx] += 1

    strict_pass_rate_by_attempt: dict[str, float] = {}
    for idx, seen in sorted(attempt_seen.items()):
        strict_pass_rate_by_attempt[str(idx)] = (strict_pass_by_attempt.get(idx, 0) / seen) if seen else 0.0

    stage_stats = {
        "parse_ok": sum(1 for r in results if r["parsed_question"]["metric"] != "unknown"),
        "retrieval_ok": sum(1 for r in results if r.get("chosen_evidence") is not None),
        "extraction_ok": sum(1 for r in results if r["extracted_values"].get("raw_value") is not None),
        "extraction_grounded_ok": sum(1 for r in results if (r.get("evidence_rows") or [])),
        "calculation_ok": sum(1 for r in results if r["calculation"].get("result") is not None),
        "verification_ok": sum(
            1
            for r in results
            if all(
                r["verification"].get(k)
                for k in [
                    "correct_metric_match",
                    "correct_time_match",
                    "unit_check",
                    "arithmetic_check",
                    "evidence_sufficiency_check",
                ]
            )
        ),
        "strict_missing_source_file": sum(
            1 for r in results if any(str(n).startswith("strict_missing_source_file=true") for n in (r.get("notes") or []))
        ),
        "parse_schema_ok": sum(1 for r in results if (r.get("schema_validation") or {}).get("parse_valid", False)),
        "solve_schema_ok": sum(1 for r in results if (r.get("schema_validation") or {}).get("solve_valid", False)),
        "grounded_rows_ok": sum(
            1
            for r in results
            if (r.get("grounding_checks") or [])
            and all(c.get("matched") for c in (r.get("grounding_checks") or []))
        ),
        "row_presence_gate_ok": sum(
            1
            for r in results
            if bool(_last_solve_artifacts(r).get("row_presence_gate", False))
        ),
        "row_grounding_gate_ok": sum(
            1
            for r in results
            if bool(_last_solve_artifacts(r).get("row_grounding_gate", False))
        ),
    }

    attempts_total = 0
    parse_calls_total = 0
    solve_calls_total = 0
    llm_calls_total = 0
    parse_prompt_chars_total = 0
    solve_prompt_chars_total = 0
    solve_prompt_chunks_total = 0
    for r in results:
        cb = r.get("call_budget") or {}
        attempts_total += int(cb.get("attempts_executed", 0) or 0)
        parse_calls_total += int(cb.get("parse_calls", 0) or 0)
        solve_calls_total += int(cb.get("solve_calls", 0) or 0)
        llm_calls_total += int(cb.get("total_llm_calls", 0) or 0)
        parse_prompt_chars_total += int(cb.get("parse_prompt_chars_total", 0) or 0)
        solve_prompt_chars_total += int(cb.get("solve_prompt_chars_total", 0) or 0)
        solve_prompt_chunks_total += int(cb.get("solve_prompt_chunks_total", 0) or 0)

    avg_attempts = (attempts_total / total) if total else 0.0
    avg_llm_calls = (llm_calls_total / total) if total else 0.0

    stage_seen = Counter()
    stage_pass = Counter()
    stage_timing_sum = Counter()
    stage_error_counts: dict[str, Counter] = {}
    for r in results:
        for a in (r.get("attempts") or []):
            for stage_name, out in ((a.get("stage_outcomes") or {}).items()):
                status = str((out or {}).get("status", "skip"))
                if status == "skip":
                    continue
                stage_seen[stage_name] += 1
                if status == "pass":
                    stage_pass[stage_name] += 1
                stage_timing_sum[stage_name] += float((out or {}).get("timing_ms", 0.0) or 0.0)
                for err in ((out or {}).get("errors") or []):
                    stage_error_counts.setdefault(stage_name, Counter())[str(err)] += 1

    stage_reliability: dict[str, dict[str, Any]] = {}
    for stage_name, seen in sorted(stage_seen.items()):
        errs = stage_error_counts.get(stage_name, Counter())
        stage_reliability[stage_name] = {
            "pass_rate": (stage_pass.get(stage_name, 0) / seen) if seen else 0.0,
            "avg_timing_ms": (stage_timing_sum.get(stage_name, 0.0) / seen) if seen else 0.0,
            "top_error_codes": dict(errs.most_common(3)),
        }

    dropped_rows_total = 0
    dropped_rows_examples = 0
    drop_reason_counts = Counter()
    for r in results:
        had_drop = False
        for a in (r.get("attempts") or []):
            solve_q = ((a.get("model_trace") or {}).get("stage_trace") or {}).get("solve_row_quality") or {}
            row_diags = solve_q.get("row_diagnostics") or []
            for d in row_diags:
                if not d.get("kept", False):
                    had_drop = True
                    dropped_rows_total += 1
                    drop_reason_counts[str(d.get("drop_reason", "unknown"))] += 1
        if had_drop:
            dropped_rows_examples += 1

    return {
        "total_examples": total,
        "correct_examples": numeric_correct,
        "accuracy": numeric_accuracy,
        "numeric_correct_examples": numeric_correct,
        "strict_grounded_correct_examples": strict_correct,
        "numeric_accuracy_1pct": numeric_accuracy,
        "strict_grounded_accuracy": strict_accuracy,
        "accepted_final_rate": (accepted_final / total) if total else 0.0,
        "strict_pass_rate_by_attempt": strict_pass_rate_by_attempt,
        "attempt_stop_reason_counts": dict(stop_reason_counts),
        "failure_breakdown": dict(failures),
        "stage_stats": stage_stats,
        "telemetry": {
            "attempts_total": attempts_total,
            "avg_attempts_per_example": avg_attempts,
            "parse_calls_total": parse_calls_total,
            "solve_calls_total": solve_calls_total,
            "llm_calls_total": llm_calls_total,
            "avg_llm_calls_per_example": avg_llm_calls,
            "parse_prompt_chars_total": parse_prompt_chars_total,
            "solve_prompt_chars_total": solve_prompt_chars_total,
            "solve_prompt_chunks_total": solve_prompt_chunks_total,
        },
        "stage_reliability": stage_reliability,
        "quality_filter_summary": {
            "examples_with_dropped_rows": dropped_rows_examples,
            "dropped_rows_total": dropped_rows_total,
            "top_drop_reasons": dict(drop_reason_counts.most_common(5)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OfficeQA proxy harness evaluation.")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--limit", type=int, default=None, help="Max examples to run")
    parser.add_argument("--ids", default=None, help="Comma-separated question IDs")
    parser.add_argument("--tag", default=None, help="Optional run tag")
    parser.add_argument("--dataset", default=None, help="Override dataset path")
    parser.add_argument("--max-attempts", type=int, default=None, help="Override pipeline.max_attempts")
    parser.add_argument("--final-gate", default=None, help="Override decision.final_gate")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)
    if args.max_attempts is not None:
        config.setdefault("pipeline", {})
        config["pipeline"]["max_attempts"] = int(args.max_attempts)
    if args.final_gate is not None:
        config.setdefault("decision", {})
        config["decision"]["final_gate"] = str(args.final_gate)

    dataset_cfg = config.get("dataset_path", "data/sample_questions.jsonl")
    dataset_path = Path(args.dataset) if args.dataset else (ROOT / dataset_cfg)
    if not dataset_path.is_absolute():
        dataset_path = (ROOT / dataset_path).resolve()

    records = load_jsonl(dataset_path)

    wanted_ids = None
    if args.ids:
        wanted_ids = {x.strip() for x in args.ids.split(",") if x.strip()}
        records = [r for r in records if str(r.get("question_id")) in wanted_ids]

    limit = args.limit if args.limit is not None else config.get("subset", {}).get("limit")
    if limit:
        records = records[: int(limit)]

    tag = sanitize_tag(args.tag or config.get("run", {}).get("default_tag", "dev"))
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    out_examples = run_dir / "examples.jsonl"

    result_dicts: list[dict[str, Any]] = []
    run_t0 = time.perf_counter()
    total_records = len(records)
    with out_examples.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(records, start=1):
            qid = str(ex.get("question_id", "unknown"))
            ex_t0 = time.perf_counter()
            print(f"[{idx}/{total_records}] start question_id={qid}", flush=True)
            result = process_example(ex, config)
            payload = result.to_dict()
            result_dicts.append(payload)
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
            f.flush()
            ex_elapsed = time.perf_counter() - ex_t0
            strict_ok = bool((payload.get("scores") or {}).get("strict_grounded_correct", False))
            numeric_ok = bool((payload.get("scores") or {}).get("numeric_correct", False))
            print(
                f"[{idx}/{total_records}] done question_id={qid} numeric={numeric_ok} strict={strict_ok} elapsed_s={ex_elapsed:.1f}",
                flush=True,
            )

    summary = summarize(result_dicts)

    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    trace_cfg = config.get("trace", {}) or {}
    if bool(trace_cfg.get("review_enabled", True)):
        trace_report = build_trace_report(result_dicts)
        with (run_dir / "trace_report.json").open("w", encoding="utf-8") as f:
            json.dump(trace_report, f, indent=2, ensure_ascii=True)
        (run_dir / "trace_report.md").write_text(render_trace_report_md(trace_report), encoding="utf-8")

    resolved_config = dict(config)
    resolved_config["resolved"] = {
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "limit": limit,
        "ids": sorted(list(wanted_ids)) if wanted_ids else None,
        "tag": tag,
    }
    with (run_dir / "config.snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2, ensure_ascii=True)

    print(f"Run directory: {run_dir}")
    print(f"Examples written: {out_examples}")
    print(f"Total elapsed seconds: {time.perf_counter() - run_t0:.1f}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
