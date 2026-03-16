from __future__ import annotations

from collections import Counter
from typing import Any


def build_trace_report(examples: list[dict[str, Any]]) -> dict[str, Any]:
    failure_counts = Counter((e.get("failure_type") or "none") for e in examples)
    parse_fail_reasons = Counter()
    evidence_mismatch = 0
    normalization_issues = 0
    missing_source_files = 0
    ungrounded_extraction = 0
    parse_schema_violations = 0
    unit_normalization_mismatch = 0

    schema_violations = 0
    total_rows = 0
    grounded_rows = 0
    source_file_aligned_rows = 0
    failing_fields = Counter()
    attempt_counts = Counter()
    strict_pass_attempt_counts = Counter()
    stop_reason_counts = Counter()
    repair_code_by_attempt: dict[int, Counter] = {}
    grounding_reason_counts = Counter()
    strict_failure_questions: list[dict[str, Any]] = []
    dropped_row_reason_counts = Counter()
    dropped_row_examples = 0
    dropped_row_total = 0
    row_presence_gate_pass = 0
    row_grounding_gate_pass = 0
    solve_attempts_seen = 0

    for ex in examples:
        notes = ex.get("notes") or []
        joined = " ".join(notes)
        schema_validation = ex.get("schema_validation") or {}
        if not schema_validation.get("parse_valid", True):
            schema_violations += 1
        if not schema_validation.get("solve_valid", True):
            schema_violations += 1

        if ex.get("failure_type") == "parse":
            if "parse_structured_invalid_json=true" in joined:
                parse_fail_reasons["invalid_json"] += 1
            elif "parse_structured_missing_fields=true" in joined:
                parse_fail_reasons["missing_fields"] += 1
                parse_schema_violations += 1
            elif "parse_fallback_used=true" in joined:
                parse_fail_reasons["fallback_used"] += 1
            else:
                parse_fail_reasons["rule_parser_unknown"] += 1
        if "evidence_alignment=false" in joined:
            evidence_mismatch += 1
        if "strict_missing_source_file=true" in joined:
            missing_source_files += 1
        if "strict_missing_evidence_rows=true" in joined or not (ex.get("evidence_rows") or []):
            ungrounded_extraction += 1
        if "predicted_parse_source=none" in joined or "expected_parse_source=none" in joined:
            normalization_issues += 1
        if "normalization_applied=True" in joined and ex.get("failure_type") in {"units", "arithmetic"}:
            unit_normalization_mismatch += 1

        for row in ex.get("evidence_rows") or []:
            if not str(row.get("source_file", "")).strip():
                failing_fields["source_file"] += 1
            if not str(row.get("row_label", "")).strip():
                failing_fields["row_label"] += 1
            if not str(row.get("column_label", "")).strip():
                failing_fields["column_label"] += 1
            if row.get("raw_value") in {None, ""}:
                failing_fields["raw_value"] += 1
            if not str(row.get("unit", "")).strip():
                failing_fields["unit"] += 1
            if not str(row.get("matched_snippet", "")).strip():
                failing_fields["matched_snippet"] += 1

        for check in ex.get("grounding_checks") or []:
            total_rows += 1
            if check.get("matched"):
                grounded_rows += 1
            if check.get("source_file_allowed"):
                source_file_aligned_rows += 1
            for rc in (check.get("reason_codes") or []):
                grounding_reason_counts[str(rc)] += 1

        if not (ex.get("scores") or {}).get("strict_grounded_correct", False):
            severity = 0
            severity += len((ex.get("schema_validation") or {}).get("parse_errors") or [])
            severity += len((ex.get("schema_validation") or {}).get("solve_errors") or [])
            severity += sum(len(c.get("reason_codes") or []) for c in (ex.get("grounding_checks") or []))
            strict_failure_questions.append(
                {
                    "question_id": ex.get("question_id"),
                    "failure_type": ex.get("failure_type"),
                    "severity": severity,
                }
            )

        for a in ex.get("attempts") or []:
            idx = int(a.get("attempt_index", 0) or 0)
            if idx <= 0:
                continue
            attempt_counts[idx] += 1
            if (a.get("scores") or {}).get("strict_grounded_correct", False):
                strict_pass_attempt_counts[idx] += 1
            sr = str(a.get("stop_reason", "")).strip()
            if sr:
                stop_reason_counts[sr] += 1
            schema = a.get("schema_validation") or {}
            solve_artifacts = ((a.get("stage_outcomes") or {}).get("solve") or {}).get("artifacts") or {}
            if solve_artifacts:
                solve_attempts_seen += 1
                if bool(solve_artifacts.get("row_presence_gate", False)):
                    row_presence_gate_pass += 1
                if bool(solve_artifacts.get("row_grounding_gate", False)):
                    row_grounding_gate_pass += 1
            for code in (schema.get("parse_errors") or []):
                repair_code_by_attempt.setdefault(idx, Counter())[f"parse:{code}"] += 1
            for code in (schema.get("solve_errors") or []):
                repair_code_by_attempt.setdefault(idx, Counter())[f"solve:{code}"] += 1
            for g in (a.get("grounding_checks") or []):
                for rc in (g.get("reason_codes") or []):
                    repair_code_by_attempt.setdefault(idx, Counter())[f"ground:{rc}"] += 1
            solve_q = ((a.get("model_trace") or {}).get("stage_trace") or {}).get("solve_row_quality") or {}
            row_diags = solve_q.get("row_diagnostics") or []
            if row_diags:
                had_drop = False
                for d in row_diags:
                    if not d.get("kept", False):
                        had_drop = True
                        dropped_row_total += 1
                        dropped_row_reason_counts[str(d.get("drop_reason", "unknown"))] += 1
                if had_drop:
                    dropped_row_examples += 1

    total_examples = len(examples)
    schema_violation_rate = (schema_violations / (2 * total_examples)) if total_examples else 0.0
    grounded_row_match_rate = (grounded_rows / total_rows) if total_rows else 0.0
    source_file_alignment_rate = (source_file_aligned_rows / total_rows) if total_rows else 0.0

    pass_by_attempt_curve: dict[str, float] = {}
    for idx, seen in sorted(attempt_counts.items()):
        pass_by_attempt_curve[str(idx)] = (strict_pass_attempt_counts.get(idx, 0) / seen) if seen else 0.0

    repair_trends: dict[str, dict[str, int]] = {}
    for idx, counter in sorted(repair_code_by_attempt.items()):
        repair_trends[str(idx)] = dict(counter.most_common(8))

    top_actions: list[str] = []
    if parse_fail_reasons:
        top_actions.append("Tighten parse prompt/schema and inspect malformed parse JSON examples.")
    if parse_schema_violations > 0 or schema_violation_rate > 0.10:
        top_actions.append("Enforce schema decoding and keep strict JSON response_format enabled.")
    if missing_source_files > 0:
        top_actions.append("Fix corpus root or fetch missing transformed TXT files for referenced source_files.")
    if ungrounded_extraction > 0 or grounded_row_match_rate < 0.7:
        top_actions.append("Improve evidence_rows grounding: require row/column/value strings that match corpus chunks.")
    if source_file_alignment_rate < 1.0:
        top_actions.append("Constrain source_file to metadata.source_files and fail on mismatches.")
    if not top_actions:
        top_actions.append("No dominant failure cluster; run 3 more examples and inspect trace manually.")
    top_actions = top_actions[:3]

    recommended_edit_target = "verifier"
    top_code = grounding_reason_counts.most_common(1)
    if parse_fail_reasons and sum(parse_fail_reasons.values()) >= max(1, (top_code[0][1] if top_code else 0)):
        recommended_edit_target = "prompt"
    elif top_code:
        if top_code[0][0] in {"source_file_mismatch", "year_mismatch"}:
            recommended_edit_target = "retrieval"
        elif top_code[0][0] in {
            "row_label_not_found",
            "column_label_not_found",
            "raw_value_not_found",
            "matched_snippet_missing",
            "matched_snippet_not_found",
        }:
            recommended_edit_target = "prompt"
        else:
            recommended_edit_target = "verifier"

    strict_failure_questions.sort(key=lambda x: x.get("severity", 0), reverse=True)

    return {
        "total_examples": total_examples,
        "failure_bucket_counts": dict(failure_counts),
        "top_parse_failure_reasons": dict(parse_fail_reasons),
        "parse_schema_violation_count": parse_schema_violations,
        "missing_source_file_count": missing_source_files,
        "ungrounded_extraction_count": ungrounded_extraction,
        "evidence_mismatch_count": evidence_mismatch,
        "numeric_normalization_issues": normalization_issues,
        "unit_normalization_mismatch_count": unit_normalization_mismatch,
        "schema_violation_rate": schema_violation_rate,
        "grounded_row_match_rate": grounded_row_match_rate,
        "source_file_alignment_rate": source_file_alignment_rate,
        "top_evidence_row_failing_fields": dict(failing_fields.most_common(5)),
        "pass_by_attempt_curve": pass_by_attempt_curve,
        "attempt_stop_reason_counts": dict(stop_reason_counts),
        "repair_code_frequency_by_attempt": repair_trends,
        "top_failing_reason_codes": dict(grounding_reason_counts.most_common(5)),
        "dropped_rows": {
            "examples_with_dropped_rows": dropped_row_examples,
            "dropped_rows_total": dropped_row_total,
            "top_drop_reasons": dict(dropped_row_reason_counts.most_common(5)),
        },
        "row_gate_metrics": {
            "solve_attempts_seen": solve_attempts_seen,
            "row_presence_gate_pass_rate": (row_presence_gate_pass / solve_attempts_seen) if solve_attempts_seen else 0.0,
            "row_grounding_gate_pass_rate": (row_grounding_gate_pass / solve_attempts_seen) if solve_attempts_seen else 0.0,
        },
        "top_strict_failure_questions": strict_failure_questions[:3],
        "recommended_edit_target": recommended_edit_target,
        "top_actions": top_actions,
    }


def render_trace_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Trace Review",
        "",
        f"- Total examples: `{report.get('total_examples', 0)}`",
        "",
        "## Failure Buckets",
    ]
    for k, v in sorted((report.get("failure_bucket_counts") or {}).items()):
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Parse Failures"]
    parse_reasons = report.get("top_parse_failure_reasons") or {}
    if parse_reasons:
        for k, v in sorted(parse_reasons.items()):
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Evidence / Normalization",
        f"- parse_schema_violation_count: `{report.get('parse_schema_violation_count', 0)}`",
        f"- missing_source_file_count: `{report.get('missing_source_file_count', 0)}`",
        f"- ungrounded_extraction_count: `{report.get('ungrounded_extraction_count', 0)}`",
        f"- evidence_mismatch_count: `{report.get('evidence_mismatch_count', 0)}`",
        f"- numeric_normalization_issues: `{report.get('numeric_normalization_issues', 0)}`",
        f"- unit_normalization_mismatch_count: `{report.get('unit_normalization_mismatch_count', 0)}`",
        f"- schema_violation_rate: `{report.get('schema_violation_rate', 0.0):.3f}`",
        f"- grounded_row_match_rate: `{report.get('grounded_row_match_rate', 0.0):.3f}`",
        f"- source_file_alignment_rate: `{report.get('source_file_alignment_rate', 0.0):.3f}`",
        "",
        "## Attempt Dynamics",
    ]
    curve = report.get("pass_by_attempt_curve") or {}
    if curve:
        for k, v in curve.items():
            lines.append(f"- attempt {k} strict_pass_rate: `{float(v):.3f}`")
    else:
        lines.append("- no attempt data")
    lines.append("")
    lines.append("## Stop Reasons")
    stops = report.get("attempt_stop_reason_counts") or {}
    if stops:
        for k, v in sorted(stops.items()):
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- none")

    lines += [
        "",
        "## Top Failing Evidence Fields",
    ]
    fields = report.get("top_evidence_row_failing_fields") or {}
    if fields:
        for k, v in fields.items():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- none")

    lines += ["", "## Repair Trends By Attempt"]
    trends = report.get("repair_code_frequency_by_attempt") or {}
    if trends:
        for attempt, details in trends.items():
            lines.append(f"- attempt `{attempt}`: {details}")
    else:
        lines.append("- none")

    lines += ["", "## Top Failing Reason Codes"]
    reason_codes = report.get("top_failing_reason_codes") or {}
    if reason_codes:
        for k, v in reason_codes.items():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- none")

    lines += ["", "## Dropped Rows"]
    dr = report.get("dropped_rows") or {}
    lines.append(f"- examples_with_dropped_rows: `{dr.get('examples_with_dropped_rows', 0)}`")
    lines.append(f"- dropped_rows_total: `{dr.get('dropped_rows_total', 0)}`")
    top_drop = dr.get("top_drop_reasons") or {}
    if top_drop:
        for k, v in top_drop.items():
            lines.append(f"- drop_reason `{k}`: {v}")
    else:
        lines.append("- no dropped-row diagnostics")

    lines += ["", "## Row Gate Metrics"]
    rg = report.get("row_gate_metrics") or {}
    lines.append(f"- solve_attempts_seen: `{rg.get('solve_attempts_seen', 0)}`")
    lines.append(f"- row_presence_gate_pass_rate: `{float(rg.get('row_presence_gate_pass_rate', 0.0)):.3f}`")
    lines.append(f"- row_grounding_gate_pass_rate: `{float(rg.get('row_grounding_gate_pass_rate', 0.0)):.3f}`")

    lines += ["", "## Strict Failure Severity (Top 3 Questions)"]
    top_q = report.get("top_strict_failure_questions") or []
    if top_q:
        for q in top_q:
            lines.append(
                f"- `{q.get('question_id')}`: failure_type=`{q.get('failure_type')}`, severity=`{q.get('severity')}`"
            )
    else:
        lines.append("- none")

    lines += ["", "## Recommended Next Edit Target"]
    lines.append(f"- `{report.get('recommended_edit_target', 'verifier')}`")

    lines += ["", "## Top 3 Actions"]
    for action in report.get("top_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines) + "\n"
