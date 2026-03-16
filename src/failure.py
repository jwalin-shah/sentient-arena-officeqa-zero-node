from __future__ import annotations

from src.types import ExampleResult


FAILURE_TYPES = {
    "parse",
    "retrieval",
    "extraction",
    "units",
    "arithmetic",
    "verification",
    "unknown",
}


def classify_failure(result: ExampleResult) -> str | None:
    scores = result.scores or {}
    strict_correct = scores.get("strict_grounded_correct")
    if isinstance(strict_correct, bool):
        if strict_correct:
            return None
    elif result.is_correct:
        return None

    notes_joined = " ".join(result.notes)
    schema_validation = result.schema_validation or {}

    if not schema_validation.get("parse_valid", True):
        return "parse"
    if "strict_parse_failed=true" in notes_joined:
        return "parse"
    if "parse_structured_invalid_json=true" in notes_joined or "parse_structured_missing_fields=true" in notes_joined:
        return "parse"

    if "strict_missing_source_file=true" in notes_joined:
        return "retrieval"

    for check in result.grounding_checks or []:
        reasons = set(check.get("reason_codes") or [])
        if "source_file_mismatch" in reasons:
            return "retrieval"

    if not schema_validation.get("solve_valid", True):
        return "extraction"
    if "strict_missing_evidence_rows=true" in notes_joined:
        return "extraction"
    if result.extracted_values.raw_value is None or len(result.evidence_rows) == 0:
        return "extraction"

    for check in result.grounding_checks or []:
        reasons = set(check.get("reason_codes") or [])
        if reasons.intersection(
            {
                "row_label_not_found",
                "column_label_not_found",
                "raw_value_not_found",
                "matched_snippet_missing",
                "matched_snippet_not_found",
            }
        ):
            return "extraction"
        if reasons.intersection({"year_mismatch", "topic_mismatch"}):
            return "verification"

    checks = result.verification
    if not checks.unit_check:
        return "units"
    if not checks.arithmetic_check:
        return "arithmetic"

    if not (checks.correct_metric_match and checks.correct_time_match and checks.evidence_sufficiency_check):
        return "verification"

    if "eval_relative_error=" in notes_joined:
        return "arithmetic"

    return "unknown"
