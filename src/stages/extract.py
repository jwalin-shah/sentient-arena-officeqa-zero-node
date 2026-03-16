from __future__ import annotations

import hashlib

from src.types import EvidenceCandidate, Extraction, ParsedQuestion


def _deterministic_fallback_value(question_id: str) -> float:
    digest = hashlib.md5(question_id.encode("utf-8")).hexdigest()[:6]
    base = int(digest, 16) % 10000
    return round(base / 10.0, 2)


def extract_values(
    question_id: str,
    parsed: ParsedQuestion,
    chosen_evidence: EvidenceCandidate | None,
    metadata: dict,
) -> tuple[Extraction, bool]:
    if chosen_evidence is None:
        return Extraction(), False

    raw_value = metadata.get("mock_value")
    used_fallback = False
    if raw_value is None:
        raw_value = _deterministic_fallback_value(question_id)
        used_fallback = True

    unit = "percent" if parsed.expected_answer_type == "percent" else "dollars"

    return (
        Extraction(
            document=chosen_evidence.document,
            table_title=chosen_evidence.table_title,
            row_label=parsed.metric,
            column_label=parsed.time_period,
            raw_value=raw_value,
            unit=unit,
        ),
        used_fallback,
    )
