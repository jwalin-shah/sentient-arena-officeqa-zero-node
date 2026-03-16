from __future__ import annotations

import re

from src.types import EvidenceCandidate, ParsedQuestion


def select_evidence(question_id: str, parsed: ParsedQuestion, metadata: dict) -> tuple[list[EvidenceCandidate], EvidenceCandidate | None]:
    source_docs = str(metadata.get("source_docs", "")).strip()
    source_files = str(metadata.get("source_files", "")).strip()
    doc_hint = str(metadata.get("doc_hint", "")).strip()
    table_hint = str(metadata.get("table_hint", "")).strip()
    if not doc_hint:
        doc_hint = source_docs or f"Treasury Bulletin {parsed.time_period}"
    if not table_hint:
        table_hint = source_files or "Financial Summary"

    candidates = [
        EvidenceCandidate(
            document=doc_hint,
            table_title=table_hint,
            score=0.90,
            rationale="Metadata source hint matched.",
        ),
        EvidenceCandidate(
            document=f"Treasury Bulletin {parsed.time_period}",
            table_title="Budget Totals",
            score=0.70,
            rationale="Year-aligned fallback candidate.",
        ),
        EvidenceCandidate(
            document="Treasury Bulletin Reference",
            table_title="Historical Appendix",
            score=0.40,
            rationale="Low-confidence generic fallback.",
        ),
    ]

    year_match = re.search(r"(19|20)\d{2}", source_docs + " " + source_files)
    if year_match and parsed.time_period != "unknown" and year_match.group(0) == parsed.time_period:
        candidates[0].score = 0.95
        candidates[0].rationale = "Metadata source year aligns with parsed question year."

    chosen = candidates[0] if candidates else None
    return candidates, chosen
