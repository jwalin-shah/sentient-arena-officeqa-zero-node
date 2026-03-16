from __future__ import annotations

import re
from typing import Any

from src.types import Calculation, EvidenceCandidate, EvidenceRow, Extraction, ParsedQuestion, Verification


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _norm_num_text(value: Any) -> str:
    # Handle common financial formatting: commas, dollar signs, and whitespace
    text = str(value or "").lower().replace(",", "").replace("$", "").strip()
    # Extract just the numeric part (e.g., "2,602.5" -> "2602.5")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return m.group(0) if m else ""


def _snippet(text: str, start: int, end: int, radius: int = 45) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return text[lo:hi]


def _find_text_match(haystack: str, needle: str) -> tuple[bool, dict[str, Any]]:
    if not needle:
        return False, {"start": None, "end": None, "snippet": "", "matched_text": ""}
    m = re.search(re.escape(needle), haystack, flags=re.IGNORECASE)
    if not m:
        # Whitespace-normalized fallback for table rendering differences.
        hay_norm = _norm_text(haystack)
        needle_norm = _norm_text(needle)
        if needle_norm and needle_norm in hay_norm:
            return True, {
                "start": None,
                "end": None,
                "snippet": needle.strip()[:220],
                "matched_text": needle.strip()[:220],
            }
        return False, {"start": None, "end": None, "snippet": "", "matched_text": ""}
    return True, {
        "start": m.start(),
        "end": m.end(),
        "snippet": _snippet(haystack, m.start(), m.end()),
        "matched_text": haystack[m.start():m.end()],
    }


def _find_numeric_match(haystack: str, raw_value: Any) -> tuple[bool, dict[str, Any]]:
    raw = str(raw_value or "").strip()
    if raw:
        ok, details = _find_text_match(haystack, raw)
        if ok:
            return True, details
    return False, {"start": None, "end": None, "snippet": "", "matched_text": ""}


def _metadata_source_files(metadata: dict[str, Any]) -> set[str]:
    raw = str((metadata or {}).get("source_files", ""))
    parts = [p.strip() for p in re.split(r"[\n,;]+", raw) if p.strip()]
    return set(parts)


def _question_terms(question: str) -> set[str]:
    terms = re.findall(r"[a-zA-Z]{4,}", question.lower())
    stop = {
        "what",
        "were",
        "from",
        "with",
        "that",
        "this",
        "using",
        "only",
        "reported",
        "values",
        "total",
        "calendar",
        "year",
        "in",
        "millions",
        "dollars",
    }
    return {t for t in terms if t not in stop}


def evaluate_grounding(
    *,
    raw_question: str,
    metadata: dict[str, Any],
    parsed: ParsedQuestion,
    evidence_rows: list[EvidenceRow],
    corpus_chunks: list[dict[str, str]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    allowed_files = _metadata_source_files(metadata)
    question_terms = _question_terms(raw_question)
    year = parsed.time_period if parsed.time_period != "unknown" else ""

    by_file: dict[str, str] = {}
    for c in corpus_chunks:
        sf = str(c.get("source_file", ""))
        by_file[sf] = by_file.get(sf, "") + "\n" + str(c.get("text", ""))

    for idx, row in enumerate(evidence_rows):
        source_file = row.source_file.strip()
        raw_text = str(by_file.get(source_file, ""))
        text = _norm_text(raw_text)
        source_file_allowed = (not allowed_files) or (source_file in allowed_files)
        raw_value_found, raw_value_match = _find_numeric_match(raw_text, row.raw_value)
        matched_snippet = str(getattr(row, "matched_snippet", "") or "").strip()
        snippet_found, snippet_match = _find_text_match(raw_text, matched_snippet)

        row_label_norm = _norm_text(row.row_label)
        column_label_norm = _norm_text(row.column_label)

        row_label_found, row_label_match = _find_text_match(text, row_label_norm)
        column_label_found, column_label_match = _find_text_match(text, column_label_norm)

        year_in_metadata = bool(year and (year in str(metadata.get("source_docs", "")) or year in str(metadata.get("source_files", ""))))
        year_aligned = True if not year else (year in text or year_in_metadata)

        row_topic_text = _norm_text(f"{row.table_or_section} {row.row_label}")
        topic_aligned = True
        if question_terms:
            topic_aligned = any(t in row_topic_text for t in question_terms)

        reason_codes: list[str] = []
        if not source_file_allowed:
            reason_codes.append("source_file_mismatch")
        if not raw_value_found:
            reason_codes.append("raw_value_not_found")
        if not row_label_found:
            reason_codes.append("row_label_not_found")
        if not column_label_found:
            reason_codes.append("column_label_not_found")
        if not matched_snippet:
            reason_codes.append("matched_snippet_missing")
        elif not snippet_found:
            reason_codes.append("matched_snippet_not_found")
        if not year_aligned:
            reason_codes.append("year_mismatch")
        if not topic_aligned:
            reason_codes.append("topic_mismatch")

        checks.append(
            {
                "row_index": idx,
                "source_file": source_file,
                "source_file_allowed": source_file_allowed,
                "raw_value_found": raw_value_found,
                "row_label_found": row_label_found,
                "column_label_found": column_label_found,
                "matched_snippet_found": snippet_found,
                "year_aligned": year_aligned,
                "topic_aligned": topic_aligned,
                "matched": len(reason_codes) == 0,
                "reason_codes": reason_codes,
                "matches": {
                    "raw_value": raw_value_match,
                    "row_label": row_label_match,
                    "column_label": column_label_match,
                    "matched_snippet": snippet_match,
                },
            }
        )
    return checks


def verify_result(
    parsed: ParsedQuestion,
    chosen_evidence: EvidenceCandidate | None,
    extraction: Extraction,
    calculation: Calculation,
    raw_question: str,
    metadata: dict,
    evidence_rows: list[EvidenceRow],
    corpus_chunks: list[dict[str, str]],
) -> Verification:
    q = raw_question.lower()

    metric_match = parsed.metric != "unknown" and (
        parsed.metric in q or parsed.metric in extraction.row_label.lower()
    )
    time_match = parsed.time_period != "unknown" and parsed.time_period in q
    unit_check = extraction.unit in {"dollars", "percent", "count", "millions of dollars", "billions of dollars", "million", "billion"}
    arithmetic_check = calculation.result is not None and calculation.formula != ""

    grounding_checks = evaluate_grounding(
        raw_question=raw_question,
        metadata=metadata,
        parsed=parsed,
        evidence_rows=evidence_rows,
        corpus_chunks=corpus_chunks,
    )
    grounded = bool(grounding_checks) and all(c.get("matched") for c in grounding_checks)
    evidence_sufficient = chosen_evidence is not None and extraction.raw_value is not None and grounded

    return Verification(
        correct_metric_match=metric_match,
        correct_time_match=time_match,
        unit_check=unit_check,
        arithmetic_check=arithmetic_check,
        evidence_sufficiency_check=evidence_sufficient,
    )
