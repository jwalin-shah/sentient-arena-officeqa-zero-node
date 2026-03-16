from __future__ import annotations

import re
from typing import Any

from src.types import ParsedQuestion


def parse_question(question: str) -> ParsedQuestion:
    q = question.lower()

    metric = "unknown"
    for candidate in [
        "public debt",
        "receipts",
        "outlays",
        "deficit",
        "surplus",
        "revenue",
        "interest",
    ]:
        if candidate in q:
            metric = candidate
            break

    entity_category = "treasury"
    if "interest" in q:
        entity_category = "interest payments"
    elif "debt" in q:
        entity_category = "federal debt"

    year_match = re.search(r"(19|20)\d{2}", q)
    time_period = year_match.group(0) if year_match else "unknown"

    if "difference" in q or "minus" in q:
        operation = "difference"
    elif "average" in q:
        operation = "average"
    elif "percent" in q or "ratio" in q:
        operation = "ratio"
    else:
        operation = "lookup"

    expected_answer_type = "percent" if ("percent" in q or "ratio" in q) else "number"

    return ParsedQuestion(
        metric=metric,
        entity_category=entity_category,
        time_period=time_period,
        operation=operation,
        expected_answer_type=expected_answer_type,
    )


def parse_question_to_dict(question: str) -> dict[str, str]:
    parsed = parse_question(question)
    return {
        "metric": parsed.metric,
        "entity_category": parsed.entity_category,
        "time_period": parsed.time_period,
        "operation": parsed.operation,
        "expected_answer_type": parsed.expected_answer_type,
    }


def _normalize_operation(operation: str) -> str:
    op = operation.strip().lower()
    if op in {"lookup", "difference", "average", "ratio"}:
        return op
    if op in {"find", "get", "identify", "retrieve"}:
        return "lookup"
    if op in {"diff", "delta", "subtract", "subtraction"}:
        return "difference"
    if op in {"mean", "geometric mean", "weighted average"}:
        return "average"
    if op in {"percent", "percentage", "rate"}:
        return "ratio"
    return "unknown"


def _normalize_time_period(value: str) -> str:
    m = re.search(r"(19|20)\d{2}", value)
    return m.group(0) if m else "unknown"


def _normalize_answer_type(value: str) -> str:
    v = value.strip().lower()
    if any(x in v for x in ["percent", "%", "ratio"]):
        return "percent"
    return "number"


def parsed_question_from_dict(payload: dict[str, Any]) -> ParsedQuestion | None:
    required = [
        "metric",
        "entity_category",
        "time_period",
        "operation",
        "expected_answer_type",
    ]
    if not all(k in payload for k in required):
        return None
    values = {k: str(payload.get(k, "")).strip() for k in required}
    if any(v == "" for v in values.values()):
        return None
    normalized = {
        "metric": values["metric"].lower(),
        "entity_category": values["entity_category"],
        "time_period": _normalize_time_period(values["time_period"]),
        "operation": _normalize_operation(values["operation"]),
        "expected_answer_type": _normalize_answer_type(values["expected_answer_type"]),
    }
    if normalized["operation"] == "unknown":
        return None
    return ParsedQuestion(**normalized)
