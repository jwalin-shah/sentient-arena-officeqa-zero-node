from __future__ import annotations

import math
import re
from typing import Any


def parse_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = text.replace("$", "").replace(",", "").strip()

    percent = text.endswith("%")
    if percent:
        text = text[:-1].strip()

    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not m:
        return None

    number = float(m.group(0))
    if negative:
        number = -number
    if percent:
        number = number / 100.0
    return number


def parse_source(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, (int, float)):
        return "number"
    if not isinstance(value, str):
        return "non_string"
    return "string"


def normalize_numeric_prediction(
    *,
    predicted: Any,
    raw_question: str,
    evidence_rows: list[dict[str, Any]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    details: dict[str, Any] = {
        "applied": False,
        "reason": "",
        "multiplier": 1.0,
        "unit_from_evidence": "",
    }
    pred = parse_numeric(predicted)
    if pred is None:
        return predicted, details

    q = raw_question.lower()
    unit = ""
    if evidence_rows:
        for row in evidence_rows:
            u = str((row or {}).get("unit", "")).lower()
            if u:
                unit = u
                break
    details["unit_from_evidence"] = unit

    if ("in millions" in q or "million" in q) and "billion" in unit:
        pred *= 1000.0
        details.update({"applied": True, "reason": "billion_to_million", "multiplier": 1000.0})
    elif ("in billions" in q or "billion" in q) and "million" in unit:
        pred /= 1000.0
        details.update({"applied": True, "reason": "million_to_billion", "multiplier": 0.001})
    elif "percent" in q and isinstance(predicted, (int, float, str)) and "%" not in str(predicted):
        # Many models emit 12.34 instead of 12.34% or 0.1234; keep conservative conversion.
        if pred > 1.0:
            pred = pred / 100.0
            details.update({"applied": True, "reason": "percent_scale_fix", "multiplier": 0.01})
    return pred, details


def compare_numeric(predicted: Any, expected: Any, tolerance: float = 0.01) -> tuple[bool, dict[str, Any]]:
    pred = parse_numeric(predicted)
    gold = parse_numeric(expected)
    details: dict[str, Any] = {
        "predicted_numeric": pred,
        "expected_numeric": gold,
        "predicted_parse_source": parse_source(predicted),
        "expected_parse_source": parse_source(expected),
        "relative_error": None,
    }

    if pred is None or gold is None:
        return False, details

    if math.isclose(gold, 0.0, abs_tol=1e-12):
        is_correct = abs(pred - gold) <= tolerance
        details["relative_error"] = abs(pred - gold)
        return is_correct, details

    rel_error = abs(pred - gold) / abs(gold)
    details["relative_error"] = rel_error
    return rel_error <= tolerance, details
