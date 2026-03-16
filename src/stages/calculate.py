from __future__ import annotations

from src.evaluator import parse_numeric
from src.types import Calculation, Extraction, ParsedQuestion


def calculate_answer(parsed: ParsedQuestion, extraction: Extraction) -> Calculation:
    value = parse_numeric(extraction.raw_value)
    if value is None:
        return Calculation(formula="", substituted_values={}, result=None)

    if parsed.operation == "ratio" and extraction.unit == "percent":
        # In stub mode we keep extracted ratio as-is.
        return Calculation(formula="ratio_value", substituted_values={"ratio_value": value}, result=value)

    if parsed.operation == "average":
        return Calculation(formula="value", substituted_values={"value": value}, result=value)

    if parsed.operation == "difference":
        return Calculation(formula="value", substituted_values={"value": value}, result=value)

    return Calculation(formula="value", substituted_values={"value": value}, result=value)
