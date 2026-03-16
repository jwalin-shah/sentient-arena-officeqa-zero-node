from __future__ import annotations

from src.types import Calculation, Extraction


def finalize_answer(calculation: Calculation, extraction: Extraction):
    if calculation.result is not None:
        return calculation.result
    return extraction.raw_value
