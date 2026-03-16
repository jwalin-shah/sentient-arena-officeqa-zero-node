#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import compare_numeric
from src.failure import classify_failure
from src.stages.parse import parsed_question_from_dict
from src.stages.verify import evaluate_grounding
from src.types import (
    AttemptResult,
    Calculation,
    EvidenceCandidate,
    EvidenceRow,
    ExampleResult,
    Extraction,
    FinalDecision,
    ParsedQuestion,
    StageOutcome,
    Verification,
)


def make_base_result() -> ExampleResult:
    return ExampleResult(
        question_id="t",
        raw_question="What was debt in 2020?",
        parsed_question=ParsedQuestion(metric="public debt", operation="lookup", time_period="2020"),
        evidence_candidates=[EvidenceCandidate("d", "t", 1.0, "r")],
        chosen_evidence=EvidenceCandidate("d", "t", 1.0, "r"),
        extracted_values=Extraction(raw_value=10, unit="dollars"),
        evidence_rows=[EvidenceRow(source_file="x.txt", row_label="public debt", column_label="2020", raw_value=10, unit="dollars")],
        calculation=Calculation(formula="value", substituted_values={"value": 10}, result=10),
        verification=Verification(True, True, True, True, True),
        final_answer=10,
        expected_answer=20,
        is_correct=False,
        failure_type=None,
        scores={"numeric_correct": False, "strict_grounded_correct": False},
        schema_validation={"parse_valid": True, "solve_valid": True},
        grounding_checks=[{"matched": True, "source_file_allowed": True, "reason_codes": []}],
        attempts=[
            AttemptResult(
                attempt_index=1,
                stage_outcomes={"parse": StageOutcome(status="pass")},
                scores={"numeric_correct": False, "strict_grounded_correct": False, "relative_error": 0.5},
                schema_validation={"parse_valid": True, "solve_valid": True, "parse_errors": [], "solve_errors": []},
            )
        ],
        final_decision=FinalDecision(accepted=False, reason="rejected_after_max_attempts", selected_attempt=1),
    )


def test_evaluator() -> None:
    ok, details = compare_numeric("101", "100", tolerance=0.01)
    assert ok
    assert details["predicted_parse_source"] == "string"
    ok, _ = compare_numeric("102", "100", tolerance=0.01)
    assert not ok
    ok, _ = compare_numeric("8%", "0.08", tolerance=0.01)
    assert ok
    ok, _ = compare_numeric("abc", "10", tolerance=0.01)
    assert not ok


def test_parse_validation() -> None:
    valid = {
        "metric": "public debt",
        "entity_category": "federal debt",
        "time_period": "2020",
        "operation": "lookup",
        "expected_answer_type": "number",
    }
    assert parsed_question_from_dict(valid) is not None
    invalid = dict(valid)
    invalid.pop("metric")
    assert parsed_question_from_dict(invalid) is None


def test_grounding_checks() -> None:
    checks = evaluate_grounding(
        raw_question="What was public debt in 2020?",
        metadata={"source_files": "x.txt"},
        parsed=ParsedQuestion(metric="public debt", time_period="2020"),
        evidence_rows=[EvidenceRow(source_file="x.txt", row_label="public debt", column_label="2020", raw_value="10", unit="dollars")],
        corpus_chunks=[{"source_file": "x.txt", "chunk_id": "x.txt#chunk1", "text": "public debt 2020 value 10"}],
    )
    assert checks and checks[0]["matched"]


def test_failures() -> None:
    r = make_base_result()
    r.schema_validation = {"parse_valid": False, "solve_valid": True}
    assert classify_failure(r) == "parse"

    r = make_base_result()
    r.grounding_checks = [{"matched": False, "source_file_allowed": False, "reason_codes": ["source_file_mismatch"]}]
    assert classify_failure(r) == "retrieval"

    r = make_base_result()
    r.schema_validation = {"parse_valid": True, "solve_valid": False}
    assert classify_failure(r) == "extraction"

    r = make_base_result()
    r.verification.unit_check = False
    assert classify_failure(r) == "units"

    r = make_base_result()
    r.verification.arithmetic_check = False
    assert classify_failure(r) == "arithmetic"

    r = make_base_result()
    r.verification.correct_time_match = False
    assert classify_failure(r) == "verification"

    r = make_base_result()
    r.scores = {"numeric_correct": True, "strict_grounded_correct": True}
    assert classify_failure(r) is None


def main() -> int:
    test_evaluator()
    test_parse_validation()
    test_grounding_checks()
    test_failures()
    print("self_test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
