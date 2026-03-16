from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ParsedQuestion:
    metric: str = "unknown"
    entity_category: str = "unknown"
    time_period: str = "unknown"
    operation: str = "lookup"
    expected_answer_type: str = "number"


@dataclass
class EvidenceCandidate:
    document: str
    table_title: str
    score: float
    rationale: str


@dataclass
class Extraction:
    document: str = "unknown"
    table_title: str = "unknown"
    row_label: str = "unknown"
    column_label: str = "unknown"
    raw_value: str | float | int | None = None
    unit: str = "unknown"


@dataclass
class Calculation:
    formula: str = ""
    substituted_values: dict[str, Any] = field(default_factory=dict)
    result: float | None = None


@dataclass
class Verification:
    correct_metric_match: bool = False
    correct_time_match: bool = False
    unit_check: bool = False
    arithmetic_check: bool = False
    evidence_sufficiency_check: bool = False


@dataclass
class EvidenceRow:
    source_file: str = ""
    table_or_section: str = ""
    row_label: str = ""
    column_label: str = ""
    raw_value: str | float | int | None = None
    unit: str = "unknown"
    matched_snippet: str = ""


@dataclass
class StageOutcome:
    status: str = "skip"  # pass|fail|skip
    errors: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    timing_ms: float = 0.0


@dataclass
class AttemptResult:
    attempt_index: int
    stage_outcomes: dict[str, StageOutcome] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)
    schema_validation: dict[str, Any] = field(default_factory=dict)
    grounding_checks: list[dict[str, Any]] = field(default_factory=list)
    model_trace: dict[str, Any] = field(default_factory=dict)
    attempt_budget: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    stop_reason: str = ""
    repair_brief: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalDecision:
    accepted: bool = False
    reason: str = ""
    selected_attempt: int = 0
    final_candidate: Any = None
    final_reason: str = ""
    gate_results: dict[str, bool] = field(default_factory=dict)
    best_numeric_candidate: Any = None
    best_numeric_attempt: int = 0
    best_strict_candidate: Any = None
    best_strict_attempt: int = 0
    selection_policy: str = "strict_plus_numeric"


@dataclass
class ExampleResult:
    question_id: str
    raw_question: str
    parsed_question: ParsedQuestion
    evidence_candidates: list[EvidenceCandidate]
    chosen_evidence: EvidenceCandidate | None
    extracted_values: Extraction
    evidence_rows: list[EvidenceRow]
    calculation: Calculation
    verification: Verification
    final_answer: str | float | int | None
    expected_answer: str | float | int | None
    is_correct: bool
    failure_type: str | None
    backend_name: str = "local_stub"
    model_trace: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)
    call_budget: dict[str, Any] = field(default_factory=dict)
    grounding_checks: list[dict[str, Any]] = field(default_factory=list)
    schema_validation: dict[str, Any] = field(default_factory=dict)
    attempts: list[AttemptResult] = field(default_factory=list)
    final_decision: FinalDecision = field(default_factory=FinalDecision)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
