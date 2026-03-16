# Zero Node Addendum: Empirical Findings vs Current Harness

Date: March 15, 2026
Project: /Users/jwalinshah/projects/sentient-arena-officeqa

## 1) Does the Zero Node framing help?
Yes. The "Adaptive Grounding" framing is useful as an architectural target because it cleanly separates:
- candidate narrowing,
- grounded extraction/verification,
- adaptive improvement loops.

This matches the failure patterns we are seeing: numeric correctness is often high, but grounding reliability is weak.

## 2) Does current implementation match the proposed structure?
Partially.

### What already matches
- PRE-FILTER (partial): hint-constrained retrieval + parse-guided retrieval query over transformed TXT corpus.
- ACTOR_LOOP (partial): staged attempts with repair briefs (`max_attempts`, failed-family retries).
- Grounding contracts: strict schema + deterministic verification + row-level diagnostics.
- AUTO_REFINE (partial): trace-driven loop exists (3Q gate + matrix reporting), but updates are still human-in-the-loop.

### What is missing vs Zero Node target
- No true parallel multi-actor extraction consensus yet (single solver call per attempt).
- Temporal graph/critic is heuristic (year/topic checks), not a formal temporal constraint graph.
- Skill evolution is manual (prompt/config/code iteration), not autonomous skill synthesis/retrieval.

## 3) Measured findings (current state)
Primary result pattern:
- Numeric answers can be correct while strict grounding fails.
- Bottleneck is evidence-row reliability and acceptance, not parse validity.

### Latest 3Q strict run
Run: `runs/20260315_202401_contract-v3-gate3-rerun`
- numeric_accuracy_1pct: `1.0`
- strict_grounded_accuracy: `0.0`
- failure_breakdown: `{ "extraction": 3 }`
- parse_schema_ok: `3/3`
- solve_schema_ok: `0/3`
- grounded_rows_ok: `0/3`
- dropped_rows_total: `11`
- top_drop_reasons: `insufficient_grounding_signal`, `below_min_rows_after_filter`

Trace signals:
- attempt 1 frequently has partial/usable evidence behavior.
- attempt 2 often degrades into `missing_evidence_rows` and fails strict gate.
- dominant repair codes: `missing_evidence_rows`, `row_grounding_gate_failed`, `row_presence_gate_failed`.

### Baseline vs candidate matrix (earlier snapshot)
Report: `reports/20260315_200549_officeqa-baseline-vs-strict/matrix_comparison.json`
- baseline dev strict: `0.0`
- candidate dev strict: `0.0`
- baseline dev numeric: `1.0`
- candidate dev numeric: `1.0`
- promotion_passed: `false`

Interpretation: contract clarity improved, but strict-grounded rate did not yet improve.

## 4) What we changed recently (contract hardening)
- Added operation-based required row counts:
  - `lookup=1`, `difference=2`, `ratio=2`, `average=2`
- Added explicit solve gates:
  - `row_presence_gate`
  - `row_grounding_gate`
- Added required `matched_snippet` per evidence row.
- Added optional LLM grounding judge path (config-controlled) to complement deterministic checks.
- Added per-example trace inspector for complete step-level audit:
  - `scripts/inspect_example_trace.py`

## 5) Why strict is still failing
- The solver still inconsistently returns rows that survive strict grounding.
- Quality filter and strict min-row rules correctly reject weak provenance.
- In some retries, model behavior shifts toward fewer/empty usable rows despite numeric answer quality.

This is expected when moving from answer-only optimization to provenance-first optimization.

## 6) Recommended next experiments (high signal)
1. Stabilize second-attempt behavior:
- Preserve successful attempt-1 evidence rows unless directly invalidated.
- Penalize retries that remove previously valid rows.

2. Strengthen row contract prompting:
- Require one short `matched_snippet` tightly around value intersection.
- Add explicit negative examples of invalid row outputs.

3. Add verifier score decomposition to promotion gate:
- Promote only when `row_presence_gate_pass_rate` and `row_grounding_gate_pass_rate` improve.
- Keep numeric as secondary metric.

4. Add light actor split (before full multi-agent):
- one extraction call focused on provenance rows,
- one calculation call consuming only accepted rows.

## 7) Bottom line
- The Zero Node architecture is directionally aligned with what this benchmark needs.
- Current harness already embodies the core loop in simplified form.
- The key unresolved gap is robust provenance extraction under strict gates.
- Iteration should stay focused on row-gate reliability rather than broader architectural expansion.
