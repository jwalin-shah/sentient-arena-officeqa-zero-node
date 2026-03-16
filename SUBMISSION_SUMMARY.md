# Submission Summary: Zero Node (Challenge 0)

Date: March 15, 2026  
Project: `/Users/jwalinshah/projects/sentient-arena-officeqa`

This submission package includes:
- Proposal: [`PROPOSAL_ZERO_NODE.md`](/Users/jwalinshah/projects/sentient-arena-officeqa/PROPOSAL_ZERO_NODE.md)
- Empirical addendum: [`ZERO_NODE_FINDINGS_ADDENDUM.md`](/Users/jwalinshah/projects/sentient-arena-officeqa/ZERO_NODE_FINDINGS_ADDENDUM.md)

## 1) What is working (Measured)
Source artifacts:
- `runs/20260315_202401_contract-v3-gate3-rerun/summary.json`
- `reports/20260315_200549_officeqa-baseline-vs-strict/matrix_comparison.md`

Measured observations:
- Numeric correctness is strong on the latest strict 3Q rerun: `numeric_accuracy_1pct = 1.0`.
- Parse stage reliability is stable in the same run: `parse_schema_ok = 3/3` and `parse pass_rate = 1.0`.
- Baseline-vs-candidate dev snapshot shows numeric parity: baseline `100%`, candidate `100%`.

## 2) What is not working (Measured)
- Strict grounding remains the blocker: latest rerun `strict_grounded_accuracy = 0.0`.
- Solver-grounded outputs are not surviving strict row gates:
  - `solve_schema_ok = 0/3`
  - `grounded_rows_ok = 0/3`
  - `failure_breakdown = {"extraction": 3}`
- Dominant dropped-row reasons: `insufficient_grounding_signal`, `below_min_rows_after_filter`.
- Dev matrix still fails strict promotion (`promotion_passed = false`).

## 3) Why this matters
The current harness can often compute the right number, but cannot consistently provide strict, audit-grade provenance rows that survive grounding gates. In enterprise settings, this is a correctness-vs-auditability gap: answer-only accuracy is insufficient when evidence traceability is a first-class requirement.

## 4) Immediate next experiments (Targeted)
1. Stabilize second-attempt behavior by preserving valid attempt-1 rows unless explicitly invalidated.
2. Strengthen row-contract prompting with tighter `matched_snippet` requirements and explicit negative examples.
3. Decompose promotion gating with explicit row-presence and row-grounding pass-rate criteria (numeric secondary).
4. Add a light actor split: separate provenance-row extraction from downstream calculation using accepted rows only.

## Claim framing for judges
- `PROPOSAL_ZERO_NODE.md` is **target architecture and hypothesis**.
- `ZERO_NODE_FINDINGS_ADDENDUM.md` and cited run/report artifacts are **measured current state**.
- The key unresolved gap is robust provenance extraction under strict gates.
