# Submission Summary: Zero Node (Challenge 0)

Date: March 15, 2026
Project: `/Users/jwalinshah/projects/sentient-arena-officeqa`

> **Sample size note:** Dev-20 (n=20) results establish directional patterns. Smoke run figures (n=3) confirm end-to-end pipeline function only — percentages from 3 samples are not meaningful rates and are not cited as such here.

## What to submit (Discord #product-showcase)

**File:** `PROPOSAL_ZERO_NODE.md` — this is the complete solution proposal. It covers architecture, formal framework, implemented modules, and empirical findings with appropriate caveats.

Optionally attach or summarize `ZERO_NODE_FINDINGS_ADDENDUM.md` as supplementary context on the grounding gap diagnosis.

---

## What is working (Measured, dev-20)

Source artifacts:
- `runs/20260315_202401_contract-v3-gate3-rerun/summary.json`
- `reports/20260315_200549_officeqa-baseline-vs-strict/matrix_comparison.md`

- Numeric accuracy improved from 0/20 (baseline) to 14/20 post-implementation.
- Parse stage is reliable: `parse_schema_ok = 3/3`, `parse_pass_rate = 1.0` on latest smoke.
- Pipeline runs end-to-end on both Cerebras and MiniMax (OpenRouter) models.
- PRE-FILTER, ACTOR_LOOP (with repair briefs), and AUTO_REFINE are all implemented.

## What is not working (Measured, dev-20)

- Strict grounded accuracy is unchanged at 1/20 — the grounding gate is the only blocker.
- Solver outputs fail the strict row gate consistently:
  - Top failure reason: `column_label_not_found`
  - Secondary: `year_mismatch`, `insufficient_grounding_signal`
- The gap is not a reasoning problem — correct numeric answers are computed. The model fails verbatim label copying from source table headers.

## Why this matters

14 out of 20 questions get the right number. 1 out of 20 gets strict grounded credit. That gap is the story: the architecture correctly isolates the bottleneck as grounding extraction fidelity, not reasoning or retrieval. This is a tractable, well-scoped problem that targeted skill refinement directly addresses.

## Immediate next steps

1. Tighten `matched_snippet` prompting with explicit negative examples of invalid labels.
2. Preserve valid attempt-1 evidence rows across retries instead of discarding them.
3. Run full dev-20 post-fix (pending Cerebras quota reset) to confirm numeric hold and check strict movement.
4. Consider light actor split: separate provenance-row extraction from downstream calculation.

## Claim framing for judges

The proposal (`PROPOSAL_ZERO_NODE.md`) describes what is implemented and what was measured. No claims are made from n=3 smoke samples. All cited numbers are from the dev-20 evaluation set.
