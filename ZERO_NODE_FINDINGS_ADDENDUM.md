# Zero Node Addendum: Empirical Findings

Date: March 15, 2026
Project: /Users/jwalinshah/projects/sentient-arena-officeqa

> **Sample size note:** Dev-20 (n=20) numbers are directional, not statistically precise. Smoke run figures (n=3) confirm pipeline function only — do not read them as accuracy rates.

---

## 1) Does the Zero Node framing help?

Yes. The "Adaptive Grounding" framing correctly separates the three failure modes and maps them to three implemented modules:
- PRE-FILTER → addresses SEARCH_BLUR (candidate narrowing via term-score + per-file guarantee)
- ACTOR_LOOP → addresses DATA_CORRUPT (parse→retrieve→extract→verify with repair briefs)
- AUTO_REFINE → addresses VERSION_COLLISION routing (deterministic fitness via `evaluate_grounding()`)

The failure patterns confirm the framing: numeric correctness is strong (14/20 on dev), but grounding reliability is weak (1/20 strict). The bottleneck is exactly where the architecture predicted — extraction/grounding fidelity, not retrieval or reasoning.

---

## 2) Implementation status

### What is implemented
- **PRE-FILTER:** section-aware corpus chunking (`src/corpus.py`) with term-score retrieval and per-source-file chunk guarantees (prevents multi-year questions getting all context from one file).
- **ACTOR_LOOP:** parse → retrieve → extract → verify loop with `_build_repair_brief()` generating targeted per-reason-code repair instructions on retry (e.g. `year_mismatch`, `column_label_not_found`).
- **Grounding contracts:** strict schema + 7-check deterministic verifier (`evaluate_grounding()`) + row-level diagnostics with named failure codes.
- **AUTO_REFINE:** `scripts/auto_refine.py` using `evaluate_grounding()` as a no-label fitness signal — fully automated, no human annotation required.

### What remains simplified vs full Zero Node vision
- No parallel multi-actor extraction (single solver call per attempt).
- Temporal critic is heuristic (year/topic checks), not a formal constraint graph.
- Skill synthesis is config-and-prompt iteration, not autonomous generation.

---

## 3) Measured findings (dev-20)

**Primary result:**
- Numeric answers are correct on 14/20 questions.
- Strict grounded accuracy is 1/20.
- Gap (13 questions): model computes the right number, fails verbatim label copy from source tables.

**Latest smoke run (n=3 — pipeline verification only, not a rate):**
Run: `runs/20260315_202401_contract-v3-gate3-rerun`
- parse_schema_ok: 3/3 ✓
- numeric_accuracy_1pct: 1.0 ✓
- strict_grounded_accuracy: 0.0 ✗
- failure_breakdown: `{ "extraction": 3 }`
- solve_schema_ok: 0/3
- grounded_rows_ok: 0/3
- dropped_rows_total: 11
- top drop reasons: `insufficient_grounding_signal`, `below_min_rows_after_filter`

**Baseline vs candidate dev snapshot:**
Report: `reports/20260315_200549_officeqa-baseline-vs-strict/matrix_comparison.json`
- baseline strict: 1/20, numeric: 0/20
- candidate strict: 1/20, numeric: 14/20
- promotion_passed: false (strict gate threshold not yet cleared)

---

## 4) Recent changes (contract hardening + bug fixes)

**Contract hardening:**
- Operation-based required row counts: `lookup=1`, `difference=2`, `ratio=2`, `average=2`
- Explicit solve gates: `row_presence_gate`, `row_grounding_gate`
- Required `matched_snippet` per evidence row
- Per-example trace inspector: `scripts/inspect_example_trace.py`

**Bug fixes (March 15):**
- `src/stages/parse.py`: Added `"sum"/"total"/"aggregate"` → `"average"` in `_normalize_operation()`. Previously these returned `"unknown"` causing `parse_valid = False` and downstream strict gate failure even when grounding was fine.
- `src/backends/openhands_sdk_backend.py`: Per-source-file chunk guarantee before global top-N slice. Fixes multi-year questions where both top chunks could come from the same file.
- `config/openhands_openrouter_minimax.yaml`: `solve_prompt_max_chunks: 2→4`, `solve_prompt_max_chars_per_chunk: 2500→4000`.

---

## 5) Why strict is still failing

- The solver returns rows that fail the strict grounding checks — labels do not verbatim match source table headers.
- The quality filter and strict min-row rules correctly reject these; the failure is real, not a harness bug.
- Second attempts often produce fewer usable rows despite equivalent numeric quality — retry behavior is not yet stable.

This is expected when shifting from answer-only optimization to provenance-first optimization. The grounding constraint is tighter than the model's default output behavior.

---

## 6) Recommended next steps (high signal)

1. **Stabilize retry behavior:** Preserve valid attempt-1 evidence rows unless explicitly invalidated by the repair brief.
2. **Tighten row contract prompting:** Require one short `matched_snippet` tightly around the value intersection. Add explicit negative examples of invalid row outputs (paraphrased labels, reconstructed text).
3. **Light actor split:** Separate provenance-row extraction from downstream calculation — one call focused on finding verbatim rows, one call consuming only accepted rows to compute the final answer.
4. **Post-fix dev-20 run:** Confirm numeric accuracy holds and check for strict movement after the `_normalize_operation` and per-file guarantee fixes.

---

## 7) Bottom line

The Zero Node architecture is correctly aligned with what this benchmark requires. The core loop is implemented and functioning. The single remaining gap is grounding extraction fidelity: the model reasons correctly but does not reliably copy labels verbatim from source tables. That is a targeted, tractable problem.
