# Empirical Methodology: Grounded Reasoning Development

## 1. Why this file exists
This document is a running experiment ledger. We keep every major method change with:
- what we changed,
- why we changed it,
- what we expected,
- what happened.
- decision: keep / revert / iterate.

The goal is to avoid repeating failed ideas and to make performance deltas auditable.

## 1.1 Reporting split
- `METHODOLOGY.md` is the mutable chronological ledger.
- `reports/<timestamp>_<tag>/matrix_comparison.md` is immutable per-milestone evidence.

## 2. Current Method (strict, trace-first)
1. Parse question into structured fields (`metric`, `entity_category`, `time_period`, `operation`, `expected_answer_type`).
2. Retrieve only OfficeQA hinted files (`metadata.source_files`) in strict runs.
3. Chunk and score text inside hinted files only, using parse-guided retrieval query.
4. Solve with strict JSON schema (`final_answer`, `evidence_rows[]`).
5. Run deterministic grounding checks + full verification once per attempt.
6. Gate final answer on `strict_plus_numeric` (numeric tolerance + grounded evidence).
7. Retry only failed stage families up to configured `max_attempts`.

## 2.1 Baseline definitions
- Primary baseline (default): **pure OfficeQA base prompt setup** via `config/openhands_cerebras_officeqa_base.yaml`.
- Optional secondary baseline: Arena-packaged prompt + 3 skills via `config/openhands_cerebras_arena_base.yaml`.
- Candidate: strict quality setup via `config/openhands_cerebras_quality_eval.yaml`.

## 3. Iteration lifecycle
| Phase | Activity | Goal |
| :--- | :--- | :--- |
| Research | Review `summary.json` + `trace_report.md` | Find highest-impact failure family |
| Change | Edit one method/prompt/skill at a time | Keep attribution clean |
| Execute | Run 1-3 questions first, then 20-question dev set | Fast signal, then stability signal |
| Audit | Compare run telemetry + strict score deltas | Verify real improvement |

## 4. Experiment log

### Step 0: Baseline (system integrity)
- Method: `local_stub`.
- Why: verify data plumbing and scoring.
- Outcome: infra works, no model intelligence.

### Step 1: LLM enablement
- Method: OpenHands SDK backend with structured output.
- Why: establish real model baseline.
- Outcome: parse/solve worked, but strict grounding weak.

### Step 2: Strict contracts
- Method: schema-valid parse/solve required in strict mode.
- Why: eliminate silent/fuzzy acceptance.
- Outcome: better failure localization (`parse` vs `extraction` vs `retrieval`).

### Step 3: Adaptive retries
- Method: attempt loop, retry failed stage families only.
- Why: avoid full reruns and improve debug signal.
- Outcome: clearer stop reasons and attempt-level traces.

### Step 4: Context cap correction
- Method: explicit solve prompt limits + timeout controls.
- Why: prevent "frozen" runs caused by oversized context.
- Outcome: predictable latency; run completion improved.

### Step 5: Hint-only strict retrieval
- Method: `max_files_from_year_fallback=0` for OfficeQA Pro.
- Why: Pro dataset has complete `source_files` hints; stay benchmark-faithful.
- Outcome: retrieval scope aligned with official hint discipline.

### Step 6: Parse-guided retrieval (current)
- Method: retrieval query now combines raw question + parsed fields, with global chunk cap.
- Why: reduce irrelevant chunks and improve literal evidence hit rate.
- Expected: fewer `raw_value_not_found` / `row_label_not_found` strict failures.
- Status: active; evaluate via 3Q smoke then 20Q dev.

### Step 7: Baseline-to-strict experiment matrix
- Method: run baseline + candidate on fixed smoke/dev splits with one command.
- Why: make baseline-vs-candidate comparison reproducible and auditable.
- Output: compact JSON + Markdown snapshot under `reports/`.
- Status: active.

### Step 8: Contract v3 (row gates + matched snippet)
- Method:
  - enforce operation-based required rows (`lookup=1`, `difference=2`, `ratio=2`, `average=2`)
  - require `matched_snippet` per evidence row
  - split solve acceptance into `row_presence_gate` and `row_grounding_gate`
- Why: reduce brittleness from extra/noisy rows and make strict failures explicit.
- Expected: better failure attribution and fewer silent “numeric-right but strict-fail” cases.
- Status: active; iterate via 3Q smoke then 20Q dev.

## 5. Base OfficeQA performance (baseline-only)
Use baseline runs only (no candidate mixing) to populate this section.

Current proxy baseline evidence (existing runs):
- Smoke strict: `33.3%` (`runs/20260315_192622_ab2-baseline-smoke`)
- Smoke numeric: `33.3%` (`runs/20260315_192622_ab2-baseline-smoke`)
- Dev strict: `5.0%` (`runs/20260315_192740_ab2-baseline-dev20`)
- Dev numeric: `5.0%` (`runs/20260315_192740_ab2-baseline-dev20`)

Note:
- These are from the previous proxy baseline config, not the new pure-OfficeQA baseline config.
- Recompute this section from `scripts/run_experiment_matrix.py` outputs after running the new baseline config.

## 6. Metrics that decide if a change is accepted
- `numeric_accuracy_1pct`
- `strict_grounded_accuracy` (primary)
- `accepted_final_rate`
- `failure_breakdown`
- telemetry:
  - `llm_calls_total`
  - `avg_llm_calls_per_example`
  - `solve_prompt_chunks_total`
  - `solve_prompt_chars_total`

## 7. Tools used for empirical audit
- `scripts/run_eval.py`
- `scripts/summarize_run.py`
- `scripts/trace_review.py`
- `scripts/run_experiment_matrix.py`
- `scripts/promotion_gate.py`

## 8. Decision policy
- Promote a method only if strict-grounded metrics improve on smoke and do not regress on dev.
- Keep one variable changed per experiment whenever possible.
