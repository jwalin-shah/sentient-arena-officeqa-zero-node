#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_examples(run_dir: Path) -> list[dict[str, Any]]:
    p = run_dir / "examples.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing examples.jsonl: {p}")
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _short(text: Any, n: int) -> str:
    s = str(text or "")
    if len(s) <= n:
        return s
    return s[:n] + "...<truncated>"


def _print_attempt(a: dict[str, Any], show_prompts: bool, show_raw: bool) -> None:
    idx = a.get("attempt_index")
    scores = a.get("scores") or {}
    schema = a.get("schema_validation") or {}
    stage_outcomes = a.get("stage_outcomes") or {}
    mt = a.get("model_trace") or {}
    st = mt.get("stage_trace") or {}
    budget = a.get("attempt_budget") or {}

    print(f"\n  Attempt {idx}")
    print(f"  - scores: numeric={scores.get('numeric_correct')} strict={scores.get('strict_grounded_correct')} rel_err={scores.get('relative_error')}")
    print(f"  - schema: parse_valid={schema.get('parse_valid')} solve_valid={schema.get('solve_valid')}")
    print(f"  - llm_budget: calls={budget.get('llm_calls', 0)} parse_calls={budget.get('parse_calls', 0)} solve_calls={budget.get('solve_calls', 0)} prompt_chars={budget.get('prompt_chars', 0)} chunks_sent={budget.get('chunks_sent', 0)}")
    print(f"  - stop_reason: {a.get('stop_reason', '')}")

    solve_out = stage_outcomes.get("solve") or {}
    solve_art = solve_out.get("artifacts") or {}
    print(
        "  - solve_gate:"
        f" row_presence={solve_art.get('row_presence_gate')}"
        f" row_grounding={solve_art.get('row_grounding_gate')}"
        f" required_min={solve_art.get('required_min_rows')}"
        f" allowed_max={solve_art.get('allowed_max_rows')}"
        f" evidence_row_count={solve_art.get('evidence_row_count')}"
    )

    print("  - stage_outcomes:")
    for name in ["parse", "retrieve", "extract", "calculate", "solve", "grounding_precheck", "verify", "decision"]:
        out = stage_outcomes.get(name) or {}
        print(
            f"    - {name}: status={out.get('status')} timing_ms={out.get('timing_ms', 0):.1f} errors={out.get('errors', [])}"
        )

    grounding_checks = a.get("grounding_checks") or []
    print(f"  - grounding_checks_count: {len(grounding_checks)}")
    if grounding_checks:
        for i, c in enumerate(grounding_checks[:5], start=1):
            print(
                f"    - row[{i-1}] matched={c.get('matched')} reasons={c.get('reason_codes', [])} "
                f"source={c.get('source_file')} raw_value_found={c.get('raw_value_found')} "
                f"row_label_found={c.get('row_label_found')} col_label_found={c.get('column_label_found')} "
                f"snippet_found={c.get('matched_snippet_found')}"
            )

    row_quality = (st.get("solve_row_quality") or {})
    if row_quality:
        print(
            "  - row_quality:"
            f" input={row_quality.get('input_count')} dedup={row_quality.get('dedup_count')}"
            f" kept={row_quality.get('kept_count')} dropped={row_quality.get('dropped_count')}"
            f" min_rows={row_quality.get('min_rows')} max_rows={row_quality.get('max_rows')}"
        )
        for d in (row_quality.get("row_diagnostics") or [])[:5]:
            print(
                f"    - diag kept={d.get('kept')} drop_reason={d.get('drop_reason')} "
                f"score={d.get('score')} reasons={d.get('reason_codes', [])}"
            )

    judge = st.get("llm_grounding_judge") or {}
    if judge:
        judge_obj = judge.get("backend_response_structured") or {}
        row_checks = judge_obj.get("row_checks") or []
        print(f"  - llm_grounding_judge: rows={len(row_checks)} overall={judge_obj.get('overall_grounded')}")
        for rc in row_checks[:5]:
            print(
                f"    - llm row_index={rc.get('row_index')} grounded={rc.get('grounded')} "
                f"confidence={rc.get('confidence')} reasons={rc.get('reason_codes', [])}"
            )

    parse_req = st.get("parse_backend_request") or {}
    solve_req = st.get("solve_backend_request") or {}
    if parse_req:
        print(f"  - parse_model: {parse_req.get('model')} tokens={parse_req.get('max_output_tokens')}")
    if solve_req:
        print(f"  - solve_model: {solve_req.get('model')} tokens={solve_req.get('max_output_tokens')}")

    if show_prompts:
        if parse_req:
            up = ((parse_req.get("prompt_payload") or {}).get("user_prompt") or "")
            print(f"  - parse_user_prompt:\n{_short(up, 3000)}")
        if solve_req:
            up = ((solve_req.get("prompt_payload") or {}).get("user_prompt") or "")
            print(f"  - solve_user_prompt:\n{_short(up, 4000)}")

    if show_raw:
        print(f"  - parse_raw_response:\n{_short(st.get('parse_backend_response_raw', ''), 3000)}")
        print(f"  - solve_raw_response:\n{_short(st.get('solve_backend_response_raw', ''), 4000)}")
        print(f"  - solve_structured:\n{json.dumps(st.get('solve_backend_response_structured', {}), ensure_ascii=True)[:4000]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect per-example attempt trace for a run.")
    parser.add_argument("--run", required=True, help="Path to runs/<run_id>")
    parser.add_argument("--qid", default=None, help="Optional question_id filter")
    parser.add_argument("--show-prompts", action="store_true", help="Print prompt payload excerpts")
    parser.add_argument("--show-raw", action="store_true", help="Print raw/structured model response excerpts")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()

    examples = _load_examples(run_dir)
    if args.qid:
        examples = [e for e in examples if str(e.get("question_id")) == str(args.qid)]

    if not examples:
        print("No examples matched.")
        return 0

    print(f"Run: {run_dir}")
    print(f"Examples selected: {len(examples)}")
    for ex in examples:
        qid = ex.get("question_id")
        scores = ex.get("scores") or {}
        fd = ex.get("final_decision") or {}
        print("\n============================================================")
        print(f"Question ID: {qid}")
        print(f"Question: {_short(ex.get('raw_question', ''), 300)}")
        print(
            f"Final: answer={ex.get('final_answer')} expected={ex.get('expected_answer')} "
            f"numeric={scores.get('numeric_correct')} strict={scores.get('strict_grounded_correct')} "
            f"failure_type={ex.get('failure_type')}"
        )
        print(
            f"Decision: accepted={fd.get('accepted')} reason={fd.get('reason')} "
            f"selected_attempt={fd.get('selected_attempt')}"
        )
        cb = ex.get("call_budget") or {}
        print(
            "Run call budget:"
            f" attempts={cb.get('attempts_executed', 0)}"
            f" llm_calls={cb.get('total_llm_calls', 0)}"
            f" parse_calls={cb.get('parse_calls', 0)}"
            f" solve_calls={cb.get('solve_calls', 0)}"
            f" parse_prompt_chars={cb.get('parse_prompt_chars_total', 0)}"
            f" solve_prompt_chars={cb.get('solve_prompt_chars_total', 0)}"
            f" solve_prompt_chunks={cb.get('solve_prompt_chunks_total', 0)}"
        )
        for a in (ex.get("attempts") or []):
            _print_attempt(a, show_prompts=args.show_prompts, show_raw=args.show_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
