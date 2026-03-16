#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def _latest_run_with_tag(tag: str) -> Path:
    run_root = ROOT / "runs"
    candidates = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.endswith(tag)])
    if not candidates:
        raise RuntimeError(f"No run directory found with tag suffix: {tag}")
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_examples(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "examples.jsonl"
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _ensure_artifacts(run_dir: Path) -> None:
    _run([sys.executable, "scripts/summarize_run.py", "--run", str(run_dir)])
    _run([sys.executable, "scripts/trace_review.py", "--run", str(run_dir)])


def _first_failing_contract_reason(example: dict[str, Any]) -> str:
    stage_order = ["parse", "retrieve", "solve", "grounding_precheck", "verify", "decision"]
    attempts = example.get("attempts") or []
    if not attempts:
        return "unknown"
    selected = int((example.get("final_decision") or {}).get("selected_attempt", 0) or 0)
    chosen = None
    if selected > 0:
        for a in attempts:
            if int(a.get("attempt_index", 0) or 0) == selected:
                chosen = a
                break
    if chosen is None:
        chosen = attempts[-1]
    stage_outcomes = (chosen or {}).get("stage_outcomes") or {}
    for stage in stage_order:
        out = stage_outcomes.get(stage) or {}
        errors = out.get("errors") or []
        if errors:
            return str(errors[0])
        if str(out.get("status", "")) == "fail":
            return f"{stage}_failed"
    checks = (chosen or {}).get("grounding_checks") or []
    for c in checks:
        for rc in (c.get("reason_codes") or []):
            return str(rc)
    return "unknown"


def _failure_severity(example: dict[str, Any]) -> int:
    score = 0
    for a in (example.get("attempts") or []):
        for out in ((a.get("stage_outcomes") or {}).values()):
            score += len((out or {}).get("errors") or [])
        for g in (a.get("grounding_checks") or []):
            score += len(g.get("reason_codes") or [])
    return score


def _top_grounding_failures(run_dir: Path, limit: int = 3) -> list[dict[str, Any]]:
    examples = _load_examples(run_dir)
    failing = [
        e for e in examples if not bool((e.get("scores") or {}).get("strict_grounded_correct", False))
    ]
    failing.sort(key=_failure_severity, reverse=True)
    top: list[dict[str, Any]] = []
    for e in failing[:limit]:
        top.append(
            {
                "question_id": e.get("question_id"),
                "question": e.get("raw_question"),
                "expected_answer": e.get("expected_answer"),
                "final_answer": e.get("final_answer"),
                "first_failing_contract_reason": _first_failing_contract_reason(e),
            }
        )
    return top


def _run_eval(config: str, dataset: str, tag: str, limit: int | None) -> Path:
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--config",
        config,
        "--dataset",
        dataset,
        "--tag",
        tag,
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    _run(cmd)
    return _latest_run_with_tag(tag)


def _promotion_gate(
    *,
    baseline_smoke: Path,
    candidate_smoke: Path,
    baseline_dev: Path,
    candidate_dev: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/promotion_gate.py",
        "--baseline-smoke",
        str(baseline_smoke),
        "--candidate-smoke",
        str(candidate_smoke),
        "--baseline-dev",
        str(baseline_dev),
        "--candidate-dev",
        str(candidate_dev),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    text = (proc.stdout or "").strip()
    if not text:
        return {
            "promotion_passed": False,
            "error": f"promotion_gate_no_output code={proc.returncode}",
        }
    try:
        out = json.loads(text)
    except Exception:
        out = {"promotion_passed": False, "error": "promotion_gate_invalid_json", "raw": text}
    out["exit_code"] = proc.returncode
    return out


def _format_pct(v: float) -> str:
    return f"{100.0 * float(v):.1f}%"


def _build_markdown(report: dict[str, Any]) -> str:
    ts = report.get("generated_at", "")
    base = report["baseline"]
    cand = report["candidate"]
    gate = report.get("promotion_gate", {})
    lines = [
        "# OfficeQA Experiment Milestone Snapshot",
        "",
        f"- Generated at: `{ts}`",
        f"- Baseline label: `{base.get('label')}`",
        f"- Candidate label: `{cand.get('label')}`",
        "",
        "## Base OfficeQA Performance (Baseline Runs Only)",
        f"- Smoke strict: `{_format_pct(base['smoke']['summary'].get('strict_grounded_accuracy', 0.0))}`",
        f"- Smoke numeric: `{_format_pct(base['smoke']['summary'].get('numeric_accuracy_1pct', 0.0))}`",
        f"- Dev strict: `{_format_pct(base['dev']['summary'].get('strict_grounded_accuracy', 0.0))}`",
        f"- Dev numeric: `{_format_pct(base['dev']['summary'].get('numeric_accuracy_1pct', 0.0))}`",
        "",
        "## Baseline vs Candidate",
        f"- Candidate smoke strict: `{_format_pct(cand['smoke']['summary'].get('strict_grounded_accuracy', 0.0))}`",
        f"- Candidate smoke numeric: `{_format_pct(cand['smoke']['summary'].get('numeric_accuracy_1pct', 0.0))}`",
        f"- Candidate dev strict: `{_format_pct(cand['dev']['summary'].get('strict_grounded_accuracy', 0.0))}`",
        f"- Candidate dev numeric: `{_format_pct(cand['dev']['summary'].get('numeric_accuracy_1pct', 0.0))}`",
        "",
        "## Strict-First Gate",
        f"- promotion_passed: `{gate.get('promotion_passed', False)}`",
        f"- smoke_improved: `{gate.get('smoke_improved', False)}`",
        f"- dev_non_regression: `{gate.get('dev_non_regression', False)}`",
        "",
        "## Failure Analysis",
        "- Baseline dev top failure breakdown: "
        f"`{base['dev']['summary'].get('failure_breakdown', {})}`",
        "- Candidate dev top failure breakdown: "
        f"`{cand['dev']['summary'].get('failure_breakdown', {})}`",
        "- Baseline dev top reason codes: "
        f"`{(base['dev']['trace_report'].get('top_failing_reason_codes') or {})}`",
        "- Candidate dev top reason codes: "
        f"`{(cand['dev']['trace_report'].get('top_failing_reason_codes') or {})}`",
        "- Candidate dev quality filter drop summary: "
        f"`{cand['dev']['summary'].get('quality_filter_summary', {})}`",
        "",
        "## Representative Failed Questions (Candidate Dev)",
    ]
    top = cand["dev"].get("top_grounding_failures") or []
    if top:
        for i, item in enumerate(top, start=1):
            lines += [
                f"{i}. question_id=`{item.get('question_id')}`",
                f"   question: {item.get('question')}",
                f"   expected_answer: `{item.get('expected_answer')}`",
                f"   final_answer: `{item.get('final_answer')}`",
                f"   first_failing_contract_reason: `{item.get('first_failing_contract_reason')}`",
            ]
    else:
        lines.append("1. none")
    lines += [
        "",
        "## Next Actions",
        "1. Fix the top candidate-dev strict failure reason first.",
        "2. Re-run smoke + dev matrix; only promote on strict improvement with no dev strict regression.",
        "3. Update one prompt/skill or one retrieval/verifier knob per iteration.",
    ]
    return "\n".join(lines) + "\n"


def _collect_run_block(label: str, split: str, run_dir: Path, config: str, dataset: str) -> dict[str, Any]:
    summary = _load_json(run_dir / "summary.json")
    trace_report = _load_json(run_dir / "trace_report.json")
    return {
        "label": label,
        "split": split,
        "run_dir": str(run_dir),
        "config": config,
        "dataset": dataset,
        "summary": summary,
        "trace_report": trace_report,
        "top_grounding_failures": _top_grounding_failures(run_dir, limit=3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline/candidate OfficeQA matrix and write milestone report.")
    parser.add_argument("--baseline-config", default="config/openhands_cerebras_officeqa_base.yaml")
    parser.add_argument("--candidate-config", default="config/openhands_cerebras_quality_eval.yaml")
    parser.add_argument("--smoke-dataset", default="data/splits/smoke.jsonl")
    parser.add_argument("--dev-dataset", default="data/splits/dev.jsonl")
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--dev-limit", type=int, default=20)
    parser.add_argument("--tag-prefix", default="matrix")
    parser.add_argument("--baseline-label", default="officeqa_base")
    parser.add_argument("--candidate-label", default="strict_candidate")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--reuse-baseline-smoke-run", default=None)
    parser.add_argument("--reuse-candidate-smoke-run", default=None)
    parser.add_argument("--reuse-baseline-dev-run", default=None)
    parser.add_argument("--reuse-candidate-dev-run", default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tags = {
        "baseline_smoke": f"{args.tag_prefix}_smoke_{args.baseline_label}",
        "candidate_smoke": f"{args.tag_prefix}_smoke_{args.candidate_label}",
        "baseline_dev": f"{args.tag_prefix}_dev_{args.baseline_label}",
        "candidate_dev": f"{args.tag_prefix}_dev_{args.candidate_label}",
    }

    if args.reuse_baseline_smoke_run and args.reuse_candidate_smoke_run and args.reuse_baseline_dev_run and args.reuse_candidate_dev_run:
        baseline_smoke_dir = Path(args.reuse_baseline_smoke_run).resolve()
        candidate_smoke_dir = Path(args.reuse_candidate_smoke_run).resolve()
        baseline_dev_dir = Path(args.reuse_baseline_dev_run).resolve()
        candidate_dev_dir = Path(args.reuse_candidate_dev_run).resolve()
    else:
        baseline_smoke_dir = _run_eval(
            args.baseline_config,
            args.smoke_dataset,
            run_tags["baseline_smoke"],
            args.smoke_limit,
        )
        candidate_smoke_dir = _run_eval(
            args.candidate_config,
            args.smoke_dataset,
            run_tags["candidate_smoke"],
            args.smoke_limit,
        )
        baseline_dev_dir = _run_eval(
            args.baseline_config,
            args.dev_dataset,
            run_tags["baseline_dev"],
            args.dev_limit,
        )
        candidate_dev_dir = _run_eval(
            args.candidate_config,
            args.dev_dataset,
            run_tags["candidate_dev"],
            args.dev_limit,
        )

    for d in [baseline_smoke_dir, candidate_smoke_dir, baseline_dev_dir, candidate_dev_dir]:
        _ensure_artifacts(d)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": {
            "label": args.baseline_label,
            "smoke": _collect_run_block(args.baseline_label, "smoke", baseline_smoke_dir, args.baseline_config, args.smoke_dataset),
            "dev": _collect_run_block(args.baseline_label, "dev", baseline_dev_dir, args.baseline_config, args.dev_dataset),
        },
        "candidate": {
            "label": args.candidate_label,
            "smoke": _collect_run_block(args.candidate_label, "smoke", candidate_smoke_dir, args.candidate_config, args.smoke_dataset),
            "dev": _collect_run_block(args.candidate_label, "dev", candidate_dev_dir, args.candidate_config, args.dev_dataset),
        },
        "promotion_gate": _promotion_gate(
            baseline_smoke=baseline_smoke_dir,
            candidate_smoke=candidate_smoke_dir,
            baseline_dev=baseline_dev_dir,
            candidate_dev=candidate_dev_dir,
        ),
    }

    out_dir = (ROOT / args.reports_dir / f"{stamp}_{args.tag_prefix}").resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "matrix_comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    (out_dir / "matrix_comparison.md").write_text(_build_markdown(report), encoding="utf-8")

    print(f"Report directory: {out_dir}")
    print(f"JSON: {out_dir / 'matrix_comparison.json'}")
    print(f"Markdown: {out_dir / 'matrix_comparison.md'}")
    print(json.dumps(report.get("promotion_gate", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
