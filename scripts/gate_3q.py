#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _latest_run(prefix: str) -> Path:
    run_root = ROOT / "runs"
    candidates = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.endswith(prefix)])
    if not candidates:
        raise RuntimeError("No run directory found after run_eval.")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 3-question iteration gate and print top actions.")
    parser.add_argument("--config", default="config/openhands_openrouter.yaml")
    parser.add_argument("--dataset", default="data/officeqa_pro.jsonl")
    parser.add_argument("--tag", default="gate3q")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--limit",
        "3",
        "--tag",
        args.tag,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    run_dir = _latest_run(args.tag)

    report_path = run_dir / "trace_report.json"
    if not report_path.exists():
        raise RuntimeError(f"Missing trace report: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    top_actions = report.get("top_actions") or []
    reason_codes = report.get("top_failing_reason_codes") or {}
    top_questions = report.get("top_strict_failure_questions") or []
    edit_target = report.get("recommended_edit_target", "verifier")

    print(f"\n3Q gate complete: {run_dir}")
    print(f"Recommended next edit target: {edit_target}")
    print("Top failing reason codes:")
    if reason_codes:
        for i, (k, v) in enumerate(reason_codes.items(), start=1):
            if i > 3:
                break
            print(f"{i}. {k}: {v}")
    else:
        print("1. none")
    print("Top strict failure questions:")
    if top_questions:
        for i, q in enumerate(top_questions[:3], start=1):
            print(
                f"{i}. {q.get('question_id')} | failure={q.get('failure_type')} | severity={q.get('severity')}"
            )
    else:
        print("1. none")
    print("Top next actions:")
    for i, action in enumerate(top_actions[:3], start=1):
        print(f"{i}. {action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
