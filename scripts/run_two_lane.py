#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def _latest_run(tag: str) -> Path:
    run_root = ROOT / "runs"
    candidates = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.endswith(tag)])
    if not candidates:
        raise RuntimeError(f"No run found for tag suffix: {tag}")
    return candidates[-1]


def _load_summary(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "summary.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _strict(summary: dict[str, Any]) -> float:
    v = summary.get("strict_grounded_accuracy", 0.0)
    return float(v) if isinstance(v, (int, float)) else 0.0


def _numeric(summary: dict[str, Any]) -> float:
    v = summary.get("numeric_accuracy_1pct", 0.0)
    return float(v) if isinstance(v, (int, float)) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-lane runner: fast smoke gate -> optional quality matrix.")
    parser.add_argument("--baseline-config", default="config/openhands_cerebras_officeqa_base.yaml")
    parser.add_argument("--fast-candidate-config", default="config/openhands_cerebras_fast_iter.yaml")
    parser.add_argument("--quality-candidate-config", default="config/openhands_cerebras_quality_eval.yaml")
    parser.add_argument("--smoke-dataset", default="data/splits/smoke.jsonl")
    parser.add_argument("--dev-dataset", default="data/splits/dev.jsonl")
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--dev-limit", type=int, default=20)
    parser.add_argument("--tag-prefix", default="two-lane")
    parser.add_argument("--min-strict-delta", type=float, default=0.000001)
    args = parser.parse_args()

    baseline_tag = f"{args.tag_prefix}_fast_smoke_baseline"
    candidate_tag = f"{args.tag_prefix}_fast_smoke_candidate"

    # Lane 1: fast smoke baseline
    _run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--config",
            args.baseline_config,
            "--dataset",
            args.smoke_dataset,
            "--limit",
            str(args.smoke_limit),
            "--tag",
            baseline_tag,
        ]
    )
    baseline_run = _latest_run(baseline_tag)

    # Lane 1: fast smoke candidate
    _run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--config",
            args.fast_candidate_config,
            "--dataset",
            args.smoke_dataset,
            "--limit",
            str(args.smoke_limit),
            "--tag",
            candidate_tag,
        ]
    )
    candidate_run = _latest_run(candidate_tag)

    baseline_summary = _load_summary(baseline_run)
    candidate_summary = _load_summary(candidate_run)
    strict_delta = _strict(candidate_summary) - _strict(baseline_summary)

    gate = {
        "baseline_smoke_run": str(baseline_run),
        "candidate_smoke_run": str(candidate_run),
        "baseline_strict": _strict(baseline_summary),
        "candidate_strict": _strict(candidate_summary),
        "baseline_numeric": _numeric(baseline_summary),
        "candidate_numeric": _numeric(candidate_summary),
        "strict_delta": strict_delta,
        "promote_to_quality": strict_delta >= float(args.min_strict_delta),
    }

    print("Fast lane gate:")
    print(json.dumps(gate, indent=2))

    if not gate["promote_to_quality"]:
        print("Skipping quality lane (strict smoke did not improve).")
        return 0

    # Lane 2: quality matrix only if fast gate passes
    _run(
        [
            sys.executable,
            "scripts/run_experiment_matrix.py",
            "--baseline-config",
            args.baseline_config,
            "--candidate-config",
            args.quality_candidate_config,
            "--smoke-dataset",
            args.smoke_dataset,
            "--dev-dataset",
            args.dev_dataset,
            "--smoke-limit",
            str(args.smoke_limit),
            "--dev-limit",
            str(args.dev_limit),
            "--tag-prefix",
            f"{args.tag_prefix}-quality",
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
