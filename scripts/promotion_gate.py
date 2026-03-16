#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing summary.json: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _metric(summary: dict, key: str) -> float:
    v = summary.get(key, 0.0)
    return float(v) if isinstance(v, (int, float)) else 0.0


def _tie_break(summary: dict) -> tuple[float, float, float]:
    grounded_row_match = float(((summary.get("stage_stats") or {}).get("grounded_rows_ok", 0)))
    numeric = _metric(summary, "numeric_accuracy_1pct")
    avg_calls = float(((summary.get("telemetry") or {}).get("avg_llm_calls_per_example", 1e9)))
    return (grounded_row_match, numeric, -avg_calls)


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict-first promotion gate across smoke/dev runs.")
    parser.add_argument("--baseline-smoke", required=True, help="runs/<id> baseline smoke run dir")
    parser.add_argument("--candidate-smoke", required=True, help="runs/<id> candidate smoke run dir")
    parser.add_argument("--baseline-dev", required=True, help="runs/<id> baseline dev run dir")
    parser.add_argument("--candidate-dev", required=True, help="runs/<id> candidate dev run dir")
    args = parser.parse_args()

    bs = _load_summary(Path(args.baseline_smoke))
    cs = _load_summary(Path(args.candidate_smoke))
    bd = _load_summary(Path(args.baseline_dev))
    cd = _load_summary(Path(args.candidate_dev))

    b_smoke = _metric(bs, "strict_grounded_accuracy")
    c_smoke = _metric(cs, "strict_grounded_accuracy")
    b_dev = _metric(bd, "strict_grounded_accuracy")
    c_dev = _metric(cd, "strict_grounded_accuracy")

    smoke_improved = c_smoke > b_smoke
    dev_non_regression = c_dev >= b_dev

    tie_break_baseline = _tie_break(bs)
    tie_break_candidate = _tie_break(cs)

    passed = smoke_improved and dev_non_regression

    print(json.dumps({
        "baseline_smoke_strict": b_smoke,
        "candidate_smoke_strict": c_smoke,
        "baseline_dev_strict": b_dev,
        "candidate_dev_strict": c_dev,
        "smoke_improved": smoke_improved,
        "dev_non_regression": dev_non_regression,
        "tie_break_baseline": {
            "grounded_rows_ok": tie_break_baseline[0],
            "numeric_accuracy_1pct": tie_break_baseline[1],
            "avg_llm_calls_per_example": -tie_break_baseline[2],
        },
        "tie_break_candidate": {
            "grounded_rows_ok": tie_break_candidate[0],
            "numeric_accuracy_1pct": tie_break_candidate[1],
            "avg_llm_calls_per_example": -tie_break_candidate[2],
        },
        "promotion_passed": passed,
    }, indent=2))

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
