#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_eval import summarize


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute summary from a run directory.")
    parser.add_argument("--run", required=True, help="Path to runs/<run_id>")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()

    examples_path = run_dir / "examples.jsonl"
    if not examples_path.exists():
        raise FileNotFoundError(f"Missing examples.jsonl: {examples_path}")

    results = []
    with examples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    summary = summarize(results)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"Recomputed summary written: {run_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
