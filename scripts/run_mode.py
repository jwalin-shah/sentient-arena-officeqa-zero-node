#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed eval mode: smoke/dev/holdout.")
    parser.add_argument("--mode", choices=["smoke", "dev", "holdout"], required=True)
    parser.add_argument("--config", default="config/openhands_cerebras.yaml")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ds = root / "data" / "splits" / f"{args.mode}.jsonl"
    if not ds.exists():
        raise FileNotFoundError(
            f"Missing split file: {ds}. Run scripts/create_splits.py first."
        )

    tag = args.tag or f"{args.mode}-run"
    cmd = [
        sys.executable,
        "scripts/run_eval.py",
        "--config",
        args.config,
        "--dataset",
        str(ds),
        "--tag",
        tag,
    ]
    subprocess.run(cmd, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
