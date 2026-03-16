#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trace_review import build_trace_report, render_trace_report_md


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate trace diagnostics for a run.")
    parser.add_argument("--run", required=True, help="Path to runs/<run_id>")
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()

    examples_path = run_dir / "examples.jsonl"
    if not examples_path.exists():
        raise FileNotFoundError(f"Missing examples.jsonl: {examples_path}")

    examples = []
    with examples_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    report = build_trace_report(examples)
    report_json = run_dir / "trace_report.json"
    report_md = run_dir / "trace_report.md"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    report_md.write_text(render_trace_report_md(report), encoding="utf-8")

    print(f"Trace report written: {report_json}")
    print(f"Trace report written: {report_md}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
