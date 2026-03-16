#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Databricks OfficeQA CSV to harness JSONL.")
    parser.add_argument("--input", required=True, help="Path to officeqa_*.csv")
    parser.add_argument("--output", required=True, help="Path to output .jsonl")
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with in_path.open("r", encoding="utf-8", newline="") as fin, out_path.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            record = {
                "question_id": row.get("uid", "").strip(),
                "question": row.get("question", "").strip(),
                "expected_answer": row.get("answer", "").strip(),
                "metadata": {
                    "difficulty": row.get("difficulty", "").strip(),
                    "source_docs": row.get("source_docs", "").strip(),
                    "source_files": row.get("source_files", "").strip(),
                },
            }
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1

    print(f"Converted {count} rows")
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
