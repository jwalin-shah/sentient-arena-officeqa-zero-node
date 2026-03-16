#!/usr/bin/env python3
import json
from pathlib import Path
import sys

def load_summary(run_dir: Path):
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/compare_runs.py <run_dir_1> <run_dir_2>")
        sys.exit(1)

    dir1 = Path(sys.argv[1])
    dir2 = Path(sys.argv[2])

    s1 = load_summary(dir1)
    s2 = load_summary(dir2)

    if not s1 or not s2:
        print("Error: Could not load summary.json from one or both directories.")
        sys.exit(1)

    print(f"{'Metric':<25} | {'Baseline':<10} | {'Experimental':<10} | {'Delta':<10}")
    print("-" * 65)
    
    metrics = [
        ("Numeric Accuracy", "numeric_accuracy"),
        ("Strict Accuracy", "strict_accuracy"),
        ("Total Questions", "total_questions"),
    ]

    for label, key in metrics:
        v1 = s1.get(key, 0)
        v2 = s2.get(key, 0)
        delta = v2 - v1 if isinstance(v1, (int, float)) else 0
        print(f"{label:<25} | {v1:<10.4f} | {v2:<10.4f} | {delta:<+10.4f}")

    print("\nFailure Counts (Lower is better):")
    f1 = s1.get("failures", {})
    f2 = s2.get("failures", {})
    all_keys = set(f1.keys()) | set(f2.keys())
    for k in sorted(all_keys):
        v1 = f1.get(k, 0)
        v2 = f2.get(k, 0)
        delta = v2 - v1
        print(f"{k:<25} | {v1:<10} | {v2:<10} | {delta:<+10}")

if __name__ == "__main__":
    main()
