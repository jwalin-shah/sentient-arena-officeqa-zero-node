#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_ids(path: Path, rows: list[dict]) -> None:
    ids = [str(r.get("question_id", "")) for r in rows]
    path.write_text(json.dumps(ids, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic smoke/dev/holdout splits.")
    parser.add_argument("--dataset", default="data/officeqa_pro.jsonl")
    parser.add_argument("--smoke", type=int, default=3)
    parser.add_argument("--dev", type=int, default=20)
    parser.add_argument("--holdout", type=int, default=20)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ds = Path(args.dataset)
    if not ds.is_absolute():
        ds = (root / ds).resolve()

    rows = sorted(load_jsonl(ds), key=lambda r: str(r.get("question_id", "")))
    smoke_rows = rows[: args.smoke]
    dev_rows = rows[args.smoke : args.smoke + args.dev]
    holdout_rows = rows[args.smoke + args.dev : args.smoke + args.dev + args.holdout]

    out = root / "data" / "splits"
    write_jsonl(out / "smoke.jsonl", smoke_rows)
    write_jsonl(out / "dev.jsonl", dev_rows)
    write_jsonl(out / "holdout.jsonl", holdout_rows)
    write_ids(out / "smoke_ids.json", smoke_rows)
    write_ids(out / "dev_ids.json", dev_rows)
    write_ids(out / "holdout_ids.json", holdout_rows)

    print(f"Wrote splits to {out}")
    print(f"smoke={len(smoke_rows)} dev={len(dev_rows)} holdout={len(holdout_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
