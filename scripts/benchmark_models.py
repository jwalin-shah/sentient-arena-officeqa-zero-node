#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_eval import load_jsonl, summarize
from src.env import load_env_file
from src.pipeline import process_example


DEFAULT_MODELS = [
    "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
    "openrouter/qwen/qwen3-coder:free",
]


def sanitize_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower() or "model"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def run_one_model(
    *,
    model: str,
    base_config: dict[str, Any],
    dataset_records: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    cfg = deepcopy(base_config)
    cfg.setdefault("backend", {})
    cfg["backend"]["name"] = "openhands_sdk"
    cfg["backend"].setdefault("openhands", {})
    cfg["backend"]["openhands"]["model"] = model

    results: list[dict[str, Any]] = []
    examples_path = out_dir / "examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as f:
        for ex in dataset_records:
            res = process_example(ex, cfg).to_dict()
            results.append(res)
            f.write(json.dumps(res, ensure_ascii=True) + "\n")

    summary = summarize(results)
    summary["model"] = model
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "config.snapshot.json", cfg)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OfficeQA harness across multiple OpenRouter models.")
    parser.add_argument("--config", default="config/openhands_openrouter.yaml", help="Base YAML config")
    parser.add_argument("--dataset", default=None, help="Override dataset path")
    parser.add_argument("--limit", type=int, default=3, help="Questions per model")
    parser.add_argument("--ids", default=None, help="Comma-separated question IDs")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model IDs")
    parser.add_argument("--tag", default="model-bench", help="Run tag")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Export it before running benchmark_models.py")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (ROOT / cfg_path).resolve()
    cfg = load_config(cfg_path)

    ds_cfg = cfg.get("dataset_path", "data/sample_questions.jsonl")
    ds_path = Path(args.dataset) if args.dataset else Path(ds_cfg)
    if not ds_path.is_absolute():
        ds_path = (ROOT / ds_path).resolve()

    records = load_jsonl(ds_path)
    if args.ids:
        keep = {x.strip() for x in args.ids.split(",") if x.strip()}
        records = [r for r in records if str(r.get("question_id")) in keep]
    records = records[: args.limit]

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise ValueError("No models provided.")

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitize_slug(args.tag)}"
    root_run = ROOT / "runs" / run_id
    root_run.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    for idx, model in enumerate(models, start=1):
        model_dir = root_run / f"{idx:02d}_{sanitize_slug(model)}"
        model_dir.mkdir(parents=True, exist_ok=False)
        summary = run_one_model(
            model=model,
            base_config=cfg,
            dataset_records=records,
            out_dir=model_dir,
        )
        rows.append(summary)

    rows_sorted = sorted(rows, key=lambda x: (x.get("accuracy", 0.0), x.get("correct_examples", 0)), reverse=True)
    comparison = {
        "run_id": run_id,
        "dataset_path": str(ds_path),
        "num_examples_per_model": len(records),
        "models": rows_sorted,
        "winner": rows_sorted[0]["model"] if rows_sorted else None,
    }
    write_json(root_run / "comparison.json", comparison)

    md_lines = [
        f"# Model Benchmark: {run_id}",
        "",
        f"- Dataset: `{ds_path}`",
        f"- Examples/model: `{len(records)}`",
        "",
        "| Rank | Model | Accuracy | Correct | Total |",
        "|---|---|---:|---:|---:|",
    ]
    for i, row in enumerate(rows_sorted, start=1):
        md_lines.append(
            f"| {i} | `{row['model']}` | {row['accuracy']:.4f} | {row['correct_examples']} | {row['total_examples']} |"
        )
    (root_run / "comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Benchmark root run: {root_run}")
    print(f"Winner: {comparison['winner']}")
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
