#!/usr/bin/env python3
"""AUTO_REFINE: Grounding-driven skill synthesis loop.

Algorithm:
1. Load a completed run directory (output of run_mode.py / run_eval.py)
2. Scan all trace files for grounding_checks with matched=False
3. Classify failures by dominant reason_codes -> failure_class
4. For each new failure_class not already in skill library:
   - Call LLM with failure example + reason code + existing skill text as context
   - Ask it to synthesize a targeted repair skill
   - Write candidate skill to arena_official/skills/auto_skill_{failure_class}.md
5. Run run_mode.py --mode smoke with the new skill added to config
6. Compare grounding pass rate before/after. If improved: keep; else: discard.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

FAILURE_CLASSES: dict[frozenset[str], str] = {
    frozenset(["matched_snippet_not_found", "matched_snippet_missing"]): "verbatim_copy_failure",
    frozenset(["raw_value_not_found"]): "value_formatting_failure",
    frozenset(["row_label_not_found", "column_label_not_found"]): "label_copy_failure",
    frozenset(["year_mismatch"]): "temporal_alignment_failure",
    frozenset(["topic_mismatch"]): "topic_alignment_failure",
}

# Map individual codes to their class
_CODE_TO_CLASS: dict[str, str] = {}
for _codes, _cls in FAILURE_CLASSES.items():
    for _code in _codes:
        _CODE_TO_CLASS[_code] = _cls


def _classify_codes(reason_codes: list[str]) -> str | None:
    """Return the dominant failure class for a list of reason_codes."""
    counts: Counter[str] = Counter()
    for code in reason_codes:
        cls = _CODE_TO_CLASS.get(code)
        if cls:
            counts[cls] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _scan_run_dir(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Scan examples.jsonl for failing grounding rows.

    Returns: {failure_class: [list of failure examples]}
    """
    examples_path = run_dir / "examples.jsonl"
    if not examples_path.exists():
        raise FileNotFoundError(f"No examples.jsonl in {run_dir}")

    failures_by_class: dict[str, list[dict[str, Any]]] = {}
    with examples_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            attempts = ex.get("attempts", [])
            for attempt in attempts:
                grounding_checks = attempt.get("grounding_checks") or []
                for check in grounding_checks:
                    if check.get("matched"):
                        continue
                    reason_codes = check.get("reason_codes", [])
                    cls = _classify_codes(reason_codes)
                    if cls is None:
                        continue
                    entry = {
                        "question_id": ex.get("question_id", ""),
                        "question": ex.get("raw_question", ""),
                        "reason_codes": reason_codes,
                        "failure_class": cls,
                        "row": check,
                        "attempt_idx": attempt.get("attempt_idx", 0),
                    }
                    failures_by_class.setdefault(cls, []).append(entry)

    return failures_by_class


def _load_skill(skill_path: Path) -> str:
    if skill_path.exists():
        return skill_path.read_text(encoding="utf-8")
    return ""


def _existing_auto_skills(skills_dir: Path) -> set[str]:
    return {p.stem.replace("auto_skill_", "") for p in skills_dir.glob("auto_skill_*.md")}


def _build_synthesis_prompt(
    failure_class: str,
    examples: list[dict[str, Any]],
    existing_skill_text: str,
) -> str:
    top_examples = examples[:3]
    examples_text = ""
    for i, ex in enumerate(top_examples, 1):
        examples_text += f"\n### Example {i}\n"
        examples_text += f"Question: {ex['question']}\n"
        examples_text += f"Reason codes: {', '.join(ex['reason_codes'])}\n"
        row = ex.get("row", {})
        if row.get("source_file"):
            examples_text += f"Source file: {row['source_file']}\n"
        if row.get("row_label"):
            examples_text += f"Row label attempted: {row.get('row_label', '')}\n"
        if row.get("matched_snippet"):
            examples_text += f"Matched snippet attempted: {row.get('matched_snippet', '')[:200]}\n"

    prompt = f"""You are a skill-synthesis assistant for an evidence extraction system.

The system extracts evidence rows from financial documents and must include verbatim text from the corpus.
The agent is consistently failing with failure class: **{failure_class}**

## Failure Examples
{examples_text}

## Existing Skill Context
{existing_skill_text[:2000] if existing_skill_text else "(none)"}

## Task
Write a concise, targeted skill document (markdown) that teaches the agent how to avoid this specific failure class.

Requirements:
- Focus only on this failure pattern: {failure_class}
- Give 2-3 concrete rules with examples
- Use imperative instructions ("DO", "NEVER", "COPY")
- Keep under 400 words
- Start with: # Skill: Auto-Repair — {failure_class.replace("_", " ").title()}

Respond with ONLY the markdown skill document.
"""
    return prompt


def _call_llm(prompt: str, config: dict[str, Any]) -> str:
    """Call the LLM to synthesize a skill. Uses OpenAI-compatible API."""
    backend_cfg = config.get("backend", {}).get("openhands", {})
    model = backend_cfg.get("model", "cerebras/qwen-3-235b-a22b-instruct-2507")
    base_url = backend_cfg.get("base_url", "https://api.cerebras.ai/v1")
    api_key_env = backend_cfg.get("api_key_env", "CEREBRAS_API_KEY")
    api_key = os.environ.get(api_key_env, "")

    try:
        from openai import OpenAI
    except ImportError:
        print("[auto_refine] openai package not available; writing placeholder skill", file=sys.stderr)
        return f"# Skill: Auto-Repair — Placeholder\n\nCould not call LLM (openai package missing).\n"

    client = OpenAI(api_key=api_key or "placeholder", base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[auto_refine] LLM call failed: {e}", file=sys.stderr)
        return f"# Skill: Auto-Repair — Error\n\nLLM synthesis failed: {e}\n"


def _measure_grounding_pass_rate(run_dir: Path) -> float:
    """Compute grounding pass rate from an existing run directory."""
    examples_path = run_dir / "examples.jsonl"
    if not examples_path.exists():
        return 0.0
    total = 0
    passed = 0
    with examples_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            attempts = ex.get("attempts", [])
            # Check if any attempt had all grounding checks passing
            for attempt in attempts:
                checks = attempt.get("grounding_checks") or []
                if checks and all(c.get("matched") for c in checks):
                    passed += 1
                    break
    return passed / total if total > 0 else 0.0


def _run_smoke_with_config(config_path: Path) -> Path | None:
    """Run smoke test and return the run directory, or None on failure."""
    import re as _re
    cmd = [
        sys.executable,
        "scripts/run_mode.py",
        "--mode", "smoke",
        "--config", str(config_path),
        "--tag", "auto-refine-validation",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[auto_refine] Smoke run failed:\n{result.stderr}", file=sys.stderr)
        return None

    # Find the most recently created run dir
    runs_dir = ROOT / "runs"
    run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in run_dirs:
        if d.is_dir() and (d / "examples.jsonl").exists():
            return d
    return None


def _patch_config_with_skill(config_path: Path, skill_path: Path) -> Path:
    """Return path to a temp config file that includes the new skill."""
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    skill_paths = (
        cfg.get("backend", {}).get("openhands", {}).get("skill_paths") or []
    )
    rel_skill = str(skill_path.relative_to(ROOT))
    if rel_skill not in skill_paths:
        skill_paths = list(skill_paths) + [rel_skill]
    cfg.setdefault("backend", {}).setdefault("openhands", {})["skill_paths"] = skill_paths

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=ROOT / "config"
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Grounding-driven skill synthesis (AUTO_REFINE)")
    parser.add_argument("--run-dir", required=True, help="Path to completed run directory")
    parser.add_argument("--config", default="config/openhands_cerebras.yaml", help="Config to use for validation runs")
    parser.add_argument("--skills-dir", default="arena_official/skills", help="Skills library directory")
    parser.add_argument("--dry-run", action="store_true", help="Synthesize skills but skip validation smoke run")
    parser.add_argument("--min-failures", type=int, default=2, help="Minimum failures to trigger skill synthesis for a class")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"[auto_refine] ERROR: run_dir does not exist: {run_dir}", file=sys.stderr)
        return 1

    config_path = (ROOT / args.config).resolve()
    if not config_path.exists():
        print(f"[auto_refine] ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    skills_dir = (ROOT / args.skills_dir).resolve()
    skills_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open() as f:
        config = yaml.safe_load(f)

    print(f"[auto_refine] Scanning run: {run_dir}")
    failures_by_class = _scan_run_dir(run_dir)

    if not failures_by_class:
        print("[auto_refine] No grounding failures found. Nothing to do.")
        return 0

    print(f"[auto_refine] Failure classes found:")
    for cls, examples in sorted(failures_by_class.items(), key=lambda x: -len(x[1])):
        print(f"  {cls}: {len(examples)} failing rows")

    # Baseline grounding pass rate
    baseline_rate = _measure_grounding_pass_rate(run_dir)
    print(f"[auto_refine] Baseline grounding pass rate: {baseline_rate:.1%}")

    existing_classes = _existing_auto_skills(skills_dir)
    print(f"[auto_refine] Existing auto-skills: {existing_classes or '(none)'}")

    synthesized: list[Path] = []

    for failure_class, examples in sorted(failures_by_class.items(), key=lambda x: -len(x[1])):
        if len(examples) < args.min_failures:
            print(f"[auto_refine] Skipping {failure_class} (only {len(examples)} failures < min={args.min_failures})")
            continue

        if failure_class in existing_classes:
            print(f"[auto_refine] Skipping {failure_class} — skill already exists")
            continue

        print(f"\n[auto_refine] Synthesizing skill for: {failure_class} ({len(examples)} failures)")

        # Find related existing skill text for context
        existing_skill_text = ""
        for skill_path in skills_dir.glob("*.md"):
            if any(kw in skill_path.name for kw in ["evidence", "grounding", "verification"]):
                existing_skill_text += _load_skill(skill_path)[:1000] + "\n"

        prompt = _build_synthesis_prompt(failure_class, examples, existing_skill_text)
        skill_text = _call_llm(prompt, config)

        skill_file = skills_dir / f"auto_skill_{failure_class}.md"
        skill_file.write_text(skill_text, encoding="utf-8")
        print(f"[auto_refine] Wrote candidate skill: {skill_file}")
        synthesized.append(skill_file)

    if not synthesized:
        print("[auto_refine] No new skills synthesized.")
        return 0

    if args.dry_run:
        print(f"[auto_refine] Dry-run: skipping validation. Synthesized {len(synthesized)} skill(s).")
        return 0

    # Validate each synthesized skill
    kept = []
    discarded = []
    for skill_file in synthesized:
        print(f"\n[auto_refine] Validating skill: {skill_file.name}")
        patched_config = _patch_config_with_skill(config_path, skill_file)
        try:
            new_run_dir = _run_smoke_with_config(patched_config)
        finally:
            patched_config.unlink(missing_ok=True)

        if new_run_dir is None:
            print(f"[auto_refine] Validation run failed for {skill_file.name} — discarding")
            skill_file.unlink(missing_ok=True)
            discarded.append(skill_file.name)
            continue

        new_rate = _measure_grounding_pass_rate(new_run_dir)
        print(f"[auto_refine] Grounding pass rate with skill: {new_rate:.1%} (baseline: {baseline_rate:.1%})")

        if new_rate > baseline_rate:
            print(f"[auto_refine] KEEP {skill_file.name} (+{new_rate - baseline_rate:.1%})")
            kept.append(skill_file.name)
        else:
            print(f"[auto_refine] DISCARD {skill_file.name} (no improvement)")
            skill_file.unlink(missing_ok=True)
            discarded.append(skill_file.name)

    print(f"\n[auto_refine] Summary:")
    print(f"  Kept:     {kept or '(none)'}")
    print(f"  Discarded: {discarded or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
