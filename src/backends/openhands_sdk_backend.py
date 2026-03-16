from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from src.env import load_env_file
from src.schemas import grounding_judge_schema, parse_schema, solve_schema
from src.stages.parse import parsed_question_from_dict
from src.types import ParsedQuestion


def _read_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        return "You are an OfficeQA agent. Return only the final numeric answer."
    return prompt_path.read_text(encoding="utf-8")


def _read_skill_texts(
    *,
    project_root: Path,
    skill_paths: list[str],
    notes: list[str],
) -> tuple[list[str], list[str]]:
    resolved_paths: list[str] = []
    blocks: list[str] = []
    for raw in skill_paths:
        p = Path(raw)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        if not p.exists():
            notes.append(f"missing_skill_path={p}")
            continue
        try:
            text = p.read_text(encoding="utf-8").strip()
        except Exception as exc:  # pragma: no cover
            notes.append(f"skill_read_error={p}:{exc}")
            continue
        if not text:
            notes.append(f"empty_skill_file={p}")
            continue
        resolved_paths.append(str(p))
        blocks.append(f"# Skill: {p.name}\n{text}")
    return resolved_paths, blocks


def _compose_system_prompt(base_prompt: str, skill_blocks: list[str]) -> str:
    if not skill_blocks:
        return base_prompt
    return f"{base_prompt.rstrip()}\n\n## Additional Skills\n\n" + "\n\n".join(skill_blocks)


def _extract_text(response: Any) -> str:
    message = getattr(response, "message", None)
    if message is None:
        return str(response)
    content = getattr(message, "content", None) or []
    chunks: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            chunks.append(text)
    if chunks:
        return "\n".join(chunks).strip()
    return str(message)


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _extract_final_answer_from_text(text: str) -> str | None:
    m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?%?", text)
    if not m:
        return None
    return m.group(0)


def _build_llm_and_common_fields(config: dict[str, Any], project_root: Path):
    notes: list[str] = []
    backend_cfg = (config.get("backend", {}) or {}).get("openhands", {}) or {}
    model = str(backend_cfg.get("model", "openrouter/meta-llama/llama-3.3-70b-instruct:free"))
    api_key_env = str(backend_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
    base_url = backend_cfg.get("base_url", "https://openrouter.ai/api/v1")
    if "openrouter.ai" in str(base_url) and "/" in model and not model.startswith("openrouter/"):
        model = f"openrouter/{model}"
    max_output_tokens = int(backend_cfg.get("max_output_tokens", 256))
    temperature = float(backend_cfg.get("temperature", 0.0))
    timeout_seconds = int(backend_cfg.get("timeout_seconds", 120))
    prompt_path = project_root / str(backend_cfg.get("system_prompt_path", "prompts/officeqa_prompt.j2"))
    trace_mode = bool(backend_cfg.get("trace_mode", True))

    load_env_file(project_root / ".env")
    api_key = os.getenv(api_key_env)
    if not api_key:
        redacted = api_key_env
        if api_key_env.startswith("sk-") or len(api_key_env) > 40:
            redacted = "<redacted_or_misconfigured>"
        notes.append(f"openhands_missing_api_key_env={redacted}")
        return None, None, None, None, None, None, None, None, None, notes

    try:
        from openhands.sdk import LLM, Message, TextContent
    except Exception as exc:  # pragma: no cover
        notes.append(f"openhands_import_error={exc}")
        return None, None, None, None, None, None, None, None, None, notes

    llm = LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        timeout=timeout_seconds,
    )
    skill_paths = backend_cfg.get("skill_paths", []) or []
    if not isinstance(skill_paths, list):
        skill_paths = []

    return (
        llm,
        Message,
        TextContent,
        model,
        prompt_path,
        [str(x) for x in skill_paths],
        trace_mode,
        max_output_tokens,
        temperature,
        notes,
    )


def _backend_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return (config.get("backend", {}) or {}).get("openhands", {}) or {}


def _structured_output_enabled(config: dict[str, Any]) -> bool:
    return bool(_backend_cfg(config).get("structured_output", True))


def _parse_schema_name(config: dict[str, Any]) -> str:
    return str(_backend_cfg(config).get("parse_schema_name", "officeqa_parse_v1"))


def _solve_schema_name(config: dict[str, Any]) -> str:
    return str(_backend_cfg(config).get("solve_schema_name", "officeqa_solve_v1"))


def _solve_max_output_tokens(config: dict[str, Any], default_tokens: int) -> int:
    return int(_backend_cfg(config).get("solve_max_output_tokens", default_tokens))


def _grounding_judge_max_output_tokens(config: dict[str, Any], default_tokens: int) -> int:
    return int(_backend_cfg(config).get("grounding_judge_max_output_tokens", min(4096, default_tokens)))


def _solve_prompt_chunk_limit(config: dict[str, Any]) -> int:
    return int(_backend_cfg(config).get("solve_prompt_max_chunks", 4))


def _solve_prompt_chars_per_chunk(config: dict[str, Any]) -> int:
    return int(_backend_cfg(config).get("solve_prompt_max_chars_per_chunk", 3500))


def _request_trace(
    *,
    model: str,
    trace_mode: bool,
    max_output_tokens: int,
    temperature: float,
    prompt_path: Path,
    prompt_kind: str,
    skill_paths: list[str],
    system_prompt: str,
    user_prompt: str,
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "prompt_kind": prompt_kind,
        "model": model,
        "trace_mode": trace_mode,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "system_prompt_path": str(prompt_path),
        "response_format": response_format or {},
        "skill_paths": skill_paths,
        "prompt_payload": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        },
    }


def run_openhands_parse(
    *,
    raw_question: str,
    metadata: dict[str, Any],
    corpus_chunks: list[dict[str, str]],
    config: dict[str, Any],
    project_root: Path,
    repair_brief: dict[str, Any] | None = None,
    attempt_index: int = 1,
) -> tuple[ParsedQuestion | None, dict[str, Any], list[str]]:
    (
        llm,
        Message,
        TextContent,
        model,
        prompt_path,
        skill_paths,
        trace_mode,
        max_output_tokens,
        temperature,
        notes,
    ) = _build_llm_and_common_fields(config, project_root)
    if llm is None:
        return None, {}, notes

    base_prompt = _read_prompt(prompt_path)
    resolved_skill_paths, skill_blocks = _read_skill_texts(
        project_root=project_root,
        skill_paths=skill_paths,
        notes=notes,
    )
    system_prompt = _compose_system_prompt(base_prompt, skill_blocks)
    preview_chunks = corpus_chunks[:2]
    corpus_preview = "\n\n".join(
        f"[{c.get('chunk_id')}] {c.get('text','')[:1200]}" for c in preview_chunks
    )
    repair_block = f"\nRepair brief (attempt {attempt_index}): {repair_brief}\n" if repair_brief else ""
    user_prompt = (
        "Parse this OfficeQA question and return ONLY schema-valid JSON with exact keys:\n"
        "metric, entity_category, time_period, operation, expected_answer_type.\n"
        "No extra keys. No markdown fences.\n\n"
        f"Question: {raw_question}\n"
        f"Metadata hints: {metadata}\n"
        f"Corpus snippets:\n{corpus_preview}\n"
        f"{repair_block}"
    )
    notes.append(f"parse_prompt_chunk_count={len(preview_chunks)}")
    notes.append(f"parse_prompt_char_count={len(corpus_preview)}")

    response_format = parse_schema(_parse_schema_name(config)) if _structured_output_enabled(config) else None
    req_trace = _request_trace(
        model=model,
        trace_mode=trace_mode,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        prompt_path=prompt_path,
        prompt_kind="parse_structured",
        skill_paths=resolved_skill_paths,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=response_format,
    )
    completion_kwargs: dict[str, Any] = {}
    if response_format is not None:
        completion_kwargs["response_format"] = response_format
    try:
        response = llm.completion(
            messages=[
                Message(role="system", content=[TextContent(text=system_prompt)]),
                Message(role="user", content=[TextContent(text=user_prompt)]),
            ],
            **completion_kwargs,
        )
    except Exception as exc:  # pragma: no cover
        notes.append(f"openhands_parse_error={exc}")
        return None, {"backend_request": req_trace}, notes

    output_text = _extract_text(response)
    parsed_json = _try_parse_json_object(output_text)
    trace = {
        "backend_request": req_trace,
        "backend_response_raw": output_text,
        "backend_response_structured": parsed_json or {},
    }
    if parsed_json is None:
        notes.append("parse_structured_invalid_json=true")
        notes.append("parse_schema_valid=false")
        return None, trace, notes

    parsed = parsed_question_from_dict(parsed_json)
    if parsed is None:
        notes.append("parse_structured_missing_fields=true")
        notes.append("parse_schema_valid=false")
        return None, trace, notes
    notes.append("parse_schema_valid=true")
    return parsed, trace, notes


def run_openhands_backend(
    *,
    raw_question: str,
    parsed_question: dict[str, Any],
    metadata: dict[str, Any],
    corpus_chunks: list[dict[str, str]],
    config: dict[str, Any],
    project_root: Path,
    repair_brief: dict[str, Any] | None = None,
    attempt_index: int = 1,
    expected_rows_hint: dict[str, int] | None = None,
) -> tuple[str | None, dict[str, Any], list[str]]:
    (
        llm,
        Message,
        TextContent,
        model,
        prompt_path,
        skill_paths,
        trace_mode,
        max_output_tokens,
        temperature,
        notes,
    ) = _build_llm_and_common_fields(config, project_root)
    if llm is None:
        return None, {}, notes

    solve_tokens = _solve_max_output_tokens(config, max_output_tokens)
    base_prompt = _read_prompt(prompt_path)
    resolved_skill_paths, skill_blocks = _read_skill_texts(
        project_root=project_root,
        skill_paths=skill_paths,
        notes=notes,
    )
    system_prompt = _compose_system_prompt(base_prompt, skill_blocks)
    prompt_chunk_limit = _solve_prompt_chunk_limit(config)
    prompt_chunk_chars = _solve_prompt_chars_per_chunk(config)
    # Guarantee at least 1 chunk per source file, then fill remainder by score
    seen_files: set[str] = set()
    guaranteed: list[dict] = []
    remainder: list[dict] = []
    for c in corpus_chunks:
        sf = c.get("source_file", "")
        if sf not in seen_files:
            seen_files.add(sf)
            guaranteed.append(c)
        else:
            remainder.append(c)
    selected = (guaranteed + remainder)[:prompt_chunk_limit]
    corpus_block = "\n\n".join(
        f"[{c.get('chunk_id')}] source_file={c.get('source_file')}\n{c.get('text','')[:prompt_chunk_chars]}"
        for c in selected
    )
    notes.append(f"solve_prompt_chunk_count={min(len(corpus_chunks), prompt_chunk_limit)}")
    notes.append(f"solve_prompt_char_count={len(corpus_block)}")
    min_rows = int((expected_rows_hint or {}).get("min_rows", 1))
    max_rows = int((expected_rows_hint or {}).get("max_rows", 4))
    notes.append(f"expected_evidence_rows_min={min_rows}")
    notes.append(f"expected_evidence_rows_max={max_rows}")
    repair_block = f"\nRepair brief (attempt {attempt_index}): {repair_brief}\n" if repair_brief else ""
    if trace_mode:
        user_prompt = (
            "Solve this OfficeQA-style question and return ONLY schema-valid JSON.\n"
            "Required keys only: final_answer, reasoning_summary, confidence, evidence_rows.\n"
            "evidence_rows is a list of objects with keys: "
            "source_file, table_or_section, row_label, column_label, raw_value, unit, matched_snippet.\n"
            f"Return ONLY the minimum evidence rows needed. Required evidence_rows count: {min_rows} to {max_rows}.\n"
            "Every evidence row must be literal from corpus chunks; never guess labels or values.\n"
            "matched_snippet must be a short exact excerpt copied from corpus chunks that contains the row evidence.\n"
            "If a row is uncertain, omit it rather than returning approximate text.\n"
            "No markdown fences. No extra keys.\n\n"
            f"Question: {raw_question}\n"
            f"Parsed fields: {parsed_question}\n"
            f"Metadata hints: {metadata}\n"
            f"Corpus chunks:\n{corpus_block}\n"
            f"{repair_block}"
        )
    else:
        user_prompt = (
            "Solve this OfficeQA-style question and return only the final numeric answer.\n\n"
            f"Question: {raw_question}\n"
            f"Parsed fields: {parsed_question}\n"
            f"Metadata hints: {metadata}\n"
            f"Corpus chunks:\n{corpus_block}\n"
        )

    response_format = solve_schema(_solve_schema_name(config)) if _structured_output_enabled(config) and trace_mode else None
    req_trace = _request_trace(
        model=model,
        trace_mode=trace_mode,
        max_output_tokens=solve_tokens,
        temperature=temperature,
        prompt_path=prompt_path,
        prompt_kind="solve_question",
        skill_paths=resolved_skill_paths,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=response_format,
    )
    completion_kwargs: dict[str, Any] = {"max_completion_tokens": solve_tokens}
    if response_format is not None:
        completion_kwargs["response_format"] = response_format
    try:
        response = llm.completion(
            messages=[
                Message(role="system", content=[TextContent(text=system_prompt)]),
                Message(role="user", content=[TextContent(text=user_prompt)]),
            ],
            **completion_kwargs,
        )
    except Exception as exc:  # pragma: no cover
        notes.append(f"openhands_completion_error={exc}")
        return None, {"backend_request": req_trace}, notes

    output_text = _extract_text(response)
    parsed_json = _try_parse_json_object(output_text)
    final_answer: str | None = None
    model_trace: dict[str, Any] = {
        "backend_request": req_trace,
        "backend_response_raw": output_text,
        "backend_response_structured": {},
    }
    if parsed_json is not None:
        model_trace["backend_response_structured"] = parsed_json
        final_value = parsed_json.get("final_answer")
        final_answer = None if final_value is None else str(final_value)
        evidence_rows = parsed_json.get("evidence_rows")
        if not isinstance(evidence_rows, list):
            notes.append("solver_missing_evidence_rows=true")
            notes.append("solve_schema_valid=false")
        else:
            notes.append("solve_schema_valid=true")
    else:
        if _structured_output_enabled(config) and trace_mode:
            notes.append("solver_invalid_json=true")
            notes.append("solve_schema_valid=false")
        else:
            final_answer = _extract_final_answer_from_text(output_text)

    notes.append(f"openhands_model={model}")
    return final_answer, model_trace, notes


def run_openhands_grounding_judge(
    *,
    raw_question: str,
    parsed_question: dict[str, Any],
    metadata: dict[str, Any],
    evidence_rows: list[dict[str, Any]],
    corpus_chunks: list[dict[str, str]],
    config: dict[str, Any],
    project_root: Path,
    attempt_index: int = 1,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[str]]:
    (
        llm,
        Message,
        TextContent,
        model,
        prompt_path,
        skill_paths,
        trace_mode,
        max_output_tokens,
        temperature,
        notes,
    ) = _build_llm_and_common_fields(config, project_root)
    if llm is None:
        return None, {}, notes

    judge_tokens = _grounding_judge_max_output_tokens(config, max_output_tokens)
    base_prompt = _read_prompt(prompt_path)
    resolved_skill_paths, skill_blocks = _read_skill_texts(
        project_root=project_root,
        skill_paths=skill_paths,
        notes=notes,
    )
    system_prompt = _compose_system_prompt(base_prompt, skill_blocks)

    prompt_chunk_limit = _solve_prompt_chunk_limit(config)
    prompt_chunk_chars = _solve_prompt_chars_per_chunk(config)
    corpus_block = "\n\n".join(
        f"[{c.get('chunk_id')}] source_file={c.get('source_file')}\n{c.get('text','')[:prompt_chunk_chars]}"
        for c in corpus_chunks[:prompt_chunk_limit]
    )
    notes.append(f"grounding_judge_prompt_chunk_count={min(len(corpus_chunks), prompt_chunk_limit)}")
    notes.append(f"grounding_judge_prompt_char_count={len(corpus_block)}")

    user_prompt = (
        "You are a grounding judge. Validate whether each evidence row is truly grounded in the corpus chunks.\n"
        "Return ONLY schema-valid JSON.\n"
        "For each row_index, set grounded=true only if source_file/row_label/column_label/raw_value are supported by the corpus text.\n"
        "Use concise reason_codes like source_file_mismatch,row_label_not_found,column_label_not_found,raw_value_not_found,snippet_not_supported.\n"
        f"Question: {raw_question}\n"
        f"Parsed fields: {parsed_question}\n"
        f"Metadata hints: {metadata}\n"
        f"Attempt: {attempt_index}\n"
        f"Evidence rows:\n{json.dumps(evidence_rows, ensure_ascii=True)}\n"
        f"Corpus chunks:\n{corpus_block}\n"
    )

    response_format = grounding_judge_schema() if _structured_output_enabled(config) and trace_mode else None
    req_trace = _request_trace(
        model=model,
        trace_mode=trace_mode,
        max_output_tokens=judge_tokens,
        temperature=temperature,
        prompt_path=prompt_path,
        prompt_kind="grounding_judge",
        skill_paths=resolved_skill_paths,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=response_format,
    )

    completion_kwargs: dict[str, Any] = {"max_completion_tokens": judge_tokens}
    if response_format is not None:
        completion_kwargs["response_format"] = response_format
    try:
        response = llm.completion(
            messages=[
                Message(role="system", content=[TextContent(text=system_prompt)]),
                Message(role="user", content=[TextContent(text=user_prompt)]),
            ],
            **completion_kwargs,
        )
    except Exception as exc:  # pragma: no cover
        notes.append(f"openhands_grounding_judge_error={exc}")
        return None, {"backend_request": req_trace}, notes

    output_text = _extract_text(response)
    parsed_json = _try_parse_json_object(output_text)
    trace = {
        "backend_request": req_trace,
        "backend_response_raw": output_text,
        "backend_response_structured": parsed_json or {},
    }
    if not isinstance(parsed_json, dict):
        notes.append("grounding_judge_invalid_json=true")
        return None, trace, notes
    notes.append("grounding_judge_valid=true")
    return parsed_json, trace, notes
