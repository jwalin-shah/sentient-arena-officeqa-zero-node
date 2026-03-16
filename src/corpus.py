from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


@dataclass
class CorpusContext:
    requested_files: list[str]
    loaded_files: list[str]
    missing_files: list[str]
    chunks: list[dict[str, str]]


def parse_source_files(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value)
    parts = re.split(r"[\n,;]+", text)
    files = [p.strip() for p in parts if p.strip()]
    return files


_SECTION_HEADER_RE = re.compile(r"(?:^[A-Z][A-Z\s\-]{4,}$|^-{3,}$)", re.MULTILINE)


def _split_by_section(text: str) -> list[str]:
    """Split on ALL-CAPS headers and --- separators. Returns sections."""
    positions = [m.start() for m in _SECTION_HEADER_RE.finditer(text)]
    if len(positions) < 2:
        return [text]
    sections = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        sections.append(text[pos:end].strip())
    return [s for s in sections if s]


def _chunk_text(text: str, *, max_chars: int = 5000, overlap: int = 200) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _query_terms(query: str) -> set[str]:
    terms = re.findall(r"[a-zA-Z]{4,}", query.lower())
    stop = {
        "what",
        "were",
        "from",
        "with",
        "that",
        "this",
        "using",
        "only",
        "reported",
        "values",
        "total",
        "calendar",
        "year",
    }
    return {t for t in terms if t not in stop}


def _score_chunk(chunk: str, query_terms: set[str]) -> float:
    text = chunk.lower()
    if not query_terms:
        return 0.0
    overlap = sum(1 for t in query_terms if t in text)
    penalty = 2 if "table of contents" in text else 0
    return float(overlap - penalty)


def build_retrieval_query(
    *,
    raw_question: str,
    parsed_question: dict[str, Any] | None = None,
) -> str:
    if not parsed_question:
        return raw_question
    fields = [
        str(parsed_question.get("metric", "")).strip(),
        str(parsed_question.get("entity_category", "")).strip(),
        str(parsed_question.get("time_period", "")).strip(),
        str(parsed_question.get("operation", "")).strip(),
        str(parsed_question.get("expected_answer_type", "")).strip(),
    ]
    parsed_text = " ".join(x for x in fields if x and x.lower() != "unknown")
    return f"{raw_question} {parsed_text}".strip()


def load_transformed_txt_context(
    *,
    source_files: list[str],
    corpus_root: Path,
    max_chunks_per_file: int = 2,
    max_chars: int = 5000,
    query: str = "",
    max_files_from_year_fallback: int = 2,
    max_total_chunks: int = 0,
) -> CorpusContext:
    loaded_files: list[str] = []
    missing_files: list[str] = []
    chunks: list[dict[str, str]] = []

    # Skill: Automated Year-Based Retrieval
    # If the metadata is missing source_files, we search for them based on the query year.
    if not source_files and query and int(max_files_from_year_fallback) > 0:
        year_match = re.search(r"(19|20)\d{2}", query)
        if year_match:
            year = year_match.group(0)
            # Find all .txt files in the corpus_root that contain the year
            potential_files = sorted(f.name for f in corpus_root.glob(f"*{year}*.txt"))
            source_files = potential_files[: int(max_files_from_year_fallback)]

    terms = _query_terms(query)
    scored_chunks_all: list[tuple[float, dict[str, str]]] = []
    for filename in source_files:
        path = corpus_root / filename
        if not path.exists():
            missing_files.append(filename)
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        sections = _split_by_section(text)
        if len(sections) > 1:
            all_chunks = sections
        else:
            all_chunks = _chunk_text(text, max_chars=max_chars)
        scored = sorted(
            enumerate(all_chunks, start=1),
            key=lambda x: _score_chunk(x[1], terms),
            reverse=True,
        )
        selected = scored[:max_chunks_per_file] if scored else []
        split_chunks = [(idx, chunk) for idx, chunk in selected]
        if not split_chunks:
            split_chunks = [(1, all_chunks[0])] if all_chunks else []
        for i, chunk in split_chunks:
            score = _score_chunk(chunk, terms)
            scored_chunks_all.append(
                (
                    score,
                    {
                        "source_file": filename,
                        "chunk_id": f"{filename}#chunk{i}",
                        "text": chunk,
                    },
                )
            )
        loaded_files.append(filename)

    scored_chunks_all.sort(key=lambda x: x[0], reverse=True)
    if int(max_total_chunks) > 0:
        scored_chunks_all = scored_chunks_all[: int(max_total_chunks)]
    chunks = [c for _, c in scored_chunks_all]

    return CorpusContext(
        requested_files=source_files,
        loaded_files=loaded_files,
        missing_files=missing_files,
        chunks=chunks,
    )
