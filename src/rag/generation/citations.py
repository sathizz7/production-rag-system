from __future__ import annotations

import re

import structlog

from rag.models import AssembledContext, Citation, SourceKind

log = structlog.get_logger()

_MARKER = re.compile(r"\[(\d+)\]")


def validate_and_build_citations(
    answer_text: str, context: AssembledContext
) -> tuple[str, list[Citation]]:
    """Strip markers that do not map to an assembled chunk; build Citations for the rest.

    Returns (cleaned_text, citations). A chunk is cited at most once. Markers are
    1-based indices into ``context.chunks`` (so ``[1]`` -> ``context.chunks[0]``).
    """
    n = len(context.chunks)
    valid_indices = {i for i in range(1, n + 1)}
    seen: set[int] = set()
    citations: list[Citation] = []

    for match in _MARKER.finditer(answer_text):
        idx = int(match.group(1))
        if idx in valid_indices and idx not in seen:
            seen.add(idx)
            chunk = context.chunks[idx - 1]
            citations.append(
                Citation(
                    marker=f"[{idx}]",
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    source_uri=chunk.source_uri,
                    source_kind=SourceKind.corpus,
                    page=chunk.page,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                )
            )

    def _strip_invalid(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if idx in valid_indices:
            return match.group(0)
        log.warning("stripped_invalid_citation", marker=match.group(0))
        return ""

    cleaned = _MARKER.sub(_strip_invalid, answer_text)
    citations.sort(key=lambda c: int(c.marker.strip("[]")))
    return cleaned, citations
