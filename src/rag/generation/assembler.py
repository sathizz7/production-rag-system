from __future__ import annotations

from rag.models import AssembledContext, Chunk
from rag.util.tokens import count_tokens


class TokenBudgetAssembler:
    """Dedupes, numbers, and packs ranked chunks under a token budget.

    Input order is assumed to be relevance order; overflow is dropped from the tail.
    chunks[i] in the result corresponds to citation marker ``[i+1]``.
    """

    def assemble(
        self, query: str, chunks: list[Chunk], token_budget: int
    ) -> AssembledContext:
        selected: list[Chunk] = []
        seen: set[str] = set()
        used_tokens = 0

        for chunk in chunks:
            if chunk.chunk_id in seen:
                continue
            block = self._format_block(len(selected) + 1, chunk)
            block_tokens = count_tokens(block)
            if selected and used_tokens + block_tokens > token_budget:
                break
            seen.add(chunk.chunk_id)
            selected.append(chunk)
            used_tokens += block_tokens

        text = "\n\n".join(
            self._format_block(i + 1, c) for i, c in enumerate(selected)
        )
        return AssembledContext(text=text, chunks=selected, token_count=count_tokens(text))

    @staticmethod
    def _format_block(marker: int, chunk: Chunk) -> str:
        return f"[{marker}] (source: {chunk.source_uri}, p.{chunk.page})\n{chunk.text}"
