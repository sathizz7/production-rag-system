from __future__ import annotations

import re

from rag.models import Document, Page
from rag.util.hashing import content_hash

_HORIZONTAL_WS = re.compile(r"[ \t\f\v]+")
_MANY_NEWLINES = re.compile(r"\n{2,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class BasicCleaner:
    """Conservative, offset-honest normalization (P0a). Layout-aware cleaning is P0c."""

    def clean(self, doc: Document) -> Document:
        cleaned_pages = [
            Page(number=p.number, text=self._clean_text(p.text)) for p in doc.pages
        ]
        new_doc = doc.model_copy(update={"pages": cleaned_pages})
        new_doc.content_hash = content_hash(new_doc.text)
        return new_doc

    @staticmethod
    def _clean_text(text: str) -> str:
        text = _CONTROL.sub("", text)
        # collapse horizontal whitespace, trim each line, collapse blank-line runs
        text = _HORIZONTAL_WS.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = _MANY_NEWLINES.sub("\n", text)
        return text.strip()
