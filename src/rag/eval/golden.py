from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

Stratum = str  # easy | ambiguous | table | adversarial | out_of_corpus | metadata


class GoldenItem(BaseModel):
    id: str
    question: str
    reference_answer: str
    relevant_doc_ids: list[str] = Field(default_factory=list)
    stratum: Stratum = "easy"


def load_golden_set(path: Path) -> list[GoldenItem]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [GoldenItem(**item) for item in data["items"]]
