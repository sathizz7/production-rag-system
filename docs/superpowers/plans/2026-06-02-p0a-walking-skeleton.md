# P0a — Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the end-to-end RAG walking skeleton — ingest a text PDF, embed and store it in Postgres+pgvector under Alembic-managed schema, retrieve by dense vector search, assemble context with span-level citations, and return a non-streaming cited JSON answer from a FastAPI `/query` endpoint, all behind `docker compose up`.

**Architecture:** Two paths over one Postgres store. The **ingestion path** (offline CLI) runs PDF → parse → clean → chunk → embed → upsert. The **query path** (FastAPI) runs embed-query → dense KNN → assemble → generate → validate-citations → JSON. Every subsystem sits behind a Protocol from the design spec §6, so P0b can add hybrid/rerank/streaming without changing signatures. Everything is **synchronous** in P0a (the API route wraps sync work in a threadpool); P0b introduces async streaming. All model calls route through **LiteLLM** with **Gemini** defaults.

**Tech Stack:** Python 3.12 + uv · FastAPI · Pydantic v2 + pydantic-settings · LiteLLM (Gemini) · PyMuPDF · tiktoken · SQLAlchemy 2 Core + psycopg3 + pgvector · Alembic · structlog · pytest + testcontainers · ruff + mypy(strict) · Docker Compose.

**Source spec:** [docs/superpowers/specs/2026-06-02-production-rag-system-design.md](../specs/2026-06-02-production-rag-system-design.md) — this plan implements **P0a only** (spec §7). P0b/P0c and Phases 1–4 get their own plans.

---

## Conventions (read once, applies to every task)

**The TDD loop** — every task follows: write a failing test → run it, see it fail → write minimal code → run it, see it pass → commit. Do not write implementation before its test.

**Running things:**
- Unit tests (fast, no Docker, no network): `uv run pytest -m "not integration and not live"` — this is the default `uv run pytest`.
- Integration tests (boot real Postgres via Docker): `uv run pytest -m integration`.
- Live tests (hit real Gemini, need `GEMINI_API_KEY`): `uv run pytest -m live` — skipped in CI and by default.
- Lint/format: `uv run ruff check .` and `uv run ruff format .`
- Types: `uv run mypy src`

**Markers** are registered in `pyproject.toml` (Task 1). Integration tests need Docker Desktop running.

**Commits:** Conventional Commits (`feat:`, `test:`, `fix:`, `chore:`, `build:`, `docs:`). Commit at the end of every task; the repo must stay green (`uv run pytest` passing) at each commit.

**Windows note:** the author's machine is Windows/PowerShell. `make` may be absent — each `make` target below is also given as the raw command. Docker Compose is OS-agnostic.

**Citation model (the core P0a invariant):** a chunk stores `char_start`/`char_end` (offsets into its parent `Document.text`) and `page`. The invariant `chunk.text == document.text[char_start:char_end]` holds by construction and is asserted in tests. The generator cites context chunks by bracket marker `[n]`; **a marker that does not map to an assembled chunk is stripped from the answer** (spec §6 "cite only assembled chunks"). Citations copy their span from the chunk — the model never invents offsets.

---

## File Structure

Files created in P0a, each with one responsibility:

```
production-rag/
├── pyproject.toml                      # uv project, deps, ruff/mypy/pytest config   [Task 1]
├── .env.example                        # GEMINI_API_KEY, DATABASE_URL, model config  [Task 1,19]
├── Makefile                            # up/ingest/query/test/migrate one-liners      [Task 19]
├── docker-compose.yml                  # api + postgres(pgvector)                      [Task 19]
├── Dockerfile                          # multi-stage, non-root, healthcheck            [Task 19]
├── alembic.ini                         # Alembic config                                [Task 10]
├── alembic/
│   ├── env.py                          # Alembic runtime (reads DATABASE_URL)          [Task 10]
│   └── versions/0001_initial_schema.py # extension + documents + chunks + HNSW index   [Task 10]
├── src/rag/
│   ├── __init__.py
│   ├── config.py                       # pydantic-settings Settings                    [Task 2]
│   ├── models.py                       # domain models + enums                         [Task 3]
│   ├── protocols.py                    # Protocol interfaces (spec §6)                 [Task 3]
│   ├── util/
│   │   ├── __init__.py
│   │   ├── hashing.py                  # content_hash(text)                            [Task 4]
│   │   └── tokens.py                   # count_tokens(text), get_encoder()             [Task 4]
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── embeddings.py               # LiteLLMEmbeddingProvider                      [Task 9]
│   │   └── llm.py                      # LiteLLMProvider (sync complete())             [Task 16]
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── sources/
│   │   │   ├── __init__.py
│   │   │   └── pdf.py                  # PdfSourceAdapter (PyMuPDF)                     [Task 5]
│   │   ├── parse.py                    # PdfParser → Document(pages, offsets)          [Task 6]
│   │   ├── clean.py                    # BasicCleaner (offset-consistent)              [Task 7]
│   │   ├── chunking/
│   │   │   ├── __init__.py
│   │   │   └── fixed.py                # FixedTokenChunker                             [Task 8]
│   │   ├── repository.py              # PgChunkRepository (upsert/soft_delete/wm)      [Task 11]
│   │   ├── pipeline.py                # IngestionPipeline (wires the write path)       [Task 12]
│   │   └── cli.py                     # `python -m rag.ingestion.cli <path>`           [Task 12]
│   ├── db.py                          # engine, Core tables, run_migrations()          [Task 10]
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── dense.py                   # DenseRetriever (pgvector KNN + filter)         [Task 13]
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── assembler.py               # TokenBudgetAssembler                          [Task 14]
│   │   ├── citations.py               # validate_and_build_citations()                [Task 15]
│   │   ├── answerer.py                # StraightLineAnswerer                          [Task 17]
│   │   └── prompts/
│   │       └── answer_v1.md           # versioned generation prompt                    [Task 16]
│   └── api/
│       ├── __init__.py
│       ├── app.py                     # create_app() factory + lifespan DI            [Task 18]
│       ├── routes.py                  # /healthz, /query (JSON)                        [Task 18]
│       └── schemas.py                 # QueryRequest/QueryResponse/CitationOut         [Task 18]
└── tests/
    ├── conftest.py                    # fixtures: sample_pdf, migrated db engine       [Task 5,11]
    ├── unit/
    │   ├── fakes.py                   # FakeEmbedder, FakeLLMProvider, FakeRetriever   [Task 9,16,17]
    │   └── test_*.py                  # per-component unit tests
    └── integration/
        └── test_*.py                  # testcontainers: repo, retrieval, pipeline, api
```

---

## Task 1: Project scaffold & toolchain

**Files:**
- Create: `pyproject.toml`, `src/rag/__init__.py`, `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`
- Test: `tests/unit/test_smoke.py`

- [ ] **Step 1: Initialize the uv project and add dependencies**

Run (from repo root):
```bash
uv init --package --name rag --python 3.12 --no-workspace
uv add fastapi "uvicorn[standard]" "pydantic>=2" pydantic-settings litellm pymupdf tiktoken "sqlalchemy>=2" "psycopg[binary]" pgvector alembic structlog
uv add --dev pytest pytest-asyncio httpx "testcontainers[postgres]" respx ruff mypy
```
Expected: `pyproject.toml` and `uv.lock` are created and `src/rag/` is scaffolded. If `uv init` added a sample `[project.scripts]` entry (e.g. `rag = "rag:main"`) or a `main()` in `__init__.py`, remove them — Step 2 overwrites `__init__.py` and Task 19 adds the real `rag-ingest` script.

- [ ] **Step 2: Ensure the package and test directories exist**

Create these empty files if `uv init` did not:
```
src/rag/__init__.py
tests/__init__.py
tests/unit/__init__.py
tests/integration/__init__.py
```

`src/rag/__init__.py`:
```python
"""Production-grade RAG system — walking skeleton (P0a)."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Append tool config to `pyproject.toml`**

Append (do not remove uv's generated `[project]`/`[build-system]`):
```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.12"
strict = true
mypy_path = "src"
packages = ["rag"]
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["pymupdf", "fitz", "pgvector.*", "testcontainers.*", "litellm", "litellm.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
addopts = "-m 'not integration and not live' -v"
markers = [
    "integration: needs a real Postgres via Docker (testcontainers)",
    "live: needs real provider API keys (GEMINI_API_KEY)",
]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 4: Write the smoke test**

`tests/unit/test_smoke.py`:
```python
def test_package_imports_and_has_version() -> None:
    import rag

    assert rag.__version__ == "0.1.0"
```

- [ ] **Step 5: Run the smoke test (expect PASS)**

Run: `uv run pytest tests/unit/test_smoke.py -v`
Expected: 1 passed. (If `pythonpath`/`packages` are wrong the import fails — fix before moving on.)

- [ ] **Step 6: Verify lint and types are clean**

Run: `uv run ruff check .` → Expected: "All checks passed!"
Run: `uv run mypy src` → Expected: "Success: no issues found".

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src tests
git commit -m "build: scaffold uv project with ruff, mypy strict, pytest markers"
```

---

## Task 2: Config (pydantic-settings)

**Files:**
- Create: `src/rag/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_config.py`:
```python
from rag.config import Settings


def test_settings_defaults_use_gemini() -> None:
    s = Settings(gemini_api_key="x", database_url="postgresql+psycopg://u:p@h:5432/db")
    assert s.generation_model == "gemini/gemini-2.5-pro"
    assert s.grader_model == "gemini/gemini-2.5-flash"
    assert s.embedding_model == "gemini/text-embedding-004"
    assert s.embedding_dim == 768


def test_settings_reads_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "envkey")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:5432/db")
    monkeypatch.setenv("GENERATION_MODEL", "gemini/gemini-2.5-flash")
    s = Settings()
    assert s.gemini_api_key == "envkey"
    assert s.generation_model == "gemini/gemini-2.5-flash"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.config'`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/config.py`:
```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Secrets / infra
    gemini_api_key: str = ""
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"

    # Models (config-driven; swap provider by changing the string)
    generation_model: str = "gemini/gemini-2.5-pro"
    grader_model: str = "gemini/gemini-2.5-flash"
    embedding_model: str = "gemini/text-embedding-004"
    embedding_dim: int = 768

    # Chunking
    chunk_tokens: int = 512
    chunk_overlap: int = 64

    # Retrieval / assembly
    retrieval_k: int = 10
    context_token_budget: int = 6000

    # Generation
    generation_temperature: float = 0.0
    generation_max_tokens: int = 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag/config.py tests/unit/test_config.py
git commit -m "feat(config): typed Settings with Gemini defaults via pydantic-settings"
```

---

## Task 3: Domain models & Protocols

**Files:**
- Create: `src/rag/models.py`, `src/rag/protocols.py`
- Test: `tests/unit/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_models.py`:
```python
from rag.models import Document, Page, RetrievalScope, SourceKind


def test_document_text_joins_pages_with_separator() -> None:
    doc = _doc(["alpha", "bravo"])
    assert doc.text == "alpha\n\nbravo"


def test_page_at_maps_offsets_to_page_numbers() -> None:
    doc = _doc(["alpha", "bravo"])  # 'alpha'=0..5, sep=5..7, 'bravo'=7..12
    assert doc.page_at(0) == 1
    assert doc.page_at(4) == 1
    assert doc.page_at(7) == 2
    assert doc.page_at(11) == 2


def test_enums_are_string_valued() -> None:
    assert RetrievalScope.corpus_only.value == "corpus_only"
    assert SourceKind.corpus.value == "corpus"


def _doc(page_texts: list[str]) -> Document:
    return Document(
        doc_id="d1",
        source_id="s1",
        source_type="pdf",
        uri="file:///x.pdf",
        pages=[Page(number=i + 1, text=t) for i, t in enumerate(page_texts)],
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.models'`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/models.py`:
```python
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

PAGE_SEPARATOR = "\n\n"


class RetrievalScope(str, Enum):
    corpus_only = "corpus_only"
    web_allowed = "web_allowed"
    web_required = "web_required"


class Provenance(str, Enum):
    dense = "dense"
    lexical = "lexical"
    fused = "fused"
    rerank = "rerank"


class SourceKind(str, Enum):
    corpus = "corpus"
    web = "web"


class RawDocument(BaseModel):
    source_id: str
    source_type: str
    uri: str
    text: str | None = None
    raw_bytes: bytes | None = None
    fetched_at: datetime | None = None
    source_etag: str | None = None
    source_last_modified: str | None = None
    license: str | None = None
    source_meta: dict = Field(default_factory=dict)


class Page(BaseModel):
    number: int  # 1-based
    text: str


class Document(BaseModel):
    doc_id: str
    source_id: str
    source_type: str
    uri: str
    document_version: int = 1
    pages: list[Page] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    content_hash: str = ""
    license: str | None = None
    deleted_at: datetime | None = None

    @property
    def text(self) -> str:
        return PAGE_SEPARATOR.join(p.text for p in self.pages)

    def page_spans(self) -> list[tuple[int, int, int]]:
        """Returns (page_number, char_start, char_end) into self.text for each page."""
        spans: list[tuple[int, int, int]] = []
        pos = 0
        last = len(self.pages) - 1
        for i, p in enumerate(self.pages):
            start = pos
            end = pos + len(p.text)
            spans.append((p.number, start, end))
            pos = end + (len(PAGE_SEPARATOR) if i < last else 0)
        return spans

    def page_at(self, offset: int) -> int:
        spans = self.page_spans()
        for number, start, end in spans:
            if start <= offset < end:
                return number
        return spans[-1][0] if spans else 0


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_uri: str
    text: str
    ordinal: int
    page: int
    char_start: int
    char_end: int
    token_count: int
    metadata: dict = Field(default_factory=dict)
    content_hash: str = ""
    chunker_name: str = ""
    chunker_version: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    provenance: Provenance


class Citation(BaseModel):
    marker: str  # e.g. "[1]"
    doc_id: str
    chunk_id: str
    source_uri: str
    source_kind: SourceKind = SourceKind.corpus
    page: int
    char_start: int
    char_end: int


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    usage: dict = Field(default_factory=dict)
    trace_id: str = ""
    retrieval_scope: RetrievalScope = RetrievalScope.corpus_only


class AssembledContext(BaseModel):
    text: str
    chunks: list[Chunk]  # order defines citation markers: chunks[i] -> "[i+1]"
    token_count: int


class Completion(BaseModel):
    text: str
    usage: dict = Field(default_factory=dict)


class Watermark(BaseModel):
    source_id: str
    etag: str | None = None
    last_modified: str | None = None
    updated_at: datetime | None = None


class MetadataFilter(BaseModel):
    region: str | None = None
    crop: str | None = None
    season: str | None = None

    def as_pairs(self) -> list[tuple[str, str]]:
        return [(k, v) for k, v in self.model_dump().items() if v is not None]


class UpsertStats(BaseModel):
    documents_upserted: int = 0
    chunks_upserted: int = 0
    skipped_unchanged: int = 0
```

`src/rag/protocols.py`:
```python
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from rag.models import (
    AssembledContext,
    Chunk,
    Completion,
    Document,
    MetadataFilter,
    RawDocument,
    ScoredChunk,
    UpsertStats,
    Watermark,
)

Vector = list[float]


@runtime_checkable
class SourceAdapter(Protocol):
    source_type: str

    def fetch(self, since: Watermark | None) -> Iterator[RawDocument]: ...


@runtime_checkable
class Parser(Protocol):
    def parse(self, raw: RawDocument) -> Document: ...


@runtime_checkable
class Cleaner(Protocol):
    def clean(self, doc: Document) -> Document: ...


@runtime_checkable
class Chunker(Protocol):
    name: str
    version: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    model: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


@runtime_checkable
class ChunkRepository(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> UpsertStats: ...
    def soft_delete(self, doc_ids: list[str]) -> int: ...
    def get_watermark(self, source_id: str) -> Watermark | None: ...
    def set_watermark(self, source_id: str, wm: Watermark) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self, query: str, k: int, filt: MetadataFilter | None
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class ContextAssembler(Protocol):
    def assemble(
        self, query: str, chunks: list[Chunk], token_budget: int
    ) -> AssembledContext: ...


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, messages: list[dict], **opts: object) -> Completion: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_models.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run mypy (these are the foundational types — keep them strict-clean)**

Run: `uv run mypy src`
Expected: Success.

- [ ] **Step 6: Commit**

```bash
git add src/rag/models.py src/rag/protocols.py tests/unit/test_models.py
git commit -m "feat(models): domain models, enums, and Protocol interfaces (spec §6)"
```

---

## Task 4: Utilities — content hash & token counting

**Files:**
- Create: `src/rag/util/__init__.py`, `src/rag/util/hashing.py`, `src/rag/util/tokens.py`
- Test: `tests/unit/test_util.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_util.py`:
```python
from rag.util.hashing import content_hash
from rag.util.tokens import count_tokens


def test_content_hash_is_stable_and_sensitive() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")
    assert len(content_hash("hello")) == 64  # sha256 hex


def test_count_tokens_monotonic() -> None:
    assert count_tokens("") == 0
    assert count_tokens("one two three") > 0
    assert count_tokens("a much longer sentence with more words") > count_tokens("short")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_util.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/util/__init__.py`:
```python
```
(empty file)

`src/rag/util/hashing.py`:
```python
import hashlib


def content_hash(text: str) -> str:
    """Stable SHA-256 hex digest of UTF-8 text — drives dedup and idempotency."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

`src/rag/util/tokens.py`:
```python
import tiktoken

# cl100k_base is a provider-agnostic budgeting proxy. Gemini does not use it for
# billing; we use it only to bound context size deterministically (spec §8).
_ENCODER = tiktoken.get_encoding("cl100k_base")


def get_encoder() -> tiktoken.Encoding:
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_util.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag/util tests/unit/test_util.py
git commit -m "feat(util): content_hash and tiktoken token counting"
```

---

## Task 5: PDF source adapter (PyMuPDF) + test fixture

**Files:**
- Create: `src/rag/ingestion/__init__.py`, `src/rag/ingestion/sources/__init__.py`, `src/rag/ingestion/sources/pdf.py`
- Create/modify: `tests/conftest.py`
- Test: `tests/unit/test_pdf_source.py`

- [ ] **Step 1: Add a deterministic 2-page PDF fixture to conftest**

`tests/conftest.py`:
```python
from pathlib import Path

import fitz  # PyMuPDF
import pytest


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """A deterministic 2-page text PDF used across ingestion tests."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Maize responds strongly to nitrogen fertilizer.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Wheat yields decline under prolonged drought stress.")
    path = tmp_path / "sample.pdf"
    doc.save(str(path))
    doc.close()
    return path
```

- [ ] **Step 2: Write the failing test**

`tests/unit/test_pdf_source.py`:
```python
from pathlib import Path

from rag.ingestion.sources.pdf import PdfSourceAdapter


def test_pdf_adapter_yields_one_rawdocument_per_file(sample_pdf_path: Path) -> None:
    adapter = PdfSourceAdapter(paths=[sample_pdf_path])
    docs = list(adapter.fetch(since=None))
    assert len(docs) == 1
    raw = docs[0]
    assert raw.source_type == "pdf"
    assert raw.uri == sample_pdf_path.as_uri()
    assert raw.raw_bytes is not None and len(raw.raw_bytes) > 0


def test_pdf_adapter_expands_directory(tmp_path: Path, sample_pdf_path: Path) -> None:
    adapter = PdfSourceAdapter(paths=[sample_pdf_path.parent])
    docs = list(adapter.fetch(since=None))
    assert len(docs) == 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_pdf_source.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Write minimal implementation**

`src/rag/ingestion/__init__.py` and `src/rag/ingestion/sources/__init__.py`: empty files.

`src/rag/ingestion/sources/pdf.py`:
```python
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from rag.models import RawDocument, Watermark


class PdfSourceAdapter:
    """Yields one RawDocument per .pdf file. Phase-0 full-scan only (since is ignored)."""

    source_type = "pdf"

    def __init__(self, paths: list[Path], source_id: str = "pdf-corpus") -> None:
        self._paths = paths
        self._source_id = source_id

    def fetch(self, since: Watermark | None) -> Iterator[RawDocument]:
        for pdf_path in self._iter_pdf_files():
            yield RawDocument(
                source_id=self._source_id,
                source_type=self.source_type,
                uri=pdf_path.as_uri(),
                raw_bytes=pdf_path.read_bytes(),
                fetched_at=datetime.now(timezone.utc),
                source_meta={"filename": pdf_path.name},
            )

    def _iter_pdf_files(self) -> Iterator[Path]:
        for p in self._paths:
            if p.is_dir():
                yield from sorted(p.rglob("*.pdf"))
            elif p.suffix.lower() == ".pdf":
                yield p
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_pdf_source.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/rag/ingestion tests/conftest.py tests/unit/test_pdf_source.py
git commit -m "feat(ingestion): PyMuPDF source adapter + PDF test fixture"
```

---

## Task 6: PDF parser (pages + offsets)

**Files:**
- Create: `src/rag/ingestion/parse.py`
- Test: `tests/unit/test_parse.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_parse.py`:
```python
from pathlib import Path

from rag.ingestion.parse import PdfParser
from rag.ingestion.sources.pdf import PdfSourceAdapter


def test_parser_builds_document_with_pages(sample_pdf_path: Path) -> None:
    raw = next(PdfSourceAdapter(paths=[sample_pdf_path]).fetch(since=None))
    doc = PdfParser().parse(raw)

    assert len(doc.pages) == 2
    assert "nitrogen" in doc.pages[0].text
    assert "drought" in doc.pages[1].text
    assert doc.source_type == "pdf"
    assert doc.content_hash != ""


def test_parser_page_offsets_are_consistent(sample_pdf_path: Path) -> None:
    raw = next(PdfSourceAdapter(paths=[sample_pdf_path]).fetch(since=None))
    doc = PdfParser().parse(raw)
    # page_at on the first char of page 2 must return 2
    _, start_p2, _ = doc.page_spans()[1]
    assert doc.page_at(start_p2) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_parse.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/ingestion/parse.py`:
```python
from __future__ import annotations

import uuid

import fitz  # PyMuPDF

from rag.models import Document, Page, RawDocument
from rag.util.hashing import content_hash


class PdfParser:
    """Extracts per-page text from a PDF's text layer (no OCR — that is P0c)."""

    def parse(self, raw: RawDocument) -> Document:
        if raw.raw_bytes is None:
            raise ValueError(f"PdfParser requires raw_bytes; got none for {raw.uri}")

        pages: list[Page] = []
        with fitz.open(stream=raw.raw_bytes, filetype="pdf") as pdf:
            for i, page in enumerate(pdf):
                text = page.get_text("text") or ""
                pages.append(Page(number=i + 1, text=text.strip()))

        doc = Document(
            doc_id=str(uuid.uuid5(uuid.NAMESPACE_URL, raw.uri)),  # stable per URI -> idempotent re-ingest
            source_id=raw.source_id,
            source_type=raw.source_type,
            uri=raw.uri,
            pages=pages,
            license=raw.license,
            metadata=dict(raw.source_meta),
        )
        doc.content_hash = content_hash(doc.text)
        return doc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_parse.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag/ingestion/parse.py tests/unit/test_parse.py
git commit -m "feat(ingestion): PDF parser producing pages with offset map"
```

---

## Task 7: Cleaner (offset-consistent normalization)

**Files:**
- Create: `src/rag/ingestion/clean.py`
- Test: `tests/unit/test_clean.py`

The cleaner normalizes **per page**, then the Document recomputes `text`/offsets from the cleaned pages — so offsets always match the stored text. Keep cleaning conservative in P0a (collapse runs of whitespace, strip control chars). Aggressive layout cleaning is P0c.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_clean.py`:
```python
from rag.ingestion.clean import BasicCleaner
from rag.models import Document, Page


def test_cleaner_collapses_whitespace_and_recomputes_hash() -> None:
    doc = Document(
        doc_id="d1",
        source_id="s1",
        source_type="pdf",
        uri="file:///x.pdf",
        pages=[Page(number=1, text="hello   \t  world\n\n\n bye")],
        content_hash="stale",
    )
    cleaned = BasicCleaner().clean(doc)
    assert cleaned.pages[0].text == "hello world\nbye"
    # offsets are derived from cleaned text, so the invariant holds for any slice
    assert cleaned.text[0:5] == "hello"
    assert cleaned.content_hash != "stale"


def test_cleaner_preserves_page_count() -> None:
    doc = Document(
        doc_id="d1",
        source_id="s1",
        source_type="pdf",
        uri="file:///x.pdf",
        pages=[Page(number=1, text="a  b"), Page(number=2, text="c   d")],
    )
    cleaned = BasicCleaner().clean(doc)
    assert [p.number for p in cleaned.pages] == [1, 2]
    assert cleaned.pages[1].text == "c d"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_clean.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/ingestion/clean.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_clean.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag/ingestion/clean.py tests/unit/test_clean.py
git commit -m "feat(ingestion): conservative offset-consistent cleaner"
```

---

## Task 8: Fixed-token chunker

**Files:**
- Create: `src/rag/ingestion/chunking/__init__.py`, `src/rag/ingestion/chunking/fixed.py`
- Test: `tests/unit/test_chunking.py`

This is the highest-value deterministic logic in ingestion — test it hard. The key invariant: `chunk.text == doc.text[chunk.char_start:chunk.char_end]`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_chunking.py`:
```python
from rag.ingestion.chunking.fixed import FixedTokenChunker
from rag.models import Document, Page
from rag.util.tokens import count_tokens


def _doc(text: str, pages: int = 1) -> Document:
    return Document(
        doc_id="d1",
        source_id="s1",
        source_type="pdf",
        uri="file:///x.pdf",
        pages=[Page(number=1, text=text)],
    )


def test_chunk_spans_reconstruct_source_text() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    doc = _doc(text)
    chunks = FixedTokenChunker(chunk_tokens=50, overlap=10).chunk(doc)

    assert len(chunks) > 1
    for ch in chunks:
        assert ch.text == doc.text[ch.char_start : ch.char_end]  # the invariant
        assert ch.token_count == count_tokens(ch.text)
        assert ch.chunker_name == "fixed"
        assert ch.chunker_version == "1"


def test_chunks_are_ordinal_and_within_budget() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    chunks = FixedTokenChunker(chunk_tokens=50, overlap=10).chunk(_doc(text))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for ch in chunks:
        assert ch.token_count <= 50


def test_overlap_creates_shared_text() -> None:
    text = " ".join(f"word{i}" for i in range(200))
    chunks = FixedTokenChunker(chunk_tokens=40, overlap=10).chunk(_doc(text))
    # consecutive chunks overlap in character space
    assert chunks[1].char_start < chunks[0].char_end


def test_pages_assigned_from_offsets() -> None:
    doc = Document(
        doc_id="d1",
        source_id="s1",
        source_type="pdf",
        uri="file:///x.pdf",
        pages=[
            Page(number=1, text=" ".join(f"a{i}" for i in range(100))),
            Page(number=2, text=" ".join(f"b{i}" for i in range(100))),
        ],
    )
    chunks = FixedTokenChunker(chunk_tokens=30, overlap=5).chunk(doc)
    assert chunks[0].page == 1
    assert chunks[-1].page == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/ingestion/chunking/__init__.py`: empty file.

`src/rag/ingestion/chunking/fixed.py`:
```python
from __future__ import annotations

import uuid

from rag.models import Chunk, Document
from rag.util.hashing import content_hash
from rag.util.tokens import get_encoder


class FixedTokenChunker:
    """Sliding fixed-size token windows with overlap.

    Char offsets are recovered from token offsets via the concatenative property
    of byte-level BPE: ``decode(tokens[:i])`` is exactly ``text[:char_start]``.
    """

    name = "fixed"
    version = "1"

    def __init__(self, chunk_tokens: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_tokens:
            raise ValueError("overlap must be smaller than chunk_tokens")
        self._chunk_tokens = chunk_tokens
        self._overlap = overlap
        self._enc = get_encoder()

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text
        tokens = self._enc.encode(text)
        if not tokens:
            return []

        stride = self._chunk_tokens - self._overlap
        chunks: list[Chunk] = []
        ordinal = 0
        for i in range(0, len(tokens), stride):
            window = tokens[i : i + self._chunk_tokens]
            char_start = len(self._enc.decode(tokens[:i]))
            chunk_text = self._enc.decode(window)
            char_end = char_start + len(chunk_text)
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.doc_id}:{ordinal}")),
                    doc_id=doc.doc_id,
                    source_uri=doc.uri,
                    text=chunk_text,
                    ordinal=ordinal,
                    page=doc.page_at(char_start),
                    char_start=char_start,
                    char_end=char_end,
                    token_count=len(window),
                    metadata=dict(doc.metadata),
                    content_hash=content_hash(chunk_text),
                    chunker_name=self.name,
                    chunker_version=self.version,
                )
            )
            ordinal += 1
            if i + self._chunk_tokens >= len(tokens):
                break
        return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_chunking.py -v`
Expected: 4 passed. (If `test_chunk_spans_reconstruct_source_text` fails, the token→char mapping is wrong — do not patch the test; fix the decode math.)

- [ ] **Step 5: Commit**

```bash
git add src/rag/ingestion/chunking tests/unit/test_chunking.py
git commit -m "feat(ingestion): fixed-token chunker with char-offset invariant"
```

---

## Task 9: Embedding provider (LiteLLM) + FakeEmbedder

**Files:**
- Create: `src/rag/providers/__init__.py`, `src/rag/providers/embeddings.py`
- Create: `tests/unit/fakes.py`
- Test: `tests/unit/test_embeddings.py`

> **LiteLLM note (spec §18):** the embedding call surface is verified against current docs: `litellm.embedding(model="gemini/text-embedding-004", input=[...])`, vectors at `resp.data[i]["embedding"]`. Before running the **live** test, re-skim https://docs.litellm.ai/docs/embedding for any change to batching limits.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_embeddings.py`:
```python
from types import SimpleNamespace

from rag.providers.embeddings import LiteLLMEmbeddingProvider


def test_embed_returns_vectors_and_batches(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_embedding(model: str, input: list[str], **kwargs):  # noqa: A002
        calls.append(list(input))
        data = [{"embedding": [float(len(t))] * 3} for t in input]
        return SimpleNamespace(data=data)

    monkeypatch.setattr("rag.providers.embeddings.litellm.embedding", fake_embedding)

    provider = LiteLLMEmbeddingProvider(
        model="gemini/text-embedding-004", dim=3, batch_size=2
    )
    vectors = provider.embed(["a", "bb", "ccc"])

    assert vectors == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    assert calls == [["a", "bb"], ["ccc"]]  # batched in groups of 2


def test_embed_empty_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag.providers.embeddings.litellm.embedding",
        lambda **k: (_ for _ in ()).throw(AssertionError("should not call")),
    )
    assert LiteLLMEmbeddingProvider(model="m", dim=3).embed([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/providers/__init__.py`: empty file.

`src/rag/providers/embeddings.py`:
```python
from __future__ import annotations

from collections.abc import Sequence

import litellm

Vector = list[float]


class LiteLLMEmbeddingProvider:
    """Embeddings via LiteLLM. Default model is Google text-embedding-004 (768-d)."""

    def __init__(self, model: str, dim: int, batch_size: int = 96) -> None:
        self.model = model
        self.dim = dim
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        out: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            resp = litellm.embedding(model=self.model, input=batch)
            out.extend([list(item["embedding"]) for item in resp.data])
        return out
```

- [ ] **Step 4: Add the fakes module (used here and in later tasks)**

`tests/unit/fakes.py`:
```python
from __future__ import annotations

from collections.abc import Sequence

from rag.models import Completion

Vector = list[float]


class FakeEmbedder:
    """Deterministic embeddings for tests. Optionally maps specific texts to vectors."""

    def __init__(self, dim: int = 3, mapping: dict[str, Vector] | None = None) -> None:
        self.model = "fake-embed"
        self.dim = dim
        self._mapping = mapping or {}

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector_for(t) for t in texts]

    def _vector_for(self, text: str) -> Vector:
        if text in self._mapping:
            return list(self._mapping[text])
        # deterministic, length-based fallback vector
        seed = float(len(text) % 7 + 1)
        return [seed] * self.dim


class FakeLLMProvider:
    """Returns a canned completion; records the messages it was given."""

    def __init__(self, reply: str = "Answer grounded in context [1].") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], **opts: object) -> Completion:
        self.calls.append(messages)
        return Completion(
            text=self.reply,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_embeddings.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/rag/providers tests/unit/test_embeddings.py tests/unit/fakes.py
git commit -m "feat(providers): LiteLLM embedding provider + test fakes"
```

---

## Task 10: DB layer + Alembic initial migration

**Files:**
- Create: `src/rag/db.py`, `alembic.ini`, `alembic/env.py`, `alembic/versions/0001_initial_schema.py`, `alembic/script.py.mako`
- Test: `tests/integration/test_migrations.py`

> **Needs Docker.** This task introduces testcontainers. The `migrated_engine` fixture added here is reused by Tasks 11–13.

- [ ] **Step 1: Write the Core tables + engine + migration runner**

`src/rag/db.py`:
```python
from __future__ import annotations

from alembic import command
from alembic.config import Config
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
)

EMBEDDING_DIM = 768  # P0a default (text-embedding-004). A model swap = a new migration.

metadata_obj = MetaData()

documents = Table(
    "documents",
    metadata_obj,
    Column("doc_id", String, primary_key=True),
    Column("source_id", String, nullable=False),
    Column("source_type", String, nullable=False),
    Column("uri", Text, nullable=False),
    Column("document_version", Integer, nullable=False, default=1),
    Column("content_hash", String, nullable=False),
    Column("metadata", JSON, nullable=False, default=dict),
    Column("license", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
)

chunks = Table(
    "chunks",
    metadata_obj,
    Column("chunk_id", String, primary_key=True),
    Column("doc_id", String, nullable=False),
    Column("source_uri", Text, nullable=False),
    Column("text", Text, nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("page", Integer, nullable=False),
    Column("char_start", Integer, nullable=False),
    Column("char_end", Integer, nullable=False),
    Column("token_count", Integer, nullable=False),
    Column("metadata", JSON, nullable=False, default=dict),
    Column("content_hash", String, nullable=False),
    Column("chunker_name", String, nullable=False),
    Column("chunker_version", String, nullable=False),
    Column("embedding_model", String, nullable=False),
    Column("embedding_dim", Integer, nullable=False),
    Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("deleted_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint("doc_id", "ordinal", name="uq_chunks_doc_ordinal"),
)

watermarks = Table(
    "source_watermarks",
    metadata_obj,
    Column("source_id", String, primary_key=True),
    Column("etag", String, nullable=True),
    Column("last_modified", String, nullable=True),
    Column("updated_at", DateTime(timezone=True), nullable=True),
)


def get_engine(database_url: str) -> Engine:
    return create_engine(database_url, future=True, pool_size=10, max_overflow=5)


def run_migrations(database_url: str) -> None:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(cfg, "head")
```

- [ ] **Step 2: Configure Alembic**

`alembic.ini` (minimal):
```ini
[alembic]
script_location = alembic
sqlalchemy.url = postgresql+psycopg://rag:rag@localhost:5432/rag

[loggers]
keys = root

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARNING
handlers = console

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

`alembic/env.py`:
```python
from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from rag.db import metadata_obj

config = context.config
if os.environ.get("DATABASE_URL"):
    config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])

target_metadata = metadata_obj


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
```

`alembic/script.py.mako`:
```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 3: Write the initial migration**

`alembic/versions/0001_initial_schema.py`:
```python
"""initial schema: pgvector extension, documents, chunks, HNSW index

Revision ID: 0001
Revises:
"""
from __future__ import annotations

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("doc_id", sa.String(), primary_key=True),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("document_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("license", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])

    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(), primary_key=True),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("chunker_name", sa.String(), nullable=False),
        sa.Column("chunker_version", sa.String(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("doc_id", "ordinal", name="uq_chunks_doc_ordinal"),
    )
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "source_watermarks",
        sa.Column("source_id", sa.String(), primary_key=True),
        sa.Column("etag", sa.String(), nullable=True),
        sa.Column("last_modified", sa.String(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("source_watermarks")
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_index("ix_chunks_doc_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
```

- [ ] **Step 4: Add the shared testcontainers fixture**

Append to `tests/conftest.py`:
```python
from collections.abc import Iterator

from sqlalchemy import Engine, text


@pytest.fixture(scope="session")
def migrated_engine() -> Iterator[Engine]:
    """Boots a real pgvector Postgres, runs Alembic migrations, yields an Engine."""
    from testcontainers.postgres import PostgresContainer

    from rag.db import get_engine, run_migrations

    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        url = pg.get_connection_url().replace("postgresql+psycopg2", "postgresql+psycopg")
        run_migrations(url)
        engine = get_engine(url)
        yield engine
        engine.dispose()


@pytest.fixture
def clean_db(migrated_engine: Engine) -> Engine:
    """Truncates tables between tests so each test starts empty."""
    with migrated_engine.begin() as conn:
        conn.execute(text("TRUNCATE chunks, documents, source_watermarks"))
    return migrated_engine
```

- [ ] **Step 5: Write the integration test**

`tests/integration/test_migrations.py`:
```python
import pytest
from sqlalchemy import Engine, inspect, text

pytestmark = pytest.mark.integration


def test_migration_creates_tables_and_hnsw_index(migrated_engine: Engine) -> None:
    insp = inspect(migrated_engine)
    assert {"documents", "chunks", "source_watermarks"} <= set(insp.get_table_names())

    with migrated_engine.connect() as conn:
        idx = conn.execute(
            text("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'")
        ).scalars().all()
    assert "ix_chunks_embedding_hnsw" in idx
```

- [ ] **Step 6: Run the integration test (Docker required)**

Run: `uv run pytest tests/integration/test_migrations.py -m integration -v`
Expected: 1 passed (first run pulls the `pgvector/pgvector:pg16` image — may take a minute).

- [ ] **Step 7: Commit**

```bash
git add src/rag/db.py alembic.ini alembic tests/conftest.py tests/integration/test_migrations.py
git commit -m "feat(db): Core schema + Alembic migration (pgvector, HNSW) + testcontainers fixture"
```

---

## Task 11: ChunkRepository (upsert / soft_delete / watermark)

**Files:**
- Create: `src/rag/ingestion/repository.py`
- Test: `tests/integration/test_repository.py`

- [ ] **Step 1: Write the failing integration test**

`tests/integration/test_repository.py`:
```python
import pytest
from sqlalchemy import Engine, text

from rag.ingestion.repository import PgChunkRepository
from rag.models import Chunk, Watermark

pytestmark = pytest.mark.integration


def _chunk(doc_id: str, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-c{ordinal}",
        doc_id=doc_id,
        source_uri="file:///x.pdf",
        text=f"text {ordinal}",
        ordinal=ordinal,
        page=1,
        char_start=ordinal * 10,
        char_end=ordinal * 10 + 6,
        token_count=2,
        content_hash=f"h{doc_id}",
        chunker_name="fixed",
        chunker_version="1",
        embedding_model="fake",
        embedding_dim=3,
    )


def test_upsert_is_idempotent(clean_db: Engine) -> None:
    repo = PgChunkRepository(clean_db)
    chunks = [_chunk("d1", 0), _chunk("d1", 1)]
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    repo.upsert(chunks, vectors)
    repo.upsert(chunks, vectors)  # second run must not duplicate

    with clean_db.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
        d = conn.execute(text("SELECT count(*) FROM documents")).scalar_one()
    assert n == 2
    assert d == 1


def test_soft_delete_tombstones_document_and_chunks(clean_db: Engine) -> None:
    repo = PgChunkRepository(clean_db)
    repo.upsert([_chunk("d1", 0)], [[0.1, 0.2, 0.3]])

    deleted = repo.soft_delete(["d1"])
    assert deleted == 1

    with clean_db.connect() as conn:
        live = conn.execute(
            text("SELECT count(*) FROM chunks WHERE deleted_at IS NULL")
        ).scalar_one()
    assert live == 0


def test_watermark_roundtrip(clean_db: Engine) -> None:
    repo = PgChunkRepository(clean_db)
    assert repo.get_watermark("s1") is None
    repo.set_watermark("s1", Watermark(source_id="s1", etag="abc"))
    wm = repo.get_watermark("s1")
    assert wm is not None and wm.etag == "abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_repository.py -m integration -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.ingestion.repository'`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/ingestion/repository.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Engine, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from rag.db import chunks as chunks_t
from rag.db import documents as documents_t
from rag.db import watermarks as watermarks_t
from rag.models import Chunk, UpsertStats, Watermark

Vector = list[float]


class PgChunkRepository:
    """The only component that touches Postgres. Idempotent via content_hash/ordinal."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> UpsertStats:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must align")
        now = datetime.now(timezone.utc)
        stats = UpsertStats()
        by_doc: dict[str, list[tuple[Chunk, Vector]]] = {}
        for ch, vec in zip(chunks, vectors, strict=True):
            by_doc.setdefault(ch.doc_id, []).append((ch, vec))

        with self._engine.begin() as conn:
            for doc_id, items in by_doc.items():
                first = items[0][0]
                doc_stmt = (
                    pg_insert(documents_t)
                    .values(
                        doc_id=doc_id,
                        source_id=first.metadata.get("source_id", "pdf-corpus"),
                        source_type="pdf",
                        uri=first.source_uri,
                        document_version=1,
                        content_hash=first.metadata.get(
                            "document_content_hash", first.content_hash
                        ),
                        metadata=first.metadata,
                        created_at=now,
                        updated_at=now,
                    )
                    .on_conflict_do_update(
                        index_elements=["doc_id"],
                        set_={
                            "content_hash": first.metadata.get(
                                "document_content_hash", first.content_hash
                            ),
                            "metadata": first.metadata,
                            "updated_at": now,
                            "deleted_at": None,
                        },
                    )
                )
                conn.execute(doc_stmt)
                stats.documents_upserted += 1

                for ch, vec in items:
                    chunk_stmt = (
                        pg_insert(chunks_t)
                        .values(
                            chunk_id=ch.chunk_id,
                            doc_id=ch.doc_id,
                            source_uri=ch.source_uri,
                            text=ch.text,
                            ordinal=ch.ordinal,
                            page=ch.page,
                            char_start=ch.char_start,
                            char_end=ch.char_end,
                            token_count=ch.token_count,
                            metadata=ch.metadata,
                            content_hash=ch.content_hash,
                            chunker_name=ch.chunker_name,
                            chunker_version=ch.chunker_version,
                            embedding_model=ch.embedding_model,
                            embedding_dim=ch.embedding_dim,
                            embedding=vec,
                            created_at=now,
                        )
                        .on_conflict_do_update(
                            constraint="uq_chunks_doc_ordinal",
                            set_={
                                "text": ch.text,
                                "embedding": vec,
                                "content_hash": ch.content_hash,
                                "char_start": ch.char_start,
                                "char_end": ch.char_end,
                                "page": ch.page,
                                "token_count": ch.token_count,
                                "deleted_at": None,
                            },
                        )
                    )
                    conn.execute(chunk_stmt)
                    stats.chunks_upserted += 1
        return stats

    def soft_delete(self, doc_ids: list[str]) -> int:
        if not doc_ids:
            return 0
        now = datetime.now(timezone.utc)
        with self._engine.begin() as conn:
            conn.execute(
                chunks_t.update()
                .where(chunks_t.c.doc_id.in_(doc_ids))
                .values(deleted_at=now)
            )
            result = conn.execute(
                documents_t.update()
                .where(documents_t.c.doc_id.in_(doc_ids))
                .values(deleted_at=now)
            )
            return int(result.rowcount or 0)

    def get_watermark(self, source_id: str) -> Watermark | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(watermarks_t).where(watermarks_t.c.source_id == source_id)
            ).mappings().first()
        if row is None:
            return None
        return Watermark(
            source_id=row["source_id"],
            etag=row["etag"],
            last_modified=row["last_modified"],
            updated_at=row["updated_at"],
        )

    def set_watermark(self, source_id: str, wm: Watermark) -> None:
        now = datetime.now(timezone.utc)
        stmt = (
            pg_insert(watermarks_t)
            .values(
                source_id=source_id,
                etag=wm.etag,
                last_modified=wm.last_modified,
                updated_at=now,
            )
            .on_conflict_do_update(
                index_elements=["source_id"],
                set_={"etag": wm.etag, "last_modified": wm.last_modified, "updated_at": now},
            )
        )
        with self._engine.begin() as conn:
            conn.execute(stmt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_repository.py -m integration -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag/ingestion/repository.py tests/integration/test_repository.py
git commit -m "feat(ingestion): Postgres chunk repository (idempotent upsert, tombstones, watermark)"
```

---

## Task 12: Ingestion pipeline + CLI

**Files:**
- Create: `src/rag/ingestion/pipeline.py`, `src/rag/ingestion/cli.py`
- Test: `tests/integration/test_pipeline.py`

- [ ] **Step 1: Write the failing integration test**

`tests/integration/test_pipeline.py`:
```python
from pathlib import Path

import pytest
from sqlalchemy import Engine, text

from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.repository import PgChunkRepository

from tests.unit.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


def _pipeline(engine: Engine) -> IngestionPipeline:
    return IngestionPipeline(
        repository=PgChunkRepository(engine),
        embedder=FakeEmbedder(dim=768),
        chunk_tokens=64,
        overlap=8,
    )


def test_pipeline_ingests_pdf_into_postgres(clean_db: Engine, sample_pdf_path: Path) -> None:
    stats = _pipeline(clean_db).ingest_paths([sample_pdf_path])

    assert stats.documents_upserted == 1
    assert stats.chunks_upserted >= 1
    with clean_db.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
        dim = conn.execute(text("SELECT embedding_dim FROM chunks LIMIT 1")).scalar_one()
    assert n == stats.chunks_upserted
    assert dim == 768


def test_pipeline_is_idempotent(clean_db: Engine, sample_pdf_path: Path) -> None:
    pipe = _pipeline(clean_db)
    first = pipe.ingest_paths([sample_pdf_path])
    pipe.ingest_paths([sample_pdf_path])
    with clean_db.connect() as conn:
        docs = conn.execute(text("SELECT count(*) FROM documents")).scalar_one()
        n_chunks = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
    assert docs == 1
    assert n_chunks == first.chunks_upserted  # stable doc_id => no duplicate chunks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_pipeline.py -m integration -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/ingestion/pipeline.py`:
```python
from __future__ import annotations

from pathlib import Path

import structlog

from rag.ingestion.chunking.fixed import FixedTokenChunker
from rag.ingestion.clean import BasicCleaner
from rag.ingestion.parse import PdfParser
from rag.ingestion.repository import PgChunkRepository
from rag.ingestion.sources.pdf import PdfSourceAdapter
from rag.models import Chunk, UpsertStats
from rag.protocols import EmbeddingProvider

log = structlog.get_logger()


class IngestionPipeline:
    """Wires the write path: PDF -> parse -> clean -> chunk -> embed -> upsert."""

    def __init__(
        self,
        repository: PgChunkRepository,
        embedder: EmbeddingProvider,
        chunk_tokens: int = 512,
        overlap: int = 64,
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._parser = PdfParser()
        self._cleaner = BasicCleaner()
        self._chunker = FixedTokenChunker(chunk_tokens=chunk_tokens, overlap=overlap)

    def ingest_paths(self, paths: list[Path]) -> UpsertStats:
        adapter = PdfSourceAdapter(paths=paths)
        total = UpsertStats()
        for raw in adapter.fetch(since=None):
            try:
                stats = self._ingest_one(raw)
            except Exception as exc:  # per-doc isolation (spec §16)
                log.error("ingest_failed", uri=raw.uri, error=str(exc))
                continue
            total.documents_upserted += stats.documents_upserted
            total.chunks_upserted += stats.chunks_upserted
        return total

    def _ingest_one(self, raw) -> UpsertStats:  # type: ignore[no-untyped-def]
        doc = self._cleaner.clean(self._parser.parse(raw))
        chunks: list[Chunk] = self._chunker.chunk(doc)
        if not chunks:
            log.warning("no_chunks", uri=raw.uri)
            return UpsertStats()
        for ch in chunks:
            ch.embedding_model = self._embedder.model
            ch.embedding_dim = self._embedder.dim
            ch.metadata = {
                **ch.metadata,
                "source_id": doc.source_id,
                "document_content_hash": doc.content_hash,
            }
        vectors = self._embedder.embed([c.text for c in chunks])
        stats = self._repo.upsert(chunks, vectors)
        log.info("ingested", uri=raw.uri, chunks=stats.chunks_upserted)
        return stats
```

`src/rag/ingestion/cli.py`:
```python
from __future__ import annotations

import argparse
from pathlib import Path

import structlog

from rag.config import get_settings
from rag.db import get_engine
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.repository import PgChunkRepository
from rag.providers.embeddings import LiteLLMEmbeddingProvider

log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG store.")
    parser.add_argument("paths", nargs="+", type=Path, help="PDF files or directories")
    args = parser.parse_args()

    settings = get_settings()
    engine = get_engine(settings.database_url)
    pipeline = IngestionPipeline(
        repository=PgChunkRepository(engine),
        embedder=LiteLLMEmbeddingProvider(
            model=settings.embedding_model, dim=settings.embedding_dim
        ),
        chunk_tokens=settings.chunk_tokens,
        overlap=settings.chunk_overlap,
    )
    stats = pipeline.ingest_paths(args.paths)
    log.info("ingest_complete", documents=stats.documents_upserted, chunks=stats.chunks_upserted)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_pipeline.py -m integration -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag/ingestion/pipeline.py src/rag/ingestion/cli.py tests/integration/test_pipeline.py
git commit -m "feat(ingestion): end-to-end pipeline + ingest CLI"
```

---

## Task 13: DenseRetriever (pgvector KNN + metadata filter)

**Files:**
- Create: `src/rag/retrieval/__init__.py`, `src/rag/retrieval/dense.py`
- Test: `tests/integration/test_dense_retrieval.py`

- [ ] **Step 1: Write the failing integration test**

`tests/integration/test_dense_retrieval.py`:
```python
import pytest
from sqlalchemy import Engine

from rag.ingestion.repository import PgChunkRepository
from rag.models import Chunk, MetadataFilter, Provenance
from rag.retrieval.dense import DenseRetriever

from tests.unit.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


def _chunk(cid: str, text: str, meta: dict | None = None) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="d1",
        source_uri="file:///x.pdf",
        text=text,
        ordinal=int(cid[-1]),
        page=1,
        char_start=0,
        char_end=len(text),
        token_count=2,
        metadata=meta or {},
        content_hash=cid,
        chunker_name="fixed",
        chunker_version="1",
        embedding_model="fake",
        embedding_dim=3,
    )


def _seed(engine: Engine) -> FakeEmbedder:
    # explicit vectors: 'near' is closest to the query vector [1,0,0]
    mapping = {"near": [1.0, 0.0, 0.0], "mid": [0.7, 0.7, 0.0], "far": [0.0, 0.0, 1.0]}
    embedder = FakeEmbedder(dim=3, mapping={**mapping, "QUERY": [1.0, 0.0, 0.0]})
    repo = PgChunkRepository(engine)
    chunks = [_chunk("c0", "near"), _chunk("c1", "mid"), _chunk("c2", "far")]
    repo.upsert(chunks, [mapping["near"], mapping["mid"], mapping["far"]])
    return embedder


def test_dense_retrieval_orders_by_cosine(clean_db: Engine) -> None:
    embedder = _seed(clean_db)
    retriever = DenseRetriever(engine=clean_db, embedder=embedder)
    results = retriever.retrieve("QUERY", k=3, filt=None)

    assert [r.chunk.text for r in results] == ["near", "mid", "far"]
    assert results[0].provenance == Provenance.dense
    assert results[0].score >= results[1].score


def test_dense_retrieval_applies_metadata_filter(clean_db: Engine) -> None:
    mapping = {"south": [1.0, 0.0, 0.0], "north": [0.9, 0.1, 0.0]}
    embedder = FakeEmbedder(dim=3, mapping={**mapping, "QUERY": [1.0, 0.0, 0.0]})
    repo = PgChunkRepository(clean_db)
    repo.upsert(
        [_chunk("c0", "south", {"region": "south"}), _chunk("c1", "north", {"region": "north"})],
        [mapping["south"], mapping["north"]],
    )
    retriever = DenseRetriever(engine=clean_db, embedder=embedder)
    results = retriever.retrieve("QUERY", k=5, filt=MetadataFilter(region="south"))
    assert [r.chunk.text for r in results] == ["south"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_dense_retrieval.py -m integration -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/retrieval/__init__.py`: empty file.

`src/rag/retrieval/dense.py`:
```python
from __future__ import annotations

from sqlalchemy import Engine, select

from rag.db import chunks as chunks_t
from rag.models import Chunk, MetadataFilter, Provenance, ScoredChunk
from rag.protocols import EmbeddingProvider


class DenseRetriever:
    """pgvector cosine KNN over live (non-tombstoned) chunks, with optional filter."""

    def __init__(self, engine: Engine, embedder: EmbeddingProvider) -> None:
        self._engine = engine
        self._embedder = embedder

    def retrieve(
        self, query: str, k: int, filt: MetadataFilter | None
    ) -> list[ScoredChunk]:
        qvec = self._embedder.embed([query])[0]
        distance = chunks_t.c.embedding.cosine_distance(qvec)
        stmt = (
            select(chunks_t, distance.label("distance"))
            .where(chunks_t.c.deleted_at.is_(None))
            .order_by(distance)
            .limit(k)
        )
        if filt is not None:
            for key, value in filt.as_pairs():
                stmt = stmt.where(chunks_t.c.metadata[key].astext == value)

        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()

        return [self._to_scored(row) for row in rows]

    @staticmethod
    def _to_scored(row) -> ScoredChunk:  # type: ignore[no-untyped-def]
        chunk = Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            source_uri=row["source_uri"],
            text=row["text"],
            ordinal=row["ordinal"],
            page=row["page"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            token_count=row["token_count"],
            metadata=row["metadata"],
            content_hash=row["content_hash"],
            chunker_name=row["chunker_name"],
            chunker_version=row["chunker_version"],
            embedding_model=row["embedding_model"],
            embedding_dim=row["embedding_dim"],
        )
        score = 1.0 - float(row["distance"])  # cosine similarity
        return ScoredChunk(chunk=chunk, score=score, provenance=Provenance.dense)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_dense_retrieval.py -m integration -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag/retrieval tests/integration/test_dense_retrieval.py
git commit -m "feat(retrieval): dense pgvector KNN retriever with metadata filter"
```

---

## Task 14: ContextAssembler (budget, dedup, citation numbering)

**Files:**
- Create: `src/rag/generation/__init__.py`, `src/rag/generation/assembler.py`
- Test: `tests/unit/test_assembler.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_assembler.py`:
```python
from rag.generation.assembler import TokenBudgetAssembler
from rag.models import Chunk


def _chunk(cid: str, text: str, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="d1",
        source_uri="file:///x.pdf",
        text=text,
        ordinal=ordinal,
        page=ordinal + 1,
        char_start=0,
        char_end=len(text),
        token_count=0,
        chunker_name="fixed",
        chunker_version="1",
    )


def test_assembler_numbers_chunks_and_builds_context() -> None:
    chunks = [_chunk("a", "alpha fact", 0), _chunk("b", "bravo fact", 1)]
    ctx = TokenBudgetAssembler().assemble("q", chunks, token_budget=1000)

    assert ctx.chunks[0].chunk_id == "a"
    assert "[1]" in ctx.text and "[2]" in ctx.text
    assert "alpha fact" in ctx.text


def test_assembler_dedupes_by_chunk_id() -> None:
    chunks = [_chunk("a", "alpha", 0), _chunk("a", "alpha", 0)]
    ctx = TokenBudgetAssembler().assemble("q", chunks, token_budget=1000)
    assert len(ctx.chunks) == 1


def test_assembler_truncates_to_budget() -> None:
    chunks = [_chunk(str(i), f"sentence number {i} here", i) for i in range(50)]
    ctx = TokenBudgetAssembler().assemble("q", chunks, token_budget=20)
    assert len(ctx.chunks) < 50
    assert ctx.token_count <= 20 or len(ctx.chunks) == 1  # always keep at least one
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_assembler.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/generation/__init__.py`: empty file.

`src/rag/generation/assembler.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_assembler.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag/generation/__init__.py src/rag/generation/assembler.py tests/unit/test_assembler.py
git commit -m "feat(generation): token-budget context assembler with citation numbering"
```

---

## Task 15: Citation validation (cite only assembled chunks)

**Files:**
- Create: `src/rag/generation/citations.py`
- Test: `tests/unit/test_citations.py`

This implements the P0a trust rule: **a citation marker that does not map to an assembled chunk is stripped from the answer text and never returned as a Citation.**

- [ ] **Step 1: Write the failing test**

`tests/unit/test_citations.py`:
```python
from rag.generation.citations import validate_and_build_citations
from rag.models import AssembledContext, Chunk


def _ctx(chunk_ids: list[str]) -> AssembledContext:
    chunks = [
        Chunk(
            chunk_id=cid,
            doc_id="d1",
            source_uri="file:///x.pdf",
            text=f"text {cid}",
            ordinal=i,
            page=i + 1,
            char_start=i * 10,
            char_end=i * 10 + 6,
            token_count=2,
            chunker_name="fixed",
            chunker_version="1",
        )
        for i, cid in enumerate(chunk_ids)
    ]
    return AssembledContext(text="ctx", chunks=chunks, token_count=1)


def test_valid_markers_become_citations() -> None:
    ctx = _ctx(["a", "b"])
    text, citations = validate_and_build_citations("Nitrogen helps [1]. Drought hurts [2].", ctx)

    assert text == "Nitrogen helps [1]. Drought hurts [2]."
    assert [c.marker for c in citations] == ["[1]", "[2]"]
    assert citations[0].chunk_id == "a"
    assert citations[0].page == 1
    assert citations[1].chunk_id == "b"


def test_out_of_range_marker_is_stripped() -> None:
    ctx = _ctx(["a"])  # only [1] is valid
    text, citations = validate_and_build_citations("Real [1]. Fabricated [2][3].", ctx)

    assert "[2]" not in text and "[3]" not in text
    assert text == "Real [1]. Fabricated ."
    assert [c.marker for c in citations] == ["[1]"]


def test_each_chunk_cited_once_even_if_repeated() -> None:
    ctx = _ctx(["a", "b"])
    _, citations = validate_and_build_citations("A [1]. Again [1]. B [2].", ctx)
    assert [c.marker for c in citations] == ["[1]", "[2]"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_citations.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/generation/citations.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_citations.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag/generation/citations.py tests/unit/test_citations.py
git commit -m "feat(generation): citation validation — cite only assembled chunks"
```

---

## Task 16: LLM provider (LiteLLM) + prompt template

**Files:**
- Create: `src/rag/providers/llm.py`, `src/rag/generation/prompts/answer_v1.md`
- Test: `tests/unit/test_llm_provider.py`

> **LiteLLM note (spec §18):** verified surface — `litellm.completion(model="gemini/gemini-2.5-pro", messages=[...])`, text at `resp.choices[0].message.content`, usage at `resp.usage`. Cost via `litellm.completion_cost(completion_response=resp)` (wrapped defensively — cost tables move).

- [ ] **Step 1: Write the failing test**

`tests/unit/test_llm_provider.py`:
```python
from types import SimpleNamespace

from rag.providers.llm import LiteLLMProvider


def test_complete_extracts_text_and_usage(monkeypatch) -> None:
    def fake_completion(model: str, messages: list[dict], **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"))],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15),
        )

    monkeypatch.setattr("rag.providers.llm.litellm.completion", fake_completion)
    monkeypatch.setattr("rag.providers.llm.litellm.completion_cost", lambda **k: 0.0009)

    provider = LiteLLMProvider(model="gemini/gemini-2.5-pro", temperature=0.0)
    result = provider.complete([{"role": "user", "content": "hi"}])

    assert result.text == "hello world"
    assert result.usage["total_tokens"] == 15
    assert result.usage["cost_usd"] == 0.0009


def test_complete_survives_cost_failure(monkeypatch) -> None:
    def fake_completion(model: str, messages: list[dict], **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="x"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    def boom(**kwargs):
        raise RuntimeError("no cost table")

    monkeypatch.setattr("rag.providers.llm.litellm.completion", fake_completion)
    monkeypatch.setattr("rag.providers.llm.litellm.completion_cost", boom)

    result = LiteLLMProvider(model="m").complete([{"role": "user", "content": "hi"}])
    assert result.usage["cost_usd"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_llm_provider.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/providers/llm.py`:
```python
from __future__ import annotations

import litellm

from rag.models import Completion


class LiteLLMProvider:
    """Non-streaming completion via LiteLLM (P0a). Streaming is added in P0b."""

    def __init__(
        self, model: str, temperature: float = 0.0, max_tokens: int = 1024
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, messages: list[dict], **opts: object) -> Completion:
        resp = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
            "cost_usd": self._safe_cost(resp),
        }
        return Completion(text=text, usage=usage)

    @staticmethod
    def _safe_cost(resp: object) -> float | None:
        try:
            return float(litellm.completion_cost(completion_response=resp))
        except Exception:
            return None
```

`src/rag/generation/prompts/answer_v1.md`:
```markdown
You are a precise assistant answering questions strictly from the provided context.

Rules:
- Use ONLY the numbered context blocks below. Do not use outside knowledge.
- After each claim, cite the supporting block with its bracket marker, e.g. [1].
- Cite only blocks that actually support the claim. Never invent a marker.
- If the context does not contain the answer, reply exactly: "I don't have relevant context to answer that."

Context:
{context}

Question: {question}

Answer:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_llm_provider.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag/providers/llm.py src/rag/generation/prompts/answer_v1.md tests/unit/test_llm_provider.py
git commit -m "feat(providers): LiteLLM completion provider + versioned answer prompt"
```

---

## Task 17: StraightLineAnswerer

**Files:**
- Create: `src/rag/generation/answerer.py`
- Test: `tests/unit/test_answerer.py`

- [ ] **Step 1: Write the failing test (all fakes — no DB, no network)**

`tests/unit/test_answerer.py`:
```python
from rag.generation.answerer import StraightLineAnswerer
from rag.generation.assembler import TokenBudgetAssembler
from rag.models import Chunk, MetadataFilter, Provenance, ScoredChunk

from tests.unit.fakes import FakeLLMProvider


class _FakeRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, k: int, filt: MetadataFilter | None) -> list[ScoredChunk]:
        return [ScoredChunk(chunk=c, score=1.0, provenance=Provenance.dense) for c in self._chunks]


def _chunk(cid: str, text: str, ordinal: int) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="d1", source_uri="file:///x.pdf", text=text, ordinal=ordinal,
        page=ordinal + 1, char_start=0, char_end=len(text), token_count=2,
        chunker_name="fixed", chunker_version="1",
    )


def test_answerer_returns_cited_answer() -> None:
    retriever = _FakeRetriever([_chunk("a", "Nitrogen helps maize.", 0)])
    answerer = StraightLineAnswerer(
        retriever=retriever,
        assembler=TokenBudgetAssembler(),
        llm=FakeLLMProvider(reply="Nitrogen helps maize [1]."),
        token_budget=1000,
        retrieval_k=5,
    )
    answer = answerer.answer("does nitrogen help maize?", filt=None)

    assert answer.text == "Nitrogen helps maize [1]."
    assert [c.chunk_id for c in answer.citations] == ["a"]
    assert answer.trace_id != ""
    assert answer.usage["total_tokens"] == 15


def test_answerer_strips_hallucinated_citation() -> None:
    retriever = _FakeRetriever([_chunk("a", "Nitrogen helps maize.", 0)])
    answerer = StraightLineAnswerer(
        retriever=retriever,
        assembler=TokenBudgetAssembler(),
        llm=FakeLLMProvider(reply="Helps [1]. Made up [2]."),
        token_budget=1000,
        retrieval_k=5,
    )
    answer = answerer.answer("q", filt=None)
    assert "[2]" not in answer.text
    assert [c.marker for c in answer.citations] == ["[1]"]


def test_answerer_no_context_path() -> None:
    answerer = StraightLineAnswerer(
        retriever=_FakeRetriever([]),
        assembler=TokenBudgetAssembler(),
        llm=FakeLLMProvider(reply="should not be used"),
        token_budget=1000,
        retrieval_k=5,
    )
    answer = answerer.answer("q", filt=None)
    assert answer.text == "I don't have relevant context to answer that."
    assert answer.citations == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_answerer.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/generation/answerer.py`:
```python
from __future__ import annotations

import time
import uuid
from importlib import resources

import structlog

from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.citations import validate_and_build_citations
from rag.models import Answer, MetadataFilter, RetrievalScope
from rag.protocols import LLMProvider, Retriever

log = structlog.get_logger()

_NO_CONTEXT = "I don't have relevant context to answer that."


def _load_prompt() -> str:
    return (
        resources.files("rag.generation.prompts")
        .joinpath("answer_v1.md")
        .read_text(encoding="utf-8")
    )


class StraightLineAnswerer:
    """P0a read path: retrieve -> assemble -> generate -> validate citations (no loop)."""

    def __init__(
        self,
        retriever: Retriever,
        assembler: TokenBudgetAssembler,
        llm: LLMProvider,
        token_budget: int,
        retrieval_k: int,
    ) -> None:
        self._retriever = retriever
        self._assembler = assembler
        self._llm = llm
        self._token_budget = token_budget
        self._retrieval_k = retrieval_k
        self._prompt = _load_prompt()

    def answer(
        self,
        query: str,
        filt: MetadataFilter | None = None,
        scope: RetrievalScope = RetrievalScope.corpus_only,
    ) -> Answer:
        trace_id = str(uuid.uuid4())
        started = time.perf_counter()
        log = structlog.get_logger().bind(trace_id=trace_id)

        scored = self._retriever.retrieve(query, self._retrieval_k, filt)
        if not scored:
            log.info("no_context")
            return Answer(text=_NO_CONTEXT, citations=[], trace_id=trace_id, retrieval_scope=scope)

        context = self._assembler.assemble(
            query, [s.chunk for s in scored], self._token_budget
        )
        messages = [
            {
                "role": "user",
                "content": self._prompt.format(context=context.text, question=query),
            }
        ]
        completion = self._llm.complete(messages)
        text, citations = validate_and_build_citations(completion.text, context)

        usage = {
            **completion.usage,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        log.info("answered", citations=len(citations), latency_ms=usage["latency_ms"])
        return Answer(
            text=text,
            citations=citations,
            usage=usage,
            trace_id=trace_id,
            retrieval_scope=scope,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_answerer.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag/generation/answerer.py tests/unit/test_answerer.py
git commit -m "feat(generation): straight-line answerer with citation validation"
```

---

## Task 18: FastAPI app (/healthz, /query JSON)

**Files:**
- Create: `src/rag/api/__init__.py`, `src/rag/api/schemas.py`, `src/rag/api/routes.py`, `src/rag/api/app.py`
- Test: `tests/integration/test_api.py` (uses TestClient + fakes — no Docker needed, but marked integration for the wiring)

- [ ] **Step 1: Write the failing test**

`tests/integration/test_api.py`:
```python
import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.generation.answerer import StraightLineAnswerer
from rag.generation.assembler import TokenBudgetAssembler
from rag.models import Chunk, MetadataFilter, Provenance, ScoredChunk

from tests.unit.fakes import FakeLLMProvider

pytestmark = pytest.mark.integration


class _FakeRetriever:
    def retrieve(self, query: str, k: int, filt: MetadataFilter | None) -> list[ScoredChunk]:
        chunk = Chunk(
            chunk_id="a", doc_id="d1", source_uri="file:///x.pdf",
            text="Nitrogen helps maize.", ordinal=0, page=1, char_start=0, char_end=21,
            token_count=4, chunker_name="fixed", chunker_version="1",
        )
        return [ScoredChunk(chunk=chunk, score=1.0, provenance=Provenance.dense)]


def _client() -> TestClient:
    answerer = StraightLineAnswerer(
        retriever=_FakeRetriever(),
        assembler=TokenBudgetAssembler(),
        llm=FakeLLMProvider(reply="Nitrogen helps maize [1]."),
        token_budget=1000,
        retrieval_k=5,
    )
    return TestClient(create_app(answerer=answerer))


def test_healthz() -> None:
    assert _client().get("/healthz").json() == {"status": "ok"}


def test_query_returns_cited_answer() -> None:
    resp = _client().post("/query", json={"query": "does nitrogen help maize?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Nitrogen helps maize [1]."
    assert body["citations"][0]["chunk_id"] == "a"
    assert body["citations"][0]["page"] == 1
    assert body["trace_id"]


def test_query_validates_empty_query() -> None:
    resp = _client().post("/query", json={"query": "   "})
    assert resp.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_api.py -m integration -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

`src/rag/api/__init__.py`: empty file.

`src/rag/api/schemas.py`:
```python
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from rag.models import Answer, MetadataFilter


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    filter: MetadataFilter | None = None

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class CitationOut(BaseModel):
    marker: str
    doc_id: str
    chunk_id: str
    source_uri: str
    source_kind: str
    page: int
    char_start: int
    char_end: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    usage: dict
    trace_id: str
    retrieval_scope: str

    @classmethod
    def from_answer(cls, answer: Answer) -> "QueryResponse":
        return cls(
            answer=answer.text,
            citations=[CitationOut(**c.model_dump(mode="json")) for c in answer.citations],
            usage=answer.usage,
            trace_id=answer.trace_id,
            retrieval_scope=answer.retrieval_scope.value,
        )
```

`src/rag/api/routes.py`:
```python
from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from rag.api.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    answerer = request.app.state.answerer
    answer = await run_in_threadpool(
        answerer.answer, body.query, body.filter
    )
    return QueryResponse.from_answer(answer)
```

`src/rag/api/app.py`:
```python
from __future__ import annotations

from fastapi import FastAPI

from rag.api.routes import router
from rag.config import get_settings
from rag.db import get_engine
from rag.generation.answerer import StraightLineAnswerer
from rag.generation.assembler import TokenBudgetAssembler
from rag.providers.embeddings import LiteLLMEmbeddingProvider
from rag.providers.llm import LiteLLMProvider
from rag.retrieval.dense import DenseRetriever


def _build_answerer() -> StraightLineAnswerer:
    settings = get_settings()
    engine = get_engine(settings.database_url)
    embedder = LiteLLMEmbeddingProvider(
        model=settings.embedding_model, dim=settings.embedding_dim
    )
    retriever = DenseRetriever(engine=engine, embedder=embedder)
    return StraightLineAnswerer(
        retriever=retriever,
        assembler=TokenBudgetAssembler(),
        llm=LiteLLMProvider(
            model=settings.generation_model,
            temperature=settings.generation_temperature,
            max_tokens=settings.generation_max_tokens,
        ),
        token_budget=settings.context_token_budget,
        retrieval_k=settings.retrieval_k,
    )


def create_app(answerer: StraightLineAnswerer | None = None) -> FastAPI:
    app = FastAPI(title="Production RAG — P0a")
    app.state.answerer = answerer or _build_answerer()
    app.include_router(router)
    return app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_api.py -m integration -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full unit suite + types (no regressions)**

Run: `uv run pytest -m "not integration and not live"` → Expected: all pass.
Run: `uv run mypy src` → Expected: Success.

- [ ] **Step 6: Commit**

```bash
git add src/rag/api tests/integration/test_api.py
git commit -m "feat(api): FastAPI app with /healthz and JSON /query"
```

---

## Task 19: Docker, Compose, Makefile, README quickstart (the "one-command up" deliverable)

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `Makefile`, `.env.example`, `README.md`
- Modify: `pyproject.toml` (add the `rag-api`/`rag-ingest` scripts)
- Test: manual smoke (documented), plus a CLI entrypoint check

This task fulfils P0a's **Done when:** "one-command up + ingest + a cited answer over dense search."

- [ ] **Step 1: Add console-script entrypoints to `pyproject.toml`**

Add under `[project]`:
```toml
[project.scripts]
rag-ingest = "rag.ingestion.cli:main"
```
Add a uvicorn-friendly module attr by ensuring `rag.api.app:create_app` works as a factory (already does).

- [ ] **Step 2: Write `.env.example`**

`.env.example`:
```bash
# Copy to .env and fill in. .env is gitignored; .env.example is committed.
GEMINI_API_KEY=your-google-ai-studio-key

# Inside docker-compose the host is the service name "postgres".
DATABASE_URL=postgresql+psycopg://rag:rag@postgres:5432/rag

# Model selection (Gemini defaults; swap by changing the string)
GENERATION_MODEL=gemini/gemini-2.5-pro
GRADER_MODEL=gemini/gemini-2.5-flash
EMBEDDING_MODEL=gemini/text-embedding-004
EMBEDDING_DIM=768
```

- [ ] **Step 3: Write the `Dockerfile` (multi-stage, non-root)**

`Dockerfile`:
```dockerfile
# ---- builder ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic
RUN uv sync --frozen --no-dev

# ---- runtime ----
FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH="/app/src"
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "rag.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Write `docker-compose.yml`**

`docker-compose.yml`:
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: rag
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag"]
      interval: 5s
      timeout: 3s
      retries: 10

  api:
    build: .
    env_file: .env
    environment:
      DATABASE_URL: postgresql+psycopg://rag:rag@postgres:5432/rag
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
    command: >
      sh -c "alembic upgrade head &&
             uvicorn rag.api.app:create_app --factory --host 0.0.0.0 --port 8000"

volumes:
  pgdata:
```

- [ ] **Step 5: Write the `Makefile`**

`Makefile`:
```makefile
.PHONY: up down ingest query test test-int migrate fmt lint type

up:            ## Build and start api + postgres
	docker compose up --build -d

down:
	docker compose down

migrate:       ## Run Alembic migrations against DATABASE_URL
	uv run alembic upgrade head

ingest:        ## Ingest PDFs: make ingest CORPUS=./data/raw
	docker compose exec api rag-ingest $(CORPUS)

query:         ## Ask a question: make query Q="does nitrogen help maize?"
	curl -s -X POST localhost:8000/query -H 'content-type: application/json' \
	  -d '{"query": "$(Q)"}' | python -m json.tool

test:          ## Unit tests (no docker)
	uv run pytest -m "not integration and not live"

test-int:      ## Integration tests (needs docker)
	uv run pytest -m integration

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

type:
	uv run mypy src
```
(Windows without `make`: run the underlying commands directly, e.g. `uv run pytest -m "not integration and not live"`.)

- [ ] **Step 6: Write the `README.md` quickstart**

`README.md`:
```markdown
# Production-Grade RAG System

Hybrid-RAG over a single Postgres + pgvector store, every model call through LiteLLM
(Gemini default). This branch ships **P0a — the walking skeleton**: PDF ingest →
dense retrieval → cited, grounded answers behind a one-command stack.

> Roadmap: P0b adds hybrid + rerank + streaming + observability + eval; see
> [the design spec](docs/superpowers/specs/2026-06-02-production-rag-system-design.md).

## Quickstart

```bash
cp .env.example .env          # add your GEMINI_API_KEY
make up                       # builds api + postgres, runs migrations
make ingest CORPUS=./data/raw # ingest your PDFs
make query Q="does nitrogen help maize?"
```

Response is JSON: a grounded `answer`, `citations` (doc/chunk/page/char span),
`usage` (tokens, cost, latency), and a `trace_id`.

## How it works (P0a)

`PDF → parse → clean → fixed-token chunk → embed (text-embedding-004) → Postgres+pgvector`
on the write side; `embed query → pgvector cosine KNN → assemble (token budget,
numbered) → generate (gemini-2.5-pro) → validate citations` on the read side. A
generated citation that does not map to an assembled chunk is stripped — the answer
never ships a fabricated source.

## Development

```bash
uv sync
make test        # unit (fast, no docker)
make test-int    # integration (testcontainers pgvector — needs docker)
make lint type
```
```

- [ ] **Step 7: Write a CLI entrypoint smoke test**

`tests/unit/test_cli_entrypoint.py`:
```python
def test_ingest_cli_is_importable_and_has_main() -> None:
    from rag.ingestion import cli

    assert callable(cli.main)
```

Run: `uv run pytest tests/unit/test_cli_entrypoint.py -v`
Expected: 1 passed.

- [ ] **Step 8: End-to-end smoke (manual, documented in the PR)**

```bash
cp .env.example .env   # set a real GEMINI_API_KEY
docker compose up --build -d
# drop a small text PDF into ./data/raw first
docker compose exec api rag-ingest /app/data/raw   # or mount the folder
curl -s -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"query":"does nitrogen help maize?"}' | python -m json.tool
docker compose down
```
Expected: JSON answer with at least one citation carrying a real `page` and `char_start/char_end`. Capture this output for the PR description — it is the P0a "Done when" evidence.

- [ ] **Step 9: Final full verification**

```bash
uv run ruff check .
uv run mypy src
uv run pytest -m "not integration and not live"
uv run pytest -m integration   # docker required
```
Expected: all green.

- [ ] **Step 10: Commit**

```bash
git add Dockerfile docker-compose.yml Makefile .env.example README.md pyproject.toml tests/unit/test_cli_entrypoint.py
git commit -m "build: dockerized one-command stack, Makefile, README quickstart (P0a done)"
```

---

## Self-Review (run before handing off)

**1. Spec coverage (P0a slice of spec §7):**

| P0a requirement | Task(s) |
|---|---|
| text-PDF ingest (PyMuPDF, no OCR) | 5, 6 |
| clean / normalize | 7 |
| chunk (strategy) | 8 |
| embed (LiteLLM, Gemini text-embedding-004 768-d) | 9 |
| Postgres+pgvector upsert, full schema (versioning cols, content_hash, deleted_at) | 10, 11 |
| Alembic migrations | 10 |
| dense vector search | 13 |
| context assembly with span-level citations | 14 |
| citation-validation rule from day one | 15, 17 |
| non-streaming cited answer | 16, 17 |
| FastAPI /query (JSON) | 18 |
| docker-compose (api + postgres) | 19 |
| one-command up + ingest + cited answer ("Done when") | 19 |
| structlog with trace_id (eng standards §12) | 17 (bind trace_id), 12 |
| failure mode: no-context path (§16) | 17 |
| failure mode: per-doc isolation on ingest (§16) | 12 |

No P0a requirement is left without a task.

**2. Deliberately deferred (NOT in P0a — do not add):** FTS/lexical/RRF hybrid, Cohere rerank, SSE streaming, Langfuse/Prometheus/Grafana, eval harness/golden set, OCR/Docling/tables, corrective loop/grader/router, the LiteLLM Router + Langfuse callback. These belong to P0b/P0c/Phase-1/Phase-2 per spec §7 and get their own plans.

**3. Type consistency check (names used identically across tasks):**
- `Chunk` fields (`source_uri`, `char_start`, `char_end`, `page`, `ordinal`, `embedding_model`, `embedding_dim`) — defined Task 3, used identically in Tasks 8, 11, 13, 14, 15.
- `AssembledContext.chunks` ordering → citation marker `[i+1]` — produced Task 14, consumed Task 15. Consistent.
- `Completion.usage` keys (`prompt_tokens`/`completion_tokens`/`total_tokens`/`cost_usd`) — produced Task 16, extended with `latency_ms` Task 17, surfaced Task 18. Consistent.
- `EmbeddingProvider` (`.model`, `.dim`, `.embed`) — Protocol Task 3, real impl Task 9, fake Task 9, consumed Tasks 12, 13. Consistent.
- `StraightLineAnswerer.answer(query, filt, scope)` — defined Task 17, called Task 18 via `run_in_threadpool(answerer.answer, body.query, body.filter)`. Signature matches (scope defaulted).
- `PgChunkRepository` (`upsert`/`soft_delete`/`get_watermark`/`set_watermark`) — Task 11, used Tasks 12, 13(seed). Consistent.
- DB Core tables (`documents`, `chunks`, `watermarks`/`source_watermarks`) — defined Task 10, used Tasks 11, 13. Table object `watermarks` maps to SQL table `source_watermarks`; consistent within code.

**4. Placeholder scan:** every code step contains complete, runnable code; no "TBD"/"similar to Task N"/"add error handling" placeholders. Live-provider verification notes point to exact LiteLLM doc URLs (intentional, per spec §18 — not placeholders).

**Known engineering choices worth flagging to the reviewer:**
- **Sync-first.** P0a is synchronous; the async route wraps work in `run_in_threadpool`. P0b introduces real async streaming. Rationale: a walking skeleton should not pay async-DB complexity before it streams.
- **Provider fakes over respx for LiteLLM.** Unit/integration tests inject fakes at the Protocol seam rather than mocking LiteLLM's provider-specific HTTP (brittle, URL-coupled). respx is reserved for genuinely HTTP-shaped adapters (web/API sources, Tavily) in later phases. A `live`-marked test (skipped by default) exercises the real Gemini calls when `GEMINI_API_KEY` is set. Net effect matches spec §13's "CI spends nothing."
- **Embedding dim hardcoded to 768 in the migration.** A model swap that changes dim is a new migration + the `reindex` CLI (Phase 3), not a P0a concern.
- **Deterministic IDs for idempotency.** `doc_id = uuid5(NAMESPACE_URL, uri)` and `chunk_id = uuid5(doc_id:ordinal)`, so re-ingesting the same source updates rows in place rather than duplicating. `content_hash` is indexed but **not unique** in P0a (the document hash flows from the pipeline via `metadata["document_content_hash"]`); cross-file content dedup and hash-based change-detection are Phase-3 concerns.

---

## Execution Handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration (superpowers:subagent-driven-development).
2. **Inline Execution** — execute tasks in this session with checkpoints (superpowers:executing-plans).
