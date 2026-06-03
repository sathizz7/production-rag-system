# P0b — Hybrid · Rerank · Streaming · Observability · Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the P0a walking skeleton into a portfolio-grade RAG core: hybrid retrieval (Postgres FTS + dense, fused by RRF), Cohere cross-encoder rerank, SSE token streaming, Langfuse + Prometheus/Grafana observability, and a stratified golden-set eval harness that reports retrieval and answer metrics separately with bootstrap confidence intervals — all behind the same Protocol seams, so nothing in P0a's signatures changes.

**Architecture:** Every P0b feature *wraps* a P0a component, never replaces it (spec §6). `DenseRetriever` is composed inside a new `HybridRetriever`; the straight-line read path gains an optional rerank step and a streaming sibling; the FastAPI app keeps its JSON `/query` and adds an SSE `/query/stream`; observability and eval are new cross-cutting modules that read the existing store and providers. All model calls still route through **LiteLLM** with **Gemini** defaults; rerank routes through `litellm.rerank()` (Cohere).

**Tech Stack:** Python 3.12 + uv · FastAPI + sse-starlette · LiteLLM (Gemini generation/embedding/judge, Cohere rerank) · Postgres 18 + pgvector (HNSW) + native FTS (tsvector/GIN) · SQLAlchemy 2 Core + psycopg3 · Alembic · prometheus-client + Prometheus/Grafana · Langfuse · numpy (bootstrap CIs) · PyYAML (golden set) · structlog · pytest · ruff + mypy(strict).

**Source spec:** [docs/superpowers/specs/2026-06-02-production-rag-system-design.md](../specs/2026-06-02-production-rag-system-design.md) — this plan implements **P0b only** (spec §7, the second slice of Phase 0). P0c (parser hardening) and Phase 1 (eval-in-CI) get their own plans. Phase 1 *promotes* this plan's `rag-eval` into a CI gate; P0b only needs it runnable locally.

**Predecessor:** [docs/superpowers/plans/2026-06-02-p0a-walking-skeleton.md](2026-06-02-p0a-walking-skeleton.md). P0a is merged to `main`. Branch P0b off `main`.

---

## Conventions (read once, applies to every task)

**The TDD loop** — every task follows: write a failing test → run it, see it fail → write minimal code → run it, see it pass → commit. Do not write implementation before its test. The `run` and `commit` steps are written tersely below to keep the plan readable; never skip them.

**Running things (Windows/PowerShell, no `make` — raw `uv` commands are authoritative):**
- Unit tests (fast, no DB, no network): `uv run pytest -m "not integration and not live"` — the default `uv run pytest`.
- Integration tests (real local Postgres+pgvector via `TEST_DATABASE_URL`): `uv run pytest -m integration`.
- Live tests/eval (real Gemini/Cohere keys): `uv run pytest -m live` and `uv run rag-eval` — skipped in CI and by default; cost real money.
- Lint/format: `uv run ruff check .` · `uv run ruff format .` — Types: `uv run mypy src`.

**Markers** (`pyproject.toml`): `integration` (local Postgres), `live` (real provider keys). No new marker is added; eval *machinery* is unit-tested against fakes, and the real eval run is the manual `uv run rag-eval`.

**Commits:** Conventional Commits. Commit at the end of every task; `uv run pytest` (unit) and `uv run ruff check .` and `uv run mypy src` must be green at every commit. Author identity stays **Sathish R / sathish01072000@gmail.com** (already the git default).

**Database:** local Postgres 18 + pgvector on `localhost:5432`. App DB from `DATABASE_URL`, integration tests from `TEST_DATABASE_URL` (e.g. `.../rag_test`). The session-scoped `migrated_engine` fixture (in `tests/conftest.py`) drops+recreates the `public` schema on the **test** DB only, then runs all migrations; `clean_db` truncates between tests. New migrations are picked up automatically by that fixture — no fixture changes needed.

**Secrets:** never commit `.env`. New keys this slice — `COHERE_API_KEY` (rerank), `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` (tracing) — are gitignored in `.env` and mirrored as placeholders in `.env.example`. Every provider key reaches LiteLLM through `apply_provider_env()` (Task 1), never by hardcoding.

**Observability runs without Docker.** The Prometheus `/metrics` endpoint and Langfuse callback work against the plain local process. Prometheus + Grafana themselves ship as an *optional* `docker-compose.observability.yml` for reviewers; the author is not required to run Docker to satisfy P0b's "Done when".

---

## Design notes locked before coding (read before Track C and Track E)

**Streaming + the citation invariant.** P0a's invariant — *an answer may cite only chunks in the assembled context; invalid markers are stripped* (`validate_and_build_citations`) — needs the **full** answer text to run. Streaming can't un-send a token. P0b resolves this honestly: the streaming answerer emits raw `token` events for UX as deltas arrive, accumulates the full text, then runs the **same** validator and emits a terminal `done` event carrying the cleaned text + validated citations. Clients render live tokens, then replace with `done.answer` and attach `done.citations`. The invariant is preserved (citations are still validated against assembled context); only the *delivery* is incremental. A mid-stream provider error emits a terminal `error` event after flushing whatever was streamed (spec §16).

**Usage/cost under streaming.** LiteLLM streaming yields content deltas; authoritative token/cost accounting is captured by the **Langfuse callback** (Task 12), not reconstructed in the answerer. The `done` event's `usage` carries `latency_ms` and a `completion_tokens_est` (tiktoken count of the streamed text) — labeled "est" so it is never mistaken for billing truth. The non-streaming JSON `/query` keeps exact usage from `litellm.completion` (unchanged P0a path via `LiteLLMProvider.complete`).

**Eval corpus is frozen and self-contained.** To keep `rag-eval` deterministic and reproducible with zero external downloads, the golden set runs against a small committed agronomy mini-corpus authored as Markdown under `eval/corpus/*.md` (public, factual, non-copyrighted content). The runner renders each `.md` to a one-page PDF at runtime (PyMuPDF, the same trick `tests/conftest.py` uses) and ingests it through the **real** pipeline (parse→clean→chunk→embed→upsert) into a dedicated DB, so eval exercises production code, not a shortcut. Curating the full 30–50-item stratified set is a data task seeded here with concrete starter items and an exact schema.

**Judge choice.** P0b ships a **custom** LLM judge (faithfulness + answer-relevance) called through the existing `LLMProvider.complete` seam on the cheap model (`gemini-2.5-flash`, temperature 0), rather than pulling in RAGAS. Rationale: keeps the thin-explicit-core philosophy (spec §4), pins determinism through our own provider, and avoids a heavy framework's parallel LLM orchestration. RAGAS remains a documented Phase-1 cross-check option (spec §8), not a P0b dependency.

---

## File map (what each new/changed file owns)

```
src/rag/
  config.py                      [MODIFY] new Settings fields + apply_provider_env extension
  models.py                      [MODIFY] AnswerEvent union (TokenEvent/DoneEvent/ErrorEvent)
  protocols.py                   [MODIFY] add Reranker; add LLMProvider.stream
  db.py                          [MODIFY] chunks.text_search (generated tsvector) column
  retrieval/
    lexical.py                   [CREATE] Postgres FTS retriever (english-frozen; provenance=lexical)
    fusion.py                    [CREATE] reciprocal_rank_fusion (pure)
    hybrid.py                    [CREATE] HybridRetriever (over-fetch candidate_k, RRF; +per-stage timing T11)
    reranked.py                  [CREATE] RerankedRetriever decorator (over-fetch + rerank; +timing T11)
  providers/
    llm.py                       [MODIFY] LiteLLMProvider.stream()
    rerank.py                    [CREATE] CohereReranker via litellm.rerank()
  generation/
    answerer.py                  [UNCHANGED by rerank] StraightLineAnswerer stays P0a (rerank is a retriever now)
    streaming.py                 [CREATE] StreamingAnswerer (answer_stream + answer; retriever-agnostic)
  observability/
    __init__.py                  [CREATE]
    metrics.py                   [CREATE] prometheus counters/histograms + middleware + render
    tracing.py                   [CREATE] configure_observability (Langfuse via LiteLLM callback)
  eval/
    __init__.py                  [CREATE]
    metrics.py                   [CREATE] hit@k, reciprocal_rank, ndcg@k (pure)
    stats.py                     [CREATE] bootstrap_ci (numpy, seeded)
    judge.py                     [CREATE] FaithfulnessJudge (custom, via LLMProvider)
    prompts/judge_v1.md          [CREATE] judge prompt
    golden.py                    [CREATE] golden-set loader + GoldenItem model
    scorecard.py                 [CREATE] render separated retrieval/answer scorecards
    corpus.py                    [CREATE] md→pdf rendering for the frozen mini-corpus
    runner.py                    [CREATE] orchestration + rag-eval CLI entry
  api/
    schemas.py                   [MODIFY] (none required; SSE serialises AnswerEvent directly)
    routes.py                    [MODIFY] add POST /query/stream (SSE) + GET /metrics
    app.py                       [MODIFY] Hybrid (+RerankedRetriever if enabled) → StreamingAnswerer; obs init

alembic/versions/0002_fts_tsvector.py   [CREATE] tsvector generated column + GIN index
eval/corpus/*.md                        [CREATE] frozen agronomy mini-corpus (Markdown)
eval/golden_set.yaml                    [CREATE] smoke-eval golden set (7 items; 30–50 = Phase 1)
monitoring/prometheus.yml               [CREATE] scrape config
monitoring/grafana/dashboard.json       [CREATE] starter dashboard
docker-compose.observability.yml        [CREATE] optional prometheus+grafana+langfuse
tests/unit/fakes.py                     [MODIFY] FakeStreamingLLM, FakeReranker, FakeRetriever
tests/unit/…                            [CREATE] per-task unit tests
tests/integration/…                     [CREATE] per-task integration tests
```

---

## Track A — Hybrid retrieval (Postgres FTS + RRF)

### Task 1: Dependencies, Settings, and provider-env wiring

**Files:**
- Modify: `pyproject.toml` (deps + mypy overrides)
- Modify: `src/rag/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Add the failing config test.** Append to `tests/unit/test_config.py`:

```python
def test_settings_has_p0b_fields() -> None:
    from rag.config import Settings

    s = Settings(_env_file=None)
    assert s.rrf_k == 60
    assert s.rerank_model == "cohere/rerank-english-v3.0"
    assert s.rerank_enabled is False          # off until a Cohere key is present
    assert s.candidate_k == 30                 # over-fetch pool for fusion + rerank
    assert s.eval_database_url == ""           # isolated eval DB; empty until configured
    assert s.langfuse_host == "https://cloud.langfuse.com"


def test_apply_provider_env_exports_optional_keys(monkeypatch) -> None:
    import os

    from rag.config import Settings, apply_provider_env

    for var in ("GEMINI_API_KEY", "COHERE_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(
        _env_file=None,
        gemini_api_key="g",
        cohere_api_key="c",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
    )
    apply_provider_env(s)
    assert os.environ["GEMINI_API_KEY"] == "g"
    assert os.environ["COHERE_API_KEY"] == "c"
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk"
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest tests/unit/test_config.py -q` → FAIL (unknown fields / keys not exported).

- [ ] **Step 3: Add dependencies.** In `pyproject.toml`, add to `[project].dependencies`:

```toml
    "sse-starlette>=2.1.0",
    "prometheus-client>=0.21.0",
    "langfuse>=2.50.0",
    "numpy>=2.1.0",
    "pyyaml>=6.0.2",
```

Extend the existing mypy override `module` list (the `[[tool.mypy.overrides]]` block) to also silence untyped third parties:

```toml
module = ["pymupdf", "fitz", "pgvector.*", "testcontainers.*", "litellm", "litellm.*", "yaml", "sse_starlette.*", "langfuse.*"]
```

Then sync: `uv sync`.

- [ ] **Step 4: Extend Settings + apply_provider_env.** In `src/rag/config.py`, add fields to `Settings` (after `context_token_budget`):

```python
    # Hybrid retrieval
    rrf_k: int = 60                       # RRF damping constant
    candidate_k: int = 30                 # over-fetch per leg before fusion; rerank pool size
    # NOTE: Postgres FTS language is frozen to 'english' in P0b. It MUST match the
    # STORED generated tsvector column (migration 0002: to_tsvector('english', ...)).
    # A runtime language knob would silently desync the query language from the column,
    # so there is intentionally none — multi-language is a future migration, not config.

    # Rerank (Cohere via LiteLLM); disabled until a key is present
    cohere_api_key: str = ""
    rerank_model: str = "cohere/rerank-english-v3.0"
    rerank_enabled: bool = False

    # Eval (dedicated DB so the harness NEVER writes the app/dev database)
    eval_database_url: str = ""           # e.g. .../rag_eval ; empty -> rag-eval errors out

    # Observability (all optional; absent keys → features no-op)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
```

Replace the body of `apply_provider_env` to export the optional keys too:

```python
def apply_provider_env(settings: Settings | None = None) -> None:
    """Export provider API keys from Settings (.env) into ``os.environ`` for LiteLLM.

    LiteLLM and the Langfuse callback read keys from the process environment, but
    pydantic-settings only loads ``.env`` into the Settings object. Call this at
    app/CLI startup so plain ``uv run`` works without ``--env-file``. Pre-existing
    environment values win (``setdefault``).
    """
    settings = settings or get_settings()
    pairs = {
        "GEMINI_API_KEY": settings.gemini_api_key,
        "COHERE_API_KEY": settings.cohere_api_key,
        "LANGFUSE_PUBLIC_KEY": settings.langfuse_public_key,
        "LANGFUSE_SECRET_KEY": settings.langfuse_secret_key,
        "LANGFUSE_HOST": settings.langfuse_host,
    }
    for key, value in pairs.items():
        if value:
            os.environ.setdefault(key, value)
```

- [ ] **Step 5: Run + commit.** `uv run pytest tests/unit/test_config.py -q` → PASS; `uv run mypy src` → clean. Commit:

```bash
git checkout -b feat/p0b-hybrid-rerank-streaming-obs-eval
git add pyproject.toml uv.lock src/rag/config.py tests/unit/test_config.py
git commit -m "build(p0b): add hybrid/rerank/obs/eval deps + Settings + provider-env keys"
```

---

### Task 2: Migration 0002 — FTS `tsvector` generated column + GIN index

**Files:**
- Create: `alembic/versions/0002_fts_tsvector.py`
- Modify: `src/rag/db.py`
- Test: `tests/integration/test_migrations.py`

- [ ] **Step 1: Add the failing integration test.** Append to `tests/integration/test_migrations.py`:

```python
def test_chunks_has_fts_column_and_index(migrated_engine) -> None:
    from sqlalchemy import text

    with migrated_engine.connect() as conn:
        col = conn.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name='chunks' AND column_name='text_search'"
            )
        ).scalar()
        idx = conn.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname='ix_chunks_text_search'")
        ).scalar()
    assert col == "tsvector"
    assert idx is not None and "gin" in idx.lower()
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest -m integration tests/integration/test_migrations.py -q` → FAIL (no `text_search` column).

- [ ] **Step 3: Write the migration.** Create `alembic/versions/0002_fts_tsvector.py`:

```python
"""fts: generated tsvector column on chunks + GIN index

Revision ID: 0002
Revises: 0001
"""
from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # STORED generated column keeps the tsvector in lockstep with text on every
    # insert/update — the repository never has to compute or pass it.
    op.execute(
        "ALTER TABLE chunks ADD COLUMN text_search tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_text_search ON chunks USING gin (text_search)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_text_search")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS text_search")
```

- [ ] **Step 4: Mirror the column in the Core table** so the lexical retriever can reference `chunks_t.c.text_search`. In `src/rag/db.py`, add imports and the column. Add to the existing `sqlalchemy` import group `Computed`, and a new import:

```python
from sqlalchemy import Computed  # add to the existing sqlalchemy import block
from sqlalchemy.dialects.postgresql import TSVECTOR
```

Then add this column to the `chunks` table definition, right after the `embedding` column:

```python
    Column(
        "text_search",
        TSVECTOR,
        Computed("to_tsvector('english', text)", persisted=True),
        nullable=True,
    ),
```

- [ ] **Step 5: Run + commit.** `uv run pytest -m integration tests/integration/test_migrations.py -q` → PASS; `uv run mypy src` → clean. Commit:

```bash
git add alembic/versions/0002_fts_tsvector.py src/rag/db.py tests/integration/test_migrations.py
git commit -m "feat(db): add generated FTS tsvector column + GIN index (migration 0002)"
```

---

### Task 3: `LexicalRetriever` — Postgres full-text search

**Files:**
- Create: `src/rag/retrieval/lexical.py`
- Test: `tests/integration/test_lexical_retrieval.py`

- [ ] **Step 1: Write the failing integration test.** Create `tests/integration/test_lexical_retrieval.py`:

```python
import pytest
from sqlalchemy import Engine

from rag.models import Provenance
from rag.retrieval.lexical import LexicalRetriever
from tests.integration.test_dense_retrieval import _chunk, _vec

pytestmark = pytest.mark.integration


def _seed(engine: Engine) -> None:
    from rag.ingestion.repository import PgChunkRepository

    repo = PgChunkRepository(engine)
    chunks = [
        _chunk("c0", "Nitrogen fertilizer increases maize yield substantially."),
        _chunk("c1", "Drought stress reduces wheat grain filling."),
        _chunk("c2", "Phosphorus supports early root development in maize."),
    ]
    repo.upsert(chunks, [_vec(1.0), _vec(1.0), _vec(1.0)])


def test_lexical_matches_query_terms(clean_db: Engine) -> None:
    _seed(clean_db)
    retriever = LexicalRetriever(engine=clean_db)
    results = retriever.retrieve("nitrogen maize yield", k=5, filt=None)

    texts = [r.chunk.text for r in results]
    assert any("Nitrogen" in t for t in texts)
    assert results[0].provenance == Provenance.lexical
    assert all("Drought" not in t for t in texts)  # no shared lexical terms


def test_lexical_returns_empty_on_no_match(clean_db: Engine) -> None:
    _seed(clean_db)
    retriever = LexicalRetriever(engine=clean_db)
    assert retriever.retrieve("zzzqxnonsense", k=5, filt=None) == []
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest -m integration tests/integration/test_lexical_retrieval.py -q` → FAIL (no module `rag.retrieval.lexical`).

- [ ] **Step 3: Implement the retriever.** Create `src/rag/retrieval/lexical.py`:

```python
from __future__ import annotations

from sqlalchemy import Engine, func, select

from rag.db import chunks as chunks_t
from rag.models import Chunk, MetadataFilter, Provenance, ScoredChunk


class LexicalRetriever:
    """Postgres full-text search over live chunks via the generated tsvector.

    Uses ``websearch_to_tsquery`` (tolerant of free-form user input) and ranks
    with ``ts_rank_cd``. The language is frozen to ``'english'`` to match the STORED
    generated tsvector column (migration 0002) — querying with a different config
    than the column was built with silently returns wrong matches. NOTE: Postgres
    FTS rank is tf-idf-ish, not BM25 — the README documents this; hybrid fusion
    (RRF) ignores absolute scores anyway.
    """

    _LANGUAGE = "english"  # MUST equal the column's to_tsvector config (migration 0002)

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def retrieve(
        self, query: str, k: int, filt: MetadataFilter | None
    ) -> list[ScoredChunk]:
        tsquery = func.websearch_to_tsquery(self._LANGUAGE, query)
        rank = func.ts_rank_cd(chunks_t.c.text_search, tsquery)
        stmt = (
            select(chunks_t, rank.label("rank"))
            .where(chunks_t.c.deleted_at.is_(None))
            .where(chunks_t.c.text_search.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(k)
        )
        if filt is not None:
            for key, value in filt.as_pairs():
                stmt = stmt.where(chunks_t.c.metadata[key].as_string() == value)

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
        return ScoredChunk(chunk=chunk, score=float(row["rank"]), provenance=Provenance.lexical)
```

- [ ] **Step 4: Run, see it pass.** `uv run pytest -m integration tests/integration/test_lexical_retrieval.py -q` → PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/rag/retrieval/lexical.py tests/integration/test_lexical_retrieval.py
git commit -m "feat(retrieval): Postgres FTS LexicalRetriever (ts_rank_cd over generated tsvector)"
```

---

### Task 4: Reciprocal Rank Fusion (pure function)

**Files:**
- Create: `src/rag/retrieval/fusion.py`
- Test: `tests/unit/test_fusion.py`

- [ ] **Step 1: Write the failing unit test.** Create `tests/unit/test_fusion.py`:

```python
from rag.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_rewards_items_ranked_high_in_both_lists() -> None:
    dense = ["a", "b", "c"]
    lexical = ["b", "a", "d"]
    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    ids = [item_id for item_id, _ in fused]
    # 'a' (ranks 0,1) and 'b' (ranks 1,0) outrank singletons 'c' and 'd'
    assert set(ids[:2]) == {"a", "b"}
    assert set(ids[2:]) == {"c", "d"}


def test_rrf_scores_are_descending_and_use_k() -> None:
    fused = reciprocal_rank_fusion([["x", "y"]], k=60)
    assert fused[0] == ("x", 1.0 / 61)
    assert fused[1] == ("y", 1.0 / 62)
    assert fused[0][1] > fused[1][1]


def test_rrf_handles_empty_input() -> None:
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest tests/unit/test_fusion.py -q` → FAIL.

- [ ] **Step 3: Implement.** Create `src/rag/retrieval/fusion.py`:

```python
from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked id-lists by Reciprocal Rank Fusion.

    Each list is in best-first order. An id at 0-based rank ``r`` in a list
    contributes ``1 / (k + r + 1)``; contributions sum across lists. Returns
    ``(id, score)`` pairs sorted by score descending. Rank-based, so it needs no
    score normalisation between dense and lexical sources (spec §4).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
```

- [ ] **Step 4: Run, see it pass.** `uv run pytest tests/unit/test_fusion.py -q` → PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/rag/retrieval/fusion.py tests/unit/test_fusion.py
git commit -m "feat(retrieval): reciprocal rank fusion (pure, rank-based)"
```

---

### Task 5: `HybridRetriever` — compose dense + lexical, fuse by RRF

**Files:**
- Create: `src/rag/retrieval/hybrid.py`
- Modify: `tests/unit/fakes.py`
- Test: `tests/unit/test_hybrid.py`, `tests/integration/test_hybrid_retrieval.py`

- [ ] **Step 1: Add a `FakeRetriever` to fakes** so hybrid logic is unit-testable with no DB. Append to `tests/unit/fakes.py`:

```python
from rag.models import Chunk, MetadataFilter, Provenance, ScoredChunk


def make_chunk(chunk_id: str, text: str = "x") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        source_uri="file:///x.pdf",
        text=text,
        ordinal=0,
        page=1,
        char_start=0,
        char_end=len(text),
        token_count=1,
    )


class FakeRetriever:
    """Returns a fixed ScoredChunk list; records the args of the last call."""

    def __init__(self, chunks: list[Chunk], provenance: Provenance = Provenance.dense) -> None:
        self._chunks = chunks
        self._provenance = provenance
        self.last_call: tuple[str, int, MetadataFilter | None] | None = None

    def retrieve(self, query: str, k: int, filt: MetadataFilter | None) -> list[ScoredChunk]:
        self.last_call = (query, k, filt)
        return [
            ScoredChunk(chunk=c, score=1.0 / (i + 1), provenance=self._provenance)
            for i, c in enumerate(self._chunks[:k])
        ]
```

- [ ] **Step 2: Write the failing unit test.** Create `tests/unit/test_hybrid.py`:

```python
from rag.models import Provenance
from rag.retrieval.hybrid import HybridRetriever
from tests.unit.fakes import FakeRetriever, make_chunk


def test_hybrid_fuses_both_sources_and_marks_provenance() -> None:
    a, b, c, d = (make_chunk(x) for x in "abcd")
    dense = FakeRetriever([a, b, c], provenance=Provenance.dense)
    lexical = FakeRetriever([b, a, d], provenance=Provenance.lexical)
    hybrid = HybridRetriever(dense=dense, lexical=lexical, rrf_k=60, candidate_k=3)

    results = hybrid.retrieve("q", k=3, filt=None)

    assert all(r.provenance == Provenance.fused for r in results)
    assert {r.chunk.chunk_id for r in results[:2]} == {"a", "b"}
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    # both legs over-fetched the candidate pool (max(k, candidate_k) = 3 here)
    assert dense.last_call == ("q", 3, None)
    assert lexical.last_call == ("q", 3, None)


def test_hybrid_over_fetches_candidate_pool_beyond_k() -> None:
    pool = [make_chunk(str(i)) for i in range(10)]
    dense = FakeRetriever(pool)
    lexical = FakeRetriever(list(reversed(pool)))
    hybrid = HybridRetriever(dense=dense, lexical=lexical, rrf_k=60, candidate_k=8)
    results = hybrid.retrieve("q", k=3, filt=None)
    assert dense.last_call == ("q", 8, None)      # asked for the pool, not just k
    assert len(results) == 3                        # but returns the fused top-k


def test_hybrid_returns_empty_when_both_empty() -> None:
    hybrid = HybridRetriever(
        dense=FakeRetriever([]), lexical=FakeRetriever([]), rrf_k=60
    )
    assert hybrid.retrieve("q", k=5, filt=None) == []
```

- [ ] **Step 3: Run, see it fail.** `uv run pytest tests/unit/test_hybrid.py -q` → FAIL.

- [ ] **Step 4: Implement.** Create `src/rag/retrieval/hybrid.py`:

```python
from __future__ import annotations

from rag.models import MetadataFilter, Provenance, ScoredChunk
from rag.protocols import Retriever
from rag.retrieval.fusion import reciprocal_rank_fusion


class HybridRetriever:
    """Runs dense + lexical retrieval and fuses their rankings with RRF.

    Each leg is over-fetched to ``max(k, candidate_k)`` candidates so RRF fuses a
    rich pool (fusing only top-k from each leg loses recall); the fused top-``k``
    are returned as ``ScoredChunk`` with ``provenance=fused`` and the RRF score.
    The same ``filt`` passes through to both legs — the geospatial metadata edge
    works identically on each.
    """

    def __init__(
        self, dense: Retriever, lexical: Retriever, rrf_k: int = 60, candidate_k: int = 30
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    def retrieve(
        self, query: str, k: int, filt: MetadataFilter | None
    ) -> list[ScoredChunk]:
        pool = max(k, self._candidate_k)
        dense = self._dense.retrieve(query, pool, filt)
        lexical = self._lexical.retrieve(query, pool, filt)
        by_id = {sc.chunk.chunk_id: sc.chunk for sc in (*dense, *lexical)}
        fused = reciprocal_rank_fusion(
            [[sc.chunk.chunk_id for sc in dense], [sc.chunk.chunk_id for sc in lexical]],
            k=self._rrf_k,
        )
        return [
            ScoredChunk(chunk=by_id[chunk_id], score=score, provenance=Provenance.fused)
            for chunk_id, score in fused[:k]
        ]
```

- [ ] **Step 5: Add an integration test over real data.** Create `tests/integration/test_hybrid_retrieval.py`:

```python
import pytest
from sqlalchemy import Engine

from rag.ingestion.repository import PgChunkRepository
from rag.models import Provenance
from rag.providers.embeddings import LiteLLMEmbeddingProvider  # noqa: F401  (import parity)
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.lexical import LexicalRetriever
from tests.integration.test_dense_retrieval import _chunk, _vec
from tests.unit.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


def test_hybrid_surfaces_lexical_only_and_dense_only_hits(clean_db: Engine) -> None:
    # 'c0' is the dense nearest; 'c1' shares the query's lexical terms only.
    mapping = {
        "Maize nitrogen response is strong.": _vec(1, 0, 0),
        "Irrigation scheduling for nitrogen uptake timing.": _vec(0, 0, 1),
        "QUERY": _vec(1, 0, 0),
    }
    embedder = FakeEmbedder(dim=768, mapping=mapping)
    repo = PgChunkRepository(clean_db)
    c0 = _chunk("c0", "Maize nitrogen response is strong.")
    c1 = _chunk("c1", "Irrigation scheduling for nitrogen uptake timing.")
    repo.upsert([c0, c1], [mapping[c0.text], mapping[c1.text]])

    hybrid = HybridRetriever(
        dense=DenseRetriever(engine=clean_db, embedder=embedder),
        lexical=LexicalRetriever(engine=clean_db, language="english"),
        rrf_k=60,
    )
    results = hybrid.retrieve("nitrogen", k=5, filt=None)
    ids = {r.chunk.chunk_id for r in results}
    assert ids == {"c0", "c1"}                       # dense-only + lexical-only both present
    assert all(r.provenance == Provenance.fused for r in results)
```

Note: the FakeEmbedder maps the literal chunk texts so the dense leg is deterministic; the query string `"QUERY"` maps to the same vector as `c0`. Adjust the embedder call — DenseRetriever embeds the *query* string, so add `"nitrogen": _vec(1, 0, 0)` to `mapping` (the query passed to `hybrid.retrieve`). Update the mapping dict accordingly before running.

- [ ] **Step 6: Run + commit.** `uv run pytest tests/unit/test_hybrid.py -q` and `uv run pytest -m integration tests/integration/test_hybrid_retrieval.py -q` → PASS; `uv run mypy src` → clean. Commit:

```bash
git add src/rag/retrieval/hybrid.py tests/unit/fakes.py tests/unit/test_hybrid.py tests/integration/test_hybrid_retrieval.py
git commit -m "feat(retrieval): HybridRetriever fusing dense+lexical via RRF"
```

---

## Track B — Cross-encoder rerank (Cohere via LiteLLM)

### Task 6: `Reranker` protocol + `CohereReranker`

**Files:**
- Modify: `src/rag/protocols.py`
- Create: `src/rag/providers/rerank.py`
- Test: `tests/unit/test_rerank.py`

- [ ] **Step 1: Write the failing unit test.** Create `tests/unit/test_rerank.py`. It monkeypatches `litellm.rerank` so no network is touched, and asserts the reranker reorders chunks by the provider's relevance and tags provenance:

```python
import litellm

from rag.models import Provenance
from rag.providers.rerank import CohereReranker
from tests.unit.fakes import make_chunk


class _FakeRerankResponse:
    # LiteLLM returns an object with a .results list of {index, relevance_score}
    def __init__(self, results: list[dict]) -> None:
        self.results = results


def test_rerank_reorders_by_relevance_and_truncates(monkeypatch) -> None:
    chunks = [make_chunk("a", "alpha"), make_chunk("b", "beta"), make_chunk("c", "gamma")]
    captured: dict = {}

    def fake_rerank(*, model, query, documents, top_n):
        captured.update(model=model, query=query, documents=documents, top_n=top_n)
        # provider says doc index 2 is best, then 0; drops index 1
        return _FakeRerankResponse([{"index": 2, "relevance_score": 0.9},
                                    {"index": 0, "relevance_score": 0.4}])

    monkeypatch.setattr(litellm, "rerank", fake_rerank)
    reranker = CohereReranker(model="cohere/rerank-english-v3.0", top_n=2)
    out = reranker.rerank("which is best?", chunks, top_n=2)

    assert [s.chunk.chunk_id for s in out] == ["c", "a"]
    assert [s.score for s in out] == [0.9, 0.4]
    assert all(s.provenance == Provenance.rerank for s in out)
    assert captured["documents"] == ["alpha", "beta", "gamma"]
    assert captured["top_n"] == 2


def test_rerank_empty_input_short_circuits(monkeypatch) -> None:
    def boom(**kwargs):  # must NOT be called
        raise AssertionError("litellm.rerank should not run on empty input")

    monkeypatch.setattr(litellm, "rerank", boom)
    assert CohereReranker().rerank("q", [], top_n=5) == []
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest tests/unit/test_rerank.py -q` → FAIL.

- [ ] **Step 3: Add the protocol.** In `src/rag/protocols.py`, add `ScoredChunk` is already imported; add the `Reranker` protocol after `Retriever`:

```python
@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: str, chunks: list[Chunk], top_n: int
    ) -> list[ScoredChunk]: ...
```

(`Chunk` and `ScoredChunk` are already in the existing model imports at the top of `protocols.py`.)

- [ ] **Step 4: Implement the reranker.** Create `src/rag/providers/rerank.py`:

```python
from __future__ import annotations

import litellm

from rag.models import Chunk, Provenance, ScoredChunk


class CohereReranker:
    """Cross-encoder rerank via ``litellm.rerank`` (Cohere by default).

    Sends the chunk texts as documents and reorders by the provider's relevance
    score, keeping the top ``top_n``. A local ``bge-reranker-v2-m3`` is the
    offline swap — change ``model`` only (spec §8). Verify the response shape
    against the current LiteLLM docs; results expose ``index`` + ``relevance_score``
    (attribute or mapping access are both handled below).
    """

    def __init__(self, model: str = "cohere/rerank-english-v3.0", top_n: int = 8) -> None:
        self.model = model
        self._top_n = top_n

    def rerank(
        self, query: str, chunks: list[Chunk], top_n: int | None = None
    ) -> list[ScoredChunk]:
        if not chunks:
            return []
        n = min(top_n or self._top_n, len(chunks))
        resp = litellm.rerank(
            model=self.model,
            query=query,
            documents=[c.text for c in chunks],
            top_n=n,
        )
        out: list[ScoredChunk] = []
        for r in resp.results:
            idx = r["index"] if isinstance(r, dict) else r.index
            score = r["relevance_score"] if isinstance(r, dict) else r.relevance_score
            out.append(
                ScoredChunk(chunk=chunks[idx], score=float(score), provenance=Provenance.rerank)
            )
        return out
```

- [ ] **Step 5: Run + commit.** `uv run pytest tests/unit/test_rerank.py -q` → PASS; `uv run mypy src` → clean. Commit:

```bash
git add src/rag/protocols.py src/rag/providers/rerank.py tests/unit/test_rerank.py
git commit -m "feat(providers): CohereReranker via litellm.rerank + Reranker protocol"
```

---

### Task 7: `RerankedRetriever` — rerank as a Retriever decorator

**Files:**
- Create: `src/rag/retrieval/reranked.py`
- Modify: `tests/unit/fakes.py`
- Test: `tests/unit/test_reranked.py`

**Design:** Rerank is a retrieval concern, so it composes as a `Retriever` that wraps another `Retriever` + a `Reranker`. It over-fetches `candidate_k` from the base, reranks, and returns the top-`k`. The answerer stays oblivious — it just gets "a retriever", no rerank parameter. This makes the eval A/B (Task 18) a one-line base-swap, gives rerank its own timing span (Task 11), and keeps `StraightLineAnswerer`/`StreamingAnswerer` **unchanged by rerank**. No `select_chunks` layer is needed.

- [ ] **Step 1: Add a `FakeReranker` to fakes.** Append to `tests/unit/fakes.py`:

```python
class FakeReranker:
    """Reverses the candidate order and keeps top_n; tags provenance=rerank."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[ScoredChunk]:
        self.calls.append((query, top_n))
        reversed_chunks = list(reversed(chunks))[:top_n]
        return [
            ScoredChunk(chunk=c, score=float(len(reversed_chunks) - i), provenance=Provenance.rerank)
            for i, c in enumerate(reversed_chunks)
        ]
```

- [ ] **Step 2: Write the failing unit test.** Create `tests/unit/test_reranked.py`:

```python
from rag.models import Provenance
from rag.retrieval.reranked import RerankedRetriever
from tests.unit.fakes import FakeReranker, FakeRetriever, make_chunk


def test_reranked_over_fetches_pool_then_returns_reranked_top_k() -> None:
    pool = [make_chunk(str(i)) for i in range(10)]
    base = FakeRetriever(pool)
    reranker = FakeReranker()
    rr = RerankedRetriever(base=base, reranker=reranker, candidate_k=8)

    results = rr.retrieve("q", k=3, filt=None)

    assert base.last_call == ("q", 8, None)             # over-fetched candidate_k
    assert reranker.calls == [("q", 3)]                  # reranked down to k
    assert [r.chunk.chunk_id for r in results] == ["9", "8", "7"]  # FakeReranker reverses
    assert all(r.provenance == Provenance.rerank for r in results)


def test_reranked_uses_k_when_larger_than_candidate_k() -> None:
    base = FakeRetriever([make_chunk(str(i)) for i in range(5)])
    rr = RerankedRetriever(base=base, reranker=FakeReranker(), candidate_k=2)
    rr.retrieve("q", k=4, filt=None)
    assert base.last_call == ("q", 4, None)              # pool = max(k, candidate_k)


def test_reranked_empty_base_short_circuits() -> None:
    reranker = FakeReranker()
    rr = RerankedRetriever(base=FakeRetriever([]), reranker=reranker, candidate_k=8)
    assert rr.retrieve("q", k=3, filt=None) == []
    assert reranker.calls == []                           # never reranks nothing
```

- [ ] **Step 3: Run, see it fail.** `uv run pytest tests/unit/test_reranked.py -q` → FAIL.

- [ ] **Step 4: Implement.** Create `src/rag/retrieval/reranked.py`:

```python
from __future__ import annotations

from rag.models import MetadataFilter, ScoredChunk
from rag.protocols import Reranker, Retriever


class RerankedRetriever:
    """A Retriever decorator: over-fetch from a base retriever, then rerank.

    ``retrieve(query, k, filt)`` fetches ``max(k, candidate_k)`` candidates from the
    base (e.g. a HybridRetriever), reranks them with a cross-encoder, and returns
    the reranked top-``k`` (``provenance=rerank``). Because it IS a Retriever, the
    answerer never knows rerank is happening, and Task 18 measures rerank lift by
    swapping the base retriever for this one — a clean before/after.
    """

    def __init__(self, base: Retriever, reranker: Reranker, candidate_k: int = 30) -> None:
        self._base = base
        self._reranker = reranker
        self._candidate_k = candidate_k

    def retrieve(
        self, query: str, k: int, filt: MetadataFilter | None
    ) -> list[ScoredChunk]:
        pool = max(k, self._candidate_k)
        scored = self._base.retrieve(query, pool, filt)
        if not scored:
            return []
        return self._reranker.rerank(query, [s.chunk for s in scored], top_n=k)
```

- [ ] **Step 5: Run + commit.** `uv run pytest tests/unit/test_reranked.py -q` → PASS; `uv run mypy src` → clean. Commit:

```bash
git add src/rag/retrieval/reranked.py tests/unit/fakes.py tests/unit/test_reranked.py
git commit -m "feat(retrieval): RerankedRetriever decorator (over-fetch candidate_k + rerank)"
```

---

## Track C — SSE streaming

### Task 8: `AnswerEvent` models + `LLMProvider.stream`

**Files:**
- Modify: `src/rag/models.py`
- Modify: `src/rag/protocols.py`
- Modify: `src/rag/providers/llm.py`
- Modify: `tests/unit/fakes.py`
- Test: `tests/unit/test_models.py`, `tests/unit/test_llm_provider.py`

- [ ] **Step 1: Write the failing model + provider tests.**

  Append to `tests/unit/test_models.py`:
```python
def test_answer_events_carry_discriminator_type() -> None:
    from rag.models import DoneEvent, ErrorEvent, TokenEvent

    assert TokenEvent(text="hi").type == "token"
    assert DoneEvent(answer="a").type == "done"
    assert ErrorEvent(message="boom").type == "error"
    # round-trips as JSON for SSE payloads
    assert TokenEvent(text="hi").model_dump_json() == '{"type":"token","text":"hi"}'
```

  Append to `tests/unit/test_llm_provider.py`:
```python
def test_litellm_provider_stream_yields_content_deltas(monkeypatch) -> None:
    import litellm

    from rag.providers.llm import LiteLLMProvider

    class _Delta:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.delta = _Delta(content)

    class _Chunk:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    def fake_completion(**kwargs):
        assert kwargs["stream"] is True
        return iter([_Chunk("Nitro"), _Chunk("gen"), _Chunk(None), _Chunk(" helps.")])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    provider = LiteLLMProvider(model="gemini/gemini-2.5-pro")
    assert list(provider.stream([{"role": "user", "content": "q"}])) == ["Nitro", "gen", " helps."]
```

- [ ] **Step 2: Run, see them fail.** `uv run pytest tests/unit/test_models.py tests/unit/test_llm_provider.py -q` → FAIL.

- [ ] **Step 3: Add the event models.** In `src/rag/models.py`, add `Literal` to the typing import and define the events near the bottom (after `Answer`):

```python
from typing import Literal  # add at the top with the other imports


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    usage: dict[str, object] = Field(default_factory=dict)
    trace_id: str = ""
    retrieval_scope: RetrievalScope = RetrievalScope.corpus_only


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
    trace_id: str = ""


AnswerEvent = TokenEvent | DoneEvent | ErrorEvent
```

- [ ] **Step 4: Add `stream` to the protocol.** In `src/rag/protocols.py`, extend `LLMProvider`:

```python
@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, messages: list[dict[str, object]], **opts: object) -> Completion: ...
    def stream(self, messages: list[dict[str, object]], **opts: object) -> Iterator[str]: ...
```

(`Iterator` is already imported from `collections.abc` at the top of `protocols.py`.)

- [ ] **Step 5: Implement `LiteLLMProvider.stream`.** In `src/rag/providers/llm.py`, add `Iterator` import and the method:

```python
from collections.abc import Iterator  # add at top
```
```python
    def stream(self, messages: list[dict[str, object]], **opts: object) -> Iterator[str]:
        resp = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
```

  Also update the docstring line in `LiteLLMProvider` from "Streaming is added in P0b." to "Non-streaming `complete` for graders/JSON; `stream` for the SSE answer path."

- [ ] **Step 6: Add a streaming fake.** Append to `tests/unit/fakes.py`:

```python
class FakeStreamingLLM:
    """Yields canned token deltas for stream(); complete() returns the joined text."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.tokens = tokens if tokens is not None else ["Answer ", "grounded ", "[1]."]

    def stream(self, messages: list[dict], **opts: object):
        yield from self.tokens

    def complete(self, messages: list[dict], **opts: object) -> Completion:
        return Completion(text="".join(self.tokens), usage={"total_tokens": 0})
```

- [ ] **Step 7: Run + commit.** `uv run pytest tests/unit/test_models.py tests/unit/test_llm_provider.py -q` → PASS; `uv run mypy src` → clean. Commit:

```bash
git add src/rag/models.py src/rag/protocols.py src/rag/providers/llm.py tests/unit/fakes.py tests/unit/test_models.py tests/unit/test_llm_provider.py
git commit -m "feat(streaming): AnswerEvent models + LLMProvider.stream (LiteLLM)"
```

---

### Task 9: `StreamingAnswerer` — token stream + terminal validated citations

**Files:**
- Create: `src/rag/generation/streaming.py`
- Test: `tests/unit/test_streaming_answerer.py`

- [ ] **Step 1: Write the failing unit test.** Create `tests/unit/test_streaming_answerer.py`:

```python
from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.streaming import StreamingAnswerer
from rag.models import DoneEvent, ErrorEvent, TokenEvent
from tests.unit.fakes import FakeRetriever, FakeStreamingLLM, make_chunk


def _answerer(llm, retriever):
    return StreamingAnswerer(
        retriever=retriever,
        assembler=TokenBudgetAssembler(),
        llm=llm,
        token_budget=1000,
        retrieval_k=3,
    )


def test_stream_emits_tokens_then_done_with_validated_citations() -> None:
    chunks = [make_chunk("a", text="alpha"), make_chunk("b", text="beta")]
    llm = FakeStreamingLLM(tokens=["alpha ", "is true ", "[1]", " [9]"])  # [9] is invalid
    events = list(_answerer(llm, FakeRetriever(chunks)).answer_stream("q"))

    tokens = [e for e in events if isinstance(e, TokenEvent)]
    done = events[-1]
    assert [t.text for t in tokens] == ["alpha ", "is true ", "[1]", " [9]"]
    assert isinstance(done, DoneEvent)
    assert "[9]" not in done.answer                      # invalid marker stripped
    assert [c.marker for c in done.citations] == ["[1]"]
    assert done.citations[0].chunk_id == "a"
    assert done.usage["completion_tokens_est"] >= 1
    assert done.trace_id


def test_stream_empty_retrieval_yields_no_context_done() -> None:
    events = list(_answerer(FakeStreamingLLM(), FakeRetriever([])).answer_stream("q"))
    assert len(events) == 1
    assert isinstance(events[0], DoneEvent)
    assert events[0].answer == "I don't have relevant context to answer that."
    assert events[0].citations == []


def test_stream_provider_error_flushes_partial_then_error_event() -> None:
    class _BoomLLM:
        def stream(self, messages, **opts):
            yield "partial "
            raise RuntimeError("provider exploded")

        def complete(self, messages, **opts):  # unused
            raise NotImplementedError

    chunks = [make_chunk("a", text="alpha")]
    events = list(_answerer(_BoomLLM(), FakeRetriever(chunks)).answer_stream("q"))
    assert isinstance(events[0], TokenEvent) and events[0].text == "partial "
    assert isinstance(events[-1], ErrorEvent)
    assert "provider exploded" in events[-1].message


def test_answer_collects_stream_into_answer_object() -> None:
    chunks = [make_chunk("a", text="alpha")]
    llm = FakeStreamingLLM(tokens=["alpha ", "[1]"])
    answer = _answerer(llm, FakeRetriever(chunks)).answer("q")
    assert answer.text.strip().endswith("[1]")
    assert answer.citations[0].chunk_id == "a"
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest tests/unit/test_streaming_answerer.py -q` → FAIL.

- [ ] **Step 3: Implement.** Create `src/rag/generation/streaming.py`:

```python
from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from importlib import resources

import structlog

from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.citations import validate_and_build_citations
from rag.models import (
    Answer,
    AnswerEvent,
    DoneEvent,
    ErrorEvent,
    MetadataFilter,
    RetrievalScope,
    TokenEvent,
)
from rag.protocols import LLMProvider, Retriever
from rag.util.tokens import count_tokens

log = structlog.get_logger()

_NO_CONTEXT = "I don't have relevant context to answer that."


def _load_prompt() -> str:
    return (
        resources.files("rag.generation.prompts")
        .joinpath("answer_v1.md")
        .read_text(encoding="utf-8")
    )


class StreamingAnswerer:
    """Streaming read path: retrieve → assemble → stream → validate.

    Emits ``TokenEvent`` per delta for live UX, then a terminal ``DoneEvent`` whose
    ``answer``/``citations`` come from the SAME post-gen validator as the JSON path
    (invalid markers stripped). A provider error flushes the partial stream and ends
    with an ``ErrorEvent`` (spec §16). ``answer()`` collects the stream into an
    ``Answer`` so the JSON ``/query`` route and tests share one implementation.

    Rerank is NOT a parameter here — inject a ``RerankedRetriever`` as ``retriever``
    and this class is unchanged (Task 7). It only ever sees "a retriever".
    """

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

    def answer_stream(
        self,
        query: str,
        filt: MetadataFilter | None = None,
        scope: RetrievalScope = RetrievalScope.corpus_only,
    ) -> Iterator[AnswerEvent]:
        trace_id = str(uuid.uuid4())
        started = time.perf_counter()
        bound_log = log.bind(trace_id=trace_id)

        scored = self._retriever.retrieve(query, self._retrieval_k, filt)
        if not scored:
            bound_log.info("no_context")
            yield DoneEvent(answer=_NO_CONTEXT, trace_id=trace_id, retrieval_scope=scope)
            return

        chunks = [s.chunk for s in scored]
        context = self._assembler.assemble(query, chunks, self._token_budget)
        messages: list[dict[str, object]] = [
            {"role": "user", "content": self._prompt.format(context=context.text, question=query)}
        ]

        parts: list[str] = []
        try:
            for delta in self._llm.stream(messages):
                parts.append(delta)
                yield TokenEvent(text=delta)
        except Exception as exc:  # flush partial, then surface as a terminal event
            bound_log.warning("stream_error", error=str(exc))
            yield ErrorEvent(message=str(exc), trace_id=trace_id)
            return

        full_text = "".join(parts)
        cleaned, citations = validate_and_build_citations(full_text, context)
        usage = {
            "completion_tokens_est": count_tokens(full_text),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        bound_log.info("answered_stream", citations=len(citations), latency_ms=usage["latency_ms"])
        yield DoneEvent(
            answer=cleaned,
            citations=citations,
            usage=usage,
            trace_id=trace_id,
            retrieval_scope=scope,
        )

    def answer(
        self,
        query: str,
        filt: MetadataFilter | None = None,
        scope: RetrievalScope = RetrievalScope.corpus_only,
    ) -> Answer:
        done: DoneEvent | None = None
        for event in self.answer_stream(query, filt, scope):
            if isinstance(event, ErrorEvent):
                raise RuntimeError(event.message)
            if isinstance(event, DoneEvent):
                done = event
        assert done is not None  # answer_stream always ends with a DoneEvent on success
        return Answer(
            text=done.answer,
            citations=done.citations,
            usage=done.usage,
            trace_id=done.trace_id,
            retrieval_scope=done.retrieval_scope,
        )
```

- [ ] **Step 4: Run, see it pass.** `uv run pytest tests/unit/test_streaming_answerer.py -q` → PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/rag/generation/streaming.py tests/unit/test_streaming_answerer.py
git commit -m "feat(generation): StreamingAnswerer (token stream + terminal validated citations)"
```

---

### Task 10: SSE `/query/stream` endpoint + app wiring (hybrid + rerank + streaming)

**Files:**
- Modify: `src/rag/api/routes.py`
- Modify: `src/rag/api/app.py`
- Test: `tests/integration/test_streaming_api.py`

- [ ] **Step 1: Write the failing integration test.** Create `tests/integration/test_streaming_api.py` (uses `TestClient`, which drives the SSE response synchronously; no DB needed — a fake retriever/LLM are injected):

```python
import json

import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.streaming import StreamingAnswerer
from tests.unit.fakes import FakeRetriever, FakeStreamingLLM, make_chunk

pytestmark = pytest.mark.integration


def _client() -> TestClient:
    chunks = [make_chunk("a", text="alpha")]
    answerer = StreamingAnswerer(
        retriever=FakeRetriever(chunks),
        assembler=TokenBudgetAssembler(),
        llm=FakeStreamingLLM(tokens=["alpha ", "is true ", "[1]"]),
        token_budget=1000,
        retrieval_k=3,
    )
    return TestClient(create_app(answerer=answerer))


def test_stream_endpoint_emits_token_then_done_events() -> None:
    with _client() as client:
        with client.stream("POST", "/query/stream", json={"query": "is alpha true?"}) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            body = "".join(resp.iter_text())

    # SSE frames: "event: <type>\ndata: <json>\n\n"
    assert "event: token" in body
    assert "event: done" in body
    done_payload = body.split("event: done")[1].split("data: ")[1].splitlines()[0]
    done = json.loads(done_payload)
    assert done["citations"][0]["chunk_id"] == "a"
    assert "[1]" in done["answer"]


def test_json_query_still_works_with_streaming_answerer() -> None:
    with _client() as client:
        resp = client.post("/query", json={"query": "is alpha true?"})
    assert resp.status_code == 200
    assert resp.json()["citations"][0]["chunk_id"] == "a"
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest -m integration tests/integration/test_streaming_api.py -q` → FAIL (no `/query/stream`).

- [ ] **Step 3: Add the SSE route.** In `src/rag/api/routes.py`, add imports and the endpoint:

```python
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool
```
```python
@router.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest) -> EventSourceResponse:
    answerer = request.app.state.answerer

    async def event_publisher() -> "AsyncIterator[dict[str, str]]":
        # answer_stream is a sync generator (LiteLLM stream is sync); run it in a
        # threadpool so the event loop is never blocked.
        gen = answerer.answer_stream(body.query, body.filter)
        async for event in iterate_in_threadpool(gen):
            yield {"event": event.type, "data": event.model_dump_json()}

    return EventSourceResponse(event_publisher())
```

  Add the typing import at the top of the file:
```python
from collections.abc import AsyncIterator
```

- [ ] **Step 4: Wire the production answerer to hybrid + rerank + streaming.** Replace `_build_answerer` in `src/rag/api/app.py` so the app composes the full P0b read path. Replace the whole file body with:

```python
from __future__ import annotations

from fastapi import FastAPI

from rag.api.routes import router
from rag.config import apply_provider_env, get_settings
from rag.db import get_engine
from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.streaming import StreamingAnswerer
from rag.observability.tracing import configure_observability
from rag.providers.embeddings import LiteLLMEmbeddingProvider
from rag.providers.llm import LiteLLMProvider
from rag.providers.rerank import CohereReranker
from rag.protocols import Retriever
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.lexical import LexicalRetriever
from rag.retrieval.reranked import RerankedRetriever


def _build_answerer() -> StreamingAnswerer:
    settings = get_settings()
    apply_provider_env(settings)
    configure_observability(settings)
    engine = get_engine(settings.database_url)
    embedder = LiteLLMEmbeddingProvider(
        model=settings.embedding_model, dim=settings.embedding_dim
    )
    retriever: Retriever = HybridRetriever(
        dense=DenseRetriever(engine=engine, embedder=embedder),
        lexical=LexicalRetriever(engine=engine),   # FTS frozen to english (migration 0002)
        rrf_k=settings.rrf_k,
        candidate_k=settings.candidate_k,
    )
    if settings.rerank_enabled:
        # Decorate the hybrid retriever — the answerer never knows rerank happened.
        retriever = RerankedRetriever(
            base=retriever,
            reranker=CohereReranker(model=settings.rerank_model),
            candidate_k=settings.candidate_k,
        )
    return StreamingAnswerer(
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


def create_app(answerer: StreamingAnswerer | None = None) -> FastAPI:
    app = FastAPI(title="Production RAG — P0b")
    app.state.answerer = answerer if answerer is not None else _build_answerer()
    app.include_router(router)
    return app
```

> **Note:** `configure_observability` is created in Task 12. Until then, comment out that import + call, or implement Task 12 before Task 10's app rewrite. (Recommended execution order: 11 → 12 → then this app wiring, or stub `configure_observability` to a no-op first.) The plan's self-contained order assumes the stub exists; if running strictly in order, add a temporary `def configure_observability(_): return False` in `src/rag/observability/tracing.py` now and flesh it out in Task 12.

- [ ] **Step 5: Run, see it pass.** `uv run pytest -m integration tests/integration/test_streaming_api.py -q` → PASS. Re-run the existing `tests/integration/test_api.py` → still PASS (JSON `/query` unchanged; it still accepts an injected answerer with `.answer()`).

- [ ] **Step 6: Commit.**

```bash
git add src/rag/api/routes.py src/rag/api/app.py tests/integration/test_streaming_api.py
git commit -m "feat(api): SSE /query/stream + wire hybrid+rerank+streaming read path"
```

---

## Track D — Observability (Prometheus + Grafana + Langfuse)

### Task 11: Prometheus metrics — counters, latency histograms, middleware, `/metrics`

**Files:**
- Create: `src/rag/observability/__init__.py` (empty)
- Create: `src/rag/observability/metrics.py`
- Modify: `src/rag/api/routes.py`, `src/rag/api/app.py`
- Test: `tests/unit/test_metrics.py`, `tests/integration/test_metrics_endpoint.py`

- [ ] **Step 1: Write the failing unit test.** Create `tests/unit/test_metrics.py`:

```python
def test_observe_stage_records_a_sample() -> None:
    from rag.observability import metrics

    before = metrics.STAGE_LATENCY.labels(stage="dense")._sum.get()
    with metrics.observe_stage("dense"):
        pass
    after = metrics.STAGE_LATENCY.labels(stage="dense")._sum.get()
    assert after >= before
    # a count sample exists for the labelled stage
    assert metrics.STAGE_LATENCY.labels(stage="dense")._count.get() >= 1.0


def test_render_latest_returns_prometheus_text() -> None:
    from rag.observability import metrics

    metrics.REQUESTS.labels(endpoint="/query", status="200").inc()
    text, content_type = metrics.render_latest()
    assert "rag_requests_total" in text
    assert "text/plain" in content_type
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest tests/unit/test_metrics.py -q` → FAIL.

- [ ] **Step 3: Implement the metrics module.** Create `src/rag/observability/__init__.py` (empty) and `src/rag/observability/metrics.py`:

```python
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Default process registry — one set of metrics for the whole app.
REQUESTS = Counter(
    "rag_requests_total", "HTTP requests handled", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds", "End-to-end HTTP latency", ["endpoint"]
)
STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds",
    "Per-stage read-path latency",
    ["stage"],  # dense | lexical | hybrid | rerank | assemble | generate
)


@contextmanager
def observe_stage(stage: str) -> Iterator[None]:
    """Time a read-path stage into ``rag_stage_latency_seconds{stage=...}``."""
    start = perf_counter()
    try:
        yield
    finally:
        STAGE_LATENCY.labels(stage=stage).observe(perf_counter() - start)


def render_latest() -> tuple[str, str]:
    """Return (exposition_text, content_type) for the /metrics endpoint."""
    return generate_latest().decode("utf-8"), CONTENT_TYPE_LATEST
```

- [ ] **Step 4: Add middleware + `/metrics` route.** In `src/rag/api/routes.py`, add the metrics endpoint:

```python
from fastapi import Response

from rag.observability import metrics
```
```python
@router.get("/metrics")
async def metrics_endpoint() -> Response:
    text, content_type = metrics.render_latest()
    return Response(content=text, media_type=content_type)
```

  In `src/rag/api/app.py`, add a request-timing middleware inside `create_app` (before `return app`):

```python
from time import perf_counter

from starlette.requests import Request as StarletteRequest

# ... inside create_app, after include_router:
    @app.middleware("http")
    async def _record_metrics(request: StarletteRequest, call_next):  # type: ignore[no-untyped-def]
        start = perf_counter()
        response = await call_next(request)
        endpoint = request.url.path
        from rag.observability import metrics

        metrics.REQUEST_LATENCY.labels(endpoint=endpoint).observe(perf_counter() - start)
        metrics.REQUESTS.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        return response
```

- [ ] **Step 5: Instrument each read-path stage separately** (spec §9 wants dense / lexical / fusion / rerank / assemble / generate as distinct p95 lines). Each stage is timed in its natural home, so timing follows whichever retriever is wired in.

  **5a — `src/rag/retrieval/hybrid.py`** (dense, lexical, fusion). Add the import and wrap each leg + the fuse:
```python
from rag.observability.metrics import observe_stage
```
```python
    def retrieve(
        self, query: str, k: int, filt: MetadataFilter | None
    ) -> list[ScoredChunk]:
        pool = max(k, self._candidate_k)
        with observe_stage("dense"):
            dense = self._dense.retrieve(query, pool, filt)
        with observe_stage("lexical"):
            lexical = self._lexical.retrieve(query, pool, filt)
        with observe_stage("fusion"):
            by_id = {sc.chunk.chunk_id: sc.chunk for sc in (*dense, *lexical)}
            fused = reciprocal_rank_fusion(
                [[sc.chunk.chunk_id for sc in dense], [sc.chunk.chunk_id for sc in lexical]],
                k=self._rrf_k,
            )
            return [
                ScoredChunk(chunk=by_id[chunk_id], score=score, provenance=Provenance.fused)
                for chunk_id, score in fused[:k]
            ]
```

  **5b — `src/rag/retrieval/reranked.py`** (rerank). Add the import and wrap the rerank call:
```python
from rag.observability.metrics import observe_stage
```
```python
        scored = self._base.retrieve(query, pool, filt)
        if not scored:
            return []
        with observe_stage("rerank"):
            return self._reranker.rerank(query, [s.chunk for s in scored], top_n=k)
```

  **5c — `src/rag/generation/streaming.py`** (assemble, generate). Add the import and wrap assembly + the stream loop (retrieval is already timed inside the retriever, so there is no "retrieve" span here):
```python
from rag.observability.metrics import observe_stage
```
```python
        chunks = [s.chunk for s in scored]
        with observe_stage("assemble"):
            context = self._assembler.assemble(query, chunks, self._token_budget)
```
```python
        parts: list[str] = []
        try:
            with observe_stage("generate"):
                for delta in self._llm.stream(messages):
                    parts.append(delta)
                    yield TokenEvent(text=delta)
```
  (Keep the existing `except Exception` and post-loop logic; only the `for` loop is now nested under `observe_stage`. `yield` inside the context manager is fine — the timer closes when the generator block exits.)

- [ ] **Step 6: Add the integration test for `/metrics`.** Create `tests/integration/test_metrics_endpoint.py`:

```python
import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.streaming import StreamingAnswerer
from tests.unit.fakes import FakeRetriever, FakeStreamingLLM, make_chunk

pytestmark = pytest.mark.integration


def test_metrics_endpoint_reports_request_counts() -> None:
    answerer = StreamingAnswerer(
        retriever=FakeRetriever([make_chunk("a", text="alpha")]),
        assembler=TokenBudgetAssembler(),
        llm=FakeStreamingLLM(tokens=["alpha ", "[1]"]),
        token_budget=1000,
        retrieval_k=3,
    )
    client = TestClient(create_app(answerer=answerer))
    client.post("/query", json={"query": "alpha?"})
    body = client.get("/metrics").text
    assert "rag_requests_total" in body
    assert "rag_stage_latency_seconds" in body
```

- [ ] **Step 7: Run + commit.** `uv run pytest tests/unit/test_metrics.py -q` and `uv run pytest -m integration tests/integration/test_metrics_endpoint.py -q` → PASS; `uv run mypy src` → clean. Commit:

```bash
git add src/rag/observability/__init__.py src/rag/observability/metrics.py src/rag/api/routes.py src/rag/api/app.py src/rag/retrieval/hybrid.py src/rag/retrieval/reranked.py src/rag/generation/streaming.py tests/unit/test_metrics.py tests/integration/test_metrics_endpoint.py
git commit -m "feat(observability): Prometheus /metrics, request + per-stage latency (dense/lexical/fusion/rerank/assemble/generate)"
```

---

### Task 12: Langfuse tracing via the LiteLLM callback (config-gated)

**Files:**
- Create: `src/rag/observability/tracing.py`
- Test: `tests/unit/test_tracing.py`

- [ ] **Step 1: Write the failing unit test.** Create `tests/unit/test_tracing.py`:

```python
def test_configure_observability_sets_litellm_callbacks_when_keys_present(monkeypatch) -> None:
    import litellm

    from rag.config import Settings
    from rag.observability.tracing import configure_observability

    monkeypatch.setattr(litellm, "success_callback", [], raising=False)
    monkeypatch.setattr(litellm, "failure_callback", [], raising=False)
    s = Settings(_env_file=None, langfuse_public_key="pk", langfuse_secret_key="sk")

    assert configure_observability(s) is True
    assert "langfuse" in litellm.success_callback
    assert "langfuse" in litellm.failure_callback


def test_configure_observability_is_noop_without_keys(monkeypatch) -> None:
    import litellm

    from rag.config import Settings
    from rag.observability.tracing import configure_observability

    monkeypatch.setattr(litellm, "success_callback", [], raising=False)
    s = Settings(_env_file=None)  # no langfuse keys
    assert configure_observability(s) is False
    assert litellm.success_callback == []
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest tests/unit/test_tracing.py -q` → FAIL.

- [ ] **Step 3: Implement.** Create `src/rag/observability/tracing.py`:

```python
from __future__ import annotations

import litellm
import structlog

from rag.config import Settings, apply_provider_env, get_settings

log = structlog.get_logger()


def configure_observability(settings: Settings | None = None) -> bool:
    """Enable the LiteLLM→Langfuse callback when Langfuse keys are configured.

    LiteLLM emits a trace per model call (cost, tokens, latency) to Langfuse via
    its built-in callback; we only register it and export the keys. Returns True
    if tracing was enabled, False if it is a no-op (no keys). Consult the current
    LiteLLM + Langfuse docs before changing callback wiring (spec §18).
    """
    settings = settings or get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return False
    apply_provider_env(settings)  # exports LANGFUSE_* into os.environ for LiteLLM
    if "langfuse" not in litellm.success_callback:
        litellm.success_callback = [*litellm.success_callback, "langfuse"]
    if "langfuse" not in litellm.failure_callback:
        litellm.failure_callback = [*litellm.failure_callback, "langfuse"]
    log.info("observability_enabled", backend="langfuse", host=settings.langfuse_host)
    return True
```

> If Task 10's app wiring was implemented with a temporary no-op stub of `configure_observability`, replace that stub with this file now.

- [ ] **Step 4: Run + commit.** `uv run pytest tests/unit/test_tracing.py -q` → PASS; `uv run mypy src` → clean. Commit:

```bash
git add src/rag/observability/tracing.py tests/unit/test_tracing.py
git commit -m "feat(observability): config-gated Langfuse tracing via LiteLLM callback"
```

---

### Task 13: Monitoring configs + optional observability compose

**Files:**
- Create: `monitoring/prometheus.yml`
- Create: `monitoring/grafana/dashboard.json`
- Create: `docker-compose.observability.yml`
- Test: none (config artifacts) — verification is structural (valid YAML/JSON, compose config parses)

> This task ships infra-as-config for reviewers; the author is **not** required to run Docker (Task 11's `/metrics` and Task 12's Langfuse callback already satisfy "Done when"). These files make the dashboards reproducible.

- [ ] **Step 1: Prometheus scrape config.** Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: rag-api
    metrics_path: /metrics
    static_configs:
      # host.docker.internal reaches the API running on the host (no-Docker dev).
      - targets: ["host.docker.internal:8000"]
```

- [ ] **Step 2: Grafana dashboard.** Create `monitoring/grafana/dashboard.json` — a minimal dashboard with four panels (request rate, p95 request latency, per-stage p95, error rate). Use this exact starter JSON:

```json
{
  "title": "RAG Service",
  "schemaVersion": 39,
  "panels": [
    {
      "type": "timeseries", "title": "Request rate (rps)",
      "targets": [{"expr": "sum(rate(rag_requests_total[1m])) by (endpoint)"}]
    },
    {
      "type": "timeseries", "title": "Request p95 (s)",
      "targets": [{"expr": "histogram_quantile(0.95, sum(rate(rag_request_latency_seconds_bucket[5m])) by (le, endpoint))"}]
    },
    {
      "type": "timeseries", "title": "Stage p95 (s)",
      "targets": [{"expr": "histogram_quantile(0.95, sum(rate(rag_stage_latency_seconds_bucket[5m])) by (le, stage))"}]
    },
    {
      "type": "timeseries", "title": "Error rate",
      "targets": [{"expr": "sum(rate(rag_requests_total{status=~\"5..\"}[5m]))"}]
    }
  ]
}
```

- [ ] **Step 3: Optional compose.** Create `docker-compose.observability.yml`:

```yaml
# Optional: `docker compose -f docker-compose.observability.yml up` to view dashboards.
# The API + Postgres run on the host; this stack only scrapes/visualises.
services:
  prometheus:
    image: prom/prometheus:v2.54.1
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    ports: ["9090:9090"]
    extra_hosts: ["host.docker.internal:host-gateway"]
  grafana:
    image: grafana/grafana:11.2.0
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin
    ports: ["3000:3000"]
    depends_on: [prometheus]
  langfuse:
    image: langfuse/langfuse:2
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/langfuse
      - NEXTAUTH_SECRET=dev-secret-change-me
      - SALT=dev-salt-change-me
      - NEXTAUTH_URL=http://localhost:3001
    ports: ["3001:3000"]
    extra_hosts: ["host.docker.internal:host-gateway"]
```

- [ ] **Step 4: Verify the configs parse.** Run:

```bash
uv run python -c "import yaml; yaml.safe_load(open('monitoring/prometheus.yml')); yaml.safe_load(open('docker-compose.observability.yml')); print('yaml ok')"
uv run python -c "import json; json.load(open('monitoring/grafana/dashboard.json')); print('json ok')"
```
Expected: `yaml ok` then `json ok`.

- [ ] **Step 5: Commit.**

```bash
git add monitoring/ docker-compose.observability.yml
git commit -m "build(observability): prometheus scrape config, grafana dashboard, optional compose"
```

---

## Track E — Eval harness (stratified golden set + separated scorecards + CIs)

### Task 14: Retrieval metrics — hit@k, reciprocal rank, nDCG@k (pure)

**Files:**
- Create: `src/rag/eval/__init__.py` (empty)
- Create: `src/rag/eval/metrics.py`
- Test: `tests/unit/test_eval_metrics.py`

- [ ] **Step 1: Write the failing unit test.** Create `tests/unit/test_eval_metrics.py`:

```python
import math

from rag.eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank


def test_hit_at_k() -> None:
    assert hit_at_k(["d2", "d1", "d3"], {"d1"}, k=2) == 1.0
    assert hit_at_k(["d2", "d3", "d1"], {"d1"}, k=2) == 0.0   # d1 is at rank 3
    assert hit_at_k([], {"d1"}, k=5) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["d2", "d1", "d3"], {"d1"}) == 0.5  # first relevant at rank 2
    assert reciprocal_rank(["d1"], {"d1"}) == 1.0
    assert reciprocal_rank(["d2", "d3"], {"d1"}) == 0.0


def test_ndcg_at_k() -> None:
    # one relevant doc at rank 2 → DCG = 1/log2(3); IDCG = 1/log2(2) = 1
    expected = (1 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_k(["d2", "d1", "d3"], {"d1"}, k=3), expected, rel_tol=1e-9)
    # perfect ranking → 1.0
    assert math.isclose(ndcg_at_k(["d1", "d2"], {"d1", "d2"}, k=2), 1.0, rel_tol=1e-9)
    assert ndcg_at_k(["d2"], {"d1"}, k=1) == 0.0
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest tests/unit/test_eval_metrics.py -q` → FAIL.

- [ ] **Step 3: Implement.** Create `src/rag/eval/__init__.py` (empty) and `src/rag/eval/metrics.py`:

```python
from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import log2


def hit_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """1.0 if any relevant id appears in the top-k retrieved ids, else 0.0."""
    relevant = set(relevant_ids)
    return 1.0 if any(doc_id in relevant for doc_id in retrieved_ids[:k]) else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """1 / rank of the first relevant id (1-based), or 0.0 if none retrieved."""
    relevant = set(relevant_ids)
    for index, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            return 1.0 / (index + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k. IDCG assumes all relevant docs ranked first."""
    relevant = set(relevant_ids)
    dcg = sum(
        1.0 / log2(index + 2)
        for index, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0
```

- [ ] **Step 4: Run + commit.** `uv run pytest tests/unit/test_eval_metrics.py -q` → PASS. Commit:

```bash
git add src/rag/eval/__init__.py src/rag/eval/metrics.py tests/unit/test_eval_metrics.py
git commit -m "feat(eval): retrieval metrics hit@k, reciprocal_rank, ndcg@k (pure)"
```

---

### Task 15: Bootstrap confidence intervals

**Files:**
- Create: `src/rag/eval/stats.py`
- Test: `tests/unit/test_eval_stats.py`

- [ ] **Step 1: Write the failing unit test.** Create `tests/unit/test_eval_stats.py`:

```python
from rag.eval.stats import bootstrap_ci


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    mean, low, high = bootstrap_ci(values, n_resamples=2000, confidence=0.95, seed=7)
    assert abs(mean - 0.7) < 1e-9
    assert low <= mean <= high
    assert 0.0 <= low <= high <= 1.0
    # same seed → identical interval
    assert bootstrap_ci(values, n_resamples=2000, confidence=0.95, seed=7) == (mean, low, high)


def test_bootstrap_ci_empty_is_zeros() -> None:
    assert bootstrap_ci([], n_resamples=100) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_single_value_has_zero_width() -> None:
    mean, low, high = bootstrap_ci([0.5], n_resamples=100, seed=1)
    assert mean == low == high == 0.5
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest tests/unit/test_eval_stats.py -q` → FAIL.

- [ ] **Step 3: Implement.** Create `src/rag/eval/stats.py`:

```python
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point_mean, ci_low, ci_high) via percentile bootstrap of the mean.

    Deterministic given ``seed`` (so eval runs are reproducible and CI bands are
    stable — spec §15). Empty input → all zeros; a single value → zero-width CI.
    """
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    point = float(arr.mean())
    if arr.size == 1:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low = float(np.percentile(means, tail * 100))
    high = float(np.percentile(means, (1.0 - tail) * 100))
    return (point, low, high)
```

- [ ] **Step 4: Run + commit.** `uv run pytest tests/unit/test_eval_stats.py -q` → PASS; `uv run mypy src` → clean. Commit:

```bash
git add src/rag/eval/stats.py tests/unit/test_eval_stats.py
git commit -m "feat(eval): seeded percentile bootstrap confidence intervals"
```

---

### Task 16: Custom LLM judge — faithfulness + answer-relevance

**Files:**
- Create: `src/rag/eval/prompts/__init__.py` (empty), `src/rag/eval/prompts/judge_v1.md`
- Create: `src/rag/eval/judge.py`
- Test: `tests/unit/test_judge.py`

- [ ] **Step 1: Write the failing unit test.** Create `tests/unit/test_judge.py`:

```python
from rag.eval.judge import FaithfulnessJudge, JudgeScore


class _CannedLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls: list[list[dict]] = []

    def complete(self, messages, **opts):
        from rag.models import Completion

        self.calls.append(messages)
        return Completion(text=self._reply, usage={})

    def stream(self, messages, **opts):  # unused by the judge
        yield ""


def test_judge_parses_scores_from_json_reply() -> None:
    llm = _CannedLLM('Here is my rating: {"faithfulness": 0.8, "answer_relevance": 0.6}')
    score = FaithfulnessJudge(llm).score("q?", "answer", "context")
    assert isinstance(score, JudgeScore)
    assert score.faithfulness == 0.8
    assert score.answer_relevance == 0.6
    assert "context" in llm.calls[0][0]["content"]


def test_judge_clamps_and_defaults_on_unparseable_reply() -> None:
    score = FaithfulnessJudge(_CannedLLM("the model rambled with no json")).score("q", "a", "c")
    assert score.faithfulness == 0.0
    assert score.answer_relevance == 0.0
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest tests/unit/test_judge.py -q` → FAIL.

- [ ] **Step 3: Write the judge prompt.** Create `src/rag/eval/prompts/__init__.py` (empty) and `src/rag/eval/prompts/judge_v1.md`:

```markdown
You are a strict evaluation judge. Rate the ANSWER against the CONTEXT and QUESTION.

Definitions:
- faithfulness: fraction of the answer's claims that are directly supported by the context (0.0–1.0). Unsupported or contradicted claims lower it.
- answer_relevance: how well the answer addresses the question, ignoring support (0.0–1.0).

Respond with ONLY a JSON object and nothing else, e.g.:
{"faithfulness": 0.9, "answer_relevance": 0.8}

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
```

- [ ] **Step 4: Implement the judge.** Create `src/rag/eval/judge.py`:

```python
from __future__ import annotations

import json
import re
from importlib import resources

from pydantic import BaseModel

from rag.protocols import LLMProvider

_JSON = re.compile(r"\{[^{}]*\}", re.DOTALL)


class JudgeScore(BaseModel):
    faithfulness: float = 0.0
    answer_relevance: float = 0.0


def _load_prompt() -> str:
    return (
        resources.files("rag.eval.prompts").joinpath("judge_v1.md").read_text(encoding="utf-8")
    )


class FaithfulnessJudge:
    """Cheap-model judge (temperature 0) scoring faithfulness + answer-relevance.

    Routes through the same ``LLMProvider.complete`` seam as the rest of the system
    so the judge model is config-driven and pinned. Unparseable replies default to
    0.0 (a non-answer is not credited) — see spec §15 on determinism.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._prompt = _load_prompt()

    def score(self, question: str, answer: str, context: str) -> JudgeScore:
        content = self._prompt.format(question=question, context=context, answer=answer)
        reply = self._llm.complete([{"role": "user", "content": content}]).text
        match = _JSON.search(reply)
        if not match:
            return JudgeScore()
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return JudgeScore()
        return JudgeScore(
            faithfulness=_clamp(data.get("faithfulness", 0.0)),
            answer_relevance=_clamp(data.get("answer_relevance", 0.0)),
        )


def _clamp(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
```

- [ ] **Step 5: Run + commit.** `uv run pytest tests/unit/test_judge.py -q` → PASS; `uv run mypy src` → clean. Commit:

```bash
git add src/rag/eval/prompts src/rag/eval/judge.py tests/unit/test_judge.py
git commit -m "feat(eval): custom faithfulness/answer-relevance judge via LLMProvider"
```

---

### Task 17: Golden set + loader + scorecard renderer

**Files:**
- Create: `eval/golden_set.yaml`
- Create: `src/rag/eval/golden.py`, `src/rag/eval/scorecard.py`
- Test: `tests/unit/test_golden.py`, `tests/unit/test_scorecard.py`

- [ ] **Step 1: Write the failing loader + scorecard tests.**

  Create `tests/unit/test_golden.py`:
```python
from pathlib import Path

from rag.eval.golden import GoldenItem, load_golden_set


def test_load_golden_set_parses_items(tmp_path: Path) -> None:
    yaml_text = """
items:
  - id: q1
    question: Does nitrogen increase maize yield?
    reference_answer: Yes, nitrogen raises maize yield.
    relevant_doc_ids: ["maize_nitrogen"]
    stratum: easy
  - id: q2
    question: What is the capital of France?
    reference_answer: This corpus does not cover that.
    relevant_doc_ids: []
    stratum: out_of_corpus
"""
    path = tmp_path / "g.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    items = load_golden_set(path)
    assert len(items) == 2
    assert isinstance(items[0], GoldenItem)
    assert items[0].id == "q1"
    assert items[0].relevant_doc_ids == ["maize_nitrogen"]
    assert items[1].stratum == "out_of_corpus"


def test_committed_golden_set_is_valid_and_stratified() -> None:
    items = load_golden_set(Path("eval/golden_set.yaml"))
    assert len(items) >= 6
    strata = {i.stratum for i in items}
    assert {"easy", "adversarial", "out_of_corpus"} <= strata
```

  Create `tests/unit/test_scorecard.py`:
```python
from rag.eval.scorecard import StratumScore, render_scorecard


def test_render_scorecard_has_separated_retrieval_and_answer_sections() -> None:
    scores = [
        StratumScore(
            stratum="easy", n=3,
            hit_at_k=(0.67, 0.40, 0.90), mrr=(0.55, 0.30, 0.80), ndcg=(0.60, 0.35, 0.85),
            faithfulness=(0.80, 0.60, 0.95), answer_relevance=(0.75, 0.55, 0.90),
        ),
    ]
    text = render_scorecard(scores, k=5)
    assert "RETRIEVAL" in text and "ANSWER" in text
    assert "easy" in text
    assert "hit@5" in text.lower() or "hit@k" in text.lower()
    assert "0.67" in text                     # point estimate rendered
    assert "0.40" in text and "0.90" in text  # CI bounds rendered
```

- [ ] **Step 2: Run, see them fail.** `uv run pytest tests/unit/test_golden.py tests/unit/test_scorecard.py -q` → FAIL.

- [ ] **Step 3: Implement the loader.** Create `src/rag/eval/golden.py`:

```python
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
```

- [ ] **Step 4: Implement the scorecard renderer.** Create `src/rag/eval/scorecard.py`:

```python
from __future__ import annotations

from pydantic import BaseModel

CI = tuple[float, float, float]  # (point, low, high)


class StratumScore(BaseModel):
    stratum: str
    n: int
    hit_at_k: CI
    mrr: CI
    ndcg: CI
    faithfulness: CI
    answer_relevance: CI


def _fmt(ci: CI) -> str:
    point, low, high = ci
    return f"{point:.2f} [{low:.2f},{high:.2f}]"


def render_scorecard(scores: list[StratumScore], k: int) -> str:
    """Render separated RETRIEVAL and ANSWER tables, per stratum, with 95% CIs.

    The first row is the ``ALL`` aggregate (the only statistically useful CI at
    smoke-eval scale). Per-stratum rows show point [low,high] too, but at small n
    those intervals are directional — read ``ALL`` for the headline.
    """
    lines: list[str] = []
    lines.append("(ALL = aggregate over every item; per-stratum CIs are directional at small n)")
    lines.append("=== RETRIEVAL (no LLM judge) ===")
    lines.append(f"{'stratum':<14}{'n':>4}  {f'hit@{k}':<20}{'mrr':<20}{'ndcg@'+str(k):<20}")
    for s in scores:
        lines.append(
            f"{s.stratum:<14}{s.n:>4}  {_fmt(s.hit_at_k):<20}{_fmt(s.mrr):<20}{_fmt(s.ndcg):<20}"
        )
    lines.append("")
    lines.append("=== ANSWER (LLM-judged, temp 0) ===")
    lines.append(f"{'stratum':<14}{'n':>4}  {'faithfulness':<20}{'answer_relevance':<20}")
    for s in scores:
        lines.append(
            f"{s.stratum:<14}{s.n:>4}  {_fmt(s.faithfulness):<20}{_fmt(s.answer_relevance):<20}"
        )
    return "\n".join(lines)
```

- [ ] **Step 5: Seed the committed golden set.** Create `eval/golden_set.yaml` — **7 smoke items** spanning easy/ambiguous/adversarial/out_of_corpus (the full 30–50 stratified set is a Phase-1 deliverable). `relevant_doc_ids` are **corpus filename stems**; the runner matches them via the stem parsed from each chunk's `source_uri` (Task 18, `stem_from_uri`), so they line up with the rendered `<stem>.pdf` docs:

```yaml
# SMOKE-EVAL golden set for the frozen mini-corpus in eval/corpus/*.md.
# Purpose in P0b: prove the harness end-to-end. It is intentionally small, so
# per-stratum confidence intervals are NOT meaningful — read the aggregate "ALL"
# row (Task 18). The statistically-honest 30-50 item stratified set (with a human
# calibration slice) is a Phase-1 deliverable, gated on the real corpus (spec §19).
# relevant_doc_ids reference the corpus filename stem (e.g. "maize_nitrogen.md" -> "maize_nitrogen").
# NOTE: no "metadata" stratum here — metadata-filtered retrieval is Phase 4, so a
# metadata row would mislabel an unfiltered query as if filtering were tested.
items:
  - id: q1
    question: How does nitrogen fertilizer affect maize yield?
    reference_answer: Nitrogen fertilizer substantially increases maize yield up to an optimal rate.
    relevant_doc_ids: ["maize_nitrogen"]
    stratum: easy
  - id: q2
    question: What causes yield loss in wheat during drought?
    reference_answer: Drought during grain filling reduces wheat yield by shortening the filling period.
    relevant_doc_ids: ["wheat_drought"]
    stratum: easy
  - id: q3
    question: Why is phosphorus important early in the season?
    reference_answer: Phosphorus supports early root development and establishment.
    relevant_doc_ids: ["phosphorus_roots"]
    stratum: easy
  - id: q4
    question: Should I apply more nitrogen than the optimal rate to be safe?
    reference_answer: No; beyond the optimal rate yield gains plateau while cost and losses rise.
    relevant_doc_ids: ["maize_nitrogen"]
    stratum: ambiguous
  - id: q5
    question: Does irrigation timing interact with nitrogen uptake?
    reference_answer: Adequate soil moisture is needed for nitrogen uptake; timing matters.
    relevant_doc_ids: ["irrigation_nitrogen"]
    stratum: ambiguous
  - id: q6
    question: Is it true that adding potassium cures all drought damage in wheat?
    reference_answer: No; the context does not support potassium curing drought damage.
    relevant_doc_ids: ["wheat_drought"]
    stratum: adversarial
  - id: q7
    question: What is the population of Tokyo?
    reference_answer: I don't have relevant context to answer that.
    relevant_doc_ids: []
    stratum: out_of_corpus
```

- [ ] **Step 6: Run + commit.** `uv run pytest tests/unit/test_golden.py tests/unit/test_scorecard.py -q` → PASS; `uv run mypy src` → clean. Commit:

```bash
git add eval/golden_set.yaml src/rag/eval/golden.py src/rag/eval/scorecard.py tests/unit/test_golden.py tests/unit/test_scorecard.py
git commit -m "feat(eval): golden-set loader, stratified starter set, scorecard renderer"
```

---

### Task 18: Frozen mini-corpus + eval runner + `rag-eval` CLI

**Files:**
- Create: `eval/corpus/*.md` (5 documents)
- Create: `src/rag/eval/corpus.py`, `src/rag/eval/runner.py`
- Modify: `pyproject.toml` (`[project.scripts]` add `rag-eval`)
- Test: `tests/unit/test_eval_runner.py`

- [ ] **Step 1: Author the frozen corpus.** Create five Markdown files under `eval/corpus/` with factual, non-copyrighted agronomy content. Filenames (stems) must match the golden set's `relevant_doc_ids`. Minimum content (one short paragraph each is fine; keep them deterministic):

  `eval/corpus/maize_nitrogen.md`:
```markdown
# Nitrogen and Maize Yield

Nitrogen fertilizer substantially increases maize yield up to an optimal rate.
Beyond that optimal rate, additional nitrogen yields diminishing returns while
raising input cost and the risk of leaching losses. Maize in the South region
shows a particularly strong nitrogen response.
```

  `eval/corpus/wheat_drought.md`:
```markdown
# Drought Stress in Wheat

Prolonged drought stress during grain filling reduces wheat yield by shortening
the grain-filling period and lowering kernel weight. Potassium does not cure
drought damage; water availability is the limiting factor.
```

  `eval/corpus/phosphorus_roots.md`:
```markdown
# Phosphorus and Root Development

Phosphorus supports early root development and seedling establishment. Adequate
early-season phosphorus availability improves nutrient and water capture later
in the season.
```

  `eval/corpus/irrigation_nitrogen.md`:
```markdown
# Irrigation Timing and Nitrogen Uptake

Nitrogen uptake depends on adequate soil moisture. Irrigation scheduling that
maintains moisture during peak demand improves nitrogen uptake efficiency;
poorly timed irrigation can reduce it.
```

  `eval/corpus/potassium_basics.md`:
```markdown
# Potassium Basics

Potassium regulates water balance and enzyme activation in crops. It is not a
remedy for drought damage but supports overall plant water relations.
```

- [ ] **Step 2: Write the failing runner test.** The runner is unit-tested with fakes (no DB, no network): a fake retriever returns deterministic ids per question, a fake judge returns fixed scores. Create `tests/unit/test_eval_runner.py`:

```python
from rag.eval.golden import GoldenItem
from rag.eval.runner import evaluate
from rag.eval.scorecard import StratumScore


class _RetrieverByQuestion:
    """Returns chunk ids keyed by question id baked into the query string."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    def retrieve(self, query, k, filt):
        from rag.models import Provenance, ScoredChunk
        from tests.unit.fakes import make_chunk

        ids = self._mapping.get(query, [])
        return [
            ScoredChunk(chunk=make_chunk(doc_id, text=doc_id), score=1.0 / (i + 1),
                        provenance=Provenance.dense)
            for i, doc_id in enumerate(ids)
        ]


class _FixedJudge:
    def score(self, question, answer, context):
        from rag.eval.judge import JudgeScore

        return JudgeScore(faithfulness=0.9, answer_relevance=0.8)


class _FixedAnswerer:
    def answer(self, query, filt=None, scope=None):
        from rag.models import Answer

        return Answer(text="grounded answer [1].")


def test_evaluate_produces_per_stratum_scores() -> None:
    items = [
        GoldenItem(id="q1", question="q1", reference_answer="", relevant_doc_ids=["maize_nitrogen"], stratum="easy"),
        GoldenItem(id="q2", question="q2", reference_answer="", relevant_doc_ids=[], stratum="out_of_corpus"),
    ]
    retriever = _RetrieverByQuestion({"q1": ["maize_nitrogen", "x"], "q2": ["y"]})
    scores = evaluate(
        items=items,
        retriever=retriever,
        answerer=_FixedAnswerer(),
        judge=_FixedJudge(),
        k=5,
        seed=0,
        doc_key=lambda sc: sc.chunk.doc_id,   # test ids are stems on doc_id; live uses source_uri stem
    )
    by_stratum = {s.stratum: s for s in scores}
    assert isinstance(by_stratum["easy"], StratumScore)
    assert by_stratum["easy"].hit_at_k[0] == 1.0          # q1 retrieved a relevant doc
    assert by_stratum["easy"].faithfulness[0] == 0.9
    assert by_stratum["out_of_corpus"].n == 1
    assert by_stratum["ALL"].n == 2                        # aggregate pools every item
```

- [ ] **Step 3: Run, see it fail.** `uv run pytest tests/unit/test_eval_runner.py -q` → FAIL.

- [ ] **Step 4: Implement the corpus renderer.** Create `src/rag/eval/corpus.py`:

```python
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF


def render_corpus_to_pdfs(corpus_dir: Path, out_dir: Path) -> list[Path]:
    """Render each eval/corpus/*.md into a one-page PDF named <stem>.pdf.

    Keeps the eval corpus diffable as text while exercising the real PDF pipeline.
    The doc_id assigned at ingest equals the file stem (see runner), which the
    golden set's relevant_doc_ids reference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_paths: list[Path] = []
    for md_path in sorted(corpus_dir.glob("*.md")):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), md_path.read_text(encoding="utf-8"))
        pdf_path = out_dir / f"{md_path.stem}.pdf"
        doc.save(str(pdf_path))
        doc.close()
        pdf_paths.append(pdf_path)
    return pdf_paths
```

- [ ] **Step 5: Implement the runner + CLI.** Create `src/rag/eval/runner.py`:

```python
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import unquote, urlparse

import structlog

from rag.eval.golden import GoldenItem, load_golden_set
from rag.eval.judge import FaithfulnessJudge, JudgeScore
from rag.eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank
from rag.eval.scorecard import StratumScore, render_scorecard
from rag.eval.stats import bootstrap_ci
from rag.models import Answer, ScoredChunk
from rag.protocols import Retriever

if TYPE_CHECKING:  # for annotations only — avoids importing heavy modules at CLI start
    from rag.config import Settings

log = structlog.get_logger()


class _Answerer(Protocol):  # evaluate only needs answer(question) -> Answer
    def answer(self, query: str) -> Answer: ...


class _Judge(Protocol):
    def score(self, question: str, answer: str, context: str) -> JudgeScore: ...


def stem_from_uri(scored: ScoredChunk) -> str:
    """Map a ScoredChunk to its corpus key = the filename stem of its source_uri.

    Ingestion sets ``doc_id = uuid5(NAMESPACE_URL, uri)`` (a hash), but the golden
    set labels docs by filename stem (e.g. ``maize_nitrogen``). The stem is
    recoverable from the chunk's ``source_uri`` (a ``file://`` URL), so retrieval
    is scored against stems on both sides — no id translation table needed.
    """
    path = unquote(urlparse(scored.chunk.source_uri).path)
    return Path(path).stem


def evaluate(
    items: list[GoldenItem],
    retriever: Retriever,
    answerer: _Answerer,
    judge: _Judge,
    k: int,
    seed: int = 0,
    doc_key: Callable[[ScoredChunk], str] = stem_from_uri,
) -> list[StratumScore]:
    """Score every golden item, aggregate per stratum with bootstrap CIs.

    Retrieval metrics compare ``doc_key(scored_chunk)`` (default: source_uri stem)
    against the golden ``relevant_doc_ids`` (also stems). Answer metrics use the
    judge over the generated answer. Out-of-corpus items (no relevant ids) are
    scored on answer metrics only; retrieval metrics are skipped for them (there
    is nothing relevant to hit), so the judge still evaluates the "I don't have
    relevant context" behaviour.
    """
    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"hit": [], "mrr": [], "ndcg": [], "faith": [], "rel": []}
    )
    for item in items:
        scored = retriever.retrieve(item.question, k, None)
        retrieved_ids = [doc_key(s) for s in scored]
        relevant = set(item.relevant_doc_ids)

        answer = answerer.answer(item.question)
        context_text = "\n".join(s.chunk.text for s in scored)
        verdict = judge.score(item.question, answer.text, context_text)

        b = buckets[item.stratum]
        if relevant:  # retrieval is only meaningful when there ARE relevant docs
            b["hit"].append(hit_at_k(retrieved_ids, relevant, k))
            b["mrr"].append(reciprocal_rank(retrieved_ids, relevant))
            b["ndcg"].append(ndcg_at_k(retrieved_ids, relevant, k))
        b["faith"].append(verdict.faithfulness)
        b["rel"].append(verdict.answer_relevance)

    def _score(label: str, b: dict[str, list[float]]) -> StratumScore:
        return StratumScore(
            stratum=label,
            n=len(b["faith"]),
            hit_at_k=bootstrap_ci(b["hit"], seed=seed),
            mrr=bootstrap_ci(b["mrr"], seed=seed),
            ndcg=bootstrap_ci(b["ndcg"], seed=seed),
            faithfulness=bootstrap_ci(b["faith"], seed=seed),
            answer_relevance=bootstrap_ci(b["rel"], seed=seed),
        )

    # Pool every item into an "ALL" aggregate. At smoke-eval scale per-stratum n is
    # tiny (1-2), so per-stratum CIs are near-degenerate; the ALL row is the only
    # statistically useful interval and is what Task 20's RESULTS headlines.
    pooled: dict[str, list[float]] = {key: [] for key in ("hit", "mrr", "ndcg", "faith", "rel")}
    for b in buckets.values():
        for key in pooled:
            pooled[key].extend(b[key])

    per_stratum = sorted((_score(s, b) for s, b in buckets.items()), key=lambda s: s.stratum)
    return [_score("ALL", pooled), *per_stratum]


def _aggregate(scores: list[StratumScore]) -> StratumScore | None:
    return next((s for s in scores if s.stratum == "ALL"), None)


def _rerank_delta(baseline: list[StratumScore], reranked: list[StratumScore]) -> str:
    """One-line rerank-lift summary on the ALL aggregate (spec §7 deliverable)."""
    b, r = _aggregate(baseline), _aggregate(reranked)
    if b is None or r is None:
        return "RERANK LIFT: n/a"
    df = r.faithfulness[0] - b.faithfulness[0]
    dn = r.ndcg[0] - b.ndcg[0]
    return (
        f"RERANK LIFT (ALL, n={r.n}): "
        f"faithfulness {b.faithfulness[0]:.2f} -> {r.faithfulness[0]:.2f} ({df:+.2f}); "
        f"ndcg {b.ndcg[0]:.2f} -> {r.ndcg[0]:.2f} ({dn:+.2f})"
    )


def _build_answerer(retriever: Retriever, settings: Settings) -> _Answerer:
    from rag.generation.assembler import TokenBudgetAssembler
    from rag.generation.streaming import StreamingAnswerer
    from rag.providers.llm import LiteLLMProvider

    return StreamingAnswerer(
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


def _run_live(golden_path: Path, corpus_dir: Path) -> str:
    """Wire real components, ingest the frozen corpus into the DEDICATED eval DB,
    then run a smoke eval — baseline (hybrid) vs hybrid+rerank A/B.

    Manual ``uv run rag-eval`` path; hits real Gemini/Cohere and costs money. Unit
    tests cover ``evaluate`` with fakes; this is the glue.
    """
    import tempfile

    from sqlalchemy import text as sa_text

    from rag.config import apply_provider_env, get_settings
    from rag.db import get_engine, run_migrations
    from rag.eval.corpus import render_corpus_to_pdfs
    from rag.ingestion.pipeline import IngestionPipeline
    from rag.ingestion.repository import PgChunkRepository
    from rag.providers.embeddings import LiteLLMEmbeddingProvider
    from rag.providers.llm import LiteLLMProvider
    from rag.providers.rerank import CohereReranker
    from rag.retrieval.dense import DenseRetriever
    from rag.retrieval.hybrid import HybridRetriever
    from rag.retrieval.lexical import LexicalRetriever
    from rag.retrieval.reranked import RerankedRetriever

    settings = get_settings()
    apply_provider_env(settings)

    # ISOLATION: eval writes its OWN database; it must never touch the app/dev DB.
    eval_url = settings.eval_database_url
    if not eval_url:
        raise SystemExit("EVAL_DATABASE_URL is unset — create a dedicated rag_eval DB first.")
    if eval_url == settings.database_url:
        raise SystemExit("EVAL_DATABASE_URL must differ from DATABASE_URL (no app-DB pollution).")

    reset = get_engine(eval_url)                 # reset the eval DB to empty, then migrate
    with reset.begin() as conn:
        conn.execute(sa_text("DROP SCHEMA IF EXISTS public CASCADE"))
        conn.execute(sa_text("CREATE SCHEMA public"))
    reset.dispose()
    run_migrations(eval_url)

    engine = get_engine(eval_url)
    embedder = LiteLLMEmbeddingProvider(model=settings.embedding_model, dim=settings.embedding_dim)
    with tempfile.TemporaryDirectory() as tmp:
        pdfs = render_corpus_to_pdfs(corpus_dir, Path(tmp))
        IngestionPipeline(
            repository=PgChunkRepository(engine),
            embedder=embedder,
            chunk_tokens=settings.chunk_tokens,
            overlap=settings.chunk_overlap,
        ).ingest_paths(pdfs)

    hybrid = HybridRetriever(
        dense=DenseRetriever(engine=engine, embedder=embedder),
        lexical=LexicalRetriever(engine=engine),
        rrf_k=settings.rrf_k,
        candidate_k=settings.candidate_k,
    )
    judge = FaithfulnessJudge(LiteLLMProvider(model=settings.grader_model, max_tokens=256))
    items = load_golden_set(golden_path)
    k = settings.retrieval_k

    # A/B: rerank lift = swap the base retriever. Both answerers are rerank-agnostic.
    baseline = evaluate(items, hybrid, _build_answerer(hybrid, settings), judge, k=k)
    out = [
        "# SMOKE EVAL — harness validation (small n; read the ALL row).",
        "# The statistically-honest 30-50 item stratified set is a Phase-1 gate (spec §19).",
        "",
        "## Baseline: hybrid (dense + lexical, RRF), no rerank",
        render_scorecard(baseline, k=k),
    ]
    if settings.rerank_enabled and settings.cohere_api_key:
        reranked_retriever = RerankedRetriever(
            base=hybrid,
            reranker=CohereReranker(model=settings.rerank_model),
            candidate_k=settings.candidate_k,
        )
        reranked = evaluate(items, reranked_retriever, _build_answerer(reranked_retriever, settings), judge, k=k)
        out += [
            "",
            "## Hybrid + Cohere rerank",
            render_scorecard(reranked, k=k),
            "",
            _rerank_delta(baseline, reranked),
        ]
    else:
        out += ["", "(rerank A/B skipped — set RERANK_ENABLED=true + COHERE_API_KEY to measure lift)"]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG golden-set smoke eval (live).")
    parser.add_argument("--golden", type=Path, default=Path("eval/golden_set.yaml"))
    parser.add_argument("--corpus", type=Path, default=Path("eval/corpus"))
    args = parser.parse_args()
    print(_run_live(args.golden, args.corpus))


if __name__ == "__main__":
    main()
```

> **Label matching is already solved — no guesswork.** Ingestion sets `doc_id = uuid5(NAMESPACE_URL, raw.uri)` (see `src/rag/ingestion/parse.py:25`), a hash, **not** the filename. So retrieval is scored by the **filename stem parsed from each chunk's `source_uri`** via the default `doc_key=stem_from_uri`, which equals the corpus stem the golden set references (e.g. `maize_nitrogen`). Because `render_corpus_to_pdfs` names each PDF `<stem>.pdf` and the PDF adapter sets `source_uri = path.as_uri()`, the stems line up by construction. The unit test passes `doc_key=lambda sc: sc.chunk.doc_id` (its fake chunks carry stems as doc_ids); the live runner uses the default. No id-translation table, no pipeline change.

- [ ] **Step 6: Register the CLI.** In `pyproject.toml`, add under `[project.scripts]`:

```toml
rag-eval = "rag.eval.runner:main"
```

- [ ] **Step 7: Run + commit.** `uv run pytest tests/unit/test_eval_runner.py -q` → PASS; `uv run mypy src` → clean. Commit:

```bash
git add eval/corpus src/rag/eval/corpus.py src/rag/eval/runner.py pyproject.toml tests/unit/test_eval_runner.py
git commit -m "feat(eval): frozen mini-corpus, evaluate() runner, rag-eval CLI"
```

---

## Track F — Demo UI, docs, and live verification

### Task 19: Minimal static SSE chat UI (streaming + citations)

**Files:**
- Create: `ui/index.html`
- Modify: `src/rag/api/app.py` (mount the static UI), `src/rag/api/routes.py` (optional root redirect)
- Test: `tests/integration/test_ui_served.py`

- [ ] **Step 1: Write the failing integration test.** Create `tests/integration/test_ui_served.py`:

```python
import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.generation.assembler import TokenBudgetAssembler
from rag.generation.streaming import StreamingAnswerer
from tests.unit.fakes import FakeRetriever, FakeStreamingLLM, make_chunk

pytestmark = pytest.mark.integration


def test_ui_index_is_served() -> None:
    answerer = StreamingAnswerer(
        retriever=FakeRetriever([make_chunk("a", text="alpha")]),
        assembler=TokenBudgetAssembler(),
        llm=FakeStreamingLLM(),
        token_budget=1000,
        retrieval_k=3,
    )
    client = TestClient(create_app(answerer=answerer))
    resp = client.get("/ui/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "query/stream" in resp.text          # the page wires to the SSE endpoint
```

- [ ] **Step 2: Run, see it fail.** `uv run pytest -m integration tests/integration/test_ui_served.py -q` → FAIL.

- [ ] **Step 3: Write the page.** Create `ui/index.html` — a zero-build page that POSTs to `/query/stream` and renders tokens live, then attaches citations from the `done` event:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Production RAG — demo</title>
  <style>
    body { font: 15px/1.5 system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; }
    #answer { white-space: pre-wrap; border: 1px solid #ddd; border-radius: 8px; padding: 16px; min-height: 80px; }
    #cites { margin-top: 12px; color: #555; font-size: 13px; }
    input, button { font: inherit; padding: 8px; }
    input { width: 72%; }
  </style>
</head>
<body>
  <h1>Production RAG</h1>
  <form id="f">
    <input id="q" placeholder="Ask the corpus…" autocomplete="off" />
    <button>Ask</button>
  </form>
  <div id="answer"></div>
  <div id="cites"></div>
  <script>
    const f = document.getElementById("f");
    const ans = document.getElementById("answer");
    const cites = document.getElementById("cites");

    f.addEventListener("submit", async (e) => {
      e.preventDefault();
      ans.textContent = ""; cites.textContent = "";
      const resp = await fetch("/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: document.getElementById("q").value }),
      });
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop();                       // keep the trailing partial frame
        for (const frame of frames) {
          const evMatch = frame.match(/^event: (.*)$/m);
          const dataMatch = frame.match(/^data: (.*)$/m);
          if (!evMatch || !dataMatch) continue;
          const payload = JSON.parse(dataMatch[1]);
          if (evMatch[1] === "token") ans.textContent += payload.text;
          else if (evMatch[1] === "done") {
            ans.textContent = payload.answer;        // replace with validated text
            cites.textContent = (payload.citations || [])
              .map(c => `${c.marker} ${c.source_uri} (p.${c.page})`).join("  ·  ");
          } else if (evMatch[1] === "error") {
            cites.textContent = "Error: " + payload.message;
          }
        }
      }
    });
  </script>
</body>
</html>
```

- [ ] **Step 4: Mount the UI.** In `src/rag/api/app.py`, mount the static directory. Add imports and a mount inside `create_app` (after `include_router`):

```python
from pathlib import Path

from fastapi.staticfiles import StaticFiles

# inside create_app, after app.include_router(router):
    ui_dir = Path(__file__).resolve().parents[3] / "ui"
    if ui_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")
```

(`parents[3]` from `src/rag/api/app.py` → repo root; `ui/` is a top-level dir. The `html=True` flag serves `index.html` at `/ui/`.)

- [ ] **Step 5: Run, see it pass.** `uv run pytest -m integration tests/integration/test_ui_served.py -q` → PASS.

- [ ] **Step 6: Commit.**

```bash
git add ui/index.html src/rag/api/app.py tests/integration/test_ui_served.py
git commit -m "feat(ui): minimal static SSE chat page (streaming tokens + citations)"
```

---

### Task 20: README, `.env.example`, Makefile, and live verification

**Files:**
- Modify: `README.md`, `.env.example`, `Makefile`
- Test: none (docs) — verification is the live smoke run

- [ ] **Step 1: Extend `.env.example`** with the new placeholders (never real values):

```bash
# Hybrid retrieval
CANDIDATE_K=30                       # over-fetch pool per leg before fusion/rerank

# Rerank (Cohere via LiteLLM) — leave RERANK_ENABLED=false until a key is set
COHERE_API_KEY=
RERANK_ENABLED=false
RERANK_MODEL=cohere/rerank-english-v3.0

# Eval — DEDICATED database; must differ from DATABASE_URL (rag-eval refuses otherwise)
EVAL_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag_eval

# Observability (optional; absent keys disable the feature)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

Create the dedicated eval database once (pgAdmin → Databases → Create → `rag_eval`); the migration runs `CREATE EXTENSION IF NOT EXISTS vector` on it. The runner resets this DB's schema on every run, so it must never be the app DB.

- [ ] **Step 2: Add Makefile targets (with raw `uv` equivalents in comments)** for parity with the spec's `make` one-liners. Append to `Makefile`:

```makefile
serve:            ## run the API with the streaming UI at /ui/
	uv run uvicorn rag.api.app:create_app --factory --port 8000   # no-make: same line

eval:             ## run the live golden-set eval (needs GEMINI_API_KEY; costs money)
	uv run rag-eval                                               # no-make: same line

obs-up:           ## optional Prometheus+Grafana+Langfuse dashboards
	docker compose -f docker-compose.observability.yml up -d
```

- [ ] **Step 3: Rewrite the README RESULTS + features sections.** Update `README.md` to document P0b. Add/replace these sections (keep P0a quickstart, extend it):
  - **Architecture:** note hybrid (dense+lexical→RRF), rerank, SSE streaming, observability, eval.
  - **Honesty callouts (spec §8):** Postgres `ts_rank` is tf-idf-ish, not BM25 (RRF ignores absolute scores; ParadeDB `pg_search` is the BM25 upgrade path); pgvector's 2000-d HNSW limit and the 768-d MRL choice.
  - **RESULTS scorecard (honest framing):** report the **aggregate `ALL` row with its CI** as the headline (per-stratum CIs are directional at smoke-eval n — say so in one line), the **rerank-lift line** from the A/B (`faithfulness X→Y`, `ndcg A→B`), and the **per-stage p95 table** (dense/lexical/fusion/rerank/assemble/generate) from `/metrics`. State plainly that the 30–50 item statistically-honest set is a Phase-1 deliverable. Leave numeric cells as `_pending live run_` until Step 5 fills them — a real placeholder to resolve in this task, not a shipped TODO.
  - **Quickstart additions:**
    ```powershell
    uv run alembic upgrade head
    uv run rag-ingest <your-pdf-dir>
    uv run uvicorn rag.api.app:create_app --factory --port 8000
    # open http://localhost:8000/ui/  → streaming chat with citations
    # metrics at http://localhost:8000/metrics
    uv run rag-eval        # live scorecard (needs GEMINI_API_KEY)
    ```

- [ ] **Step 4: Full green check.** Run the whole suite:
```bash
uv run pytest -m "not integration and not live" -q      # unit
uv run pytest -m integration -q                          # integration (local Postgres)
uv run ruff check . ; uv run mypy src
```
Expected: all PASS, ruff clean, mypy clean.

- [ ] **Step 5: Live verification (the P0b "Done when").** With real keys + `EVAL_DATABASE_URL` in `.env`:
  1. `uv run alembic upgrade head` (applies migration 0002 to the app DB).
  2. Start the server; open `http://localhost:8000/ui/`; confirm a query **streams tokens** then shows validated citations.
  3. `curl http://localhost:8000/metrics` → confirm `rag_stage_latency_seconds` has samples for `dense`, `lexical`, `fusion`, `assemble`, `generate` (and `rerank` once enabled); read off each stage's p95.
  4. `RERANK_ENABLED=true` + `COHERE_API_KEY`, then `uv run rag-eval` → confirm it (a) targets `rag_eval`, not the app DB, (b) prints **separated** RETRIEVAL/ANSWER tables with an `ALL` aggregate, and (c) prints the **RERANK LIFT** line (the spec-promised before/after number).
  5. Paste the `ALL`-row numbers, the rerank-lift line, and the per-stage p95 into the README RESULTS section.

- [ ] **Step 6: Commit.**

```bash
git add README.md .env.example Makefile
git commit -m "docs(p0b): README results scorecard, streaming/obs/eval quickstart, honesty callouts"
```

---

## Self-Review (performed while writing this plan)

**1. Spec coverage (spec §7 "P0b — Hybrid, rerank, streaming, observability, eval"):**

| P0b requirement (spec §7) | Task(s) |
|---|---|
| Postgres FTS + RRF hybrid | 2 (tsvector), 3 (lexical, english-frozen), 4 (RRF), 5 (hybrid + candidate_k over-fetch) |
| Cohere rerank (`litellm.rerank`; bge swap) | 6 (reranker), 7 (`RerankedRetriever` decorator), 10 (wired when enabled) |
| **Rerank lift measured** (spec §7 "before/after number") | 18 (`_run_live` A/B: hybrid vs hybrid+rerank → `RERANK LIFT` line) |
| SSE streaming | 8 (stream API), 9 (StreamingAnswerer), 10 (SSE endpoint), 19 (demo UI) |
| Langfuse traces | 12 |
| Prometheus `/metrics` + Grafana | 11 (metrics), 13 (dashboards/compose) |
| Golden set (smoke now; 30–50 = Phase 1) | 17 (loader + 7-item smoke set, honestly framed) |
| `make eval` scorecard, retrieval+answer **separated**, **bootstrap CIs** | 14 (metrics), 15 (CIs), 16 (judge), 17 (scorecard + ALL aggregate), 18 (runner/CLI), 20 (`make eval`) |
| per-stage p95 captured (dense/lexical/fusion/rerank/assemble/generate) | 11 (`observe_stage` in hybrid/reranked/streaming), 20 (read off `/metrics`) |
| Eval isolation (no app-DB pollution) | 18 (`EVAL_DATABASE_URL`; runner refuses if unset or == app DB) |
| "Done when": streaming cited answers; separated scorecards w/ CIs; Langfuse+Grafana render | 10+19 (stream), 18+20 (eval), 12+13 (dashboards) |

**Honesty callouts (spec §8):** ts_rank-not-BM25 (Task 3 docstring + Task 20 README), pgvector 2000-d/MRL (Task 20 README), smoke-eval CIs directional / ALL-row headline (Tasks 17/18/20). **Cost seam (spec §8.4):** judge uses the cheap `grader_model` (Task 18). **Failure modes (spec §16):** mid-stream error → flush + error event (Task 9). All present.

**Deferred-by-design (NOT in this plan, correctly):** online sampled quality scoring and the eval-in-CI GitHub Action are **Phase 1**, not P0b (spec §7) — this plan makes `rag-eval` runnable locally and leaves the seam. P0c parser hardening (Docling/OCR) is separate. The statistically-honest 30–50 item stratified set + human calibration slice is a **Phase-1** deliverable gated on the real corpus (spec §19); P0b ships a 7-item smoke set and is explicit that per-stratum CIs are directional at this n (no overselling). The `metadata` stratum is intentionally **absent** — metadata-filtered retrieval is Phase 4.

**2. Placeholder scan:** No "TODO/TBD/implement later" in code steps; every code step shows complete code. One *intentional* resolve-in-task marker, explicit: the README RESULTS cells (`_pending live run_`, filled in Task 20 Step 5). The eval label-matching (doc_id vs stem) is fully resolved in-plan via `stem_from_uri` (Task 18) — no implementation-time guesswork. The Task 10 `configure_observability` ordering note gives an explicit stub-or-reorder instruction.

**3. Type/name consistency (checked across tasks):** `Retriever.retrieve(query, k, filt) -> list[ScoredChunk]` — the one seam every retriever implements: `DenseRetriever`/`LexicalRetriever`/`HybridRetriever` (Task 5) and `RerankedRetriever` (Task 7), all consumed identically by `StreamingAnswerer` (Task 9) and `evaluate` (Task 18). `Reranker.rerank(query, chunks, top_n) -> list[ScoredChunk]` — Task 6 protocol matches `CohereReranker`/`FakeReranker` and the `RerankedRetriever` call `reranker.rerank(query, chunks, top_n=k)` (Task 7). `HybridRetriever(dense, lexical, rrf_k=60, candidate_k=30)` and `RerankedRetriever(base, reranker, candidate_k=30)` — defs (Tasks 5/7) match construction in Tasks 10/18. `AnswerEvent = TokenEvent | DoneEvent | ErrorEvent` with `.type` discriminator + `.model_dump_json()` — defined Task 8, consumed Tasks 9/10/19. `StreamingAnswerer(retriever, assembler, llm, token_budget, retrieval_k)` — **no rerank params** (rerank is in the retriever); identical in Tasks 9 (def), 10/18/19 (construction). `StratumScore` fields (`stratum,n,hit_at_k,mrr,ndcg,faithfulness,answer_relevance` as `(point,low,high)` triples) — identical in Tasks 17 (def) and 18 (construction); `evaluate` returns an `ALL` row first (Task 18) consumed by `_rerank_delta` + the renderer. `observe_stage(stage)` + `STAGE_LATENCY`/`REQUESTS`/`render_latest` — Task 11 def matches its usage in hybrid/reranked/streaming. `bootstrap_ci(values, n_resamples, confidence, seed) -> (point, low, high)` — Task 15 def matches Task 18 calls. No mismatches found.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-03-p0b-hybrid-rerank-streaming-obs-eval.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — a fresh subagent per task, two-stage review between tasks, fast iteration. Matches how P0a was built. **REQUIRED SUB-SKILL:** superpowers:subagent-driven-development.

**2. Inline Execution** — execute tasks in this session with checkpoints. **REQUIRED SUB-SKILL:** superpowers:executing-plans.

**Suggested execution order note:** implement Task 12 (`configure_observability`) before Task 10's app rewrite, OR drop the temporary no-op stub named in Task 10 Step 4, to keep the tree importable at every commit. Everything else runs in numeric order.

**Which approach?**
