# Production-Grade RAG System

Hybrid-RAG over a single Postgres + pgvector store, every model call through LiteLLM
(Gemini default). **P0b** ships hybrid retrieval, optional Cohere cross-encoder rerank,
SSE token streaming, Prometheus/Langfuse observability, and a golden-set eval harness.
P0a delivered the walking skeleton: PDF ingest → dense retrieval → cited, grounded answers.

> Full design spec: [docs/superpowers/specs/2026-06-02-production-rag-system-design.md](docs/superpowers/specs/2026-06-02-production-rag-system-design.md)

---

## Quickstart (local — primary path)

Prerequisites: Python 3.12+, `uv`, and a local PostgreSQL 18 + pgvector on `:5432`.

```powershell
# 1. In pgAdmin, create databases: rag_db, rag_test, rag_eval
# 2. Copy and fill the env file
cp .env.example .env          # fill GEMINI_API_KEY + your Postgres password

uv sync
uv run alembic upgrade head                                   # build schema (+ CREATE EXTENSION vector)
uv run rag-ingest <your-pdf-dir>                              # ingest PDFs
uv run uvicorn rag.api.app:create_app --factory --port 8000   # start the API

# Streaming chat with citations:   http://localhost:8000/ui/
# Prometheus metrics endpoint:     http://localhost:8000/metrics
# Health check:                    http://localhost:8000/healthz

uv run rag-eval   # live smoke eval (needs GEMINI_API_KEY + a dedicated rag_eval DB)
```

Or with `make`:

```bash
make migrate
make serve                    # start API on :8000
make ingest CORPUS=./data/raw
make query Q="does nitrogen help maize?"
make eval                     # live golden-set eval (costs money)
make obs-up                   # optional Prometheus + Grafana + Langfuse stack
```

`POST /query` returns JSON: a grounded `answer`, `citations` (doc/chunk/page/char span),
`usage` (tokens, latency), and a `trace_id`. `POST /query/stream` streams the same answer
as Server-Sent Events and emits validated citations only after generation completes.

---

## Architecture (P0b)

```
PDF → parse → clean → fixed-token chunk → embed (gemini-embedding-001, 768-d)
    → Postgres + pgvector (HNSW) + FTS tsvector column
                      │
          ┌───────────┴───────────┐
          │                       │
   dense KNN (pgvector)    lexical FTS (ts_rank)
          │                       │
          └────── RRF fusion ─────┘
                      │
              [optional] Cohere cross-encoder rerank
                      │
           token-budget assembler → numbered context
                      │
           gemini-2.5-pro (streaming or batch)
                      │
           citation validator → strip fabricated refs
                      │
                  answer + citations
```

**Write path:** `rag-ingest` parses PDFs (PyMuPDF), cleans text (conservative offset-
preserving pass), chunks at 512 tokens with 64-token overlap, embeds with
`gemini-embedding-001` at 768 dimensions (Matryoshka truncation), and stores chunks in
Postgres with both a pgvector column and a pre-computed `tsvector` column for FTS.

**Read path (P0b):**

1. **Dense retrieval** — pgvector cosine KNN, fetches `CANDIDATE_K` (default 30) chunks.
2. **Lexical retrieval** — Postgres FTS (`ts_rank` score) against the `tsvector` column,
   fetches another `CANDIDATE_K` pool. Language is frozen to `english` — it must match
   the stored `to_tsvector('english', ...)` column; a runtime language knob would silently
   desync queries from the index, so there is intentionally none.
3. **RRF fusion** — Reciprocal Rank Fusion merges both ranked lists into a single ordering.
   RRF ignores absolute scores (only rank positions matter), so the ts_rank vs. BM25
   distinction has no effect on fusion quality (see Honesty callouts below).
4. **Rerank (optional)** — `RerankedRetriever` is a decorator that wraps any retriever and
   re-scores the fused pool with Cohere's cross-encoder via LiteLLM. Enable by setting
   `RERANK_ENABLED=true` and `COHERE_API_KEY`.
5. **Assemble** — top-k chunks are packed within a `CONTEXT_TOKEN_BUDGET` (default 6 000
   tokens), numbered for citation indexing.
6. **Generate** — `gemini-2.5-pro` produces the answer. `POST /query/stream` streams
   tokens via SSE; citations are validated and appended after the final token.
7. **Citation validation** — any citation that does not map to an assembled chunk is
   stripped. The answer never ships a fabricated source.

**Observability:**

- **Langfuse** traces (span per stage) — enabled when `LANGFUSE_PUBLIC_KEY` and
  `LANGFUSE_SECRET_KEY` are set; no-ops otherwise.
- **Prometheus** metrics at `GET /metrics` — per-stage p95 latency histograms, request
  counts, error rates.
- **Grafana** dashboard — `make obs-up` starts the full
  Prometheus + Grafana + Langfuse compose stack.

Every subsystem sits behind a Protocol (spec §6), so swapping a retriever, embedder, or
generator requires only a one-line config change.

---

## Honesty callouts

**ts_rank is not BM25.**
Postgres `ts_rank` is a tf-idf-ish scorer, not the probabilistic BM25 used by Elasticsearch
or Solr. In this system that distinction is harmless: RRF fuses by rank position, ignoring
absolute scores entirely, so a tf-idf rank and a true BM25 rank produce equivalent inputs
to the fusion step. The upgrade path to real BM25 is
[ParadeDB `pg_search`](https://docs.paradedb.com/documentation/full-text/overview) — a
Postgres extension that exposes BM25 scoring while keeping the same table schema.

**pgvector HNSW index limit is 2 000 dimensions.**
pgvector's HNSW index supports a maximum of 2 000 dimensions. `gemini-embedding-001`
natively produces 3 072-d vectors; we truncate to 768 d via Matryoshka Representation
Learning (MRL) by setting `output_dimensionality=768` in the embedding call. 768 d sits
well inside the limit and preserves retrieval quality (Google reports < 1% NDCG loss vs.
full 3 072-d on most tasks).

---

## Results (P0b smoke eval)

The headline metric is the `ALL` aggregate row. Per-stratum confidence intervals are
directional at smoke-eval n — a statistically robust 30–50 item set per stratum is a
Phase-1 deliverable.

### Quality

| Metric | ALL | CI (95%) |
|--------|-----|----------|
| Faithfulness | _pending live run_ | _pending live run_ |
| Answer relevance | _pending live run_ | _pending live run_ |
| Hit@k | _pending live run_ | _pending live run_ |
| MRR | _pending live run_ | _pending live run_ |
| nDCG | _pending live run_ | _pending live run_ |
| Rerank lift (MRR delta) | _pending live run_ | _pending live run_ |

### Latency (p95 per stage)

| Stage | p95 |
|-------|-----|
| Dense retrieval | _pending live run_ |
| Lexical retrieval | _pending live run_ |
| RRF fusion | _pending live run_ |
| Rerank | _pending live run_ |
| Assemble | _pending live run_ |
| Generate (TTFB) | _pending live run_ |

---

## Development

```bash
uv sync
make test        # unit tests (fast, no Postgres required)
make test-int    # integration tests (needs local Postgres+pgvector + TEST_DATABASE_URL in .env)
make lint        # ruff check
make type        # mypy strict
make fmt         # ruff format
```

Or without `make`:

```powershell
uv run pytest -m "not integration and not live" -q   # unit
uv run pytest -m integration -q                       # integration (local Postgres)
uv run ruff check .
uv run mypy src
```

---

## Configuration reference

All settings live in `.env` (copied from `.env.example`). Key P0b additions:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CANDIDATE_K` | `30` | Over-fetch pool per retrieval leg before fusion/rerank |
| `COHERE_API_KEY` | _(empty)_ | Cohere API key — required when `RERANK_ENABLED=true` |
| `RERANK_ENABLED` | `false` | Enable Cohere cross-encoder rerank |
| `RERANK_MODEL` | `cohere/rerank-english-v3.0` | LiteLLM model string for reranker |
| `EVAL_DATABASE_URL` | _(empty)_ | Dedicated eval DB; `rag-eval` refuses to run if this equals `DATABASE_URL` |
| `LANGFUSE_PUBLIC_KEY` | _(empty)_ | Enables Langfuse tracing when set |
| `LANGFUSE_SECRET_KEY` | _(empty)_ | Langfuse secret |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse endpoint |

P0a variables (`GEMINI_API_KEY`, `DATABASE_URL`, `TEST_DATABASE_URL`, `GENERATION_MODEL`,
`EMBEDDING_MODEL`, `EMBEDDING_DIM`) remain unchanged.

---

## Optional: Docker (reviewers)

A `docker-compose.yml` and `Dockerfile` are included as a convenience for reviewers who
lack a local Postgres installation. **This is not the author's primary development path
and has not been exercised in the local dev environment.**

```bash
cp .env.example .env   # fill GEMINI_API_KEY (DATABASE_URL is pre-set for Docker)
make up                # docker compose up --build -d (starts api + pgvector/pgvector:pg16)
# wait for api to be healthy, then:
docker compose exec api rag-ingest /app/data/raw
make query Q="does nitrogen help maize?"
make down
```

The `api` service runs migrations automatically on startup (`alembic upgrade head`) and
connects to the bundled `pgvector/pgvector:pg16` Postgres container via
`DATABASE_URL=postgresql+psycopg://rag:rag@postgres:5432/rag`.
