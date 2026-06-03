# Production-Grade RAG System

Hybrid-RAG over a single Postgres + pgvector store, every model call through LiteLLM
(Gemini default). This branch ships **P0a — the walking skeleton**: PDF ingest →
dense retrieval → cited, grounded answers behind a FastAPI `/query` endpoint.

> Roadmap: P0b adds hybrid + rerank + streaming + observability + eval; see
> [the design spec](docs/superpowers/specs/2026-06-02-production-rag-system-design.md).

## Quickstart (local — primary path)

Prerequisites: Python 3.12+, `uv`, and a local PostgreSQL 18 + pgvector (e.g. via pgAdmin).

```bash
# 1. In pgAdmin, create databases `rag` and `rag_test`
# 2. Copy the example env file and fill in your credentials
cp .env.example .env          # fill GEMINI_API_KEY + your postgres password

uv sync
make migrate                  # builds schema (+ CREATE EXTENSION vector)
make serve                    # start the API on :8000
make ingest CORPUS=./data/raw # ingest PDFs from a directory
make query Q="does nitrogen help maize?"
```

Response is JSON: a grounded `answer`, `citations` (doc/chunk/page/char span),
`usage` (tokens, latency), and a `trace_id`.

## How it works (P0a)

`PDF → parse → clean → fixed-token chunk → embed (text-embedding-004) → Postgres+pgvector`
on the write side; `embed query → pgvector cosine KNN → assemble (token budget,
numbered) → generate (gemini-2.5-pro) → validate citations` on the read side. A
generated citation that does not map to an assembled chunk is stripped — the answer
never ships a fabricated source.

Every subsystem sits behind a Protocol (spec §6) so P0b can add hybrid/rerank/streaming
without changing signatures. All model calls route through LiteLLM (swap provider by
changing the model string in `.env`).

## Development

```bash
uv sync
make test        # unit tests (fast, no Postgres required)
make test-int    # integration tests (needs local Postgres+pgvector + TEST_DATABASE_URL in .env)
make lint        # ruff check
make type        # mypy strict
make fmt         # ruff format
```

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
