# Production-Grade RAG System — Design Spec

- **Author:** Sathish R
- **Date:** 2026-06-02
- **Status:** Approved design — ready for implementation planning
- **Roadmap slot:** Project 01 / 15 (Tier 1), expanded into a phased flagship
- **Repo:** `production-rag` (new, dedicated)

---

## 1. Summary

Build a thin, explicitly-coded hybrid-RAG core on a single Postgres + pgvector store, route every model call through LiteLLM with Gemini as the default, and hide each subsystem behind a stable interface so four marquee features phase in without rewrites. Each phase ships as a complete artifact with a résumé-grade metric. A hard stop-line at every phase boundary guarantees the repo is never half-built.

The four marquee features, layered on the core: **eval-in-CI**, **corrective/self-reflective RAG**, **multi-source + incremental ingestion**, and a **light geospatial-metadata edge**. Observability and monitoring (§14), data-lifecycle versioning (§10), citation-span validation (§6), and statistically-honest evaluation (§15) are built in, not bolted on.

## 2. Goals and non-goals

**Goals**

- Demonstrate AI Engineering, LLMOps, Data Engineering, Backend Engineering, and System Design in one coherent repo.
- Stay reproducible: a reviewer runs `docker compose up`, ingests the corpus, and queries it.
- Produce real numbers (faithfulness, rerank lift, hallucination reduction, per-stage latency, cost) and put them in the README.
- Keep clean seams so later roadmap projects (P12 cost-router, P13 LLMOps platform) dock on without rework.

**Non-goals** (deferred to their own roadmap projects)

- Full VLM-over-imagery pipeline → **P09** (this repo carries only a light geospatial-metadata layer).
- Kubernetes autoscaling and self-hosted model serving → **P10**.
- A full LLMOps control plane (drift detection, threshold alerting, prompt registry UI) → **P13**; this repo seeds it with eval-in-CI, versioned prompts, service metrics, and online quality scores.
- A standalone guardrails/safety product → **P08**.

## 3. Context and the scope decision

The roadmap positions RAG as a focused ~2-week Tier 1 project and already breaks the heavy pieces into separate projects (P05 eval, P06 corrective RAG, P08 guardrails, P12 router, P13 LLMOps). The original brief for this repo described a system that absorbed all of them — a capstone, not Project 01.

We resolved the tension in favor of the roadmap's "depth beats breadth" north star: a **strong core plus a small set of marquee features**, built in strict phases with a hard stop-line, so ambition never produces a sprawling half-built repo. Heavy multimodal work stays in P09; the two repos cross-reference each other.

An architecture review (2026-06-02) tightened the plan further: Phase 0 was split into three shippable slices, and data-lifecycle, capacity, citation-correctness, eval-rigor, and web-fallback-trust obligations were added where cheap and on-goal. Enterprise-only obligations (per-user permissions, bounding-box citations, multi-annotator agreement, a full ops runbook) were explicitly cut (§20).

## 4. Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Ambition | Strong core + marquee features, strict phases, stop-line | Depth over buzzword breadth; never half-built |
| Corpus | Public agriculture / agronomy (research papers, FAO/USDA/extension PDFs) | Reproducible *and* tied to the author's domain edge; messy real PDFs give a genuine ingestion story |
| Model hosting | API-first | Simplest code, strongest eval numbers |
| Model wrapper | **LiteLLM** | One interface for completion + embedding + rerank; native Langfuse callback; built-in cost tracking; Router for retries/fallbacks/caching |
| Default models | Gemini — `gemini-2.5-pro` (generation), `gemini-2.5-flash` (grader/router/judge) | Strong, cheap, swappable by config string |
| Architecture | Thin & explicit core; framework only where it earns its place | Highest engineering-taste signal; one-command runnable |
| Store | Single Postgres 16 + pgvector 0.7 | Vectors, lexical FTS, metadata, and ingestion state in one place; no cross-store consistency problem |
| Hybrid fusion | Reciprocal Rank Fusion (RRF) | Rank-based; no score-normalization headaches |
| Rerank | Cohere Rerank via `litellm.rerank()`; local `bge-reranker-v2-m3` swap | Gemini has no first-class rerank API |
| Orchestration | LangGraph only for the Phase-2 corrective loop | A state machine earns its place in a cyclic graph, nowhere else |
| Phase 0 granularity | Split into P0a (skeleton) / P0b (hybrid+rerank+SSE+obs+eval) / P0c (OCR/tables) | A reliable walking-skeleton MVP de-risks the build before hardening |
| Data lifecycle | Versioning columns + soft-delete tombstones + Alembic migrations + reindex CLI | Honest re-embed/re-chunk story; real data-engineering maturity |
| Citation correctness | Span offsets (page/char) + "cite only assembled chunks" validation + drift tests | Trust signal; citations verified, not assumed |
| Eval methodology | Stratified golden set, separate retrieval/answer gates, bootstrap CIs, tolerance bands | Statistically honest gating; avoids brittle/flaky CI |
| Observability & monitoring | Level B — Langfuse tracing + Prometheus/Grafana service metrics + online sampled quality scoring | Honors the `notes.txt` priority; covers LLM-native and SRE-style monitoring; alerting + drift stay in P13 |

## 5. Architecture and data flow

Two paths meet at one store. Ingestion only writes Postgres; the query path only reads it. They share a schema, not code — so you rebuild the index without touching the API, and load-test the API against a frozen index. Every box tagged `[Phase N]` is additive; P0a is the unbroken flow minus FTS, rerank, streaming, the grader, the router, and the self-check.

```
INGESTION PATH  (offline / async — the "write" side)

  SOURCES                    one adapter → one normalized Document
  ├─ PDF        [P0a/P0c]
  ├─ HTML/web   [Phase 3]    ┌──────────┐   ┌──────────┐   ┌──────────┐
  ├─ API/DB     [Phase 3] ─► │  Loader  │─► │  Clean / │─► │ Chunker  │─►
  └─ (pluggable SourceAdapter)│ + Parse │   │Normalize │   │(strategy)│
                             └──────────┘   └──────────┘   └────┬─────┘
        ┌───────────────────────────────────────────────────────┘
        ▼
   ┌──────────┐   ┌─────────────────────────────────────────────────┐
   │ Embedder │─► │  Indexer / Upsert  → POSTGRES + pgvector         │
   │  (API)   │   │  • documents  • chunks  • embeddings (HNSW)      │
   └──────────┘   │  • metadata (region/crop/season)  • FTS tsvector │
                  │  • content_hash (dedup)  • source_watermark      │ ◄─ [Phase 3
                  │  • version cols (chunker/embedding)  • deleted_at │     incremental
                  └─────────────────────────────────────────────────┘     + tombstones]

QUERY PATH  (online / sync-streaming — the "read" side)

  Client ─► FastAPI  /query (SSE stream, retrieval_scope)
                ▼
        ┌───────────────┐  [Phase 2] simple? ──────────────────────────┐
        │ Adaptive route│  complex? ─► full corrective loop            │
        └───────┬───────┘                                              ▼
                ▼                                                ┌──────────────────┐
        ┌──────────────┐   ┌───────────────────────────┐        │  Reranker        │
        │  Retriever   │─► │ Hybrid: pgvector dense KNN │──────► │ (Cohere via      │
        │ (+ metadata  │   │  + Postgres FTS lexical    │        │  LiteLLM rerank) │
        │   filter)    │   │  fused via RRF             │        └────────┬─────────┘
        └──────────────┘   └───────────────────────────┘                 ▼
   [Phase 2] ┌─────────────────┐  weak ctx? ─► rewrite query / web fallback ─┐
             │ Retrieval grader │ ─────────────────────────────────(loop)────┘
             └────────┬─────────┘  ok ▼
                      ▼
        ┌──────────────────┐   ┌───────────────┐   ┌──────────────────────────┐
        │ Context assembly │─► │  LLM generate │─► │ Citation validation +    │
        │ (budget,dedup,   │   │  (streaming,  │   │ [Phase 2] groundedness   │
        │  citation map)   │   │   citations)  │   │  self-check gate         │
        └──────────────────┘   └───────────────┘   └────────────┬─────────────┘
                                                                 ▼
                                       streamed tokens + validated source citations

CROSS-CUTTING
  • Observability: Langfuse traces every node (latency / tokens / $ / quality) via LiteLLM callback
  • Monitoring: Prometheus /metrics + Grafana (req rate · p50/p95/p99 · errors · $/day);
                online sampled quality scoring (faithfulness/groundedness on live traffic → Langfuse)
  • Citation validation: an answer may cite only chunks present in assembled context (post-gen check)
  • Provider seam: LiteLLM (generation · embedding · rerank), swap by config string
  • Config: pydantic-settings, one typed Settings object
  • Eval harness: stratified golden set → retrieval + answer metrics  ──[Phase 1]──► CI gate
```

## 6. Component and interface contracts

Interfaces stay fixed across phases. Phases add implementations or wrap existing ones — they never change a signature. That property is what makes the stop-line real.

**Shared domain models**

```python
RawDocument   # source_id, source_type, uri, raw_bytes|text, fetched_at,
              # source_etag, source_last_modified, license, source_meta
Document      # doc_id, document_version, text, structure(tables/sections),
              # metadata{region,crop,season}, content_hash, deleted_at
Chunk         # chunk_id, doc_id, text, ordinal, metadata, token_count,
              # chunker_name, chunker_version, embedding_model, embedding_dim
ScoredChunk   # chunk + score + provenance("dense"|"lexical"|"fused"|"rerank")
Citation      # marker, doc_id, chunk_id, source_uri, source_kind("corpus"|"web"),
              # page, char_start, char_end
Answer        # text, citations[], usage{tokens,$,latency}, trace_id, retrieval_scope
```

**Ingestion-path contracts**

```python
class SourceAdapter(Protocol):          # Phase 0: Pdf; Phase 3: Html, Api/DB
    source_type: str
    def fetch(self, since: Watermark | None) -> Iterator[RawDocument]: ...
    #  since=None → full scan (Phase 0); a watermark (etag/last_modified) → incremental (Phase 3).

class Parser(Protocol):    def parse(self, raw) -> Document      # OCR fallback, table extraction [P0c]
class Cleaner(Protocol):   def clean(self, doc) -> Document
class Chunker(Protocol):   name; version; def chunk(self, doc) -> list[Chunk]   # strategy; eval picks the winner
class EmbeddingProvider(Protocol):  model; dim; def embed(self, texts) -> list[Vector]   # LiteLLM, batched + retried

class ChunkRepository(Protocol):        # the only thing that touches Postgres
    def upsert(self, chunks, vectors) -> UpsertStats     # idempotent via content_hash
    def soft_delete(self, doc_ids) -> int                # tombstone; retrieval excludes deleted_at
    def get_watermark(self, source_id) -> Watermark | None
    def set_watermark(self, source_id, wm) -> None
    def reindex(self, *, embedding_model=None, source_id=None) -> ReindexStats   # re-embed targeted chunks
    #  Schema evolves via Alembic migrations. Versioning columns let reindex target
    #  "all chunks from embedding model X" or "all docs from source Y".
```

**Query-path contracts**

```python
class Retriever(Protocol):
    def retrieve(self, query, k, filt: MetadataFilter | None) -> list[ScoredChunk]: ...
    #  HybridRetriever composes DenseRetriever + LexicalRetriever, fuses by RRF.
    #  filt carries region/crop/season — the light geospatial edge lives here, for free.

class Reranker(Protocol):         def rerank(self, query, chunks, top_n) -> list[ScoredChunk]
class ContextAssembler(Protocol): def assemble(self, query, chunks, token_budget) -> AssembledContext
class LLMProvider(Protocol):
    def stream(self, messages, **opts) -> Iterator[Token]    # generation (LiteLLM acompletion)
    def complete(self, messages, **opts) -> Completion       # graders/judges (non-streaming)

class Answerer:                   # orchestrates the read path; P0a = a straight line
    def answer(self, query, filt, scope=RetrievalScope.corpus_only) -> Iterator[AnswerEvent]
    #  yields tokens + citations. Enforces citation validity: the answer may cite ONLY
    #  chunks present in assembled context; a post-gen check strips/flags any that are not.
    #  scope (corpus_only | web_allowed | web_required) governs the Phase-2 web fallback.
```

**Phase-2 additions — they wrap, never replace**

```python
class RetrievalGrader(Protocol):     def grade(self, query, chunks) -> Grade       # cheap LLM: relevant/sufficient?
class QueryRewriter(Protocol):       def rewrite(self, query, reason) -> list[str]
class GroundednessChecker(Protocol): def check(self, answer, context) -> Verdict   # hallucination gate

# CorrectiveAnswerer wraps the SAME Retriever/Reranker/Answerer in a LangGraph loop:
#   retrieve → grade → (rewrite | web-fallback → retrieve)* → assemble → generate → validate → self-check
# Web fallback (Tavily) is gated by retrieval_scope + a domain allow/blocklist; web chunks are
# labeled source_kind="web" so corpus-vs-web provenance is visible in the answer.
# AdaptiveRouter: trivial query → plain Answerer; else → CorrectiveAnswerer.
# Both expose the identical answer() signature → the FastAPI layer never knows which ran.
```

The FastAPI route depends only on `Answerer.answer()`. Whether a plain straight-line (P0a) or an adaptive corrective loop (Phase 2) runs behind it is an injected detail — so every phase ships behind the same API, and every unit tests against fakes with zero network.

## 7. Phase plan

Each phase (and sub-phase) is a finished, demoable artifact with a number. Start the next slice only when the current "Done when" holds and its metric is captured. If you stop anywhere, what exists is whole.

### Phase 0 — Core flagship / MVP (three shippable slices)

**P0a — Walking skeleton · ~4–5 days**
- **Ships:** text-PDF ingest (PyMuPDF, no OCR yet) → clean → chunk → embed → Postgres+pgvector upsert with the full schema (versioning columns, `content_hash`, `deleted_at`) under **Alembic** migrations → dense vector search → context assembly with **span-level citations** → **non-streaming** cited answer → FastAPI `/query` (JSON) → docker-compose (api + postgres). Citation-validation rule enforced from day one.
- **Done when:** one-command up + ingest + a cited (page/char) answer over dense search.
- **Unlocks:** the end-to-end skeleton and a first faithfulness reading.

**P0b — Hybrid, rerank, streaming, observability, eval · ~4–5 days**
- **Ships:** Postgres FTS + RRF hybrid; Cohere rerank; SSE streaming; Langfuse traces + Prometheus `/metrics` + Grafana; a **stratified** 30–50 item golden set + `make eval` scorecard that reports **retrieval and answer metrics separately, with bootstrap CIs**; per-stage p95 captured.
- **Done when:** streaming cited answers; `make eval` prints separated retrieval/answer scorecards with CIs; Langfuse + Grafana render.
- **Unlocks:** "Hybrid + cross-encoder rerank raised faithfulness X→Y and context-precision A→B; per-stage retrieval p95 < Z ms," plus a clean before/after rerank-lift number.

**P0c — Parser hardening · ~2–3 days**
- **Ships:** Docling for multi-column + tables; Tesseract OCR fallback on empty text layers; **citation-drift tests** across clean/chunk; dead-letter quarantine for failed docs.
- **Done when:** a scanned / table-heavy PDF ingests correctly; citation offsets survive cleaning; failed docs are quarantined, not fatal.
- **Unlocks:** "handles messy real PDFs" with evidence. *(Bounding-box / table-cell citations deferred — only if Docling yields coordinates cheaply.)*

### Phase 1 — Eval-in-CI quality gate · ~1 week

- **Ships:** `make eval` promoted into GitHub Actions over a frozen mini-corpus + the stratified golden set; **two separate gates** (retrieval metrics, answer metrics); committed baselines with **tolerance bands** (not brittle single numbers) and **bootstrap-CI** comparison; CI posts the per-stratum scorecard as a PR comment; judge pinned at temperature 0; a minimum human-labeled **calibration slice** anchors judge-vs-human agreement. Adds **online sampled quality scoring** — a configurable % of live queries get the cheap faithfulness judge, logged to Langfuse.
- **Done when:** a PR that worsens retrieval *or* answer quality beyond the band turns CI red with a diffed per-stratum scorecard; sampled live queries show a quality score in Langfuse.
- **Unlocks:** "Eval-gated CI (separate retrieval/answer gates, CI-banded) blocking regressions; judge-vs-human agreement X%; answer quality monitored on live traffic."

### Phase 2 — Corrective/self-reflective RAG + adaptive routing · ~1–1.5 weeks

- **Ships:** `RetrievalGrader` + `QueryRewriter` + web-search fallback (Tavily) + `GroundednessChecker`, wired into a LangGraph `CorrectiveAnswerer` with a max-iteration cap; `AdaptiveRouter` (cheap model: trivial → fast path, complex/low-confidence → corrective). Web fallback is gated by `retrieval_scope` + a **domain allow/blocklist**, web chunks **labeled `source_kind="web"`**, and an **eval slice verifies web evidence never overrides authoritative corpus docs**. Benchmarked against the P0b baseline on the adversarial / out-of-corpus strata.
- **Done when:** the hard strata show lower hallucination; the router proves most easy queries skip the loop; the web-override eval slice passes.
- **Unlocks:** "Corrective grading + self-reflection cut hallucinated answers X% on out-of-corpus queries; adaptive routing held p50 latency flat by sending Y% down the fast path; scoped, provenance-labeled web fallback."

### Phase 3 — Multi-source + incremental ingestion · ~1–1.5 weeks

- **Ships:** `HtmlSourceAdapter` (trafilatura) + one `Api/DB` adapter through the same pipeline; the incremental path (`source_etag`/`source_last_modified` watermark + `content_hash` → upsert only changed, **tombstone upstream deletions**); a scheduled/worker re-index (arq); the **`reindex` CLI** (`--embedding-model` / `--source`); per-source `license` metadata; an optional small event-driven trigger (webhook or watched folder).
- **Done when:** modifying one doc re-indexes only that doc and removing one upstream tombstones it (show upsert/delete stats); a model-swap `reindex` re-embeds only the targeted chunks; the query reflects all three.
- **Unlocks:** "Multi-source ingestion (PDF+web+API) with incremental upsert, tombstoned deletes, and targeted re-indexing — N docs/hr, M% of embedding calls saved vs full re-index."

### Phase 4 — Light geospatial-metadata edge · ~3–4 days

- **Ships:** region/crop/season tagging during ingestion (rules + LLM tagging); metadata filters exposed in `/query`; a **metadata-filtered ANN degradation benchmark** (filtered vs unfiltered recall) with the chosen mitigation documented; a demo of filtered retrieval ("answer using only South-region maize docs"); a cross-link to P09.
- **Done when:** a filtered query provably restricts retrieval to matching docs *and* the recall-under-filter benchmark is recorded.
- **Unlocks:** "Geospatial-aware retrieval (region/crop/season filtering) with measured recall-under-filter" — the differentiation hook that seeds P09.

**Stop-line:** portfolio-proud from the end of **P0b + Phase 1** (hybrid+rerank+eval+observability+CI gate), flagship-proud from the end of **Phase 2**. Phases 3–4 are depth bonuses. Cumulative ≈ **6–8 weeks** part-time for everything; ≈ 4 weeks reaches the end of Phase 2.

## 8. Tech stack per layer

| Layer | Pick | Why / trade-off |
|---|---|---|
| Runtime | Python 3.12 + uv | Fast, reproducible lockfile |
| PDF parse | PyMuPDF [P0a] · Docling [P0c] multi-column+tables · Tesseract OCR fallback [P0c] | Fast text first; layout/OCR added in hardening |
| HTML (P3) | trafilatura + httpx | Best-in-class boilerplate stripping |
| API/DB (P3) | httpx · SQLAlchemy Core | Thin, async, no ORM ceremony |
| Migrations | Alembic | Versioned schema, run in CI |
| Chunking | Own strategies (recursive/layout-aware + semantic), tiktoken for budgeting | Own it; the eval harness picks the winner |
| Model wrapper | LiteLLM | Unifies completion + embedding + rerank; Langfuse callback; cost tracking; Router |
| Embeddings | Google `text-embedding-004` (768-d) default; `gemini-embedding-001` for MRL dims | 768-d indexes cleanly under pgvector's limit |
| Vector + lexical + meta + state | Postgres 16 + pgvector 0.7 (HNSW), native FTS (tsvector/GIN) | One store, four jobs |
| Fusion | Reciprocal Rank Fusion (RRF) | Rank-based; no score normalization |
| Rerank | Cohere Rerank via `litellm.rerank()`; `bge-reranker-v2-m3` local swap | Gemini has no first-class rerank |
| Generation | `gemini/gemini-2.5-pro` behind LiteLLM | Quality where the user sees it |
| Grader / router / judge | `gemini/gemini-2.5-flash` | The loop and eval run many cheap calls |
| Corrective loop (P2) | LangGraph (only here) | State machine + checkpointing in a cyclic graph |
| Web fallback (P2) | Tavily + domain allow/blocklist | LLM-oriented search, scoped and provenance-labeled |
| API | FastAPI + uvicorn + sse-starlette, Pydantic v2 | Async streaming + typed contracts |
| Config | pydantic-settings | One typed Settings object; 12-factor |
| Observability | Langfuse (self-hosted) + structlog | Per-node cost/tokens/latency/quality via LiteLLM callback |
| Service metrics | prometheus-client (`/metrics`) + Prometheus + Grafana | SRE view: request rate, latency percentiles, error rate, $/day |
| Online quality | Sampled live faithfulness/groundedness judge → Langfuse | Watches answer quality on real traffic, not just offline eval |
| Eval | RAGAS + custom hit@k/MRR/nDCG + bootstrap CIs (numpy) | Standard RAG metrics + retrieval metrics + statistical honesty |
| Ingest worker (P3) | arq (async Redis queue) or APScheduler for MVP | Light; Celery noted as scale-path, not built |
| Demo UI | Minimal static HTML/JS chat hitting SSE | Zero build; shows streaming + citations |
| Tests | pytest · pytest-asyncio · testcontainers (real pgvector) · respx (API mocks) | Unit on fakes, integration on ephemeral Postgres |
| Standards | ruff · mypy strict · pre-commit · Makefile · multi-stage Docker | The hygiene a reviewer skims for first |
| Deploy | docker-compose; K8s documented as scaling path | One command; real K8s deferred to P10/P13 |

**Sharp engineering details that prove depth**

1. **pgvector's index limit is 2000 dimensions.** A 3072-d embedding (`gemini-embedding-001` at full size) will not take an HNSW index naively. Two documented ways out: truncate via the Matryoshka `dimensions` parameter, or index the full vector with pgvector's `halfvec` (half-precision, up to 4096-d, ~half the index size, negligible recall loss). Defaulting to `text-embedding-004` at 768-d sidesteps it — but the README shows the trade.
2. **Postgres `ts_rank` is not BM25** — it is tf-idf-ish. The README says so, fuses it with dense via RRF (which ignores absolute scores), and names ParadeDB `pg_search` (real BM25 in Postgres via Tantivy) as the single-store upgrade. Most "hybrid BM25" repos mislabel this; this one will not.
3. **Eval determinism:** judge pinned, temperature 0, frozen mini-corpus, bootstrap-CI tolerance bands → a red CI build signals a real regression, not judge noise.
4. **Cost discipline by design:** the cheap model serves the many grader/router/judge calls; the frontier model serves only the single user-facing generation — the exact seam P12 later exploits.

## 9. Capacity assumptions and performance targets

The repo demonstrates that the operational envelope is understood and measured, not that it runs at enterprise scale. Stated assumptions:

- **Corpus scale:** ~5k–50k documents, ~200k–1M chunks, 768-d vectors. Estimated HNSW index ~1–4 GB; fits a single Postgres comfortably.
- **Concurrency:** single-digit QPS (demo/portfolio), not a load-balanced fleet.

**Per-stage p95 latency targets** (measured via Prometheus, reported in the README):

| Stage | p95 target |
|---|---|
| Dense KNN | < 50 ms |
| Lexical FTS | < 40 ms |
| Hybrid (RRF) | < 80 ms |
| Rerank (Cohere) | < 350 ms |
| Generation (time-to-first-token) | < 1.2 s |

**ANN tuning:** HNSW `m` and `ef_construction` set at build; `ef_search` tuned against the recall/latency curve and documented.

**Known risk — metadata-filtered ANN:** HNSW recall can degrade sharply under a restrictive metadata filter (post-filtering drops candidates below `ef_search`). A benchmark (Phase 4) measures filtered vs unfiltered recall on region/crop/season, and the README documents the chosen mitigation (pre-filtering, partial indexes, or a widened `ef_search`).

**Light operational notes:** enable `pg_stat_statements`; a small fixed connection pool (≈10–20) sized to the worker count; repeated upserts create dead tuples, so rely on autovacuum plus a periodic `VACUUM (ANALYZE)` after large re-index runs. A full capacity-planning runbook is out of scope (§20).

## 10. Data lifecycle and versioning

Ingestion is a living dataset, not a one-shot load:

- **Versioning columns** — every chunk records `chunker_name`, `chunker_version`, `embedding_model`, `embedding_dim`; every document records `document_version` and `content_hash`. This makes "which model/strategy produced this vector" answerable and enables targeted re-indexing.
- **Soft deletes** — a removed upstream document is tombstoned (`deleted_at`) and excluded from retrieval, never hard-deleted mid-run. Incremental ingestion (Phase 3) detects upstream removals.
- **Change detection** — `source_etag` / `source_last_modified` drive the incremental watermark for web/API sources; `content_hash` drives dedup for files.
- **Re-indexing** — a `reindex` CLI re-embeds targeted subsets: `--embedding-model <old>` after a model swap, or `--source <id>` after a source change, without rebuilding the whole corpus.
- **Migrations** — the schema evolves through Alembic migrations, committed and run in CI.
- **License/governance** — each source carries a `license` field (the corpus is open-access). Per-user access policy and permission enforcement are out of scope for a single public corpus (§20).

## 11. Repo structure

```
production-rag/
├── README.md              # problem → arch diagram → RESULTS scorecard → 1-cmd quickstart → demo
├── pyproject.toml · uv.lock · Makefile · .env.example
├── docker-compose.yml · Dockerfile        # api · postgres+pgvector · langfuse · prometheus · grafana · (worker·redis P3)
├── alembic/  versions/    # schema migrations
├── .github/workflows/  ci.yml (lint·type·unit·integration)   eval.yml (Phase-1 gates + PR scorecard)
├── docs/  architecture.md · decisions/ (ADR-lite) · superpowers/specs/<this spec>
├── eval/  golden_set.yaml (stratified: Q · reference · relevant-doc-ids · stratum)   baselines/ (thresholds + bands)
├── monitoring/  prometheus.yml · grafana/ (dashboard json)
├── src/rag/
│   ├── config.py              # pydantic-settings Settings
│   ├── models.py              # Document · Chunk · ScoredChunk · Citation · Answer
│   ├── providers/             # LiteLLM-backed seams: llm.py · embeddings.py · rerank.py
│   ├── ingestion/
│   │   ├── sources/           # SourceAdapter: pdf.py [P0a/P0c] · html.py · api.py [P3]
│   │   ├── parse.py · clean.py · embed.py · repository.py · pipeline.py · reindex.py
│   │   └── chunking/          # fixed.py · semantic.py · layout.py
│   ├── retrieval/  dense.py · lexical.py · hybrid.py (RRF + filter) · rerank.py
│   ├── generation/  assembler.py · answerer.py · citations.py (validation) · prompts/ (versioned)
│   ├── corrective/ [P2]  grader.py · rewriter.py · websearch.py · groundedness.py · router.py · graph.py
│   ├── eval/  metrics.py (RAGAS + hit@k/MRR/nDCG) · stats.py (bootstrap CIs) · runner.py
│   ├── observability/  tracing.py (langfuse + structlog + LiteLLM callback) · metrics.py (prometheus) · online_quality.py
│   └── api/  app.py · routes.py (/query SSE · /ingest · /healthz · /metrics) · schemas.py
├── workers/ [P3]  arq incremental-ingest worker
├── ui/            minimal static chat page (SSE + citations)
└── tests/  unit/ (fakes, no net) · integration/ (testcontainers pgvector + respx) · conftest.py
```

## 12. Engineering standards

mypy strict · ruff lint+format · pre-commit · 12-factor config · structlog with a `trace_id` on every log · ADR-lite decision records · Makefile one-liners (`make up/ingest/eval/reindex/test`) · multi-stage non-root Docker with healthcheck · secrets only via env (`.env.example` committed, `.env` never) · versioned prompt templates as files (seeds P13 prompt-versioning) · Alembic migrations run in CI.

## 13. Testing strategy

- **Unit** — every interface against fakes (`FakeRetriever`, `FakeLLMProvider` yielding canned tokens, `FakeEmbedder`). Fast, no network. Test-first for the deterministic high-value logic: RRF fusion, **citation mapping + validation**, **citation-drift after clean/chunk**, chunkers, content-hash dedup + watermark.
- **Integration** — testcontainers boots a real Postgres+pgvector; verifies upsert idempotency, incremental re-index, tombstoned deletes, the `reindex` CLI, and dense/lexical/hybrid retrieval over seeded data, plus the **metadata-filtered recall benchmark**. respx mocks all LiteLLM/Cohere/Tavily HTTP, so CI spends nothing.
- **Eval-as-gate (P1)** — the stratified golden-set eval runs in its own workflow over a frozen mini-corpus, judge pinned at temperature 0, retrieval and answer gated separately with bootstrap-CI tolerance bands.

## 14. Observability and monitoring

First-class concern (Level B), across two complementary layers; heavy control-plane parts go to P13.

- **Layer 1 — LLM-native (Langfuse).** The LiteLLM callback streams every model call into Langfuse: cost, tokens, latency, and quality scores, per node and per trace. Curated dashboard screenshots become a README deliverable.
- **Layer 2 — service-level (Prometheus + Grafana).** The FastAPI app exposes a Prometheus `/metrics` endpoint (request rate, p50/p95/p99 latency, error rate, $/day). Prometheus scrapes it; a small Grafana dashboard ships in docker-compose — the SRE view Langfuse does not cover.
- **Online quality scoring.** A configurable sample of live queries gets a cheap faithfulness judge (Phase 1), extended with groundedness once Phase 2 lands. Scores log to Langfuse, so quality is watched on real traffic.
- **Deferred to P13:** threshold alerting (Slack/webhook) and input/output drift detection. This repo leaves the seams — metrics, online scores, traces — for P13 to consume.

## 15. Evaluation methodology

Eval splits into two independent concerns, gated separately:

- **Retrieval quality (no LLM judge):** hit@k, MRR, nDCG against the golden set's relevant-doc labels. Deterministic, fast, the first CI gate.
- **Answer quality (LLM-judged):** RAGAS faithfulness, answer-relevance, context precision/recall, plus a **citation-faithfulness check** (does each cited chunk actually support its claim?). Judge pinned at temperature 0.

**Stratified golden set** — easy, ambiguous, table-based, adversarial, out-of-corpus, and metadata-filtered slices, scored per stratum (an aggregate alone hides where the system fails).

**Statistical honesty**
- Score deltas between runs report **bootstrap confidence intervals**, not bare point differences.
- CI gates use **tolerance bands** ("faithfulness must not drop more than X within the 95% CI"), so judge noise does not cause false red builds.
- A minimum human-labeled **calibration slice** anchors judge-vs-human agreement. Inter-annotator agreement applies only if more than one annotator labels; a solo project uses a single calibrated annotator as the baseline (§20).

**Online quality scoring (Phase 1+)** — a sample of live queries gets the cheap judge, logged to Langfuse, so quality is monitored in production, not just in CI.

## 16. Failure modes — designed in, not patched on

- **Ingestion:** per-doc try/except → failed docs land in a dead-letter table with a reason, the batch continues; OCR fallback on an empty text layer; idempotent upsert makes re-runs safe; the watermark makes a run resumable; upstream deletions tombstone, never silently vanish.
- **Retrieval:** zero results or an empty filter → an explicit "no relevant context" path. Never hallucinate to fill silence.
- **Citations:** a generated citation absent from assembled context is stripped and logged; the answer never ships a fabricated source.
- **Providers:** LiteLLM Router handles rate-limits and timeouts with retries plus model fallback; a per-request budget guard caps spend.
- **Generation:** token-budget overflow → the assembler truncates by relevance; a mid-stream error flushes the partial answer and emits an SSE error event.
- **Corrective loop (P2):** a hard max-iteration cap prevents infinite loops; if confidence stays low, the system returns its best answer flagged low-confidence with citations, never a silent fabrication; the groundedness gate forces one regeneration or an honest "insufficient evidence"; web evidence is scope-gated and provenance-labeled so it cannot quietly override the corpus.
- **Measured, not assumed:** the stratified golden set includes empty, adversarial, and out-of-corpus queries, so the failure paths are scored, not hoped for.

## 17. The README as a deliverable

problem → architecture diagram → results scorecard (per-stratum, with the real numbers and per-stage p95) → one-command quickstart → key design decisions (the pgvector/RRF/BM25-honesty/citation-validation calls) → curated Langfuse + Grafana dashboard shots → 60–90s demo. That sequence converts "another RAG repo" into "this person ships production systems."

## 18. Implementation notes

- **LiteLLM:** consult the current LiteLLM docs before writing the provider, embedding, rerank, Langfuse-callback, and Router code. Those APIs move; do not code them from memory.
- **Gemini default:** the generation, grader/router/judge, and embedding models default to Gemini/Google, selected by config string so any provider swap is a one-line change.

## 19. Open questions and risks

- **Corpus assembly** — pick the exact public ag/agronomy sources and licensing before P0a (candidates: agricultural research papers, FAO/USDA reports, extension-service guides). Mitigate licensing/download friction by selecting open-access sources.
- **Golden-set labeling effort** — a stratified 30–50 item set with reference answers, relevant-doc ids, and a calibration slice takes real time; budget for it inside P0b.
- **Rerank cost** — Cohere Rerank adds per-query cost; the local `bge-reranker` swap caps it if needed.
- **Gemini rate limits** — free-tier limits may throttle bulk embedding/reindex; LiteLLM Router retries/backoff plus batch sizing mitigate this.

## 20. Out of scope (cross-references and review cuts)

Roadmap hand-offs: heavy VLM multimodal → **P09** · model serving + K8s → **P10** · cost-router + semantic cache → **P12** · full LLMOps control plane (threshold alerting, drift detection, prompt registry UI) → **P13** · standalone guardrails → **P08**. This repo leaves clean seams for each: the LiteLLM Router, versioned prompts, eval-in-CI, the Prometheus metrics, the online quality scores, and the `Retriever` interface.

Cut as YAGNI for a single public-corpus portfolio (documented here as future extensions, not built):

- **Per-user access policy / permission enforcement** and upstream source-permission changes — multi-tenant governance with no users and one public corpus.
- **Bounding-box / table-cell citations** — heavy layout-coordinate plumbing; page + char offsets deliver the correctness win instead.
- **Multi-annotator inter-annotator-agreement processes** — a solo project uses a single calibrated annotator.
- **A full capacity-planning / ops runbook** — the operational envelope is documented (§9) but not operated.
