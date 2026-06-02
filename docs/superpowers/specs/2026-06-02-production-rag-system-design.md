# Production-Grade RAG System — Design Spec

- **Author:** Sathish R
- **Date:** 2026-06-02
- **Status:** Approved design — ready for implementation planning
- **Roadmap slot:** Project 01 / 15 (Tier 1), expanded into a phased flagship
- **Repo:** `production-rag` (new, dedicated)

---

## 1. Summary

Build a thin, explicitly-coded hybrid-RAG core on a single Postgres + pgvector store, route every model call through LiteLLM with Gemini as the default, and hide each subsystem behind a stable interface so four marquee features phase in without rewrites. Each phase ships as a complete artifact with a résumé-grade metric. A hard stop-line at every phase boundary guarantees the repo is never half-built.

The four marquee features, layered on the core: **eval-in-CI**, **corrective/self-reflective RAG**, **multi-source + incremental ingestion**, and a **light geospatial-metadata edge**. Observability and monitoring are a first-class concern throughout (Level B — see §12).

## 2. Goals and non-goals

**Goals**

- Demonstrate AI Engineering, LLMOps, Data Engineering, Backend Engineering, and System Design in one coherent repo.
- Stay reproducible: a reviewer runs `docker compose up`, ingests the corpus, and queries it.
- Produce real numbers (faithfulness, rerank lift, hallucination reduction, latency, cost) and put them in the README.
- Keep clean seams so later roadmap projects (P12 cost-router, P13 LLMOps platform) dock on without rework.

**Non-goals** (deferred to their own roadmap projects)

- Full VLM-over-imagery pipeline → **P09** (this repo carries only a light geospatial-metadata layer).
- Kubernetes autoscaling and self-hosted model serving → **P10**.
- A full LLMOps control plane (drift detection, threshold alerting, prompt registry UI) → **P13**; this repo seeds it with eval-in-CI, versioned prompts, service metrics, and online quality scores.
- A standalone guardrails/safety product → **P08**.

## 3. Context and the scope decision

The roadmap positions RAG as a focused ~2-week Tier 1 project and already breaks the heavy pieces into separate projects (P05 eval, P06 corrective RAG, P08 guardrails, P12 router, P13 LLMOps). The original brief for this repo described a system that absorbed all of them — a capstone, not Project 01.

We resolved the tension in favor of the roadmap's "depth beats breadth" north star: a **strong core plus a small set of marquee features**, built in strict phases with a hard stop-line, so ambition never produces a sprawling half-built repo. Heavy multimodal work stays in P09; the two repos cross-reference each other.

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
| Observability & monitoring | Level B — Langfuse tracing + Prometheus/Grafana service metrics + online sampled quality scoring | Honors the `notes.txt` priority; covers LLM-native and SRE-style monitoring; alerting + drift stay in P13 |

## 5. Architecture and data flow

Two paths meet at one store. Ingestion only writes Postgres; the query path only reads it. They share a schema, not code — so you rebuild the index without touching the API, and load-test the API against a frozen index. Every box tagged `[Phase N]` is additive; Phase 0 is the unbroken flow minus the grader, router, and self-check.

```
INGESTION PATH  (offline / async — the "write" side)

  SOURCES                    one adapter → one normalized Document
  ├─ PDF        [Phase 0]
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
                  └─────────────────────────────────────────────────┘     incremental]

QUERY PATH  (online / sync-streaming — the "read" side)

  Client ─► FastAPI  /query (SSE stream)
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
        │ Context assembly │─► │  LLM generate │─► │ [Phase 2] groundedness   │
        │ (budget,dedup,   │   │  (streaming,  │   │  self-check gate         │
        │  citation map)   │   │   citations)  │   └────────────┬─────────────┘
        └──────────────────┘   └───────────────┘                ▼
                                                   streamed tokens + source citations

CROSS-CUTTING
  • Observability: Langfuse traces every node (latency / tokens / $ / quality) via LiteLLM callback
  • Monitoring: Prometheus /metrics + Grafana (req rate · p50/p95/p99 · errors · $/day);
                online sampled quality scoring (faithfulness/groundedness on live traffic → Langfuse)
  • Provider seam: LiteLLM (generation · embedding · rerank), swap by config string
  • Config: pydantic-settings, one typed Settings object
  • Eval harness: golden set → RAGAS metrics  ──[Phase 1]──► runs as CI gate
```

## 6. Component and interface contracts

Interfaces stay fixed across phases. Phases add implementations or wrap existing ones — they never change a signature. That property is what makes the stop-line real.

**Shared domain models**

```python
RawDocument   # source_id, source_type, uri, raw_bytes|text, fetched_at, source_meta
Document      # doc_id, text, structure(tables/sections), metadata{region,crop,season}, content_hash
Chunk         # chunk_id, doc_id, text, ordinal, metadata, token_count
ScoredChunk   # chunk + score + provenance("dense"|"lexical"|"fused"|"rerank")
Citation      # marker, doc_id, chunk_id, source_uri, span
Answer        # text, citations[], usage{tokens,$,latency}, trace_id
```

**Ingestion-path contracts**

```python
class SourceAdapter(Protocol):          # Phase 0: Pdf; Phase 3: Html, Api/DB
    source_type: str
    def fetch(self, since: Watermark | None) -> Iterator[RawDocument]: ...
    #  since=None → full scan (Phase 0); a watermark → incremental (Phase 3). Same signature.

class Parser(Protocol):    def parse(self, raw) -> Document      # OCR fallback, table extraction
class Cleaner(Protocol):   def clean(self, doc) -> Document
class Chunker(Protocol):   name; def chunk(self, doc) -> list[Chunk]   # strategy; eval picks the winner
class EmbeddingProvider(Protocol):  dim; def embed(self, texts) -> list[Vector]   # LiteLLM, batched + retried

class ChunkRepository(Protocol):        # the only thing that touches Postgres
    def upsert(self, chunks, vectors) -> UpsertStats     # idempotent via content_hash
    def get_watermark(self, source_id) -> Watermark | None
    def set_watermark(self, source_id, wm) -> None
    #  Phase-3 incremental falls out of upsert idempotency + watermarks. No new interface.
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

class Answerer:                   # orchestrates the read path; Phase 0 = a straight line
    def answer(self, query, filt) -> Iterator[AnswerEvent]   # yields tokens + citations (SSE)
```

**Phase-2 additions — they wrap, never replace**

```python
class RetrievalGrader(Protocol):     def grade(self, query, chunks) -> Grade       # cheap LLM: relevant/sufficient?
class QueryRewriter(Protocol):       def rewrite(self, query, reason) -> list[str]
class GroundednessChecker(Protocol): def check(self, answer, context) -> Verdict   # hallucination gate

# CorrectiveAnswerer wraps the SAME Retriever/Reranker/Answerer in a LangGraph loop:
#   retrieve → grade → (rewrite | web-fallback → retrieve)* → assemble → generate → self-check
# AdaptiveRouter: trivial query → plain Answerer; else → CorrectiveAnswerer.
# Both expose the identical answer() signature → the FastAPI layer never knows which ran.
```

**Phase-1 eval — a consumer of the pipeline, not part of it**

```python
class Evaluator:
    def run(self, dataset: GoldenSet, pipeline: Answerer) -> EvalReport
    #  retrieval: hit@k, MRR, nDCG   |   RAGAS: faithfulness, answer-relevance, context precision/recall
    #  make eval → JSON report → CI compares vs committed thresholds → gate (Phase 1)
```

The FastAPI route depends only on `Answerer.answer()`. Whether a plain straight-line (Phase 0) or an adaptive corrective loop (Phase 2) runs behind it is an injected detail — so every phase ships behind the same API, and every unit tests against fakes with zero network.

## 7. Phase plan (0 → 4)

Each phase is a finished, demoable artifact with a number. Start phase N+1 only when phase N's "Done when" holds and its metric is captured. If you stop anywhere, what exists is whole.

### Phase 0 — Core flagship / MVP · ~1.5–2 weeks

- **Ships:** PDF ingest (parse + tables + OCR fallback) → clean → chunk → embed → Postgres+pgvector upsert (metadata + FTS) → hybrid retrieve (dense+FTS, RRF) → Cohere rerank → context assembly with citations → streaming cited answers over FastAPI SSE → Langfuse traces + a Prometheus `/metrics` endpoint → `docker compose up` (api + postgres + langfuse + prometheus + grafana) → README with architecture diagram. Plus a 30–50 item golden set and a local `make eval` scorecard.
- **Done when:** one-command up, one-command ingest, ask a question → cited streamed answer; `make eval` prints RAGAS + retrieval metrics; the Langfuse trace and the Grafana dashboard both render.
- **Unlocks:** "Hybrid + cross-encoder rerank raised faithfulness X→Y and context-precision A→B; retrieval p95 < Z ms," plus a clean before/after rerank-lift number.

### Phase 1 — Eval-in-CI quality gate · ~3–5 days

- **Ships:** `make eval` promoted into GitHub Actions over a frozen mini-corpus + golden set; committed baseline thresholds; CI fails on faithfulness/precision regression and posts the scorecard as a PR comment; judge model pinned at temperature 0; a small human-labeled slice calibrates the judge. Also adds **online sampled quality scoring** — a configurable percentage of live queries get the cheap faithfulness judge, logged to Langfuse and surfaced on the dashboards (extended with the groundedness signal once Phase 2 lands).
- **Done when:** a PR that worsens retrieval turns CI red with a diffed scorecard; sampled live queries show a quality score in Langfuse.
- **Unlocks:** "Eval-gated CI blocking quality regressions; judge-vs-human agreement X%; answer quality monitored on live traffic." The strongest LLMOps signal and the dock for P13.

### Phase 2 — Corrective/self-reflective RAG + adaptive routing · ~1–1.5 weeks

- **Ships:** `RetrievalGrader` + `QueryRewriter` + web-search fallback (Tavily) + `GroundednessChecker`, wired into a LangGraph `CorrectiveAnswerer` with a max-iteration cap; `AdaptiveRouter` (cheap model: trivial → fast path, complex/low-confidence → corrective). Benchmarked against the Phase-0 baseline on a hard / out-of-corpus slice.
- **Done when:** the hard slice shows lower hallucination; the router proves most easy queries skip the loop.
- **Unlocks:** "Corrective grading + self-reflection cut hallucinated answers X% on out-of-corpus queries; adaptive routing held p50 latency flat by sending Y% down the fast path."

### Phase 3 — Multi-source + incremental ingestion · ~1 week

- **Ships:** `HtmlSourceAdapter` (trafilatura) + one `Api/DB` adapter through the same pipeline; the incremental path (watermark + content_hash → upsert only changed); a scheduled/worker re-index (arq); an optional small event-driven trigger (webhook or watched folder) to claim "event-driven" honestly.
- **Done when:** modifying one doc re-indexes only that doc (show upsert stats); the query reflects the change.
- **Unlocks:** "Multi-source ingestion (PDF+web+API) with incremental upsert re-indexing only changed docs — N docs/hr, M% of embedding calls saved vs full re-index."

### Phase 4 — Light geospatial-metadata edge · ~2–3 days

- **Ships:** region/crop/season tagging during ingestion (rules + LLM tagging); metadata filters exposed in `/query`; a demo of filtered retrieval ("answer using only South-region maize docs"); a cross-link to P09.
- **Done when:** a filtered query provably restricts retrieval to matching docs.
- **Unlocks:** "Geospatial-aware retrieval (region/crop/season filtering)" — the differentiation hook that seeds P09.

**Stop-line:** portfolio-proud from the end of Phase 1, flagship-proud from the end of Phase 2. Phases 3–4 are depth bonuses. Cumulative ≈ 5–6.5 weeks part-time for all five; ≈ 3 weeks reaches the end of Phase 2.

## 8. Tech stack per layer

| Layer | Pick | Why / trade-off |
|---|---|---|
| Runtime | Python 3.12 + uv | Fast, reproducible lockfile |
| PDF parse | Docling primary · PyMuPDF fast-path · Tesseract OCR fallback | Docling handles multi-column + tables; OCR only when the text layer is missing |
| HTML (P3) | trafilatura + httpx | Best-in-class boilerplate stripping |
| API/DB (P3) | httpx · SQLAlchemy Core | Thin, async, no ORM ceremony |
| Chunking | Own strategies (recursive/layout-aware + semantic), tiktoken for budgeting | Own it; the eval harness picks the winner |
| Model wrapper | LiteLLM | Unifies completion + embedding + rerank; Langfuse callback; cost tracking; Router |
| Embeddings | Google `text-embedding-004` (768-d) default; `gemini-embedding-001` for MRL dims | 768-d indexes cleanly under pgvector's limit |
| Vector + lexical + meta + state | Postgres 16 + pgvector 0.7 (HNSW), native FTS (tsvector/GIN) | One store, four jobs |
| Fusion | Reciprocal Rank Fusion (RRF) | Rank-based; no score normalization |
| Rerank | Cohere Rerank via `litellm.rerank()`; `bge-reranker-v2-m3` local swap | Gemini has no first-class rerank |
| Generation | `gemini/gemini-2.5-pro` behind LiteLLM | Quality where the user sees it |
| Grader / router / judge | `gemini/gemini-2.5-flash` | The loop and eval run many cheap calls |
| Corrective loop (P2) | LangGraph (only here) | State machine + checkpointing in a cyclic graph |
| Web fallback (P2) | Tavily | LLM-oriented search, clean JSON |
| API | FastAPI + uvicorn + sse-starlette, Pydantic v2 | Async streaming + typed contracts |
| Config | pydantic-settings | One typed Settings object; 12-factor |
| Observability | Langfuse (self-hosted) + structlog | Per-node cost/tokens/latency/quality via LiteLLM callback |
| Service metrics | prometheus-client (`/metrics`) + Prometheus + Grafana | SRE view: request rate, latency percentiles, error rate, $/day; complements Langfuse's call-level view |
| Online quality | Sampled live faithfulness/groundedness judge → Langfuse | Watches answer quality on real traffic, not just offline eval |
| Eval | RAGAS + custom hit@k/MRR/nDCG | Standard RAG metrics + retrieval metrics RAGAS omits |
| Ingest worker (P3) | arq (async Redis queue) or APScheduler for MVP | Light; Celery noted as scale-path, not built |
| Demo UI | Minimal static HTML/JS chat hitting SSE | Zero build; shows streaming + citations |
| Tests | pytest · pytest-asyncio · testcontainers (real pgvector) · respx (API mocks) | Unit on fakes, integration on ephemeral Postgres |
| Standards | ruff · mypy strict · pre-commit · Makefile · multi-stage Docker | The hygiene a reviewer skims for first |
| Deploy | docker-compose; K8s documented as scaling path | One command; real K8s deferred to P10/P13 |

**Sharp engineering details that prove depth**

1. **pgvector's index limit is 2000 dimensions.** A 3072-d embedding (`gemini-embedding-001` at full size) will not take an HNSW index naively. Two documented ways out: truncate via the Matryoshka `dimensions` parameter, or index the full vector with pgvector's `halfvec` (half-precision, up to 4096-d, ~half the index size, negligible recall loss). Defaulting to `text-embedding-004` at 768-d sidesteps it — but the README shows the trade.
2. **Postgres `ts_rank` is not BM25** — it is tf-idf-ish. The README says so, fuses it with dense via RRF (which ignores absolute scores), and names ParadeDB `pg_search` (real BM25 in Postgres via Tantivy) as the single-store upgrade. Most "hybrid BM25" repos mislabel this; this one will not.
3. **Eval determinism:** judge pinned, temperature 0, frozen mini-corpus → a red CI build signals a real regression, not judge noise.
4. **Cost discipline by design:** the cheap model serves the many grader/router/judge calls; the frontier model serves only the single user-facing generation — the exact seam P12 later exploits.

## 9. Repo structure

```
production-rag/
├── README.md              # problem → arch diagram → RESULTS scorecard → 1-cmd quickstart → demo
├── pyproject.toml · uv.lock · Makefile · .env.example
├── docker-compose.yml · Dockerfile        # api · postgres+pgvector · langfuse · prometheus · grafana · (worker·redis P3)
├── .github/workflows/  ci.yml (lint·type·unit·integration)   eval.yml (Phase-1 gate + PR scorecard)
├── docs/  architecture.md · decisions/ (ADR-lite) · superpowers/specs/<this spec>
├── eval/  golden_set.yaml (Q · reference · relevant-doc-ids)   baselines/ (committed thresholds)
├── monitoring/  prometheus.yml · grafana/ (dashboard json)
├── src/rag/
│   ├── config.py              # pydantic-settings Settings
│   ├── models.py              # Document · Chunk · ScoredChunk · Citation · Answer
│   ├── providers/             # LiteLLM-backed seams: llm.py · embeddings.py · rerank.py
│   ├── ingestion/
│   │   ├── sources/           # SourceAdapter: pdf.py [P0] · html.py · api.py [P3]
│   │   ├── parse.py · clean.py · embed.py · repository.py · pipeline.py
│   │   └── chunking/          # fixed.py · semantic.py · layout.py
│   ├── retrieval/  dense.py · lexical.py · hybrid.py (RRF + filter) · rerank.py
│   ├── generation/  assembler.py · answerer.py · prompts/ (versioned)
│   ├── corrective/ [P2]  grader.py · rewriter.py · websearch.py · groundedness.py · router.py · graph.py
│   ├── eval/  metrics.py (RAGAS + hit@k/MRR/nDCG) · runner.py
│   ├── observability/  tracing.py (langfuse + structlog + LiteLLM callback) · metrics.py (prometheus) · online_quality.py
│   └── api/  app.py · routes.py (/query SSE · /ingest · /healthz · /metrics) · schemas.py
├── workers/ [P3]  arq incremental-ingest worker
├── ui/            minimal static chat page (SSE + citations)
└── tests/  unit/ (fakes, no net) · integration/ (testcontainers pgvector + respx) · conftest.py
```

## 10. Engineering standards

mypy strict · ruff lint+format · pre-commit · 12-factor config · structlog with a `trace_id` on every log · ADR-lite decision records · Makefile one-liners (`make up/ingest/eval/test`) · multi-stage non-root Docker with healthcheck · secrets only via env (`.env.example` committed, `.env` never) · versioned prompt templates as files (seeds P13 prompt-versioning).

## 11. Testing strategy

- **Unit** — every interface against fakes (`FakeRetriever`, `FakeLLMProvider` yielding canned tokens, `FakeEmbedder`). Fast, no network. Test-first for the deterministic high-value logic: RRF fusion, citation mapping, chunkers, content-hash dedup + watermark.
- **Integration** — testcontainers boots a real Postgres+pgvector; verifies upsert idempotency, incremental re-index, and dense/lexical/hybrid retrieval over seeded data. respx mocks all LiteLLM/Cohere/Tavily HTTP, so CI spends nothing.
- **Eval-as-gate (P1)** — the golden-set eval runs in its own workflow over a frozen mini-corpus, judge pinned at temperature 0, for reproducible red/green.

## 12. Observability and monitoring

This repo treats observability and monitoring as a first-class concern (Level B), across two complementary layers, and leaves the heavy control-plane parts to P13.

- **Layer 1 — LLM-native (Langfuse).** The LiteLLM callback streams every model call into Langfuse: cost, tokens, latency, and quality scores, per node and per trace. Langfuse's dashboards give live cost/latency/quality views; curated screenshots become a README deliverable.
- **Layer 2 — service-level (Prometheus + Grafana).** The FastAPI app exposes a Prometheus `/metrics` endpoint (request rate, p50/p95/p99 latency, error rate, and a $/day cost counter). Prometheus scrapes it; a small Grafana dashboard ships in docker-compose. This is the SRE-style view Langfuse does not cover, and a recognizable backend/infra signal.
- **Online quality scoring.** Beyond offline eval, a configurable sample of live queries gets a cheap faithfulness judge (Phase 1), extended with the groundedness signal once Phase 2 lands. Scores log to Langfuse, so answer quality is watched on real traffic, not just in CI.
- **Deferred to P13:** threshold alerting (Slack/webhook on latency/cost/hallucination spikes) and input/output drift detection. This repo leaves the seams — metrics, online scores, and traces — for P13 to consume.

## 13. Failure modes — designed in, not patched on

- **Ingestion:** per-doc try/except → failed docs land in a dead-letter table with a reason, the batch continues; OCR fallback on an empty text layer; idempotent upsert makes re-runs safe; the watermark makes a run resumable.
- **Retrieval:** zero results or an empty filter → an explicit "no relevant context" path. Never hallucinate to fill silence.
- **Providers:** LiteLLM Router handles rate-limits and timeouts with retries plus model fallback; a per-request budget guard caps spend.
- **Generation:** token-budget overflow → the assembler truncates by relevance; a mid-stream error flushes the partial answer and emits an SSE error event.
- **Corrective loop (P2):** a hard max-iteration cap prevents infinite loops; if confidence stays low, the system returns its best answer flagged low-confidence with citations, never a silent fabrication; the groundedness gate forces one regeneration or an honest "insufficient evidence."
- **Measured, not assumed:** the golden set includes empty, adversarial, and out-of-corpus queries, so the failure paths are scored, not hoped for.

## 14. The README as a deliverable

problem → architecture diagram → results scorecard with the real numbers → one-command quickstart → key design decisions (the pgvector/RRF/BM25-honesty calls) → curated Langfuse + Grafana dashboard shots → 60–90s demo. That sequence converts "another RAG repo" into "this person ships production systems."

## 15. Implementation notes

- **LiteLLM:** consult the current LiteLLM docs before writing the provider, embedding, rerank, Langfuse-callback, and Router code. Those APIs move; do not code them from memory.
- **Gemini default:** the generation, grader/router/judge, and embedding models default to Gemini/Google, selected by config string so any provider swap is a one-line change.

## 16. Open questions and risks

- **Corpus assembly** — pick the exact public ag/agronomy sources and licensing before Phase 0 (candidates: agricultural research papers, FAO/USDA reports, extension-service guides). Risk: licensing or download friction; mitigate by selecting open-access sources.
- **Golden-set labeling effort** — 30–50 quality Q&A with reference answers and relevant-doc ids takes real time; budget for it inside Phase 0.
- **Rerank cost** — Cohere Rerank adds per-query cost; the local `bge-reranker` swap caps it if needed.
- **Gemini rate limits** — free-tier limits may throttle bulk embedding; LiteLLM Router retries/backoff plus batch sizing mitigate this.

## 17. Out of scope (cross-references)

Heavy VLM multimodal → **P09** · model serving + K8s → **P10** · cost-router + semantic cache → **P12** · full LLMOps control plane (threshold alerting, drift detection, prompt registry UI) → **P13** · standalone guardrails → **P08**. This repo leaves clean seams for each: the LiteLLM Router, versioned prompts, eval-in-CI, the Prometheus metrics, the online quality scores, and the `Retriever` interface.
