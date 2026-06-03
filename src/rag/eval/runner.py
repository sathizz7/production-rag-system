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
    scored on answer metrics only; retrieval metrics are skipped for them.
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

    pooled: dict[str, list[float]] = {key: [] for key in ("hit", "mrr", "ndcg", "faith", "rel")}
    for b in buckets.values():
        for key in pooled:
            pooled[key].extend(b[key])

    per_stratum = sorted((_score(s, b) for s, b in buckets.items()), key=lambda s: s.stratum)
    return [_score("ALL", pooled), *per_stratum]


def _aggregate(scores: list[StratumScore]) -> StratumScore | None:
    return next((s for s in scores if s.stratum == "ALL"), None)


def _rerank_delta(baseline: list[StratumScore], reranked: list[StratumScore]) -> str:
    """One-line rerank-lift summary on the ALL aggregate (spec deliverable)."""
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
        raise SystemExit("EVAL_DATABASE_URL is unset - create a dedicated rag_eval DB first.")
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
    # gemini-2.5-flash is a "thinking" model: a small budget is consumed by reasoning
    # and leaves no JSON, scoring 0. Give the judge headroom (verified: 256 -> 0.0, 2048 -> 1.0).
    judge = FaithfulnessJudge(LiteLLMProvider(model=settings.grader_model, max_tokens=2048))
    items = load_golden_set(golden_path)
    k = settings.retrieval_k

    # A/B: rerank lift = swap the base retriever. Both answerers are rerank-agnostic.
    baseline = evaluate(items, hybrid, _build_answerer(hybrid, settings), judge, k=k)
    out = [
        "# SMOKE EVAL - harness validation (small n; read the ALL row).",
        "# The statistically-honest 30-50 item stratified set is a Phase-1 gate.",
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
        reranked = evaluate(
            items, reranked_retriever, _build_answerer(reranked_retriever, settings), judge, k=k
        )
        out += [
            "",
            "## Hybrid + Cohere rerank",
            render_scorecard(reranked, k=k),
            "",
            _rerank_delta(baseline, reranked),
        ]
    else:
        out += [
            "",
            "(rerank A/B skipped - set RERANK_ENABLED=true + COHERE_API_KEY to measure lift)",
        ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG golden-set smoke eval (live).")
    parser.add_argument("--golden", type=Path, default=Path("eval/golden_set.yaml"))
    parser.add_argument("--corpus", type=Path, default=Path("eval/corpus"))
    args = parser.parse_args()
    print(_run_live(args.golden, args.corpus))


if __name__ == "__main__":
    main()
