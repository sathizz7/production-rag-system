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
