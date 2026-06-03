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
