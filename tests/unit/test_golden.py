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
