import json
from pathlib import Path

import pytest

from evaluation.run import evaluate
from vectorstore import Chunk, build_retriever

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "evaluation" / "cases.json").read_text()
)
CHUNKS = [Chunk(**document) for document in FIXTURE["documents"]]


@pytest.mark.parametrize(
    "name", ["tfidf", "bm25", "dense_lsa", "hybrid", "hybrid_rerank"]
)
def test_retriever_finds_exact_policy(name):
    retriever = build_retriever(name)
    retriever.add_documents(CHUNKS)
    assert (
        retriever.retrieve("application log retention", k=1)[0].chunk.id == "retention"
    )


def test_benchmark_has_declared_scale_and_metrics():
    result = evaluate()
    assert result["corpus_documents"] == 20
    assert result["answerable_queries"] == 60
    assert result["unanswerable_queries"] == 12
    assert result["retrievers"]["hybrid_rerank"]["recall_at_3"] >= 0.8


def test_unknown_retriever_is_rejected():
    with pytest.raises(ValueError, match="Unknown retriever"):
        build_retriever("magic")
