"""Run the deterministic retrieval and citation-integrity benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from agent import GroundingError, RAGAnswer, RAGCitationAgent
from vectorstore import Chunk, build_retriever

RETRIEVERS = ("tfidf", "bm25", "dense_lsa", "hybrid", "hybrid_rerank")


def evaluate_retriever(name: str, chunks: list[Chunk], queries: list[dict]) -> dict:
    retriever = build_retriever(name)
    index_started = time.perf_counter()
    retriever.add_documents(chunks)
    index_ms = (time.perf_counter() - index_started) * 1000
    rows, latencies = [], []
    for case in queries:
        started = time.perf_counter()
        results = retriever.retrieve(case["query"], k=3)
        latencies.append((time.perf_counter() - started) * 1000)
        ids = [item.chunk.id for item in results]
        relevant = set(case["relevant"])
        ranks = [rank for rank, chunk_id in enumerate(ids, 1) if chunk_id in relevant]
        rows.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "query": case["query"],
                "relevant": case["relevant"],
                "retrieved": ids,
                "top_1_hit": bool(ids and ids[0] in relevant) if relevant else None,
                "recall_at_3": len(relevant.intersection(ids)) / len(relevant)
                if relevant
                else None,
                "reciprocal_rank_at_3": 1 / min(ranks)
                if ranks
                else (0 if relevant else None),
                "no_result": not ids,
            }
        )
    answerable = [row for row in rows if row["relevant"]]
    unanswerable = [row for row in rows if not row["relevant"]]
    by_kind = {}
    for kind in ("exact", "paraphrase"):
        selected = [row for row in answerable if row["kind"] == kind]
        by_kind[kind] = {
            "queries": len(selected),
            "top_1_accuracy": round(
                sum(row["top_1_hit"] for row in selected) / len(selected), 4
            ),
            "recall_at_3": round(
                sum(row["recall_at_3"] for row in selected) / len(selected), 4
            ),
        }
    return {
        "index_ms": round(index_ms, 3),
        "median_query_ms": round(statistics.median(latencies), 3),
        "top_1_accuracy": round(
            sum(row["top_1_hit"] for row in answerable) / len(answerable), 4
        ),
        "recall_at_3": round(
            sum(row["recall_at_3"] for row in answerable) / len(answerable), 4
        ),
        "mrr_at_3": round(
            sum(row["reciprocal_rank_at_3"] for row in answerable) / len(answerable), 4
        ),
        "no_result_rate_unanswerable": round(
            sum(row["no_result"] for row in unanswerable) / len(unanswerable), 4
        ),
        "by_query_kind": by_kind,
        "top_1_miss_query_ids": [
            row["id"] for row in answerable if not row["top_1_hit"]
        ],
        "unanswerable_query_ids_with_results": [
            row["id"] for row in unanswerable if not row["no_result"]
        ],
    }


def citation_checks(chunks: list[Chunk]) -> list[dict]:
    agent = RAGCitationAgent(retriever=build_retriever("tfidf"), client=object())
    evidence = {chunk.id: chunk for chunk in chunks}
    base = {
        "answer": "Employees receive 20 days of leave.",
        "grounded": True,
        "self_reported_confidence": 0.9,
        "citations": [
            {
                "chunk_id": "leave",
                "source": "people-handbook",
                "supporting_quote": "20 days of paid annual leave",
            }
        ],
    }
    cases = [
        ("valid_evidence", base, True),
        (
            "invented_chunk",
            {**base, "citations": [{**base["citations"][0], "chunk_id": "missing"}]},
            False,
        ),
        (
            "wrong_source",
            {**base, "citations": [{**base["citations"][0], "source": "elsewhere"}]},
            False,
        ),
        (
            "fabricated_quote",
            {
                **base,
                "citations": [{**base["citations"][0], "supporting_quote": "90 days"}],
            },
            False,
        ),
        ("missing_citations", {**base, "citations": []}, False),
        (
            "wrong_answer_with_real_quote",
            {**base, "answer": "Employees receive 90 days of leave."},
            True,
        ),
    ]
    checks = []
    for name, data, expected in cases:
        try:
            agent._check_grounding(RAGAnswer(**data), evidence)
            accepted = True
        except GroundingError:
            accepted = False
        checks.append({"case": name, "accepted": accepted, "expected": expected})
    return checks


def evaluate() -> dict:
    fixture = json.loads(Path(__file__).with_name("cases.json").read_text())
    chunks = [Chunk(**document) for document in fixture["documents"]]
    benchmark = {
        name: evaluate_retriever(name, chunks, fixture["queries"])
        for name in RETRIEVERS
    }
    checks = citation_checks(chunks)
    return {
        "mode": "offline_synthetic_no_llm",
        "dataset": fixture["dataset"],
        "corpus_documents": len(chunks),
        "answerable_queries": sum(bool(q["relevant"]) for q in fixture["queries"]),
        "unanswerable_queries": sum(not q["relevant"] for q in fixture["queries"]),
        "retrievers": benchmark,
        "citation_checks": checks,
        "limitations": [
            "Synthetic single-relevant-document corpus; results do not estimate production quality.",
            "Dense LSA is local truncated SVD, not a pretrained neural embedding model.",
            "No-result rate is not calibrated abstention accuracy; score scales differ by retriever.",
            "Citation checks verify provenance and verbatim quotes, not claim-to-evidence entailment.",
            "Latency is a local single-process measurement and excludes model generation.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    result = evaluate()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.summary:
        print("retriever\ttop1\trecall@3\tmrr@3\tmedian_ms")
        for name, metrics in result["retrievers"].items():
            print(
                f"{name}\t{metrics['top_1_accuracy']:.3f}\t{metrics['recall_at_3']:.3f}"
                f"\t{metrics['mrr_at_3']:.3f}\t{metrics['median_query_ms']:.3f}"
            )
    else:
        print(json.dumps(result, indent=2))
    if any(
        check["accepted"] != check["expected"] for check in result["citation_checks"]
    ):
        raise SystemExit("Citation-integrity regression")
