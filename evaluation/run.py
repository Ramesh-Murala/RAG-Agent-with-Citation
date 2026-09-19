"""Deterministic retrieval and citation-integrity checks; no model/API calls."""
import argparse
import json
from pathlib import Path

from agent import GroundingError, RAGAnswer, RAGCitationAgent
from vectorstore import Chunk, TfidfRetriever


def evaluate():
    fixture = json.loads(Path(__file__).with_name("cases.json").read_text())
    retriever = TfidfRetriever()
    chunks = [Chunk(**d) for d in fixture["documents"]]
    retriever.add_documents(chunks)
    rows = []
    for case in fixture["queries"]:
        ids = [r.chunk.id for r in retriever.retrieve(case["query"], k=3)]
        relevant = set(case["relevant"])
        ranks = [i + 1 for i, id_ in enumerate(ids) if id_ in relevant]
        rows.append(dict(query=case["query"], relevant=case["relevant"], retrieved=ids,
                         recall_at_3=len(relevant.intersection(ids)) / len(relevant) if relevant else None,
                         reciprocal_rank=1 / min(ranks) if ranks else 0,
                         empty_retrieval=not ids))
    answerable = [r for r in rows if r["relevant"]]
    unanswerable = [r for r in rows if not r["relevant"]]
    agent = RAGCitationAgent(retriever=retriever, client=object())
    base = dict(answer="20 days of leave", grounded=True, self_reported_confidence=0.9,
                citations=[dict(chunk_id="leave", source="handbook", supporting_quote="20 days of paid leave")])
    cases = [
        ("valid_evidence", base, True),
        ("invented_chunk", {**base, "citations": [dict(chunk_id="missing", source="handbook", supporting_quote="20 days")]}, False),
        ("wrong_source", {**base, "citations": [dict(chunk_id="leave", source="elsewhere", supporting_quote="20 days")]}, False),
        ("fabricated_quote", {**base, "citations": [dict(chunk_id="leave", source="handbook", supporting_quote="90 days")]}, False),
        ("missing_citations", {**base, "citations": []}, False),
        ("wrong_answer_with_real_quote", {**base, "answer": "Employees receive 90 days of leave."}, True),
    ]
    checks = []
    for name, data, expected in cases:
        try:
            agent._check_grounding(RAGAnswer(**data), {c.id: c for c in chunks})
            accepted = True
        except GroundingError:
            accepted = False
        checks.append(dict(case=name, accepted=accepted, expected=expected))
    return dict(
        mode="offline_fixture_no_llm", corpus_documents=len(chunks), answerable_queries=len(answerable),
        unanswerable_queries=len(unanswerable),
        recall_at_3=round(sum(r["recall_at_3"] for r in answerable) / len(answerable), 4),
        mrr_at_3=round(sum(r["reciprocal_rank"] for r in answerable) / len(answerable), 4),
        empty_retrieval_rate_on_unanswerable=sum(r["empty_retrieval"] for r in unanswerable) / len(unanswerable),
        queries=rows, citation_checks=checks,
        limitation="Hand-authored tiny corpus; no generation-quality, scale, cost, or semantic-entailment claim. A real quote can accompany a false answer.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate()
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)
    if any(c["accepted"] != c["expected"] for c in result["citation_checks"]):
        raise SystemExit("Citation-integrity regression")
