"""
example.py
Demo for Project #02 -- RAG Agent with Citation Grounding.

Builds a tiny internal knowledge base (fictional company policies), then
runs three questions chosen to exercise the three paths through the agent:

  1. Well-covered question       -> high confidence, clean citations.
  2. Partially-covered question  -> low confidence, fallback search kicks in.
  3. Out-of-scope question       -> low confidence, fallback finds nothing,
                                     agent honestly says it doesn't know.

Requires ANTHROPIC_API_KEY to be set in the environment.
"""

from __future__ import annotations

import os
import re

from agent import RAGCitationAgent, RAGResult
from vectorstore import Chunk, TfidfRetriever, simple_chunk_text

KNOWLEDGE_BASE = [
    (
        "pto_policy.md",
        (
            "Full-time employees accrue 15 days of paid time off (PTO) per year, "
            "credited monthly at a rate of 1.25 days/month. Up to 5 unused PTO "
            "days may be rolled over into the next calendar year; anything beyond "
            "that is forfeited on December 31st. PTO requests must be submitted "
            "through the HR portal at least 5 business days in advance."
        ),
    ),
    (
        "expense_policy.md",
        (
            "Employees may expense reasonable business travel costs including "
            "airfare, lodging, and ground transportation. Meals are reimbursed up "
            "to $75/day while traveling. All expenses over $25 require an itemized "
            "receipt uploaded to the finance portal within 30 days of the "
            "purchase. Personal entertainment expenses are never reimbursable."
        ),
    ),
    (
        "remote_work_policy.md",
        (
            "Employees may work remotely up to 3 days per week with manager "
            "approval. Fully remote arrangements require VP-level sign-off and "
            "are reviewed quarterly. The company does not currently provide a "
            "stipend for home office equipment."
        ),
    ),
]


def build_knowledge_base() -> TfidfRetriever:
    retriever = TfidfRetriever()
    for source, text in KNOWLEDGE_BASE:
        retriever.add_documents(simple_chunk_text(text, source=source))
    return retriever


def mock_web_search_fallback(query: str):
    """
    Stand-in for a real search API (Tavily, Brave Search, Bing, etc). Swap
    this out for an actual HTTP call in production -- the agent only cares
    that it gets back a list of Chunk objects.
    """
    lookup = {
        "sabbatical": (
            "web_search:hr_blog",
            (
                "According to common industry practice, sabbaticals are typically "
                "offered after 5-7 years of tenure, but this varies widely by "
                "company and is not something we can confirm from internal docs."
            ),
        ),
    }
    for keyword, (source, text) in lookup.items():
        if keyword in query.lower():
            return [Chunk(id=f"fallback-{keyword}", text=text, source=source)]
    return []  # nothing found -- agent should say so honestly


class FakeToolUse:
    def __init__(self, payload):
        self.type = "tool_use"
        self.input = payload


class FakeMessages:
    def __init__(self, responder):
        self.responder = responder

    def create(self, **kwargs):
        user_content = kwargs["messages"][0]["content"]
        question_match = re.search(r"Question:\s*(.*)", user_content, re.DOTALL)
        context_block = user_content.split("Context:\n", 1)[1].split(
            "\n\nQuestion:", 1
        )[0]
        question = question_match.group(1).strip() if question_match else ""
        payload = self.responder(question, context_block)
        return type("Response", (), {"content": [FakeToolUse(payload)]})()


class FakeAnthropicClient:
    def __init__(self, responder):
        self.messages = FakeMessages(responder)


def build_fake_response(question: str, context_block: str):
    q = question.lower()
    chunk_ids = re.findall(r"chunk_id=([^\s\]]+)", context_block)
    if not chunk_ids:
        chunk_ids = ["fake-chunk"]

    def citation_for(text: str, chunk_id: str):
        return {
            "source": "knowledge_base",
            "chunk_id": chunk_id,
            "supporting_quote": text,
        }

    if "pto" in q or "paid time off" in q or "roll over" in q:
        return {
            "answer": "Full-time employees receive 15 days of PTO per year and can roll over up to 5 unused days into the next calendar year.",
            "citations": [
                citation_for(
                    "Full-time employees accrue 15 days of paid time off (PTO) per year.",
                    chunk_ids[0],
                ),
                citation_for(
                    "Up to 5 unused PTO days may be rolled over into the next calendar year; anything beyond that is forfeited on December 31st.",
                    chunk_ids[0],
                ),
            ],
            "self_reported_confidence": 0.9,
            "grounded": True,
        }

    if "sabbatical" in q:
        return {
            "answer": "The available internal documents do not contain a sabbatical policy. The fallback search also found no internal policy for this topic.",
            "citations": [],
            "self_reported_confidence": 0.1,
            "grounded": False,
        }

    return {
        "answer": "I couldn't find a policy in the available local knowledge base for that question.",
        "citations": [],
        "self_reported_confidence": 0.1,
        "grounded": False,
    }


def print_result(result: RAGResult) -> None:
    print(f"\n{'=' * 70}")
    print(f"Q: {result.query}")
    print(f"{'=' * 70}")
    print(f"Answer: {result.answer.answer}")
    print(f"Grounded: {result.answer.grounded}")
    print(f"Retrieval confidence: {result.retrieval_confidence:.2f}")
    print(f"Self-reported confidence: {result.answer.self_reported_confidence:.2f}")
    print(f"Final blended confidence: {result.final_confidence:.2f}")
    print(f"Low confidence flag: {result.is_low_confidence}")
    print(f"Used fallback search: {result.used_fallback}")
    if result.answer.citations:
        print("Citations:")
        for c in result.answer.citations:
            print(f'  - [{c.source} / {c.chunk_id}] "{c.supporting_quote}"')
    else:
        print("Citations: (none)")
    if len(result.attempts) > 1:
        print(f"(Needed {len(result.attempts)} attempts due to validation retries)")


def main() -> None:
    use_fake_llm = not os.environ.get("ANTHROPIC_API_KEY")

    retriever = build_knowledge_base()
    client = FakeAnthropicClient(build_fake_response) if use_fake_llm else None
    agent = RAGCitationAgent(
        retriever=retriever,
        top_k=3,
        low_confidence_threshold=0.45,
        fallback_search_fn=mock_web_search_fallback,
        client=client,
    )

    if use_fake_llm:
        print(
            "Running in local offline mode with a fake LLM client (no API key required)."
        )

    questions = [
        "How many PTO days do full-time employees get, and can they roll them over?",
        "What is the company's sabbatical leave policy?",
        "What's the parking reimbursement policy for the Austin office?",
    ]

    for q in questions:
        result = agent.query(q)
        print_result(result)


if __name__ == "__main__":
    main()
