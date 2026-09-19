from types import SimpleNamespace

import pytest

from agent import GroundingError, RAGAnswer, RAGCitationAgent
from vectorstore import Chunk, TfidfRetriever, simple_chunk_text


CHUNK = Chunk(id="pto", source="handbook", text="Employees receive 20 days of paid leave each year.")


def answer(**changes):
    data = dict(answer="Employees receive 20 days of paid leave.", grounded=True,
                self_reported_confidence=0.9,
                citations=[dict(chunk_id="pto", source="handbook", supporting_quote="20 days of paid leave")])
    data.update(changes)
    return data


class FakeClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=next(self.outputs))])


def make_agent(outputs, **kwargs):
    retriever = TfidfRetriever()
    retriever.add_documents([CHUNK])
    client = FakeClient(outputs)
    return RAGCitationAgent(retriever=retriever, client=client, **kwargs), client


@pytest.mark.parametrize("citation", [
    dict(chunk_id="invented", source="handbook", supporting_quote="20 days"),
    dict(chunk_id="pto", source="invented", supporting_quote="20 days"),
    dict(chunk_id="pto", source="handbook", supporting_quote="90 days"),
    dict(chunk_id="pto", source="handbook", supporting_quote="   "),
])
def test_rejects_invalid_evidence(citation):
    agent, _ = make_agent([])
    with pytest.raises(GroundingError):
        agent._check_grounding(RAGAnswer(**answer(citations=[citation])), {CHUNK.id: CHUNK})


def test_retry_uses_validation_feedback():
    agent, client = make_agent([answer(citations=[]), answer()])
    result = agent.query("paid leave")
    assert result.answer.grounded
    assert len(result.attempts) == 2
    assert "must include at least one citation" in client.calls[1]["messages"][0]["content"]


def test_exhaustion_abstains():
    agent, client = make_agent([answer(citations=[])] * 2, max_retries=1)
    result = agent.query("paid leave")
    assert not result.answer.grounded
    assert result.final_confidence <= 0.3
    assert len(client.calls) == 2


def test_missing_context_skips_generation():
    agent, client = make_agent([])
    result = agent.query("spaceships")
    assert not result.answer.grounded
    assert result.is_low_confidence
    assert not client.calls


def test_fallback_runs_once_and_validates_new_sources():
    calls = []
    def search(query):
        calls.append(query)
        return [Chunk(id="space", source="search", text="Spaceships require fuel.")]
    data = answer(citations=[dict(chunk_id="space", source="search", supporting_quote="Spaceships require fuel.")])
    agent, client = make_agent([data], fallback_search_fn=search)
    result = agent.query("spaceships")
    assert result.used_fallback and result.answer.grounded
    assert len(calls) == len(client.calls) == 1


def test_valid_quote_does_not_prove_answer_entailment():
    # Deliberate counterexample: this structural checker cannot catch a wrong
    # answer accompanied by a real, but contradictory, quotation.
    agent, _ = make_agent([])
    candidate = RAGAnswer(**answer(answer="Employees receive 90 days of paid leave."))
    agent._check_grounding(candidate, {CHUNK.id: CHUNK})


def test_whitespace_normalization():
    agent, _ = make_agent([])
    data = answer(citations=[dict(chunk_id="pto", source="handbook", supporting_quote="20  days\nof paid leave")])
    agent._check_grounding(RAGAnswer(**data), {CHUNK.id: CHUNK})


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"max_retries": -1}, {"low_confidence_threshold": float("nan")}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        make_agent([], **kwargs)


def test_retriever_rejects_duplicate_ids_without_changing_index():
    r = TfidfRetriever()
    r.add_documents([CHUNK])
    with pytest.raises(ValueError):
        r.add_documents([CHUNK])
    assert len(r.retrieve("leave")) == 1


def test_invalid_chunking_configuration():
    with pytest.raises(ValueError):
        simple_chunk_text("a long document", "test", chunk_size=5, overlap=5)
