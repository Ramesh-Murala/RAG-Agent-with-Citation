#RAG Agent with Citation Grounding

Retrieve context, generate an answer with sources, flag low-confidence
responses, fall back to search. Same philosophy as Project #01: don't trust
the model to freeform an answer -- force it through a validated schema, and
feed validation errors back in for a corrective retry. Here the schema is
the citation set, and "validation" includes a **grounding check**: every
citation must point at a chunk that was actually retrieved, or the attempt
is rejected and retried.

## Why this prevents hallucination at scale

Two independent failure modes are handled separately:

1. **The model invents a fact not in the context.** Caught by asking the
   model to self-report `grounded: false` when the context is insufficient,
   combined with a retrieval-based confidence signal it can't fake.
2. **The model invents a citation** (cites a source that doesn't exist, or
   attributes a real claim to the wrong chunk). Caught structurally: every
   `chunk_id` in the response is checked against the set of chunk_ids that
   were actually retrieved. A hallucinated citation raises a
   `GroundingError`, which feeds the exact error text back to the model for
   a retry -- the same "specific corrective signal" pattern from Project #01.

## Architecture

```
vectorstore.py   Chunk, TfidfRetriever -- offline retrieval, no embedding API
agent.py         RAGCitationAgent      -- generate, validate, score, fall back
example.py       Runnable demo over a small fictional company knowledge base
```

### Retrieval (`vectorstore.py`)

Uses scikit-learn TF-IDF + cosine similarity instead of a real embedding
model. This is a deliberate scope cut: it means the whole retrieval half of
the pipeline runs offline with zero API calls and zero extra credentials,
which keeps the demo self-contained. It also means retrieval is a **lexical
overlap** signal, not semantic similarity -- it will miss a chunk that
answers the question using different words than the query. Swap
`TfidfRetriever` for a real vector DB (pgvector, Pinecone, Chroma,
Weaviate...) plus a real embedding model in production; the agent only
depends on the `retrieve(query, k) -> List[RetrievedChunk]` interface, so
the swap is contained to one class.

### Generation + grounding retry (`agent.py`)

`RAGCitationAgent.query()`:

1. Retrieve top-k chunks for the question.
2. Call Claude with `tool_choice` forced to a `submit_rag_answer` tool whose
   `input_schema` is generated directly from the `RAGAnswer` Pydantic model
   (answer, citations, self-reported confidence, grounded flag) -- same
   forced tool-calling pattern as Project #01.
3. Validate the response: JSON parse -> Pydantic schema -> grounding check
   (do all cited `chunk_id`s exist in the retrieved set?). Any failure
   appends the exact error text to the next prompt and retries, up to
   `max_retries` (default 2). If every attempt fails, the agent returns a
   safe `grounded=False, confidence=0.0` answer rather than crashing or
   silently returning bad data.
4. Blend a final confidence score from the retrieval score and the model's
   self-reported confidence (see below).
5. If confidence is low **and** a `fallback_search_fn` was provided, call it,
   append whatever it returns as extra context, and regenerate once. This
   only fires once per query -- it does not loop.

### Confidence heuristic

```
final_confidence = 0.35 * retrieval_score + 0.65 * self_reported_confidence
```
capped at 0.3 if `grounded=False`, and capped at 0.35 if there are no
citations at all. **This is a heuristic, not a calibrated probability** --
TF-IDF cosine scores aren't comparable across queries in any rigorous way,
and a model's self-reported confidence is itself an LLM output, not ground
truth. Treat `is_low_confidence` as "worth a second look or a fallback,"
not as a statistically validated uncertainty estimate. A production system
would want to calibrate this against labeled data (does confidence < 0.45
actually correlate with wrong answers on your domain?) rather than trust
the fixed weights above.

### Fallback search

`fallback_search_fn: Callable[[str], List[Chunk]]` is a pluggable seam, not
a batteries-included web search integration -- `example.py` wires up a
`mock_web_search_fallback()` that returns canned results for one keyword,
to keep the demo runnable with no extra API key. Replace it with a real
call to Tavily, Brave Search, Bing, or your internal search API; the agent
only needs back a `List[Chunk]`.

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python example.py
```

```python
from agent import RAGCitationAgent
from vectorstore import TfidfRetriever, simple_chunk_text

retriever = TfidfRetriever()
retriever.add_documents(simple_chunk_text(my_document_text, source="handbook.pdf"))

agent = RAGCitationAgent(retriever=retriever, fallback_search_fn=my_search_fn)
result = agent.query("What's our PTO rollover policy?")

print(result.answer.answer)
print(result.is_low_confidence, result.final_confidence)
for c in result.answer.citations:
    print(c.source, c.chunk_id, c.supporting_quote)
```

## Out of scope (available on request, same as Project #01)

- Async/batch querying across many questions at once
- Exponential backoff / rate-limit handling on the Anthropic API calls
- A formal test suite (the logic above was checked with mocked-client
  scripts during development, but nothing is committed as `pytest` tests)
- A real embedding-based retriever or real search API integration
- Persisting the document store (everything is in-memory per process)
- Multi-turn conversational RAG (each `query()` call is independent; no
  chat history is threaded through)
