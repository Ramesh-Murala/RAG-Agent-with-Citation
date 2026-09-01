"""
agent.py
Project #02 -- RAG Agent with Citation Grounding

Pipeline: retrieve -> generate a structured, cited answer -> score confidence
-> fall back to a pluggable search function when confidence is low.

Design carries over the philosophy from Project #01 (Structured Output
Agent): don't trust the model to freeform an answer, force it through a
validated schema, and feed validation errors back in for a corrective
retry. Here the schema is the citation set, and "validation" includes a
grounding check -- every citation must point at a chunk that was actually
retrieved, or the attempt is rejected and retried.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import anthropic
from pydantic import BaseModel, Field, ValidationError

from vectorstore import Chunk, RetrievedChunk, TfidfRetriever

logger = logging.getLogger("rag_citation_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL = "claude-sonnet-4-6"


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

class Citation(BaseModel):
    source: str = Field(..., description="The `source` label of the chunk this citation comes from")
    chunk_id: str = Field(..., description="The `chunk_id` of the chunk this citation comes from")
    supporting_quote: str = Field(
        ..., description="A short (<20 word) excerpt from the chunk that directly supports the claim"
    )


class RAGAnswer(BaseModel):
    answer: str = Field(..., description="The answer to the user's question, grounded only in the provided context")
    citations: List[Citation] = Field(
        default_factory=list, description="Citations backing each factual claim in the answer"
    )
    self_reported_confidence: float = Field(
        ..., ge=0, le=1, description="Model's own confidence that the answer is fully supported by the context"
    )
    grounded: bool = Field(
        ..., description="False if the context did not contain enough information to answer confidently"
    )


@dataclass
class AttemptRecord:
    attempt: int
    raw_output: str
    error: Optional[str] = None


@dataclass
class RAGResult:
    query: str
    answer: RAGAnswer
    retrieved_chunks: List[RetrievedChunk]
    retrieval_confidence: float
    final_confidence: float
    is_low_confidence: bool
    used_fallback: bool
    attempts: List[AttemptRecord] = field(default_factory=list)


class GroundingError(Exception):
    """Raised when the model cites a chunk_id that wasn't actually retrieved."""


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------

class RAGCitationAgent:
    def __init__(
        self,
        retriever: Optional[TfidfRetriever] = None,
        model: str = DEFAULT_MODEL,
        top_k: int = 4,
        low_confidence_threshold: float = 0.45,
        max_retries: int = 2,
        fallback_search_fn: Optional[Callable[[str], List[Chunk]]] = None,
        client: Optional[anthropic.Anthropic] = None,
    ):
        self.retriever = retriever or TfidfRetriever()
        self.model = model
        self.top_k = top_k
        self.low_confidence_threshold = low_confidence_threshold
        self.max_retries = max_retries
        self.fallback_search_fn = fallback_search_fn
        self.client = client or anthropic.Anthropic()

    def add_documents(self, chunks: List[Chunk]) -> None:
        self.retriever.add_documents(chunks)

    # ---- public API ----------------------------------------------------

    def query(self, question: str) -> RAGResult:
        retrieved = self.retriever.retrieve(question, k=self.top_k)
        retrieval_confidence = retrieved[0].score if retrieved else 0.0

        answer, attempts = self._generate_grounded_answer(question, retrieved)
        final_confidence = self._combine_confidence(retrieval_confidence, answer)
        is_low = self._is_low_confidence(final_confidence, answer, retrieved)
        used_fallback = False

        if is_low and self.fallback_search_fn is not None:
            logger.info("Low confidence (%.2f) for %r -- invoking fallback search", final_confidence, question)
            fallback_chunks = self.fallback_search_fn(question)
            if fallback_chunks:
                used_fallback = True
                combined = retrieved + [RetrievedChunk(chunk=c, score=0.0) for c in fallback_chunks]
                answer, retry_attempts = self._generate_grounded_answer(question, combined)
                attempts.extend(retry_attempts)
                retrieved = combined
                # retrieval_confidence intentionally NOT recomputed from fallback:
                # it's a different source type (e.g. live web), scored 0.0 by
                # convention so it never inflates the TF-IDF-based signal.
                final_confidence = self._combine_confidence(retrieval_confidence, answer)
                is_low = self._is_low_confidence(final_confidence, answer, retrieved)

        if is_low:
            logger.warning("Returning low-confidence answer for %r (confidence=%.2f)", question, final_confidence)

        return RAGResult(
            query=question,
            answer=answer,
            retrieved_chunks=retrieved,
            retrieval_confidence=retrieval_confidence,
            final_confidence=final_confidence,
            is_low_confidence=is_low,
            used_fallback=used_fallback,
            attempts=attempts,
        )

    # ---- generation + validation retry loop -----------------------------

    def _generate_grounded_answer(
        self, question: str, chunks: List[RetrievedChunk]
    ) -> tuple[RAGAnswer, List[AttemptRecord]]:
        valid_chunk_ids = {rc.chunk.id for rc in chunks}
        context_block = self._format_context(chunks)
        attempts: List[AttemptRecord] = []
        error_feedback = ""

        for attempt_num in range(1, self.max_retries + 2):  # first try + retries
            raw = self._call_llm(question, context_block, error_feedback)
            try:
                data = json.loads(raw)
                candidate = RAGAnswer.model_validate(data)
                self._check_grounding(candidate, valid_chunk_ids)
                attempts.append(AttemptRecord(attempt=attempt_num, raw_output=raw))
                return candidate, attempts
            except (json.JSONDecodeError, ValidationError, GroundingError) as exc:
                error_text = str(exc)
                logger.info("Attempt %d failed validation: %s", attempt_num, error_text)
                attempts.append(AttemptRecord(attempt=attempt_num, raw_output=raw, error=error_text))
                error_feedback = (
                    f"Your previous response failed validation with this error:\n{error_text}\n"
                    "Fix it and respond again with ONLY a valid tool call matching the schema. "
                    f"You may only cite these chunk_ids: {sorted(valid_chunk_ids)}."
                )

        # Exhausted retries: return a safe, explicitly ungrounded fallback
        # rather than raising, so a flaky final attempt doesn't crash the caller.
        logger.error("All %d attempts failed validation for query %r", self.max_retries + 1, question)
        fallback = RAGAnswer(
            answer="I wasn't able to produce a reliably grounded answer from the available context.",
            citations=[],
            self_reported_confidence=0.0,
            grounded=False,
        )
        return fallback, attempts

    def _check_grounding(self, answer: RAGAnswer, valid_chunk_ids: set) -> None:
        bad_ids = [c.chunk_id for c in answer.citations if c.chunk_id not in valid_chunk_ids]
        if bad_ids:
            raise GroundingError(
                f"Citation(s) reference chunk_id(s) not present in the retrieved context: {bad_ids}"
            )

    # ---- LLM call --------------------------------------------------------

    def _call_llm(self, question: str, context_block: str, error_feedback: str) -> str:
        system = (
            "You are a careful research assistant. Answer ONLY using the numbered "
            "context chunks provided. Every factual claim must be backed by a "
            "citation to a chunk_id that appears in the context. If the context "
            "does not contain enough information to answer, set grounded=false, "
            "give a low self_reported_confidence, and say so in the answer rather "
            "than guessing."
        )
        user_content = f"Context:\n{context_block}\n\nQuestion: {question}"
        if error_feedback:
            user_content += f"\n\n{error_feedback}"

        tool = {
            "name": "submit_rag_answer",
            "description": "Submit the grounded answer with citations.",
            "input_schema": RAGAnswer.model_json_schema(),
        }

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "submit_rag_answer"},
            messages=[{"role": "user", "content": user_content}],
        )

        for block in response.content:
            if block.type == "tool_use":
                return json.dumps(block.input)
        raise RuntimeError("Model did not return a tool_use block despite forced tool_choice")

    # ---- confidence heuristic --------------------------------------------

    def _combine_confidence(self, retrieval_confidence: float, answer: RAGAnswer) -> float:
        """
        Blend the bag-of-words retrieval score with the model's self-reported
        confidence. This is a heuristic, not a calibrated probability -- it's
        weighted toward the model's own assessment (0.65) since it can see
        whether the context actually answers the question, while the
        retrieval score alone only measures lexical overlap (0.35).
        """
        capped_retrieval = min(retrieval_confidence, 1.0)
        blended = 0.35 * capped_retrieval + 0.65 * answer.self_reported_confidence
        if not answer.grounded:
            blended = min(blended, 0.3)
        if not answer.citations:
            blended = min(blended, 0.35)
        return round(blended, 3)

    def _is_low_confidence(self, final_confidence: float, answer: RAGAnswer, chunks: List[RetrievedChunk]) -> bool:
        return (
            final_confidence < self.low_confidence_threshold
            or not answer.grounded
            or (not chunks and not answer.citations)
        )

    @staticmethod
    def _format_context(chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "(no context retrieved)"
        lines = []
        for rc in chunks:
            lines.append(
                f"[chunk_id={rc.chunk.id} source={rc.chunk.source!r}]\n{rc.chunk.text}"
            )
        return "\n\n".join(lines)
