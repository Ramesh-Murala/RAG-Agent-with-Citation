"""
vectorstore.py
Lightweight in-memory document store with TF-IDF retrieval.

This deliberately avoids external embedding APIs so the retrieval half of
the pipeline runs fully offline, with no network dependency and no extra
API key. Swap `TfidfRetriever` for a real vector DB (pgvector, Pinecone,
Chroma, Weaviate...) in production -- see README "Extending" section for
where the seam is.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def simple_chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
    metadata: Optional[dict] = None,
) -> List[Chunk]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    metadata = metadata or {}
    text = re.sub(r"\s+", " ", text).strip()
    chunks: List[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(id=str(uuid.uuid4())[:8], text=piece, source=source, metadata=metadata))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


class TfidfRetriever:
    """Minimal TF-IDF + cosine-similarity retriever.

    Similarity scores are NOT a calibrated probability -- they're a bag-of-
    words overlap signal. The agent treats them as one input to a confidence
    heuristic, not ground truth. See README for why this matters.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        self.chunks: List[Chunk] = []

    def add_documents(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)
        self._reindex()

    def _reindex(self) -> None:
        if not self.chunks:
            self._matrix = None
            return
        texts = [c.text for c in self.chunks]
        self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        if not self.chunks or self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked_idx = np.argsort(scores)[::-1][:k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]))
            for i in ranked_idx
            if scores[i] > 0
        ]
