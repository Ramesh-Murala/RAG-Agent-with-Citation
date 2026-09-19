"""Deterministic offline retrieval baselines for the agent and benchmark.

``DenseLSARetriever`` is a local latent-semantic baseline, not a neural
embedding model. Keeping that distinction explicit prevents benchmark results
from implying capabilities the project does not implement.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


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


class Retriever(Protocol):
    chunks: list[Chunk]

    def add_documents(self, chunks: list[Chunk]) -> None: ...

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]: ...


def _validate_documents(existing: list[Chunk], new: list[Chunk]) -> list[Chunk]:
    candidate = existing + new
    if len({c.id for c in candidate}) != len(candidate):
        raise ValueError("Chunk IDs must be unique")
    if any(not c.text.strip() for c in candidate):
        raise ValueError("Chunks must contain text")
    return candidate


def _top_results(
    chunks: list[Chunk], scores: np.ndarray, k: int
) -> list[RetrievedChunk]:
    if k < 1:
        raise ValueError("k must be positive")
    ranked = np.argsort(scores)[::-1][:k]
    return [
        RetrievedChunk(chunks[i], float(scores[i])) for i in ranked if scores[i] > 0
    ]


def simple_chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    overlap: int = 150,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    metadata = metadata or {}
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(str(uuid.uuid4())[:8], piece, source, metadata.copy()))
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


class TfidfRetriever:
    """Word and bigram TF-IDF lexical baseline."""

    def __init__(self):
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = None
        self.chunks: list[Chunk] = []

    def add_documents(self, chunks: list[Chunk]) -> None:
        candidate = _validate_documents(self.chunks, chunks)
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = (
            vectorizer.fit_transform([c.text for c in candidate]) if candidate else None
        )
        self.chunks, self._vectorizer, self._matrix = candidate, vectorizer, matrix

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k must be positive")
        if not self.chunks or self._matrix is None:
            return []
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix)[0]
        return _top_results(self.chunks, scores, k)


class BM25Retriever:
    """Small Okapi BM25 implementation with no search-service dependency."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0

    def add_documents(self, chunks: list[Chunk]) -> None:
        candidate = _validate_documents(self.chunks, chunks)
        tokens = [tokenize(c.text) for c in candidate]
        df: Counter[str] = Counter()
        for terms in tokens:
            df.update(set(terms))
        self.chunks, self._tokens, self._df = candidate, tokens, df
        self._avgdl = sum(map(len, tokens)) / len(tokens) if tokens else 0.0

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k must be positive")
        if not self.chunks:
            return []
        q_terms = tokenize(query)
        n_docs = len(self.chunks)
        scores = np.zeros(n_docs)
        for i, terms in enumerate(self._tokens):
            tf = Counter(terms)
            length_norm = 1 - self.b + self.b * len(terms) / max(self._avgdl, 1)
            for term in q_terms:
                freq = tf[term]
                if not freq:
                    continue
                idf = math.log(
                    1 + (n_docs - self._df[term] + 0.5) / (self._df[term] + 0.5)
                )
                scores[i] += idf * freq * (self.k1 + 1) / (freq + self.k1 * length_norm)
        return _top_results(self.chunks, scores, k)


class DenseLSARetriever:
    """Dense latent-semantic baseline built from local TF-IDF + SVD."""

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions
        self.chunks: list[Chunk] = []
        self._vectorizer = None
        self._projector = None
        self._matrix = None

    def add_documents(self, chunks: list[Chunk]) -> None:
        candidate = _validate_documents(self.chunks, chunks)
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        sparse = vectorizer.fit_transform([c.text for c in candidate])
        max_dimensions = min(sparse.shape) - 1
        if max_dimensions < 1:
            raise ValueError(
                "Dense LSA retrieval needs at least two documents and terms"
            )
        projector = make_pipeline(
            TruncatedSVD(
                n_components=min(self.dimensions, max_dimensions), random_state=42
            ),
            Normalizer(copy=False),
        )
        matrix = projector.fit_transform(sparse)
        self.chunks = candidate
        self._vectorizer, self._projector, self._matrix = vectorizer, projector, matrix

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k must be positive")
        if not self.chunks or self._matrix is None:
            return []
        query_sparse = self._vectorizer.transform([query])
        if query_sparse.nnz == 0:
            return []
        query_dense = self._projector.transform(query_sparse)
        scores = cosine_similarity(query_dense, self._matrix)[0]
        scores[scores < 1e-9] = 0
        return _top_results(self.chunks, scores, k)


class HybridRetriever:
    """Fuse BM25 and dense-LSA ranks, then optionally rerank candidates."""

    def __init__(self, rerank: bool = False, rrf_constant: int = 60):
        self.rerank = rerank
        self.rrf_constant = rrf_constant
        self.lexical = BM25Retriever()
        self.dense = DenseLSARetriever()
        self.chunks: list[Chunk] = []

    def add_documents(self, chunks: list[Chunk]) -> None:
        candidate = _validate_documents(self.chunks, chunks)
        lexical, dense = BM25Retriever(), DenseLSARetriever()
        lexical.add_documents(candidate)
        dense.add_documents(candidate)
        self.chunks, self.lexical, self.dense = candidate, lexical, dense

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k must be positive")
        candidate_k = min(len(self.chunks), max(k * 4, 10))
        scores: dict[str, float] = {}
        by_id = {c.id: c for c in self.chunks}
        for results in (
            self.lexical.retrieve(query, candidate_k),
            self.dense.retrieve(query, candidate_k),
        ):
            for rank, result in enumerate(results, 1):
                scores[result.chunk.id] = scores.get(result.chunk.id, 0) + 1 / (
                    self.rrf_constant + rank
                )
        if self.rerank:
            q_terms = set(tokenize(query))
            for chunk_id in scores:
                text_terms = set(tokenize(by_id[chunk_id].text))
                coverage = len(q_terms & text_terms) / max(len(q_terms), 1)
                phrase = 1.0 if query.lower() in by_id[chunk_id].text.lower() else 0.0
                scores[chunk_id] += 0.012 * coverage + 0.004 * phrase
        ranked = sorted(scores, key=lambda item: (-scores[item], item))[:k]
        return [RetrievedChunk(by_id[item], scores[item]) for item in ranked]


def build_retriever(name: str) -> Retriever:
    """Construct a benchmark/demo retriever by stable public name."""
    factories = {
        "tfidf": TfidfRetriever,
        "bm25": BM25Retriever,
        "dense_lsa": DenseLSARetriever,
        "hybrid": HybridRetriever,
        "hybrid_rerank": lambda: HybridRetriever(rerank=True),
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown retriever {name!r}; choose from {sorted(factories)}"
        ) from exc
