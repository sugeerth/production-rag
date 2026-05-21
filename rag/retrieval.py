"""Retrieval: query understanding, search, and reranking."""

import re
from dataclasses import dataclass

import numpy as np

import config
from rag.embeddings import embed_query
from rag.indexing import hybrid_search, vector_search


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    chunks: list[dict]
    query_original: str
    query_rewritten: str
    retrieval_scores: list[float]
    top_score: float
    is_confident: bool


# Pre-compiled regex patterns for query expansion
_EXPANSIONS = [
    (re.compile(r"\bAPI\b"), "API Application Programming Interface"),
    (re.compile(r"\bML\b"), "ML Machine Learning"),
    (re.compile(r"\bNLP\b"), "NLP Natural Language Processing"),
    (re.compile(r"\bRAG\b"), "RAG Retrieval Augmented Generation"),
    (re.compile(r"\bDB\b"), "DB Database"),
    (re.compile(r"\bUI\b"), "UI User Interface"),
    (re.compile(r"\bLLM\b"), "LLM Large Language Model"),
]


def rewrite_query(query: str) -> str:
    """Simple query rewriting: clean up and expand the query.

    In production, you'd use an LLM for this. Here we do rule-based cleanup.
    """
    # Remove filler words for search
    query = query.strip().rstrip("?").strip()

    expanded = query
    for pattern, replacement in _EXPANSIONS:
        expanded = pattern.sub(replacement, expanded)

    return expanded


def rerank_results(query_embedding: np.ndarray, results: list[dict],
                   top_k: int = None) -> list[dict]:
    """Rerank results using embedding similarity.

    Uses pre-fetched embeddings from search results to avoid redundant
    embedding computations.
    """
    top_k = top_k or config.TOP_K_RERANK

    if not results:
        return []

    # Use pre-fetched embeddings from search results
    chunk_embs = np.array([r["embedding"] for r in results], dtype=np.float32)

    # Compute cosine similarity (embeddings are already normalized)
    similarities = np.dot(chunk_embs, query_embedding)

    # Combine original score with re-ranking score
    for i, result in enumerate(results):
        original_score = result.get("score", 0)
        rerank_score = float(similarities[i])
        # Weighted combination: favor rerank score
        result["rerank_score"] = rerank_score
        result["combined_score"] = 0.4 * original_score + 0.6 * rerank_score

    # Sort by combined score
    results.sort(key=lambda x: x["combined_score"], reverse=True)

    return results[:top_k]


def retrieve(query: str, top_k: int = None,
             use_reranking: bool = True) -> RetrievalResult:
    """Full retrieval pipeline: rewrite -> search -> rerank."""
    top_k_final = top_k or config.TOP_K_RERANK

    # Step 1: Query understanding / rewriting
    rewritten = rewrite_query(query)

    # Step 2: Embed and search
    query_embedding = embed_query(rewritten)
    candidates = hybrid_search(query_embedding, rewritten, top_k=config.TOP_K_RETRIEVAL)

    if not candidates:
        return RetrievalResult(
            chunks=[],
            query_original=query,
            query_rewritten=rewritten,
            retrieval_scores=[],
            top_score=0.0,
            is_confident=False,
        )

    # Step 3: Reranking (pass query_embedding to avoid re-embedding)
    if use_reranking and len(candidates) > top_k_final:
        ranked = rerank_results(query_embedding, candidates, top_k=top_k_final)
    else:
        ranked = candidates[:top_k_final]

    # Compute confidence
    scores = [r.get("combined_score", r.get("score", 0)) for r in ranked]
    top_score = max(scores) if scores else 0.0
    is_confident = top_score >= config.RETRIEVAL_SCORE_THRESHOLD

    return RetrievalResult(
        chunks=ranked,
        query_original=query,
        query_rewritten=rewritten,
        retrieval_scores=scores,
        top_score=top_score,
        is_confident=is_confident,
    )
