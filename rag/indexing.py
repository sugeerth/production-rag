"""Indexing: vector store (ChromaDB) and BM25 index."""

import os
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import logging
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
import numpy as np
from rank_bm25 import BM25Okapi

import config
from rag.ingestion import DocumentChunk
from rag.embeddings import embed_texts

_chroma_client = None
_bm25_index = None
_bm25_chunk_ids = []
_bm25_texts = []
_search_executor = ThreadPoolExecutor(max_workers=2)


def get_chroma_client() -> chromadb.ClientAPI:
    """Get or create the ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(config.CHROMA_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection():
    """Get or create the document collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[DocumentChunk]) -> dict:
    """Index chunks into ChromaDB and build BM25 index."""
    if not chunks:
        return {"indexed": 0}

    collection = get_collection()
    batch_size = 100
    total_indexed = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]
        metadatas = []
        for c in batch:
            meta = dict(c.metadata)
            # ChromaDB metadata values must be str, int, float, or bool
            for k, v in list(meta.items()):
                if isinstance(v, (list, dict)):
                    meta[k] = json.dumps(v)
            metadatas.append(meta)

        embeddings = embed_texts(texts).tolist()

        # Upsert to handle re-indexing
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total_indexed += len(batch)

    # Incrementally extend BM25 index
    new_ids = [c.chunk_id for c in chunks]
    new_texts = [c.text for c in chunks]
    _extend_bm25(new_ids, new_texts)

    return {"indexed": total_indexed, "collection_size": collection.count()}


def _rebuild_bm25():
    """Rebuild the BM25 index from all chunks in ChromaDB."""
    global _bm25_index, _bm25_chunk_ids, _bm25_texts

    collection = get_collection()
    count = collection.count()

    if count == 0:
        _bm25_index = None
        _bm25_chunk_ids = []
        _bm25_texts = []
        return

    # Get all documents
    results = collection.get(include=["documents"])
    _bm25_chunk_ids = results["ids"]
    _bm25_texts = results["documents"]

    # Tokenize for BM25
    tokenized = [doc.lower().split() for doc in _bm25_texts]
    _bm25_index = BM25Okapi(tokenized)


def _extend_bm25(new_ids: list[str], new_texts: list[str]):
    """Incrementally extend the BM25 index with new chunks (no ChromaDB fetch)."""
    global _bm25_index, _bm25_chunk_ids, _bm25_texts

    if _bm25_index is None and not _bm25_chunk_ids:
        # Cold start: just build from what we have
        _bm25_chunk_ids = list(new_ids)
        _bm25_texts = list(new_texts)
    else:
        # Append new entries (handle re-indexing by replacing existing ids)
        existing = set(_bm25_chunk_ids)
        for cid, text in zip(new_ids, new_texts):
            if cid in existing:
                idx = _bm25_chunk_ids.index(cid)
                _bm25_texts[idx] = text
            else:
                _bm25_chunk_ids.append(cid)
                _bm25_texts.append(text)

    if _bm25_texts:
        tokenized = [doc.lower().split() for doc in _bm25_texts]
        _bm25_index = BM25Okapi(tokenized)


def vector_search(query_embedding: np.ndarray, top_k: int = None,
                  filters: dict = None) -> list[dict]:
    """Search ChromaDB with a query embedding."""
    top_k = top_k or config.TOP_K_RETRIEVAL
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_params = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances", "embeddings"],
    }

    if filters:
        where = {}
        for k, v in filters.items():
            where[k] = v
        if where:
            query_params["where"] = where

    results = collection.query(**query_params)

    search_results = []
    for i in range(len(results["ids"][0])):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score: 1 - (distance/2)
        distance = results["distances"][0][i]
        score = 1.0 - (distance / 2.0)
        search_results.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": score,
            "source": "vector",
            "embedding": results["embeddings"][0][i],
        })

    return search_results


def bm25_search(query: str, top_k: int = None) -> list[dict]:
    """Search using BM25 (lexical/keyword search)."""
    global _bm25_index, _bm25_chunk_ids, _bm25_texts

    if _bm25_index is None:
        _rebuild_bm25()

    if _bm25_index is None or not _bm25_chunk_ids:
        return []

    top_k = top_k or config.TOP_K_RETRIEVAL
    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Filter to positive scores and collect IDs for batch fetch
    max_score = max(scores) if max(scores) > 0 else 1.0
    valid = [(idx, scores[idx] / max_score) for idx in top_indices if scores[idx] > 0]

    if not valid:
        return []

    batch_ids = [_bm25_chunk_ids[idx] for idx, _ in valid]
    collection = get_collection()
    batch_result = collection.get(ids=batch_ids, include=["metadatas", "embeddings"])

    # Build lookup dicts
    meta_lookup = dict(zip(batch_result["ids"], batch_result["metadatas"]))
    emb_lookup = dict(zip(batch_result["ids"], batch_result["embeddings"]))

    results = []
    for idx, normalized_score in valid:
        cid = _bm25_chunk_ids[idx]
        results.append({
            "chunk_id": cid,
            "text": _bm25_texts[idx],
            "metadata": meta_lookup.get(cid, {}),
            "score": normalized_score,
            "source": "bm25",
            "embedding": emb_lookup.get(cid),
        })

    return results


def hybrid_search(query_embedding: np.ndarray, query_text: str,
                  top_k: int = None, alpha: float = None) -> list[dict]:
    """Combine vector and BM25 search results with reciprocal rank fusion."""
    top_k = top_k or config.TOP_K_RETRIEVAL
    alpha = alpha if alpha is not None else config.HYBRID_ALPHA

    # Run vector and BM25 search in parallel
    vec_future = _search_executor.submit(vector_search, query_embedding, top_k)
    bm25_future = _search_executor.submit(bm25_search, query_text, top_k)
    vector_results = vec_future.result()
    bm25_results = bm25_future.result()

    # Reciprocal Rank Fusion
    k_rrf = 60  # RRF constant
    scores = {}
    texts = {}
    metadatas = {}
    embeddings = {}

    for rank, r in enumerate(vector_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + alpha * (1.0 / (k_rrf + rank + 1))
        texts[cid] = r["text"]
        metadatas[cid] = r["metadata"]
        if r.get("embedding") is not None:
            embeddings[cid] = r["embedding"]

    for rank, r in enumerate(bm25_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + (1 - alpha) * (1.0 / (k_rrf + rank + 1))
        texts[cid] = r["text"]
        metadatas[cid] = r["metadata"]
        if cid not in embeddings and r.get("embedding") is not None:
            embeddings[cid] = r["embedding"]

    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

    results = []
    for cid in sorted_ids:
        results.append({
            "chunk_id": cid,
            "text": texts[cid],
            "metadata": metadatas[cid],
            "score": scores[cid],
            "source": "hybrid",
            "embedding": embeddings.get(cid),
        })

    return results


def get_index_stats() -> dict:
    """Get statistics about the current index."""
    collection = get_collection()
    count = collection.count()
    return {
        "total_chunks": count,
        "bm25_indexed": len(_bm25_chunk_ids),
        "collection_name": config.COLLECTION_NAME,
    }


def clear_index():
    """Clear all indexed data."""
    global _bm25_index, _bm25_chunk_ids, _bm25_texts
    client = get_chroma_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    _bm25_index = None
    _bm25_chunk_ids = []
    _bm25_texts = []
