"""Embedding strategy: compute and manage vector embeddings."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

import config

_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of texts, returning an array of shape (n, dim)."""
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


@lru_cache(maxsize=128)
def embed_query(query: str) -> np.ndarray:
    """Embed a single query string (cached)."""
    model = get_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )
    arr = np.array(embedding[0], dtype=np.float32)
    arr.flags.writeable = False
    return arr
