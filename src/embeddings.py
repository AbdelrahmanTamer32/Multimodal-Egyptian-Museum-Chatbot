from __future__ import annotations
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

_EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedder


def embed_query(text: str) -> np.ndarray:
    """
    E5 best practice: prefix with 'query:'
    """
    emb = get_embedder().encode([f"query: {text}"], normalize_embeddings=True)
    return emb.astype("float32")


def embed_passages(passages: List[str]) -> np.ndarray:
    """
    E5 best practice: prefix with 'passage:'
    """
    embs = get_embedder().encode([f"passage: {p}" for p in passages], normalize_embeddings=True)
    return embs.astype("float32")