from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
import faiss

from src.embeddings_bge_large_en_v15 import embed_passages, embed_query


def build_faiss_index(passages: List[str]) -> faiss.Index:
    passage_embs = embed_passages(passages)
    dim = passage_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embs)
    return index


def save_index(index: faiss.Index, index_path: str | Path) -> None:
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_index(index_path: str | Path) -> faiss.Index:
    return faiss.read_index(str(index_path))


def save_meta(meta: List[Dict[str, Any]], meta_path: str | Path) -> None:
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta(meta_path: str | Path) -> List[Dict[str, Any]]:
    meta_path = Path(meta_path)
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def retrieve(index: faiss.Index, meta: List[Dict[str, Any]], passages: List[str], query: str, top_k: int = 3):
    q_emb = embed_query(query)
    scores, idxs = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        results.append({
            "score": float(score),
            "passage": passages[idx],
            "meta": meta[idx]
        })
    return results