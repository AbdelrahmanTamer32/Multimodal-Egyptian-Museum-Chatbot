from __future__ import annotations
from pathlib import Path
import json

from src.kb import load_kb, build_passages
from src.retrieval import load_index, load_meta, retrieve


def reciprocal_rank(retrieved_ids: list[str], expected_id: str) -> float:
    for rank, artifact_id in enumerate(retrieved_ids, start=1):
        if artifact_id == expected_id:
            return 1.0 / rank
    return 0.0


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    kb_path = root / "data" / "kb" / "artifacts_kb.json"
    index_path = root / "data" / "indexes" / "faiss.index"
    meta_path = root / "data" / "indexes" / "meta.json"
    eval_path = root / "data" / "eval" / "text_eval.json"

    artifacts = load_kb(kb_path)
    passages, _ = build_passages(artifacts)
    index = load_index(index_path)
    meta = load_meta(meta_path)

    with eval_path.open("r", encoding="utf-8") as f:
        eval_samples = json.load(f)

    total = 0
    top1_hits = 0
    top5_hits = 0
    mrr_sum = 0.0

    for sample in eval_samples:
        query = sample["query"]
        expected_id = sample["expected_id"]

        results = retrieve(index, meta, passages, query, top_k=5)
        retrieved_ids = [r["meta"]["id"] for r in results]

        total += 1

        if len(retrieved_ids) >= 1 and retrieved_ids[0] == expected_id:
            top1_hits += 1

        if expected_id in retrieved_ids:
            top5_hits += 1

        mrr_sum += reciprocal_rank(retrieved_ids, expected_id)

    top1 = top1_hits / total if total else 0.0
    top5 = top5_hits / total if total else 0.0
    mrr = mrr_sum / total if total else 0.0

    print(f"Total queries: {total}")
    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")
    print(f"MRR: {mrr:.4f}")


if __name__ == "__main__":
    main()