from __future__ import annotations
from pathlib import Path
import json

from src.kb import load_kb, build_passages
from src.retrieval import load_index, load_meta, retrieve


def reciprocal_rank_exact(retrieved_ids: list[str], expected_id: str) -> float:
    for rank, artifact_id in enumerate(retrieved_ids, start=1):
        if artifact_id == expected_id:
            return 1.0 / rank
    return 0.0


def reciprocal_rank_title(retrieved_titles: list[str], expected_title: str) -> float:
    for rank, title in enumerate(retrieved_titles, start=1):
        if title == expected_title:
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

    id_to_title = {
        artifact["id"]: (artifact.get("title_en") or "").strip()
        for artifact in artifacts
        if artifact.get("id")
    }

    total = 0

    strict_top1_hits = 0
    strict_top5_hits = 0
    strict_mrr_sum = 0.0

    title_top1_hits = 0
    title_top5_hits = 0
    title_mrr_sum = 0.0

    for sample in eval_samples:
        query = sample["query"]
        expected_id = sample["expected_id"]
        expected_title = id_to_title.get(expected_id, "").strip()

        results = retrieve(index, meta, passages, query, top_k=5)
        retrieved_ids = [r["meta"]["id"] for r in results]
        retrieved_titles = [r["meta"]["title"] for r in results]

        total += 1

        # Strict metrics
        if len(retrieved_ids) >= 1 and retrieved_ids[0] == expected_id:
            strict_top1_hits += 1
        if expected_id in retrieved_ids:
            strict_top5_hits += 1
        strict_mrr_sum += reciprocal_rank_exact(retrieved_ids, expected_id)

        # Title-aware metrics
        if len(retrieved_titles) >= 1 and retrieved_titles[0] == expected_title:
            title_top1_hits += 1
        if expected_title in retrieved_titles:
            title_top5_hits += 1
        title_mrr_sum += reciprocal_rank_title(retrieved_titles, expected_title)

    print(f"Total queries: {total}")
    print()
    print("STRICT METRICS (exact ID match)")
    print(f"Top-1 Accuracy: {strict_top1_hits / total:.4f}")
    print(f"Top-5 Accuracy: {strict_top5_hits / total:.4f}")
    print(f"MRR: {strict_mrr_sum / total:.4f}")
    print()
    print("TITLE-AWARE METRICS (same title match)")
    print(f"Top-1 Accuracy: {title_top1_hits / total:.4f}")
    print(f"Top-5 Accuracy: {title_top5_hits / total:.4f}")
    print(f"MRR: {title_mrr_sum / total:.4f}")


if __name__ == "__main__":
    main()