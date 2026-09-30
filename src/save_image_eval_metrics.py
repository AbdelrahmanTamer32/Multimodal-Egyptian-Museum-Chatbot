from __future__ import annotations
from pathlib import Path
import json

from src.image_retrieval import ImageRetriever


def reciprocal_rank(retrieved_ids: list[str], expected_id: str) -> float:
    for rank, artifact_id in enumerate(retrieved_ids, start=1):
        if artifact_id == expected_id:
            return 1.0 / rank
    return 0.0


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    eval_path = root / "data" / "eval" / "image_eval.json"
    out_path = root / "data" / "eval" / "image_eval_metrics.json"

    retriever = ImageRetriever(project_root=root)

    with eval_path.open("r", encoding="utf-8") as f:
        eval_samples = json.load(f)

    total = 0
    top1_hits = 0
    top5_hits = 0
    mrr_sum = 0.0

    for sample in eval_samples:
        image_path = sample["image_path"]
        expected_id = sample["expected_id"]

        results = retriever.search(image_path, top_k=5)
        retrieved_ids = [r["id"] for r in results]

        total += 1

        if len(retrieved_ids) >= 1 and retrieved_ids[0] == expected_id:
            top1_hits += 1

        if expected_id in retrieved_ids:
            top5_hits += 1

        mrr_sum += reciprocal_rank(retrieved_ids, expected_id)

    metrics = {
        "total_images": total,
        "top1_accuracy": top1_hits / total if total else 0.0,
        "top5_accuracy": top5_hits / total if total else 0.0,
        "mrr": mrr_sum / total if total else 0.0
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()