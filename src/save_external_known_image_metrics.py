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
    manifest_path = root / "data" / "external_image_tests" / "known_manifest.json"
    out_path = root / "data" / "eval" / "external_known_image_metrics.json"

    retriever = ImageRetriever(project_root=root)

    with manifest_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    total = 0
    top1_hits = 0
    top5_hits = 0
    mrr_sum = 0.0

    detailed_results = []

    for sample in samples:
        image_path = root / sample["image_path"]
        expected_id = sample["expected_id"]

        results = retriever.search(image_path, top_k=5)
        retrieved_ids = [r["id"] for r in results]

        total += 1

        top1_ok = len(retrieved_ids) >= 1 and retrieved_ids[0] == expected_id
        top5_ok = expected_id in retrieved_ids
        rr = reciprocal_rank(retrieved_ids, expected_id)

        if top1_ok:
            top1_hits += 1
        if top5_ok:
            top5_hits += 1
        mrr_sum += rr

        detailed_results.append({
            "image_path": str(image_path),
            "expected_id": expected_id,
            "top1_ok": top1_ok,
            "top5_ok": top5_ok,
            "retrieved_top5": results
        })

    metrics = {
        "total_images": total,
        "top1_accuracy": top1_hits / total if total else 0.0,
        "top5_accuracy": top5_hits / total if total else 0.0,
        "mrr": mrr_sum / total if total else 0.0,
        "detailed_results": detailed_results
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()