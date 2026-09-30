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

    top1 = top1_hits / total if total else 0.0
    top5 = top5_hits / total if total else 0.0
    mrr = mrr_sum / total if total else 0.0

    print(f"Total external known images: {total}")
    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-5 Accuracy: {top5:.4f}")
    print(f"MRR: {mrr:.4f}")

    print("\nDetailed results:\n")
    for item in detailed_results:
        print(f"Image: {item['image_path']}")
        print(f"Expected ID: {item['expected_id']}")
        print(f"Top-1 correct: {item['top1_ok']}")
        print(f"Top-5 correct: {item['top5_ok']}")
        print("Retrieved top 5:")
        for r in item["retrieved_top5"]:
            print(f"  - {r['id']} | {r['title']} | score={r['score']:.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()