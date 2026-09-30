from __future__ import annotations
from pathlib import Path
import json

from src.image_retrieval import ImageRetriever


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    unknown_dir = root / "data" / "external_image_tests" / "unknown"
    out_path = root / "data" / "eval" / "external_unknown_image_summary.json"

    retriever = ImageRetriever(project_root=root)

    image_files = sorted([
        p for p in unknown_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])

    detailed_results = []
    top_scores = []

    for image_path in image_files:
        results = retriever.search(image_path, top_k=5)
        top_score = results[0]["score"] if results else 0.0
        top_scores.append(top_score)

        detailed_results.append({
            "image_path": str(image_path),
            "top_score": top_score,
            "retrieved_top5": results
        })

    summary = {
        "total_images": len(image_files),
        "average_top_score": sum(top_scores) / len(top_scores) if top_scores else 0.0,
        "min_top_score": min(top_scores) if top_scores else 0.0,
        "max_top_score": max(top_scores) if top_scores else 0.0,
        "detailed_results": detailed_results
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    main()