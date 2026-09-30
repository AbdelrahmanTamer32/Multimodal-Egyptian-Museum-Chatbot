from __future__ import annotations
from pathlib import Path
import json

from src.image_retrieval import ImageRetriever


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    unknown_dir = root / "data" / "external_image_tests" / "unknown"
    out_path = root / "data" / "eval" / "external_unknown_image_results.json"

    retriever = ImageRetriever(project_root=root)

    image_files = sorted([
        p for p in unknown_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])

    all_results = []

    for image_path in image_files:
        results = retriever.search(image_path, top_k=5)

        all_results.append({
            "image_path": str(image_path),
            "top_score": results[0]["score"] if results else None,
            "retrieved_top5": results
        })

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Total unknown external images: {len(image_files)}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()