from __future__ import annotations
from pathlib import Path

from src.image_retrieval import ImageRetriever


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    unknown_dir = root / "data" / "external_image_tests" / "unknown"

    retriever = ImageRetriever(project_root=root)

    image_files = sorted([
        p for p in unknown_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])

    print(f"Total unknown external images: {len(image_files)}\n")

    for image_path in image_files:
        results = retriever.search(image_path, top_k=5)

        print(f"Image: {image_path}")
        print("Retrieved top 5:")
        for r in results:
            print(f"  - {r['id']} | {r['title']} | score={r['score']:.4f}")
        print("-" * 60)


if __name__ == "__main__":
    main()