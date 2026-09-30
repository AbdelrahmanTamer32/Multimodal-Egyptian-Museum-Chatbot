from __future__ import annotations
from pathlib import Path
import json


def has_image(artifact: dict) -> bool:
    image_primary = (artifact.get("image_primary") or "").strip()
    image_small = (artifact.get("image_small") or "").strip()
    return bool(image_primary or image_small)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    test_path = root / "data" / "splits" / "artifacts_test.json"
    out_dir = root / "data" / "splits"

    with test_path.open("r", encoding="utf-8") as f:
        test_artifacts = json.load(f)

    total_test = len(test_artifacts)
    image_test_artifacts = [a for a in test_artifacts if has_image(a)]
    image_test_ids = [a["id"] for a in image_test_artifacts]

    with (out_dir / "image_test_ids.json").open("w", encoding="utf-8") as f:
        json.dump(image_test_ids, f, ensure_ascii=False, indent=2)

    print(f"Total test artifacts: {total_test}")
    print(f"Test artifacts with images: {len(image_test_artifacts)}")
    print(f"Test artifacts without images: {total_test - len(image_test_artifacts)}")
    print("Saved: data/splits/image_test_ids.json")


if __name__ == "__main__":
    main()