from __future__ import annotations
from pathlib import Path
import json


def has_image_path(artifact_id: str, image_meta: list[dict]) -> str:
    for item in image_meta:
        if item.get("id") == artifact_id:
            return (item.get("img_path") or "").strip()
    return ""


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    image_test_ids_path = root / "data" / "splits" / "image_test_ids.json"
    image_meta_path = root / "data" / "indexes" / "image_meta.json"
    out_dir = root / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    with image_test_ids_path.open("r", encoding="utf-8") as f:
        image_test_ids = json.load(f)

    with image_meta_path.open("r", encoding="utf-8") as f:
        image_meta = json.load(f)

    eval_samples = []

    for artifact_id in image_test_ids:
        img_path = has_image_path(artifact_id, image_meta)
        if not img_path:
            continue

        eval_samples.append({
            "image_path": img_path,
            "expected_id": artifact_id
        })

    out_path = out_dir / "image_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(eval_samples, f, ensure_ascii=False, indent=2)

    print(f"Image-capable test IDs: {len(image_test_ids)}")
    print(f"Image eval samples created: {len(eval_samples)}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()