from __future__ import annotations
from pathlib import Path
import json
import random


def main() -> None:
    # Project root = parent of /src
    root = Path(__file__).resolve().parent.parent

    kb_path = root / "data" / "kb" / "artifacts_kb_with_images.json"
    out_dir = root / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ratio = 0.2
    seed = 42

    # Load KB
    with kb_path.open("r", encoding="utf-8") as f:
        artifacts = json.load(f)

    if not isinstance(artifacts, list):
        raise ValueError("Expected artifacts_kb_with_images.json to contain a list of artifacts.")

    # Keep only entries that have an ID
    artifacts = [a for a in artifacts if isinstance(a, dict) and a.get("id")]

    if not artifacts:
        raise ValueError("No valid artifacts with 'id' found in KB.")

    # Shuffle reproducibly
    random.seed(seed)
    random.shuffle(artifacts)

    # Split
    n_total = len(artifacts)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test

    train_artifacts = artifacts[:n_train]
    test_artifacts = artifacts[n_train:]

    train_ids = [a["id"] for a in train_artifacts]
    test_ids = [a["id"] for a in test_artifacts]

    # Save outputs
    with (out_dir / "train_ids.json").open("w", encoding="utf-8") as f:
        json.dump(train_ids, f, ensure_ascii=False, indent=2)

    with (out_dir / "test_ids.json").open("w", encoding="utf-8") as f:
        json.dump(test_ids, f, ensure_ascii=False, indent=2)

    with (out_dir / "artifacts_train.json").open("w", encoding="utf-8") as f:
        json.dump(train_artifacts, f, ensure_ascii=False, indent=2)

    with (out_dir / "artifacts_test.json").open("w", encoding="utf-8") as f:
        json.dump(test_artifacts, f, ensure_ascii=False, indent=2)

    print(f"Total artifacts: {n_total}")
    print(f"Train artifacts: {len(train_artifacts)}")
    print(f"Test artifacts: {len(test_artifacts)}")
    print(f"Saved split files to: {out_dir}")


if __name__ == "__main__":
    main()