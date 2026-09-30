from __future__ import annotations
from pathlib import Path
import json
from collections import Counter, defaultdict


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    kb_path = root / "data" / "kb" / "artifacts_kb.json"
    failures_path = root / "data" / "eval" / "text_retrieval_failures.json"

    with kb_path.open("r", encoding="utf-8") as f:
        artifacts = json.load(f)

    with failures_path.open("r", encoding="utf-8") as f:
        failures = json.load(f)

    title_to_ids = defaultdict(list)
    for artifact in artifacts:
        title = (artifact.get("title_en") or "").strip()
        artifact_id = artifact.get("id")
        if title and artifact_id:
            title_to_ids[title].append(artifact_id)

    duplicate_titles = {title: ids for title, ids in title_to_ids.items() if len(ids) > 1}

    ambiguous_failures = 0

    for failure in failures:
        expected_id = failure["expected_id"]

        expected_title = None
        for artifact in artifacts:
            if artifact.get("id") == expected_id:
                expected_title = (artifact.get("title_en") or "").strip()
                break

        if expected_title and expected_title in duplicate_titles:
            ambiguous_failures += 1

    print(f"Total artifacts: {len(artifacts)}")
    print(f"Unique duplicated titles: {len(duplicate_titles)}")
    print(f"Total failures: {len(failures)}")
    print(f"Failures caused by duplicate-title ambiguity: {ambiguous_failures}")


if __name__ == "__main__":
    main()