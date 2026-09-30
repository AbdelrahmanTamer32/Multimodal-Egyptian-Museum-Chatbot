from __future__ import annotations
from pathlib import Path
import json


def make_queries(artifact: dict) -> list[str]:
    title = (artifact.get("title_en") or "").strip()
    if not title:
        return []

    return [
        f"tell me about {title}",
        f"what is {title}",
        f"give me information about {title}",
    ]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    test_path = root / "data" / "splits" / "artifacts_test.json"
    out_dir = root / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    with test_path.open("r", encoding="utf-8") as f:
        test_artifacts = json.load(f)

    eval_samples = []

    for artifact in test_artifacts:
        artifact_id = artifact.get("id")
        reference_answer = (artifact.get("text_en") or "").strip()

        if not artifact_id or not reference_answer:
            continue

        for query in make_queries(artifact):
            eval_samples.append({
                "query": query,
                "expected_id": artifact_id,
                "reference_answer": reference_answer
            })

    out_path = out_dir / "text_answer_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(eval_samples, f, ensure_ascii=False, indent=2)

    print(f"Test artifacts: {len(test_artifacts)}")
    print(f"Answer eval samples created: {len(eval_samples)}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()