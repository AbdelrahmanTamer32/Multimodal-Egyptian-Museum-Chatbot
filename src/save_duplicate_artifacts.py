from __future__ import annotations
from pathlib import Path
import json
from collections import defaultdict


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    kb_path = root / "data" / "kb" / "artifacts_kb.json"

    out_dir = root / "data" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "duplicate_titles_summary.json"
    details_path = out_dir / "duplicate_titles_details.json"

    with kb_path.open("r", encoding="utf-8") as f:
        artifacts = json.load(f)

    title_to_artifacts = defaultdict(list)

    for artifact in artifacts:
        title = (artifact.get("title_en") or artifact.get("title") or "").strip()
        if title:
            title_to_artifacts[title].append(artifact)

    duplicate_summary = []
    duplicate_details = []

    for title, items in title_to_artifacts.items():
        if len(items) > 1:
            duplicate_summary.append({
                "title": title,
                "count": len(items)
            })

            duplicate_details.append({
                "title": title,
                "count": len(items),
                "artifacts": items
            })

    duplicate_summary.sort(key=lambda x: (-x["count"], x["title"]))
    duplicate_details.sort(key=lambda x: (-x["count"], x["title"]))

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(duplicate_summary, f, ensure_ascii=False, indent=2)

    with details_path.open("w", encoding="utf-8") as f:
        json.dump(duplicate_details, f, ensure_ascii=False, indent=2)

    print(f"Total artifacts: {len(artifacts)}")
    print(f"Duplicate titles found: {len(duplicate_summary)}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved details to: {details_path}")


if __name__ == "__main__":
    main()