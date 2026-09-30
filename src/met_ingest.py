from __future__ import annotations
import csv
import json
from pathlib import Path


def is_egypt_related(row: dict) -> bool:
    """
    Filter rules for Egyptian-related objects.
    We use multiple fields because the Met dataset varies.
    """
    hay = " ".join([
        row.get("Department", ""),
        row.get("Culture", ""),
        row.get("Geography Type", ""),
        row.get("Geography", ""),
        row.get("Object Name", ""),
        row.get("Classification", ""),
        row.get("Title", ""),
        row.get("Object Begin Date", ""),
        row.get("Object End Date", ""),
    ]).lower()

    keywords = [
        "egypt", "egyptian", "ancient egypt", "ptolemaic", "phara", "pharaoh",
        "coptic", "demotic", "hieroglyph"
    ]
    return any(k in hay for k in keywords)


def build_text_en(row: dict) -> str:
    """
    Build a richer English passage from Met fields so the chatbot can answer questions.
    """
    title = row.get("Title", "").strip()
    obj_name = row.get("Object Name", "").strip()
    classification = row.get("Classification", "").strip()
    department = row.get("Department", "").strip()
    culture = row.get("Culture", "").strip()

    dynasty = row.get("Dynasty", "").strip()
    period = row.get("Period", "").strip()
    reign = row.get("Reign", "").strip()

    date = row.get("Object Date", "").strip()
    begin_date = row.get("Object Begin Date", "").strip()
    end_date = row.get("Object End Date", "").strip()

    medium = row.get("Medium", "").strip()
    dimensions = row.get("Dimensions", "").strip()

    geography = row.get("Geography", "").strip()
    excavation = row.get("Excavation", "").strip()
    credit = row.get("Credit Line", "").strip()
    repository = row.get("Repository", "").strip()

    tags = row.get("Tags", "").strip()
    url = row.get("Link Resource", "").strip()

    parts = []
    if title:
        parts.append(f"Title: {title}.")
    if obj_name:
        parts.append(f"Object type: {obj_name}.")
    if classification:
        parts.append(f"Classification: {classification}.")
    if department:
        parts.append(f"Department: {department}.")
    if culture:
        parts.append(f"Culture: {culture}.")

    if period:
        parts.append(f"Period: {period}.")
    if dynasty:
        parts.append(f"Dynasty: {dynasty}.")
    if reign:
        parts.append(f"Reign: {reign}.")

    if date:
        parts.append(f"Date: {date}.")
    elif begin_date or end_date:
        parts.append(f"Date range: {begin_date} to {end_date}.")

    if medium:
        parts.append(f"Material/Medium: {medium}.")
    if dimensions:
        parts.append(f"Dimensions: {dimensions}.")

    if geography:
        parts.append(f"Geography: {geography}.")
    if excavation:
        parts.append(f"Excavation: {excavation}.")
    if credit:
        parts.append(f"Credit line: {credit}.")
    if repository:
        parts.append(f"Repository: {repository}.")

    if tags:
        parts.append(f"Tags: {tags}.")
    if url:
        parts.append(f"More info: {url}")

    return " ".join(parts).strip()


def main():
    project_root = Path(__file__).resolve().parents[1]

    met_csv = project_root / "data" / "datasets" / "met" / "MetObjects.csv"
    out_kb = project_root / "data" / "kb" / "artifacts_kb.json"

    # How many Egyptian items to export (start small for speed, then increase)
    MAX_ITEMS = 1500

    artifacts = []
    with met_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not is_egypt_related(row):
                continue

            object_id = row.get("Object ID", "").strip()
            title = row.get("Title", "").strip()
            if not object_id or not title:
                continue

            text_en = build_text_en(row)

            artifacts.append({
                "id": f"met_{object_id}",
                "title_en": title,
                "text_en": text_en,
                # aliases can be expanded later; keep empty for now
                "aliases": {"ar": [], "de": [], "fr": []}
            })

            if len(artifacts) >= MAX_ITEMS:
                break

    out_kb.parent.mkdir(parents=True, exist_ok=True)
    out_kb.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Wrote KB:", out_kb)
    print("✅ Items:", len(artifacts))
    print("Example:", artifacts[0]["title_en"] if artifacts else "No items found")


if __name__ == "__main__":
    main()