from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_kb(kb_path: str | Path) -> List[Dict[str, Any]]:
    """
    Loads artifacts_kb.json (a list of artifact objects).
    """
    kb_path = Path(kb_path)
    with kb_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("artifacts_kb.json must be a JSON list of artifacts.")
    return data


def build_passages(artifacts: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Convert KB entries into passages for retrieval.

    Returns:
      passages: list[str] (what we embed)
      meta: list[dict] (aligned metadata: id + title)
    """
    passages: List[str] = []
    meta: List[Dict[str, Any]] = []

    for a in artifacts:
        art_id = a.get("id", "")
        title = a.get("title_en", a.get("title", ""))
        text = a.get("text_en", a.get("text", ""))

        if not title or not text:
            continue

        passages.append(f"{title}: {text}")
        meta.append({"id": art_id, "title": title})

    if not passages:
        raise ValueError("No valid artifacts found in KB (missing title/text).")

    return passages, meta