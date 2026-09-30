from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json

from .image_retrieval import ImageRetriever
from .pipeline_text import TextChatbot


class ImageChatbot:
    def __init__(
        self,
        project_root: str | Path,
        kb_with_images: str | Path = "data/kb/artifacts_kb_with_images.json",
    ):
        self.project_root = Path(project_root)
        self.img_retriever = ImageRetriever(project_root=self.project_root)
        self.text_bot = TextChatbot(project_root=self.project_root)

        kb_path = self.project_root / kb_with_images
        artifacts: List[dict] = json.loads(kb_path.read_text(encoding="utf-8"))

        # Build id -> image urls map
        self.id_to_images: Dict[str, Dict[str, str]] = {}
        for a in artifacts:
            art_id = a.get("id")
            if not art_id:
                continue
            self.id_to_images[art_id] = {
                "image_small": (a.get("image_small") or "").strip(),
                "image_primary": (a.get("image_primary") or "").strip(),
            }

    def ask_image(self, image_path: str | Path, question: str = "", lang: str = "en") -> Dict[str, Any]:
        raw_results = self.img_retriever.search(image_path, top_k=3)

        # Replace local img_path with public URLs (Android-friendly)
        results = []
        for r in raw_results:
            rid = r["id"]
            urls = self.id_to_images.get(rid, {})
            results.append({
                "score": r["score"],
                "id": rid,
                "title": r["title"],
                "image_small": urls.get("image_small", ""),
                "image_primary": urls.get("image_primary", ""),
            })

        top = results[0]
        top_score = top["score"]

        # Default question if user didn't type anything
        if not question or not question.strip():
            question = "What is this artifact?"

        # Force output language from app selection
        question_for_bot = f"[LANG={lang}] {question}"

        # confidence gate for clean images
        if top_score < 0.75:
            out = self.text_bot.ask(question_for_bot)
            out["image_match"] = results
            out["image_top_score"] = top_score
            return out

        title = top["title"]
        art_id = top["id"]

        urls = self.id_to_images.get(art_id, {})
        image_small = urls.get("image_small", "")
        image_primary = urls.get("image_primary", "")

        # Keep the user's question text, but anchor to the identified artifact title
        combined_q = f"{question} (Artifact: {title})"
        out = self.text_bot.ask(f"[LANG={lang}] {combined_q}")

        out["image_match"] = results
        out["image_top_score"] = top_score
        out["matched_artifact"] = {
            "id": art_id,
            "title": title,
            "image_small": image_small,
            "image_primary": image_primary,
        }
        return out