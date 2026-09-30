from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
import torch
from PIL import Image
import open_clip


class ImageRetriever:
    def __init__(
        self,
        project_root: str | Path,
        index_path: str | Path = "data/indexes/faiss_images.index",
        meta_path: str | Path = "data/indexes/image_meta.json",
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ):
        self.project_root = Path(project_root)
        self.index_path = self.project_root / index_path
        self.meta_path = self.project_root / meta_path

        self.index = faiss.read_index(str(self.index_path))
        self.meta: List[Dict[str, Any]] = json.loads(self.meta_path.read_text(encoding="utf-8"))

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device)
        self.model.eval()

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype("float32")

    def search(self, image_path: str | Path, top_k: int = 3):
        q = self.embed_image(image_path)
        scores, idxs = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            m = self.meta[int(idx)]
            results.append({
                "score": float(score),
                "id": m["id"],
                "title": m["title"],
                "img_path": m["img_path"],
            })
        return results