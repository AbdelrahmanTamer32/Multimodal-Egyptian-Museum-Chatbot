from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import faiss
import torch
from PIL import Image
import open_clip


def main():
    project_root = Path(__file__).resolve().parents[1]

    kb_path = project_root / "data" / "kb" / "artifacts_kb_images_only.json"
    img_dir = project_root / "data" / "images" / "thumbs"

    out_index = project_root / "data" / "indexes" / "faiss_images.index"
    out_meta = project_root / "data" / "indexes" / "image_meta.json"

    artifacts = json.loads(kb_path.read_text(encoding="utf-8"))

    # Map artifact id -> title (and image file path)
    items = []
    for a in artifacts:
        art_id = a.get("id", "")
        title = a.get("title_en", "")
        img_path = img_dir / f"{art_id}.jpg"
        if art_id and title and img_path.exists():
            items.append({"id": art_id, "title": title, "img_path": str(img_path)})

    print("Items with local thumbnails:", len(items))
    if not items:
        raise RuntimeError("No thumbnails found. Check data/images/thumbs/.")
    
    if not torch.cuda.is_available():
       raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch or check your GPU drivers.")
    device = "cuda"
    print("Using device:", device)

    # OpenCLIP model (good default)
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()

    embs = []
    meta = []

    with torch.no_grad():
        for i, it in enumerate(items, start=1):
            img = Image.open(it["img_path"]).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)

            feat = model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)  # normalize
            feat_np = feat.cpu().numpy().astype("float32")[0]

            embs.append(feat_np)
            meta.append({"id": it["id"], "title": it["title"], "img_path": it["img_path"]})

            if i % 200 == 0:
                print(f"Embedded {i}/{len(items)}")

    embs = np.vstack(embs).astype("float32")
    dim = embs.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
    index.add(embs)

    out_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_index))
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved image index:", out_index)
    print(" Saved image meta:", out_meta)
    print(" Vectors:", embs.shape)


if __name__ == "__main__":
    main()