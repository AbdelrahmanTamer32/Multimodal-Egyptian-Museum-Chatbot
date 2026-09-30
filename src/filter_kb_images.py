from __future__ import annotations
import json
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    in_path = project_root / "data" / "kb" / "artifacts_kb_with_images.json"
    out_path = project_root / "data" / "kb" / "artifacts_kb_images_only.json"

    artifacts = json.loads(in_path.read_text(encoding="utf-8"))

    keep = []
    for a in artifacts:
        small = (a.get("image_small") or "").strip()
        primary = (a.get("image_primary") or "").strip()
        if small or primary:
            keep.append(a)

    out_path.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(" Saved:", out_path)
    print(" Items with images:", len(keep))
    print("Example:", keep[0]["title_en"] if keep else "NONE")

if __name__ == "__main__":
    main()