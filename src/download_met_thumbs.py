from __future__ import annotations
import json
import time
from pathlib import Path

import requests


def safe_filename(s: str) -> str:
    # make a Windows-safe filename
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in s)


def main():
    project_root = Path(__file__).resolve().parents[1]
    kb_path = project_root / "data" / "kb" / "artifacts_kb_images_only.json"

    out_dir = project_root / "data" / "images" / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = json.loads(kb_path.read_text(encoding="utf-8"))

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BachelorMuseumGuide/1.0",
        "Accept": "*/*",
    })

    ok = 0
    skipped = 0
    failed = 0

    for i, a in enumerate(artifacts, start=1):
        art_id = a.get("id", f"item_{i}")
        url = (a.get("image_small") or a.get("image_primary") or "").strip()
        if not url:
            failed += 1
            continue

        fname = safe_filename(art_id) + ".jpg"
        fpath = out_dir / fname

        if fpath.exists() and fpath.stat().st_size > 0:
            skipped += 1
            continue

        try:
            r = session.get(url, timeout=30)
            if r.status_code != 200 or not r.content:
                failed += 1
                continue
            fpath.write_bytes(r.content)
            ok += 1
        except Exception:
            failed += 1

        # light throttle
        if i % 25 == 0:
            time.sleep(0.4)

        if i % 200 == 0:
            print(f"Downloaded {ok} | skipped {skipped} | failed {failed} | processed {i}/{len(artifacts)}")

    print("\n Thumbnails folder:", out_dir)
    print(" Downloaded:", ok)
    print("  Skipped (already existed):", skipped)
    print(" Failed:", failed)


if __name__ == "__main__":
    main()