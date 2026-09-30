from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Tuple, Optional

import requests

API_URL = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BachelorMuseumGuide/1.0",
    "Accept": "application/json",
})


def fetch_object(object_id: str, max_retries: int = 5) -> Tuple[Optional[dict], Optional[str]]:
    """
    Fetch a Met object JSON via requests.Session with retries + backoff.
    Returns (data, error_string).
    """
    url = API_URL.format(object_id)

    for attempt in range(1, max_retries + 1):
        try:
            resp = SESSION.get(url, timeout=20)

            # Success
            if resp.status_code == 200:
                return resp.json(), None

            # Permanent
            if resp.status_code == 404:
                return None, "HTTPError 404"

            # Temporary blocks / throttling
            if resp.status_code in (403, 429, 500, 502, 503, 504):
                wait = min(30, 2 ** attempt)  # exponential backoff up to 30s
                time.sleep(wait)
                continue

            # Other unexpected codes
            return None, f"HTTPError {resp.status_code}"

        except requests.RequestException as e:
            wait = min(30, 2 ** attempt)
            time.sleep(wait)
            last_err = repr(e)

    return None, "Retries exhausted"


def main():
    project_root = Path(__file__).resolve().parents[1]

    kb_path = project_root / "data" / "kb" / "artifacts_kb.json"
    out_path = project_root / "data" / "kb" / "artifacts_kb_with_images.json"

    artifacts = json.loads(kb_path.read_text(encoding="utf-8"))

    updated = []
    ok = 0
    no_img = 0
    failed = 0

    for i, a in enumerate(artifacts, start=1):
        art_id = a.get("id", "")

        # your ids are like: met_123456
        if not art_id.startswith("met_"):
            updated.append(a)
            continue

        object_id = art_id.replace("met_", "").strip()

        data, err = fetch_object(object_id, max_retries=5)
        if data is None:
            failed += 1
            if failed <= 10:
                print(f"[FAIL {failed}] object_id={object_id} error={err}")
            updated.append(a)
            continue

        primary = data.get("primaryImage", "") or ""
        small = data.get("primaryImageSmall", "") or ""

        a2 = dict(a)
        a2["image_primary"] = primary
        a2["image_small"] = small

        if small or primary:
            ok += 1
        else:
            no_img += 1

        updated.append(a2)

        # gentle rate limit (still good practice)
        if i % 25 == 0:
            time.sleep(0.5)

        if i % 100 == 0:
            print(f"Processed {i}/{len(artifacts)} | with_img={ok} | no_img={no_img} | failed={failed}")

    out_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n✅ Saved:", out_path)
    print("With images:", ok)
    print("No images:", no_img)
    print("Failed API:", failed)


if __name__ == "__main__":
    main()