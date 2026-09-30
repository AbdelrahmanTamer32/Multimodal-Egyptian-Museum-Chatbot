from __future__ import annotations
import json
import urllib.request
from typing import Optional, List

DEFAULT_MODEL = "qwen2.5:3b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"

def generate(prompt: str, max_tokens: int = 220, model: str = DEFAULT_MODEL, stop: Optional[List[str]] = None) -> str:
    """
    Send a prompt to the local Ollama server and return the response text.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_predict": max_tokens
        }
    }
    if stop:
        payload["options"]["stop"] = stop

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data.get("response", "").strip()