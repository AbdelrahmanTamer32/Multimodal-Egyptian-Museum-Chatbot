from __future__ import annotations
import re


def clean_output(text: str, max_sentences: int = 5) -> str:
    """
    Removes unwanted meta text and keeps the answer concise.
    """
    if not text:
        return text

    bad_patterns = [
        r"(?i)note:.*",
        r"(?i)remark:.*",
        r"(?i)see also:.*",
        r"(?i)anmerkung.*",
        r"(?i)\*.*\*",          # removes *smiling* style stage directions
    ]
    for pat in bad_patterns:
        text = re.sub(pat, "", text).strip()

    # Remove bracketed content
    text = re.sub(r"\[.*?\]", "", text).strip()
    text = re.sub(r"\(.*?\)", "", text).strip()

    # Keep only first N sentences
    sentences = re.split(r"(?<=[.!؟?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(sentences[:max_sentences]).strip()