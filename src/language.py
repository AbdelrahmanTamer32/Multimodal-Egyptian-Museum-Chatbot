from __future__ import annotations
from lingua import Language, LanguageDetectorBuilder
import re
_detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH,
    Language.ARABIC,
    Language.GERMAN,
    Language.FRENCH,
    Language.ITALIAN,
    Language.SPANISH,

    # NEW languages
    Language.CHINESE,
    Language.RUSSIAN,
    Language.PORTUGUESE,
    Language.TURKISH,
    Language.JAPANESE,

    
).build()

_LANG_MAP = {
    "ENGLISH": "en",
    "ARABIC": "ar",
    "GERMAN": "de",
    "FRENCH": "fr",
    "ITALIAN": "it",
    "SPANISH": "es",
    "CHINESE": "zh",
    "RUSSIAN": "ru",
    "PORTUGUESE": "pt",
    "TURKISH": "tr",
    "JAPANESE": "ja",
 
}



def detect_lang(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "en"

    low = t.lower()

    # Strong script checks
    if re.search(r"[\u0600-\u06FF]", t):
        return "ar"

    # Cyrillic (Russian) script check
    if re.search(r"[\u0400-\u04FF]", t):
        return "ru"
     # CJK script checks
    if re.search(r"[\u3040-\u30FF]", t):  # Hiragana/Katakana
        return "ja"
    if re.search(r"[\uAC00-\uD7AF]", t):  # Hangul
        return "ko"
    if re.search(r"[\u4E00-\u9FFF]", t):  # CJK Unified Ideographs (Chinese chars)
        return "zh"

    # Phrase heuristics (very reliable for your use case)
    if any(p in low for p in ["parle-moi", "qu'est-ce", "quelle est", "explique", "décris"]):
        return "fr"
    if any(p in low for p in ["was ist", "erzähl", "beschreibe", "wann wurde", "hauptstadt"]):
        return "de"
    if any(p in low for p in ["háblame", "qué es", "cuál es", "explícame", "describe"]):
        return "es"
    if any(p in low for p in ["parlami", "che cos'è", "qual è", "spiegami", "descrivi"]):
        return "it"
    # Portuguese heuristics
    if any(p in low for p in ["fale sobre", "fala sobre", "conte-me sobre", "explique", "descreva"]):
        return "pt"
    # Turkish heuristics
    if any(p in low for p in ["hakkında", "anlat", "açıkla", "nedir", "ne zaman"]):
        return "tr"

    # Short latin text fallback (avoid mis-detecting as random languages)
    if len(t) < 30 and re.fullmatch(r"[A-Za-z0-9\s\?\!\.\,\-\']+", t):
        return "en"

    lang = _detector.detect_language_of(t)
    if lang is None:
        return "en"

    code = _LANG_MAP.get(lang.name, lang.name.lower())

    # Allow these UI languages (extend as needed)
    allowed = {"en","ar","de","fr","es","it","zh","ru","pt","tr","ja"}
    return code if code in allowed else "en"