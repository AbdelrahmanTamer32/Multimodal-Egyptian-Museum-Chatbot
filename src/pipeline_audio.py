from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from faster_whisper import WhisperModel

from .pipeline_text import TextChatbot


def _lang_from_path(p: str) -> str | None:
    low = p.replace("\\", "/").lower()

    # Folder aliases (your folder names)
    folder_map = {
        "/ar/": "ar",
        "/de/": "de",
        "/fr/": "fr",
        "/eng/": "en",
        "/en/": "en",

        "/ru/": "ru",
        "/chi/": "zh",
        "/jap/": "ja",
        "/port/": "pt",
        "/turk/": "tr",

        # keep these too in case you add later
        "/zh/": "zh",
        "/ja/": "ja",
        "/ko/": "ko",
        "/pt/": "pt",
        "/tr/": "tr",
        "/es/": "es",
        "/it/": "it",
    }

    for k, v in folder_map.items():
        if k in low:
            return v

    # Filename-based detection (works for refuse folder)
    fname = low.split("/")[-1]
    # support both zh/chi, ja/jap, ko/kor, pt/port, tr/turk
    if fname.startswith(("ar",)) or "ar_" in fname or "_ar" in fname:
        return "ar"
    if fname.startswith(("de",)) or "de_" in fname or "_de" in fname:
        return "de"
    if fname.startswith(("fr",)) or "fr_" in fname or "_fr" in fname:
        return "fr"
    if fname.startswith(("ru",)) or "ru_" in fname or "_ru" in fname:
        return "ru"
    if fname.startswith(("zh", "chi")) or "zh_" in fname or "chi_" in fname or "_zh" in fname or "_chi" in fname:
        return "zh"
    if fname.startswith(("ja", "jap")) or "ja_" in fname or "jap_" in fname or "_ja" in fname or "_jap" in fname:
        return "ja"
    if fname.startswith(("pt", "port")) or "pt_" in fname or "port_" in fname or "_pt" in fname or "_port" in fname:
        return "pt"
    if fname.startswith(("tr", "turk")) or "tr_" in fname or "turk_" in fname or "_tr" in fname or "_turk" in fname:
        return "tr"
    if fname.startswith(("en", "eng")) or "en_" in fname or "eng_" in fname or "_en" in fname or "_eng" in fname:
        return "en"

    return None

class AudioChatbot:
    def __init__(self, project_root: str | Path, whisper_size: str = "small"):
        self.project_root = Path(project_root)

        # Whisper on GPU (requires CUDA working)
        self.whisper = WhisperModel(whisper_size, device="cuda", compute_type="float16")

        self.text_bot = TextChatbot(project_root=self.project_root)

    def transcribe(self, audio_path: str | Path) -> Dict[str, str]:
        audio_path = Path(audio_path)
        segments, info = self.whisper.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return {"whisper_lang": info.language, "text": text}

    def ask_audio(self, audio_path: str | Path) -> Dict[str, Any]:
        asr = self.transcribe(audio_path)

        forced_lang = _lang_from_path(str(audio_path))
        if forced_lang:
            # Force a language hint so output language matches folder (de/ar/fr/eng)
            question = f"[LANG={forced_lang}] {asr['text']}"
        else:
            question = asr["text"]

        out = self.text_bot.ask(question)
        out["transcript"] = asr["text"]
        out["whisper_lang"] = asr["whisper_lang"]
        out["forced_lang"] = forced_lang
        return out