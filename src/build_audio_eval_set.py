from __future__ import annotations
from pathlib import Path
import json

from src.pipeline_audio import AudioChatbot


FOLDER_LANG_MAP = {
    "ar": "ar",
    "chi": "zh",
    "de": "de",
    "eng": "en",
    "fr": "fr",
    "jap": "ja",
    "port": "pt",
    "ru": "ru",
    "turk": "tr",
}


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    audio_root = project_root / "audio tests"
    output_path = project_root / "data" / "eval" / "audio_eval_draft.json"

    bot = AudioChatbot(project_root=project_root)
    results = []

    for folder_name, lang_code in FOLDER_LANG_MAP.items():
        folder_path = audio_root / folder_name
        if not folder_path.exists():
            print(f"Skipping missing folder: {folder_path}")
            continue

        audio_files = sorted(folder_path.glob("*.mp3"))

        for audio_file in audio_files:
            print(f"Transcribing: {audio_file}")
            try:
                out = bot.ask_audio(audio_file)
                transcript = out.get("transcript", "").strip()
            except Exception as e:
                print(f"Failed on {audio_file}: {e}")
                transcript = ""

            results.append({
                "audio_file": str(audio_file.relative_to(project_root)).replace("/", "\\"),
                "lang": lang_code,
                "reference": transcript
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved draft audio eval set to: {output_path}")


if __name__ == "__main__":
    main()