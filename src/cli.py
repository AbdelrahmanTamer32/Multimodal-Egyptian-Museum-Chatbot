from __future__ import annotations
import argparse
from pathlib import Path

from .pipeline_text import TextChatbot
from .pipeline_audio import AudioChatbot


def main():
    parser = argparse.ArgumentParser(description="Bachelor Project Prototype CLI")
    parser.add_argument("--text", type=str, default=None, help="Ask a question (text)")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (mp3/wav)")
    parser.add_argument("--root", type=str, default=".", help="Project root folder")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    if args.text:
        bot = TextChatbot(project_root=root)
        out = bot.ask(args.text)
        print("\nLANG:", out["lang"])
        print("SCORE:", round(out["top_score"], 3))
        print("Q_NORMALIZED:", out.get("normalized_question"))
        print("ANSWER:", out["answer"])
        if "answer_en" in out:
            print("ANSWER_EN:", out["answer_en"])
        print("SOURCES:", out["sources"])
        return

    if args.audio:
        bot = AudioChatbot(project_root=root)
        out = bot.ask_audio(args.audio)
        print("\nWHISPER_LANG:", out["whisper_lang"])
        print("TRANSCRIPT:", out["transcript"])
        print("LANG:", out["lang"])
        print("SCORE:", round(out["top_score"], 3))
        print("ANSWER:", out["answer"])
        if "answer_en" in out:
            print("ANSWER_EN:", out["answer_en"])
        print("SOURCES:", out["sources"])
        return

    print("Please provide --text or --audio")


if __name__ == "__main__":
    main()