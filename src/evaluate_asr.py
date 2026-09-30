from __future__ import annotations
from pathlib import Path
import json
import argparse
from typing import List, Dict, Any

from faster_whisper import WhisperModel


def levenshtein_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[n][m]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="e.g. tiny, base, small, medium, large-v3")
    parser.add_argument("--eval_file", default="data/eval/audio_eval.json")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute_type", default="float16")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--force_language", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    eval_path = project_root / args.eval_file
    output_path = project_root / args.output_file

    eval_items = load_eval_set(eval_path)

    model = WhisperModel(args.model_name, device=args.device, compute_type=args.compute_type)

    results = []
    wers = []
    cers = []

    for item in eval_items:
        audio_path = project_root / item["audio_file"]
        lang = item["lang"]
        reference = item["reference"]

        transcribe_kwargs = {
            "beam_size": args.beam_size
        }

        if args.force_language:
            transcribe_kwargs["language"] = lang

        segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)
        transcript = " ".join(seg.text.strip() for seg in segments).strip()

        wer = word_error_rate(reference, transcript)
        cer = char_error_rate(reference, transcript)

        wers.append(wer)
        cers.append(cer)

        results.append({
            "audio_file": item["audio_file"],
            "lang": lang,
            "reference": reference,
            "prediction": transcript,
            "detected_language": getattr(info, "language", None),
            "wer": wer,
            "cer": cer
        })

        print(f"Done: {item['audio_file']} | WER={wer:.4f} | CER={cer:.4f}")

    summary = {
        "model_name": args.model_name,
        "beam_size": args.beam_size,
        "force_language": args.force_language,
        "avg_wer": sum(wers) / len(wers) if wers else 0.0,
        "avg_cer": sum(cers) / len(cers) if cers else 0.0,
        "num_files": len(results),
        "results": results
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== FINAL SUMMARY ===")
    print(f"Model: {args.model_name}")
    print(f"Beam size: {args.beam_size}")
    print(f"Force language: {args.force_language}")
    print(f"Average WER: {summary['avg_wer']:.4f}")
    print(f"Average CER: {summary['avg_cer']:.4f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()