from __future__ import annotations
from pathlib import Path
import json

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from src.pipeline_text import TextChatbot


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    eval_path = root / "data" / "eval" / "text_answer_eval.json"
    out_path = root / "data" / "eval" / "text_answer_metrics.json"

    bot = TextChatbot(project_root=root)

    with eval_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    smoothie = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    total = 0
    bleu_sum = 0.0
    rouge1_sum = 0.0
    rougeL_sum = 0.0

    detailed_results = []

    for sample in samples:
        query = sample["query"]
        expected_id = sample["expected_id"]
        reference = sample["reference_answer"]

        result = bot.ask(query)
        generated = (result.get("answer") or "").strip()

        reference_tokens = reference.split()
        generated_tokens = generated.split()

        bleu = sentence_bleu(
            [reference_tokens],
            generated_tokens,
            smoothing_function=smoothie
        )

        rouge_scores = scorer.score(reference, generated)
        rouge1_f = rouge_scores["rouge1"].fmeasure
        rougeL_f = rouge_scores["rougeL"].fmeasure

        total += 1
        bleu_sum += bleu
        rouge1_sum += rouge1_f
        rougeL_sum += rougeL_f

        detailed_results.append({
            "query": query,
            "expected_id": expected_id,
            "generated_answer": generated,
            "reference_answer": reference,
            "bleu": bleu,
            "rouge1_f": rouge1_f,
            "rougeL_f": rougeL_f
        })

    metrics = {
        "total_samples": total,
        "average_bleu": bleu_sum / total if total else 0.0,
        "average_rouge1_f": rouge1_sum / total if total else 0.0,
        "average_rougeL_f": rougeL_sum / total if total else 0.0,
        "detailed_results": detailed_results
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Total samples: {total}")
    print(f"Average BLEU: {metrics['average_bleu']:.4f}")
    print(f"Average ROUGE-1 F1: {metrics['average_rouge1_f']:.4f}")
    print(f"Average ROUGE-L F1: {metrics['average_rougeL_f']:.4f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()