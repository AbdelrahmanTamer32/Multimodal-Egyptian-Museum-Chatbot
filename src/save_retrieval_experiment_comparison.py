from __future__ import annotations
from pathlib import Path
import json


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    eval_dir = root / "data" / "eval"

    comparison = {
        "baseline": {
            "technique": "single-passage dense retrieval",
            "embedding_model": "intfloat/multilingual-e5-large",
            "top1_accuracy": 0.4889,
            "top5_accuracy": 0.6700,
            "mrr": 0.5580
        },
        "structured_passages": {
            "technique": "structured passage construction",
            "embedding_model": "intfloat/multilingual-e5-large",
            "top1_accuracy": 0.4844,
            "top5_accuracy": 0.6656,
            "mrr": 0.5515
        },
        "rerank": {
            "technique": "retrieve top-10 then lexical rerank",
            "embedding_model": "intfloat/multilingual-e5-large",
            "top1_accuracy": 0.4944,
            "top5_accuracy": 0.6722,
            "mrr": 0.5621
        },
        "e5_base_model": {
            "technique": "single-passage dense retrieval",
            "embedding_model": "intfloat/multilingual-e5-base",
            "top1_accuracy": 0.4744,
            "top5_accuracy": 0.6678,
            "mrr": 0.5472
        }
    }

    out_path = eval_dir / "retrieval_experiment_comparison.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"Saved comparison to: {out_path}")


if __name__ == "__main__":
    main()