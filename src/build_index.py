from __future__ import annotations
from pathlib import Path

from src.kb import load_kb, build_passages
from src.retrieval import build_faiss_index, save_index, save_meta


def main():
    project_root = Path(__file__).resolve().parents[1]

    kb_path = project_root / "data" / "kb" / "artifacts_kb.json"
    index_path = project_root / "data" / "indexes" / "faiss.index"
    meta_path = project_root / "data" / "indexes" / "meta.json"

    artifacts = load_kb(kb_path)
    passages, meta = build_passages(artifacts)

    index = build_faiss_index(passages)

    save_index(index, index_path)
    save_meta(meta, meta_path)

    print("✅ Index built successfully!")
    print("KB:", kb_path)
    print("Index:", index_path)
    print("Meta:", meta_path)
    print("Passages:", len(passages))


if __name__ == "__main__":
    main()