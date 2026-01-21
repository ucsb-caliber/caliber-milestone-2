import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


QUESTIONS_DB = Path("layout_debug/questions.json")
OUT_NPZ = Path("layout_debug/question_embeddings.npz")

MODEL_NAME = "BAAI/bge-large-en-v1.5"


BATCH_SIZE = 32
NORMALIZE = True 


def load_questions_db(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "ingestions" not in data or not isinstance(data["ingestions"], list):
        raise ValueError("Expected nested format: {'ingestions': [...]} in questions.json")
    return data


def extract_records(data: Dict[str, Any]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Returns:
      ids:   question_id list
      texts: question text list
      meta:  per-question provenance metadata (parallel list)
    """
    ids: List[str] = []
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    for ing in data["ingestions"]:
        created_at = ing.get("created_at")
        source_pdf = ing.get("source_pdf")
        exam_id = ing.get("exam_id")
        ingestion_id = ing.get("ingestion_id")

        for q in ing.get("questions", []):
            qid = (q.get("question_id") or "").strip()
            text = (q.get("text") or "").strip()
            if not qid or not text:
                continue

            ids.append(qid)
            texts.append(text)
            meta.append(
                {
                    "question_id": qid,
                    "exam_id": exam_id,
                    "source_pdf": source_pdf,
                    "created_at": created_at,
                    "ingestion_id": ingestion_id,
                    "start_page": q.get("start_page"),
                    "page_nums": q.get("page_nums", []),
                    "text_hash": q.get("text_hash"),
                    "image_crops": q.get("image_crops", []),
                }
            )

    return ids, texts, meta


def main():
    data = load_questions_db(QUESTIONS_DB)
    ids, texts, meta = extract_records(data)

    print(f"Loaded {len(ids)} questions to embed from {QUESTIONS_DB}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=NORMALIZE,
    )

    # Store compactly
    embeddings = np.asarray(embeddings, dtype=np.float32)
    ids_arr = np.asarray(ids, dtype=object)
    meta_json = np.asarray([json.dumps(m, ensure_ascii=False) for m in meta], dtype=object)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        model_name=np.asarray(MODEL_NAME, dtype=object),
        normalized=np.asarray(NORMALIZE, dtype=bool),
        ids=ids_arr,
        embeddings=embeddings,
        meta_json=meta_json, 
    )

    print(f"Wrote embeddings to: {OUT_NPZ}")
    print(f"Embeddings shape: {embeddings.shape} (n_questions, dim)")


if __name__ == "__main__":
    main()
