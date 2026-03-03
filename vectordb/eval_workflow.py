"""
Lightweight evaluator for vectordb category assignment.

Modes:
- exemplar-only: use preset exemplars, no live DB seed
- live-only: seed live DB with train split, disable exemplars
- hybrid: seed live DB and use exemplars
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

# Allow running as script: python vectordb/eval_workflow.py (project root on path)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_labeled_questions(json_path: Path) -> list[dict]:
    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    rows: list[dict] = []
    for ingestion in payload.get("ingestions", []):
        for q in ingestion.get("questions", []):
            text = (q.get("text") or "").strip()
            category = (q.get("category") or "").strip()
            if text and category:
                rows.append({"text": text, "category": category})
    return rows


def _make_seed_payload(rows: list[dict]) -> dict:
    return {
        "schema_version": "1.0",
        "ingestions": [
            {
                "ingestion_id": "eval_seed",
                "created_at": "2026-03-02T00:00:00Z",
                "source_pdf": "eval",
                "exam_id": "eval",
                "questions": [{"text": r["text"], "category": r["category"]} for r in rows],
            }
        ],
    }


def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-path",
        default=str(Path(__file__).with_name("questions_categories.json")),
        help="Labeled questions JSON used for evaluation.",
    )
    parser.add_argument(
        "--mode",
        choices=["exemplar-only", "live-only", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=3, help="Nearest neighbors for assign_category.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Vote threshold for assign_category.")
    parser.add_argument(
        "--embedding-max-length",
        type=int,
        default=1024,
        help="Max token length used during embedding (lower is more memory-safe).",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path).resolve()
    rows = _load_labeled_questions(json_path)
    if len(rows) < 2:
        raise RuntimeError(f"Need at least 2 labeled rows; found {len(rows)} in {json_path}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    split_idx = max(1, min(len(rows) - 1, int(len(rows) * args.train_ratio)))
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]

    print(f"Loaded {len(rows)} labeled questions from {json_path}")
    print(f"Mode={args.mode} train={len(train_rows)} test={len(test_rows)} n={args.n} threshold={args.threshold}")

    with tempfile.TemporaryDirectory(prefix="vectordb_eval_") as tmpdir:
        db_path = Path(tmpdir) / "eval_questionbank.db"
        chroma_dir = Path(tmpdir) / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)

        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

        # Import only after DATABASE_URL is set.
        from vectordb.vectordb_workflow import DBWorkflow

        use_exemplars = args.mode in ("exemplar-only", "hybrid")
        workflow = DBWorkflow(
            has_categories=True,
            use_exemplars=use_exemplars,
            chroma_persist_dir=str(chroma_dir),
            embedding_max_length=args.embedding_max_length,
        )

        if args.mode in ("live-only", "hybrid"):
            workflow.populate(_make_seed_payload(train_rows))

        y_true: list[str] = []
        y_pred: list[str] = []
        for r in test_rows:
            pred = workflow.assign_category(
                r["text"],
                n=args.n,
                threshold=args.threshold,
            )
            y_true.append(r["category"])
            y_pred.append(pred)

        total = len(y_true)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        review = sum(1 for p in y_pred if p == "Review")
        print(f"Accuracy: {_safe_div(correct, total):.3f} ({correct}/{total})")
        print(f"Review rate: {_safe_div(review, total):.3f} ({review}/{total})")

        labels = sorted(set(y_true) | set(y_pred))
        tp = Counter()
        pred_count = Counter(y_pred)
        true_count = Counter(y_true)
        for t, p in zip(y_true, y_pred):
            if t == p:
                tp[t] += 1

        print("\nPer-category metrics")
        for label in labels:
            precision = _safe_div(tp[label], pred_count[label])
            recall = _safe_div(tp[label], true_count[label])
            f1 = _safe_div(2 * precision * recall, precision + recall)
            print(
                f"- {label}: precision={precision:.3f} recall={recall:.3f} "
                f"f1={f1:.3f} support={true_count[label]}"
            )

        confusion: dict[tuple[str, str], int] = defaultdict(int)
        for t, p in zip(y_true, y_pred):
            if t != p:
                confusion[(t, p)] += 1
        if confusion:
            print("\nTop confusions")
            for (t, p), cnt in sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"- true={t} predicted={p}: {cnt}")
        else:
            print("\nNo confusions on this split.")


if __name__ == "__main__":
    main()
