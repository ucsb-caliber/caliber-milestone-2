"""
Small fixed-index generation run for spot-checking (concept, API MCQ, OOP, recursion).

Does not touch layout_debug/variants.json — writes layout_debug/variants_eval.json by default.

From the server directory:

  python -m variant_gen.eval_batch
  python -m variant_gen.eval_batch --exam-id practicefinal3 --indices 0,1,2
  python -m variant_gen.eval_batch --indices 1,2,3,10,14
  python -m variant_gen.eval_batch --ingestion -1 --indices 0,1,2

``--exam-id`` selects the ingestion whose ``exam_id`` matches (last match wins).
Otherwise ``--ingestion`` is the 0-based index among ingestions (``-1`` = last).
Default indices assume a ~15+ question set; shorten ``--indices`` for smaller exams.

By default questions are parsed from ``exam_tests/*.pdf``. Optional ``--db`` loads a layout JSON file.

Or use VS Code / Cursor "Run Python File" on this module — the path bootstrap below makes that work.
"""

import argparse
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    _server_dir = Path(__file__).resolve().parent.parent
    if str(_server_dir) not in sys.path:
        sys.path.insert(0, str(_server_dir))

from variant_gen import EXAM_TESTS_DIR, generate_variant
from variant_gen.exam_tests_questions import load_questions_database
from variant_gen.ingestion_resolve import resolve_ingestion_index

# Default indices span a mid-sized exam; use --indices for shorter ingestions.
DEFAULT_INDICES = [1, 2, 3, 5, 8, 10, 12, 14]
DEFAULT_OUT = EXAM_TESTS_DIR / "variants_eval.json"


def main():
    ap = argparse.ArgumentParser(description="Run variant generation on a few indices for eval.")
    ap.add_argument(
        "--indices",
        default=",".join(str(i) for i in DEFAULT_INDICES),
        help=f"comma-separated 0-based indices (default: {DEFAULT_INDICES})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output JSON path (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=None,
        help="optional questions.json; default is exam_tests PDFs or VARIANT_GEN_QUESTIONS_JSON",
    )
    ap.add_argument(
        "--ingestion",
        type=int,
        default=0,
        help="0-based ingestion index (-1=last; default 0); ignored if --exam-id is set",
    )
    ap.add_argument(
        "--exam-id",
        default=None,
        metavar="ID",
        help="select ingestion by exam_id (exact match; last match wins); overrides --ingestion",
    )
    args = ap.parse_args()
    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]

    try:
        source_data = load_questions_database(args.db)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"Error loading questions: {e}", file=sys.stderr)
        sys.exit(1)
    ingestions = source_data["ingestions"]
    try:
        ing_idx = resolve_ingestion_index(
            ingestions,
            exam_id=args.exam_id,
            ingestion_index=args.ingestion,
        )
    except (ValueError, IndexError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    ing = ingestions[ing_idx]
    questions = ing["questions"]
    ing_id = ing.get("ingestion_id", "?")
    print(
        f"Using ingestion [{ing_idx}] exam_id={ing.get('exam_id', '?')} {ing_id} ({len(questions)} questions)"
    )

    rows = []
    for idx in indices:
        if idx < 0 or idx >= len(questions):
            rows.append(
                {
                    "ingestion_index": ing_idx,
                    "ingestion_id": ing_id,
                    "index": idx,
                    "error": "index out of range",
                    "variant": None,
                }
            )
            print(f"\n=== index {idx} OUT OF RANGE (0–{len(questions) - 1}) ===")
            continue
        q = questions[idx]
        qid = q.get("question_id")
        preview = (q.get("text") or "")[:72].replace("\n", " ")
        print(f"\n=== index {idx} | {qid} ===")
        print(f"    {preview}...")
        t0 = time.time()
        variant = generate_variant(
            idx,
            db_path=args.db,
            ingestion_index=ing_idx,
            questions_db=source_data,
        )
        elapsed = time.time() - t0
        if variant:
            print(f"    ok ({elapsed:.1f}s) [{variant.get('scenario_domain', '?')}]")
            rows.append(
                {
                    "ingestion_index": ing_idx,
                    "ingestion_id": ing_id,
                    "index": idx,
                    "original_id": qid,
                    "variant": variant,
                }
            )
        else:
            print(f"    fail ({elapsed:.1f}s)")
            rows.append(
                {
                    "ingestion_index": ing_idx,
                    "ingestion_id": ing_id,
                    "index": idx,
                    "original_id": qid,
                    "variant": None,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
