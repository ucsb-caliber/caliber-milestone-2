"""
Small fixed-index generation run for spot-checking (concept, API MCQ, OOP, recursion).

Does not touch layout_debug/variants.json — writes layout_debug/variants_eval.json by default.

From the server directory:

  python -m variant_gen.eval_batch
  python -m variant_gen.eval_batch --indices 1,2,3,10,14

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

from variant_gen import DB_PATH, generate_variant

DEFAULT_INDICES = [1, 2, 3, 5, 10, 11, 14]
DEFAULT_OUT = DB_PATH.parent / "variants_eval.json"


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
        help="questions.json (default: variant_gen DB_PATH)",
    )
    args = ap.parse_args()
    db = args.db or DB_PATH
    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]

    with open(db, "r", encoding="utf-8") as f:
        questions = json.load(f)["ingestions"][-1]["questions"]

    rows = []
    for idx in indices:
        if idx < 0 or idx >= len(questions):
            rows.append({"index": idx, "error": "index out of range", "variant": None})
            print(f"\n=== index {idx} OUT OF RANGE (0–{len(questions) - 1}) ===")
            continue
        q = questions[idx]
        qid = q.get("question_id")
        preview = (q.get("text") or "")[:72].replace("\n", " ")
        print(f"\n=== index {idx} | {qid} ===")
        print(f"    {preview}...")
        t0 = time.time()
        variant = generate_variant(idx, db_path=db)
        elapsed = time.time() - t0
        if variant:
            print(f"    ok ({elapsed:.1f}s) [{variant.get('scenario_domain', '?')}]")
            rows.append({"index": idx, "original_id": qid, "variant": variant})
        else:
            print(f"    fail ({elapsed:.1f}s)")
            rows.append({"index": idx, "original_id": qid, "variant": None})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
