"""
Full-ingestion batch: generate variants for all questions, append to layout_debug/variants.json.

From the server directory:

  python -m variant_gen.generation_batch_runner

Or use VS Code / Cursor "Run Python File" on this module — the path bootstrap below makes that work.
"""

import json
import os
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    _server_dir = Path(__file__).resolve().parent.parent
    if str(_server_dir) not in sys.path:
        sys.path.insert(0, str(_server_dir))

from variant_gen import DB_PATH, generate_variant

OUTPUT_PATH = DB_PATH.parent / "variants.json"

DEBUG_BATCH = False


def load_json_safe(path):
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []


def save_atomic(data, path):
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(str(temp_path), str(path))


def main():
    if not DB_PATH.exists():
        print(f"Error: Input file not found at {DB_PATH}")
        return

    print(f"Loading questions from: {DB_PATH}")

    with open(DB_PATH, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    questions = source_data["ingestions"][-1]["questions"]
    total_questions = len(questions)

    existing_results = load_json_safe(OUTPUT_PATH)
    processed_ids = {item["original_id"] for item in existing_results}
    results = list(existing_results)

    print(f"Total Questions: {total_questions}")
    print(f"Already Completed: {len(processed_ids)}")
    print(f"Output: {OUTPUT_PATH}")
    print("Starting Batch Generation...\n")
    print("-" * 50)

    stats = {"success": 0, "fail": 0, "skipped": len(processed_ids)}

    for i, q in enumerate(questions):
        q_id = q.get("question_id", f"index_{i}")
        q_text = q.get("text", "")[:60].replace("\n", " ") + "..."

        if not q.get("text"):
            if DEBUG_BATCH:
                print(f"   [Skip] Empty text at index {i}")
            continue

        if q_id in processed_ids:
            if DEBUG_BATCH:
                print(f"   [Skip] ID {q_id} already processed.")
            continue

        print(f"Processing [{i+1}/{total_questions}]: ID {q_id}")
        print(f"   Context: \"{q_text}\"")

        try:
            start_time = time.time()

            variant = generate_variant(i, db_path=DB_PATH)

            duration = time.time() - start_time

            if variant:
                print(f"   Success ({duration:.1f}s) [{variant.get('scenario_domain', '?')}]")
                results.append(variant)
                processed_ids.add(q_id)
                stats["success"] += 1
            else:
                print(f"   Failed ({duration:.1f}s)")
                stats["fail"] += 1

            save_atomic(results, OUTPUT_PATH)

            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nBatch stopped by user. Progress saved.")
            save_atomic(results, OUTPUT_PATH)
            break
        except Exception as e:
            print(f"   Critical Error: {e}")
            stats["fail"] += 1
            continue

        print("-" * 50)

    print("\n" + "=" * 40)
    print("          BATCH COMPLETE")
    print("=" * 40)
    print(f"  New Successful:   {stats['success']}")
    print(f"  Previously Done:  {stats['skipped']}")
    print(f"  Failed/Skipped:   {stats['fail']}")
    print(f"  Output Saved To:  {OUTPUT_PATH}")
    print("=" * 40)


if __name__ == "__main__":
    main()
