import json
import time
from pathlib import Path
from generator import run_pipeline

# --- configuration ---
BASE_DIR = Path(__file__).parent.parent 
DB_PATH = BASE_DIR / "server" / "layout_debug" / "questions.json"
OUTPUT_PATH = BASE_DIR / "server" / "layout_debug" / "variants.json"

# set true to see raw dumps
DEBUG_BATCH = False 

def load_json_safe(path):
    if not path.exists(): return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

def save_atomic(data, path):
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # atomic replace
    if path.exists():
        path.unlink()
    temp_path.rename(path)

def main():
    # setup and validation
    if not DB_PATH.exists():
        print(f"Error: Input file not found at {DB_PATH}")
        return

    print(f"Loading questions from: {DB_PATH.name}")
    
    # load source questions
    with open(DB_PATH, "r", encoding="utf-8") as f:
        source_data = json.load(f)
    
    # assume we always want the latest ingestion
    questions = source_data["ingestions"][-1]["questions"]
    total_questions = len(questions)
    
    # resume Logic
    # load existing results to skip already-processed questions
    existing_results = load_json_safe(OUTPUT_PATH)
    processed_ids = {item["original_id"] for item in existing_results}
    
    # initialize results with what we already have
    results = existing_results
    
    print(f"Total Questions: {total_questions}")
    print(f"Already Completed: {len(processed_ids)}")
    print(f"Starting Batch Generation...\n")
    print("-" * 50)

    stats = {
        "success": 0, 
        "fail": 0, 
        "skipped": len(processed_ids)
    }

    # processing Loop
    for i, q in enumerate(questions):
        q_id = q.get("question_id", f"index_{i}")
        q_text = q.get("text", "")[:50].replace("\n", " ") + "..." # snippet for logging

        # skip checks
        if not q.get("text"):
            if DEBUG_BATCH: print(f"   [Skip] Empty text at index {i}")
            continue
            
        if q_id in processed_ids:
            # prin skipping for all if are in debug mode
            if DEBUG_BATCH: print(f"   [Skip] ID {q_id} already processed.")
            continue

        # user feedback
        print(f"Processing [{i+1}/{total_questions}]: ID {q_id}")
        print(f"   Context: \"{q_text}\"")

        # run pipeline
        try:
            start_time = time.time()
            
            variant = run_pipeline(i)
            
            duration = time.time() - start_time
            
            if variant:
                print(f"   Success ({duration:.1f}s)")
                results.append(variant)
                processed_ids.add(q_id)
                stats["success"] += 1
            else:
                print(f"   Failed Verification ({duration:.1f}s)")
                stats["fail"] += 1

            # atomic save
            save_atomic(results, OUTPUT_PATH)
            
            # cooldown for gpu or api
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\nBatch stopped by user.")
            break
        except Exception as e:
            print(f"   Critical Error: {e}")
            stats["fail"] += 1
            continue
        
        print("-" * 50)

    # final summary
    print("\n" + "="*30)
    print("       BATCH COMPLETE       ")
    print("="*30)
    print(f"New Successful:   {stats['success']}")
    print(f"Previously Done:  {stats['skipped']}")
    print(f"Failed/Skipped:   {stats['fail']}")
    print(f"Output Saved To:  {OUTPUT_PATH}")
    print("="*30)

if __name__ == "__main__":
    main()
