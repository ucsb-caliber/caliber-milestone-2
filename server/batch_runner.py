import json
import time
from pathlib import Path
from generator import run_pipeline

# config
DB_PATH = Path("layout_debug/questions.json")
OUTPUT_PATH = Path("layout_debug/variants.json")

def main():
    if not DB_PATH.exists():
        print("error: questions.json not found")
        return

    with open(DB_PATH, "r") as f:
        data = json.load(f)
    
    # get questions from the latest ingestion
    questions = data["ingestions"][-1]["questions"]
    total = len(questions)
    
    print(f"Starting batch generation: {total} questions total")
    
    results = []
    stats = {"success": 0, "fail": 0}

    for i in range(total):
        # skip empty questions
        if not questions[i].get("text"):
            continue

        try:
            # run the pipeline for one question
            variant = run_pipeline(i)
            
            if variant:
                results.append(variant)
                stats["success"] += 1
            else:
                stats["fail"] += 1
                
            # append to file every time so we don't lose data if it crashes
            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            
            # small sleep to prevent ollama from choking
            time.sleep(0.5)

        except Exception as e:
            print(f"Critical error at index {i}: {e}")
            continue

    print("\n--- Batch complete ---")
    print(f"Successfully generated: {stats['success']}")
    print(f"Failed verification:   {stats['fail']}")
    print(f"Results saved to:      {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
