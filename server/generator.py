import json
import requests
import base64
import re
from pathlib import Path

# config
DB_PATH = Path("layout_debug/questions.json")
OLLAMA_URL = "http://localhost:11434/api/chat"
# TODO: change models based on hardware available
VISION_MODEL = "llama3.2-vision"
CODING_MODEL = "qwen2.5-coder:7b"
JUDGING_MODEL = "llama3.1:8b"
DEBUG = False  # set to True to see full logs
MAX_RETRIES = 3 

# --- helper functions ---
def encode_image(path):
    if not Path(path).exists(): return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def call_ollama(prompt, model, images=None):
    if DEBUG:
        print(f"\n[DEBUG] Calling model: {model}")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images or []}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.4, "num_ctx": 8192}
    }
    try:
        # added timeout to prevent hanging on vision models
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        content = response.json()['message']['content']

        if DEBUG:
            print(f"[DEBUG] Raw response: {content}")

        return json.loads(content)
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] API error: {e}")
        return None

def should_skip_question(text):
    t = text.lower()
    junk_triggers = [
        "honor code", "academic integrity", "policies", 
        "adhere to", "judicial board", "leaving this question blank",
        "docstrings", "merely examples", "by selecting", "agree that"
    ]
    if any(trigger in t for trigger in junk_triggers):
        return True
    return False

def should_use_vision(q_data):
    text = q_data.get("text", "")
    images = q_data.get("image_crops", [])
    if not images:
        return False
    
    vision_keywords = ["shown", "diagram", "figure", "graph", "tree", "circuit", "table", "plot", "chart"]
    for k in vision_keywords:
        if k in text.lower():
            return True
    
    has_mcq = re.search(r"(\b[A-E1-5][\.\)]\s)|(\([a-e]\))", text)
    is_tf = "true" in text.lower() and "false" in text.lower()
    
    if (has_mcq or is_tf) and len(text) > 30:
        return False
    if len(text) > 60:
        return False
    
    return True

def detect_format(text):
    text_lower = text.lower()
    if "true" in text_lower and "false" in text_lower:
        if len(text) < 200 or "select" in text_lower:
            return "TRUE_FALSE"
    has_mcq = re.search(r"(?:^|\n|\s)(?:[A-E]|[1-5])[\.\)]\s+\w+", text)
    if has_mcq:
        return "MCQ"
    return "FREE_RESPONSE"

def judge_equivalence(intended, solved):
    prompt = f"""
    You are a lenient Computer Science TA grading a student's answer.

    Key (Correct Logic): "{intended}"
    Student Code/Answer: "{solved}"
    
    TASK: Determine if the Student's logic produces the correct result.
    GRADING RULES:
    1. IGNORE syntax differences (e.g., 'for loop' vs 'sum()' function are EQUAL).
    2. IGNORE variable names.
    3. If the logic solves the problem described in the Key, MARK IT TRUE.
    4. Only mark FALSE if the logic is fundamentally wrong.
    
    Return JSON: {{ "match": true, "reason": "..." }}
    """
    return call_ollama(prompt, JUDGING_MODEL)

def normalize_answer(ans):
    s = str(ans).strip().upper()
    if s in ['TRUE', 'T', 'YES']: return 'TRUE'
    if s in ['FALSE', 'F', 'NO']: return 'FALSE'
    match = re.search(r"(?:^|\s|\.|^OPTION\s)([A-E1-5])(?:$|\s|\.|[\)])", s)
    val = match.group(1) if match else s[:1]

    mapping = {'1':'A', '2':'B', '3':'C', '4':'D', '5':'E'}
    return mapping.get(val, val)

# --- main pipeline ---
def run_pipeline(index):
    if not DB_PATH.exists(): return
    with open(DB_PATH, "r", encoding="utf-8") as f: db = json.load(f)
    q = db["ingestions"][-1]["questions"][index]
    
    print(f"\n--- Starting pipeline for question index {index} ---")

    if should_skip_question(q['text']):
        print("Skipping: Detected as non-question (policy/instructions).")
        return None
    
    use_vision = should_use_vision(q)
    gen_model = VISION_MODEL if use_vision else CODING_MODEL
    images = [encode_image(q["image_crops"][0])] if (use_vision and q.get("image_crops")) else []
    
    # format detection
    forced_type = detect_format(q['text'])
    print(f"Format: {forced_type} | Gen Model: {gen_model}")

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"Attempt {attempt}/{MAX_RETRIES}...")
        
        # generation
        gen_prompt = f"""
        Create a LOGICAL VARIANT of this CS question.
        Original: {q['text']}
        
        INSTRUCTIONS:
        1. KEEP the same concept and difficulty.
        2. CHANGE specific values, logic structures (e.g. while to for), or scenarios.
        3. You MUST generate a "{forced_type}" question.
        4. Ensure the 'correct_answer' is complete and matches the format.
        
        OUTPUT JSON:
        {{
            "type": "{forced_type}",
            "variant_text": "...",
            "options": {{...}} or null,
            "correct_answer": "..."
        }}
        """
        
        variant = call_ollama(gen_prompt, gen_model, images)
        if not variant: 
            print("Generation returned null.")
            continue

        if DEBUG:
            print(f"[DEBUG] Variant text: {variant.get('variant_text')}")

        # verification using qwen coder
        verify_prompt = f"""
        Solve this question accurately.
        Question: {variant['variant_text']}
        Options: {variant.get('options', 'None')}
        
        INSTRUCTIONS:
        - If MCQ: Return the Label (A, B, C, D, E).
        - If True/False: Return 'True' or 'False'.
        - If Free Response: Write the code or explanation clearly. Do NOT return 'None'.
        
        OUTPUT JSON:
        {{
            "reasoning": "Brief logic...",
            "final_answer": "..."
        }}
        """
        
        solution = call_ollama(verify_prompt, CODING_MODEL)
        if not solution: 
            print("Solver returned null.")
            continue

        if DEBUG:
            print(f"[DEBUG] Solver answer: {solution.get('final_answer')}")

        # logic check
        gen_ans = str(variant.get('correct_answer')).strip()
        sol_ans = str(solution.get('final_answer')).strip()
        
        verified = False
        if forced_type in ["MCQ", "TRUE_FALSE"]:
            g_val = normalize_answer(gen_ans)
            s_val = normalize_answer(sol_ans)
            
            if g_val == s_val:
                verified = True
            else:
                print(f"Mismatch: Generator='{g_val}' vs Solver='{s_val}'")
        else:
            print("Judging answer equivalence...")
            judgment = judge_equivalence(gen_ans, sol_ans)
            if judgment and judgment.get('match'):
                verified = True
            else:
                reason = judgment.get('reason') if judgment else 'No response'
                print(f"Judge failed: {reason}")

        if verified:
            print(f"Success: variant verified on attempt {attempt}")
            return {
                "original_id": q.get("question_id"),
                "type": forced_type,
                "question": variant['variant_text'],
                "options": variant.get('options'),
                "answer": gen_ans
            }
            
    print(f"Failed verification after {MAX_RETRIES} attempts")
    return None

if __name__ == "__main__":
    run_pipeline(3)
