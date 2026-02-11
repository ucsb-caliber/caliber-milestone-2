import json
import requests
import base64
import re
from pathlib import Path

# config
DB_PATH = Path("layout_debug/questions.json")
OLLAMA_URL = "http://localhost:11434/api/chat"
DEBUG = True  # set to True to see full logs

# --- helper functions ---
def encode_image(path):
    if not Path(path).exists(): return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def call_ollama(prompt, model, images=None):
    if DEBUG:
        print(f"\n[DEBUG] Calling model: {model}")
        print(f"[DEBUG] Prompt preview: {prompt}")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images or []}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": 4096}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        content = response.json()['message']['content']

        if DEBUG:
            print(f"[DEBUG] Raw response: {content}")

        return json.loads(content)
    except Exception as e:
        print(f"api error: {e}")
        return None

def should_use_vision(q_data):
    text = q_data.get("text", "")
    images = q_data.get("image_crops", [])
    if not images:
        if DEBUG: print("[DEBUG] Routing: no images found -> text mode")
        return False
    
    vision_keywords = ["shown", "diagram", "figure", "graph", "tree", "circuit", "table"]
    for k in vision_keywords:
        if k in text.lower():
            if DEBUG: print(f"[DEBUG] Routing: found visual keyword '{k}' -> vision mode")
            return True
    
    # text heuristics
    has_mcq = re.search(r"(\b[A-E1-5][\.\)]\s)|(\([a-e]\))", text)
    is_tf = "true" in text.lower() and "false" in text.lower()
    
    if (has_mcq or is_tf) and len(text) > 40:
        if DEBUG: print("[DEBUG] Routing: detected text-only mcq/tf pattern -> text mode")
        return False
    
    if DEBUG: print("[DEBUG] Routing: default fallback -> vision mode")
    return True

def detect_format(text):
    text_lower = text.lower()
    
    # check for true/false pattern
    if "true" in text_lower and "false" in text_lower:
        if len(text) < 200 or "select" in text_lower:
            return "TRUE_FALSE"

    # check for mcq pattern (A. B. C. or 1. 2. 3.)
    # looks for A. or (a) at start of lines or sentences
    has_mcq = re.search(r"(?:^|\n|\s)(?:[A-E]|[1-5])[\.\)]\s+\w+", text)
    if has_mcq:
        return "MCQ"
        
    return "FREE_RESPONSE"

def judge_equivalence(intended, solved):
    prompt = f"""
    You are a lenient Computer Science TA grading a student's answer.
    Key (Correct Answer): "{intended}"
    Student Response:     "{solved}"
    
    TASK: Determine if the Student Response is correct based on the Key.
    
    GRADING RULES:
    1. If the student includes the core concept from the Key, MARK IT TRUE.
    2. If the student adds extra correct details, MARK IT TRUE.
    3. Only mark FALSE if the student contradicts the key.
    
    Return JSON: {{ "match": true, "reason": "..." }}
    """
    return call_ollama(prompt, "llama3.2")

def normalize_answer(ans):
    s = str(ans).strip().upper()
    if s in ['TRUE', 'T']: return 'TRUE'
    if s in ['FALSE', 'F']: return 'FALSE'
    
    match = re.search(r"(?:^|\s|\.|^OPTION\s)([A-E1-5])(?:$|\s|\.|[\)])", s)
    val = match.group(1) if match else s[:1]
    
    mapping = {'1':'A', '2':'B', '3':'C', '4':'D', '5':'E'}
    return mapping.get(val, val)

# --- main pipeline ---
def run_pipeline(index):
    if not DB_PATH.exists(): return
    with open(DB_PATH, "r") as f: db = json.load(f)
    q = db["ingestions"][-1]["questions"][index]
    
    print(f"\n--- Starting pipeline for question {index} ---")
    
    # routing
    use_vision = should_use_vision(q)
    gen_model = "llama3.2-vision" if use_vision else "llama3.2"
    images = [encode_image(q["image_crops"][0])] if (use_vision and q.get("image_crops")) else []
    
    # format detection
    forced_type = detect_format(q['text'])
    print(f"[DEBUG] Detected format: {forced_type}")

    # generation
    gen_prompt = f"""
    Create a variant of this CS question.
    Original: {q['text']}
    
    INSTRUCTIONS:
    1. You MUST generate a "{forced_type}" question.
    2. Do NOT change the format.
    
    FORMAT RULES:
    - If MCQ: Provide "options" {{ "A": "...", "B": "...", ... }}.
    - If TRUE_FALSE: Set "options" to null. Answer is "True" or "False".
    - If FREE_RESPONSE: Set "options" to null. Answer is a string.
    
    OUTPUT JSON:
    {{
        "type": "{forced_type}",
        "variant_text": "The question text...",
        "options": {{...}} or null,
        "correct_answer": "..."
    }}
    """
    
    variant = call_ollama(gen_prompt, gen_model, images)
    if not variant: return

    if DEBUG:
        print(f"[DEBUG] Generated variant text: {variant.get('variant_text')}")

    # verification
    verify_prompt = f"""
    Solve this question.
    Question: {variant['variant_text']}
    Options: {variant.get('options', 'None')}
    
    INSTRUCTIONS:
    - If MCQ: Return the Label (A, B, C, D, E).
    - If True/False: Return 'True' or 'False'.
    - If Free Response: Provide a clear, accurate explanation. Include key details.
    
    OUTPUT JSON:
    {{
        "reasoning": "Brief logic...",
        "final_answer": "..."
    }}
    """
    
    solution = call_ollama(verify_prompt, "llama3.2")
    if not solution: return

    # judgment
    if DEBUG:
        print(f"[DEBUG] Solver reasoning: {solution.get('reasoning')}")
        print(f"[DEBUG] Solver raw answer: {solution.get('final_answer')}")

    # logic check
    gen_ans = str(variant.get('correct_answer')).strip()
    sol_ans = str(solution.get('final_answer')).strip()
    
    verified = False
    
    if forced_type in ["MCQ", "TRUE_FALSE"]:
        g_val = normalize_answer(gen_ans)
        s_val = normalize_answer(sol_ans)
        
        if DEBUG:
            print(f"[DEBUG] Normalized comparison: generator='{g_val}' vs solver='{s_val}'")

        if g_val == s_val:
            verified = True
        else:
            print(f"Mismatch: {g_val} vs {s_val}")
            
    else:
        print("Judging answer equivalence")
        judgment = judge_equivalence(gen_ans, sol_ans)
        if DEBUG:
            print(f"[DEBUG] Judge result: {judgment}")

        if judgment and judgment.get('match'):
            verified = True
        else:
            print(f"Judgment failed: {judgment.get('reason')}")

    # result
    if verified:
        print("Success: variant verified")
        return {
            "original_id": q.get("question_id"),
            "type": forced_type,
            "question": variant['variant_text'],
            "options": variant.get('options'),
            "answer": gen_ans
        }
    else:
        print("Failed verification")
        return None

if __name__ == "__main__":
    run_pipeline(3)
