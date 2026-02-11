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
        print(f"\n[DEBUG] calling model: {model}")
        print(f"[DEBUG] prompt preview: {prompt}")

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
            print(f"[DEBUG] raw response: {content}")
            
        return json.loads(content)
    except Exception as e:
        print(f"api error: {e}")
        return None

def should_use_vision(q_data):
    text = q_data.get("text", "")
    images = q_data.get("image_crops", [])
    
    if not images:
        if DEBUG: print("[DEBUG] routing: no images found -> text mode")
        return False
    
    # vision keywords
    vision_keywords = ["shown", "diagram", "figure", "graph", "tree", "circuit", "table"]
    for k in vision_keywords:
        if k in text.lower():
            if DEBUG: print(f"[DEBUG] routing: found visual keyword '{k}' -> vision mode")
            return True
    
    # text heuristics
    has_mcq = re.search(r"(\b[A-E1-5][\.\)]\s)|(\([a-e]\))", text)
    is_tf = "true" in text.lower() and "false" in text.lower()
    
    if (has_mcq or is_tf) and len(text) > 40:
        if DEBUG: print("[DEBUG] routing: detected text-only mcq/tf pattern -> text mode")
        return False
    
    if DEBUG: print("[DEBUG] routing: default fallback -> vision mode")
    return True

def judge_equivalence(intended, solved):
    prompt = f"""
    You are a lenient Computer Science TA grading a student's answer.
    
    Key (Correct Answer): "{intended}"
    Student Response:     "{solved}"
    
    TASK: Determine if the Student Response is correct based on the Key.
    
    GRADING RULES:
    1. If the student includes the core concept from the Key, MARK IT TRUE.
    2. If the student adds extra correct details or context, MARK IT TRUE.
    3. If the student is much more detailed than the key but correct, MARK IT TRUE.
    4. Only mark FALSE if the student contradicts the key or is factually wrong.
    
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

# --- main pipline ---
def run_pipeline(index):
    if not DB_PATH.exists(): return
    with open(DB_PATH, "r") as f: db = json.load(f)
    q = db["ingestions"][-1]["questions"][index]
    
    print(f"\n--- starting pipeline for question {index} ---")
    
    # routing
    use_vision = should_use_vision(q)
    gen_model = "llama3.2-vision" if use_vision else "llama3.2"
    images = [encode_image(q["image_crops"][0])] if (use_vision and q.get("image_crops")) else []

    # generation
    gen_prompt = f"""
    Create a variant of this CS question.
    Original: {q['text']}
    
    INSTRUCTIONS:
    1. Identify the format: MCQ, TRUE_FALSE, or FREE_RESPONSE.
    2. Generate a variant using the SAME format.
    
    FORMAT RULES:
    - If MCQ: Provide "options" {{ "A": "...", "B": "...", ... }}.
    - If TRUE_FALSE: Set "options" to null. The answer MUST be "True" or "False".
    - If FREE_RESPONSE: Set "options" to null.
    
    OUTPUT JSON:
    {{
        "type": "MCQ" or "TRUE_FALSE" or "FREE_RESPONSE",
        "variant_text": "The question text...",
        "options": {{...}} or null,
        "correct_answer": "The label (A), boolean (True/False), or code string"
    }}
    """
    
    variant = call_ollama(gen_prompt, gen_model, images)
    if not variant: return

    if DEBUG:
        print(f"[DEBUG] full generated object:\n{json.dumps(variant, indent=2)}")

    q_type = variant.get("type", "FREE_RESPONSE")

    # verification
    verify_prompt = f"""
    Solve this question.
    Question: {variant['variant_text']}
    Options: {variant.get('options', 'None')}
    
    INSTRUCTIONS:
    - If MCQ: Return the Label (A, B, C, D, E).
    - If True/False: Return 'True' or 'False'.
    - If Free Response: Return a CONCISE 1-2 sentence answer. Do not lecture.
    
    OUTPUT JSON:
    {{
        "reasoning": "Step-by-step logic...",
        "final_answer": "..."
    }}
    """
    
    solution = call_ollama(verify_prompt, "llama3.2")
    if not solution: return
    
    if DEBUG:
        print(f"[DEBUG] solver reasoning: {solution.get('reasoning')}")
        print(f"[DEBUG] solver raw answer: {solution.get('final_answer')}")

    # logic check
    gen_ans = str(variant.get('correct_answer')).strip()
    sol_ans = str(solution.get('final_answer')).strip()
    
    verified = False
    
    if q_type in ["MCQ", "TRUE_FALSE"]:
        g_val = normalize_answer(gen_ans)
        s_val = normalize_answer(sol_ans)
        
        if DEBUG:
            print(f"[DEBUG] normalized comparison: generator='{g_val}' vs solver='{s_val}'")
        
        if g_val == s_val:
            verified = True
        else:
            print(f"mismatch: {g_val} vs {s_val}")
            
    else:
        print("judging answer equivalence")
        judgment = judge_equivalence(gen_ans, sol_ans)
        if DEBUG:
            print(f"[DEBUG] judge result: {judgment}")
            
        if judgment and judgment.get('match'):
            verified = True
        else:
            print(f"judgment failed: {judgment.get('reason')}")

    if verified:
        print("success: variant verified")
        return variant
    else:
        print("failed verification")
        return None

if __name__ == "__main__":
    run_pipeline(5)
