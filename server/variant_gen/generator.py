import ast
import json
import requests
import base64
import re
from pathlib import Path
from difflib import SequenceMatcher
import os
from dotenv import load_dotenv

from .question_contract import (
    build_question_contract,
    fence_lang,
    free_response_cs_vocabulary_lost,
    language_display,
    scenario_from_contract,
)
from .policies import get_policy

_SERVER_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_SERVER_DIR / ".env")
load_dotenv()

# config (repo root = server/variant_gen -> server -> repo)
BASE_DIR = _SERVER_DIR.parent
DB_PATH = BASE_DIR / "layout_debug" / "questions.json"
OLLAMA_URL = "http://localhost:11434/api/chat"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Model selection ──────────────────────────────────────────────────
# Two models: one for text (generation + verification), one for vision.
# The pipeline is 2 LLM calls per question: generate → verify.
#
#   GPU               VRAM   TEXT_MODEL           VISION_MODEL
#   ──────────────────────────────────────────────────────────
#   GTX 1080 / 3060    8 GB   qwen2.5-coder:7b     moondream
#   RTX 3090 / 4080   16+ GB  qwen2.5-coder:7b     llama3.2-vision
#
# Set VISION_MODEL = None to disable vision entirely.
#
# Local: LLM_BACKEND unset or ollama → TEXT_MODEL is an Ollama tag (e.g. qwen2.5-coder:14b).
# Cloud: LLM_BACKEND=openrouter, OPENROUTER_API_KEY, OPENROUTER_MODEL (e.g. openai/gpt-4o-mini).
#        Optional: OPENROUTER_SEND_IMAGES=0 if the chosen model is text-only.
#        QUESTION_LANG=python|cpp|java|generic overrides auto-detect (see variant_gen.question_contract).
#        Routing / reskin policy: build_question_contract (future: QUESTION_ROUTER=llm).
#
TEXT_MODEL   = "qwen2.5-coder:14b"   # Ollama tag, or fallback when OPENROUTER_MODEL is empty
VISION_MODEL = "moondream"           # only for deciding if a crop exists; JSON gen uses TEXT_MODEL / OpenRouter

DEBUG = False
MAX_RETRIES = 3
BASE_TEMPERATURE = 0.4
SIMILARITY_THRESHOLD = 0.6
# Explanation / "why" questions keep most of the stem; only the hook changes — allow higher overlap.
SIMILARITY_THRESHOLD_EXPLANATION = 0.72

# ── Algorithm extraction ─────────────────────────────────────────────
# Keyword-based detection of the core algorithmic concept.
# Used to tell the generator which structure must be preserved.
_ALGORITHM_PATTERNS = [
    ("hash map lookup",     ["hash", "dictionary", "dict", "lookup", "two sum", "key", "mapping"]),
    ("graph traversal",     ["graph", "bfs", "dfs", "breadth", "depth", "adjacen", "vertex", "edge", "shortest path", "dijkstra", "spanning tree", "kruskal"]),
    ("dynamic programming", ["dynamic programming", "memoiz", "tabulation", "subproblem", "overlapping", "optimal substructure", "longest common", "knapsack"]),
    ("sorting / ordering",  ["sort", "order", "rank", "arrange", "ascending", "descending", "merge sort", "quicksort", "bubble"]),
    ("binary search",       ["binary search", "bisect", "sorted array", "log n", "divide and conquer"]),
    ("recursion",           ["recurs", "base case", "recursive"]),
    ("tree traversal",      ["tree", "preorder", "inorder", "postorder", "binary tree", "bst", "traversal"]),
    ("linked list",         ["linked list", "singly linked", "head", "node", "next pointer", "reverse"]),
    ("stack / queue",       ["stack", "queue", "push", "pop", "peek", "fifo", "lifo"]),
    ("set operations",      ["set", "intersection", "union", "difference", "common elements"]),
    ("string manipulation", ["string", "substring", "reverse", "palindrome", "anagram", "character"]),
    ("array search",        ["array", "list", "find", "search", "index", "element"]),
    ("greedy",              ["greedy", "optimal", "locally optimal"]),
    ("class design / OOP",  ["class", "inherit", "object", "method", "__init__", "__str__", "attribute"]),
]


def extract_algorithm(text):
    t = text.lower()
    matches = []
    for algo_name, keywords in _ALGORITHM_PATTERNS:
        score = sum(1 for kw in keywords if kw in t)
        if score > 0:
            matches.append((algo_name, score))
    if not matches:
        return "general problem solving"
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0][0]


# Scenario lists and routing: question_contract.py (QuestionContract).

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def encode_image(path, data_url=False):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    if not data_url:
        return b64
    mime = _IMAGE_MIME.get(p.suffix.lower(), "image/png")
    return f"data:{mime};base64,{b64}"


def _llm_backend():
    b = os.getenv("LLM_BACKEND", "ollama").strip().lower()
    return b if b in ("ollama", "openrouter") else "ollama"


def _openrouter_model_id():
    return os.getenv("OPENROUTER_MODEL", "").strip()


def _parse_llm_json(content):
    if not content or not isinstance(content, str):
        return None
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"  Model returned invalid JSON: {e}")
        if DEBUG:
            print(f"[DEBUG] Raw content was: {content[:2000]}")
        return None


_ollama_reachable = None


def _ollama_chat(prompt, model, image_paths, temp):
    global _ollama_reachable
    images = []
    for p in image_paths:
        b64 = encode_image(p)
        if b64:
            images.append(b64)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images}],
        "stream": False,
        "format": "json",
        "options": {"temperature": temp, "num_ctx": 8192},
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        _ollama_reachable = True
        content = response.json()["message"]["content"]
        if DEBUG:
            print(f"[DEBUG] Raw response: {content}")
        return _parse_llm_json(content)
    except requests.ConnectionError:
        if _ollama_reachable is None:
            print(f"\n  ERROR: Cannot connect to Ollama at {OLLAMA_URL}")
            print("  Is Ollama running?  Try: ollama serve")
            _ollama_reachable = False
        return None
    except requests.Timeout:
        print("  Request timed out (120s).")
        return None
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        print(f"  Ollama returned HTTP {status}: {e}")
        return None
    except Exception as e:
        print(f"  Unexpected error: {type(e).__name__}: {e}")
        return None


def _openrouter_chat(prompt, image_paths, temp):
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("  OPENROUTER_API_KEY is not set.")
        return None
    mid = _openrouter_model_id()
    if not mid:
        print("  OPENROUTER_MODEL is not set (example: openai/gpt-4o-mini).")
        return None

    if image_paths:
        parts = [{"type": "text", "text": prompt}]
        for p in image_paths:
            url = encode_image(p, data_url=True)
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
        user_content = parts
    else:
        user_content = prompt

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    ref = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    if ref:
        headers["HTTP-Referer"] = ref
    title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
    if title:
        headers["X-Title"] = title

    payload = {
        "model": mid,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temp,
        "response_format": {"type": "json_object"},
    }
    try:
        response = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=120)
        if not response.ok:
            try:
                err = response.json()
                detail = err.get("error", err)
            except Exception:
                detail = response.text[:500]
            print(f"  OpenRouter HTTP {response.status_code}: {detail}")
            return None
        data = response.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        if DEBUG:
            print(f"[DEBUG] Raw response: {content}")
        return _parse_llm_json(content)
    except requests.Timeout:
        print("  Request timed out (120s).")
        return None
    except Exception as e:
        print(f"  Unexpected error: {type(e).__name__}: {e}")
        return None


def call_llm(prompt, model, image_paths=None, temperature=None):
    """Generate or verify: JSON object in, JSON object out."""
    image_paths = [Path(x) for x in (image_paths or []) if x]
    temp = temperature if temperature is not None else BASE_TEMPERATURE
    backend = _llm_backend()
    label = _openrouter_model_id() if backend == "openrouter" else model
    if DEBUG:
        print(f"\n[DEBUG] Calling model ({backend}): {label}")

    if backend == "openrouter":
        return _openrouter_chat(prompt, image_paths, temp)
    return _ollama_chat(prompt, model, image_paths, temp)


def should_skip_question(text):
    t = text.lower()
    junk_triggers = [
        "honor code",
        "academic integrity",
        "policies",
        "adhere to",
        "judicial board",
        "leaving this question blank",
        "docstrings",
        "merely examples",
        "by selecting",
        "agree that",
    ]
    return any(trigger in t for trigger in junk_triggers)


def _is_code_or_written_answer_question(text):
    """Coding / written exam items: OCR text is authoritative; skip vision for generation."""
    t = text.lower()
    return any(
        p in t
        for p in (
            "write a function",
            "write a class",
            "write pseudocode",
            "recursive function",
            "def ",
            "__init__",
            "class named",
            "inherits",
        )
    )


def should_use_vision(q_data):
    """Whether we may attach an image to the *text* model (if it supports vision)."""
    if not VISION_MODEL:
        return False

    text = q_data.get("text", "")
    images = q_data.get("image_crops", [])
    if not images:
        return False

    if _is_code_or_written_answer_question(text):
        return False

    vision_keywords = [
        "shown", "diagram", "figure", "graph", "tree",
        "circuit", "table", "plot", "chart",
    ]
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


def text_model_supports_images():
    if _llm_backend() == "openrouter":
        return os.getenv("OPENROUTER_SEND_IMAGES", "1").lower() not in ("0", "false", "no")
    return "vision" in TEXT_MODEL.lower()


def detect_format(text):
    text_lower = text.lower()
    if "true" in text_lower and "false" in text_lower:
        if len(text) < 200 or "select" in text_lower:
            return "TRUE_FALSE"
    has_mcq = re.search(r"(?:^|\n|\s)(?:[A-E]|[1-5])[\.\)]\s+\w+", text)
    if has_mcq:
        return "MCQ"
    return "FREE_RESPONSE"


def _similarity_threshold_for_original(original_text):
    t = (original_text or "").lower()
    if any(
        k in t
        for k in (
            "explain ",
            "why is it",
            "why is ",
            "important to ",
            "difference between",
            "what is the difference",
            "name at least",
            "how do they differ",
        )
    ):
        return SIMILARITY_THRESHOLD_EXPLANATION
    return SIMILARITY_THRESHOLD


def is_too_similar(original_text, variant_text):
    thresh = _similarity_threshold_for_original(original_text)
    ratio = SequenceMatcher(None, original_text.lower(), variant_text.lower()).ratio()
    if DEBUG:
        print(f"[DEBUG] Similarity ratio: {ratio:.2f} (threshold {thresh:.2f})")
    return ratio > thresh


_BAD_PLACEHOLDER_FRAGMENTS = (
    "1-2 sentence real-world scenario",
    "the full problem statement combining",
    "the clear problem definition",
    "any constraints or rules carried over",
    "the complete correct answer",
)

_META_ANSWER_SNIPPETS = (
    "the correct answer should",
    "the student should",
    "your answer should be",
    "acceptable responses",
    "grading rubric",
    "must be a function definition that",
    "solution should use",
)


def _original_asks_for_code_submission(original_text):
    """True when the source exam text asks students to submit code (not prose-only)."""
    t = (original_text or "").lower()
    return any(
        p in t
        for p in (
            "write a function",
            "write a class",
            "write pseudocode",
            "recursive function",
            "implement a function",
            "define a function",
        )
    )


def _variant_asks_for_python_code(variant):
    blob = " ".join(
        str(variant.get(k) or "")
        for k in ("variant_text", "task", "constraints")
    ).lower()
    return any(
        p in blob
        for p in (
            "write a function",
            "write a class",
            "write pseudocode",
            "recursive function",
            "implement a function",
            "define a function",
        )
    )


def _stem_asks_for_non_method(stem_lower):
    s = stem_lower or ""
    if "method" not in s or not re.search(r"\bnot\b", s):
        return False
    if re.search(r"\bnot\s+a\s+list\s+method\b", s):
        return True
    if re.search(r"\bnot\b.?\s*(one\s+of\s+)?the\s+following", s):
        return True
    if re.search(r"which\b.*\bnot\b.*\bmethod\b", s):
        return True
    return False


def _mcq_correct_option_label(correct_answer, options):
    """Resolve correct_answer to an options dict key (supports 1–5 vs A–E mismatch)."""
    if not options or not isinstance(options, dict):
        return None, None
    ca = str(correct_answer).strip()
    candidates = {ca, normalize_answer(ca)}
    m = re.match(r"^([A-E])", ca, re.I)
    if m:
        L = m.group(1).upper()
        candidates.add(L)
        candidates.add(str(ord(L) - ord("A") + 1))
    m = re.match(r"^([1-5])", ca)
    if m:
        num = m.group(1)
        candidates.add(num)
        candidates.add(chr(ord("A") + int(num) - 1))
    na = normalize_answer(ca)
    if na in ("A", "B", "C", "D", "E"):
        candidates.add(str(ord(na) - ord("A") + 1))
    for k in candidates:
        if k and k in options:
            return k, str(options[k]).strip().lower()
    return None, None


def _answer_snippet_in_stem(stem_blob, answer, min_len=28):
    if not answer or len(answer) < min_len:
        return False
    a = answer.strip().lower()
    s = (stem_blob or "").lower()
    win = min(72, max(min_len, len(a) // 2))
    if len(a) < win:
        return a in s
    for i in range(0, len(a) - win + 1, max(1, win // 3)):
        if a[i : i + win] in s:
            return True
    return False


def _norm_code_line(line):
    return re.sub(r"\s+", " ", (line or "").strip())


def _free_response_code_leaked_into_stem(stem_blob, ca_str, lang):
    """
    True when the student-facing stem repeats substantive lines from correct_answer.
    Typical when ingest merged the answer key with the prompt (OOP “write a class” items).
    Only applies when the model answer defines a class — pure ``def`` solutions (e.g. recursion)
    often legitimately echo the function signature in the stem.
    """
    if lang != "python":
        return False
    if not _answer_looks_like_code(ca_str, lang):
        return False
    code = _extract_python_for_parse(ca_str)
    if len(code) < 60 or "class " not in code:
        return False

    ans_lines = []
    for line in code.splitlines():
        n = _norm_code_line(line)
        if len(n) < 12 or n.startswith("#"):
            continue
        ans_lines.append(n)
    if len(ans_lines) < 3:
        return False
    ans_set = set(ans_lines)

    stem_lines = {_norm_code_line(l) for l in (stem_blob or "").splitlines()}
    stem_lines.discard("")
    overlap = ans_set & stem_lines
    if len(overlap) >= 4:
        return True
    if len(overlap) >= 2 and sum(len(x) for x in overlap) >= 100:
        return True
    if len(overlap) >= 3 and len(overlap) / len(ans_set) >= 0.45:
        return True
    return False


def _free_response_fenced_code_matches_answer(stem_blob, ca_str, lang):
    """
    Full solution pasted in a ```python``` block in the stem (common when ingest merged answer key).
    Catches recursion / function tasks, not only class-based OOP.
    """
    if lang != "python" or not _answer_looks_like_code(ca_str, lang):
        return False
    ca_code = _extract_python_for_parse(ca_str)
    if len(ca_code) < 40:
        return False
    ca_compact = re.sub(r"\s+", " ", ca_code.strip())
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", stem_blob or "", re.DOTALL | re.IGNORECASE)
    for blk in blocks:
        b = blk.strip()
        if len(b) < 35:
            continue
        b_compact = re.sub(r"\s+", " ", b)
        if SequenceMatcher(None, ca_compact, b_compact).ratio() > 0.82:
            return True
    return False


def _extract_python_for_parse(s):
    # Kept for backwards-compat with local helpers; policy module owns parsing rules.
    if not s:
        return ""
    m = re.search(r"```(?:python)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s.strip()


def _line_looks_like_code_definition(line, lang):
    if lang == "python":
        return bool(re.search(r"^\s*(def|class)\s+\w", line))
    if lang in ("cpp", "java", "generic"):
        if re.search(r"^\s*(class|struct|template|namespace)\b", line):
            return True
        if lang == "java" and re.search(r"^\s*(public|private|protected)\s+(\w+\s+)*\w+\s+\w+\s*\(", line):
            return True
        if re.search(
            r"^\s*(inline\s+)?(static\s+)?(void|bool|int|long|double|float|char|auto|string|unsigned)\b.+\([^)]*\)",
            line,
        ):
            return True
        if re.search(r"^\s*[\w:<>,\s&*]+\s+\w+\s*\([^)]*\)\s*\{?\s*$", line):
            return True
    return bool(re.search(r"^\s*(def|class)\s+\w", line))


def _variant_code_blob_heuristic(variant_text, lang):
    vt = variant_text or ""
    if vt.count("```") >= 2:
        return False
    if vt.count("\n") >= 6:
        return False
    lines = vt.splitlines()
    has_code_line = any(_line_looks_like_code_definition(l, lang) for l in lines)
    if not has_code_line:
        return False
    if len(vt) > 480 and vt.count("\n") < 4:
        return True
    for line in lines:
        if _line_looks_like_code_definition(line, lang) and len(line) > 160:
            return True
    return False


def _answer_looks_like_code(ca, lang):
    s = ca or ""
    sl = s.lower()
    if lang == "python":
        return "def " in s or "class " in sl
    if lang == "cpp":
        return bool(
            re.search(r"\b(class|struct|void|int|bool)\b", sl)
            and ("{" in s or ";" in s)
        ) or "#include" in s
    if lang == "java":
        return "class " in sl or re.search(r"\b(public|private)\s+.*\(", s) is not None
    return bool(re.search(r"def\s+\w+|class\s+\w+", sl) or ("{" in s and ";" in s))


def _free_response_correct_answer_invalid(correct_answer, variant, lang, contract, original_text):
    if correct_answer is None:
        ca = ""
    elif isinstance(correct_answer, str):
        ca = correct_answer.strip()
    else:
        ca = str(correct_answer).strip()
    if not ca:
        return "empty correct_answer"
    ca_lower = ca.lower()
    for frag in _META_ANSWER_SNIPPETS:
        if frag in ca_lower:
            return "correct_answer is meta/rubric, not a concrete solution"

    require_code = _variant_asks_for_python_code(variant)
    if (
        require_code
        and contract.mode == "conceptual"
        and not _original_asks_for_code_submission(original_text or "")
    ):
        require_code = False

    if require_code:
        if not _answer_looks_like_code(ca, lang):
            return f"coding task requires correct_answer with real {language_display(lang)} code"

    return None


def _autofix_list_method_mcq(variant, original_text, lang):
    """If only one option is a real list method (or one is not, for NOT-questions), set letter."""
    if lang != "python":
        return False
    pol = get_policy("python")
    return pol.list_method_mcq_autofix(
        variant,
        original_text,
        normalize_answer=normalize_answer,
        mcq_correct_option_label=_mcq_correct_option_label,
    )


def is_invalid_variant(variant, forced_type, expected_mcq_options, original_text, contract):
    """Reject models that echo JSON schema instructions or return garbage."""
    lang = contract.language
    vt_raw = (variant.get("variant_text") or "").strip()
    vt = vt_raw.lower()
    if len(vt) < 40:
        return "variant_text too short"

    for frag in _BAD_PLACEHOLDER_FRAGMENTS:
        if frag in vt:
            return f"placeholder text in variant_text: {frag[:40]}"

    for field in ("storyline", "task"):
        val = (variant.get(field) or "").strip().lower()
        for frag in _BAD_PLACEHOLDER_FRAGMENTS:
            if frag in val:
                return f"placeholder in {field}"

    ca = variant.get("correct_answer")
    if isinstance(ca, (list, dict)):
        return "correct_answer is not a string"
    if ca is None:
        ca = ""

    stem_blob = " ".join(
        str(variant.get(k) or "") for k in ("storyline", "task", "variant_text", "constraints")
    )

    if forced_type == "FREE_RESPONSE":
        fr_err = _free_response_correct_answer_invalid(
            ca, variant, lang, contract, original_text
        )
        if fr_err:
            return fr_err
        if free_response_cs_vocabulary_lost(original_text, variant, contract):
            return "conceptual FR drifted off-topic (keep CS terms from the original, not a novelty theme)"
        ca_str = ca if isinstance(ca, str) else str(ca)
        if _free_response_code_leaked_into_stem(stem_blob, ca_str, lang):
            return (
                "variant_text repeats correct_answer implementation (answer key in ingest?) — "
                "keep specs in stem, code only in correct_answer"
            )
        if _free_response_fenced_code_matches_answer(stem_blob, ca_str, lang):
            return (
                "fenced code in variant_text matches correct_answer — remove full solution from stem "
                "(spec or empty starter only)"
            )
        skip_overlap = _answer_looks_like_code(ca_str, lang)
        if not skip_overlap and _answer_snippet_in_stem(stem_blob, ca_str):
            return "answer text appears to be copied into the stem"
        if _variant_code_blob_heuristic(vt_raw, lang):
            return (
                f"code in variant_text is crammed (use fenced ```{fence_lang(lang)} ``` blocks and line breaks)"
            )
        if lang == "python":
            pol = get_policy("python")
            if pol and not pol.answer_parseable(ca_str):
                return "correct_answer has invalid Python syntax"
            if pol and pol.answer_has_mutable_defaults(ca_str) and not pol.original_shows_mutable_defaults(original_text):
                return "correct_answer uses mutable default [] or {}; use None unless original does"

    if forced_type == "MCQ":
        opts = variant.get("options")
        if not opts or not isinstance(opts, dict):
            return "MCQ missing or invalid options"
        n = len(opts)
        if expected_mcq_options >= 2 and n != expected_mcq_options:
            return f"expected {expected_mcq_options} MCQ options, got {n}"
        vals = [str(v).strip() for v in opts.values()]
        if vals and all(re.match(r"^0\.\d+$", v) for v in vals if v):
            return "MCQ options look like garbage probabilities"

        if lang == "python":
            pol = get_policy("python")
            if pol:
                err = pol.validate_list_method_mcq(variant, original_text, vt, _mcq_correct_option_label)
                if err:
                    return err

    return None


def normalize_answer(ans):
    s = str(ans).strip().upper()
    if s in ["TRUE", "T", "YES"]:
        return "TRUE"
    if s in ["FALSE", "F", "NO"]:
        return "FALSE"
    match = re.search(r"(?:^|\s|\.|^OPTION\s)([A-E1-5])(?:$|\s|\.|[\)])", s)
    val = match.group(1) if match else s[:1]

    mapping = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    return mapping.get(val, val)


def _count_options(text):
    """Count how many MCQ options appear in the original question text."""
    letter_opts = re.findall(r"(?:^|\n|\s)[A-E][\.\)]\s", text)
    num_opts = re.findall(r"(?:^|\n|\s)[1-5][\.\)]\s", text)
    count = max(len(letter_opts), len(num_opts))
    return count if count >= 2 else 0


def _prompt_extras(original_text, forced_type, style, contract):
    t = (original_text or "").lower()
    lang = contract.language
    chunks = []

    if forced_type == "MCQ" and lang == "python":
        list_api = (
            "list method" in t
            or ("method" in t and "list" in t)
            or "elements to a list" in t
            or ("to a list" in t and "method" in t)
        )
        if list_api:
            chunks.append(
                "Python LIST API: say clearly the choices are Python built-in list methods. "
                "Each option must be a real list method name "
                "(append, extend, pop, insert, remove, clear, sort, reverse, copy, count, index). "
                "No fake names (merge, add_friends, combine). If the original asks which is NOT a method, "
                "correct_answer must be the letter for an option that is not a list method."
            )
        elif "dict" in t and "method" in t:
            chunks.append(
                "Python DICT API: options must be real dict methods; no invented names."
            )
        elif "set" in t and "method" in t:
            chunks.append(
                "Python SET API: options must be real set methods; no invented names."
            )
    elif forced_type == "MCQ" and lang == "cpp" and "vector" in t and "method" in t:
        chunks.append(
            "C++: if asking about std::vector member functions, options must be real vector API names "
            "for the standard used in the course."
        )

    if forced_type == "FREE_RESPONSE":
        if contract.mode == "conceptual":
            chunks.append(
                f"CONCEPTUAL: stay in {language_display(lang)} and CS ideas from the original; theme is only a hook. "
                "Do not replace with unrelated domains. Do not put the full solution in variant_text or task."
            )
            if not _original_asks_for_code_submission(original_text or ""):
                chunks.append(
                    "The original asks for explanation or comparison only — do NOT turn it into a coding exercise "
                    "(e.g. do not add 'write a function' or require implementation) unless the source explicitly "
                    "asked for code."
                )
        code_in_stem = any(
            k in t
            for k in (
                "write a function",
                "write a class",
                "recursive function",
                "write pseudocode",
            )
        ) or "def " in (original_text or "") or "void " in (original_text or "")
        oop_fr = contract.mode == "class_design" and lang == "python"
        if oop_fr:
            fence = fence_lang(lang)
            chunks.append(
                "INGEST MAY INCLUDE THE ANSWER KEY BELOW THE PROMPT: The original often pastes the full solution. "
                "variant_text must be the STUDENT-FACING question only—requirements in prose or short bullets. "
                "Do NOT copy class/method bodies from the solution into variant_text, storyline, or task; "
                "put the complete working code ONLY in correct_answer. "
                f"If a starter is essential, use at most one ```{fence}``` block with `pass` or `...` only, "
                "not the real implementation."
            )
        elif code_in_stem:
            fence = fence_lang(lang)
            chunks.append(
                f"Put code in variant_text inside ```{fence} ... ``` with normal line breaks and indentation."
            )
        if lang == "python" and (style == "swe" or "class named" in t or "write a class" in t):
            chunks.append(
                "OOP correct_answer: no mutable defaults (def f(self, a=[], b={})); use None in __init__ "
                "unless the original clearly shows that pattern."
            )

    if not chunks:
        return ""
    return "\nEXTRA RULES (from original):\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(chunks, start=1))


def _build_generation_prompt(original_text, forced_type, scenario, algorithm, contract):
    style = scenario.get("style", "reskin")
    lang = contract.language
    ld = language_display(lang)

    if style == "swe":
        context_block = f"""SCENARIO TO USE:
- Domain: {scenario["domain"]}
- Setting: {scenario["context"]}
- Use entities like: {", ".join(scenario["entities"])}

Rename classes, methods, and attributes to fit the domain naturally.
For example, a "Book" class might become a "Deployment" class; a "Library" might become a "ServiceRegistry"."""
    elif style == "conceptual":
        context_block = f"""VARIANT STYLE — CONCEPTUAL (definition / compare / name-at-least-N):
- The original tests CS knowledge (types, language rules, complexity, data structures, etc.).
- Keep the SAME technical subject: if it asks about sequences, lists, tuples, or strings, the variant
  must still be about those ideas in {ld} (or the same language as the original).
- You may add a one-sentence hook, but do NOT reframe the question into a non-CS domain
  (no parking lots, recipes, sports teams, travel, etc.).
- Preserve technical vocabulary students must use (e.g. sequence, tuple, immutable)."""
    else:
        context_block = f"""THEME TO USE: {scenario["theme"]}
- Swap abstract variables and entities for: {scenario["swap_nouns"]}
- Example of this style: {scenario["example"]}

Keep the problem structure almost identical to the original. Just replace the abstract
nouns/numbers with concrete, tangible things from the theme. Think of it like a word-problem
reskin: "find two numbers that sum to target" becomes "find two packages whose weights
add up to the truck's capacity"."""

    extras = _prompt_extras(original_text, forced_type, style, contract)

    format_rules = ""
    if forced_type == "MCQ":
        n_opts = _count_options(original_text) or 4
        lang_mcq = (
            "If testing library/container methods, name the type (e.g. Python list, std::vector, Java ArrayList) "
            "and use only real API names from that language."
            if lang != "generic"
            else "Use real API names for the language implied by the original."
        )
        format_rules = f"""
FORMAT CONSTRAINTS (MCQ):
- Keep the EXACT same question direction. If the original asks "which IS", ask "which IS".
  If it asks "which is NOT", ask "which is NOT". Do NOT flip it.
- Produce exactly {n_opts} options, labeled with the same scheme as the original.
- {lang_mcq}
- One option must be clearly correct; distractors should be plausible but wrong."""
    elif forced_type == "TRUE_FALSE":
        format_rules = """
FORMAT CONSTRAINTS (TRUE/FALSE):
- The statement must be clearly true or false with no ambiguity.
- Keep the same truth value as the original if possible."""
    else:
        format_rules = f"""
FORMAT CONSTRAINTS (FREE RESPONSE):
- If the original asks to write a function, the variant must ask to write a function.
- If the original asks to write a class, the variant must ask to write a class.
- If the original asks for an explanation, the variant must ask for an explanation.
- Preserve the same level of detail expected in the answer.
- correct_answer MUST be non-empty.
- If the question asks for code, correct_answer must be valid {ld} source (not prose about what it "should" do).
- If the question asks for a prose explanation, correct_answer must be a concrete model answer (real sentences),
  not grading instructions (never start with "The student should" or "The correct answer should").
- Do not put the full model answer in variant_text or task; only correct_answer holds it."""

    return f"""You are creating a variant of a CS exam question. The variant should feel like a
concrete, real-world scenario — not an abstract math or textbook exercise.

TARGET LANGUAGE: {ld}. All code, API names, and syntax in the variant must match this language.

ORIGINAL QUESTION:
\"\"\"{original_text}\"\"\"

CORE ALGORITHM: {algorithm}
This algorithmic structure MUST be preserved in the variant. Do not change what kind of
algorithm is needed to solve it.

{context_block}
{format_rules}
{extras}

STRUCTURAL RULES:
1. The variant must test the SAME concept, at the SAME difficulty, using the SAME question
   format as the original. Only the theme/nouns/values change.
2. The underlying algorithm ({algorithm}) must remain the same.
3. Replace generic variables (x, y, n, a, b) with descriptive names from the theme.
4. The problem should read like a real situation someone might actually encounter.
5. If the original has code, the variant must too — use fitting function/class names.
6. You MUST produce a "{forced_type}" question.
7. Do NOT mention the original question or call this a "variant".
8. NEVER paste schema instructions into fields. Every string field must be real exam prose
   a student would read — not phrases like "1-2 sentence scenario" or "full problem statement".

OUTPUT JSON (shape only — replace values with your own complete text):
{{
    "type": "{forced_type}",
    "storyline": "<brief real-world hook, 1-2 sentences>",
    "task": "<what the student must compute or implement>",
    "constraints": "<rules from the original, or empty string>",
    "variant_text": "<entire question: hook + task + constraints; coherent and self-contained>",
    "options": {{"A": "...", "B": "...", ...}} or null,
    "correct_answer": "<single correct answer string>"
}}"""


def _build_verify_prompt(variant_text, options, forced_type, claimed_answer, contract):
    ld = language_display(contract.language)
    if forced_type in ("MCQ", "TRUE_FALSE"):
        return f"""Solve the following problem. Think step by step. Use {ld} rules where the question involves code or APIs.

Question:
\"\"\"{variant_text}\"\"\"

Options: {json.dumps(options) if options else "N/A"}

{"Return ONLY the letter label (A, B, C, D, or E)." if forced_type == "MCQ" else "Return exactly 'True' or 'False'."}

OUTPUT JSON:
{{
    "reasoning": "step-by-step logic...",
    "final_answer": "..."
}}"""

    conceptual = contract.mode == "conceptual"
    code_only_invalid = (
        ""
        if conceptual
        else f"- The question requires code in {ld} but the claimed answer is only English with no real source.\n"
    )
    conceptual_note = (
        "NOTE: The question asks for explanation or comparison — prose-only answers are valid. "
        "Do not mark claimed_answer incorrect solely because it is not code.\n\n"
        if conceptual
        else ""
    )

    return f"""You are verifying a practice problem ({ld} where relevant). First judge if the question is valid,
then solve it if it is, and check whether the claimed answer is correct.

Question:
\"\"\"{variant_text}\"\"\"

Claimed correct answer:
\"\"\"{claimed_answer}\"\"\"

{conceptual_note}INVALID QUESTION — set claimed_answer_is_correct to false if ANY apply:
- The question contains template/placeholder phrases (e.g. instructions meant for the author,
  not the student), garbled code, or is incoherent.
- The question does not ask for a clear programming task when it should (e.g. nonsense numbers
  instead of code).
- The claimed answer is not a plausible answer type for the question (e.g. a bare list of
  decimals for a coding problem).
- The claimed answer is a rubric or author note ("The correct answer should...", "The student should...",
  "must be a function that...") instead of the actual code or model prose the student would submit.
{code_only_invalid}- The question stem already contains a complete or nearly complete implementation that matches the claimed
  answer (solution given away to students—often from merged answer keys).

STEPS (if the question is valid):
1. Solve the question yourself. Show your reasoning.
2. Compare your solution to the claimed answer.
3. They do NOT need to be identical — just logically equivalent.
   Ignore variable names, formatting, and minor syntax differences.
   Only mark incorrect if the logic is fundamentally wrong.

OUTPUT JSON:
{{
    "reasoning": "your solution and comparison...",
    "final_answer": "your own answer to the question",
    "claimed_answer_is_correct": true or false
}}"""


def generate_variant(index, db_path=None):
    path = db_path or DB_PATH
    if not path.exists():
        print(f"DB not found at {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)

    questions = db["ingestions"][-1]["questions"]
    if index < 0 or index >= len(questions):
        print(f"Index {index} out of range (0-{len(questions)-1})")
        return None

    q = questions[index]

    if should_skip_question(q.get("text", "")):
        print("Skipping: Detected as non-question (policy/instructions).")
        return None

    use_vision = should_use_vision(q)
    gen_model = TEXT_MODEL
    image_paths = []
    if use_vision and text_model_supports_images() and q.get("image_crops"):
        image_paths = [q["image_crops"][0]]
    elif use_vision and not text_model_supports_images() and DEBUG:
        print(
            "[DEBUG] Vision requested but images disabled for this backend/model; "
            "using question text only."
        )

    forced_type = detect_format(q.get("text", ""))
    algorithm = extract_algorithm(q.get("text", ""))
    expected_mcq_options = _count_options(q.get("text", ""))
    if forced_type == "MCQ" and expected_mcq_options < 2:
        expected_mcq_options = 4
    gen_label = _openrouter_model_id() if _llm_backend() == "openrouter" else gen_model
    contract = build_question_contract(q.get("text", ""))
    print(
        f"Format: {forced_type} | Algorithm: {algorithm} | Mode: {contract.mode} | "
        f"Lang: {contract.language} | Reskin: {contract.allow_thematic_reskin} | Gen Model: {gen_label}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        temperature = BASE_TEMPERATURE + (attempt - 1) * 0.15
        print(f"Attempt {attempt}/{MAX_RETRIES} (temp={temperature:.2f})...")

        scenario = scenario_from_contract(contract)
        label = scenario.get("domain") or scenario.get("theme", "?")
        print(f"  Scenario: {label} ({scenario['style']})")

        # --- Call 1: Generate the variant ---
        gen_prompt = _build_generation_prompt(
            q["text"], forced_type, scenario, algorithm, contract
        )
        variant = call_llm(gen_prompt, gen_model, image_paths, temperature=temperature)
        if not variant or "variant_text" not in variant:
            print("  Generation returned null or missing variant_text.")
            continue

        if forced_type == "MCQ" and _autofix_list_method_mcq(
            variant, q.get("text", ""), contract.language
        ):
            print(f"  List-method MCQ: normalized correct_answer -> {variant.get('correct_answer')}")

        bad = is_invalid_variant(
            variant, forced_type, expected_mcq_options, q.get("text", ""), contract
        )
        if bad:
            print(f"  Invalid variant: {bad}")
            continue

        if DEBUG:
            print(f"[DEBUG] Variant text: {variant.get('variant_text')}")

        sim_thresh = _similarity_threshold_for_original(q["text"])
        if is_too_similar(q["text"], variant["variant_text"]):
            print(f"  Variant too similar to original (>{sim_thresh:.0%}), retrying.")
            continue

        gen_ans = str(variant.get("correct_answer", "")).strip()
        if not gen_ans:
            print("  Generator produced empty correct_answer.")
            continue
        # Belt-and-suspenders after any JSON coercion edge cases
        late_fr = _free_response_correct_answer_invalid(
            gen_ans, variant, contract.language, contract, q.get("text", "")
        )
        if forced_type == "FREE_RESPONSE" and late_fr:
            print(f"  Invalid correct_answer: {late_fr}")
            continue

        # --- Call 2: Verify the variant (solve + judge in one pass) ---
        verify_prompt = _build_verify_prompt(
            variant["variant_text"],
            variant.get("options"),
            forced_type,
            gen_ans,
            contract,
        )

        solution = call_llm(verify_prompt, TEXT_MODEL, None)
        if not solution or "final_answer" not in solution:
            print("  Solver returned null or missing final_answer.")
            continue

        if DEBUG:
            print(f"[DEBUG] Solver answer: {solution.get('final_answer')}")

        verified = False
        if forced_type in ["MCQ", "TRUE_FALSE"]:
            g_val = normalize_answer(gen_ans)
            s_val = normalize_answer(solution["final_answer"])
            if g_val == s_val:
                verified = True
            else:
                print(f"  Mismatch: Generator='{g_val}' vs Solver='{s_val}'")
        else:
            verified = bool(solution.get("claimed_answer_is_correct"))
            if not verified:
                reason = solution.get("reasoning", "no reasoning provided")
                print(f"  Verifier rejected: {reason[:120]}...")

        if verified:
            print(f"  Verified on attempt {attempt}")
            return {
                "original_id": q.get("question_id"),
                "type": forced_type,
                "language": contract.language,
                "question_mode": contract.mode,
                "routing": contract.routing_source,
                "algorithm": algorithm,
                "storyline": variant.get("storyline", ""),
                "task": variant.get("task", ""),
                "constraints": variant.get("constraints", ""),
                "question": variant["variant_text"],
                "options": variant.get("options"),
                "answer": gen_ans,
                "scenario_domain": scenario.get("domain") or scenario.get("theme"),
                "scenario_style": scenario["style"],
            }

    print(f"Failed verification after {MAX_RETRIES} attempts")
    return None


if __name__ == "__main__":
    generate_variant(3)
