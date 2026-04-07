import json
import requests
import base64
import re
import random
from pathlib import Path
from difflib import SequenceMatcher

# config
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "layout_debug" / "questions.json"
OLLAMA_URL = "http://localhost:11434/api/chat"

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
TEXT_MODEL   = "qwen2.5-coder:7b"   # generation + verification (must follow JSON reliably)
VISION_MODEL = "moondream"          # NOT used for JSON generation — too small / wrong modality

DEBUG = False
MAX_RETRIES = 3
BASE_TEMPERATURE = 0.4
SIMILARITY_THRESHOLD = 0.6

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


# ── Scenario domains ─────────────────────────────────────────────────
# Aligned with backlog spec: logistics, finance, healthcare, robotics,
# social networks, cybersecurity, e-commerce — plus extras.
THEMATIC_RESKINS = [
    {
        "theme": "Logistics & Shipping",
        "swap_nouns": "packages, crates, shipments, pallets, delivery trucks, warehouses, weight limits",
        "example": "Two Sum → find two packages whose weights add up to the truck's capacity",
    },
    {
        "theme": "Finance & Banking",
        "swap_nouns": "transactions, accounts, balances, transfers, interest rates, portfolios, deposits",
        "example": "sum → calculate total deposits; search → find a fraudulent transaction",
    },
    {
        "theme": "Healthcare & Patient Records",
        "swap_nouns": "patients, doctors, appointments, medications, dosages, rooms, wait times",
        "example": "priority queue → triage patients by severity; scheduling → assign time slots",
    },
    {
        "theme": "Robotics & Automation",
        "swap_nouns": "robots, sensors, waypoints, motor speeds, task queues, coordinates, battery levels",
        "example": "graph → plan robot's path through waypoints; queue → process sensor readings in order",
    },
    {
        "theme": "Social Network",
        "swap_nouns": "users, friend lists, posts, likes, followers, messages, notifications",
        "example": "graph → mutual friends; set operations → common followers between two users",
    },
    {
        "theme": "Cybersecurity",
        "swap_nouns": "login attempts, IP addresses, access logs, threat scores, firewall rules, permissions, tokens",
        "example": "search → find suspicious IP; filtering → block IPs that exceed failed login threshold",
    },
    {
        "theme": "E-Commerce",
        "swap_nouns": "products, prices, stock counts, shopping carts, discounts, categories, customer orders",
        "example": "filtering → remove out-of-stock items; sum → calculate cart total with discounts",
    },
    {
        "theme": "Recipe & Cooking",
        "swap_nouns": "ingredients, recipes, portions, cooking times, temperatures, pantry items",
        "example": "sorting → sort recipes by prep time; linked list → chain of cooking steps",
    },
    {
        "theme": "Music Playlist",
        "swap_nouns": "songs, playlists, albums, artists, track durations, play counts, genres",
        "example": "search → find a song in a shuffled playlist; sum → total playlist duration",
    },
    {
        "theme": "Sports & Athletics",
        "swap_nouns": "players, teams, scores, game rounds, standings, match results, rankings",
        "example": "max/min → find the top scorer; recursion → tournament bracket elimination",
    },
    {
        "theme": "Travel & Navigation",
        "swap_nouns": "cities, routes, distances, flights, travel times, layovers, fuel costs",
        "example": "graph traversal → find shortest route; DP → cheapest sequence of flights",
    },
    {
        "theme": "File System & Documents",
        "swap_nouns": "files, folders, file sizes, extensions, paths, permissions, timestamps",
        "example": "tree traversal → list all files in nested folders; search → find a file by name",
    },
    {
        "theme": "School & Grades",
        "swap_nouns": "students, courses, grades, assignments, GPAs, semesters, classrooms",
        "example": "dictionary → student grade lookup; sorting → rank students by GPA",
    },
    {
        "theme": "Parking Lot",
        "swap_nouns": "cars, parking spots, license plates, entry times, exit times, fees, levels",
        "example": "stack → last car in first car out; search → find an open spot",
    },
]

SWE_SCENARIOS = [
    {
        "domain": "Web API Development",
        "context": "You are building a REST API backend.",
        "entities": ["endpoints", "HTTP requests", "JSON responses", "users", "products", "orders"],
    },
    {
        "domain": "DevOps & CI/CD",
        "context": "You are maintaining a deployment pipeline.",
        "entities": ["containers", "build stages", "deployment targets", "log entries", "service instances"],
    },
    {
        "domain": "Chat Application",
        "context": "You are developing a real-time messaging platform.",
        "entities": ["messages", "channels", "users", "read receipts", "file attachments", "notifications"],
    },
    {
        "domain": "Testing & QA",
        "context": "You are building a test framework.",
        "entities": ["test cases", "test suites", "mock objects", "coverage reports", "build outputs"],
    },
    {
        "domain": "Package Manager / CLI Tool",
        "context": "You are building a CLI tool that manages project dependencies.",
        "entities": ["packages", "version constraints", "dependency trees", "lock files", "install targets"],
    },
]


def _classify_question(text):
    """Route to reskin vs full SWE based on question content."""
    t = text.lower()
    if any(kw in t for kw in ["write a class", "class named", "inherits", "initializer", "__init__", "__str__"]):
        return "class_design"
    if any(kw in t for kw in ["describe", "explain", "what is the difference", "what are the runtimes"]):
        return "conceptual"
    return "algorithmic"


def pick_scenario(question_text):
    qtype = _classify_question(question_text)
    if qtype == "class_design":
        return {"style": "swe", **random.choice(SWE_SCENARIOS)}
    return {"style": "reskin", **random.choice(THEMATIC_RESKINS)}


def encode_image(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


_ollama_reachable = None

def call_ollama(prompt, model, images=None, temperature=None):
    global _ollama_reachable

    if DEBUG:
        print(f"\n[DEBUG] Calling model: {model}")

    temp = temperature if temperature is not None else BASE_TEMPERATURE
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": images or []}],
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

        return json.loads(content)
    except requests.ConnectionError:
        if _ollama_reachable is None:
            print(f"\n  ERROR: Cannot connect to Ollama at {OLLAMA_URL}")
            print(f"  Is Ollama running?  Try: ollama serve")
            _ollama_reachable = False
        return None
    except requests.Timeout:
        print("  Request timed out (120s).")
        return None
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        print(f"  Ollama returned HTTP {status}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"  Model returned invalid JSON: {e}")
        if DEBUG:
            print(f"[DEBUG] Raw content was: {content}")
        return None
    except Exception as e:
        print(f"  Unexpected error: {type(e).__name__}: {e}")
        return None


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


def is_too_similar(original_text, variant_text):
    ratio = SequenceMatcher(None, original_text.lower(), variant_text.lower()).ratio()
    if DEBUG:
        print(f"[DEBUG] Similarity ratio: {ratio:.2f}")
    return ratio > SIMILARITY_THRESHOLD


_BAD_PLACEHOLDER_FRAGMENTS = (
    "1-2 sentence real-world scenario",
    "the full problem statement combining",
    "the clear problem definition",
    "any constraints or rules carried over",
    "the complete correct answer",
)


def is_invalid_variant(variant, forced_type, expected_mcq_options):
    """Reject models that echo JSON schema instructions or return garbage."""
    vt = (variant.get("variant_text") or "").strip().lower()
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


def _build_generation_prompt(original_text, forced_type, scenario, algorithm):
    style = scenario.get("style", "reskin")

    if style == "swe":
        context_block = f"""SCENARIO TO USE:
- Domain: {scenario["domain"]}
- Setting: {scenario["context"]}
- Use entities like: {", ".join(scenario["entities"])}

Rename classes, methods, and attributes to fit the domain naturally.
For example, a "Book" class might become a "Deployment" class; a "Library" might become a "ServiceRegistry"."""
    else:
        context_block = f"""THEME TO USE: {scenario["theme"]}
- Swap abstract variables and entities for: {scenario["swap_nouns"]}
- Example of this style: {scenario["example"]}

Keep the problem structure almost identical to the original. Just replace the abstract
nouns/numbers with concrete, tangible things from the theme. Think of it like a word-problem
reskin: "find two numbers that sum to target" becomes "find two packages whose weights
add up to the truck's capacity"."""

    format_rules = ""
    if forced_type == "MCQ":
        n_opts = _count_options(original_text) or 4
        format_rules = f"""
FORMAT CONSTRAINTS (MCQ):
- Keep the EXACT same question direction. If the original asks "which IS", ask "which IS".
  If it asks "which is NOT", ask "which is NOT". Do NOT flip it.
- Produce exactly {n_opts} options, labeled with the same scheme as the original.
- The options must test the same KIND of knowledge (e.g., if the original lists Python
  methods, the variant must also list Python methods — just for a different type/context).
- One option must be clearly correct; the distractors should be plausible but wrong."""
    elif forced_type == "TRUE_FALSE":
        format_rules = """
FORMAT CONSTRAINTS (TRUE/FALSE):
- The statement must be clearly true or false with no ambiguity.
- Keep the same truth value as the original if possible."""
    else:
        format_rules = """
FORMAT CONSTRAINTS (FREE RESPONSE):
- If the original asks to write a function, the variant must ask to write a function.
- If the original asks to write a class, the variant must ask to write a class.
- If the original asks for an explanation, the variant must ask for an explanation.
- Preserve the same level of detail expected in the answer."""

    return f"""You are creating a variant of a CS exam question. The variant should feel like a
concrete, real-world scenario — not an abstract math or textbook exercise.

ORIGINAL QUESTION:
\"\"\"{original_text}\"\"\"

CORE ALGORITHM: {algorithm}
This algorithmic structure MUST be preserved in the variant. Do not change what kind of
algorithm is needed to solve it.

{context_block}
{format_rules}

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


def _build_verify_prompt(variant_text, options, forced_type, claimed_answer):
    if forced_type in ("MCQ", "TRUE_FALSE"):
        return f"""Solve the following problem. Think step by step.

Question:
\"\"\"{variant_text}\"\"\"

Options: {json.dumps(options) if options else "N/A"}

{"Return ONLY the letter label (A, B, C, D, or E)." if forced_type == "MCQ" else "Return exactly 'True' or 'False'."}

OUTPUT JSON:
{{
    "reasoning": "step-by-step logic...",
    "final_answer": "..."
}}"""

    return f"""You are verifying a practice problem. First judge if the question is valid,
then solve it if it is, and check whether the claimed answer is correct.

Question:
\"\"\"{variant_text}\"\"\"

Claimed correct answer:
\"\"\"{claimed_answer}\"\"\"

INVALID QUESTION — set claimed_answer_is_correct to false if ANY apply:
- The question contains template/placeholder phrases (e.g. instructions meant for the author,
  not the student), garbled code, or is incoherent.
- The question does not ask for a clear programming task when it should (e.g. nonsense numbers
  instead of code).
- The claimed answer is not a plausible answer type for the question (e.g. a bare list of
  decimals for a coding problem).

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

    print(f"\n--- Starting pipeline for question index {index} ---")

    if should_skip_question(q.get("text", "")):
        print("Skipping: Detected as non-question (policy/instructions).")
        return None

    use_vision = should_use_vision(q)
    gen_model = TEXT_MODEL
    images = []
    if use_vision and text_model_supports_images() and q.get("image_crops"):
        img = encode_image(q["image_crops"][0])
        if img:
            images = [img]
    elif use_vision and not text_model_supports_images() and DEBUG:
        print(
            "[DEBUG] Vision-eligible question but TEXT_MODEL has no vision; "
            "generating from OCR text only."
        )

    forced_type = detect_format(q.get("text", ""))
    algorithm = extract_algorithm(q.get("text", ""))
    expected_mcq_options = _count_options(q.get("text", ""))
    if forced_type == "MCQ" and expected_mcq_options < 2:
        expected_mcq_options = 4
    print(f"Format: {forced_type} | Algorithm: {algorithm} | Gen Model: {gen_model}")

    for attempt in range(1, MAX_RETRIES + 1):
        temperature = BASE_TEMPERATURE + (attempt - 1) * 0.15
        print(f"Attempt {attempt}/{MAX_RETRIES} (temp={temperature:.2f})...")

        scenario = pick_scenario(q.get("text", ""))
        label = scenario.get("domain") or scenario.get("theme", "?")
        print(f"  Scenario: {label} ({scenario['style']})")

        # --- Call 1: Generate the variant ---
        gen_prompt = _build_generation_prompt(q["text"], forced_type, scenario, algorithm)
        variant = call_ollama(gen_prompt, gen_model, images, temperature=temperature)
        if not variant or "variant_text" not in variant:
            print("  Generation returned null or missing variant_text.")
            continue

        bad = is_invalid_variant(variant, forced_type, expected_mcq_options)
        if bad:
            print(f"  Invalid variant: {bad}")
            continue

        if DEBUG:
            print(f"[DEBUG] Variant text: {variant.get('variant_text')}")

        if is_too_similar(q["text"], variant["variant_text"]):
            print(f"  Variant too similar to original (>{SIMILARITY_THRESHOLD:.0%}), retrying.")
            continue

        gen_ans = str(variant.get("correct_answer", "")).strip()
        if not gen_ans:
            print("  Generator produced empty correct_answer.")
            continue

        # --- Call 2: Verify the variant (solve + judge in one pass) ---
        verify_prompt = _build_verify_prompt(
            variant["variant_text"],
            variant.get("options"),
            forced_type,
            gen_ans,
        )

        solution = call_ollama(verify_prompt, TEXT_MODEL)
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
