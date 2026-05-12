"""
Single place to decide *how* we treat an unseen question before generation.

Today routing is mostly rule-based (keywords + light language detection), with an
optional LLM assist:

  1. Keep the QuestionContract field names stable (including ``question_format``).
  2. ``QUESTION_ROUTER=llm`` — see ``question_router.route_stem``: one small JSON call
     for ``question_format`` and ``language`` only; mode and reskin stay rule-derived.
  3. Validators and prompts consume only QuestionContract — not scattered stems.

Adding a new “kind” of question should mean: extend the contract (if needed),
adjust one router, and add a narrow invariant — not new ad hoc branches
throughout variant_gen.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .question_inputs import count_options, detect_format

_SERVER_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_SERVER_DIR / ".env")
load_dotenv()

# ── Scenario domains (consumers: scenario_from_contract only) ────────
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

_CLASS_DESIGN_MARKERS = (
    "write a class",
    "class named",
    "inherits",
    "initializer",
    "__init__",
    "__str__",
)

# Substrings in lowercased text → treat as conceptual (no silly thematic reskin).
_CONCEPTUAL_MARKERS = (
    "describe ",
    "explain ",
    "what is the difference",
    "difference between",
    "what are the runtimes",
    "name at least",
    "how do they differ",
    "why is",
    "why ",
    "what are",
    "define ",
)


def _mentions_cpp_as_required_language(tl: str) -> bool:
    """
    True when the stem is actually about C++ coursework.

    PDFs often say things like "no C++ component whatsoever" on otherwise pure C
    questions — a naive `'c++' in text` substring match misroutes those to cpp.
    """
    if "c++" not in tl:
        return False
    for m in re.finditer(r"c\+\+", tl):
        window = tl[max(0, m.start() - 20) : m.start()]
        # "no C++ …" still contains the letters c++ — skip those hits.
        if re.search(r"\b(no|not|without|instead of|avoid|never|isn't|isnt)\s*$", window):
            continue
        return True
    return False


def infer_programming_language(text: str) -> str:
    """Guess language from raw exam text; validators and prompts follow this."""
    o = os.getenv("QUESTION_LANG", "").strip().lower()
    if o:
        o = {"py": "python", "c++": "cpp", "cxx": "cpp", "cplusplus": "cpp"}.get(o, o)
        if o in ("python", "cpp", "java", "generic"):
            return o
    tl = (text or "").lower()
    t = text or ""
    # Java / AP CS A: snippets often lack import/main; generics and keywords are the tell.
    if (
        re.search(r"\b(import\s+java|java\.util|public\s+static\s+void\s+main)\b", tl)
        or "system.out" in tl
        or "system.in" in tl
        or re.search(r"\bpublic\s+(class|interface)\b", tl)
        or re.search(r"\bimplements\b", tl)
        or re.search(r"\b(class|interface)\s+\w+\s+extends\s+\w+", tl)
        or re.search(
            r"\b(arraylist|linkedlist|hashmap|hashset|treeset|priorityqueue|list|map|set|queue|stack)\s*<",
            tl,
        )
        or " instanceof " in tl
        or re.search(r"\bboolean\s+\w+", tl)
        or "@override" in tl
        or re.search(r"\b(private|protected)\s+(int|boolean|double|char)\s+\w+\s*;", t, re.I)
        or re.search(r"\bString\s+[a-zA-Z_]\w*\s*[;=]", t)
    ):
        return "java"
    # Stanford CS107-style C (vector + Vector* API). Not C++; prompts/validators use "generic".
    if re.search(
        r"\bVector(New|Append|Dispose|Delete|Insert|Replace|Split|Length|Nth)\s*\(",
        t,
    ) and "std::" not in t:
        return "generic"
    # C++ / intro-C course signals (keep broad: many PDFs lack #include in snippets)
    if (
        _mentions_cpp_as_required_language(tl)
        or "cplusplus" in tl
        or "std::" in t
        or "#include" in t
        or "namespace std" in tl
        or "nullptr" in tl
        or "static_cast" in tl
        or "using namespace" in tl
        or re.search(r"\bcout\b", tl)
        or re.search(r"\bcin\b", tl)
        or re.search(r"\bendl\b", tl)
        or re.search(r"\bvector\s*<\s*\w+\s*>", t)
        or re.search(r"\b(int|double|float|long|unsigned|char|bool|void)\s+\w+\s*=\s*[^;\n]+;", t)
    ):
        return "cpp"
    if re.search(r"^\s*def\s+\w+", t, re.MULTILINE) or re.search(
        r"\b(__init__|elif\s+|\bself\.|\bprint\s*\()\b", tl
    ):
        return "python"
    if "python" in tl or "list method" in tl:
        return "python"
    return "python"


def language_display(lang: str) -> str:
    return {
        "python": "Python",
        "cpp": "C++",
        "java": "Java",
        "generic": "the same language as the original",
    }[lang]


def fence_lang(lang: str) -> str:
    return {"python": "python", "cpp": "cpp", "java": "java", "generic": "text"}[lang]


def question_mode(text: str) -> str:
    """High-level routing: OOP exercise vs conceptual vs algorithmic word-problem reskin."""
    t = (text or "").lower()
    if any(kw in t for kw in _CLASS_DESIGN_MARKERS):
        return "class_design"
    if any(m in t for m in _CONCEPTUAL_MARKERS):
        return "conceptual"
    return "algorithmic"


def looks_like_structural_trace_task(text: str) -> bool:
    """
    Generic signal: the student must produce intermediate structure or multi-step traces.

    Used to turn off thematic reskin (parking lots, recipes, …) without naming specific
    assignments or data structures — applies to heaps, trees, sorts, automata traces, etc.
    """
    t = (text or "").lower()
    if any(
        s in t
        for s in (
            "after each",
            "following each",
            "step-by-step",
            "step by step",
            "successive ",
            "intermediate state",
            "show the state",
            "state after",
            "draw the",
            "illustrate each",
            "trace the",
            " record the state",
        )
    ):
        return True
    if "starting from" in t and any(
        w in t
        for w in (
            "insert",
            "delete",
            "deletemin",
            "delete min",
            "remove",
            "operation",
            "performed",
        )
    ):
        return True
    if "after performing" in t and any(
        w in t for w in ("operation", "insertion", "deletion", "step", "removal")
    ):
        return True
    return False


def looks_like_named_function_write_task(text: str) -> bool:
    """
    Specs that pin a function name/signature — reskinning tends to drift types and fail verify.

    Kept generic (any PDF that says "write a function named …"), not tied to one course.
    """
    t = re.sub(r"\s+", " ", (text or "").lower())
    return bool(
        re.search(r"\bwrite\s+a\s+function\s+named\b", t)
        or re.search(r"\bwrite\s+a\s+function\s+called\b", t)
        or re.search(r"\bimplement\s+a\s+function\s+named\b", t)
        or re.search(r"\bdefine\s+a\s+function\s+named\b", t)
    )


def _conceptual_cs_only_scenario() -> dict:
    """Same theme block as conceptual mode — CS surface, no unrelated domains."""
    return {
        "style": "conceptual",
        "theme": "Computer science course (same domain as the original)",
        "swap_nouns": "(do not use — keep CS vocabulary from the original)",
        "example": "Keep questions about sequences as questions about sequences; only tighten prose.",
    }


def expected_mcq_options_for_stem(text: str, question_format: str, language: str) -> int:
    """
    MCQ option count passed to ``is_invalid_variant``. 0 means do not enforce length
    (C++ PDFs: numbered code lines look like A./1. options; ambiguous stems).
    """
    if question_format != "MCQ":
        return 0
    n = count_options(text)
    if n < 2:
        n = 4
    if language == "cpp":
        return 0
    return n


@dataclass
class QuestionContract:
    """Immutable-ish bundle of routing decisions for one source question."""

    language: str
    mode: str
    allow_thematic_reskin: bool
    question_format: str
    expected_mcq_options: int
    routing_source: str = "rules"


def build_question_contract(text: str) -> QuestionContract:
    """
    Rule-based stem router (language, mode, reskin, MCQ vs FR vs TF).

    For optional LLM assist on format/language only, use ``route_stem`` in
    ``question_router.py`` (``QUESTION_ROUTER=llm``).
    """
    lang = infer_programming_language(text)
    mode = question_mode(text)
    allow = (
        mode == "algorithmic"
        and not looks_like_structural_trace_task(text)
        and not looks_like_named_function_write_task(text)
    )
    qf = detect_format(text)
    emcq = expected_mcq_options_for_stem(text, qf, lang)
    return QuestionContract(
        language=lang,
        mode=mode,
        allow_thematic_reskin=allow,
        question_format=qf,
        expected_mcq_options=emcq,
        routing_source="rules",
    )


def scenario_from_contract(contract: QuestionContract) -> dict:
    """Maps contract.mode → the scenario dict expected by _build_generation_prompt."""
    if contract.mode == "class_design":
        return {"style": "swe", **random.choice(SWE_SCENARIOS)}
    if contract.mode == "conceptual":
        return _conceptual_cs_only_scenario()
    if not contract.allow_thematic_reskin:
        # Algorithmic but trace-heavy, named-function spec, etc. — keep CS framing like conceptual.
        return _conceptual_cs_only_scenario()
    return {"style": "reskin", **random.choice(THEMATIC_RESKINS)}


# ── CS vocabulary drift (conceptual FR only) ─────────────────────────

_FR_CS_KEYWORDS = frozenset({
    "python", "sequence", "sequences", "tuple", "tuples", "string", "strings",
    "list", "lists", "dictionary", "dictionaries", "dict", "set", "sets",
    "boolean", "booleans", "recursion", "recursive", "iterable", "iterables",
    "mutable", "immutable", "immutability", "inheritance", "polymorphism",
    "algorithm", "complexity", "array", "arrays", "stack", "stacks", "queue", "queues",
    "tree", "trees", "graph", "graphs", "hash", "pointer", "heap",
    "encapsulation", "namespace", "compiler", "interpreter", "garbage",
})

_FR_CS_SYNONYM_GROUPS = (
    frozenset({
        "sequence", "sequences", "tuple", "tuples", "string", "strings",
        "list", "lists", "iterable", "iterables", "array", "arrays",
    }),
    frozenset({"dictionary", "dictionaries", "dict"}),
    frozenset({"set", "sets"}),
    frozenset({"recursion", "recursive"}),
    frozenset({"boolean", "booleans"}),
)

_ALL_FR_CS_GROUP_TERMS = frozenset().union(*_FR_CS_SYNONYM_GROUPS)


def _fr_cs_keywords_in_text(text: str) -> set:
    tl = (text or "").lower()
    return {k for k in _FR_CS_KEYWORDS if re.search(rf"\b{re.escape(k)}\b", tl)}


def _fr_variant_covers_cs_terms(o_kw: set, v_kw: set) -> bool:
    if o_kw & v_kw:
        return True
    for g in _FR_CS_SYNONYM_GROUPS:
        if o_kw & g and not (v_kw & g):
            return False
    leftover = o_kw - _ALL_FR_CS_GROUP_TERMS
    return not leftover or bool(leftover & v_kw)


def free_response_cs_vocabulary_lost(original_text: str, variant: dict, contract: QuestionContract) -> bool:
    """
    True if variant should be rejected: conceptual question dropped CS terms
    (e.g. sequences → parking lots). Verifier alone often misses this.
    """
    if contract.mode != "conceptual":
        return False
    o_kw = _fr_cs_keywords_in_text(original_text)
    if not o_kw:
        return False
    blob = " ".join(
        str(variant.get(k) or "")
        for k in ("storyline", "task", "variant_text", "constraints")
    )
    v_kw = _fr_cs_keywords_in_text(blob)
    return not _fr_variant_covers_cs_terms(o_kw, v_kw)
