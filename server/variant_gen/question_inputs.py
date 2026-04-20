"""Classify raw exam text: skip rules, vision, format, coarse algorithm tag."""

import re
from typing import Any, Dict

from .config import _ALGORITHM_PATTERNS, openrouter_vision_enabled


def extract_algorithm(text: str) -> str:
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


def should_skip_question(text: str) -> bool:
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


def _is_code_or_written_answer_question(text: str) -> bool:
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


def should_use_vision(q_data: Dict[str, Any]) -> bool:
    if not openrouter_vision_enabled():
        return False

    text = q_data.get("text", "")
    images = q_data.get("image_crops", [])
    if not images:
        return False

    if _is_code_or_written_answer_question(text):
        return False

    vision_keywords = [
        "shown",
        "diagram",
        "figure",
        "graph",
        "tree",
        "circuit",
        "table",
        "plot",
        "chart",
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


def _looks_like_numbered_written_subproblems(text: str) -> bool:
    """Multi-part exam prompts use '1. Foo 2. Bar' like MCQ option lines — treat as written, not MCQ."""
    t = (text or "").lower()
    numbered = len(re.findall(r"(?:^|\n)\s*\d+\.\s+\S", text or ""))
    if numbered < 2:
        return False
    if "for each of the following" in t:
        return True
    if ("worst-case runtime" in t or "worst case runtime" in t) and (
        "explanation" in t or "brief" in t or "sentence" in t
    ):
        return True
    if "give a" in t and "runtime" in t and ("explanation" in t or "brief" in t):
        return True
    return False


def detect_format(text: str) -> str:
    text_lower = text.lower()
    if "true" in text_lower and "false" in text_lower:
        if len(text) < 200 or "select" in text_lower:
            return "TRUE_FALSE"
    # Coding / written tasks often include numbered lines or lettered bullets; classify before MCQ heuristics.
    if re.search(
        r"\b(?:write\s+pseudocode|write\s+a\s+(?:function|class|method)|recursive\s+function)\b",
        text_lower,
    ):
        return "FREE_RESPONSE"
    has_mcq = re.search(r"(?:^|\n|\s)(?:[A-E]|[1-5])[\.\)]\s+\w+", text)
    if has_mcq and _looks_like_numbered_written_subproblems(text):
        return "FREE_RESPONSE"
    if has_mcq:
        return "MCQ"
    # Scanned exams: "o  A   option text" or letter on its own line without \w+ immediately after
    letter_marks = len(re.findall(r"(?:^|\n|\s)[A-E][\.\)]\s*", text, flags=re.IGNORECASE))
    # "o  A   option" (circle / scan noise; letter not always followed by .)
    o_letter_opts = len(re.findall(r"(?:^|\n|\s)o\s+([A-E])\b", text, flags=re.IGNORECASE))
    # AP / textbook style "(a)  I only" through "(e)  ..." — not matched by bare "A." above.
    paren_letter_opts = len(re.findall(r"(?:^|\n|\s)\(\s*([A-E])\s*\)", text, flags=re.IGNORECASE))
    mcq_phrases = (
        "which of the following",
        "which of these",
        "which one of the following",
        "select all",
        "choose the",
        "all of the following are",
    )
    if (
        letter_marks >= 2 or o_letter_opts >= 2 or paren_letter_opts >= 2
    ) and any(k in text_lower for k in mcq_phrases):
        return "MCQ"
    return "FREE_RESPONSE"


def count_options(text: str) -> int:
    # Only treat explicit "A)" / "A." / "1)" / "1." patterns as options.
    # Tree/traversal dumps can look like dozens of fake options — cap and bail to 0.
    letter_opts = re.findall(r"(?:^|\n)\s*[A-E][\.\)]\s+\S", text)
    num_opts = re.findall(r"(?:^|\n)\s*[1-5][\.\)]\s+\S", text)
    count = max(len(letter_opts), len(num_opts))
    if count < 2:
        return 0
    if count > 10:
        return 0
    return count
