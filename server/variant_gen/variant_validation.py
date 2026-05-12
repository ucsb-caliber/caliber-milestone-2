"""Deterministic checks on model JSON: placeholders, MCQ shape, similarity."""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Optional, Tuple

from .config import DEBUG, SIMILARITY_THRESHOLD, SIMILARITY_THRESHOLD_EXPLANATION
from .policies import get_policy
from .question_contract import (
    QuestionContract,
    fence_lang,
    free_response_cs_vocabulary_lost,
    language_display,
)


def similarity_threshold_for_original(original_text: str) -> float:
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


def is_too_similar(original_text: str, variant_text: str) -> bool:
    thresh = similarity_threshold_for_original(original_text)
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


def original_asks_for_code_submission(original_text: Optional[str]) -> bool:
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
            "complete the following program",
            "complete the following code",
            "write the following function",
            "implement the following",
        )
    )


def _variant_asks_for_code_submission(variant: Dict[str, Any]) -> bool:
    """True if the *variant* text asks the student to submit code (any language)."""
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


def normalize_answer(ans: Any) -> str:
    s = str(ans).strip().upper()
    if s in ["TRUE", "T", "YES"]:
        return "TRUE"
    if s in ["FALSE", "F", "NO"]:
        return "FALSE"
    # Accept any single-letter option label, not just A–E.
    match = re.search(r"(?:^|\s|\.|^OPTION\s)([A-Z1-5])(?:$|\s|\.|[\)])", s)
    if match:
        val = match.group(1)
    else:
        # e.g. "The answer is (B)" or trailing letter after junk / numeric distractors
        m2 = re.search(r"\b([A-Z])\b", s)
        val = m2.group(1) if m2 else s[:1]

    mapping = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    return mapping.get(val, val)


def mcq_correct_option_label(correct_answer: Any, options: Any) -> Tuple[Optional[str], Optional[str]]:
    if not options or not isinstance(options, dict):
        return None, None
    ca = str(correct_answer).strip()
    candidates = {ca, normalize_answer(ca)}
    m = re.match(r"^([A-Z])", ca, re.I)
    if m:
        L = m.group(1).upper()
        candidates.add(L)
        # Only add numeric mapping for the canonical A–E range.
        if "A" <= L <= "E":
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


def _answer_looks_like_code(ca: str, lang: str) -> bool:
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


def _line_looks_like_code_definition(line: str, lang: str) -> bool:
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


def _variant_code_blob_heuristic(variant_text: str, lang: str) -> bool:
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


def free_response_correct_answer_invalid(
    correct_answer: Any,
    variant: Dict[str, Any],
    lang: str,
    contract: QuestionContract,
    original_text: Optional[str],
) -> Optional[str]:
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

    # Only treat as "must submit code" if the *source* asked for it. Reskins often
    # add "write a function…" even for trace-the-output / fill-in questions.
    require_code = _variant_asks_for_code_submission(variant) and original_asks_for_code_submission(
        original_text or ""
    )

    if require_code:
        if not _answer_looks_like_code(ca, lang):
            return f"coding task requires correct_answer with real {language_display(lang)} code"

    return None


def autofix_list_method_mcq(variant: Dict[str, Any], original_text: str, lang: str) -> bool:
    if lang != "python":
        return False
    pol = get_policy("python")
    return pol.list_method_mcq_autofix(
        variant,
        original_text,
        normalize_answer=normalize_answer,
        mcq_correct_option_label=mcq_correct_option_label,
    )


def is_invalid_variant(
    variant: Dict[str, Any],
    forced_type: str,
    expected_mcq_options: int,
    original_text: str,
    contract: QuestionContract,
) -> Optional[str]:
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

    if forced_type == "FREE_RESPONSE":
        fr_err = free_response_correct_answer_invalid(
            ca, variant, lang, contract, original_text
        )
        if fr_err:
            return fr_err
        if free_response_cs_vocabulary_lost(original_text, variant, contract):
            return "conceptual FR drifted off-topic (keep CS terms from the original, not a novelty theme)"
        ca_str = ca if isinstance(ca, str) else str(ca)
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
        # Avoid false positives on legitimate numeric MCQs (decimals, measurements).
        if (
            vals
            and len(vals) >= 4
            and all(re.match(r"^0\.\d+$", v) for v in vals if v)
            and not any(re.search(r"[a-zA-Z]", v) for v in vals if v)
        ):
            return "MCQ options look like garbage probabilities"

        if lang == "python":
            pol = get_policy("python")
            if pol:
                err = pol.validate_list_method_mcq(variant, original_text, vt, mcq_correct_option_label)
                if err:
                    return err

    return None
