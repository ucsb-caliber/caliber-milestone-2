"""LLM system/user strings for variant generation and verification."""

import json
from typing import Any, Dict, Optional

from .config import verify_variant_text_max_chars
from .question_contract import (
    QuestionContract,
    fence_lang,
    language_display,
    looks_like_named_function_write_task,
)
from .question_inputs import count_options
from .variant_validation import original_asks_for_code_submission


def _verify_output_only_hint(variant_text: str) -> str:
    """Extra verify instructions when the stem is clearly 'what is the output' style."""
    t = (variant_text or "").lower()
    if not any(
        p in t
        for p in (
            "what is the output",
            "what is the exact output",
            "exact output of",
            "output when the following",
            "output when the code",
            "output of the following",
            "output of the program",
            "when the following code",
            "when the following program",
            "when executed",
            "when run",
            "indicate the output",
        )
    ):
        return ""
    return """OUTPUT-ONLY: If the question asks only for printed or literal program output (not writing new code),
compare claimed_answer to your computed result. Set claimed_answer_is_correct true if they match in substance
after trimming leading and trailing whitespace. Minor formatting (extra spaces or newlines) is fine.
Do not mark false because claimed_answer is brief. If values, line order, or shown literals disagree, mark false.

"""


def _truncate_for_verify_block(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    head = max(2000, max_chars // 2 - 60)
    tail = max_chars - head - 140
    if tail < 800:
        tail = 800
        head = max(2000, max_chars - tail - 140)
    return (
        s[:head]
        + f"\n\n[... omitted {len(s) - head - tail} characters from middle of question ...]\n\n"
        + s[-tail:]
    )


def prompt_extras(original_text: str, forced_type: str, style: str, contract: QuestionContract) -> str:
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
            if not original_asks_for_code_submission(original_text or ""):
                chunks.append(
                    "The original asks for explanation or comparison only — do NOT turn it into a coding exercise "
                    "(e.g. do not add 'write a function' or require implementation) unless the source explicitly "
                    "asked for code."
                )
        elif contract.mode == "algorithmic" and not contract.allow_thematic_reskin:
            chunks.append(
                "MULTI-STEP / STRUCTURE TASK: Stay in normal CS and data-structure vocabulary. "
                "Do not wrap the problem in an unrelated real-world metaphor. "
                "Preserve every requirement to show intermediate states, diagrams, or step-by-step structure "
                "as clearly as in the original (same level of detail)."
            )
        if looks_like_named_function_write_task(original_text or ""):
            chunks.append(
                f"NAMED FUNCTION: Keep the required function name and parameter types as in the prompt. "
                f"correct_answer must be complete {language_display(lang)} that satisfies the spec—no language mix-ups."
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


def build_generation_prompt(
    original_text: str,
    forced_type: str,
    scenario: Dict[str, Any],
    algorithm: str,
    contract: QuestionContract,
) -> str:
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

    extras = prompt_extras(original_text, forced_type, style, contract)

    format_rules = ""
    if forced_type == "MCQ":
        n_opts = count_options(original_text) or 4
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
9. Keep variant_text and options reasonably compact (avoid repeating the same code block many times);
   downstream verification must fit in a single JSON object.

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


def build_verify_prompt(
    variant_text: str,
    options: Optional[Dict[str, Any]],
    forced_type: str,
    claimed_answer: str,
    contract: QuestionContract,
) -> str:
    ld = language_display(contract.language)
    vlim = verify_variant_text_max_chars()
    vt = _truncate_for_verify_block(variant_text, vlim)
    trunc_note = (
        "\nNOTE: Question text above may be truncated for size; solve from what is shown.\n"
        if len(variant_text or "") > vlim
        else ""
    )
    if forced_type in ("MCQ", "TRUE_FALSE"):
        # Avoid hard-coding A–E: some sources legitimately use more labels.
        allowed = None
        if forced_type == "MCQ" and options and isinstance(options, dict):
            labels = [str(k).strip() for k in options.keys() if str(k).strip()]
            # Keep single-token labels (A, B, 1, 2, etc.) and stable ordering.
            labels = [l for l in labels if len(l) <= 2]
            if labels:
                allowed = ", ".join(labels)
        return f"""Solve the following problem. Think step by step. Use {ld} rules where the question involves code or APIs.
Keep "reasoning" under ~1200 characters so the reply stays valid JSON (no huge unescaped strings).

Question:
\"\"\"{vt}\"\"\"
{trunc_note}
Options: {json.dumps(options) if options else "N/A"}

{"Return ONLY one option label from: " + allowed + "." if (forced_type == "MCQ" and allowed) else ("Return ONLY the letter/number option label." if forced_type == "MCQ" else "Return exactly 'True' or 'False'.")}

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
    output_only = _verify_output_only_hint(vt)

    return f"""You are verifying a practice problem ({ld} where relevant). First judge if the question is valid,
then solve it if it is, and check whether the claimed answer is correct.
Keep "reasoning" under ~2000 characters so the reply stays valid JSON.

Question:
\"\"\"{vt}\"\"\"
{trunc_note}
Claimed correct answer:
\"\"\"{claimed_answer}\"\"\"

{conceptual_note}{output_only}INVALID QUESTION — set claimed_answer_is_correct to false if ANY apply:
- The question contains template/placeholder phrases (e.g. instructions meant for the author,
  not the student), garbled code, or is incoherent.
- The question does not ask for a clear programming task when it should (e.g. nonsense numbers
  instead of code).
- The claimed answer is not a plausible answer type for the question (e.g. a bare list of
  decimals for a coding problem).
- The claimed answer is a rubric or author note ("The correct answer should...", "The student should...",
  "must be a function that...") instead of the actual code or model prose the student would submit.
{code_only_invalid}
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
