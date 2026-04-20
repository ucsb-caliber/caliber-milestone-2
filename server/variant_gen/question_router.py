"""
Stem routing: one entry point for format + QuestionContract.

``QUESTION_ROUTER=rules`` (default): same behavior as ``build_question_contract`` — all fields
from heuristics.

``QUESTION_ROUTER=llm``: one small JSON call classifies ``question_format`` and ``language`` only;
``mode`` and ``allow_thematic_reskin`` stay rule-derived (stable prompts / reskin safety). On LLM
failure, falls back to full rules and ``routing_source=rules``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .config import (
    generation_source_max_chars,
    question_router_name,
    question_router_timeout_sec,
    resolved_question_router_model,
)
from .llm_client import call_llm
from .question_contract import (
    QuestionContract,
    build_question_contract,
    expected_mcq_options_for_stem,
)

_ALLOWED_FORMATS = frozenset({"MCQ", "FREE_RESPONSE", "TRUE_FALSE"})
_ALLOWED_LANGS = frozenset({"python", "cpp", "java", "generic"})


def _clip_stem(text: str, limit: int) -> str:
    t = text or ""
    if len(t) <= limit:
        return t
    head = (limit * 2) // 3
    tail = limit - head - 80
    if tail < 800:
        tail = 800
        head = max(400, limit - tail - 80)
    return t[:head] + "\n\n[... truncated ...]\n\n" + t[-tail:]


def _normalize_format(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "MULTIPLE_CHOICE": "MCQ",
        "MULTIPLECHOICE": "MCQ",
        "CHOICE": "MCQ",
        "FREE_RESPONSE": "FREE_RESPONSE",
        "FREERESPONSE": "FREE_RESPONSE",
        "FR": "FREE_RESPONSE",
        "FREE": "FREE_RESPONSE",
        "TRUE_FALSE": "TRUE_FALSE",
        "TRUEFALSE": "TRUE_FALSE",
        "T_F": "TRUE_FALSE",
        "TF": "TRUE_FALSE",
    }
    s = aliases.get(s, s)
    return s if s in _ALLOWED_FORMATS else None


def _normalize_lang(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    s = {"py": "python", "c++": "cpp", "cplusplus": "cpp", "cxx": "cpp"}.get(s, s)
    return s if s in _ALLOWED_LANGS else None


def _llm_router_prompt(stem: str) -> str:
    return (
        "You classify an exam question stem for an automated variant generator.\n"
        "Return ONLY one JSON object (no markdown) with exactly these keys:\n"
        '  "question_format": one of MCQ, FREE_RESPONSE, TRUE_FALSE\n'
        '  "language": one of python, cpp, java, generic\n'
        "Rules:\n"
        "- MCQ: student picks labeled options (A–E, 1–5, or (a)–(e)), including Roman-numeral clauses "
        "with lettered answer choices.\n"
        "- TRUE_FALSE: explicit true/false style.\n"
        "- FREE_RESPONSE: written answer, traces, code to write, proofs, etc.\n"
        "- generic: stem is clearly C, Scheme, mixed course-specific code, or language is ambiguous; "
        "do not default to python unless the stem is actually Python.\n"
        "- java: AP-style or explicit Java syntax (class/interface, public static void, etc.).\n"
        "- cpp: C++ is the tested language (classes, std::, etc.); not for stems that only say "
        '"no C++" or are pure C with a C API.\n'
        "Stem:\n---\n"
        f"{stem}\n---\n"
    )


def _try_llm_route_stem(text: str) -> Optional[QuestionContract]:
    """LLM supplies format + language; mode and reskin from rules."""
    base = build_question_contract(text)
    stem = _clip_stem(text, generation_source_max_chars())
    prompt = _llm_router_prompt(stem)
    model = resolved_question_router_model()
    timeout = question_router_timeout_sec()
    raw = call_llm(prompt, image_paths=None, temperature=0.0, model=model, timeout_sec=timeout)
    if not raw or not isinstance(raw, dict):
        return None
    fmt = _normalize_format(raw.get("question_format"))
    lang = _normalize_lang(raw.get("language"))
    if fmt is None or lang is None:
        return None
    return QuestionContract(
        language=lang,
        mode=base.mode,
        allow_thematic_reskin=base.allow_thematic_reskin,
        question_format=fmt,
        expected_mcq_options=expected_mcq_options_for_stem(text, fmt, lang),
        routing_source="llm",
    )


def route_stem(text: str) -> QuestionContract:
    """
    Single entry point for ``question_format`` + contract fields.

    Env:
        QUESTION_ROUTER — ``rules`` (default) or ``llm``
        QUESTION_ROUTER_MODEL — optional OpenRouter slug (defaults to OPENROUTER_MODEL)
        QUESTION_ROUTER_TIMEOUT — HTTP read timeout seconds (default 25)
    """
    name = question_router_name()
    if name != "llm":
        return build_question_contract(text)
    merged = _try_llm_route_stem(text)
    if merged is not None:
        return merged
    print("  Question router: LLM route failed or invalid JSON; using rules.")
    return build_question_contract(text)


def telemetry_routing_line(question_id: str, contract: QuestionContract) -> str:
    """One JSON line for log aggregation (stderr or grep)."""
    payload = {
        "event": "stem_routed",
        "question_id": question_id,
        "routing_source": contract.routing_source,
        "question_format": contract.question_format,
        "language": contract.language,
        "mode": contract.mode,
        "allow_thematic_reskin": contract.allow_thematic_reskin,
    }
    return f"[variant_gen:telemetry] {json.dumps(payload, ensure_ascii=False)}"


def telemetry_outcome_line(question_id: str, outcome: str, detail: str = "") -> str:
    """Outcome after all retries (success not emitted here — batch runner already logs)."""
    payload = {
        "event": "variant_outcome",
        "question_id": question_id,
        "outcome": outcome,
        "detail": (detail or "")[:500],
    }
    return f"[variant_gen:telemetry] {json.dumps(payload, ensure_ascii=False)}"
