"""
Orchestrates one variant: load question → classify → LLM generate → validate → LLM verify.

Supporting modules:
  ``config`` — paths, model names, thresholds.
  ``question_inputs`` — skip/vision/format and coarse algorithm tag from raw text.
  ``prompts`` — generation and verification prompt strings.
  ``variant_validation`` — deterministic JSON checks and answer normalization.
  ``llm_client`` — OpenRouter JSON calls.
  ``exam_tests_questions`` — default question DB from PDFs (or env override).
"""

from pathlib import Path

from .config import (
    BASE_TEMPERATURE,
    DEBUG,
    MAX_RETRIES,
    generation_source_max_chars,
    openrouter_timeout_verify,
    resolved_openrouter_model,
    telemetry_enabled,
)
from .exam_tests_questions import load_questions_database
from .llm_client import call_llm, text_model_supports_images
from .prompts import build_generation_prompt, build_verify_prompt
from .question_contract import scenario_from_contract
from .question_inputs import (
    extract_algorithm,
    should_skip_question,
    should_use_vision,
)
from .question_router import route_stem, telemetry_outcome_line, telemetry_routing_line
from .variant_validation import (
    autofix_list_method_mcq,
    free_response_correct_answer_invalid,
    is_invalid_variant,
    is_too_similar,
    normalize_answer,
    similarity_threshold_for_original,
)


def _clip_text_for_generation(text: str, limit: int) -> str:
    t = text or ""
    if len(t) <= limit:
        return t
    head = (limit * 2) // 3
    tail = limit - head - 80
    if tail < 2000:
        tail = 2000
        head = limit - tail - 80
    return t[:head] + "\n\n[... source truncated ...]\n\n" + t[-tail:]


def generate_variant(index, db_path=None, ingestion_index=-1, questions_db=None):
    """
    ``db_path``: optional path to a questions.json-shaped file (layout output or export).
    If omitted, questions come from ``exam_tests/*.pdf`` (or ``VARIANT_GEN_QUESTIONS_JSON``).
    ``questions_db``: optional pre-loaded dict (same schema); avoids re-parsing in batch loops.
    """
    if questions_db is not None:
        db = questions_db
    elif db_path is not None:
        db = load_questions_database(Path(db_path))
    else:
        db = load_questions_database(None)

    ingestions = db["ingestions"]
    if not ingestions:
        print("No ingestions (add PDFs under exam_tests/ or set VARIANT_GEN_QUESTIONS_JSON).")
        return None
    try:
        ing = ingestions[ingestion_index]
    except IndexError:
        print(
            f"Ingestion index {ingestion_index} out of range "
            f"(0-{len(ingestions) - 1}, or negative for from end)"
        )
        return None

    questions = ing["questions"]
    if index < 0 or index >= len(questions):
        print(f"Index {index} out of range (0-{len(questions)-1})")
        return None

    q = questions[index]

    if should_skip_question(q.get("text", "")):
        print("Skipping: Detected as non-question (policy/instructions).")
        return None

    use_vision = should_use_vision(q)
    image_paths = []
    if use_vision and text_model_supports_images() and q.get("image_crops"):
        image_paths = [q["image_crops"][0]]
    elif use_vision and not text_model_supports_images() and DEBUG:
        print(
            "[DEBUG] Vision requested but OPENROUTER_VISION / OPENROUTER_SEND_IMAGES disabled; "
            "using question text only."
        )

    algorithm = extract_algorithm(q.get("text", ""))
    contract = route_stem(q.get("text", "") or "")
    forced_type = contract.question_format
    expected_mcq_options = contract.expected_mcq_options
    gen_label = resolved_openrouter_model()
    print(
        f"Format: {forced_type} | Algorithm: {algorithm} | Mode: {contract.mode} | "
        f"Lang: {contract.language} | Reskin: {contract.allow_thematic_reskin} | "
        f"Route: {contract.routing_source} | Gen Model: {gen_label}"
    )
    qid = q.get("question_id") or ""
    if telemetry_enabled():
        print(telemetry_routing_line(qid, contract), flush=True)

    for attempt in range(1, MAX_RETRIES + 1):
        temperature = BASE_TEMPERATURE + (attempt - 1) * 0.15
        print(f"Attempt {attempt}/{MAX_RETRIES} (temp={temperature:.2f})...")

        scenario = scenario_from_contract(contract)
        label = scenario.get("domain") or scenario.get("theme", "?")
        print(f"  Scenario: {label} ({scenario['style']})")

        gen_prompt = build_generation_prompt(
            _clip_text_for_generation(q.get("text", "") or "", generation_source_max_chars()),
            forced_type,
            scenario,
            algorithm,
            contract,
        )
        variant = call_llm(gen_prompt, image_paths=image_paths, temperature=temperature)
        if not variant or "variant_text" not in variant:
            print("  Generation returned null or missing variant_text.")
            continue

        if forced_type == "MCQ" and autofix_list_method_mcq(
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

        sim_thresh = similarity_threshold_for_original(q["text"])
        if is_too_similar(q["text"], variant["variant_text"]):
            print(f"  Variant too similar to original (>{sim_thresh:.0%}), retrying.")
            continue

        gen_ans = str(variant.get("correct_answer", "")).strip()
        if not gen_ans:
            print("  Generator produced empty correct_answer.")
            continue
        late_fr = free_response_correct_answer_invalid(
            gen_ans, variant, contract.language, contract, q.get("text", "")
        )
        if forced_type in ("FREE_RESPONSE", "CODING") and late_fr:
            print(f"  Invalid correct_answer: {late_fr}")
            continue

        verify_prompt = build_verify_prompt(
            variant["variant_text"],
            variant.get("options"),
            forced_type,
            gen_ans,
            contract,
        )

        solution = call_llm(
            verify_prompt,
            image_paths=None,
            temperature=temperature,
            timeout_sec=openrouter_timeout_verify(),
        )
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
            if telemetry_enabled():
                print(telemetry_outcome_line(qid, "verified", f"attempt={attempt}"), flush=True)
            return {
                "original_id": q.get("question_id"),
                "source_ingestion_id": ing.get("ingestion_id"),
                "source_ingestion_index": (
                    ingestion_index
                    if ingestion_index >= 0
                    else len(ingestions) + ingestion_index
                ),
                "type": forced_type,
                "autograde_kind": contract.autograde_kind,
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
    if telemetry_enabled():
        print(telemetry_outcome_line(qid, "failed_all_retries", ""), flush=True)
    return None


if __name__ == "__main__":
    generate_variant(3)
