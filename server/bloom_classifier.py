"""
Bloom's Taxonomy classifier using the Bloominzer model (Hugging Face).
Use this for a fast, library-based implementation instead of building from scratch.

Model: https://huggingface.co/uw-vta/bloominzer-0.1
Labels (original): Knowledge, Comprehension, Application, Analysis, Synthesis, Evaluation
We map to revised taxonomy: Remember, Understand, Apply, Analyze, Evaluate, Create.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Map Bloominzer (original Bloom) labels to revised taxonomy
ORIGINAL_TO_REVISED: Dict[str, str] = {
    "Knowledge": "Remember",
    "Comprehension": "Understand",
    "Application": "Apply",
    "Analysis": "Analyze",
    "Synthesis": "Create",
    "Evaluation": "Evaluate",
}

# Short template-based reasoning per level (for interpretability without an LLM)
REASONING_TEMPLATES: Dict[str, str] = {
    "Remember": "This question asks for recall of facts, terms, or basic concepts.",
    "Understand": "This question requires explaining ideas or concepts in one's own words.",
    "Apply": "This question requires using information or a procedure in a new situation.",
    "Analyze": "This question requires breaking material into parts and seeing how they relate.",
    "Evaluate": "This question requires making a judgment based on criteria and standards.",
    "Create": "This question requires putting elements together to form a new whole or pattern.",
}

_PIPELINE: Any = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from transformers import pipeline
        _PIPELINE = pipeline(
            "text-classification",
            model="uw-vta/bloominzer-0.1",
        )
    return _PIPELINE


def classify(question_text: str) -> Dict[str, Any]:
    """
    Classify a question into one Bloom's Taxonomy level (revised).
    Returns level, confidence, and a short reasoning string.
    """
    text = (question_text or "").strip()
    if not text:
        return {
            "bloom_level": None,
            "bloom_confidence": 0.0,
            "bloom_reasoning": "No question text provided.",
        }

    pipe = _get_pipeline()
    result = pipe(text, top_k=1)
    if not result:
        return {
            "bloom_level": None,
            "bloom_confidence": 0.0,
            "bloom_reasoning": "Model returned no prediction.",
        }

    item = result[0]
    original_label = item.get("label", "")
    score = float(item.get("score", 0.0))
    revised_level = ORIGINAL_TO_REVISED.get(original_label, original_label)
    reasoning = REASONING_TEMPLATES.get(revised_level, f"Classified as {revised_level}.")

    return {
        "bloom_level": revised_level,
        "bloom_confidence": score,
        "bloom_reasoning": reasoning,
    }


def classify_batch(question_texts: List[str]) -> List[Dict[str, Any]]:
    """Classify multiple questions. May be more efficient than calling classify() in a loop."""
    return [classify(t) for t in question_texts]


if __name__ == "__main__":
    # Quick test
    out = classify("What is the time complexity of Dijkstra's algorithm?")
    print(out)
