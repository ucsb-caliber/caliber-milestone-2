"""Variant generation pipeline (LLM generate → validate → verify).

Public API is loaded on first access so ``python -m variant_gen.generator`` does not
pre-import the generator via the package __init__.
"""

from typing import Any

__all__ = ["DB_PATH", "EXAM_TESTS_DIR", "generate_variant", "load_questions_database"]


def __getattr__(name: str) -> Any:
    if name == "DB_PATH":
        from .config import DB_PATH as _DB_PATH

        return _DB_PATH
    if name == "EXAM_TESTS_DIR":
        from .exam_tests_questions import EXAM_TESTS_DIR as _EXAM_TESTS_DIR

        return _EXAM_TESTS_DIR
    if name == "load_questions_database":
        from .exam_tests_questions import load_questions_database as _load

        return _load
    if name == "generate_variant":
        from .generator import generate_variant as _generate_variant

        return _generate_variant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
