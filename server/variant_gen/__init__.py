"""Variant generation pipeline (LLM generate → validate → verify).

Public API is loaded on first access so ``python -m variant_gen.generator`` does not
pre-import the generator via the package __init__.
"""

from typing import Any

__all__ = ["DB_PATH", "generate_variant"]


def __getattr__(name: str) -> Any:
    if name == "DB_PATH":
        from .generator import DB_PATH as _DB_PATH

        return _DB_PATH
    if name == "generate_variant":
        from .generator import generate_variant as _generate_variant

        return _generate_variant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
