"""Resolve which ingestion block to use in the questions database (layout JSON or exam_tests PDFs)."""

from typing import Any, Dict, List, Optional


def resolve_ingestion_index(
    ingestions: List[Dict[str, Any]],
    *,
    exam_id: Optional[str] = None,
    ingestion_index: int = 0,
) -> int:
    """Return a 0-based index into ``ingestions``.

    If ``exam_id`` is set, match ``ing["exam_id"]`` exactly (after strip). If several
    match, the **last** occurrence in file order wins. Otherwise ``ingestion_index``
    uses normal list semantics, including negatives (``-1`` = last).
    """
    if not ingestions:
        raise IndexError("questions.json has no ingestions")

    if exam_id is not None:
        key = exam_id.strip()
        if not key:
            raise ValueError("exam_id must be non-empty when provided")
        hits = [i for i, ing in enumerate(ingestions) if (ing.get("exam_id") or "") == key]
        if not hits:
            raise ValueError(f"no ingestion with exam_id={key!r}")
        return hits[-1]

    n = len(ingestions)
    idx = ingestion_index if ingestion_index >= 0 else n + ingestion_index
    if idx < 0 or idx >= n:
        raise IndexError(
            f"ingestion index {ingestion_index} out of range for {n} ingestion(s)"
        )
    return idx
