"""
Build the questions.json-shaped dict used by variant_gen directly from PDFs in
``<repo>/exam_tests``, without going through the layout pipeline.

Override with env ``VARIANT_GEN_QUESTIONS_JSON=/path/to/file.json`` to load a
fixed JSON file instead (same schema: top-level ``ingestions`` list).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_SERVER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = _SERVER_DIR.parent
EXAM_TESTS_DIR = REPO_ROOT / "exam_tests"

# Line starts a new question when it looks like a problem header, not a code line number or "1. union".
_LIST_METHOD_WORDS = (
    "union|extend|append|pop|sort|reverse|remove|insert|index|count|copy|"
    "clear|keys|values|items|get|setdefault"
)
_QOPEN = (
    r"What|Which|Whom|Whose|Suppose|The following|All of the|All of|After|"
    r"If\s+the|If\s+a|If\s+an|If\s+all|If\s+each|If\s+you|If\s+we|If\s+this|If\s+that|If\s+two|"
    r"If\s+there|If\s+your|If\s+the\s+input|If\s+the\s+following|How|Name|Explain|"
    r"Determine|Consider|Using\s+the|Using\s+a|Using\s+an|Write|Complete|Indicate|Fill|Circle|True|False|List|Show|Define|"
    r"State|Why|Compute|Design|Find|Prove|Provide|Given|Assume|Implement|Each|Every|Mark|Select|"
    r"Match|Identify|Outline|Describe|Compare|Arrange|Order|Evaluate|Calculate|Simplify|Convert|"
    r"Trace|Draw|Sketch|Rank|Sort|For each|For every|By selecting|Consider the"
)
_PROBLEM_HEADER = re.compile(
    r"(?mi)^[ \t]*(?:"
    r"Problem\s+(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|\d{1,3})\s*[:\.]?\s*"
    r"|Question\s+\d+\s*$"
    r"|\d{1,3}\)\s+"
    rf"|\d{{1,3}}\.(?:[ \t]|\n)+(?!(?:{_LIST_METHOD_WORDS})\b)(?=(?:{_QOPEN}|\())"
    r")",
)

# Trim trailing solution sections only when the heading is unambiguous (not "Solutions can be found …").
_BOILERPLATE_TAIL = re.compile(
    r"(?mis)(?:^|\n)\s*(?:"
    r"answer\s*key|works\s+cited|"
    r"solutions?\s+to\s+(?:the\s+)?(?:problems|exercises|questions)|"
    r"detailed\s+solutions"
    r")\b.*\Z"
)


def _pdf_relative_path(pdf: Path) -> str:
    try:
        return str(pdf.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(pdf)


def _stable_question_id(stem: str, index: int, text: str) -> str:
    h = hashlib.sha1(f"{stem}:{index}:{text[:200]}".encode("utf-8")).hexdigest()[:16]
    return f"q_exam_{stem}_{index:03d}_{h}"


def _extract_pdf_text(path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pdfminer.six is required to read exam_tests PDFs. "
            "Install dependencies from the repo requirements.txt."
        ) from e
    raw = extract_text(str(path)) or ""
    # Normalize odd PDF whitespace but keep paragraph breaks
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Page breaks from pdfminer are often \f; they are not newlines for ^ in regex, so
    # question-start patterns like "3.  Consider" would fail to match and multiple items
    # get glued into one blob (bad for MCQ verify).
    raw = raw.replace("\f", "\n")
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def _trim_solutions_block(text: str) -> str:
    m = _BOILERPLATE_TAIL.search(text)
    if m:
        return text[: m.start()].strip()
    return text


def split_pdf_text_into_questions(text: str) -> List[str]:
    """
    Heuristic split of full exam text into standalone question strings.
    Tuned for common CS exam PDFs (Problem N, Question N, 1. / 1) prefixes).
    """
    text = _trim_solutions_block(text)
    if not text:
        return []

    matches = list(_PROBLEM_HEADER.finditer(text))
    if not matches:
        return [text] if len(text) >= 40 else []

    blocks: List[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if len(chunk) >= 40:
            blocks.append(chunk)
    return blocks


def _ingestion_for_pdf(pdf: Path) -> Dict[str, Any]:
    stem = pdf.stem
    full = _extract_pdf_text(pdf)
    chunks = split_pdf_text_into_questions(full)
    questions = []
    for i, chunk in enumerate(chunks):
        questions.append(
            {
                "question_id": _stable_question_id(stem, i, chunk),
                "start_page": None,
                "page_nums": [],
                "text": chunk,
                "text_hash": None,
                "image_crops": [],
                "type": None,
                "metadata": {"source": "exam_tests_pdf", "pdf": _pdf_relative_path(pdf)},
            }
        )
    return {
        "ingestion_id": f"ing_exam_tests_{stem}",
        "created_at": None,
        "source_pdf": _pdf_relative_path(pdf),
        "exam_id": stem,
        "questions": questions,
    }


def build_questions_db_from_exam_tests_dir(
    directory: Optional[Path] = None,
    *,
    pdf_glob: str = "*.pdf",
) -> Dict[str, Any]:
    root = directory or EXAM_TESTS_DIR
    if not root.is_dir():
        return {"schema_version": "1.0", "ingestions": []}

    pdfs = sorted(root.glob(pdf_glob), key=lambda p: p.name.lower())
    ingestions: List[Dict[str, Any]] = []
    for pdf in pdfs:
        if not pdf.is_file():
            continue
        try:
            ingestions.append(_ingestion_for_pdf(pdf))
        except Exception:
            # Skip unreadable PDFs; caller may log
            continue
    return {"schema_version": "1.0", "ingestions": ingestions}


def load_questions_database(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    If ``db_path`` is set, load that JSON file (ignores env / exam_tests PDFs).
    Else if ``VARIANT_GEN_QUESTIONS_JSON`` is set in the environment, load it.
    Otherwise build from ``exam_tests/*.pdf``.
    """
    if db_path is not None:
        data = json.loads(Path(db_path).read_text(encoding="utf-8"))
    else:
        env_path = os.environ.get("VARIANT_GEN_QUESTIONS_JSON", "").strip()
        if env_path:
            data = json.loads(Path(env_path).read_text(encoding="utf-8"))
        else:
            data = build_questions_db_from_exam_tests_dir()
    if not isinstance(data, dict) or "ingestions" not in data:
        raise ValueError("Questions database must be a dict with an 'ingestions' list.")
    return data
