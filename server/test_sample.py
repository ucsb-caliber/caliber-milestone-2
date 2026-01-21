import os
import re
import json
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import layoutparser as lp
from pdf2image import convert_from_path

import numpy as np
import cv2
import pytesseract

import matplotlib.pyplot as plt


# ===================== CONFIG =====================

PDF_PATH = "exam_tests/practicefinal3.pdf"
OUTPUT_DIR = "layout_debug"

START_PAGE = 1
END_PAGE = 10

Y_TOL = 25            # reading-order tolerance

SAVE_CROPS = True
SHOW_CROPS = True     # set False if running headless (no GUI)
CROP_PADDING = 10     # pixels of padding around bbox

QUESTIONS_DB_FILENAME = "questions.json"  # stored inside OUTPUT_DIR


# ===================== QUESTION DETECTION =====================

QUESTION_START_PATTERNS = [
    r"^\s*Problem\s+\d+\b",
    r"^\s*Question\s+\d+\b",
    r"^\s*Q\s*\d+\b",
]

QUESTION_START_RE = re.compile("|".join(QUESTION_START_PATTERNS), re.IGNORECASE)

NUMBERED_ITEM_SPLIT_RE = re.compile(r"(?=\n?\s*\d+\s*[\.\)])")


def is_question_start(text: str) -> bool:
    return bool(QUESTION_START_RE.match(text.strip()))


# ===================== DATA STRUCTURES =====================

@dataclass
class Block:
    page: int
    bbox: Tuple[int, int, int, int]
    text: str


@dataclass
class Question:
    start_page: int
    blocks: List[Block] = field(default_factory=list)
    text_units: List[str] = field(default_factory=list)

    question_id: str = ""
    qtype: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    image_crops: List[str] = field(default_factory=list)

    def add_block(self, block: Block):
        self.blocks.append(block)
        self.text_units.append(block.text)

    @property
    def text(self) -> str:
        return "\n".join(self.text_units).strip()

    def bboxes_by_page(self) -> Dict[int, Tuple[int, int, int, int]]:
        by_page: Dict[int, List[Tuple[int, int, int, int]]] = {}
        for b in self.blocks:
            by_page.setdefault(b.page, []).append(b.bbox)

        out: Dict[int, Tuple[int, int, int, int]] = {}
        for p, bboxes in by_page.items():
            xs1 = [bb[0] for bb in bboxes]
            ys1 = [bb[1] for bb in bboxes]
            xs2 = [bb[2] for bb in bboxes]
            ys2 = [bb[3] for bb in bboxes]
            out[p] = (min(xs1), min(ys1), max(xs2), max(ys2))
        return out

    def page_nums(self) -> List[int]:
        return sorted({b.page for b in self.blocks})


# ===================== HELPERS =====================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_exam_id(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "exam_unknown"
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s or "exam_unknown"


def stable_hash16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_ingestion_id(created_at: str, source_pdf: str, exam_id: str) -> str:
    seed = f"{created_at}::{source_pdf}::{exam_id}"
    return "ing_" + stable_hash16(seed)


def make_question_id(exam_id: str, ingestion_id: str, question_index: int) -> str:
    seed = f"{exam_id}::{ingestion_id}::q{question_index}"
    return "q_" + stable_hash16(seed)


def atomic_write_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_questions_db(db_path: Path) -> Dict[str, Any]:
    if not db_path.exists():
        return {"schema_version": "1.0", "ingestions": []}

    try:
        raw = db_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"schema_version": "1.0", "ingestions": []}
        if "ingestions" not in data or not isinstance(data["ingestions"], list):
            data["ingestions"] = []
        if "schema_version" not in data:
            data["schema_version"] = "1.0"
        return data
    except Exception:
        # If corrupted, start fresh but preserve the broken file
        backup = db_path.with_suffix(db_path.suffix + f".bak.{uuid.uuid4().hex}")
        try:
            db_path.replace(backup)
        except Exception:
            pass
        return {"schema_version": "1.0", "ingestions": []}


# ===================== IMAGE + OCR =====================

def pil_to_bgr_np(pil_img) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def ocr_crop(bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox
    h, w = bgr.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    text = pytesseract.image_to_string(gray, config="--oem 3 --psm 6")
    return text.replace("\x0c", "").strip()


# ===================== READING ORDER =====================

def sort_blocks_reading_order(blocks: List[Block], y_tol: int) -> List[Block]:
    blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    rows: List[List[Block]] = []

    for b in blocks:
        placed = False
        for row in rows:
            if abs(b.bbox[1] - row[0].bbox[1]) <= y_tol:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    ordered: List[Block] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b.bbox[0]))

    return ordered


# ===================== TEXT NORMALIZATION =====================

def split_block_text(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in NUMBERED_ITEM_SPLIT_RE.split(text) if p.strip()]
    return parts if parts else [text.strip()]


# ===================== MAIN PARSER =====================

def parse_pdf_to_questions(pdf_path: Path) -> List[Question]:
    pages = convert_from_path(str(pdf_path))
    last_page = len(pages)

    start = max(1, START_PAGE)
    end = END_PAGE or last_page
    end = min(end, last_page)

    model = lp.AutoLayoutModel("lp://efficientdet/PubLayNet/tf_efficientdet_d1")

    all_questions: List[Question] = []
    current_question: Optional[Question] = None

    for page_idx in range(start - 1, end):
        page_num = page_idx + 1
        page_img = pages[page_idx]
        bgr = pil_to_bgr_np(page_img)

        layout = model.detect(page_img)

        page_blocks: List[Block] = []

        for b in layout:
            if b.type not in ("Text", "Title", "List"):
                continue

            x1, y1, x2, y2 = map(int, b.block.coordinates)
            text = ocr_crop(bgr, (x1, y1, x2, y2))
            if not text:
                continue

            for unit in split_block_text(text):
                page_blocks.append(Block(
                    page=page_num,
                    bbox=(x1, y1, x2, y2),
                    text=unit
                ))

        page_blocks = sort_blocks_reading_order(page_blocks, Y_TOL)

        for block in page_blocks:
            if is_question_start(block.text):
                if current_question is not None:
                    all_questions.append(current_question)
                current_question = Question(start_page=block.page)
                current_question.add_block(block)
            else:
                if current_question is not None:
                    current_question.add_block(block)

    if current_question is not None:
        all_questions.append(current_question)

    return all_questions


# ===================== CROPPING / DISPLAY =====================

def clamp_bbox(bbox: Tuple[int, int, int, int], w: int, h: int, pad: int = 0) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0, 0)
    return (x1, y1, x2, y2)


def crop_and_output_questions(pdf_path: Path, questions: List[Question], crops_dir: Path):
    pages = convert_from_path(str(pdf_path))
    crops_dir.mkdir(parents=True, exist_ok=True)

    for qi, q in enumerate(questions, 1):
        per_page = q.bboxes_by_page()

        for page_num, bbox in sorted(per_page.items()):
            page_img = pages[page_num - 1]  # PIL
            w, h = page_img.size

            x1, y1, x2, y2 = clamp_bbox(bbox, w, h, pad=CROP_PADDING)
            if (x1, y1, x2, y2) == (0, 0, 0, 0):
                continue

            crop = page_img.crop((x1, y1, x2, y2))

            if SAVE_CROPS:
                # Use question_id to avoid collisions across runs
                out_path = crops_dir / f"{q.question_id}_p{page_num:03d}.png"
                crop.save(out_path)
                q.image_crops.append(str(out_path))

            if SHOW_CROPS:
                plt.figure()
                plt.imshow(crop)
                plt.axis("off")
                plt.title(f"{q.question_id} (page {page_num})")
                plt.show()


# ===================== DB APPEND (NESTED INGESTIONS) =====================

def append_ingestion_to_db(
    db_path: Path,
    created_at: str,
    source_pdf: str,
    exam_id: str,
    ingestion_id: str,
    questions: List[Question],
):
    db = load_questions_db(db_path)

    ingestion_obj = {
        "ingestion_id": ingestion_id,
        "created_at": created_at,
        "source_pdf": source_pdf,
        "exam_id": exam_id,
        "questions": [
            {
                "question_id": q.question_id,
                "start_page": q.start_page,
                "page_nums": q.page_nums(),
                "text": q.text,
                "text_hash": stable_hash16(q.text.lower().strip()),
                "image_crops": q.image_crops,
                "type": q.qtype,         # placeholder
                "metadata": q.metadata,  # placeholder
            }
            for q in questions
        ],
    }

    db["ingestions"].append(ingestion_obj)
    atomic_write_json(db_path, db)


# ===================== ENTRY POINT =====================

def main():
    pdf_path = Path(PDF_PATH)
    assert pdf_path.exists(), f"PDF not found: {pdf_path.resolve()}"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # User-provided exam id (used for provenance + crop folder naming)
    user_exam_id = input("Enter exam id: ").strip()
    exam_id = normalize_exam_id(user_exam_id)

    created_at = utc_now_iso()
    source_pdf = str(pdf_path)

    ingestion_id = make_ingestion_id(created_at=created_at, source_pdf=source_pdf, exam_id=exam_id)

    # Parse questions
    questions = parse_pdf_to_questions(pdf_path)

    # Assign question_ids + placeholders
    for i, q in enumerate(questions, 1):
        q.question_id = make_question_id(exam_id=exam_id, ingestion_id=ingestion_id, question_index=i)
        q.qtype = None
        q.metadata = {}

    print(f"\nDetected {len(questions)} questions")
    print(f"exam_id={exam_id}")
    print(f"ingestion_id={ingestion_id}\n")

    for i, q in enumerate(questions, 1):
        preview = q.text.replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "..."
        print(f"[{i:02d}] {q.question_id} start_page={q.start_page} pages={q.page_nums()}")
        print(f"     {preview}\n")

    # Crop output directory organized by exam_id / ingestion_id
    crops_dir = Path(OUTPUT_DIR) / "crops" / exam_id / ingestion_id
    crop_and_output_questions(pdf_path, questions, crops_dir=crops_dir)

    # Append to nested-ingestions DB
    db_path = Path(OUTPUT_DIR) / QUESTIONS_DB_FILENAME
    append_ingestion_to_db(
        db_path=db_path,
        created_at=created_at,
        source_pdf=source_pdf,
        exam_id=exam_id,
        ingestion_id=ingestion_id,
        questions=questions,
    )

    print(f"Appended ingestion to: {db_path}")
    print(f"Crops saved under: {crops_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
