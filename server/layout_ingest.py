import os
import shutil
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
from PIL import Image
import pytesseract
import concurrent.futures
import matplotlib.pyplot as plt
import torch
import requests
from pytesseract import Output



_TORCH_LOAD_ORIG = torch.load

def torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD_ORIG(*args, **kwargs)

torch.load = torch_load_compat


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

# Run Bloom's Taxonomy classification on each question during ingest (requires transformers/torch)
ENABLE_BLOOM_CLASSIFICATION = True

DEBUG = True
DEBUG_DRAW_LAYOUT = False   # <-- IMPORTANT: avoids Pillow10 layoutparser crash


# ===================== QUESTION DETECTION =====================

QUESTION_START_PATTERNS = [
    r"^\s*Problem\s+\d+\b",
    r"^\s*Question\s+\d+\b",
    r"^\s*Q\s*\d+\b",
    r"^\s*\d+\.\s+.+[. ?:].*"
]

QUESTION_START_RE = re.compile("|".join(QUESTION_START_PATTERNS), re.IGNORECASE)
NUMBERED_ITEM_SPLIT_RE = re.compile(r"(?=\n?\s*\d+\s*[\.\)])")


# ===================== DATA STRUCTURES =====================

@dataclass
class Block:
    page: int
    bbox: Tuple[int, int, int, int]
    text: str
    btype: str


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
        backup = db_path.with_suffix(db_path.suffix + f".bak.{uuid.uuid4().hex}")
        try:
            db_path.replace(backup)
        except Exception:
            pass
        return {"schema_version": "1.0", "ingestions": []}


def is_question_start(block: Block) -> bool:
    return bool(QUESTION_START_RE.match(block.text.strip()))


# ===================== BLOOM'S TAXONOMY =====================

def run_bloom_on_questions(questions: List[Question]) -> None:
    """Run Bloom classifier on each question and set q.metadata with bloom_level, bloom_confidence, bloom_reasoning."""
    if not questions:
        return
    try:
        from server import bloom_classifier
    except ImportError:
        import sys
        _server_dir = Path(__file__).resolve().parent
        if str(_server_dir) not in sys.path:
            sys.path.insert(0, str(_server_dir))
        import bloom_classifier  # type: ignore[assignment]
    for q in questions:
        result = bloom_classifier.classify(q.text)
        q.metadata["bloom_level"] = result.get("bloom_level")
        q.metadata["bloom_confidence"] = result.get("bloom_confidence")
        q.metadata["bloom_reasoning"] = result.get("bloom_reasoning")


# -------------------- Debug drawing wrapper --------------------

def safe_draw_layout_debug(page_img: Image.Image, layout: lp.Layout, save_path: str):
    """
    layoutparser's draw_box uses Pillow FreeTypeFont.getsize(), removed in Pillow 10+.
    If you really want debug visuals, either:
      - pin Pillow<10, or
      - patch layoutparser, or
      - keep this disabled (default).
    This wrapper prevents your pipeline from crashing.
    """
    try:
        lp.draw_box(page_img, layout, box_width=3, show_element_type=True).save(save_path)
    except Exception as e:
        print(f"[warn] draw_box failed (Pillow/layoutparser compat): {e}")


# ===================== OCR + PARSING =====================

def get_text_within_box(ocr_data: Dict, bbox: Tuple[int, int, int, int], conf_thresh: int = 50) -> str:
    x1, y1, x2, y2 = bbox
    words = []

    for i in range(len(ocr_data["text"])):
        try:
            conf = float(ocr_data["conf"][i])
        except Exception:
            conf = -1
        if conf < conf_thresh:
            continue

        cx = ocr_data["left"][i] + ocr_data["width"][i] / 2
        cy = ocr_data["top"][i] + ocr_data["height"][i] / 2

        if x1 <= cx <= x2 and y1 <= cy <= y2:
            w = ocr_data["text"][i]
            if w:
                words.append(w)

    return " ".join(words).strip()


def parse_page(layout: lp.Layout, page_img: Image.Image, page_num: int) -> List[Block]:
    ocr_data = pytesseract.image_to_data(
        page_img,
        output_type=Output.DICT,
        config="--oem 3 --psm 6 -l eng"
    )

    page_blocks: List[Block] = []
    for b in layout:
        x1, y1, x2, y2 = map(int, b.block.coordinates)
        text = get_text_within_box(ocr_data, (x1, y1, x2, y2))
        if not text:
            continue
        page_blocks.append(Block(page=page_num, bbox=(x1, y1, x2, y2), text=text, btype=b.type))

    return page_blocks


# ===================== LAYOUT FORMATTING =====================

def keep_largest_blocks(layout: lp.Layout, threshold: int = 0.9) -> lp.Layout:
    sorted_layout = sorted(layout, key=lambda x: x.block.area, reverse=True)

    keep = []
    while sorted_layout:
        large_block = sorted_layout.pop(0)
        keep.append(large_block)

        remaining = []
        for small_block in sorted_layout:
            x1 = max(large_block.block.x_1, small_block.block.x_1)
            y1 = max(large_block.block.y_1, small_block.block.y_1)
            x2 = min(large_block.block.x_2, small_block.block.x_2)
            y2 = min(large_block.block.y_2, small_block.block.y_2)

            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            small_area = small_block.block.area
            coverage = inter_area / small_area if small_area > 0 else 0

            if coverage < threshold:
                remaining.append(small_block)

        sorted_layout = remaining

    return lp.Layout(keep)


def sort_layout_reading_order(layout: lp.Layout, y_tol: int) -> lp.Layout:
    layout = sorted(layout, key=lambda b: (int(b.block.coordinates[1]), int(b.block.coordinates[0])))
    rows: List[List[lp.TextBlock]] = []

    for b in layout:
        placed = False
        for row in rows:
            if abs(b.block.coordinates[1] - row[0].block.coordinates[1]) <= y_tol:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    ordered: List[lp.TextBlock] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b.block.coordinates[0]))

    return lp.Layout(ordered)


# ===================== MAIN PARSER =====================

def parse_pdf_to_questions(pages: List[Image.Image], model: Any) -> List[Question]:
    last_page = len(pages)
    start = max(1, START_PAGE)
    end = END_PAGE or last_page
    end = min(end, last_page)

    all_questions: List[Question] = []
    current_question: Optional[Question] = None

    max_workers = max(1, (os.cpu_count() or 2) - 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for page_idx in range(start - 1, end):
            page_num = page_idx + 1
            page_img = pages[page_idx]

            layout = model.detect(page_img)
            layout = keep_largest_blocks(layout, threshold=0.9)
            layout = sort_layout_reading_order(layout, Y_TOL)

            if DEBUG and DEBUG_DRAW_LAYOUT:
                Path("pages").mkdir(exist_ok=True)
                save_path = os.path.join("pages", f"debug_page_{page_num}.png")
                safe_draw_layout_debug(page_img, layout, save_path)

            futures.append((page_num, executor.submit(parse_page, layout, page_img, page_num)))

        futures.sort(key=lambda x: x[0])

        for _, fut in futures:
            for block in fut.result():
                if is_question_start(block) and block.btype in ['Text', 'Title']:
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


def crop_and_output_questions(pages: List[Image.Image], questions: List[Question], crops_dir: Path):
    crops_dir.mkdir(parents=True, exist_ok=True)

    for q in questions:
        per_page = q.bboxes_by_page()

        for page_num, bbox in sorted(per_page.items()):
            page_img = pages[page_num - 1]
            w, h = page_img.size

            x1, y1, x2, y2 = clamp_bbox(bbox, w, h, pad=CROP_PADDING)
            if (x1, y1, x2, y2) == (0, 0, 0, 0):
                continue

            crop = page_img.crop((x1, y1, x2, y2))

            if SAVE_CROPS:
                out_path = crops_dir / f"{q.question_id}_p{page_num:03d}.png"
                crop.save(out_path)
                q.image_crops.append(str(out_path))

            if SHOW_CROPS:
                plt.figure()
                plt.imshow(crop)
                plt.axis("off")
                plt.title(f"{q.question_id} (page {page_num})")
                plt.show()


# ===================== DB APPEND =====================

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
                "type": q.qtype,
                "metadata": q.metadata,
                "bloom_level": q.metadata.get("bloom_level"),
                "bloom_confidence": q.metadata.get("bloom_confidence"),
                "bloom_reasoning": q.metadata.get("bloom_reasoning"),
            }
            for q in questions
        ],
    }

    db["ingestions"].append(ingestion_obj)
    atomic_write_json(db_path, db)


# ===================== MODEL DOWNLOAD + LOADING =====================

def _download_file(url: str, dest: Path, *, min_bytes: int, forbid_html: bool = True, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size >= min_bytes:
        if forbid_html:
            with dest.open("rb") as f:
                head = f.read(1)
            if head != b"<":
                return dest
            dest.unlink(missing_ok=True)
        else:
            return dest

    tmp = dest.with_suffix(dest.suffix + f".tmp.{uuid.uuid4().hex}")

    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    size = tmp.stat().st_size
    if size < min_bytes:
        head = tmp.read_bytes()[:200]
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download too small ({size} bytes) from {url}. Head={head!r}")

    if forbid_html:
        head1 = tmp.read_bytes()[:1]
        if head1 == b"<":
            head200 = tmp.read_bytes()[:200]
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded HTML instead of model file from {url}. Head={head200!r}")

    tmp.replace(dest)
    return dest


def load_layout_model():
    """
    We’ll keep Detectron2 attempt, but your pipeline is already working with EfficientDet.
    Also: Detectron2 config.yml can legitimately be ~5KB, so min_bytes lowered.
    """
    print("Loading model...")

    label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

    cache_root = Path.home() / ".cache" / "caliber_layout_models"
    cache_root.mkdir(parents=True, exist_ok=True)

    # --- Detectron2 PubLayNet via HF (optional) ---
    det_cfg_url = "https://huggingface.co/nlpconnect/PubLayNet-faster_rcnn_R_50_FPN_3x/resolve/main/config.yml"
    det_wts_url = "https://huggingface.co/nlpconnect/PubLayNet-faster_rcnn_R_50_FPN_3x/resolve/main/model_final.pth"
    det_cfg_path = cache_root / "publaynet_fasterrcnn" / "config.yml"
    det_wts_path = cache_root / "publaynet_fasterrcnn" / "model_final.pth"

    try:
        _download_file(det_cfg_url, det_cfg_path, min_bytes=2_000, forbid_html=True)
        _download_file(det_wts_url, det_wts_path, min_bytes=50_000_000, forbid_html=True)

        model = lp.Detectron2LayoutModel(
            config_path=str(det_cfg_path),
            model_path=str(det_wts_path),
            label_map=label_map,
            extra_config=[
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.2,
                "MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.1,
            ],
        )
        print("Done! (Detectron2 PubLayNet via HF)")
        return model
    except Exception as e:
        print(f"[warn] Detectron2 (HF) failed: {e}")

    # --- EfficientDet PubLayNet via HF (your run succeeds here) ---
    eff_url = "https://huggingface.co/layoutparser/efficientdet/resolve/main/PubLayNet/tf_efficientdet_d1/publaynet-tf_efficientdet_d1.pth.tar"
    eff_path = cache_root / "publaynet_effdet" / "publaynet-tf_efficientdet_d1.pth.tar"

    _download_file(eff_url, eff_path, min_bytes=50_000_000, forbid_html=True)

    if hasattr(lp, "EfficientDetLayoutModel"):
        model = lp.EfficientDetLayoutModel(
            "tf_efficientdet_d1",
            model_path=str(eff_path),
            label_map=label_map
        )
    else:
        model = lp.models.effdet.layoutmodel.EfficientDetLayoutModel(
            "tf_efficientdet_d1",
            model_path=str(eff_path),
            label_map=label_map
        )

    print("Done! (EfficientDet PubLayNet via HF)")
    return model


# ===================== ENTRY POINT =====================

def main():
    pdf_path = Path(PDF_PATH)
    assert pdf_path.exists(), f"PDF not found: {pdf_path.resolve()}"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    user_exam_id = input("Enter exam id: ").strip()
    exam_id = normalize_exam_id(user_exam_id)

    created_at = utc_now_iso()
    source_pdf = str(pdf_path)
    ingestion_id = make_ingestion_id(created_at=created_at, source_pdf=source_pdf, exam_id=exam_id)

    if DEBUG:
        if os.path.exists('pages'):
            shutil.rmtree('pages')
        os.makedirs('pages', exist_ok=True)
        with open("Output.txt", "w") as _:
            pass

    model = load_layout_model()

    pages = convert_from_path(str(pdf_path))

    questions = parse_pdf_to_questions(pages, model)

    for i, q in enumerate(questions, 1):
        if DEBUG:
            with open("Output.txt", "a") as text_file:
                text_file.write(f"{q.text}\n")
                text_file.write("=====" * 20 + "\n")

        q.question_id = make_question_id(exam_id=exam_id, ingestion_id=ingestion_id, question_index=i)
        q.qtype = None
        q.metadata = {}

    if ENABLE_BLOOM_CLASSIFICATION:
        print("Running Bloom's Taxonomy classification...")
        run_bloom_on_questions(questions)
        print("Bloom classification done.")

    print(f"\nDetected {len(questions)} questions")
    print(f"exam_id={exam_id}")
    print(f"ingestion_id={ingestion_id}\n")

    for i, q in enumerate(questions, 1):
        preview = q.text.replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "..."
        print(f"[{i:02d}] {q.question_id} start_page={q.start_page} pages={q.page_nums()}")
        print(f"     {preview}\n")

    crops_dir = Path(OUTPUT_DIR) / "crops" / exam_id / ingestion_id
    crop_and_output_questions(pages, questions, crops_dir=crops_dir)

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
