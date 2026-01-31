import os
import io
import base64
import shutil
import subprocess
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
import numpy as np
import cv2
import pytesseract
import time
import concurrent.futures
import matplotlib.pyplot as plt
from pytesseract import Output
# ===================== CONFIG =====================

OUTPUT_DIR = "layout_debug"

START_PAGE = 1
END_PAGE = 10

Y_TOL = 25            # reading-order tolerance

SAVE_CROPS = False
SHOW_CROPS = False    # set False if running headless (no GUI)
CROP_PADDING = 10     # pixels of padding around bbox

QUESTIONS_DB_FILENAME = "questions.json"  # stored inside OUTPUT_DIR

DEBUG = False

MODELS = {
    '0': 'faster_rcnn_R_50_FPN_3x',
    '1': 'mask_rcnn_X_101_32x8d_FPN_3x'
}
# ===================== QUESTION DETECTION =====================

QUESTION_START_PATTERNS = [
    r"^\s*Problem\s+\d+\b",
    r"^\s*Question\s+\d+\b",
    r"^\s*Q\s*\d+\b",
    r"^\d+\.\s+.*[\.\?:]$"
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

def is_question_start(block: Block) -> bool:
    return bool(QUESTION_START_RE.match(block.text.strip()))


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


def annotate_figure(img_bytes: bytes, prompt: str) -> str:
    import ollama
    response = ollama.chat(
        model='qwen3-vl:235b-cloud',
        messages=[{'role': 'user', 'content': f'This is a figure from a CS exam which procedes this question text: {prompt}. Output ONLY a short description of the figure. Do NOT ouput any additional text, notes/comments, or conversational phrases. Do NOT answer the question text.', 'images': [img_bytes]}]
    )
    return response['message']['content']


# ===================== IMAGE + OCR =====================

def image_to_bytes(image: Image.Image, extension: str, quality: int = 100) -> str: 
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=extension, quality=quality)
    img_bytes = img_byte_arr.getvalue()
    
    return img_bytes


def get_text_within_box(ocr_data: Dict, bbox: Tuple[int, int, int, int], conf_thresh: int = 50) -> str:
    x1, y1, x2, y2 = bbox
    words = []
    x1 -= 30 # helped capture question starts
    x1 = max(0, x1)
    for i in range(len(ocr_data["text"])):
        if int(ocr_data["conf"][i]) < conf_thresh:
            continue

        cx = ocr_data["left"][i] + ocr_data["width"][i] / 2
        cy = ocr_data["top"][i] + ocr_data["height"][i] / 2

        if x1 <= cx <= x2 and y1 <= cy <= y2:
            words.append(ocr_data["text"][i])

    return " ".join(words)


def parse_page(layout: lp.Layout, page_img: Image.Image, page_num: int) -> List[Block]:
    ocr_data = pytesseract.image_to_data( # process entire page once
        page_img,
        output_type=Output.DICT,
        config="--oem 3 --psm 6 -l eng"
    )
    page_blocks: List[Block] = []
    for b in layout:    
        x1, y1, x2, y2 = map(int, b.block.coordinates)
        if b.type == 'Figure': 
            prompt = page_blocks[-1].text if page_blocks else ""
            crop = page_img.crop((x1, y1, x2, y2))
            figure_bytes = image_to_bytes(crop, 'JPEG', 80)
            text = annotate_figure(figure_bytes, prompt)
        else: 
            text = get_text_within_box(ocr_data, (x1, y1, x2, y2)) # match bbox coordinates to text

        if text.strip():
           page_blocks.append(Block(
                page=page_num,
                bbox=(x1, y1, x2, y2),
                text=text,
                btype = b.type
            ))       
    return page_blocks


# ===================== LAYOUT FORMATTING =====================

def keep_largest_blocks(layout: lp.Layout, threshold: int=0.9) -> lp.Layout: # Filters out redundant blocks from layout
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
            
            if small_area > 0:
                coverage = inter_area / small_area
            else:
                coverage = 0
            
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


# ===================== TEXT NORMALIZATION =====================

def split_block_text(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in QUESTION_START_RE.split(text) if p.strip()]
    return parts if parts else [text.strip()]


# ===================== MAIN PARSER =====================

def parse_pdf_to_questions(pages: List[Image.Image], model: Any) -> List[Question]:
    start = max(1, START_PAGE)
    end = min(END_PAGE, len(pages))

    all_questions: List[Question] = []
    current_question: Optional[Question] = None

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = []
        for page_idx in range(start - 1, end):
            page_num = page_idx + 1
            page_img = pages[page_idx]

            layout = model.detect(page_img)
            layout = keep_largest_blocks(layout, threshold=0.8)
            layout = sort_layout_reading_order(layout, Y_TOL)
            
            if DEBUG:
                SAVE_PATH = os.path.join('pages', f"debug_page_{page_num}.png")
                lp.draw_box(page_img, layout, box_width=3, show_element_type=True).save(SAVE_PATH) 

            future = executor.submit(
                parse_page,
                layout,
                page_img,
                page_num,
            )
            futures.append(future)
        
        for future in futures:
            for block in future.result():                 
                if is_question_start(block):
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
    user_exam_id = input("Enter exam id: ").strip()
    exam_id = normalize_exam_id(user_exam_id)
    
    pdf_path = Path(f'exam_tests/{user_exam_id}.pdf') # easier debugging
    assert pdf_path.exists(), f"PDF not found: {pdf_path.resolve()}"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    while True:
        user_model = input("\n[0] faster_rcnn_R_50_FPN_3x(Faster)\n[1] mask_rcnn_X_101_32x8d_FPN_3x(More Accurate)\nEnter 0 or 1 For Model Selection: ")
        if user_model in ['0', '1']:
            user_model = MODELS[user_model]
            break

    created_at = utc_now_iso()
    source_pdf = str(pdf_path)

    ingestion_id = make_ingestion_id(created_at=created_at, source_pdf=source_pdf, exam_id=exam_id)

    if DEBUG:
        if os.path.exists('pages'):
            shutil.rmtree('pages')
        os.makedirs('pages')
        with open("Output.txt", "w") as text_file:
            pass

    print(f'Loading {user_model}...')
    try:
        model = lp.Detectron2LayoutModel(
            config_path = f'lp://PubLayNet/{user_model}', 
            model_path = f'{user_model}.pth', # manually added model path needed on Windows
            extra_config=[
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.2,  
                "MODEL.ROI_HEADS.NMS_THRESH_TEST", 0.1
            ],
            label_map= {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )
    except:
        print('Falling back to tf_efficientdet_d1')
        model = lp.AutoLayoutModel("lp://efficientdet/PubLayNet/tf_efficientdet_d1")
    print('Finished loading model.')

    # Parse questions
    pages = convert_from_path(source_pdf)
    questions = parse_pdf_to_questions(pages, model)

    # Assign question_ids + placeholders
    for i, q in enumerate(questions, 1):
        if DEBUG:
            with open("Output.txt", "a") as text_file:
                text_file.write(f"{q.text}\n")
                text_file.write("=====" * 20 + "\n")
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
    crop_and_output_questions(pages, questions, crops_dir=crops_dir)

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