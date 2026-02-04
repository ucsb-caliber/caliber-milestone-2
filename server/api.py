# server/api.py
from __future__ import annotations

import asyncio
import importlib.util
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool


# ----------------------------
# Paths 
# ----------------------------
SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent

INDEX_HTML = SERVER_DIR / "index.html"
STATIC_DIR = SERVER_DIR / "static" 
INGEST_SCRIPT = SERVER_DIR / "layout_ingest.py"  

UPLOADS_DIR = SERVER_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

#output
DEFAULT_OUTPUT_DIR = SERVER_DIR / "layout_debug"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Dynamic import of layout.ingest.py
# ----------------------------
def _load_ingest_module() -> Any:
    if not INGEST_SCRIPT.exists():
        raise FileNotFoundError(f"Missing ingest script: {INGEST_SCRIPT}")

    spec = importlib.util.spec_from_file_location("layout_ingest_mod", str(INGEST_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create import spec for ingest module")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


INGEST_MOD = _load_ingest_module()


# ----------------------------
# Model caching 
# ----------------------------
_MODEL_LOCK = threading.Lock()
_CACHED_MODEL: Optional[Any] = None


def _get_model() -> Any:
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    with _MODEL_LOCK:
        if _CACHED_MODEL is None:
            _CACHED_MODEL = INGEST_MOD.load_layout_model()
    return _CACHED_MODEL


# ----------------------------
# Core ingestion runner
# ----------------------------
def _run_ingest(
    pdf_path: Path,
    exam_id: str,
    start_page: int,
    end_page: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Runs your existing pipeline in "server mode" (no GUI popups),
    using your module's functions directly.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Make module behave like server mode (no plt.show)
    INGEST_MOD.SHOW_CROPS = False
    INGEST_MOD.SAVE_CROPS = True

    # Use API-provided page range 
    INGEST_MOD.START_PAGE = max(1, int(start_page))
    INGEST_MOD.END_PAGE = int(end_page) if end_page is not None else 0

    # Output dir
    INGEST_MOD.OUTPUT_DIR = str(output_dir)

    created_at = INGEST_MOD.utc_now_iso()
    exam_id_norm = INGEST_MOD.normalize_exam_id(exam_id)
    source_pdf = str(pdf_path)

    ingestion_id = INGEST_MOD.make_ingestion_id(
        created_at=created_at, source_pdf=source_pdf, exam_id=exam_id_norm
    )

    # Load model (cached)
    model = _get_model()

    # Render pages
    pages = INGEST_MOD.convert_from_path(str(pdf_path))

    # Parse questions
    questions = INGEST_MOD.parse_pdf_to_questions(pages, model)

    # Assign IDs, metadata
    for i, q in enumerate(questions, 1):
        q.question_id = INGEST_MOD.make_question_id(
            exam_id=exam_id_norm, ingestion_id=ingestion_id, question_index=i
        )
        q.qtype = None
        q.metadata = {}

    # Crop images
    crops_dir = output_dir / "crops" / exam_id_norm / ingestion_id
    INGEST_MOD.crop_and_output_questions(pages, questions, crops_dir=crops_dir)

    # Append to DB
    db_path = output_dir / INGEST_MOD.QUESTIONS_DB_FILENAME
    INGEST_MOD.append_ingestion_to_db(
        db_path=db_path,
        created_at=created_at,
        source_pdf=source_pdf,
        exam_id=exam_id_norm,
        ingestion_id=ingestion_id,
        questions=questions,
    )

    # Build response summary
    q_summaries = []
    for i, q in enumerate(questions, 1):
        preview = q.text.replace("\n", " ").strip()
        if len(preview) > 240:
            preview = preview[:240] + "…"
        q_summaries.append(
            {
                "index": i,
                "question_id": q.question_id,
                "start_page": q.start_page,
                "page_nums": q.page_nums(),
                "text_preview": preview,
                "image_crops": q.image_crops,
            }
        )

    return {
        "ok": True,
        "exam_id": exam_id_norm,
        "created_at": created_at,
        "ingestion_id": ingestion_id,
        "source_pdf": source_pdf,
        "output_dir": str(output_dir),
        "db_path": str(db_path),
        "num_questions": len(questions),
        "questions": q_summaries,
    }


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Caliber Ingest API")

# Serve /static/app.js, etc.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    if not INDEX_HTML.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Missing {INDEX_HTML}. Put your index.html in server/index.html",
        )
    return INDEX_HTML.read_text(encoding="utf-8")


@app.post("/api/ingest")
async def ingest(
    exam_id: str = Form(...),
    start_page: int = Form(1),
    end_page: int = Form(0),
    pdf: UploadFile = File(...),
) -> JSONResponse:
    if pdf.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")

    # Save upload to disk 
    upload_id = uuid.uuid4().hex
    safe_name = (pdf.filename or "upload.pdf").replace("/", "_").replace("\\", "_")
    pdf_path = UPLOADS_DIR / f"{upload_id}_{safe_name}"

    data = await pdf.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")

    pdf_path.write_bytes(data)

    # Run heavy work off the event loop
    try:
        result = await run_in_threadpool(
            _run_ingest,
            pdf_path,
            exam_id,
            start_page,
            end_page,
            DEFAULT_OUTPUT_DIR,
        )
        return JSONResponse(result)
    except Exception as e:
        # Keep error readable for frontend
        return JSONResponse(
            {"ok": False, "error": str(e), "pdf_saved_as": str(pdf_path)},
            status_code=500,
        )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}
