
import gc
import os
import shutil
from pathlib import Path

from pdf2image import convert_from_path
from server.layout_ingest import (
    append_ingestion_to_db,
    crop_and_output_questions,
    load_layout_model,
    make_ingestion_id,
    make_question_id,
    normalize_exam_id,
    parse_pdf_to_questions,
    utc_now_iso,
)
from vectordb.vectordb_workflow import DBWorkflow

PDF_PATH = "exam_tests/practicefinal3.pdf"
OUTPUT_DIR = "layout_debug"

START_PAGE = 1
END_PAGE = 10

Y_TOL = 25            # reading-order tolerance

SAVE_CROPS = True
SHOW_CROPS = True     # set False if running headless (no GUI)
CROP_PADDING = 10     # pixels of padding around bbox

QUESTIONS_DB_FILENAME = "questions.json"  # stored inside OUTPUT_DIR

DEBUG = True
DEBUG_DRAW_LAYOUT = False   # <-- IMPORTANT: avoids Pillow10 layoutparser crash


class CombinedWorkflow:
    def __init__(self):
        self.json_path = OUTPUT_DIR + "/" + QUESTIONS_DB_FILENAME

    def run_layout_ingest(self):
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

    def run_vectordb(self):
        workflow = DBWorkflow(json_path=str(self.json_path), has_categories=False)
        workflow.run()

        # Release embedding model before exit so FlagEmbedding's __del__ runs while
        # the interpreter is still intact (avoids AttributeError: 'NoneType' ... SIGTERM).
        workflow.embedding_model = None
        gc.collect()

    def run(self):
        self.run_layout_ingest()
        self.run_vectordb()

if __name__ == "__main__":
    workflow = CombinedWorkflow()
    workflow.run()