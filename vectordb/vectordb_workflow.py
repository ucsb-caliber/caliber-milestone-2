"""
Workflow to load questions_categories.json into the SQLite vectordb (Categories + Questions).
"""
import json
import os
import sys
from collections import Counter

# Allow running as script: python vectordb/vectordb_workflow.py (project root on path)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from FlagEmbedding import BGEM3FlagModel

from vectordb.chroma_db import QuestionVectorDB
from vectordb.database import create_all, get_engine, get_session_factory
from vectordb.models import Category, Question

# Default path to the JSON input file (next to this module)
_DEFAULT_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "questions_categories.json",
)


class DBWorkflow:
    def __init__(
        self,
        json_path: str | None = None,
        database_url: str | None = None,
        chroma_persist_dir: str | None = None,
        has_categories: bool = True,
    ) -> None:
        self.json_path = json_path or _DEFAULT_JSON_PATH
        self.engine = get_engine(database_url)
        self.Session = get_session_factory(self.engine)
        create_all(self.engine)
        self.vector_db = QuestionVectorDB(persist_directory=chroma_persist_dir)
        self.embedding_model = self.load_embedding()
        self.has_categories = has_categories

    def load_embedding(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = False):
        """Load the embedding model (e.g. BGE-M3)."""
        return BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def load_data(self) -> dict:
        """Load and parse the JSON file at self.json_path. Returns the parsed dict."""
        with open(self.json_path, encoding="utf-8") as f:
            return json.load(f)

    def get_category(self, question_id: str) -> str | None:
        """Return the category name for the given question_id, or None if not found."""
        with self.Session() as session:
            question = session.query(Question).filter_by(question_id=question_id).first()
            if question is None:
                return None
            return question.category.category_name
    def get_embedding(self, text: str) -> list[float]:
        """Convert string text into a vector embedding using the loaded model (call load_embedding first)."""
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not loaded. Call load_embedding() first.")
        dense = self.embedding_model.encode([text], max_length=8192)["dense_vecs"]
        return dense[0].tolist()
    def assign_category(
        self,
        question_text: str,
        n: int = 5,
        threshold: float = 0.5,
    ) -> str:
        """
        Assign a category from the question text using nearest neighbors in the vector DB.
        Gets the top n closest questions; if > threshold fraction are in the same category,
        returns that category, otherwise returns "Review".
        """
        if not question_text.strip():
            print("[assign_category] assigned: Review (empty text)")
            return "Review"
        embedding = self.get_embedding(question_text)
        result = self.vector_db.get_n_closest(embedding, n=n)
        ids = result.get("ids") or []
        if not ids:
            print("[assign_category] assigned: Review (no neighbors)")
            return "Review"
        categories = []
        for qid in ids:
            cat = self.get_category(qid)
            if cat is not None:
                categories.append(cat)
        if not categories:
            print("[assign_category] assigned: Review (no categories from neighbors)")
            return "Review"
        (most_common_cat, count) = Counter(categories).most_common(1)[0]
        if count > len(categories) * threshold:
            print(f"[assign_category] assigned: {most_common_cat}")
            return most_common_cat
        print("[assign_category] assigned: Review (no majority)")
        return "Review"

    def add(self) -> None:
        """Load data, assign a category to each question via assign_category, then call populate with this data."""
        data = self.load_data()
        for ingestion in data.get("ingestions", []):
            for q in ingestion.get("questions", []):
                q["category"] = self.assign_category(q.get("text", ""), n=3, threshold=0.5)
        self.populate(data)

    def populate_sql(self, data: dict) -> list[dict]:
        """Populate Category and Question tables from data. Returns the list of question dicts."""
        all_questions: list[dict] = []
        category_names: set[str] = set()
        for ingestion in data.get("ingestions", []):
            for q in ingestion.get("questions", []):
                all_questions.append(q)
                if q.get("category"):
                    category_names.add(q["category"])

        with self.Session() as session:
            name_to_id: dict[str, int] = {}
            for name in sorted(category_names):
                cat = session.query(Category).filter_by(category_name=name).first()
                if cat is None:
                    cat = Category(category_name=name)
                    session.add(cat)
                    session.flush()
                name_to_id[name] = cat.category_id

            for q in all_questions:
                category_name = q.get("category")
                if not category_name or category_name not in name_to_id:
                    continue
                question_id = q["question_id"]
                if session.query(Question).filter_by(question_id=question_id).first() is not None:
                    continue
                category_id = name_to_id[category_name]
                session.add(
                    Question(
                        question_id=question_id,
                        category_id=category_id,
                        start_page=q.get("start_page"),
                        page_nums=q.get("page_nums"),
                        text=q.get("text"),
                        text_hash=q.get("text_hash"),
                        image_crops=q.get("image_crops"),
                        type=q.get("type"),
                        metadata_=q.get("metadata"),
                    )
                )
            session.commit()

        return all_questions

    def populate_chroma(self, all_questions: list[dict]) -> None:
        """Add vector embeddings for each question to ChromaDB."""
        for q in all_questions:
            text = q.get("text") or ""
            if not text:
                continue
            question_id = q["question_id"]
            embedding = self.get_embedding(text)
            self.vector_db.add_embedding(question_id, embedding)

    def populate(self, data: dict) -> None:
        """Populate both SQL and ChromaDB from data."""
        all_questions = self.populate_sql(data)
        self.populate_chroma(all_questions)

    def run(self) -> None:
        """Run the workflow: populate from JSON. If has_categories, use existing categories; else assign via add()."""
        if self.has_categories:
            data = self.load_data()
            self.populate(data)
        else:
            self.add()


if __name__ == "__main__":
    import gc

    _json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "questions_no_categories.json",
    )
    workflow = DBWorkflow(json_path=_json_path, has_categories=False)
    workflow.run()

    # Release embedding model before exit so FlagEmbedding's __del__ runs while
    # the interpreter is still intact (avoids AttributeError: 'NoneType' ... SIGTERM).
    workflow.embedding_model = None
    gc.collect()
