"""
Workflow to load questions_categories.json into the SQLite vectordb (Questions with tags).
Uses Option A for vector lookup: Chroma document ID is str(Question.id); get_category looks up by Question.id.
"""
import json
import os
import sys
from collections import defaultdict

# Allow running as script: python vectordb/vectordb_workflow.py (project root on path)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from FlagEmbedding import BGEM3FlagModel

from vectordb.chroma_db import QuestionVectorDB
from vectordb.database import create_db_and_tables, engine, get_session
from vectordb.models import Question

# Default path to the JSON input file (next to this module)
_DEFAULT_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "questions_categories.json",
)
_DEFAULT_EXEMPLARS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "category_exemplars.json",
)
_LIVE_COLLECTION = "question_embeddings"
_EXEMPLAR_COLLECTION = "question_embeddings_exemplars"


class DBWorkflow:
    def __init__(
        self,
        json_path: str | None = None,
        database_url: str | None = None,
        chroma_persist_dir: str | None = None,
        has_categories: bool = True,
        exemplar_json_path: str | None = None,
        use_exemplars: bool = True,
        live_weight: float = 1.0,
        exemplar_weight: float = 1.25,
        embedding_max_length: int = 2048,
    ) -> None:
        self.json_path = json_path or _DEFAULT_JSON_PATH
        self.engine = engine  # shared engine from vectordb.database (database_url ignored for now)
        create_db_and_tables()
        self.vector_db = QuestionVectorDB(
            persist_directory=chroma_persist_dir,
            collection_name=_LIVE_COLLECTION,
        )
        self.embedding_model = self.load_embedding()
        self.has_categories = has_categories
        self.use_exemplars = use_exemplars
        self.live_weight = live_weight
        self.exemplar_weight = exemplar_weight
        self.embedding_max_length = embedding_max_length
        self.exemplar_json_path = exemplar_json_path or _DEFAULT_EXEMPLARS_PATH
        self.exemplar_db = None
        self.exemplar_labels: dict[str, str] = {}
        if self.use_exemplars:
            self.exemplar_db = QuestionVectorDB(
                persist_directory=chroma_persist_dir,
                collection_name=_EXEMPLAR_COLLECTION,
            )
            self._load_exemplars()

    def load_embedding(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = False):
        """Load the embedding model (e.g. BGE-M3)."""
        return BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def load_data(self) -> dict:
        """Load and parse the JSON file at self.json_path. Returns the parsed dict."""
        with open(self.json_path, encoding="utf-8") as f:
            return json.load(f)

    def get_category(self, question_id: str) -> str | None:
        """Return the category/tag for the given question_id (str(Question.id) from Chroma; Option A), or None if not found."""
        gen = get_session()
        session = next(gen)
        try:
            try:
                id_val = int(question_id)
            except ValueError:
                return None
            question = session.get(Question, id_val)
            if question is None:
                return None
            return question.tags or None
        finally:
            try:
                next(gen)
            except StopIteration:
                pass
    def get_embedding(self, text: str) -> list[float]:
        """Convert string text into a vector embedding using the loaded model (call load_embedding first)."""
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not loaded. Call load_embedding() first.")
        dense = self.embedding_model.encode([text], max_length=self.embedding_max_length)["dense_vecs"]
        return dense[0].tolist()
    def assign_category(
        self,
        question_text: str,
        n: int = 5,
        threshold: float = 0.5,
        embedding: list[float] | None = None,
    ) -> str:
        """
        Assign a category from the question text using nearest neighbors in the vector DB.
        Gets the top n closest questions; if > threshold fraction are in the same category,
        returns that category, otherwise returns "Review".
        If embedding is provided, it is used instead of computing it from question_text.
        """
        if not question_text.strip():
            print("[assign_category] assigned: Review (empty text)")
            return "Review"
        if embedding is None:
            embedding = self.get_embedding(question_text)
        vote_weights = self._collect_vote_weights(embedding=embedding, n=n)
        if not vote_weights:
            print("[assign_category] assigned: Review (no neighbor labels)")
            return "Review"
        ordered_votes = sorted(vote_weights.items(), key=lambda x: x[1], reverse=True)
        (most_common_cat, best_weight) = ordered_votes[0]
        total_weight = sum(vote_weights.values())
        if best_weight > total_weight * threshold:
            print(f"[assign_category] assigned: {most_common_cat} (weighted vote)")
            return most_common_cat
        print("[assign_category] assigned: Review (no weighted majority)")
        return "Review"

    def _distance_to_similarity(self, distance: float | None) -> float:
        """Convert Chroma cosine distance to a bounded similarity in [0, 1]."""
        if distance is None:
            return 1.0
        return max(0.0, min(1.0, 1.0 - float(distance)))

    def _collect_vote_weights(self, embedding: list[float], n: int) -> dict[str, float]:
        """Collect weighted votes from live question DB and optional exemplar DB."""
        vote_weights: dict[str, float] = defaultdict(float)

        live_result = self.vector_db.get_n_closest(embedding, n=n)
        live_ids = live_result.get("ids") or []
        live_distances = live_result.get("distances") or []
        for idx, qid in enumerate(live_ids):
            label = self.get_category(qid)
            if label is None:
                continue
            distance = live_distances[idx] if idx < len(live_distances) else None
            vote_weights[label] += self.live_weight * self._distance_to_similarity(distance)

        if self.exemplar_db is not None:
            exemplar_result = self.exemplar_db.get_n_closest(embedding, n=n)
            exemplar_ids = exemplar_result.get("ids") or []
            exemplar_distances = exemplar_result.get("distances") or []
            for idx, exemplar_id in enumerate(exemplar_ids):
                label = self.exemplar_labels.get(exemplar_id)
                if label is None:
                    continue
                distance = exemplar_distances[idx] if idx < len(exemplar_distances) else None
                vote_weights[label] += self.exemplar_weight * self._distance_to_similarity(distance)

        return dict(vote_weights)

    def _iter_exemplar_questions(self, payload: dict) -> list[dict]:
        """
        Return exemplar records as dicts with at least text/category.
        Supports both:
        - {"exemplars": [...]}
        - ingestion-shaped payload with "ingestions" -> "questions"
        """
        exemplars: list[dict] = []
        if isinstance(payload.get("exemplars"), list):
            for ex in payload["exemplars"]:
                if isinstance(ex, dict):
                    exemplars.append(ex)
            return exemplars

        for ingestion in payload.get("ingestions", []):
            for q in ingestion.get("questions", []):
                if isinstance(q, dict):
                    exemplars.append(q)
        return exemplars

    def _load_exemplars(self) -> None:
        """Load exemplar labels and vectors into the exemplar collection if configured."""
        if self.exemplar_db is None:
            return
        if not os.path.exists(self.exemplar_json_path):
            print(f"[exemplars] file not found: {self.exemplar_json_path} (continuing without exemplars)")
            return
        with open(self.exemplar_json_path, encoding="utf-8") as f:
            payload = json.load(f)

        exemplars = self._iter_exemplar_questions(payload)
        for idx, ex in enumerate(exemplars):
            text = (ex.get("text") or "").strip()
            label = (ex.get("category") or "").strip()
            if not text or not label:
                continue
            exemplar_id = str(ex.get("id") or ex.get("question_id") or f"ex_{idx}")
            embedding = ex.get("embedding")
            if embedding is None:
                embedding = self.get_embedding(text)
            self.exemplar_db.add_embedding(exemplar_id, embedding)
            self.exemplar_labels[exemplar_id] = label
        print(f"[exemplars] loaded {len(self.exemplar_labels)} exemplars")

    def add(self) -> None:
        """Load data, assign a category to each question via assign_category, then call populate with this data."""
        data = self.load_data()
        for ingestion in data.get("ingestions", []):
            for q in ingestion.get("questions", []):
                text = q.get("text", "")
                embedding = self.get_embedding(text) if text.strip() else None
                if embedding is not None:
                    q["_embedding"] = embedding
                q["category"] = self.assign_category(text, n=3, threshold=0.5, embedding=embedding)
        self.populate(data)

    def populate_sql(self, data: dict) -> list[dict]:
        """Populate Question table from data (category stored in Question.tags). Returns the list of question dicts with _question_id set."""
        all_questions: list[dict] = []
        for ingestion in data.get("ingestions", []):
            for q in ingestion.get("questions", []):
                all_questions.append(q)

        gen = get_session()
        session = next(gen)
        try:
            for q in all_questions:
                question = Question(
                    text=q.get("text", ""),
                    tags=q.get("category", ""),
                    user_id="system",
                    title="",
                )
                session.add(question)
                session.flush()
                q["_question_id"] = question.id
            session.commit()
        finally:
            try:
                next(gen)
            except StopIteration:
                pass

        return all_questions

    def populate_chroma(self, all_questions: list[dict]) -> None:
        """Add vector embeddings for each question to ChromaDB. Uses precomputed _embedding if present.
        Chroma document ID is str(Question.id) (Option A) so get_category(id) can look up by Question.id."""
        for q in all_questions:
            text = q.get("text") or ""
            if not text and "_embedding" not in q:
                continue
            chroma_id = str(q["_question_id"]) if "_question_id" in q else q.get("question_id")
            if chroma_id is None:
                continue
            embedding = q.get("_embedding")
            if embedding is None:
                embedding = self.get_embedding(text)
            self.vector_db.add_embedding(chroma_id, embedding)

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
        "question_no_categories.json",
    )
    workflow = DBWorkflow(json_path=_json_path, has_categories=False)
    workflow.run()

    # Release embedding model before exit so FlagEmbedding's __del__ runs while
    # the interpreter is still intact (avoids AttributeError: 'NoneType' ... SIGTERM).
    workflow.embedding_model = None
    gc.collect()
