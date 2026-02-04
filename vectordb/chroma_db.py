"""
ChromaDB vector database for question embeddings.
Supports add by question_id, remove by question_id, and query n nearest neighbors.
"""
import os
from typing import Sequence

import chromadb
from chromadb.config import Settings

# Default path for persistent Chroma data (inside this package)
_DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chroma_persist",
)
_COLLECTION_NAME = "question_embeddings"


def get_chroma_client(persist_directory: str | None = None):
    """Create a Chroma PersistentClient. Data is stored on disk."""
    path = persist_directory or _DEFAULT_PERSIST_DIR
    return chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
    )


class QuestionVectorDB:
    """ChromaDB-backed store for question embeddings keyed by question_id."""

    def __init__(self, persist_directory: str | None = None) -> None:
        self._client = get_chroma_client(persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_embedding(self, question_id: str, embedding: Sequence[float]) -> None:
        """Add or overwrite a single embedding for the given question_id."""
        self._collection.upsert(
            ids=[question_id],
            embeddings=[list(embedding)],
        )

    def add_embeddings(
        self,
        question_ids: list[str],
        embeddings: list[Sequence[float]],
    ) -> None:
        """Add or overwrite multiple embeddings. Lengths of question_ids and embeddings must match."""
        if len(question_ids) != len(embeddings):
            raise ValueError("question_ids and embeddings must have the same length")
        self._collection.upsert(
            ids=question_ids,
            embeddings=[list(e) for e in embeddings],
        )

    def remove_embedding(self, question_id: str) -> None:
        """Remove the embedding for the given question_id. No-op if id not present."""
        self._collection.delete(ids=[question_id])

    def get_n_closest(
        self,
        query_embedding: Sequence[float],
        n: int,
        include_distances: bool = True,
    ) -> dict:
        """
        Return the n closest embeddings to the query embedding.

        Returns a dict with "ids" (list of question_ids) and optionally "distances"
        (list of distances, if include_distances=True). Order is nearest first.
        """
        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n,
            include=["distances"] if include_distances else [],
        )
        out = {"ids": result["ids"][0] if result["ids"] else []}
        if include_distances and result.get("distances"):
            out["distances"] = result["distances"][0]
        return out
