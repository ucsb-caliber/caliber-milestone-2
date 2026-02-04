"""
SQLAlchemy models for Categories and Questions (SQLite).
"""
from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Category(Base):
    """Category: has a name, id, and many questions."""

    __tablename__ = "categories"

    category_id = Column(Integer, primary_key=True, autoincrement=True)
    category_name = Column(String(255), unique=True, nullable=False, index=True)

    # One category has many questions
    questions = relationship(
        "Question",
        back_populates="category",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Category(category_id={self.category_id!r}, category_name={self.category_name!r})>"


class Question(Base):
    """
    Question: all fields from questions_categories.json question objects,
    plus a foreign key to its category.
    """

    __tablename__ = "questions"

    # Primary key
    question_id = Column(String(64), primary_key=True)

    # Foreign key to category (each question belongs to one category)
    category_id = Column(
        Integer,
        ForeignKey("categories.category_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    category = relationship("Category", back_populates="questions")

    # Fields from questions_categories.json
    start_page = Column(Integer, nullable=True)
    page_nums = Column(JSON, nullable=True)  # list of ints; SQLite stores as JSON text
    text = Column(Text, nullable=True)
    text_hash = Column(String(64), nullable=True, index=True)
    image_crops = Column(JSON, nullable=True)  # list of strings
    type = Column(String(64), nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)  # dict; avoid shadowing .metadata

    def __repr__(self) -> str:
        return f"<Question(question_id={self.question_id!r}, category_id={self.category_id})>"
