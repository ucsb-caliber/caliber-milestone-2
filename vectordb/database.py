"""
Database setup: engine, session, and table creation for the vectordb SQLite DB.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from vectordb.models import Base

_base = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATABASE_URL = f"sqlite:///{os.path.join(_base, 'questions.db')}"


def get_engine(url: str | None = None):
    """Create SQLAlchemy engine for SQLite."""
    url = url or DEFAULT_DATABASE_URL
    return create_engine(
        url,
        connect_args={"check_same_thread": False},
        echo=False,
    )


def get_session_factory(engine=None):
    """Return a session factory bound to the given engine or default."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def create_all(engine=None):
    """Create all tables (categories, questions)."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
