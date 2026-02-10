import os
from sqlmodel import SQLModel, create_engine, Session
from dotenv import load_dotenv

load_dotenv()

# Import models so SQLModel.metadata is populated (same schema as milestone-one)
from vectordb import models  # noqa: F401

# Use SQLite by default for easy local development (same default as milestone-one)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/questionbank.db")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    # SQLite specific settings
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=True)
else:
    # For PostgreSQL or other databases
    engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables():
    """Create all tables in the database."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Dependency to get a database session."""
    with Session(engine) as session:
        yield session


def get_engine(url=None):
    """Return an engine for the given URL or default DATABASE_URL. Optional compatibility helper."""
    u = url or DATABASE_URL
    if u.startswith("sqlite"):
        return create_engine(u, connect_args={"check_same_thread": False}, echo=True)
    return create_engine(u, echo=True)
