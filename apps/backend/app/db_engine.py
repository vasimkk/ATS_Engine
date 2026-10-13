"""PostgreSQL engine/session plumbing for the SQLAlchemy data layer.

Every ``Database`` instance owns its own engines (one async for the document
tables, one sync for the encrypted ``api_keys`` table read on the synchronous
LLM hot path) built from these factories.
"""

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.models import Base
from app.config import DATABASE_URL

__all__ = ["Base", "make_async_engine", "make_sync_engine", "init_models_sync"]

def _get_async_url() -> str:
    """Build the asyncpg URL from the DATABASE_URL."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL must be set in .env")
    if DATABASE_URL.startswith("postgresql://"):
        return DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    return DATABASE_URL

def _get_sync_url() -> str:
    """Build the psycopg2 URL from the DATABASE_URL."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL must be set in .env")
    if DATABASE_URL.startswith("postgresql://"):
        return DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)
    return DATABASE_URL

def make_async_engine(path: Path) -> AsyncEngine:
    """Create the async engine (``asyncpg``) for the document tables."""
    # path is ignored now since we use DATABASE_URL
    engine = create_async_engine(_get_async_url(), future=True, pool_pre_ping=True)
    return engine

def make_sync_engine(path: Path) -> Engine:
    """Create the sync engine used for the encrypted api_keys table."""
    # path is ignored now since we use DATABASE_URL
    engine = create_engine(_get_sync_url(), future=True, pool_pre_ping=True)
    return engine

def init_models_sync(engine: Engine) -> None:
    """Create all tables (idempotent) using a sync engine connection."""
    # Ensure pgvector extension exists
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        
    Base.metadata.create_all(engine)
