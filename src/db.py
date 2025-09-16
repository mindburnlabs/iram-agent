"""
Database utilities: async engine, session management, and schema creation
"""
import os
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()
_engine: Optional[AsyncEngine] = None
_Session: Optional[sessionmaker] = None


def get_database_url() -> Optional[str]:
    url = os.getenv("DATABASE_URL")
    # Ensure async driver prefix
    if url and url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def get_engine() -> Optional[AsyncEngine]:
    global _engine
    if _engine is None:
        db_url = get_database_url()
        if not db_url:
            return None
        _engine = create_async_engine(db_url, future=True, pool_pre_ping=True)
    return _engine


def get_sessionmaker() -> Optional[sessionmaker]:
    global _Session
    if _Session is None:
        engine = get_engine()
        if engine is None:
            return None
        _Session = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    return _Session


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional scope around a series of operations."""
    SessionLocal = get_sessionmaker()
    if SessionLocal is None:
        raise RuntimeError("Database not configured (DATABASE_URL missing)")
    async with SessionLocal() as session:  # type: ignore[arg-type]
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all() -> bool:
    """Create all tables if the database is configured."""
    engine = get_engine()
    if engine is None:
        return False
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return True


async def init_database():
    """Initialize database and create tables if needed."""
    await create_all()


async def close_database():
    """Close database connections."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with session_scope() as session:
        yield session
