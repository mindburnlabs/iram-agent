"""
SQLAlchemy models for IRAM
"""
from datetime import datetime
import uuid
from typing import Optional

from sqlalchemy import String, DateTime, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="queued")  # queued, running, completed, failed
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0-100

    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)