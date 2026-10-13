"""SQLAlchemy ORM models for Resume Matcher.
A single declarative ``Base`` backs all tables (doc tables migrated from
TinyDB plus the new ``applications`` and ``api_keys`` tables). The facade in
``app/database.py`` converts ORM rows to plain dicts so the rest of the app
never sees ORM objects — preserving the TinyDB-era contracts.
"""

from datetime import datetime, timezone
from typing import Any
from sqlalchemy import JSON, Boolean, Float, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Timestamps are stored as strings (not native datetimes) to preserve the
    TinyDB-era behavior: code compares them lexically and returns them to
    clients verbatim.
    """
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    """Declarative base shared by every table."""


class Resume(Base):
    """A resume document (master or tailored)."""

    __tablename__ = "resumes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    resume_id: Mapped[str] = mapped_column(String, index=True)
    res_embedding: Mapped[Any | None] = mapped_column(Vector(768), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String, default="md")
    filename: Mapped[str | None] = mapped_column(String, nullable=True)
    is_master: Mapped[bool] = mapped_column(Boolean, default=False)
    parent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    processed_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    processing_status: Mapped[str] = mapped_column(String, default="pending")
    processing_token: Mapped[str | None] = mapped_column(String, nullable=True)


    created_at: Mapped[str] = mapped_column(String, default=_utcnow_iso)
    updated_at: Mapped[str] = mapped_column(String, default=_utcnow_iso)

    __table_args__ = (
        # At most one master resume. Partial unique index enforces the invariant
        # at the storage layer; the facade serializes compound designation
        # changes with a SQLite writer transaction.
        Index(
            "ux_resumes_single_master",
            "is_master",
            unique=True,
            postgresql_where=text("is_master = true"),
        ),
    )


class Job(Base):
    """A job description.

    Only the stable columns are first-class; everything the pipeline attaches
    dynamically (``job_keywords``, ``job_keywords_hash``, ``preview_hash``,
    ``preview_hashes``, ``preview_prompt_id``, ``company``, ``role``) lives in
    ``metadata_json``. The facade flattens that map to top-level keys on read
    and merges non-core keys into it on update, reproducing TinyDB semantics.
    """

    __tablename__ = "jobs"

    job_id: Mapped[int] = mapped_column("id", Integer, primary_key=True)
    content: Mapped[str] = mapped_column("description", Text)
    created_at: Mapped[str] = mapped_column(String, default=_utcnow_iso)
    # Metadata extracted downstream
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata_json", JSON, default=dict)
    
    # Explicit Skills Columns
    from sqlalchemy.dialects.postgresql import ARRAY
    required_skills: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    preferred_skills: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    
    # Precomputed sentence-transformer embedding
    job_embedding: Mapped[Any | None] = mapped_column(Vector(768), nullable=True)


class Improvement(Base):
    """A tailoring result linking an original resume, a tailored resume, and a job."""

    __tablename__ = "improvements"

    request_id: Mapped[str] = mapped_column(String, primary_key=True)
    original_resume_id: Mapped[str] = mapped_column(String)
    tailored_resume_id: Mapped[str] = mapped_column(String, index=True)
    job_id: Mapped[str] = mapped_column(String)
    improvements: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[str] = mapped_column(String, default=_utcnow_iso)


class TailoringPreview(Base):
    """An accepted preview, bounded confirmation claim and immutable result."""

    __tablename__ = "tailoring_previews"
    __table_args__ = (Index("ix_preview_compatibility", "source_id", "job_id", "payload_hash", "created_at"),)

    improvements: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)

    preview_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(String, index=True)
    job_id: Mapped[str] = mapped_column(String, index=True)
    payload_hash: Mapped[str] = mapped_column(String)
    source_hash: Mapped[str] = mapped_column(String)
    job_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[str] = mapped_column(String)
    expires_at: Mapped[str] = mapped_column(String, index=True)
    result_resume_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )
    claim_token: Mapped[str | None] = mapped_column(String, nullable=True)
    claim_expires_at: Mapped[str | None] = mapped_column(String, nullable=True)
    response_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class Application(Base):
    """A Kanban application-tracker card."""

    __tablename__ = "applications"
    __table_args__ = (
        # Concurrency-safe dedupe: a card is unique per (job, applied resume).
        # The app-level select-then-insert relies on this to collapse races.
        UniqueConstraint("job_id", "resume_id", name="uq_application_job_resume"),
    )

    application_id: Mapped[str] = mapped_column(String, primary_key=True)
    job_id: Mapped[str] = mapped_column(String, index=True)
    # The applied/tailored resume shown in the modal and opened by "Edit".
    resume_id: Mapped[str] = mapped_column(String, index=True)
    # Optional base resume the tailored one descends from (powers "stack" grouping).
    master_resume_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="applied", index=True)
    company: Mapped[str | None] = mapped_column(String, nullable=True)
    role: Mapped[str | None] = mapped_column(String, nullable=True)
    applied_at: Mapped[str | None] = mapped_column(String, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String, default=_utcnow_iso)
    updated_at: Mapped[str] = mapped_column(String, default=_utcnow_iso)


class JobMatch(Base):
    """Stores precalculated ATS scores between a Resume and a Job."""

    __tablename__ = "job_matches"
    __table_args__ = (
        UniqueConstraint("job_id", "resume_id", name="uq_job_match_job_resume"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    resume_id: Mapped[str] = mapped_column(String, index=True, default="Unknown")
    job_id: Mapped[str] = mapped_column(String, index=True)
    match_score: Mapped[float] = mapped_column("ats_score", Float)
    score_data_json: Mapped[dict | None] = mapped_column("raw_response", JSON, nullable=True)
    enhanced_resume_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    enhanced_ats_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[str] = mapped_column("computed_at", String, default=_utcnow_iso)


class ApiKey(Base):
    """An encrypted LLM provider API key.

    ``provider`` is the *key-store* provider name (e.g. ``google`` for the
    ``gemini`` LLM provider, via ``_PROVIDER_KEY_MAP``). Only ciphertext is
    stored; plaintext exists in memory only at call time.
    """

    __tablename__ = "api_keys"

    provider: Mapped[str] = mapped_column(String, primary_key=True)
    ciphertext: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[str] = mapped_column(String, default=_utcnow_iso)


class GapAnalysis(Base):
    """Stores the qualitative AI gap analysis between a resume and a job."""

    __tablename__ = "gap_analysis"
    __table_args__ = (
        UniqueConstraint("job_id", "resume_id", name="uq_gap_analysis_job_resume"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    resume_id: Mapped[str] = mapped_column(String, index=True)
    job_id: Mapped[str] = mapped_column(String, index=True)
    
    domain_fit_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    math_engine_critique: Mapped[str | None] = mapped_column(Text, nullable=True)
    missing_skills_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    
    created_at: Mapped[str] = mapped_column(String, default=_utcnow_iso)

