"""
IRAM Database Models

Comprehensive SQLAlchemy models for all data entities including profiles,
posts, jobs, analyses, metrics, audit logs, and user management.
"""

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, Float, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, Enum
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional
import enum

from .db import Base


class JobStatus(enum.Enum):
    """Job execution status enumeration."""
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class UserRole(enum.Enum):
    """User role enumeration."""
    admin = "admin"
    analyst = "analyst"
    viewer = "viewer"
    api_client = "api_client"


class AnalysisType(enum.Enum):
    """Analysis type enumeration."""
    profile = "profile"
    comparison = "comparison"
    hashtag = "hashtag"
    trend = "trend"
    custom = "custom"


class ScrapingMethod(enum.Enum):
    """Scraping method enumeration."""
    instagrapi = "instagrapi"
    playwright = "playwright"
    public_api = "public_api"


# User Management Models
class User(Base):
    """User accounts and authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(50), unique=True, nullable=True, index=True)
    password_hash = Column(String(255), nullable=True)  # For local auth
    full_name = Column(String(100), nullable=True)
    
    # External auth
    external_id = Column(String(255), nullable=True, index=True)
    auth_provider = Column(String(50), nullable=True)  # supabase, github, google
    
    # Status and metadata
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.viewer, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    
    # Usage tracking
    total_jobs = Column(Integer, default=0, nullable=False)
    total_analyses = Column(Integer, default=0, nullable=False)
    
    # Relationships
    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_users_auth', 'auth_provider', 'external_id'),
        Index('ix_users_active', 'is_active', 'role'),
    )


class ApiKey(Base):
    """API keys for external clients."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Permissions and limits
    scopes = Column(JSON, nullable=False, default=list)  # List of allowed scopes
    rate_limit = Column(Integer, default=60, nullable=False)  # Requests per minute
    daily_quota = Column(Integer, nullable=True)  # Daily request limit
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Usage tracking
    total_requests = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    __table_args__ = (
        Index('ix_api_keys_active', 'is_active', 'expires_at'),
    )


# Job and Task Management
class Job(Base):
    """Analysis jobs and background tasks."""
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Task details
    task = Column(Text, nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.queued, nullable=False, index=True)
    progress = Column(Integer, default=0, nullable=False)
    
    # Data and results
    payload = Column(JSON, nullable=True, default=dict)  # Input parameters
    result = Column(JSON, nullable=True)  # Output results
    error = Column(JSON, nullable=True)  # Error information
    artifacts = Column(JSON, nullable=True, default=list)  # File references
    
    # Execution metadata
    priority = Column(Integer, default=0, nullable=False)
    attempts = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=3, nullable=False)
    
    # Resource usage
    execution_time_ms = Column(Integer, nullable=True)
    memory_usage_mb = Column(Integer, nullable=True)
    cost_estimate = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Scheduled jobs
    scheduled_for = Column(DateTime(timezone=True), nullable=True)
    recurring = Column(Boolean, default=False, nullable=False)
    cron_expression = Column(String(100), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="jobs")
    analysis = relationship("Analysis", back_populates="job", uselist=False)
    
    __table_args__ = (
        Index('ix_jobs_status_priority', 'status', 'priority'),
        Index('ix_jobs_scheduled', 'scheduled_for', 'status'),
        Index('ix_jobs_user_status', 'user_id', 'status'),
        CheckConstraint('progress >= 0 AND progress <= 100', name='ck_job_progress'),
    )


# Instagram Data Models
class InstagramProfile(Base):
    """Instagram profile information."""
    __tablename__ = "instagram_profiles"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(30), unique=True, nullable=False, index=True)
    
    # Profile metadata
    user_id = Column(String(50), nullable=True, index=True)  # Instagram internal ID
    full_name = Column(String(150), nullable=True)
    biography = Column(Text, nullable=True)
    
    # Metrics
    followers = Column(Integer, nullable=True, default=0)
    following = Column(Integer, nullable=True, default=0)
    posts_count = Column(Integer, nullable=True, default=0)
    
    # Status
    is_verified = Column(Boolean, default=False, nullable=False)
    is_private = Column(Boolean, default=False, nullable=False)
    is_business = Column(Boolean, default=False, nullable=False)
    
    # Media
    profile_pic_url = Column(Text, nullable=True)
    external_url = Column(Text, nullable=True)
    
    # Business info
    category = Column(String(100), nullable=True)
    contact_info = Column(JSON, nullable=True, default=dict)
    
    # Data freshness
    last_scraped_at = Column(DateTime(timezone=True), nullable=True)
    scraping_method = Column(Enum(ScrapingMethod), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    # Relationships
    posts = relationship("InstagramPost", back_populates="profile", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="profile")
    metrics = relationship("ProfileMetric", back_populates="profile", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_profiles_metrics', 'followers', 'following', 'posts_count'),
        Index('ix_profiles_status', 'is_private', 'is_verified', 'is_business'),
    )


class InstagramPost(Base):
    """Instagram post data."""
    __tablename__ = "instagram_posts"
    
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey("instagram_profiles.id"), nullable=False)
    
    # Post identifiers
    post_id = Column(String(50), unique=True, nullable=False, index=True)
    shortcode = Column(String(20), unique=True, nullable=False, index=True)
    
    # Content
    caption = Column(Text, nullable=True)
    media_type = Column(String(20), nullable=False)  # photo, video, carousel
    media_urls = Column(JSON, nullable=True, default=list)
    
    # Engagement metrics
    likes = Column(Integer, default=0, nullable=False)
    comments = Column(Integer, default=0, nullable=False)
    views = Column(Integer, nullable=True)  # For videos
    
    # Metadata
    is_video = Column(Boolean, default=False, nullable=False)
    duration = Column(Integer, nullable=True)  # Video duration in seconds
    hashtags = Column(JSON, nullable=True, default=list)
    mentions = Column(JSON, nullable=True, default=list)
    location = Column(JSON, nullable=True)
    
    # Timestamps
    posted_at = Column(DateTime(timezone=True), nullable=False, index=True)
    scraped_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    profile = relationship("InstagramProfile", back_populates="posts")
    
    __table_args__ = (
        Index('ix_posts_engagement', 'likes', 'comments'),
        Index('ix_posts_profile_date', 'profile_id', 'posted_at'),
        Index('ix_posts_media_type', 'media_type', 'is_video'),
    )


# Analysis and Insights
class Analysis(Base):
    """Analysis results and insights."""
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True, unique=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    profile_id = Column(Integer, ForeignKey("instagram_profiles.id"), nullable=True)
    
    # Analysis details
    analysis_type = Column(Enum(AnalysisType), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Configuration
    parameters = Column(JSON, nullable=True, default=dict)
    timeframe_start = Column(DateTime(timezone=True), nullable=True)
    timeframe_end = Column(DateTime(timezone=True), nullable=True)
    
    # Results
    insights = Column(JSON, nullable=True, default=list)
    metrics = Column(JSON, nullable=True, default=dict)
    recommendations = Column(JSON, nullable=True, default=list)
    
    # Metadata
    data_points = Column(Integer, default=0, nullable=False)
    confidence_score = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    
    # Processing info
    processing_time_ms = Column(Integer, nullable=True)
    llm_tokens_used = Column(Integer, nullable=True)
    cost_estimate = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    # Relationships
    job = relationship("Job", back_populates="analysis")
    user = relationship("User", back_populates="analyses")
    profile = relationship("InstagramProfile", back_populates="analyses")
    artifacts = relationship("AnalysisArtifact", back_populates="analysis", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_analyses_type_date', 'analysis_type', 'created_at'),
        Index('ix_analyses_profile', 'profile_id', 'analysis_type'),
        Index('ix_analyses_user', 'user_id', 'created_at'),
    )


class AnalysisArtifact(Base):
    """Files and artifacts associated with analyses."""
    __tablename__ = "analysis_artifacts"
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False)
    
    # File details
    filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_type = Column(String(50), nullable=False)  # screenshot, report, data, etc.
    mime_type = Column(String(100), nullable=True)
    file_size = Column(Integer, nullable=True)
    
    # Metadata
    title = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    artifact_metadata = Column(JSON, nullable=True, default=dict)
    
    # Storage
    storage_backend = Column(String(50), default="local", nullable=False)
    public_url = Column(Text, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    analysis = relationship("Analysis", back_populates="artifacts")
    
    __table_args__ = (
        Index('ix_artifacts_type', 'file_type', 'analysis_id'),
    )


# Metrics and Tracking
class ProfileMetric(Base):
    """Time-series metrics for Instagram profiles."""
    __tablename__ = "profile_metrics"
    
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey("instagram_profiles.id"), nullable=False)
    
    # Metrics snapshot
    followers = Column(Integer, nullable=False)
    following = Column(Integer, nullable=False)
    posts_count = Column(Integer, nullable=False)
    
    # Engagement metrics
    avg_likes = Column(Float, nullable=True)
    avg_comments = Column(Float, nullable=True)
    engagement_rate = Column(Float, nullable=True)
    
    # Growth metrics
    follower_growth = Column(Integer, nullable=True)
    following_growth = Column(Integer, nullable=True)
    posts_growth = Column(Integer, nullable=True)
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    
    # Relationships
    profile = relationship("InstagramProfile", back_populates="metrics")
    
    __table_args__ = (
        Index('ix_metrics_profile_date', 'profile_id', 'recorded_at'),
        UniqueConstraint('profile_id', 'recorded_at', name='uq_profile_metric_date'),
    )


class UsageMetric(Base):
    """System usage metrics and analytics."""
    __tablename__ = "usage_metrics"
    
    id = Column(Integer, primary_key=True)
    
    # Metric details
    metric_type = Column(String(50), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)
    
    # Dimensions
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    tags = Column(JSON, nullable=True, default=dict)
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('ix_usage_type_date', 'metric_type', 'recorded_at'),
        Index('ix_usage_user_type', 'user_id', 'metric_type'),
    )


# System and Audit
class AuditLog(Base):
    """Audit trail for system actions."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True)
    
    # Actor information
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"), nullable=True)
    ip_address = Column(String(45), nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    
    # Action details
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100), nullable=True)
    
    # Change details
    old_values = Column(JSON, nullable=True)
    new_values = Column(JSON, nullable=True)
    extra_metadata = Column(JSON, nullable=True, default=dict)
    
    # Request context
    request_id = Column(String(50), nullable=True, index=True)
    endpoint = Column(String(200), nullable=True)
    method = Column(String(10), nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('ix_audit_action_date', 'action', 'created_at'),
        Index('ix_audit_resource', 'resource_type', 'resource_id'),
        Index('ix_audit_user_date', 'user_id', 'created_at'),
    )


class SystemConfig(Base):
    """System configuration and feature flags."""
    __tablename__ = "system_config"
    
    id = Column(Integer, primary_key=True)
    
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(JSON, nullable=True)
    value_type = Column(String(20), nullable=False, default="string")
    
    # Metadata
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=True, index=True)
    is_sensitive = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    @validates('key')
    def validate_key(self, key, value):
        """Validate configuration key format."""
        if not value or not value.replace('_', '').replace('.', '').isalnum():
            raise ValueError("Configuration key must contain only alphanumeric characters, dots, and underscores")
        return value.lower()


# Helper functions for model operations
def create_user_with_defaults(email: str, username: Optional[str] = None, **kwargs) -> User:
    """Create a user with sensible defaults."""
    return User(
        email=email.lower(),
        username=username.lower() if username else None,
        role=kwargs.get('role', UserRole.viewer),
        is_active=kwargs.get('is_active', True),
        **{k: v for k, v in kwargs.items() if k not in ['role', 'is_active']}
    )


def create_job_with_defaults(task: str, user_id: Optional[int] = None, **kwargs) -> Job:
    """Create a job with sensible defaults."""
    return Job(
        task=task,
        user_id=user_id,
        status=kwargs.get('status', JobStatus.queued),
        priority=kwargs.get('priority', 0),
        payload=kwargs.get('payload', {}),
        **{k: v for k, v in kwargs.items() if k not in ['status', 'priority', 'payload']}
    )


def create_profile_with_defaults(username: str, **kwargs) -> InstagramProfile:
    """Create an Instagram profile with sensible defaults."""
    return InstagramProfile(
        username=username.lower(),
        followers=kwargs.get('followers', 0),
        following=kwargs.get('following', 0),
        posts_count=kwargs.get('posts_count', 0),
        **{k: v for k, v in kwargs.items() if k not in ['followers', 'following', 'posts_count']}
    )
