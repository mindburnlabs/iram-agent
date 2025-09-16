"""
IRAM API Models

Comprehensive Pydantic models for request/response validation,
OpenAPI documentation, and type safety across all API endpoints.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Job execution status enumeration."""
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class AnalysisDepth(str, Enum):
    """Analysis depth levels."""
    basic = "basic"
    standard = "standard"
    comprehensive = "comprehensive"


class MetricType(str, Enum):
    """Profile comparison metric types."""
    engagement = "engagement"
    growth = "growth"
    content = "content"
    audience = "audience"
    activity = "activity"


# Base Models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + 'Z'
        }


class ErrorResponse(BaseResponse):
    """Standard error response model."""
    error: bool = Field(True, description="Error flag")
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    trace_id: Optional[str] = Field(None, description="Request trace ID for debugging")
    remediation: Optional[str] = Field(None, description="Suggested remediation steps")


# Health and Status Models
class HealthResponse(BaseResponse):
    """Health check response."""
    status: str = Field("healthy", description="Service health status")
    service: str = Field("iram-mcp-server", description="Service identifier")
    version: str = Field(..., description="Service version")


class DependencyStatus(BaseModel):
    """Dependency health status."""
    database: bool = Field(..., description="Database connectivity")
    llm_provider: bool = Field(..., description="LLM provider availability")
    instagram_auth: bool = Field(..., description="Instagram authentication status")
    redis: bool = Field(False, description="Redis cache availability")


class ServiceCapabilities(BaseModel):
    """Service capabilities and features."""
    database: bool = Field(..., description="Database storage available")
    redis: bool = Field(..., description="Redis caching available")
    llm_provider: bool = Field(..., description="LLM analysis available")
    instagram_auth: bool = Field(..., description="Authenticated Instagram access")
    public_fallback: bool = Field(..., description="Public Instagram access fallback")


class FeatureFlags(BaseModel):
    """Feature toggle status."""
    topic_modeling: bool = Field(..., description="Topic modeling analysis")
    computer_vision: bool = Field(..., description="Image/video analysis")
    sentiment_analysis: bool = Field(..., description="Content sentiment analysis")
    playwright: bool = Field(..., description="Browser-based scraping")
    instagrapi: bool = Field(..., description="API-based Instagram access")
    background_jobs: bool = Field(..., description="Asynchronous job processing")
    scheduling: bool = Field(..., description="Scheduled analysis tasks")
    webhooks: bool = Field(..., description="Webhook notifications")
    rate_limiting: bool = Field(..., description="Request rate limiting")


class StatusResponse(BaseResponse):
    """Comprehensive service status."""
    ready: bool = Field(..., description="Overall service readiness")
    dependencies: DependencyStatus = Field(..., description="Dependency health status")
    capabilities: ServiceCapabilities = Field(..., description="Available service capabilities")
    features: FeatureFlags = Field(..., description="Enabled features")
    environment: str = Field(..., description="Deployment environment")
    version: str = Field(..., description="Service version")


# Request Models
class TaskRequest(BaseModel):
    """Generic task execution request."""
    task: str = Field(
        ..., 
        min_length=10,
        max_length=2000,
        description="Natural language task description",
        example="Analyze @username's recent posts for engagement patterns"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context and parameters for the task"
    )
    
    @validator('task')
    def validate_task(cls, v):
        if not v.strip():
            raise ValueError("Task description cannot be empty")
        return v.strip()


class ProfileAnalysisRequest(BaseModel):
    """Instagram profile analysis request."""
    username: str = Field(
        ...,
        min_length=1,
        max_length=30,
        regex=r'^[a-zA-Z0-9._]+$',
        description="Instagram username (without @ symbol)",
        example="example_user"
    )
    include_posts: bool = Field(
        True,
        description="Include recent posts in analysis"
    )
    post_limit: int = Field(
        50,
        ge=1,
        le=200,
        description="Maximum number of posts to analyze"
    )
    analysis_depth: AnalysisDepth = Field(
        AnalysisDepth.standard,
        description="Depth of analysis to perform"
    )
    include_engagement: bool = Field(
        True,
        description="Calculate engagement metrics and trends"
    )
    include_content_analysis: bool = Field(
        True,
        description="Perform AI-powered content analysis"
    )
    timeframe_days: int = Field(
        30,
        ge=1,
        le=365,
        description="Analysis timeframe in days"
    )


class ProfileComparisonRequest(BaseModel):
    """Profile comparison request."""
    usernames: List[str] = Field(
        ...,
        min_items=2,
        max_items=10,
        description="List of usernames to compare"
    )
    metrics: Optional[List[MetricType]] = Field(
        None,
        description="Specific metrics to focus on in comparison"
    )
    timeframe_days: int = Field(
        30,
        ge=1,
        le=365,
        description="Comparison timeframe in days"
    )
    include_content_analysis: bool = Field(
        True,
        description="Include content theme analysis in comparison"
    )
    
    @validator('usernames')
    def validate_usernames(cls, v):
        for username in v:
            if not username.strip():
                raise ValueError("Username cannot be empty")
            if not username.replace('.', '').replace('_', '').isalnum():
                raise ValueError(f"Invalid username format: {username}")
        return [u.strip().lower() for u in v]


class HashtagAnalysisRequest(BaseModel):
    """Hashtag trend analysis request."""
    hashtags: List[str] = Field(
        ...,
        min_items=1,
        max_items=20,
        description="List of hashtags to analyze (without # symbol)"
    )
    timeframe_days: int = Field(
        7,
        ge=1,
        le=90,
        description="Analysis timeframe in days"
    )
    include_related: bool = Field(
        True,
        description="Include related hashtags in analysis"
    )
    limit: int = Field(
        100,
        ge=10,
        le=1000,
        description="Maximum number of posts to analyze per hashtag"
    )
    
    @validator('hashtags')
    def validate_hashtags(cls, v):
        clean_tags = []
        for tag in v:
            # Remove # if present
            clean_tag = tag.strip().lstrip('#').lower()
            if not clean_tag:
                raise ValueError("Hashtag cannot be empty")
            if not clean_tag.replace('_', '').isalnum():
                raise ValueError(f"Invalid hashtag format: {tag}")
            clean_tags.append(clean_tag)
        return clean_tags


# Response Models
class TaskResponse(BaseResponse):
    """Task execution response."""
    success: bool = Field(..., description="Task execution success status")
    task_id: Optional[str] = Field(None, description="Unique task identifier")
    result: Optional[Dict[str, Any]] = Field(None, description="Task execution results")
    error: Optional[str] = Field(None, description="Error message if task failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional execution metadata")


class JobResponse(BaseResponse):
    """Job creation/status response."""
    job_id: int = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: int = Field(
        0,
        ge=0,
        le=100,
        description="Job completion percentage"
    )
    task: str = Field(..., description="Task description")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    message: Optional[str] = Field(None, description="Status message")


class InstagramPost(BaseModel):
    """Instagram post data model."""
    id: str = Field(..., description="Post ID")
    shortcode: str = Field(..., description="Post shortcode")
    caption: Optional[str] = Field(None, description="Post caption")
    media_type: str = Field(..., description="Media type (photo, video, carousel)")
    likes: int = Field(..., ge=0, description="Number of likes")
    comments: int = Field(..., ge=0, description="Number of comments")
    timestamp: datetime = Field(..., description="Post creation timestamp")
    media_url: Optional[str] = Field(None, description="Media URL")
    is_video: bool = Field(False, description="Whether post contains video")
    hashtags: List[str] = Field(default_factory=list, description="Extracted hashtags")
    mentions: List[str] = Field(default_factory=list, description="Extracted mentions")


class EngagementMetrics(BaseModel):
    """Profile engagement metrics."""
    engagement_rate: float = Field(..., ge=0, le=1, description="Overall engagement rate")
    avg_likes: float = Field(..., ge=0, description="Average likes per post")
    avg_comments: float = Field(..., ge=0, description="Average comments per post")
    best_posting_times: List[str] = Field(
        default_factory=list,
        description="Optimal posting hours"
    )
    engagement_trend: str = Field(..., description="Engagement trend direction")
    top_performing_posts: List[InstagramPost] = Field(
        default_factory=list,
        max_items=5,
        description="Top performing posts by engagement"
    )


class ContentAnalysis(BaseModel):
    """Content analysis results."""
    dominant_themes: List[str] = Field(
        default_factory=list,
        description="Primary content themes"
    )
    sentiment_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Sentiment analysis breakdown"
    )
    content_categories: Dict[str, int] = Field(
        default_factory=dict,
        description="Content category distribution"
    )
    avg_sentiment_score: float = Field(
        0.0,
        ge=-1,
        le=1,
        description="Average sentiment score"
    )
    hashtag_performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Hashtag engagement performance"
    )
    language_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Content language distribution"
    )


class ProfileData(BaseModel):
    """Instagram profile information."""
    username: str = Field(..., description="Instagram username")
    full_name: Optional[str] = Field(None, description="Full display name")
    biography: Optional[str] = Field(None, description="Profile biography")
    followers: int = Field(..., ge=0, description="Follower count")
    following: int = Field(..., ge=0, description="Following count") 
    posts_count: int = Field(..., ge=0, description="Total posts count")
    is_verified: bool = Field(False, description="Verification status")
    is_private: bool = Field(False, description="Privacy status")
    profile_pic_url: Optional[str] = Field(None, description="Profile picture URL")
    external_url: Optional[str] = Field(None, description="External website URL")
    category: Optional[str] = Field(None, description="Business category")
    contact_info: Dict[str, str] = Field(
        default_factory=dict,
        description="Contact information"
    )


class ProfileAnalysisResponse(BaseResponse):
    """Profile analysis response."""
    username: str = Field(..., description="Analyzed username")
    profile_data: ProfileData = Field(..., description="Profile information")
    posts: List[InstagramPost] = Field(
        default_factory=list,
        description="Recent posts analyzed"
    )
    engagement_metrics: EngagementMetrics = Field(..., description="Engagement analysis")
    content_analysis: ContentAnalysis = Field(..., description="Content analysis results")
    insights: List[str] = Field(
        default_factory=list,
        description="AI-generated insights and observations"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis metadata and execution info"
    )


class ProfileComparisonResponse(BaseResponse):
    """Profile comparison response."""
    usernames: List[str] = Field(..., description="Compared usernames")
    profiles: Dict[str, ProfileData] = Field(..., description="Profile data for each user")
    comparison_metrics: Dict[str, Any] = Field(
        ...,
        description="Comparative metrics and statistics"
    )
    insights: List[str] = Field(
        default_factory=list,
        description="Comparison insights and observations"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Strategic recommendations based on comparison"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Comparison metadata and execution info"
    )


class HashtagTrend(BaseModel):
    """Hashtag trend data."""
    hashtag: str = Field(..., description="Hashtag name")
    post_count: int = Field(..., ge=0, description="Number of posts")
    engagement_rate: float = Field(..., ge=0, description="Average engagement rate")
    trend_direction: str = Field(..., description="Trend direction (up, down, stable)")
    related_hashtags: List[str] = Field(
        default_factory=list,
        description="Related hashtags"
    )


class HashtagAnalysisResponse(BaseResponse):
    """Hashtag analysis response."""
    hashtags: List[str] = Field(..., description="Analyzed hashtags")
    trends: List[HashtagTrend] = Field(..., description="Hashtag trend data")
    top_posts: List[InstagramPost] = Field(
        default_factory=list,
        description="Top performing posts for analyzed hashtags"
    )
    insights: List[str] = Field(
        default_factory=list,
        description="Trend insights and observations"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Hashtag strategy recommendations"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis metadata"
    )


# Job Management Models
class JobSummary(BaseModel):
    """Job summary for list responses."""
    job_id: int = Field(..., description="Job identifier")
    task: str = Field(..., description="Task description")
    status: JobStatus = Field(..., description="Job status")
    progress: int = Field(..., ge=0, le=100, description="Completion percentage")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    user_id: Optional[int] = Field(None, description="Owner user ID")


class JobListResponse(BaseResponse):
    """Job list response with pagination."""
    jobs: List[JobSummary] = Field(..., description="List of jobs")
    total: int = Field(..., ge=0, description="Total number of jobs")
    limit: int = Field(..., ge=1, description="Results limit")
    offset: int = Field(..., ge=0, description="Results offset")


class JobDetailResponse(BaseResponse):
    """Detailed job information."""
    job_id: int = Field(..., description="Job identifier")
    task: str = Field(..., description="Task description")
    status: JobStatus = Field(..., description="Job status")
    progress: int = Field(..., ge=0, le=100, description="Completion percentage")
    payload: Optional[Dict[str, Any]] = Field(None, description="Task parameters")
    result: Optional[Dict[str, Any]] = Field(None, description="Task results")
    error: Optional[Dict[str, Any]] = Field(None, description="Error information")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    user_id: Optional[int] = Field(None, description="Owner user ID")


# User and Authentication Models
class User(BaseModel):
    """User model for authentication."""
    id: int = Field(..., description="User ID")
    email: Optional[str] = Field(None, description="User email")
    username: Optional[str] = Field(None, description="Username")
    is_active: bool = Field(True, description="Account active status")
    created_at: datetime = Field(..., description="Account creation timestamp")
    
    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """Login request model."""
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=6, description="User password")


class LoginResponse(BaseResponse):
    """Login response model."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiry in seconds")
    user: User = Field(..., description="User information")


# Configuration Models
class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(..., description="Requests per minute limit")
    burst_limit: int = Field(..., description="Burst requests limit")
    max_request_size: int = Field(..., description="Maximum request size in bytes")


class ConfigResponse(BaseResponse):
    """Service configuration response."""
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Deployment environment")
    debug: bool = Field(..., description="Debug mode status")
    capabilities: ServiceCapabilities = Field(..., description="Service capabilities")
    features: FeatureFlags = Field(..., description="Feature flags")
    limits: RateLimitConfig = Field(..., description="Rate limiting configuration")


# Metrics Models
class ServiceMetrics(BaseModel):
    """Service performance metrics."""
    uptime_seconds: float = Field(..., ge=0, description="Service uptime in seconds")
    total_requests: int = Field(..., ge=0, description="Total requests served")
    error_rate: float = Field(..., ge=0, le=1, description="Error rate percentage")
    avg_response_time_ms: float = Field(..., ge=0, description="Average response time")
    active_connections: int = Field(..., ge=0, description="Active connections")


class JobMetrics(BaseModel):
    """Job execution metrics."""
    total_jobs: int = Field(..., ge=0, description="Total jobs created")
    jobs_by_status: Dict[str, int] = Field(..., description="Job count by status")
    avg_execution_time_ms: float = Field(..., ge=0, description="Average execution time")
    success_rate: float = Field(..., ge=0, le=1, description="Job success rate")


class MetricsResponse(BaseResponse):
    """Service metrics response."""
    service: ServiceMetrics = Field(..., description="Service performance metrics")
    jobs: Optional[JobMetrics] = Field(None, description="Job execution metrics")
    environment: str = Field(..., description="Environment")
    version: str = Field(..., description="Service version")