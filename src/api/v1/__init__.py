"""
IRAM API v1 - Modern versioned API endpoints

This module provides the v1 API endpoints with comprehensive request/response
validation, OpenAPI documentation, and proper error handling.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from ...config import get_config
from ...logging_config import get_logger
from ...middleware import create_error_handler
from ...utils import validate_instagram_username
from ..models import *
from ..dependencies import get_agent, get_current_user, check_rate_limits

logger = get_logger(__name__)

# Create API v1 router
router = APIRouter(prefix="/api/v1", tags=["API v1"])

# Health endpoints for v1
@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check_v1():
    """
    Health check endpoint for load balancers and monitoring systems.
    
    Returns basic service status and version information.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="iram-mcp-server",
        version=get_config().version
    )

@router.get("/status", response_model=StatusResponse, summary="Service Status")
async def service_status():
    """
    Comprehensive service status including capabilities and dependencies.
    
    This endpoint provides detailed information about service readiness,
    configured features, and dependency health.
    """
    config = get_config()
    from ...mcp_server import _db_ready
    
    dependencies = DependencyStatus(
        database=_db_ready if config.has_database() else True,
        llm_provider=config.has_llm_provider(),
        instagram_auth=config.has_instagram_auth() or config.instagram.public_fallback,
        redis=config.has_redis()
    )
    
    capabilities = ServiceCapabilities(
        database=config.has_database(),
        redis=config.has_redis(),
        llm_provider=config.has_llm_provider(),
        instagram_auth=config.has_instagram_auth(),
        public_fallback=config.instagram.public_fallback
    )
    
    features = FeatureFlags(
        topic_modeling=config.features.enable_topic_modeling,
        computer_vision=config.features.enable_computer_vision,
        sentiment_analysis=config.features.enable_sentiment_analysis,
        playwright=config.features.enable_playwright,
        instagrapi=config.features.enable_instagrapi,
        background_jobs=config.features.enable_background_jobs,
        scheduling=config.features.enable_scheduling,
        webhooks=config.features.enable_webhooks,
        rate_limiting=config.features.enable_rate_limiting
    )
    
    return StatusResponse(
        ready=all([
            dependencies.database,
            dependencies.llm_provider,
            dependencies.instagram_auth
        ]),
        dependencies=dependencies,
        capabilities=capabilities,
        features=features,
        environment=config.environment,
        version=config.version,
        timestamp=datetime.utcnow()
    )

# Analysis endpoints
@router.post("/analysis/execute", response_model=TaskResponse, summary="Execute Analysis Task")
async def execute_analysis_task(
    request: TaskRequest,
    agent = Depends(get_agent),
    current_user = Depends(get_current_user),
    _rate_limit = Depends(check_rate_limits)
):
    """
    Execute a high-level Instagram research and analysis task.
    
    This endpoint accepts natural language task descriptions and executes
    them using the IRAM agent orchestrator. Tasks can include profile analysis,
    content research, trend identification, and comparative studies.
    
    **Examples:**
    - "Analyze @username's recent posts for engagement patterns"
    - "Compare follower growth between @user1 and @user2 over the last 30 days" 
    - "Research trending hashtags in the fitness niche"
    """
    try:
        logger.info(
            f"Executing analysis task for user {current_user.id if current_user else 'anonymous'}",
            extra={
                "task": request.task,
                "user_id": current_user.id if current_user else None,
                "context_keys": list(request.context.keys()) if request.context else []
            }
        )
        
        result = await agent.execute_task(request.task, request.context)
        
        return TaskResponse(
            success=result.get("success", False),
            task_id=result.get("task_id"),
            result=result.get("result"),
            error=result.get("error"),
            metadata={
                "execution_time_ms": result.get("execution_time_ms"),
                "tokens_used": result.get("tokens_used"),
                "cost_estimate": result.get("cost_estimate")
            },
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Analysis task execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")

@router.post("/analysis/schedule", response_model=JobResponse, summary="Schedule Background Analysis")
async def schedule_analysis_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    agent = Depends(get_agent),
    current_user = Depends(get_current_user),
    _rate_limit = Depends(check_rate_limits)
):
    """
    Schedule an analysis task for background execution.
    
    Use this endpoint for long-running or resource-intensive analysis tasks
    that should not block the API response. Returns a job ID that can be used
    to track progress and retrieve results.
    """
    from ...mcp_server import _db_ready
    
    if not _db_ready:
        raise HTTPException(status_code=503, detail="Background job scheduling requires database")
    
    try:
        from ...db import session_scope
        from ...models import Job
        
        # Create job record
        async with session_scope() as session:
            job = Job(
                task=request.task,
                status="queued",
                progress=0,
                payload=request.context or {},
                user_id=current_user.id if current_user else None
            )
            session.add(job)
            await session.flush()
            job_id = job.id
        
        # Schedule background execution
        def run_background_task(job_id: int):
            import asyncio
            try:
                # Execute task
                result = asyncio.run(agent.execute_task(request.task, request.context))
                
                # Update job with result
                async def update_job():
                    async with session_scope() as session:
                        db_job = await session.get(Job, job_id)
                        if db_job:
                            db_job.status = "completed" if result.get("success") else "failed"
                            db_job.progress = 100
                            db_job.result = result
                
                asyncio.run(update_job())
                logger.info(f"Background job {job_id} completed successfully")
                
            except Exception as e:
                # Update job with error
                async def fail_job():
                    async with session_scope() as session:
                        db_job = await session.get(Job, job_id)
                        if db_job:
                            db_job.status = "failed"
                            db_job.error = {"message": str(e), "type": type(e).__name__}
                
                asyncio.run(fail_job())
                logger.error(f"Background job {job_id} failed: {e}", exc_info=True)
        
        background_tasks.add_task(run_background_task, job_id)
        
        return JobResponse(
            job_id=job_id,
            status="queued",
            progress=0,
            task=request.task,
            created_at=datetime.utcnow(),
            message="Task scheduled for background execution"
        )
        
    except Exception as e:
        logger.error(f"Failed to schedule background task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to schedule task")

# Profile endpoints  
@router.post("/profiles/analyze", response_model=ProfileAnalysisResponse, summary="Analyze Instagram Profile")
async def analyze_profile(
    request: ProfileAnalysisRequest,
    agent = Depends(get_agent),
    current_user = Depends(get_current_user),
    _rate_limit = Depends(check_rate_limits)
):
    """
    Perform comprehensive analysis of an Instagram profile.
    
    This endpoint fetches profile data, recent posts, and performs AI-powered
    analysis including sentiment analysis, engagement patterns, content themes,
    and growth trends.
    """
    if not validate_instagram_username(request.username):
        raise HTTPException(status_code=400, detail="Invalid Instagram username format")
    
    try:
        # Build analysis task
        task_parts = [f"Analyze Instagram profile @{request.username}"]
        
        context = {
            "username": request.username,
            "include_posts": request.include_posts,
            "post_limit": request.post_limit,
            "analysis_depth": request.analysis_depth,
            "include_engagement": request.include_engagement,
            "include_content_analysis": request.include_content_analysis,
            "timeframe_days": request.timeframe_days
        }
        
        if request.include_posts:
            task_parts.append(f"Include analysis of last {request.post_limit} posts")
        
        if request.include_engagement:
            task_parts.append("Calculate engagement metrics and trends")
        
        if request.include_content_analysis:
            task_parts.append("Perform content theme and sentiment analysis")
        
        task = ". ".join(task_parts)
        
        logger.info(
            f"Starting profile analysis for @{request.username}",
            extra={
                "username": request.username,
                "analysis_depth": request.analysis_depth,
                "user_id": current_user.id if current_user else None
            }
        )
        
        result = await agent.execute_task(task, context)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Analysis failed"))
        
        analysis_data = result.get("result", {})
        
        return ProfileAnalysisResponse(
            username=request.username,
            profile_data=analysis_data.get("profile", {}),
            posts=analysis_data.get("posts", []),
            engagement_metrics=analysis_data.get("engagement", {}),
            content_analysis=analysis_data.get("content_analysis", {}),
            insights=analysis_data.get("insights", []),
            metadata={
                "analysis_depth": request.analysis_depth,
                "posts_analyzed": len(analysis_data.get("posts", [])),
                "execution_time_ms": result.get("execution_time_ms"),
                "data_freshness": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile analysis failed for @{request.username}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Profile analysis failed: {str(e)}")

@router.post("/profiles/compare", response_model=ProfileComparisonResponse, summary="Compare Instagram Profiles")
async def compare_profiles(
    request: ProfileComparisonRequest,
    agent = Depends(get_agent),
    current_user = Depends(get_current_user),
    _rate_limit = Depends(check_rate_limits)
):
    """
    Compare multiple Instagram profiles across various metrics.
    
    This endpoint analyzes multiple profiles and provides comparative insights
    including follower growth, engagement rates, content strategies, and
    competitive positioning.
    """
    # Validate usernames
    for username in request.usernames:
        if not validate_instagram_username(username):
            raise HTTPException(status_code=400, detail=f"Invalid username: {username}")
    
    if len(request.usernames) < 2:
        raise HTTPException(status_code=400, detail="At least 2 profiles required for comparison")
    
    if len(request.usernames) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 profiles allowed for comparison")
    
    try:
        context = {
            "usernames": request.usernames,
            "metrics": request.metrics,
            "timeframe_days": request.timeframe_days,
            "include_content_analysis": request.include_content_analysis
        }
        
        task = f"Compare Instagram profiles: {', '.join(['@' + u for u in request.usernames])}"
        if request.metrics:
            task += f". Focus on metrics: {', '.join(request.metrics)}"
        
        logger.info(
            f"Starting profile comparison",
            extra={
                "usernames": request.usernames,
                "metrics": request.metrics,
                "user_id": current_user.id if current_user else None
            }
        )
        
        result = await agent.execute_task(task, context)
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Comparison failed"))
        
        comparison_data = result.get("result", {})
        
        return ProfileComparisonResponse(
            usernames=request.usernames,
            profiles=comparison_data.get("profiles", {}),
            comparison_metrics=comparison_data.get("metrics", {}),
            insights=comparison_data.get("insights", []),
            recommendations=comparison_data.get("recommendations", []),
            metadata={
                "profiles_compared": len(request.usernames),
                "metrics_analyzed": request.metrics,
                "timeframe_days": request.timeframe_days,
                "execution_time_ms": result.get("execution_time_ms")
            },
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Profile comparison failed: {str(e)}")

# Job management endpoints
@router.get("/jobs", response_model=JobListResponse, summary="List Analysis Jobs")
async def list_jobs(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    status: Optional[str] = Query(None, description="Filter by job status"),
    current_user = Depends(get_current_user)
):
    """
    List analysis jobs with pagination and filtering.
    
    Returns a paginated list of analysis jobs, optionally filtered by status.
    Jobs are ordered by creation time (most recent first).
    """
    from ...mcp_server import _db_ready
    
    if not _db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        from ...db import session_scope
        from ...models import Job
        from sqlalchemy import desc, and_
        
        async with session_scope() as session:
            # Build query
            query = session.query(Job)
            
            # Filter by user if authenticated
            if current_user:
                query = query.filter(Job.user_id == current_user.id)
            
            # Filter by status
            if status:
                query = query.filter(Job.status == status)
            
            # Apply pagination
            jobs = await query.order_by(desc(Job.created_at)).limit(limit).offset(offset).all()
            
            # Get total count
            count_query = session.query(Job)
            if current_user:
                count_query = count_query.filter(Job.user_id == current_user.id)
            if status:
                count_query = count_query.filter(Job.status == status)
            
            total_count = await count_query.count()
            
            job_list = []
            for job in jobs:
                job_list.append({
                    "job_id": job.id,
                    "task": job.task,
                    "status": job.status,
                    "progress": job.progress,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at,
                    "user_id": job.user_id
                })
            
            return JobListResponse(
                jobs=job_list,
                total=total_count,
                limit=limit,
                offset=offset,
                timestamp=datetime.utcnow()
            )
            
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve jobs")

@router.get("/jobs/{job_id}", response_model=JobDetailResponse, summary="Get Job Details")
async def get_job_details(
    job_id: int = Path(..., description="Job ID"),
    current_user = Depends(get_current_user)
):
    """
    Get detailed information about a specific analysis job.
    
    Returns comprehensive job information including status, progress,
    results, and any error details.
    """
    from ...mcp_server import _db_ready
    
    if not _db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        from ...db import session_scope
        from ...models import Job
        
        async with session_scope() as session:
            job = await session.get(Job, job_id)
            
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            
            # Check access permissions
            if current_user and job.user_id != current_user.id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            return JobDetailResponse(
                job_id=job.id,
                task=job.task,
                status=job.status,
                progress=job.progress,
                payload=job.payload,
                result=job.result,
                error=job.error,
                created_at=job.created_at,
                updated_at=job.updated_at,
                user_id=job.user_id,
                timestamp=datetime.utcnow()
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve job details")