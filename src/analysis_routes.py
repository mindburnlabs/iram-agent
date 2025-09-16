"""
Analysis API Routes

RESTful API endpoints for Instagram analysis with intelligent caching,
ML model results, and comprehensive insights.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from .analysis_service import (
    get_analysis_service, EnhancedAnalysisService,
    AnalysisType, AnalysisStatus, CacheStrategy, AnalysisResult
)
from .auth import get_current_active_user, require_permission, Permission, User
from .logging_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/analysis", tags=["analysis"])


# Pydantic models
class AnalysisRequest(BaseModel):
    """Analysis request model."""
    analysis_types: List[AnalysisType] = Field(..., min_items=1)
    force_refresh: bool = False
    include_metadata: bool = True


class SentimentAnalysisResponse(BaseModel):
    """Sentiment analysis response model."""
    overall_sentiment: float
    sentiment_classification: str
    confidence: float
    texts_analyzed: int
    sentiment_distribution: Dict[str, int]
    created_at: str
    cache_strategy: str


class EngagementAnalysisResponse(BaseModel):
    """Engagement analysis response model."""
    engagement_rate: float
    average_likes: float
    average_comments: float
    total_interactions: int
    posts_analyzed: int
    engagement_trend: str
    peak_engagement_hours: List[int]
    top_performing_content_types: List[str]
    created_at: str
    cache_strategy: str


class HashtagAnalysisResponse(BaseModel):
    """Hashtag analysis response model."""
    total_hashtags_found: int
    average_hashtags_per_post: float
    top_performing_hashtags: Dict[str, Dict[str, Any]]
    hashtag_diversity_score: float
    recommended_hashtags: List[str]
    created_at: str
    cache_strategy: str


class AnalysisJobResponse(BaseModel):
    """Analysis job response model."""
    job_id: str
    username: str
    analysis_types: List[str]
    status: AnalysisStatus
    created_at: str
    estimated_completion: Optional[str] = None
    progress: Optional[float] = None


class AnalysisSummaryResponse(BaseModel):
    """Analysis summary response model."""
    username: str
    analyses: Dict[str, Dict[str, Any]]
    cache_stats: Dict[str, Any]


# Individual analysis endpoints

@router.get("/profiles/{username}/sentiment", response_model=SentimentAnalysisResponse)
async def get_sentiment_analysis(
    username: str,
    force_refresh: bool = Query(False, description="Force refresh analysis"),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYSIS))
):
    """Get sentiment analysis for a profile."""
    try:
        analysis_service = await get_analysis_service()
        result = await analysis_service.analyze_profile_sentiment(
            username, 
            force_refresh=force_refresh
        )
        
        logger.info(f"Sentiment analysis retrieved for {username} by user {current_user.email}", extra={
            "username": username,
            "user_id": current_user.id,
            "from_cache": not force_refresh
        })
        
        return SentimentAnalysisResponse(
            **result.data,
            created_at=result.created_at.isoformat(),
            cache_strategy=result.cache_strategy.value
        )
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed for {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/{username}/engagement", response_model=EngagementAnalysisResponse)
async def get_engagement_analysis(
    username: str,
    force_refresh: bool = Query(False, description="Force refresh analysis"),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYSIS))
):
    """Get engagement analysis for a profile."""
    try:
        analysis_service = await get_analysis_service()
        result = await analysis_service.analyze_content_engagement(
            username,
            force_refresh=force_refresh
        )
        
        logger.info(f"Engagement analysis retrieved for {username} by user {current_user.email}")
        
        return EngagementAnalysisResponse(
            **result.data,
            created_at=result.created_at.isoformat(),
            cache_strategy=result.cache_strategy.value
        )
        
    except Exception as e:
        logger.error(f"Engagement analysis failed for {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/{username}/hashtags", response_model=HashtagAnalysisResponse)
async def get_hashtag_analysis(
    username: str,
    force_refresh: bool = Query(False, description="Force refresh analysis"),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYSIS))
):
    """Get hashtag analysis for a profile."""
    try:
        analysis_service = await get_analysis_service()
        result = await analysis_service.analyze_hashtag_performance(
            username,
            force_refresh=force_refresh
        )
        
        logger.info(f"Hashtag analysis retrieved for {username} by user {current_user.email}")
        
        return HashtagAnalysisResponse(
            **result.data,
            created_at=result.created_at.isoformat(),
            cache_strategy=result.cache_strategy.value
        )
        
    except Exception as e:
        logger.error(f"Hashtag analysis failed for {username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Comprehensive analysis endpoints

@router.post("/profiles/{username}/comprehensive")
async def run_comprehensive_analysis(
    username: str,
    analysis_request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.CREATE_ANALYSIS))
):
    """Run comprehensive analysis with multiple analysis types."""
    try:
        job_id = f"analysis_{username}_{current_user.id}_{int(time.time())}"
        
        # Start background analysis
        background_tasks.add_task(
            _run_comprehensive_analysis_background,
            job_id,
            username,
            analysis_request.analysis_types,
            analysis_request.force_refresh,
            current_user.id
        )
        
        logger.info(f"Comprehensive analysis started for {username} by user {current_user.email}", extra={
            "username": username,
            "user_id": current_user.id,
            "analysis_types": [at.value for at in analysis_request.analysis_types],
            "job_id": job_id
        })
        
        return AnalysisJobResponse(
            job_id=job_id,
            username=username,
            analysis_types=[at.value for at in analysis_request.analysis_types],
            status=AnalysisStatus.PENDING,
            created_at=datetime.utcnow().isoformat(),
            estimated_completion=(datetime.utcnow() + timedelta(minutes=5)).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to start comprehensive analysis for {username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start analysis")


async def _run_comprehensive_analysis_background(
    job_id: str,
    username: str,
    analysis_types: List[AnalysisType],
    force_refresh: bool,
    user_id: int
):
    """Background task for comprehensive analysis."""
    import time
    from datetime import datetime, timedelta
    
    try:
        analysis_service = await get_analysis_service()
        results = {}
        
        logger.info(f"Starting comprehensive analysis job {job_id} for {username}")
        
        for analysis_type in analysis_types:
            try:
                if analysis_type == AnalysisType.PROFILE_SENTIMENT:
                    result = await analysis_service.analyze_profile_sentiment(
                        username, force_refresh=force_refresh
                    )
                elif analysis_type == AnalysisType.ENGAGEMENT_ANALYSIS:
                    result = await analysis_service.analyze_content_engagement(
                        username, force_refresh=force_refresh
                    )
                elif analysis_type == AnalysisType.HASHTAG_ANALYSIS:
                    result = await analysis_service.analyze_hashtag_performance(
                        username, force_refresh=force_refresh
                    )
                else:
                    logger.warning(f"Analysis type {analysis_type.value} not implemented yet")
                    continue
                
                results[analysis_type.value] = result.data
                
                # Add small delay between analyses
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Analysis {analysis_type.value} failed for {username}: {e}")
                results[analysis_type.value] = {"error": str(e)}
        
        logger.info(f"Comprehensive analysis job {job_id} completed for {username}")
        
        # TODO: Store job results in database or cache for retrieval
        
    except Exception as e:
        logger.error(f"Comprehensive analysis job {job_id} failed: {e}")


@router.get("/profiles/{username}/summary", response_model=AnalysisSummaryResponse)
async def get_analysis_summary(
    username: str,
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYSIS))
):
    """Get summary of all available analyses for a profile."""
    try:
        analysis_service = await get_analysis_service()
        summary = await analysis_service.get_analysis_summary(username)
        
        logger.debug(f"Analysis summary retrieved for {username} by user {current_user.email}")
        
        return AnalysisSummaryResponse(**summary)
        
    except Exception as e:
        logger.error(f"Failed to get analysis summary for {username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analysis summary")


# Cache management endpoints

@router.delete("/profiles/{username}/cache")
async def invalidate_profile_analyses(
    username: str,
    current_user: User = Depends(require_permission(Permission.CREATE_ANALYSIS))
):
    """Invalidate all cached analyses for a profile."""
    try:
        analysis_service = await get_analysis_service()
        invalidated_count = await analysis_service.invalidate_user_analyses(username)
        
        logger.info(f"Invalidated {invalidated_count} analyses for {username} by user {current_user.email}")
        
        return {
            "message": f"Invalidated {invalidated_count} cached analyses for @{username}",
            "invalidated_count": invalidated_count
        }
        
    except Exception as e:
        logger.error(f"Failed to invalidate analyses for {username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to invalidate cache")


@router.delete("/cache/analysis-type/{analysis_type}")
async def invalidate_analysis_type_cache(
    analysis_type: AnalysisType,
    current_user: User = Depends(require_permission(Permission.MANAGE_SYSTEM))
):
    """Invalidate all cached analyses of a specific type (admin only)."""
    try:
        analysis_service = await get_analysis_service()
        invalidated_count = await analysis_service.invalidate_analysis_type(analysis_type)
        
        logger.info(f"Invalidated {invalidated_count} {analysis_type.value} analyses by user {current_user.email}")
        
        return {
            "message": f"Invalidated {invalidated_count} {analysis_type.value} analyses",
            "analysis_type": analysis_type.value,
            "invalidated_count": invalidated_count
        }
        
    except Exception as e:
        logger.error(f"Failed to invalidate {analysis_type.value} analyses: {e}")
        raise HTTPException(status_code=500, detail="Failed to invalidate cache")


@router.get("/cache/stats")
async def get_analysis_cache_stats(
    current_user: User = Depends(require_permission(Permission.VIEW_SYSTEM))
):
    """Get analysis cache statistics."""
    try:
        analysis_service = await get_analysis_service()
        stats = await analysis_service.get_cache_statistics()
        
        return {
            "service": "analysis",
            "cache_statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get analysis cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache statistics")


# Batch analysis endpoints

@router.post("/profiles/batch/sentiment")
async def batch_sentiment_analysis(
    usernames: List[str] = Field(..., min_items=1, max_items=10),
    force_refresh: bool = False,
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYSIS))
):
    """Run sentiment analysis for multiple profiles."""
    try:
        analysis_service = await get_analysis_service()
        results = {}
        errors = []
        
        for username in usernames:
            try:
                result = await analysis_service.analyze_profile_sentiment(
                    username, force_refresh=force_refresh
                )
                results[username] = {
                    **result.data,
                    "created_at": result.created_at.isoformat(),
                    "cache_strategy": result.cache_strategy.value
                }
                
            except Exception as e:
                errors.append({
                    "username": username,
                    "error": str(e)
                })
        
        logger.info(f"Batch sentiment analysis completed for {len(usernames)} profiles by user {current_user.email}", extra={
            "requested_count": len(usernames),
            "success_count": len(results),
            "error_count": len(errors)
        })
        
        return {
            "results": results,
            "errors": errors,
            "summary": {
                "requested": len(usernames),
                "successful": len(results),
                "failed": len(errors)
            }
        }
        
    except Exception as e:
        logger.error(f"Batch sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Batch analysis failed")


# Analysis comparison endpoints

@router.post("/profiles/compare")
async def compare_profiles(
    usernames: List[str] = Field(..., min_items=2, max_items=5),
    analysis_types: List[AnalysisType] = Field(default=[AnalysisType.ENGAGEMENT_ANALYSIS]),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYSIS))
):
    """Compare analysis results across multiple profiles."""
    try:
        analysis_service = await get_analysis_service()
        comparison_data = {}
        
        for username in usernames:
            user_analyses = {}
            
            for analysis_type in analysis_types:
                try:
                    if analysis_type == AnalysisType.PROFILE_SENTIMENT:
                        result = await analysis_service.analyze_profile_sentiment(username)
                    elif analysis_type == AnalysisType.ENGAGEMENT_ANALYSIS:
                        result = await analysis_service.analyze_content_engagement(username)
                    elif analysis_type == AnalysisType.HASHTAG_ANALYSIS:
                        result = await analysis_service.analyze_hashtag_performance(username)
                    else:
                        continue
                    
                    user_analyses[analysis_type.value] = result.data
                    
                except Exception as e:
                    user_analyses[analysis_type.value] = {"error": str(e)}
            
            comparison_data[username] = user_analyses
        
        logger.info(f"Profile comparison completed for {len(usernames)} profiles by user {current_user.email}")
        
        return {
            "comparison": comparison_data,
            "profiles": usernames,
            "analysis_types": [at.value for at in analysis_types],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Profile comparison failed: {e}")
        raise HTTPException(status_code=500, detail="Profile comparison failed")


# Export analysis data

@router.get("/profiles/{username}/export")
async def export_analysis_data(
    username: str,
    format: str = Query("json", regex="^(json|csv)$"),
    include_metadata: bool = True,
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYSIS))
):
    """Export all analysis data for a profile."""
    try:
        analysis_service = await get_analysis_service()
        summary = await analysis_service.get_analysis_summary(username)
        
        # Get detailed data for available analyses
        export_data = {
            "username": username,
            "exported_at": datetime.utcnow().isoformat(),
            "exported_by": current_user.email,
            "analyses": {}
        }
        
        for analysis_type_str, analysis_info in summary["analyses"].items():
            if analysis_info.get("available") and analysis_info.get("valid"):
                analysis_type = AnalysisType(analysis_type_str)
                
                try:
                    cached_result = await analysis_service.get_cached_analysis(username, analysis_type)
                    if cached_result:
                        export_data["analyses"][analysis_type_str] = {
                            "data": cached_result.data,
                            "metadata": cached_result.metadata if include_metadata else None,
                            "created_at": cached_result.created_at.isoformat(),
                            "cache_strategy": cached_result.cache_strategy.value
                        }
                except Exception as e:
                    export_data["analyses"][analysis_type_str] = {"error": str(e)}
        
        if format == "csv":
            # TODO: Implement CSV export
            raise HTTPException(status_code=501, detail="CSV export not yet implemented")
        
        logger.info(f"Analysis data exported for {username} by user {current_user.email}")
        
        return export_data
        
    except Exception as e:
        logger.error(f"Failed to export analysis data for {username}: {e}")
        raise HTTPException(status_code=500, detail="Export failed")