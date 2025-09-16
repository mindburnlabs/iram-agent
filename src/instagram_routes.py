"""
Instagram API Routes

RESTful API endpoints for Instagram profile and content analysis
with comprehensive caching and authentication.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field

from .instagram_service import (
    get_instagram_service, EnhancedInstagramService,
    InstagramAPIError, RateLimitError, AuthenticationError, ProfileNotFoundError
)
from .auth import get_current_active_user, require_permission, Permission, User
from .logging_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/instagram", tags=["instagram"])


# Pydantic models
class ProfileResponse(BaseModel):
    """Instagram profile response model."""
    username: str
    full_name: Optional[str] = None
    biography: Optional[str] = None
    followers_count: Optional[int] = None
    following_count: Optional[int] = None
    media_count: Optional[int] = None
    is_verified: bool = False
    is_private: bool = False
    profile_pic_url: Optional[str] = None
    external_url: Optional[str] = None
    category: Optional[str] = None
    business_category: Optional[str] = None
    scraped_at: str
    method: str
    cache_key: str


class PostData(BaseModel):
    """Instagram post data model."""
    id: Optional[str] = None
    code: Optional[str] = None
    media_type: Optional[str] = None
    caption: Optional[str] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    taken_at: Optional[str] = None
    thumbnail_url: Optional[str] = None
    video_url: Optional[str] = None
    alt_text: Optional[str] = None


class PostsResponse(BaseModel):
    """Instagram posts response model."""
    username: str
    posts: List[PostData]
    total_found: int
    scraped_at: str
    method: str
    cache_key: str


class CacheWarmupRequest(BaseModel):
    """Cache warmup request model."""
    usernames: List[str] = Field(..., min_items=1, max_items=50)
    include_posts: bool = True


class CacheWarmupResponse(BaseModel):
    """Cache warmup response model."""
    success: List[str]
    failed: List[Dict[str, Any]]
    total: int
    message: str


# Profile endpoints

@router.get("/profiles/{username}", response_model=ProfileResponse)
async def get_profile(
    username: str,
    force_refresh: bool = Query(False, description="Force refresh from Instagram API"),
    current_user: User = Depends(require_permission(Permission.VIEW_PROFILES))
):
    """Get Instagram profile information."""
    try:
        instagram_service = await get_instagram_service()
        profile_data = await instagram_service.get_profile_info(username, force_refresh=force_refresh)
        
        logger.info(f"Profile retrieved for {username} by user {current_user.email}", extra={
            "username": username,
            "user_id": current_user.id,
            "method": profile_data.get("method"),
            "from_cache": not force_refresh
        })
        
        return ProfileResponse(**profile_data)
        
    except ProfileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile {username} not found")
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except InstagramAPIError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/{username}/cached", response_model=Optional[ProfileResponse])
async def get_cached_profile(
    username: str,
    current_user: User = Depends(require_permission(Permission.VIEW_PROFILES))
):
    """Get cached profile information without API call."""
    try:
        instagram_service = await get_instagram_service()
        profile_data = await instagram_service.get_cached_profile(username)
        
        if not profile_data:
            return None
        
        logger.debug(f"Cached profile retrieved for {username} by user {current_user.email}")
        
        return ProfileResponse(**profile_data)
        
    except Exception as e:
        logger.error(f"Failed to get cached profile for {username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cached profile")


# Posts endpoints

@router.get("/profiles/{username}/posts", response_model=PostsResponse)
async def get_posts(
    username: str,
    limit: int = Query(50, ge=1, le=200, description="Number of posts to retrieve"),
    force_refresh: bool = Query(False, description="Force refresh from Instagram API"),
    current_user: User = Depends(require_permission(Permission.VIEW_PROFILES))
):
    """Get Instagram posts for a profile."""
    try:
        instagram_service = await get_instagram_service()
        posts_data = await instagram_service.get_user_posts(username, limit, force_refresh=force_refresh)
        
        logger.info(f"Posts retrieved for {username} by user {current_user.email}", extra={
            "username": username,
            "user_id": current_user.id,
            "posts_count": posts_data.get("total_found", 0),
            "method": posts_data.get("method"),
            "from_cache": not force_refresh
        })
        
        # Convert posts to PostData models
        posts = [PostData(**post) for post in posts_data.get("posts", [])]
        posts_data["posts"] = posts
        
        return PostsResponse(**posts_data)
        
    except ProfileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Profile {username} not found")
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except InstagramAPIError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/{username}/posts/cached", response_model=Optional[PostsResponse])
async def get_cached_posts(
    username: str,
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(require_permission(Permission.VIEW_PROFILES))
):
    """Get cached posts without API call."""
    try:
        instagram_service = await get_instagram_service()
        posts_data = await instagram_service.get_cached_posts(username, limit)
        
        if not posts_data:
            return None
        
        logger.debug(f"Cached posts retrieved for {username} by user {current_user.email}")
        
        # Convert posts to PostData models
        posts = [PostData(**post) for post in posts_data.get("posts", [])]
        posts_data["posts"] = posts
        
        return PostsResponse(**posts_data)
        
    except Exception as e:
        logger.error(f"Failed to get cached posts for {username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cached posts")


# Analysis endpoints

@router.post("/profiles/{username}/analyze")
async def analyze_profile(
    username: str,
    background_tasks: BackgroundTasks,
    include_posts: bool = Query(True, description="Include post analysis"),
    current_user: User = Depends(require_permission(Permission.ANALYZE_PROFILES))
):
    """Start comprehensive profile analysis (background job)."""
    try:
        # TODO: Implement background job for analysis
        # For now, return a placeholder response
        
        background_tasks.add_task(
            _analyze_profile_background,
            username,
            current_user.id,
            include_posts
        )
        
        logger.info(f"Profile analysis started for {username} by user {current_user.email}", extra={
            "username": username,
            "user_id": current_user.id,
            "include_posts": include_posts
        })
        
        return {
            "message": f"Analysis started for @{username}",
            "status": "queued",
            "username": username,
            "job_id": f"analysis_{username}_{current_user.id}",  # Placeholder
            "estimated_time": "2-5 minutes"
        }
        
    except Exception as e:
        logger.error(f"Failed to start analysis for {username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start analysis")


async def _analyze_profile_background(username: str, user_id: int, include_posts: bool):
    """Background task for profile analysis."""
    # TODO: Implement actual analysis logic
    logger.info(f"Background analysis running for {username} (user {user_id})")


# Cache management endpoints

@router.post("/cache/warm", response_model=CacheWarmupResponse)
async def warm_cache(
    warmup_request: CacheWarmupRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.VIEW_PROFILES))
):
    """Warm cache with multiple profiles."""
    try:
        instagram_service = await get_instagram_service()
        
        # Run warmup in background
        background_tasks.add_task(
            _warm_cache_background,
            warmup_request.usernames,
            current_user.id
        )
        
        logger.info(f"Cache warmup started for {len(warmup_request.usernames)} profiles by user {current_user.email}")
        
        return CacheWarmupResponse(
            success=[],
            failed=[],
            total=len(warmup_request.usernames),
            message=f"Cache warmup started for {len(warmup_request.usernames)} profiles"
        )
        
    except Exception as e:
        logger.error(f"Failed to start cache warmup: {e}")
        raise HTTPException(status_code=500, detail="Failed to start cache warmup")


async def _warm_cache_background(usernames: List[str], user_id: int):
    """Background task for cache warmup."""
    try:
        instagram_service = await get_instagram_service()
        results = await instagram_service.warm_cache(usernames)
        
        logger.info(f"Cache warmup completed for user {user_id}: {results}")
        
    except Exception as e:
        logger.error(f"Cache warmup background task failed: {e}")


@router.delete("/cache/profiles/{username}")
async def invalidate_profile_cache(
    username: str,
    current_user: User = Depends(require_permission(Permission.VIEW_PROFILES))
):
    """Invalidate cached profile data."""
    try:
        instagram_service = await get_instagram_service()
        await instagram_service.invalidate_profile_cache(username)
        
        logger.info(f"Profile cache invalidated for {username} by user {current_user.email}")
        
        return {"message": f"Profile cache invalidated for @{username}"}
        
    except Exception as e:
        logger.error(f"Failed to invalidate profile cache for {username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to invalidate cache")


@router.delete("/cache/profiles/{username}/posts")
async def invalidate_posts_cache(
    username: str,
    current_user: User = Depends(require_permission(Permission.VIEW_PROFILES))
):
    """Invalidate cached posts data."""
    try:
        instagram_service = await get_instagram_service()
        await instagram_service.invalidate_posts_cache(username)
        
        logger.info(f"Posts cache invalidated for {username} by user {current_user.email}")
        
        return {"message": f"Posts cache invalidated for @{username}"}
        
    except Exception as e:
        logger.error(f"Failed to invalidate posts cache for {username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to invalidate cache")


# Service management endpoints

@router.get("/service/stats")
async def get_service_stats(
    current_user: User = Depends(require_permission(Permission.VIEW_SYSTEM))
):
    """Get Instagram service statistics."""
    try:
        instagram_service = await get_instagram_service()
        stats = await instagram_service.get_cache_stats()
        
        return {
            "service": "instagram",
            "cache_stats": stats,
            "timestamp": "now"
        }
        
    except Exception as e:
        logger.error(f"Failed to get service stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service stats")


@router.post("/service/authenticate")
async def reauthenticate_service(
    current_user: User = Depends(require_permission(Permission.MANAGE_SYSTEM))
):
    """Re-authenticate Instagram service."""
    try:
        instagram_service = await get_instagram_service()
        success = await instagram_service.authenticate()
        
        logger.info(f"Instagram service re-authentication {'successful' if success else 'failed'} by user {current_user.email}")
        
        return {
            "message": "Re-authentication completed",
            "authenticated": success,
            "timestamp": "now"
        }
        
    except Exception as e:
        logger.error(f"Failed to re-authenticate service: {e}")
        raise HTTPException(status_code=500, detail="Failed to re-authenticate service")


# Batch operations

@router.post("/profiles/batch")
async def get_profiles_batch(
    usernames: List[str] = Field(..., min_items=1, max_items=20),
    force_refresh: bool = False,
    current_user: User = Depends(require_permission(Permission.VIEW_PROFILES))
):
    """Get multiple profiles in batch."""
    try:
        instagram_service = await get_instagram_service()
        results = []
        errors = []
        
        for username in usernames:
            try:
                profile_data = await instagram_service.get_profile_info(username, force_refresh=force_refresh)
                results.append(ProfileResponse(**profile_data))
                
            except Exception as e:
                errors.append({
                    "username": username,
                    "error": str(e)
                })
        
        logger.info(f"Batch profile retrieval for {len(usernames)} profiles by user {current_user.email}", extra={
            "requested_count": len(usernames),
            "success_count": len(results),
            "error_count": len(errors)
        })
        
        return {
            "profiles": results,
            "errors": errors,
            "summary": {
                "requested": len(usernames),
                "successful": len(results),
                "failed": len(errors)
            }
        }
        
    except Exception as e:
        logger.error(f"Batch profile retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Batch operation failed")