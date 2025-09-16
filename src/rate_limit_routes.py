"""
Rate Limiting Management Routes

API endpoints for managing and monitoring Redis-backed rate limiting
with comprehensive statistics and configuration management.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .rate_limiter import (
    get_rate_limiter, RedisRateLimiter, RateLimitConfig, 
    RateLimitAlgorithm, RateLimitScope
)
from .auth import get_current_active_user, require_permission, Permission, User
from .logging_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/rate-limit", tags=["rate-limiting"])


# Pydantic models
class RateLimitStatusResponse(BaseModel):
    """Rate limit status response model."""
    algorithm: str
    limit: int
    remaining: int
    window: int
    scope: str
    current_usage: Optional[int] = None
    message: Optional[str] = None


class RateLimitCheckRequest(BaseModel):
    """Rate limit check request model."""
    config_name: str
    identifier: str
    endpoint: str = ""
    cost: int = Field(1, ge=1, le=100)


class RateLimitCheckResponse(BaseModel):
    """Rate limit check response model."""
    allowed: bool
    limit: int
    remaining: int
    scope: Optional[str] = None
    algorithm: Optional[str] = None
    retry_after: Optional[int] = None
    reset_time: Optional[str] = None


class CustomRateLimitConfigRequest(BaseModel):
    """Custom rate limit configuration request."""
    name: str = Field(..., min_length=1, max_length=50)
    limit: int = Field(..., ge=1, le=10000)
    window: int = Field(..., ge=1, le=86400)  # Max 24 hours
    algorithm: RateLimitAlgorithm
    scope: RateLimitScope
    burst_limit: Optional[int] = None
    burst_window: Optional[int] = None


class RateLimitStatsResponse(BaseModel):
    """Rate limit statistics response model."""
    active_configs: List[str]
    total_rate_limit_keys: int
    algorithm_distribution: Dict[str, int]
    scope_distribution: Dict[str, int]
    timestamp: str


# Rate limit checking endpoints

@router.post("/check", response_model=RateLimitCheckResponse)
async def check_rate_limit_endpoint(
    request: RateLimitCheckRequest,
    current_user: User = Depends(require_permission(Permission.MANAGE_SYSTEM))
):
    """Check rate limit for specific identifier and config."""
    try:
        limiter = await get_rate_limiter()
        result = await limiter.check_rate_limit(
            config_name=request.config_name,
            identifier=request.identifier,
            endpoint=request.endpoint,
            cost=request.cost
        )
        
        logger.info(f"Rate limit checked for {request.config_name}:{request.identifier} by {current_user.email}")
        
        return RateLimitCheckResponse(
            allowed=result.allowed,
            limit=result.limit,
            remaining=result.remaining,
            scope=result.scope,
            algorithm=result.algorithm,
            retry_after=result.retry_after,
            reset_time=result.reset_time.isoformat() if result.reset_time else None
        )
        
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{config_name}", response_model=RateLimitStatusResponse)
async def get_rate_limit_status(
    config_name: str,
    identifier: str = Query(..., description="User ID, IP address, or API key"),
    endpoint: str = Query("", description="Optional endpoint path"),
    current_user: User = Depends(require_permission(Permission.VIEW_SYSTEM))
):
    """Get current rate limit status without consuming quota."""
    try:
        limiter = await get_rate_limiter()
        status = await limiter.get_rate_limit_status(
            config_name=config_name,
            identifier=identifier,
            endpoint=endpoint
        )
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        return RateLimitStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get rate limit status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rate limit management endpoints

@router.delete("/reset/{config_name}")
async def reset_rate_limit_endpoint(
    config_name: str,
    identifier: str = Query(..., description="User ID, IP address, or API key"),
    endpoint: str = Query("", description="Optional endpoint path"),
    current_user: User = Depends(require_permission(Permission.MANAGE_SYSTEM))
):
    """Reset rate limit for specific identifier."""
    try:
        limiter = await get_rate_limiter()
        success = await limiter.reset_rate_limit(
            config_name=config_name,
            identifier=identifier,
            endpoint=endpoint
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Rate limit config '{config_name}' not found")
        
        logger.info(f"Rate limit reset for {config_name}:{identifier} by {current_user.email}")
        
        return {
            "message": f"Rate limit reset successfully for {config_name}:{identifier}",
            "config_name": config_name,
            "identifier": identifier,
            "endpoint": endpoint
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset rate limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration management endpoints

@router.post("/configs")
async def add_custom_rate_limit_config(
    config_request: CustomRateLimitConfigRequest,
    current_user: User = Depends(require_permission(Permission.MANAGE_SYSTEM))
):
    """Add a custom rate limit configuration."""
    try:
        limiter = await get_rate_limiter()
        
        # Create configuration
        config = RateLimitConfig(
            limit=config_request.limit,
            window=config_request.window,
            algorithm=config_request.algorithm,
            burst_limit=config_request.burst_limit or config_request.limit,
            burst_window=config_request.burst_window,
            scope=config_request.scope
        )
        
        # Add to rate limiter
        limiter.add_custom_config(config_request.name, config)
        
        logger.info(f"Custom rate limit config '{config_request.name}' added by {current_user.email}")
        
        return {
            "message": f"Rate limit configuration '{config_request.name}' added successfully",
            "config": {
                "name": config_request.name,
                "limit": config_request.limit,
                "window": config_request.window,
                "algorithm": config_request.algorithm.value,
                "scope": config_request.scope.value,
                "burst_limit": config.burst_limit,
                "burst_window": config.burst_window
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to add custom rate limit config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs")
async def list_rate_limit_configs(
    current_user: User = Depends(require_permission(Permission.VIEW_SYSTEM))
):
    """List all available rate limit configurations."""
    try:
        limiter = await get_rate_limiter()
        
        configs = {}
        for name, config in limiter.default_configs.items():
            configs[name] = {
                "limit": config.limit,
                "window": config.window,
                "algorithm": config.algorithm.value,
                "scope": config.scope.value,
                "burst_limit": config.burst_limit,
                "burst_window": config.burst_window
            }
        
        return {
            "configs": configs,
            "total_configs": len(configs)
        }
        
    except Exception as e:
        logger.error(f"Failed to list rate limit configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/configs/{config_name}")
async def remove_rate_limit_config(
    config_name: str,
    current_user: User = Depends(require_permission(Permission.MANAGE_SYSTEM))
):
    """Remove a custom rate limit configuration."""
    try:
        limiter = await get_rate_limiter()
        
        # Prevent removal of built-in configs
        builtin_configs = [
            "api_global", "api_per_user", "api_per_ip", 
            "instagram_per_user", "analysis_per_user"
        ]
        
        if config_name in builtin_configs:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot remove built-in configuration '{config_name}'"
            )
        
        success = limiter.remove_config(config_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Configuration '{config_name}' not found")
        
        logger.info(f"Rate limit config '{config_name}' removed by {current_user.email}")
        
        return {
            "message": f"Rate limit configuration '{config_name}' removed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove rate limit config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Statistics and monitoring endpoints

@router.get("/stats", response_model=RateLimitStatsResponse)
async def get_rate_limit_stats(
    current_user: User = Depends(require_permission(Permission.VIEW_SYSTEM))
):
    """Get comprehensive rate limiting statistics."""
    try:
        limiter = await get_rate_limiter()
        stats = await limiter.get_global_stats()
        
        from datetime import datetime
        return RateLimitStatsResponse(
            **stats,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get rate limit stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def rate_limiter_health_check(
    current_user: User = Depends(require_permission(Permission.VIEW_SYSTEM))
):
    """Health check for rate limiting service."""
    try:
        limiter = await get_rate_limiter()
        
        # Test basic functionality with a dummy check
        test_result = await limiter.check_rate_limit(
            config_name="api_per_ip",
            identifier="health-check",
            endpoint="test",
            cost=0  # Don't consume quota
        )
        
        return {
            "status": "healthy",
            "redis_available": limiter.cache is not None,
            "configs_loaded": len(limiter.default_configs),
            "test_successful": True
        }
        
    except Exception as e:
        logger.error(f"Rate limiter health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "redis_available": False,
            "test_successful": False
        }


# User-specific rate limit endpoints

@router.get("/user/status")
async def get_user_rate_limit_status(
    config_name: str = Query("api_per_user", description="Rate limit configuration"),
    endpoint: str = Query("", description="Optional endpoint path"),
    current_user: User = Depends(get_current_active_user)
):
    """Get current user's rate limit status."""
    try:
        limiter = await get_rate_limiter()
        status = await limiter.get_rate_limit_status(
            config_name=config_name,
            identifier=str(current_user.id),
            endpoint=endpoint
        )
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        return {
            "user_id": current_user.id,
            "email": current_user.email,
            "rate_limit": status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user rate limit status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/user/reset")
async def reset_user_rate_limits(
    current_user: User = Depends(get_current_active_user)
):
    """Reset current user's rate limits (self-service)."""
    try:
        limiter = await get_rate_limiter()
        
        # Reset multiple user-related configs
        user_configs = ["api_per_user", "instagram_per_user", "analysis_per_user"]
        reset_results = {}
        
        for config_name in user_configs:
            try:
                success = await limiter.reset_rate_limit(
                    config_name=config_name,
                    identifier=str(current_user.id)
                )
                reset_results[config_name] = success
            except Exception as e:
                reset_results[config_name] = False
                logger.warning(f"Failed to reset {config_name} for user {current_user.id}: {e}")
        
        logger.info(f"Rate limits reset for user {current_user.email}: {reset_results}")
        
        return {
            "message": "Rate limits reset completed",
            "user_id": current_user.id,
            "reset_results": reset_results,
            "successful_resets": sum(1 for success in reset_results.values() if success)
        }
        
    except Exception as e:
        logger.error(f"Failed to reset user rate limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))