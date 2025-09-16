"""
Redis-backed Distributed Rate Limiting

Comprehensive rate limiting implementation using Redis for distributed
applications with multiple algorithms and intelligent strategies.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import hashlib

from .config import get_config
from .cache import get_cache, rate_limit_key
from .logging_config import get_logger

logger = get_logger(__name__)


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW_LOG = "sliding_window_log"
    SLIDING_WINDOW_COUNTER = "sliding_window_counter"


class RateLimitScope(str, Enum):
    """Rate limiting scopes."""
    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    USER_ENDPOINT = "user_endpoint"
    IP_ENDPOINT = "ip_endpoint"


class RateLimitResult:
    """Rate limit check result."""
    
    def __init__(
        self,
        allowed: bool,
        limit: int,
        remaining: int,
        reset_time: Optional[datetime] = None,
        retry_after: Optional[int] = None,
        scope: Optional[str] = None,
        algorithm: Optional[str] = None
    ):
        self.allowed = allowed
        self.limit = limit
        self.remaining = remaining
        self.reset_time = reset_time
        self.retry_after = retry_after
        self.scope = scope
        self.algorithm = algorithm
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "limit": self.limit,
            "remaining": self.remaining,
            "reset_time": self.reset_time.isoformat() if self.reset_time else None,
            "retry_after": self.retry_after,
            "scope": self.scope,
            "algorithm": self.algorithm
        }


class RateLimitConfig:
    """Rate limit configuration."""
    
    def __init__(
        self,
        limit: int,
        window: int,  # seconds
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW_COUNTER,
        burst_limit: Optional[int] = None,
        burst_window: Optional[int] = None,
        scope: RateLimitScope = RateLimitScope.IP,
        key_func: Optional[callable] = None
    ):
        self.limit = limit
        self.window = window
        self.algorithm = algorithm
        self.burst_limit = burst_limit or limit
        self.burst_window = burst_window or min(60, window // 4)
        self.scope = scope
        self.key_func = key_func
    
    def generate_key(self, identifier: str, endpoint: str = "") -> str:
        """Generate rate limit key based on scope."""
        if self.key_func:
            return self.key_func(identifier, endpoint)
        
        base_key = f"ratelimit:{self.scope.value}"
        
        if self.scope == RateLimitScope.GLOBAL:
            return f"{base_key}:global"
        elif self.scope == RateLimitScope.USER:
            return f"{base_key}:user:{identifier}"
        elif self.scope == RateLimitScope.IP:
            return f"{base_key}:ip:{identifier}"
        elif self.scope == RateLimitScope.API_KEY:
            key_hash = hashlib.md5(identifier.encode()).hexdigest()[:8]
            return f"{base_key}:apikey:{key_hash}"
        elif self.scope == RateLimitScope.ENDPOINT:
            return f"{base_key}:endpoint:{endpoint}"
        elif self.scope == RateLimitScope.USER_ENDPOINT:
            return f"{base_key}:user:{identifier}:endpoint:{endpoint}"
        elif self.scope == RateLimitScope.IP_ENDPOINT:
            return f"{base_key}:ip:{identifier}:endpoint:{endpoint}"
        else:
            return f"{base_key}:{identifier}"


class RedisRateLimiter:
    """Redis-backed distributed rate limiter."""
    
    def __init__(self):
        self.config = get_config()
        self.cache = None
        
        # Default configurations
        self.default_configs = {
            "api_global": RateLimitConfig(
                limit=1000, 
                window=3600,  # 1 hour
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW_COUNTER,
                scope=RateLimitScope.GLOBAL
            ),
            "api_per_user": RateLimitConfig(
                limit=100,
                window=3600,  # 1 hour
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW_COUNTER,
                scope=RateLimitScope.USER
            ),
            "api_per_ip": RateLimitConfig(
                limit=60,
                window=3600,  # 1 hour
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW_COUNTER,
                burst_limit=10,
                burst_window=60,  # 1 minute
                scope=RateLimitScope.IP
            ),
            "instagram_per_user": RateLimitConfig(
                limit=30,
                window=60,  # 1 minute
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                scope=RateLimitScope.USER
            ),
            "analysis_per_user": RateLimitConfig(
                limit=20,
                window=3600,  # 1 hour
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW_COUNTER,
                scope=RateLimitScope.USER_ENDPOINT
            )
        }
        
        logger.info("Redis rate limiter initialized")
    
    async def initialize(self):
        """Initialize the rate limiter."""
        self.cache = await get_cache()
    
    async def check_rate_limit(
        self,
        config_name: str,
        identifier: str,
        endpoint: str = "",
        cost: int = 1,
        custom_config: Optional[RateLimitConfig] = None
    ) -> RateLimitResult:
        """Check if request is within rate limits."""
        try:
            if not self.cache:
                await self.initialize()
            
            config = custom_config or self.default_configs.get(config_name)
            if not config:
                logger.warning(f"Rate limit config '{config_name}' not found")
                return RateLimitResult(allowed=True, limit=0, remaining=0)
            
            # Generate cache key
            cache_key = config.generate_key(identifier, endpoint)
            
            # Apply rate limiting algorithm
            if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                result = await self._check_token_bucket(cache_key, config, cost)
            elif config.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                result = await self._check_fixed_window(cache_key, config, cost)
            elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW_LOG:
                result = await self._check_sliding_window_log(cache_key, config, cost)
            elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW_COUNTER:
                result = await self._check_sliding_window_counter(cache_key, config, cost)
            else:
                result = RateLimitResult(allowed=True, limit=config.limit, remaining=config.limit)
            
            # Add metadata
            result.scope = config.scope.value
            result.algorithm = config.algorithm.value
            
            # Check burst limits if configured
            if config.burst_limit < config.limit and result.allowed:
                burst_result = await self._check_burst_limit(cache_key, config, cost)
                if not burst_result.allowed:
                    result = burst_result
                    result.scope = f"{config.scope.value}_burst"
            
            # Log rate limit hits
            if not result.allowed:
                logger.warning(f"Rate limit exceeded for {config_name}: {identifier} on {endpoint}")
            
            return result
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if rate limiting fails
            return RateLimitResult(allowed=True, limit=0, remaining=0)
    
    async def _check_token_bucket(
        self, 
        cache_key: str, 
        config: RateLimitConfig, 
        cost: int
    ) -> RateLimitResult:
        """Implement token bucket algorithm."""
        current_time = time.time()
        
        # Get current bucket state
        bucket_data = await self.cache.get(cache_key)
        
        if bucket_data:
            tokens = bucket_data.get("tokens", config.limit)
            last_refill = bucket_data.get("last_refill", current_time)
        else:
            tokens = config.limit
            last_refill = current_time
        
        # Calculate tokens to add based on time elapsed
        time_passed = current_time - last_refill
        tokens_to_add = int(time_passed * (config.limit / config.window))
        tokens = min(config.limit, tokens + tokens_to_add)
        
        # Check if we have enough tokens
        if tokens >= cost:
            tokens -= cost
            allowed = True
            remaining = tokens
        else:
            allowed = False
            remaining = tokens
        
        # Update bucket state
        new_bucket_data = {
            "tokens": tokens,
            "last_refill": current_time
        }
        
        await self.cache.set(cache_key, new_bucket_data, ttl=config.window * 2)
        
        # Calculate retry after
        retry_after = None
        if not allowed:
            tokens_needed = cost - tokens
            retry_after = int(tokens_needed * (config.window / config.limit))
        
        return RateLimitResult(
            allowed=allowed,
            limit=config.limit,
            remaining=remaining,
            retry_after=retry_after
        )
    
    async def _check_fixed_window(
        self, 
        cache_key: str, 
        config: RateLimitConfig, 
        cost: int
    ) -> RateLimitResult:
        """Implement fixed window algorithm."""
        current_time = time.time()
        window_start = int(current_time // config.window) * config.window
        window_key = f"{cache_key}:{window_start}"
        
        # Get current count
        current_count = await self.cache.get(window_key) or 0
        
        # Check if request is allowed
        if current_count + cost <= config.limit:
            allowed = True
            new_count = current_count + cost
            remaining = config.limit - new_count
        else:
            allowed = False
            new_count = current_count
            remaining = max(0, config.limit - current_count)
        
        # Update count
        if allowed:
            await self.cache.set(window_key, new_count, ttl=config.window * 2)
        
        # Calculate reset time and retry after
        reset_time = datetime.fromtimestamp(window_start + config.window)
        retry_after = int(window_start + config.window - current_time) if not allowed else None
        
        return RateLimitResult(
            allowed=allowed,
            limit=config.limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after
        )
    
    async def _check_sliding_window_log(
        self, 
        cache_key: str, 
        config: RateLimitConfig, 
        cost: int
    ) -> RateLimitResult:
        """Implement sliding window log algorithm."""
        current_time = time.time()
        window_start = current_time - config.window
        
        # Get current request log
        request_log = await self.cache.get(cache_key) or []
        
        # Remove old entries
        request_log = [timestamp for timestamp in request_log if timestamp > window_start]
        
        # Check if request is allowed
        if len(request_log) + cost <= config.limit:
            allowed = True
            # Add new request timestamps
            for _ in range(cost):
                request_log.append(current_time)
            remaining = config.limit - len(request_log)
        else:
            allowed = False
            remaining = max(0, config.limit - len(request_log))
        
        # Update request log
        if allowed:
            await self.cache.set(cache_key, request_log, ttl=config.window * 2)
        
        # Calculate retry after
        retry_after = None
        if not allowed and request_log:
            oldest_request = min(request_log)
            retry_after = int(oldest_request + config.window - current_time)
        
        return RateLimitResult(
            allowed=allowed,
            limit=config.limit,
            remaining=remaining,
            retry_after=retry_after
        )
    
    async def _check_sliding_window_counter(
        self, 
        cache_key: str, 
        config: RateLimitConfig, 
        cost: int
    ) -> RateLimitResult:
        """Implement sliding window counter algorithm."""
        current_time = time.time()
        current_window = int(current_time // config.window)
        previous_window = current_window - 1
        
        current_key = f"{cache_key}:{current_window}"
        previous_key = f"{cache_key}:{previous_window}"
        
        # Get counts for current and previous windows
        current_count = await self.cache.get(current_key) or 0
        previous_count = await self.cache.get(previous_key) or 0
        
        # Calculate weighted count
        window_progress = (current_time % config.window) / config.window
        weighted_count = (previous_count * (1 - window_progress)) + current_count
        
        # Check if request is allowed
        if weighted_count + cost <= config.limit:
            allowed = True
            new_count = current_count + cost
            remaining = max(0, config.limit - int(weighted_count + cost))
        else:
            allowed = False
            new_count = current_count
            remaining = max(0, config.limit - int(weighted_count))
        
        # Update current window count
        if allowed:
            await self.cache.set(current_key, new_count, ttl=config.window * 2)
        
        # Calculate retry after
        retry_after = None
        if not allowed:
            # Estimate time until enough capacity is available
            retry_after = int(config.window * (1 - window_progress))
        
        return RateLimitResult(
            allowed=allowed,
            limit=config.limit,
            remaining=remaining,
            retry_after=retry_after
        )
    
    async def _check_burst_limit(
        self, 
        cache_key: str, 
        config: RateLimitConfig, 
        cost: int
    ) -> RateLimitResult:
        """Check burst rate limits."""
        burst_key = f"{cache_key}:burst"
        burst_config = RateLimitConfig(
            limit=config.burst_limit,
            window=config.burst_window,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW_COUNTER,
            scope=config.scope
        )
        
        return await self._check_sliding_window_counter(burst_key, burst_config, cost)
    
    async def reset_rate_limit(
        self, 
        config_name: str, 
        identifier: str, 
        endpoint: str = ""
    ) -> bool:
        """Reset rate limit for a specific identifier."""
        try:
            config = self.default_configs.get(config_name)
            if not config:
                return False
            
            cache_key = config.generate_key(identifier, endpoint)
            
            # Clear all related keys
            patterns = [
                cache_key,
                f"{cache_key}:*",
                f"{cache_key}:burst"
            ]
            
            cleared = 0
            for pattern in patterns:
                cleared += await self.cache.clear_pattern(pattern)
            
            logger.info(f"Reset rate limit for {config_name}: {identifier} (cleared {cleared} keys)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
            return False
    
    async def get_rate_limit_status(
        self, 
        config_name: str, 
        identifier: str, 
        endpoint: str = ""
    ) -> Dict[str, Any]:
        """Get current rate limit status without consuming quota."""
        try:
            config = self.default_configs.get(config_name)
            if not config:
                return {"error": f"Config '{config_name}' not found"}
            
            cache_key = config.generate_key(identifier, endpoint)
            
            # Check status based on algorithm
            if config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                bucket_data = await self.cache.get(cache_key) or {}
                tokens = bucket_data.get("tokens", config.limit)
                
                return {
                    "algorithm": config.algorithm.value,
                    "limit": config.limit,
                    "remaining": tokens,
                    "window": config.window,
                    "scope": config.scope.value
                }
            
            elif config.algorithm == RateLimitAlgorithm.SLIDING_WINDOW_COUNTER:
                current_time = time.time()
                current_window = int(current_time // config.window)
                previous_window = current_window - 1
                
                current_key = f"{cache_key}:{current_window}"
                previous_key = f"{cache_key}:{previous_window}"
                
                current_count = await self.cache.get(current_key) or 0
                previous_count = await self.cache.get(previous_key) or 0
                
                window_progress = (current_time % config.window) / config.window
                weighted_count = (previous_count * (1 - window_progress)) + current_count
                
                return {
                    "algorithm": config.algorithm.value,
                    "limit": config.limit,
                    "remaining": max(0, config.limit - int(weighted_count)),
                    "window": config.window,
                    "scope": config.scope.value,
                    "current_usage": int(weighted_count)
                }
            
            else:
                return {
                    "algorithm": config.algorithm.value,
                    "limit": config.limit,
                    "window": config.window,
                    "scope": config.scope.value,
                    "message": "Status check not implemented for this algorithm"
                }
                
        except Exception as e:
            logger.error(f"Failed to get rate limit status: {e}")
            return {"error": str(e)}
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics."""
        try:
            stats = {
                "active_configs": list(self.default_configs.keys()),
                "total_rate_limit_keys": 0,
                "algorithm_distribution": {},
                "scope_distribution": {}
            }
            
            # Count active rate limit keys
            rate_limit_keys = await self.cache.clear_pattern("ratelimit:*")
            stats["total_rate_limit_keys"] = rate_limit_keys
            
            # Analyze configurations
            for config in self.default_configs.values():
                algorithm = config.algorithm.value
                scope = config.scope.value
                
                stats["algorithm_distribution"][algorithm] = stats["algorithm_distribution"].get(algorithm, 0) + 1
                stats["scope_distribution"][scope] = stats["scope_distribution"].get(scope, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get global rate limit stats: {e}")
            return {"error": str(e)}
    
    def add_custom_config(self, name: str, config: RateLimitConfig):
        """Add a custom rate limit configuration."""
        self.default_configs[name] = config
        logger.info(f"Added custom rate limit config: {name}")
    
    def remove_config(self, name: str) -> bool:
        """Remove a rate limit configuration."""
        if name in self.default_configs:
            del self.default_configs[name]
            logger.info(f"Removed rate limit config: {name}")
            return True
        return False


# Global rate limiter instance
_rate_limiter: Optional[RedisRateLimiter] = None


async def get_rate_limiter() -> RedisRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RedisRateLimiter()
        await _rate_limiter.initialize()
    return _rate_limiter


# Convenience functions
async def check_rate_limit(
    config_name: str,
    identifier: str,
    endpoint: str = "",
    cost: int = 1
) -> RateLimitResult:
    """Convenience function to check rate limits."""
    limiter = await get_rate_limiter()
    return await limiter.check_rate_limit(config_name, identifier, endpoint, cost)


async def reset_rate_limit(
    config_name: str,
    identifier: str,
    endpoint: str = ""
) -> bool:
    """Convenience function to reset rate limits."""
    limiter = await get_rate_limiter()
    return await limiter.reset_rate_limit(config_name, identifier, endpoint)