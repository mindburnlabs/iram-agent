"""
IRAM Redis Caching Layer

Comprehensive caching implementation with Redis backend, TTL management,
invalidation strategies, and fallback support.
"""

import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import wraps
import asyncio

try:
    import redis.asyncio as redis
    from redis.asyncio import ConnectionError, TimeoutError
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    ConnectionError = Exception
    TimeoutError = Exception
    REDIS_AVAILABLE = False

from .config import get_config
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    ttl: int
    created_at: datetime
    hit_count: int = 0
    last_accessed: Optional[datetime] = None


class IRamCache:
    """IRAM Redis cache implementation with fallback to in-memory cache."""
    
    def __init__(self):
        self.config = get_config()
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, CacheEntry] = {}
        self._connection_pool = None
        self._is_redis_available = False
    
    async def initialize(self):
        """Initialize Redis connection with fallback to memory cache."""
        if not REDIS_AVAILABLE:
            logger.info("Redis package not available, using in-memory cache")
            return
            
        if not self.config.has_redis():
            logger.info("Redis not configured, using in-memory cache")
            return
        
        try:
            # Create connection pool
            if self.config.redis.url:
                self._connection_pool = redis.ConnectionPool.from_url(
                    self.config.redis.url,
                    max_connections=self.config.redis.max_connections,
                    encoding='utf-8',
                    decode_responses=True
                )
            else:
                self._connection_pool = redis.ConnectionPool(
                    host=self.config.redis.host,
                    port=self.config.redis.port,
                    db=self.config.redis.db,
                    password=self.config.redis.password,
                    max_connections=self.config.redis.max_connections,
                    encoding='utf-8',
                    decode_responses=True
                )
            
            # Create Redis client
            self.redis_client = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self.redis_client.ping()
            self._is_redis_available = True
            
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}. Falling back to memory cache.")
            self.redis_client = None
            self._is_redis_available = False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self._is_redis_available and self.redis_client:
                # Try Redis first
                value = await self.redis_client.get(key)
                if value is not None:
                    try:
                        # Try JSON decode first
                        return json.loads(value)
                    except json.JSONDecodeError:
                        # Fall back to pickle
                        return pickle.loads(value.encode('latin1'))
                return None
            
            else:
                # Use memory cache
                entry = self.memory_cache.get(key)
                if entry:
                    # Check if expired
                    if entry.created_at + timedelta(seconds=entry.ttl) > datetime.utcnow():
                        entry.hit_count += 1
                        entry.last_accessed = datetime.utcnow()
                        return entry.value
                    else:
                        # Remove expired entry
                        del self.memory_cache[key]
                return None
                
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        serialize_method: str = "json"
    ) -> bool:
        """Set value in cache with optional TTL."""
        if ttl is None:
            ttl = self.config.redis.default_ttl
        
        try:
            # Serialize value
            if serialize_method == "json":
                try:
                    serialized_value = json.dumps(value, default=str)
                except (TypeError, ValueError):
                    # Fall back to pickle
                    serialized_value = pickle.dumps(value).decode('latin1')
            else:
                serialized_value = pickle.dumps(value).decode('latin1')
            
            if self._is_redis_available and self.redis_client:
                # Store in Redis
                await self.redis_client.setex(key, ttl, serialized_value)
                return True
                
            else:
                # Store in memory cache
                self.memory_cache[key] = CacheEntry(
                    key=key,
                    value=value,
                    ttl=ttl,
                    created_at=datetime.utcnow()
                )
                return True
                
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            if self._is_redis_available and self.redis_client:
                result = await self.redis_client.delete(key)
                return result > 0
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            if self._is_redis_available and self.redis_client:
                return await self.redis_client.exists(key) > 0
            else:
                entry = self.memory_cache.get(key)
                if entry:
                    # Check if expired
                    if entry.created_at + timedelta(seconds=entry.ttl) > datetime.utcnow():
                        return True
                    else:
                        del self.memory_cache[key]
                return False
                
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key."""
        try:
            if self._is_redis_available and self.redis_client:
                return await self.redis_client.expire(key, ttl)
            else:
                entry = self.memory_cache.get(key)
                if entry:
                    # Update TTL by adjusting created_at
                    entry.created_at = datetime.utcnow()
                    entry.ttl = ttl
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            if self._is_redis_available and self.redis_client:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    return await self.redis_client.delete(*keys)
                return 0
            else:
                # Pattern matching for memory cache
                import fnmatch
                keys_to_delete = [
                    key for key in self.memory_cache.keys()
                    if fnmatch.fnmatch(key, pattern)
                ]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                return len(keys_to_delete)
                
        except Exception as e:
            logger.error(f"Cache clear pattern error for pattern {pattern}: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if self._is_redis_available and self.redis_client:
                info = await self.redis_client.info()
                return {
                    "backend": "redis",
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0),
                    "expired_keys": info.get("expired_keys", 0),
                }
            else:
                # Memory cache stats
                total_hits = sum(entry.hit_count for entry in self.memory_cache.values())
                return {
                    "backend": "memory",
                    "total_keys": len(self.memory_cache),
                    "total_hits": total_hits,
                    "memory_usage_bytes": sum(
                        len(pickle.dumps(entry.value)) 
                        for entry in self.memory_cache.values()
                    )
                }
                
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"backend": "unknown", "error": str(e)}
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries in memory cache."""
        if self._is_redis_available:
            # Redis handles expiration automatically
            return 0
        
        expired_keys = []
        now = datetime.utcnow()
        
        for key, entry in self.memory_cache.items():
            if entry.created_at + timedelta(seconds=entry.ttl) <= now:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            self._is_redis_available = False
        
        if self._connection_pool:
            await self._connection_pool.disconnect()
            self._connection_pool = None
        
        # Clear memory cache
        self.memory_cache.clear()
        
        logger.info("Cache closed")


# Cache key generators
def generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from arguments."""
    key_data = f"{args}_{kwargs}"
    return hashlib.md5(key_data.encode()).hexdigest()


def profile_cache_key(username: str) -> str:
    """Generate cache key for Instagram profile."""
    return f"profile:{username.lower()}"


def posts_cache_key(username: str, limit: int = 50) -> str:
    """Generate cache key for Instagram posts."""
    return f"posts:{username.lower()}:{limit}"


def analysis_cache_key(profile_id: int, analysis_type: str) -> str:
    """Generate cache key for analysis results."""
    return f"analysis:{profile_id}:{analysis_type}"


def user_session_key(user_id: int) -> str:
    """Generate cache key for user session."""
    return f"session:user:{user_id}"


def rate_limit_key(ip_address: str) -> str:
    """Generate cache key for rate limiting."""
    return f"ratelimit:{ip_address}"


# Cache decorators
def cached(
    ttl: int = 3600,
    key_func: Optional[Callable] = None,
    serialize_method: str = "json"
):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = await get_cache()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                await cache.set(cache_key, result, ttl, serialize_method)
                logger.debug(f"Cached result for key: {cache_key}")
            
            return result
        return wrapper
    return decorator


def cache_invalidate(pattern: str):
    """Decorator to invalidate cache entries after function execution."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache
            cache = await get_cache()
            invalidated = await cache.clear_pattern(pattern)
            if invalidated > 0:
                logger.info(f"Invalidated {invalidated} cache entries matching pattern: {pattern}")
            
            return result
        return wrapper
    return decorator


# Global cache instance
_cache_instance: Optional[IRamCache] = None


async def get_cache() -> IRamCache:
    """Get the global cache instance."""
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = IRamCache()
        await _cache_instance.initialize()
    
    return _cache_instance


async def initialize_cache():
    """Initialize the global cache."""
    await get_cache()


async def close_cache():
    """Close the global cache."""
    global _cache_instance
    if _cache_instance:
        await _cache_instance.close()
        _cache_instance = None


# Cache warming functions
async def warm_cache():
    """Warm up the cache with commonly accessed data."""
    logger.info("Starting cache warm-up")
    
    try:
        # In a real implementation, you'd pre-load:
        # - Popular profiles
        # - Recent analysis results
        # - System configuration
        # - User sessions
        
        logger.info("Cache warm-up completed")
        
    except Exception as e:
        logger.error(f"Cache warm-up failed: {e}")


# Cache monitoring and maintenance
async def cache_maintenance():
    """Perform cache maintenance tasks."""
    cache = await get_cache()
    
    # Clean up expired entries (for memory cache)
    expired_count = await cache.cleanup_expired()
    
    # Get cache statistics
    stats = await cache.get_stats()
    logger.info(f"Cache maintenance completed. Stats: {stats}")
    
    return {
        "expired_cleaned": expired_count,
        "stats": stats
    }


# Context manager for cache transactions
class CacheTransaction:
    """Context manager for cache transactions with rollback support."""
    
    def __init__(self):
        self.operations = []
        self.cache = None
    
    async def __aenter__(self):
        self.cache = await get_cache()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            # Rollback operations on error
            await self._rollback()
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in transaction."""
        self.operations.append(("set", key, value, ttl))
        return await self.cache.set(key, value, ttl)
    
    async def delete(self, key: str):
        """Delete key in transaction."""
        self.operations.append(("delete", key))
        return await self.cache.delete(key)
    
    async def _rollback(self):
        """Rollback operations (simplified implementation)."""
        logger.warning("Rolling back cache transaction")
        
        # In a real implementation, you'd need to store original values
        # to properly rollback. This is a simplified version.
        for operation in reversed(self.operations):
            try:
                if operation[0] == "set":
                    await self.cache.delete(operation[1])
                # Rollback for delete would require storing original value
            except Exception as e:
                logger.error(f"Rollback operation failed: {e}")
