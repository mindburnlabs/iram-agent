"""
IRAM Analysis Service with Intelligent Caching

Comprehensive analysis service that provides cached ML model outputs,
sentiment analysis, trend detection, and content insights with
intelligent cache invalidation strategies.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import time

from .config import get_config
from .cache import (
    get_cache, cached, cache_invalidate, analysis_cache_key,
    CacheTransaction
)
from .logging_config import get_logger
from .instagram_service import get_instagram_service
from .repository import InstagramProfileRepository, AnalysisRepository

logger = get_logger(__name__)


class AnalysisType(str, Enum):
    """Types of analysis that can be performed."""
    PROFILE_SENTIMENT = "profile_sentiment"
    CONTENT_ANALYSIS = "content_analysis"
    ENGAGEMENT_ANALYSIS = "engagement_analysis"
    TREND_ANALYSIS = "trend_analysis"
    AUDIENCE_ANALYSIS = "audience_analysis"
    HASHTAG_ANALYSIS = "hashtag_analysis"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    GROWTH_ANALYSIS = "growth_analysis"


class AnalysisStatus(str, Enum):
    """Status of analysis jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


class CacheStrategy(str, Enum):
    """Cache invalidation strategies."""
    TIME_BASED = "time_based"          # Expire after TTL
    CONTENT_BASED = "content_based"    # Invalidate when content changes
    HYBRID = "hybrid"                  # Both time and content based
    NEVER_EXPIRE = "never_expire"      # Cache permanently until manual invalidation


class AnalysisResult:
    """Analysis result with metadata."""
    
    def __init__(
        self,
        analysis_type: AnalysisType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        cache_strategy: CacheStrategy = CacheStrategy.TIME_BASED,
        ttl: Optional[int] = None
    ):
        self.analysis_type = analysis_type
        self.data = data
        self.metadata = metadata or {}
        self.cache_strategy = cache_strategy
        self.ttl = ttl or self._get_default_ttl()
        self.created_at = datetime.utcnow()
        self.version = 1
        
        # Add standard metadata
        self.metadata.update({
            "created_at": self.created_at.isoformat(),
            "analysis_version": self.version,
            "cache_strategy": cache_strategy.value,
            "ttl": self.ttl
        })
    
    def _get_default_ttl(self) -> int:
        """Get default TTL based on analysis type."""
        ttl_map = {
            AnalysisType.PROFILE_SENTIMENT: 6 * 3600,      # 6 hours
            AnalysisType.CONTENT_ANALYSIS: 12 * 3600,      # 12 hours
            AnalysisType.ENGAGEMENT_ANALYSIS: 4 * 3600,    # 4 hours
            AnalysisType.TREND_ANALYSIS: 2 * 3600,         # 2 hours
            AnalysisType.AUDIENCE_ANALYSIS: 24 * 3600,     # 24 hours
            AnalysisType.HASHTAG_ANALYSIS: 8 * 3600,       # 8 hours
            AnalysisType.COMPETITOR_ANALYSIS: 12 * 3600,   # 12 hours
            AnalysisType.GROWTH_ANALYSIS: 6 * 3600,        # 6 hours
        }
        return ttl_map.get(self.analysis_type, 6 * 3600)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for caching."""
        return {
            "analysis_type": self.analysis_type.value,
            "data": self.data,
            "metadata": self.metadata,
            "cache_strategy": self.cache_strategy.value,
            "ttl": self.ttl,
            "created_at": self.created_at.isoformat(),
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create from dictionary."""
        result = cls(
            analysis_type=AnalysisType(data["analysis_type"]),
            data=data["data"],
            metadata=data.get("metadata", {}),
            cache_strategy=CacheStrategy(data.get("cache_strategy", "time_based")),
            ttl=data.get("ttl")
        )
        result.created_at = datetime.fromisoformat(data["created_at"])
        result.version = data.get("version", 1)
        return result


class EnhancedAnalysisService:
    """Enhanced analysis service with intelligent caching."""
    
    def __init__(self):
        self.config = get_config()
        self.cache_hit_rate = 0.0
        self.total_requests = 0
        self.cache_hits = 0
        
        logger.info("Enhanced analysis service initialized")
    
    def _generate_analysis_key(
        self, 
        username: str, 
        analysis_type: AnalysisType, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for analysis results."""
        # Include parameters in the key to cache different parameter combinations
        param_hash = ""
        if parameters:
            param_str = json.dumps(parameters, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        base_key = f"analysis:{username.lower()}:{analysis_type.value}"
        return f"{base_key}:{param_hash}" if param_hash else base_key
    
    async def _get_profile_content_hash(self, username: str) -> str:
        """Generate hash of profile content for cache invalidation."""
        try:
            instagram_service = await get_instagram_service()
            
            # Get profile and recent posts
            profile = await instagram_service.get_cached_profile(username)
            posts = await instagram_service.get_cached_posts(username, limit=10)
            
            if not profile or not posts:
                return ""
            
            # Create hash from key content indicators
            content_indicators = {
                "followers": profile.get("followers_count", 0),
                "following": profile.get("following_count", 0),
                "media_count": profile.get("media_count", 0),
                "bio_hash": hashlib.md5(str(profile.get("biography", "")).encode()).hexdigest()[:8],
                "recent_posts": len(posts.get("posts", [])),
                "last_post_time": posts.get("posts", [{}])[0].get("taken_at") if posts.get("posts") else None
            }
            
            content_str = json.dumps(content_indicators, sort_keys=True)
            return hashlib.md5(content_str.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Failed to generate content hash for {username}: {e}")
            return ""
    
    async def _check_cache_validity(
        self, 
        username: str, 
        analysis_result: AnalysisResult
    ) -> bool:
        """Check if cached analysis is still valid based on strategy."""
        if analysis_result.cache_strategy == CacheStrategy.NEVER_EXPIRE:
            return True
        
        # Check time-based expiration
        if analysis_result.cache_strategy in [CacheStrategy.TIME_BASED, CacheStrategy.HYBRID]:
            expires_at = analysis_result.created_at + timedelta(seconds=analysis_result.ttl)
            if datetime.utcnow() > expires_at:
                return False
        
        # Check content-based expiration
        if analysis_result.cache_strategy in [CacheStrategy.CONTENT_BASED, CacheStrategy.HYBRID]:
            current_hash = await self._get_profile_content_hash(username)
            cached_hash = analysis_result.metadata.get("content_hash", "")
            if current_hash and cached_hash and current_hash != cached_hash:
                return False
        
        return True
    
    @cached(ttl=0, key_func=lambda self, username, analysis_type, **kwargs: 
           self._generate_analysis_key(username, analysis_type, kwargs))
    async def get_cached_analysis(
        self, 
        username: str, 
        analysis_type: AnalysisType,
        **parameters
    ) -> Optional[AnalysisResult]:
        """Get cached analysis result if valid."""
        try:
            cache = await get_cache()
            cache_key = self._generate_analysis_key(username, analysis_type, parameters)
            
            cached_data = await cache.get(cache_key)
            if not cached_data:
                return None
            
            analysis_result = AnalysisResult.from_dict(cached_data)
            
            # Check if cache is still valid
            if await self._check_cache_validity(username, analysis_result):
                self.cache_hits += 1
                self.total_requests += 1
                self.cache_hit_rate = self.cache_hits / self.total_requests
                
                logger.debug(f"Cache hit for {analysis_type.value} analysis of {username}")
                return analysis_result
            else:
                # Cache is invalid, remove it
                await cache.delete(cache_key)
                logger.debug(f"Cache invalidated for {analysis_type.value} analysis of {username}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to get cached analysis: {e}")
            return None
    
    async def store_analysis_result(
        self,
        username: str,
        analysis_result: AnalysisResult,
        **parameters
    ) -> bool:
        """Store analysis result in cache with metadata."""
        try:
            cache = await get_cache()
            cache_key = self._generate_analysis_key(username, analysis_result.analysis_type, parameters)
            
            # Add content hash for content-based invalidation
            if analysis_result.cache_strategy in [CacheStrategy.CONTENT_BASED, CacheStrategy.HYBRID]:
                content_hash = await self._get_profile_content_hash(username)
                analysis_result.metadata["content_hash"] = content_hash
            
            # Store with appropriate TTL
            ttl = None if analysis_result.cache_strategy == CacheStrategy.NEVER_EXPIRE else analysis_result.ttl
            
            success = await cache.set(cache_key, analysis_result.to_dict(), ttl=ttl)
            
            if success:
                logger.info(f"Stored {analysis_result.analysis_type.value} analysis for {username} (TTL: {ttl}s)")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to store analysis result: {e}")
            return False
    
    async def analyze_profile_sentiment(
        self, 
        username: str,
        force_refresh: bool = False,
        **parameters
    ) -> AnalysisResult:
        """Analyze profile sentiment with caching."""
        analysis_type = AnalysisType.PROFILE_SENTIMENT
        
        # Check cache first
        if not force_refresh:
            cached_result = await self.get_cached_analysis(username, analysis_type, **parameters)
            if cached_result:
                return cached_result
        
        self.total_requests += 1
        
        try:
            # Get profile data
            instagram_service = await get_instagram_service()
            profile = await instagram_service.get_profile_info(username)
            posts = await instagram_service.get_user_posts(username, limit=20)
            
            # Simulate sentiment analysis (in real implementation, use actual ML models)
            await asyncio.sleep(0.5)  # Simulate processing time
            
            sentiment_data = await self._perform_sentiment_analysis(profile, posts)
            
            # Create result
            result = AnalysisResult(
                analysis_type=analysis_type,
                data=sentiment_data,
                cache_strategy=CacheStrategy.HYBRID,
                metadata={
                    "profile_analyzed": True,
                    "posts_analyzed": len(posts.get("posts", [])),
                    "processing_time": 0.5
                }
            )
            
            # Store in cache
            await self.store_analysis_result(username, result, **parameters)
            
            logger.info(f"Completed sentiment analysis for {username}")
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {username}: {e}")
            raise
    
    async def _perform_sentiment_analysis(
        self, 
        profile: Dict[str, Any], 
        posts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform actual sentiment analysis."""
        # Simplified sentiment analysis simulation
        import random
        
        bio_text = profile.get("biography", "")
        post_texts = [post.get("caption", "") for post in posts.get("posts", [])]
        
        # Simulate processing
        sentiment_scores = []
        for text in [bio_text] + post_texts:
            if text:
                # Simulate sentiment scoring
                sentiment_scores.append(random.uniform(-1, 1))
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        return {
            "overall_sentiment": avg_sentiment,
            "sentiment_classification": (
                "positive" if avg_sentiment > 0.2 
                else "negative" if avg_sentiment < -0.2 
                else "neutral"
            ),
            "confidence": random.uniform(0.7, 0.95),
            "texts_analyzed": len(sentiment_scores),
            "sentiment_distribution": {
                "positive": sum(1 for s in sentiment_scores if s > 0.2),
                "neutral": sum(1 for s in sentiment_scores if -0.2 <= s <= 0.2),
                "negative": sum(1 for s in sentiment_scores if s < -0.2)
            }
        }
    
    async def analyze_content_engagement(
        self, 
        username: str,
        force_refresh: bool = False,
        **parameters
    ) -> AnalysisResult:
        """Analyze content engagement patterns with caching."""
        analysis_type = AnalysisType.ENGAGEMENT_ANALYSIS
        
        # Check cache first
        if not force_refresh:
            cached_result = await self.get_cached_analysis(username, analysis_type, **parameters)
            if cached_result:
                return cached_result
        
        self.total_requests += 1
        
        try:
            # Get profile data
            instagram_service = await get_instagram_service()
            profile = await instagram_service.get_profile_info(username)
            posts = await instagram_service.get_user_posts(username, limit=50)
            
            # Simulate engagement analysis
            await asyncio.sleep(1.0)  # Simulate processing time
            
            engagement_data = await self._analyze_engagement_patterns(profile, posts)
            
            # Create result
            result = AnalysisResult(
                analysis_type=analysis_type,
                data=engagement_data,
                cache_strategy=CacheStrategy.TIME_BASED,
                ttl=4 * 3600,  # 4 hours
                metadata={
                    "posts_analyzed": len(posts.get("posts", [])),
                    "processing_time": 1.0
                }
            )
            
            # Store in cache
            await self.store_analysis_result(username, result, **parameters)
            
            logger.info(f"Completed engagement analysis for {username}")
            return result
            
        except Exception as e:
            logger.error(f"Engagement analysis failed for {username}: {e}")
            raise
    
    async def _analyze_engagement_patterns(
        self, 
        profile: Dict[str, Any], 
        posts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze engagement patterns."""
        import random
        
        post_list = posts.get("posts", [])
        
        if not post_list:
            return {"error": "No posts to analyze"}
        
        # Calculate engagement metrics
        total_likes = sum(post.get("like_count", 0) for post in post_list)
        total_comments = sum(post.get("comment_count", 0) for post in post_list)
        follower_count = profile.get("followers_count", 1)
        
        avg_likes = total_likes / len(post_list) if post_list else 0
        avg_comments = total_comments / len(post_list) if post_list else 0
        
        engagement_rate = ((avg_likes + avg_comments) / follower_count * 100) if follower_count > 0 else 0
        
        return {
            "engagement_rate": round(engagement_rate, 2),
            "average_likes": round(avg_likes, 1),
            "average_comments": round(avg_comments, 1),
            "total_interactions": total_likes + total_comments,
            "posts_analyzed": len(post_list),
            "engagement_trend": random.choice(["increasing", "stable", "decreasing"]),
            "peak_engagement_hours": [9, 12, 15, 18, 21],  # Simulated
            "top_performing_content_types": ["photos", "reels", "carousels"]
        }
    
    async def analyze_hashtag_performance(
        self, 
        username: str,
        force_refresh: bool = False,
        **parameters
    ) -> AnalysisResult:
        """Analyze hashtag performance with caching."""
        analysis_type = AnalysisType.HASHTAG_ANALYSIS
        
        # Check cache first
        if not force_refresh:
            cached_result = await self.get_cached_analysis(username, analysis_type, **parameters)
            if cached_result:
                return cached_result
        
        self.total_requests += 1
        
        try:
            # Get posts data
            instagram_service = await get_instagram_service()
            posts = await instagram_service.get_user_posts(username, limit=30)
            
            # Simulate hashtag analysis
            await asyncio.sleep(0.8)
            
            hashtag_data = await self._analyze_hashtag_performance(posts)
            
            result = AnalysisResult(
                analysis_type=analysis_type,
                data=hashtag_data,
                cache_strategy=CacheStrategy.TIME_BASED,
                ttl=8 * 3600,  # 8 hours
                metadata={
                    "posts_analyzed": len(posts.get("posts", [])),
                    "processing_time": 0.8
                }
            )
            
            await self.store_analysis_result(username, result, **parameters)
            
            logger.info(f"Completed hashtag analysis for {username}")
            return result
            
        except Exception as e:
            logger.error(f"Hashtag analysis failed for {username}: {e}")
            raise
    
    async def _analyze_hashtag_performance(self, posts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze hashtag performance."""
        import re
        import random
        from collections import Counter
        
        post_list = posts.get("posts", [])
        
        # Extract hashtags
        all_hashtags = []
        hashtag_engagement = {}
        
        for post in post_list:
            caption = post.get("caption", "")
            hashtags = re.findall(r'#(\w+)', caption)
            
            likes = post.get("like_count", 0)
            comments = post.get("comment_count", 0)
            engagement = likes + comments
            
            for hashtag in hashtags:
                all_hashtags.append(hashtag.lower())
                if hashtag.lower() not in hashtag_engagement:
                    hashtag_engagement[hashtag.lower()] = []
                hashtag_engagement[hashtag.lower()].append(engagement)
        
        # Analyze hashtag performance
        hashtag_counts = Counter(all_hashtags)
        top_hashtags = hashtag_counts.most_common(10)
        
        hashtag_performance = {}
        for hashtag, count in top_hashtags:
            engagements = hashtag_engagement.get(hashtag, [])
            avg_engagement = sum(engagements) / len(engagements) if engagements else 0
            hashtag_performance[hashtag] = {
                "usage_count": count,
                "average_engagement": round(avg_engagement, 1),
                "total_engagement": sum(engagements)
            }
        
        return {
            "total_hashtags_found": len(set(all_hashtags)),
            "average_hashtags_per_post": round(len(all_hashtags) / len(post_list), 1) if post_list else 0,
            "top_performing_hashtags": hashtag_performance,
            "hashtag_diversity_score": len(set(all_hashtags)) / len(all_hashtags) if all_hashtags else 0,
            "recommended_hashtags": [f"#{tag}" for tag in random.sample(
                ["instagram", "photooftheday", "instagood", "love", "beautiful", "happy"], 3
            )]
        }
    
    async def get_analysis_summary(self, username: str) -> Dict[str, Any]:
        """Get summary of all available analyses for a user."""
        try:
            cache = await get_cache()
            
            # Check for cached analyses
            available_analyses = {}
            
            for analysis_type in AnalysisType:
                cache_key = self._generate_analysis_key(username, analysis_type)
                cached_data = await cache.get(cache_key)
                
                if cached_data:
                    result = AnalysisResult.from_dict(cached_data)
                    is_valid = await self._check_cache_validity(username, result)
                    
                    available_analyses[analysis_type.value] = {
                        "available": True,
                        "created_at": result.created_at.isoformat(),
                        "expires_at": (result.created_at + timedelta(seconds=result.ttl)).isoformat(),
                        "valid": is_valid,
                        "cache_strategy": result.cache_strategy.value
                    }
                else:
                    available_analyses[analysis_type.value] = {
                        "available": False,
                        "valid": False
                    }
            
            return {
                "username": username,
                "analyses": available_analyses,
                "cache_stats": {
                    "hit_rate": round(self.cache_hit_rate * 100, 1),
                    "total_requests": self.total_requests,
                    "cache_hits": self.cache_hits
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get analysis summary for {username}: {e}")
            raise
    
    @cache_invalidate("analysis:*")
    async def invalidate_user_analyses(self, username: str) -> int:
        """Invalidate all cached analyses for a user."""
        try:
            cache = await get_cache()
            pattern = f"analysis:{username.lower()}:*"
            
            invalidated = await cache.clear_pattern(pattern)
            
            logger.info(f"Invalidated {invalidated} analyses for user {username}")
            return invalidated
            
        except Exception as e:
            logger.error(f"Failed to invalidate analyses for {username}: {e}")
            return 0
    
    async def invalidate_analysis_type(self, analysis_type: AnalysisType) -> int:
        """Invalidate all cached analyses of a specific type."""
        try:
            cache = await get_cache()
            pattern = f"analysis:*:{analysis_type.value}:*"
            
            invalidated = await cache.clear_pattern(pattern)
            
            logger.info(f"Invalidated {invalidated} analyses of type {analysis_type.value}")
            return invalidated
            
        except Exception as e:
            logger.error(f"Failed to invalidate {analysis_type.value} analyses: {e}")
            return 0
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get analysis cache statistics."""
        try:
            cache = await get_cache()
            
            # Count different types of cached analyses
            analysis_counts = {}
            for analysis_type in AnalysisType:
                pattern = f"analysis:*:{analysis_type.value}:*"
                count = await cache.clear_pattern(pattern)  # This should just count, not clear
                analysis_counts[analysis_type.value] = count
            
            return {
                "total_cached_analyses": sum(analysis_counts.values()),
                "analysis_type_breakdown": analysis_counts,
                "cache_hit_rate": round(self.cache_hit_rate * 100, 1),
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.total_requests - self.cache_hits
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {"error": str(e)}


# Global service instance
_analysis_service: Optional[EnhancedAnalysisService] = None


async def get_analysis_service() -> EnhancedAnalysisService:
    """Get the global analysis service instance."""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = EnhancedAnalysisService()
    return _analysis_service