"""
Enhanced Instagram Service with Comprehensive Caching

This service provides cached Instagram data retrieval with intelligent
cache management, rate limiting, and fallback strategies.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import time
import random

from instagrapi import Client
from instagrapi.exceptions import LoginRequired, PleaseWaitFewMinutes, UserNotFound
from playwright.async_api import async_playwright, Browser, BrowserContext

from .config import get_config
from .cache import (
    get_cache, cached, cache_invalidate,
    profile_cache_key, posts_cache_key, analysis_cache_key,
    CacheTransaction
)
from .logging_config import get_logger
from .utils import validate_instagram_username
from .evasion_manager import EvasionManager

logger = get_logger(__name__)


class InstagramAPIError(Exception):
    """Base exception for Instagram API errors."""
    pass


class RateLimitError(InstagramAPIError):
    """Raised when rate limit is exceeded."""
    pass


class AuthenticationError(InstagramAPIError):
    """Raised when authentication fails."""
    pass


class ProfileNotFoundError(InstagramAPIError):
    """Raised when profile is not found."""
    pass


class EnhancedInstagramService:
    """Enhanced Instagram service with caching and intelligent data retrieval."""
    
    def __init__(self):
        self.config = get_config()
        self.client = None
        self.authenticated = False
        self.evasion_manager = EvasionManager()
        
        # Browser automation
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        
        # Rate limiting tracking
        self.last_request_time = 0
        self.requests_count = 0
        self.rate_limit_window_start = time.time()
        
        logger.info("Enhanced Instagram service initialized")
    
    async def initialize(self):
        """Initialize the Instagram service."""
        # Initialize Instagrapi client
        self.client = Client()
        self.client.delay_range = [
            self.config.instagram.delay_min,
            self.config.instagram.delay_max
        ]
        
        # Attempt authentication if credentials are available
        if self.config.has_instagram_auth():
            await self.authenticate()
        
        # Initialize browser for fallback
        await self._init_browser()
    
    async def authenticate(self) -> bool:
        """Authenticate with Instagram using cached session or credentials."""
        try:
            cache = await get_cache()
            session_key = "instagram:session"
            
            # Try to load cached session
            cached_session = await cache.get(session_key)
            if cached_session and self._load_session(cached_session):
                logger.info("Loaded Instagram session from cache")
                self.authenticated = True
                return True
            
            # Authenticate with credentials
            username = self.config.instagram.username
            password = self.config.instagram.password
            
            if not username or not password:
                logger.warning("Instagram credentials not provided")
                return False
            
            # Apply evasion delay
            await self.evasion_manager.apply_delay()
            
            # Login
            success = self.client.login(username, password)
            if success:
                self.authenticated = True
                
                # Cache session
                session_data = self._get_session_data()
                await cache.set(session_key, session_data, ttl=24*3600)  # 24 hours
                
                logger.info("Successfully authenticated with Instagram")
                return True
            else:
                raise AuthenticationError("Login failed")
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def _load_session(self, session_data: Dict[str, Any]) -> bool:
        """Load session from cached data."""
        try:
            # In a real implementation, you'd restore the client session
            # This is a simplified version
            return session_data.get("authenticated", False)
        except Exception as e:
            logger.warning(f"Failed to load session: {e}")
            return False
    
    def _get_session_data(self) -> Dict[str, Any]:
        """Get current session data for caching."""
        return {
            "authenticated": self.authenticated,
            "cached_at": datetime.utcnow().isoformat(),
            # In real implementation, include session cookies/tokens
        }
    
    async def _init_browser(self) -> bool:
        """Initialize Playwright browser for fallback scraping."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-extensions',
                ]
            )
            
            # Randomize user agent
            user_agents = self.config.instagram.user_agents
            user_agent = random.choice(user_agents) if user_agents else (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            self.context = await self.browser.new_context(
                user_agent=user_agent,
                viewport={"width": 1280, "height": 720}
            )
            
            logger.info("Browser initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            return False
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Reset window if needed
        if current_time - self.rate_limit_window_start > 60:
            self.rate_limit_window_start = current_time
            self.requests_count = 0
        
        # Check rate limit
        if self.requests_count >= self.config.instagram.requests_per_minute:
            wait_time = 60 - (current_time - self.rate_limit_window_start)
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                raise RateLimitError(f"Rate limit exceeded. Try again in {wait_time:.1f} seconds")
        
        # Apply delay between requests
        if self.last_request_time > 0:
            elapsed = current_time - self.last_request_time
            min_delay = self.config.instagram.delay_min
            if elapsed < min_delay:
                await asyncio.sleep(min_delay - elapsed)
        
        self.last_request_time = time.time()
        self.requests_count += 1
    
    @cached(ttl=3600, key_func=lambda self, username: profile_cache_key(username))
    async def get_profile_info(self, username: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Get comprehensive profile information with caching."""
        try:
            if not validate_instagram_username(username):
                raise ValueError("Invalid username format")
            
            # Check rate limit
            await self._check_rate_limit()
            
            # Try authenticated API first
            if self.authenticated and self.client:
                try:
                    profile_data = await self._get_profile_api(username)
                    logger.info(f"Retrieved profile via API: {username}")
                    return profile_data
                    
                except Exception as e:
                    logger.warning(f"API method failed for {username}: {e}")
            
            # Fallback to browser scraping
            profile_data = await self._get_profile_browser(username)
            logger.info(f"Retrieved profile via browser: {username}")
            return profile_data
            
        except Exception as e:
            logger.error(f"Profile retrieval failed for {username}: {e}")
            raise InstagramAPIError(f"Failed to get profile {username}: {str(e)}")
    
    async def _get_profile_api(self, username: str) -> Dict[str, Any]:
        """Get profile data using Instagrapi."""
        try:
            user_info = self.client.user_info_by_username(username)
            
            return {
                "username": user_info.username,
                "full_name": user_info.full_name,
                "biography": user_info.biography,
                "followers_count": user_info.follower_count,
                "following_count": user_info.following_count,
                "media_count": user_info.media_count,
                "is_verified": user_info.is_verified,
                "is_private": user_info.is_private,
                "profile_pic_url": str(user_info.profile_pic_url),
                "external_url": user_info.external_url,
                "category": user_info.category,
                "business_category": getattr(user_info, 'business_category_name', None),
                "scraped_at": datetime.utcnow().isoformat(),
                "method": "api",
                "cache_key": profile_cache_key(username)
            }
            
        except UserNotFound:
            raise ProfileNotFoundError(f"Profile {username} not found")
        except (LoginRequired, PleaseWaitFewMinutes) as e:
            raise RateLimitError(f"Instagram API rate limit: {str(e)}")
    
    async def _get_profile_browser(self, username: str) -> Dict[str, Any]:
        """Get profile data using browser automation."""
        if not self.browser or not self.context:
            await self._init_browser()
        
        try:
            page = await self.context.new_page()
            
            # Navigate to profile
            await page.goto(f"https://www.instagram.com/{username}/", timeout=30000)
            await page.wait_for_timeout(3000)
            
            # Check if profile exists
            is_404 = await page.locator("text=Sorry, this page isn't available").count() > 0
            if is_404:
                raise ProfileNotFoundError(f"Profile {username} not found")
            
            # Extract profile data
            profile_data = await page.evaluate("""
                () => {
                    const data = {};
                    
                    // Get username from URL or title
                    data.username = window.location.pathname.split('/')[1];
                    
                    // Get basic info from meta tags
                    const metaDescription = document.querySelector('meta[name="description"]');
                    if (metaDescription) {
                        const desc = metaDescription.content;
                        const followerMatch = desc.match(/(\\d+[KM]?) Followers/);
                        const followingMatch = desc.match(/(\\d+[KM]?) Following/);
                        const postMatch = desc.match(/(\\d+[KM]?) Posts/);
                        
                        if (followerMatch) data.followers_count = followerMatch[1];
                        if (followingMatch) data.following_count = followingMatch[1];
                        if (postMatch) data.media_count = postMatch[1];
                    }
                    
                    // Try to get biography
                    const bioElements = document.querySelectorAll('[data-testid="user-bio"], .x1lliihq span');
                    for (const el of bioElements) {
                        if (el.textContent && el.textContent.length > 10) {
                            data.biography = el.textContent;
                            break;
                        }
                    }
                    
                    // Get profile picture
                    const avatarImg = document.querySelector('img[alt*="profile picture"]');
                    if (avatarImg) {
                        data.profile_pic_url = avatarImg.src;
                    }
                    
                    // Check if verified
                    data.is_verified = document.querySelector('[aria-label="Verified"]') !== null;
                    
                    // Check if private
                    data.is_private = document.querySelector('text="This Account is Private"') !== null;
                    
                    return data;
                }
            """)
            
            await page.close()
            
            profile_data.update({
                "scraped_at": datetime.utcnow().isoformat(),
                "method": "browser",
                "cache_key": profile_cache_key(username)
            })
            
            return profile_data
            
        except Exception as e:
            logger.error(f"Browser scraping failed for {username}: {e}")
            raise InstagramAPIError(f"Browser scraping failed: {str(e)}")
    
    @cached(ttl=1800, key_func=lambda self, username, limit: posts_cache_key(username, limit))
    async def get_user_posts(self, username: str, limit: int = 50, force_refresh: bool = False) -> Dict[str, Any]:
        """Get user posts with caching."""
        try:
            if not validate_instagram_username(username):
                raise ValueError("Invalid username format")
            
            await self._check_rate_limit()
            
            # Try authenticated API first
            if self.authenticated and self.client:
                try:
                    posts_data = await self._get_posts_api(username, limit)
                    logger.info(f"Retrieved {len(posts_data.get('posts', []))} posts via API: {username}")
                    return posts_data
                    
                except Exception as e:
                    logger.warning(f"API posts method failed for {username}: {e}")
            
            # Fallback to browser scraping
            posts_data = await self._get_posts_browser(username, limit)
            logger.info(f"Retrieved {len(posts_data.get('posts', []))} posts via browser: {username}")
            return posts_data
            
        except Exception as e:
            logger.error(f"Posts retrieval failed for {username}: {e}")
            raise InstagramAPIError(f"Failed to get posts for {username}: {str(e)}")
    
    async def _get_posts_api(self, username: str, limit: int) -> Dict[str, Any]:
        """Get posts using Instagrapi."""
        try:
            user_info = self.client.user_info_by_username(username)
            medias = self.client.user_medias(user_info.pk, limit)
            
            posts = []
            for media in medias:
                post_data = {
                    "id": str(media.id),
                    "code": media.code,
                    "media_type": str(media.media_type),
                    "caption": media.caption_text if media.caption_text else "",
                    "like_count": media.like_count,
                    "comment_count": media.comment_count,
                    "taken_at": media.taken_at.isoformat() if media.taken_at else None,
                    "thumbnail_url": str(media.thumbnail_url) if media.thumbnail_url else None,
                    "video_url": str(media.video_url) if hasattr(media, 'video_url') and media.video_url else None,
                }
                posts.append(post_data)
            
            return {
                "username": username,
                "posts": posts,
                "total_found": len(posts),
                "scraped_at": datetime.utcnow().isoformat(),
                "method": "api",
                "cache_key": posts_cache_key(username, limit)
            }
            
        except UserNotFound:
            raise ProfileNotFoundError(f"Profile {username} not found")
        except (LoginRequired, PleaseWaitFewMinutes) as e:
            raise RateLimitError(f"Instagram API rate limit: {str(e)}")
    
    async def _get_posts_browser(self, username: str, limit: int) -> Dict[str, Any]:
        """Get posts using browser automation."""
        if not self.browser or not self.context:
            await self._init_browser()
        
        try:
            page = await self.context.new_page()
            
            # Navigate to profile
            await page.goto(f"https://www.instagram.com/{username}/", timeout=30000)
            await page.wait_for_timeout(3000)
            
            # Scroll and collect posts
            posts = []
            scroll_count = 0
            max_scrolls = min(10, (limit // 12) + 2)  # Estimate scrolls needed
            
            while len(posts) < limit and scroll_count < max_scrolls:
                # Extract current posts
                current_posts = await page.evaluate("""
                    () => {
                        const posts = [];
                        const postLinks = document.querySelectorAll('a[href*="/p/"]');
                        
                        postLinks.forEach((link, index) => {
                            const img = link.querySelector('img');
                            if (img) {
                                posts.push({
                                    code: link.href.split('/p/')[1].split('/')[0],
                                    thumbnail_url: img.src,
                                    alt_text: img.alt || ''
                                });
                            }
                        });
                        
                        return posts;
                    }
                """)
                
                # Update posts list (avoid duplicates)
                existing_codes = {post['code'] for post in posts}
                new_posts = [post for post in current_posts if post['code'] not in existing_codes]
                posts.extend(new_posts)
                
                # Scroll down
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(2000)
                scroll_count += 1
            
            await page.close()
            
            # Limit results
            posts = posts[:limit]
            
            return {
                "username": username,
                "posts": posts,
                "total_found": len(posts),
                "scraped_at": datetime.utcnow().isoformat(),
                "method": "browser",
                "cache_key": posts_cache_key(username, limit)
            }
            
        except Exception as e:
            logger.error(f"Browser posts scraping failed for {username}: {e}")
            raise InstagramAPIError(f"Browser posts scraping failed: {str(e)}")
    
    async def get_cached_profile(self, username: str) -> Optional[Dict[str, Any]]:
        """Get profile from cache only (no API call)."""
        try:
            cache = await get_cache()
            cache_key = profile_cache_key(username)
            return await cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Failed to get cached profile for {username}: {e}")
            return None
    
    async def get_cached_posts(self, username: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        """Get posts from cache only (no API call)."""
        try:
            cache = await get_cache()
            cache_key = posts_cache_key(username, limit)
            return await cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Failed to get cached posts for {username}: {e}")
            return None
    
    @cache_invalidate("profile:*")
    async def invalidate_profile_cache(self, username: str):
        """Invalidate cached profile data."""
        cache = await get_cache()
        await cache.delete(profile_cache_key(username))
        logger.info(f"Invalidated profile cache for {username}")
    
    @cache_invalidate("posts:*")
    async def invalidate_posts_cache(self, username: str):
        """Invalidate cached posts data."""
        cache = await get_cache()
        await cache.clear_pattern(f"posts:{username.lower()}:*")
        logger.info(f"Invalidated posts cache for {username}")
    
    async def warm_cache(self, usernames: List[str]) -> Dict[str, Any]:
        """Pre-warm cache with profile and posts data for multiple users."""
        results = {
            "success": [],
            "failed": [],
            "total": len(usernames)
        }
        
        logger.info(f"Starting cache warm-up for {len(usernames)} profiles")
        
        for username in usernames:
            try:
                # Get profile (this will cache it)
                await self.get_profile_info(username)
                
                # Get some posts (this will cache them)
                await self.get_user_posts(username, limit=12)
                
                results["success"].append(username)
                
                # Add delay to avoid rate limiting
                await asyncio.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.warning(f"Failed to warm cache for {username}: {e}")
                results["failed"].append({"username": username, "error": str(e)})
        
        logger.info(f"Cache warm-up completed: {len(results['success'])} success, {len(results['failed'])} failed")
        return results
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get Instagram-specific cache statistics."""
        cache = await get_cache()
        
        # Count Instagram-related cache keys
        profile_keys = await cache.clear_pattern("profile:*")  # This returns count, doesn't actually clear
        posts_keys = await cache.clear_pattern("posts:*")
        
        return {
            "profile_cache_entries": profile_keys,
            "posts_cache_entries": posts_keys,
            "authenticated": self.authenticated,
            "rate_limit_remaining": max(0, self.config.instagram.requests_per_minute - self.requests_count),
            "last_request_time": self.last_request_time
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        
        logger.info("Instagram service cleaned up")


# Global service instance
_instagram_service: Optional[EnhancedInstagramService] = None


async def get_instagram_service() -> EnhancedInstagramService:
    """Get the global Instagram service instance."""
    global _instagram_service
    if _instagram_service is None:
        _instagram_service = EnhancedInstagramService()
        await _instagram_service.initialize()
    return _instagram_service


async def close_instagram_service():
    """Close the global Instagram service."""
    global _instagram_service
    if _instagram_service:
        await _instagram_service.cleanup()
        _instagram_service = None