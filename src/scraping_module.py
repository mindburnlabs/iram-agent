"""
Instagram Research Agent MCP (IRAM) - Scraping Module

This module handles Instagram content scraping using both Instagrapi (private API)
and Playwright (browser automation) for comprehensive data extraction.
"""

import os
import time
import random
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import asyncio

from instagrapi import Client
from instagrapi.exceptions import LoginRequired, PleaseWaitFewMinutes, UserNotFound
from playwright.async_api import async_playwright
import requests

from .evasion_manager import EvasionManager
from .utils import get_logger, validate_instagram_username

logger = get_logger(__name__)


class InstagramScraper:
    """Main Instagram scraping class with multiple data sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Instagram scraper."""
        self.config = config or {}
        
        # Initialize Instagrapi client
        self.client = Client()
        self.client.delay_range = [1, 3]
        
        # Initialize evasion manager
        self.evasion_manager = EvasionManager(config)
        
        # Session management
        self.authenticated = False
        self.last_request_time = 0
        
        # Initialize browser context
        self.playwright = None
        self.browser = None
        self.context = None
        
        logger.info("Instagram scraper initialized")
    
    async def authenticate(self) -> bool:
        """Authenticate with Instagram using credentials."""
        try:
            username = self.config.get("instagram_username") or os.getenv("INSTAGRAM_USERNAME")
            password = self.config.get("instagram_password") or os.getenv("INSTAGRAM_PASSWORD")
            
            if not username or not password:
                logger.warning("Instagram credentials not provided, using public-only mode")
                return False
            
            # Apply evasion delay
            await self.evasion_manager.apply_delay()
            
            # Login with Instagrapi
            success = self.client.login(username, password)
            if success:
                self.authenticated = True
                logger.info("Successfully authenticated with Instagram")
                return True
            else:
                logger.error("Failed to authenticate with Instagram")
                return False
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    async def init_browser(self) -> bool:
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
            
            self.context = await self.browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 720}
            )
            
            logger.info("Browser initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            return False
    
    def get_profile_info(self, username: str) -> Dict[str, Any]:
        """Get comprehensive profile information."""
        try:
            if not validate_instagram_username(username):
                return {"error": "Invalid username format"}
            
            # Try authenticated API first
            if self.authenticated:
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
                        "scraped_at": datetime.utcnow().isoformat(),
                        "method": "instagrapi"
                    }
                except Exception as e:
                    logger.warning(f"Instagrapi failed for {username}, trying browser method: {e}")
            
            # Fallback to browser scraping
            return asyncio.run(self._scrape_profile_browser(username))
            
        except Exception as e:
            logger.error(f"Profile scraping failed for {username}: {e}")
            return {"error": str(e), "username": username}
    
    async def _scrape_profile_browser(self, username: str) -> Dict[str, Any]:
        """Scrape profile using browser automation."""
        try:
            if not self.browser:
                await self.init_browser()
            
            page = await self.context.new_page()
            await page.goto(f"https://www.instagram.com/{username}/")
            
            # Wait for page load
            await page.wait_for_timeout(3000)
            
            # Extract profile data
            profile_data = await page.evaluate("""
                () => {
                    const data = {};
                    
                    // Get basic info from meta tags and page structure
                    const title = document.title;
                    data.username = title.split('(')[0].replace('@', '').trim();
                    
                    // Try to extract stats
                    const statsElements = document.querySelectorAll('span[title], a span');
                    const stats = Array.from(statsElements).map(el => el.textContent).filter(text => text && /\\d/.test(text));
                    
                    if (stats.length >= 3) {
                        data.media_count = stats[0];
                        data.followers_count = stats[1];
                        data.following_count = stats[2];
                    }
                    
                    // Get biography
                    const bioElement = document.querySelector('div.-vDIg span, div[data-testid="user-bio"]');
                    if (bioElement) {
                        data.biography = bioElement.textContent;
                    }
                    
                    // Get profile picture
                    const avatarImg = document.querySelector('img[alt*="profile picture"], img[data-testid="user-avatar"]');
                    if (avatarImg) {
                        data.profile_pic_url = avatarImg.src;
                    }
                    
                    return data;
                }
            """)
            
            await page.close()
            
            profile_data.update({
                "scraped_at": datetime.utcnow().isoformat(),
                "method": "browser"
            })
            
            return profile_data
            
        except Exception as e:
            logger.error(f"Browser scraping failed for {username}: {e}")
            return {"error": str(e), "username": username, "method": "browser"}
    
    def get_user_posts(self, username: str, limit: int = 50) -> Dict[str, Any]:
        """Get user's recent posts."""
        try:
            if not self.authenticated:
                return {"error": "Authentication required for post scraping", "username": username}
            
            user_id = self.client.user_id_from_username(username)
            medias = self.client.user_medias(user_id, limit)
            
            posts = []
            for media in medias:
                post_data = {
                    "id": media.id,
                    "code": media.code,
                    "media_type": media.media_type,
                    "caption": media.caption_text if media.caption_text else "",
                    "like_count": media.like_count,
                    "comment_count": media.comment_count,
                    "taken_at": media.taken_at.isoformat() if media.taken_at else None,
                    "thumbnail_url": str(media.thumbnail_url) if media.thumbnail_url else None,
                }
                
                # Add location if available
                if media.location:
                    post_data["location"] = {
                        "name": media.location.name,
                        "address": media.location.address
                    }
                
                posts.append(post_data)
            
            return {
                "username": username,
                "posts": posts,
                "total_posts": len(posts),
                "scraped_at": datetime.utcnow().isoformat(),
                "method": "instagrapi"
            }
            
        except Exception as e:
            logger.error(f"Post scraping failed for {username}: {e}")
            return {"error": str(e), "username": username}
    
    def get_user_stories(self, username: str) -> Dict[str, Any]:
        """Get user's current stories."""
        try:
            if not self.authenticated:
                return {"error": "Authentication required for story scraping", "username": username}
            
            user_id = self.client.user_id_from_username(username)
            stories = self.client.user_stories(user_id)
            
            story_data = []
            for story in stories:
                story_info = {
                    "id": story.id,
                    "media_type": story.media_type,
                    "taken_at": story.taken_at.isoformat() if story.taken_at else None,
                    "expiring_at": story.expiring_at.isoformat() if story.expiring_at else None,
                    "thumbnail_url": str(story.thumbnail_url) if story.thumbnail_url else None,
                }
                story_data.append(story_info)
            
            return {
                "username": username,
                "stories": story_data,
                "total_stories": len(story_data),
                "scraped_at": datetime.utcnow().isoformat(),
                "method": "instagrapi"
            }
            
        except Exception as e:
            logger.error(f"Story scraping failed for {username}: {e}")
            return {"error": str(e), "username": username}
    
    def get_followers(self, username: str, limit: int = 100) -> Dict[str, Any]:
        """Get user's followers."""
        try:
            if not self.authenticated:
                return {"error": "Authentication required for follower scraping", "username": username}
            
            user_id = self.client.user_id_from_username(username)
            followers = self.client.user_followers(user_id, amount=limit)
            
            follower_data = []
            for follower_id, follower_info in followers.items():
                follower_data.append({
                    "id": follower_id,
                    "username": follower_info.username,
                    "full_name": follower_info.full_name,
                    "is_verified": follower_info.is_verified,
                    "profile_pic_url": str(follower_info.profile_pic_url)
                })
            
            return {
                "username": username,
                "followers": follower_data,
                "total_followers": len(follower_data),
                "scraped_at": datetime.utcnow().isoformat(),
                "method": "instagrapi"
            }
            
        except Exception as e:
            logger.error(f"Follower scraping failed for {username}: {e}")
            return {"error": str(e), "username": username}
    
    def search_users(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for users matching the query."""
        try:
            if self.authenticated:
                users = self.client.search_users(query, amount=limit)
                return [
                    {
                        "id": user.pk,
                        "username": user.username,
                        "full_name": user.full_name,
                        "is_verified": user.is_verified,
                        "follower_count": user.follower_count,
                        "profile_pic_url": str(user.profile_pic_url)
                    }
                    for user in users
                ]
            else:
                # Use public search methods
                return self._public_user_search(query, limit)
                
        except Exception as e:
            logger.error(f"User search failed for query '{query}': {e}")
            return []
    
    def _public_user_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Public user search without authentication."""
        try:
            # This would implement a public search method
            # For now, return empty list as public search is limited
            logger.warning("Public user search not fully implemented")
            return []
            
        except Exception as e:
            logger.error(f"Public user search failed: {e}")
            return []
    
    def search_hashtags(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for hashtags."""
        try:
            if self.authenticated:
                hashtags = self.client.search_hashtags(query, amount=limit)
                return [
                    {
                        "id": hashtag.id,
                        "name": hashtag.name,
                        "media_count": hashtag.media_count
                    }
                    for hashtag in hashtags
                ]
            else:
                logger.warning("Hashtag search requires authentication")
                return []
                
        except Exception as e:
            logger.error(f"Hashtag search failed for query '{query}': {e}")
            return []
    
    def search_general(self, query: str) -> Dict[str, Any]:
        """General search combining users and hashtags."""
        try:
            users = self.search_users(query, 10)
            hashtags = self.search_hashtags(query, 10)
            
            return {
                "query": query,
                "users": users,
                "hashtags": hashtags,
                "total_results": len(users) + len(hashtags),
                "scraped_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"General search failed for query '{query}': {e}")
            return {"error": str(e), "query": query}
    
    def scrape_content(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Generic content scraping method."""
        target = query.get("target", "")
        content_type = query.get("content_type", "profile")
        limit = query.get("limit", 50)
        
        if content_type == "profile":
            return self.get_profile_info(target)
        elif content_type == "posts":
            return self.get_user_posts(target, limit)
        elif content_type == "stories":
            return self.get_user_stories(target)
        elif content_type == "followers":
            return self.get_followers(target, limit)
        else:
            return {"error": f"Unknown content type: {content_type}"}
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            logger.info("Scraper resources cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to clean up resources."""
        try:
            if self.browser or self.playwright:
                asyncio.create_task(self.cleanup())
        except:
            pass