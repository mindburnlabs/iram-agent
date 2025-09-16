"""
Instagram Tools Module for MCP Server

This module consolidates Instagram functionality including DMs, media downloads,
business insights, and posting capabilities. Integrates functionality from:
- trypeggy/instagram_dm_mcp
- jlbadano/ig-mcp

Provides unified JSON-RPC tools for MCP server integration.
"""

import os
import json
import asyncio
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, BinaryIO
from pathlib import Path
import base64
import mimetypes

from instagrapi import Client
from instagrapi.exceptions import (
    LoginRequired, 
    ChallengeRequired, 
    PleaseWaitFewMinutes, 
    RateLimitError,
    MediaNotFound
)
from PIL import Image
import requests
from io import BytesIO

from .utils import get_logger
from .config import get_config
from .evasion_manager import EvasionManager

logger = get_logger(__name__)


class InstagramMCPTools:
    """Unified Instagram tools for MCP server integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Instagram MCP tools."""
        self.config = get_config()
        self.custom_config = config or {}
        
        # Initialize Instagram client
        self.client = None
        self.session_file = self.config.instagram.session_file
        self.logged_in = False
        
        # Initialize evasion manager
        self.evasion_manager = EvasionManager(config=self.custom_config)
        
        # Rate limiting
        self.last_request_time = 0
        self.request_delay = 2.0  # Minimum delay between requests
        
        logger.info("Instagram MCP Tools initialized")
    
    async def _ensure_client(self) -> Client:
        """Ensure Instagram client is initialized and logged in."""
        if self.client is None:
            self.client = Client()
            
            # Configure client settings
            self.client.request_timeout = 10
            self.client.delay_range = [1, 3]
            
            # Apply evasion settings
            await self._apply_evasion_settings()
        
        if not self.logged_in:
            await self._login()
        
        return self.client
    
    async def _apply_evasion_settings(self):
        """Apply evasion settings to Instagram client."""
        try:
            # Set user agent
            user_agent = self.evasion_manager.get_user_agent()
            self.client.set_user_agent(user_agent)
            
            # Set device settings for authenticity
            device = {
                'app_version': '239.0.0.10.109',
                'android_version': '10',
                'android_release': '10.0',
                'dpi': '420dpi',
                'resolution': '1080x2340',
                'manufacturer': 'Samsung',
                'device': 'SM-G973F',
                'model': 'galaxy_s10',
                'cpu': 'exynos9820'
            }
            self.client.set_device(device)
            
            logger.debug("Applied evasion settings to Instagram client")
            
        except Exception as e:
            logger.warning(f"Failed to apply evasion settings: {e}")
    
    async def _login(self):
        """Login to Instagram with proper error handling."""
        try:
            username = self.config.instagram.username
            password = self.config.instagram.password
            
            if not username or not password:
                raise ValueError("Instagram credentials not configured")
            
            # Try to load existing session
            session_path = Path(self.session_file)
            if session_path.exists():
                try:
                    self.client.load_settings(str(session_path))
                    self.client.login(username, password)
                    self.logged_in = True
                    logger.info("Logged in using existing session")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load existing session: {e}")
                    session_path.unlink(missing_ok=True)
            
            # Fresh login
            self.client.login(username, password)
            
            # Save session
            self.client.dump_settings(str(session_path))
            self.logged_in = True
            
            logger.info(f"Successfully logged in as @{username}")
            
        except ChallengeRequired as e:
            logger.error(f"Instagram challenge required: {e}")
            raise ValueError("Instagram login requires verification. Please complete the challenge manually.")
        
        except PleaseWaitFewMinutes:
            logger.error("Instagram rate limiting detected")
            raise ValueError("Instagram is rate limiting. Please wait and try again.")
        
        except Exception as e:
            logger.error(f"Instagram login failed: {e}")
            raise ValueError(f"Login failed: {str(e)}")
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            delay = self.request_delay - time_since_last
            await asyncio.sleep(delay)
        
        self.last_request_time = time.time()
    
    # ==== USER PROFILE TOOLS ====
    
    async def fetch_profile(self, username: str) -> Dict[str, Any]:
        """Fetch comprehensive user profile information."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            # Remove @ if present
            username = username.lstrip('@')
            
            # Get user info
            user_info = client.user_info_by_username(username)
            
            # Get additional metrics if it's a business account
            insights = None
            if user_info.is_business:
                try:
                    insights = await self.get_business_insights(username)
                except Exception as e:
                    logger.warning(f"Failed to get business insights: {e}")
            
            return {
                "success": True,
                "profile": {
                    "pk": user_info.pk,
                    "username": user_info.username,
                    "full_name": user_info.full_name,
                    "biography": user_info.biography,
                    "external_url": user_info.external_url,
                    "follower_count": user_info.follower_count,
                    "following_count": user_info.following_count,
                    "media_count": user_info.media_count,
                    "is_private": user_info.is_private,
                    "is_verified": user_info.is_verified,
                    "is_business": user_info.is_business,
                    "profile_pic_url": user_info.profile_pic_url,
                    "category": user_info.category,
                },
                "insights": insights,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch profile for @{username}: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ==== DIRECT MESSAGING TOOLS ====
    
    async def send_direct_message(
        self, 
        username: str, 
        message: str, 
        media_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a direct message to a user."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            username = username.lstrip('@')
            user_id = client.user_id_from_username(username)
            
            # Send message with or without media
            if media_path and Path(media_path).exists():
                # Send with media
                result = client.direct_send_photo(
                    path=media_path,
                    user_ids=[user_id],
                    text=message
                )
            else:
                # Send text only
                result = client.direct_send(message, user_ids=[user_id])
            
            return {
                "success": True,
                "message_id": result.id if hasattr(result, 'id') else None,
                "recipient": username,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to send DM to @{username}: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_direct_messages(
        self, 
        username: Optional[str] = None, 
        limit: int = 20
    ) -> Dict[str, Any]:
        """Get direct messages from inbox or specific user."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            if username:
                # Get messages from specific user
                username = username.lstrip('@')
                user_id = client.user_id_from_username(username)
                thread = client.direct_thread(user_id)
                messages = thread.messages[:limit] if thread and thread.messages else []
            else:
                # Get all inbox messages
                threads = client.direct_threads(amount=limit)
                messages = []
                for thread in threads[:5]:  # Limit to 5 threads
                    if thread.messages:
                        messages.extend(thread.messages[:5])  # 5 messages per thread
            
            formatted_messages = []
            for msg in messages[:limit]:
                formatted_messages.append({
                    "id": msg.id,
                    "text": msg.text or "",
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "user_id": msg.user_id,
                    "is_sent_by_viewer": msg.is_sent_by_viewer,
                    "media_type": msg.item_type
                })
            
            return {
                "success": True,
                "messages": formatted_messages,
                "count": len(formatted_messages),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get direct messages: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ==== MEDIA TOOLS ====
    
    async def download_media(
        self, 
        media_url_or_shortcode: str, 
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Download media (photo/video) from Instagram."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            # Determine if it's a URL or shortcode
            if 'instagram.com' in media_url_or_shortcode:
                # Extract shortcode from URL
                parts = media_url_or_shortcode.split('/')
                shortcode = None
                for i, part in enumerate(parts):
                    if part in ['p', 'reel', 'tv']:
                        shortcode = parts[i + 1]
                        break
                if not shortcode:
                    raise ValueError("Could not extract shortcode from URL")
            else:
                shortcode = media_url_or_shortcode
            
            # Get media info
            media = client.media_info_by_shortcode(shortcode)
            
            # Create output directory
            if not output_dir:
                output_dir = tempfile.mkdtemp(prefix="instagram_media_")
            else:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            downloaded_files = []
            
            if media.media_type == 1:  # Photo
                photo_path = client.photo_download(media.pk, output_dir)
                downloaded_files.append(str(photo_path))
            elif media.media_type == 2:  # Video
                video_path = client.video_download(media.pk, output_dir)
                downloaded_files.append(str(video_path))
            elif media.media_type == 8:  # Carousel (multiple media)
                album_path = client.album_download(media.pk, output_dir)
                if isinstance(album_path, list):
                    downloaded_files.extend([str(p) for p in album_path])
                else:
                    downloaded_files.append(str(album_path))
            
            return {
                "success": True,
                "media_id": media.pk,
                "shortcode": shortcode,
                "media_type": media.media_type,
                "downloaded_files": downloaded_files,
                "output_dir": output_dir,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to download media: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def upload_photo(
        self, 
        image_path: str, 
        caption: str = "",
        location: Optional[Dict[str, Any]] = None,
        usertags: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Upload a photo to Instagram."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            if not Path(image_path).exists():
                raise ValueError(f"Image file not found: {image_path}")
            
            # Upload photo
            media = client.photo_upload(
                path=image_path,
                caption=caption,
                location=location,
                usertags=usertags
            )
            
            return {
                "success": True,
                "media_id": media.pk,
                "shortcode": media.code,
                "caption": caption,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to upload photo: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def upload_video(
        self, 
        video_path: str, 
        caption: str = "",
        thumbnail_path: Optional[str] = None,
        location: Optional[Dict[str, Any]] = None,
        usertags: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Upload a video to Instagram."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            if not Path(video_path).exists():
                raise ValueError(f"Video file not found: {video_path}")
            
            # Upload video
            media = client.video_upload(
                path=video_path,
                caption=caption,
                thumbnail=thumbnail_path,
                location=location,
                usertags=usertags
            )
            
            return {
                "success": True,
                "media_id": media.pk,
                "shortcode": media.code,
                "caption": caption,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ==== BUSINESS INSIGHTS TOOLS ====
    
    async def get_business_insights(
        self, 
        username: Optional[str] = None,
        period: str = "week"  # day, week, month
    ) -> Dict[str, Any]:
        """Get business account insights."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            if username:
                username = username.lstrip('@')
                user_id = client.user_id_from_username(username)
            else:
                user_id = client.user_id
            
            # Get account insights
            insights = client.insights_account(user_id, period)
            
            # Get media insights for recent posts
            medias = client.user_medias(user_id, amount=10)
            media_insights = []
            
            for media in medias[:5]:  # Limit to 5 recent posts
                try:
                    media_insight = client.insights_media(media.pk)
                    media_insights.append({
                        "media_id": media.pk,
                        "shortcode": media.code,
                        "insights": media_insight
                    })
                except Exception as e:
                    logger.warning(f"Failed to get insights for media {media.pk}: {e}")
            
            return {
                "success": True,
                "account_insights": insights,
                "media_insights": media_insights,
                "period": period,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get business insights: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ==== STORY TOOLS ====
    
    async def get_user_stories(self, username: str) -> Dict[str, Any]:
        """Get user's current stories."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            username = username.lstrip('@')
            user_id = client.user_id_from_username(username)
            stories = client.user_stories(user_id)
            
            formatted_stories = []
            for story in stories:
                formatted_stories.append({
                    "id": story.pk,
                    "media_type": story.media_type,
                    "thumbnail_url": story.thumbnail_url,
                    "video_url": story.video_url if hasattr(story, 'video_url') else None,
                    "taken_at": story.taken_at.isoformat() if story.taken_at else None,
                    "expires_at": story.expiring_at.isoformat() if story.expiring_at else None
                })
            
            return {
                "success": True,
                "username": username,
                "stories": formatted_stories,
                "count": len(formatted_stories),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get stories for @{username}: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def upload_story_photo(
        self, 
        image_path: str,
        mentions: Optional[List[str]] = None,
        hashtags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Upload a photo story."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            if not Path(image_path).exists():
                raise ValueError(f"Image file not found: {image_path}")
            
            # Prepare mentions
            story_mentions = []
            if mentions:
                for username in mentions:
                    user_id = client.user_id_from_username(username.lstrip('@'))
                    story_mentions.append({
                        "user_id": user_id,
                        "x": 0.5,  # Center position
                        "y": 0.5,
                        "width": 0.4,
                        "height": 0.2
                    })
            
            # Prepare hashtags
            story_hashtags = []
            if hashtags:
                for tag in hashtags[:3]:  # Limit to 3 hashtags
                    story_hashtags.append({
                        "hashtag": tag.lstrip('#'),
                        "x": 0.5,
                        "y": 0.8,
                        "width": 0.3,
                        "height": 0.1
                    })
            
            # Upload story
            story = client.photo_upload_to_story(
                path=image_path,
                mentions=story_mentions,
                hashtags=story_hashtags
            )
            
            return {
                "success": True,
                "story_id": story.pk,
                "mentions": mentions or [],
                "hashtags": hashtags or [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to upload story: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ==== ENGAGEMENT TOOLS ====
    
    async def like_media(self, media_id_or_shortcode: str) -> Dict[str, Any]:
        """Like a media post."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            # Get media ID if shortcode provided
            if not media_id_or_shortcode.isdigit():
                media = client.media_info_by_shortcode(media_id_or_shortcode)
                media_id = media.pk
            else:
                media_id = media_id_or_shortcode
            
            result = client.media_like(media_id)
            
            return {
                "success": True,
                "media_id": media_id,
                "liked": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to like media: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def comment_media(
        self, 
        media_id_or_shortcode: str, 
        comment: str
    ) -> Dict[str, Any]:
        """Comment on a media post."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            # Get media ID if shortcode provided
            if not media_id_or_shortcode.isdigit():
                media = client.media_info_by_shortcode(media_id_or_shortcode)
                media_id = media.pk
            else:
                media_id = media_id_or_shortcode
            
            result = client.media_comment(media_id, comment)
            
            return {
                "success": True,
                "media_id": media_id,
                "comment": comment,
                "comment_id": result.pk if hasattr(result, 'pk') else None,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to comment on media: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def follow_user(self, username: str) -> Dict[str, Any]:
        """Follow a user."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            username = username.lstrip('@')
            user_id = client.user_id_from_username(username)
            result = client.user_follow(user_id)
            
            return {
                "success": True,
                "username": username,
                "followed": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to follow @{username}: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    # ==== UTILITY METHODS ====
    
    async def get_media_info(self, media_id_or_shortcode: str) -> Dict[str, Any]:
        """Get detailed media information."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            if media_id_or_shortcode.isdigit():
                media = client.media_info(media_id_or_shortcode)
            else:
                media = client.media_info_by_shortcode(media_id_or_shortcode)
            
            return {
                "success": True,
                "media": {
                    "pk": media.pk,
                    "id": media.id,
                    "code": media.code,
                    "taken_at": media.taken_at.isoformat() if media.taken_at else None,
                    "media_type": media.media_type,
                    "thumbnail_url": media.thumbnail_url,
                    "like_count": media.like_count,
                    "comment_count": media.comment_count,
                    "caption_text": media.caption_text,
                    "user": {
                        "username": media.user.username,
                        "full_name": media.user.full_name,
                        "is_verified": media.user.is_verified
                    }
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get media info: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def search_users_advanced(
        self, 
        query: str, 
        limit: int = 20
    ) -> Dict[str, Any]:
        """Advanced user search with filtering."""
        try:
            await self._rate_limit()
            client = await self._ensure_client()
            
            users = client.search_users(query)[:limit]
            
            formatted_users = []
            for user in users:
                formatted_users.append({
                    "pk": user.pk,
                    "username": user.username,
                    "full_name": user.full_name,
                    "is_verified": user.is_verified,
                    "is_private": user.is_private,
                    "follower_count": user.follower_count,
                    "profile_pic_url": user.profile_pic_url
                })
            
            return {
                "success": True,
                "query": query,
                "users": formatted_users,
                "count": len(formatted_users),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to search users: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def logout(self) -> Dict[str, Any]:
        """Logout and cleanup."""
        try:
            if self.client and self.logged_in:
                self.client.logout()
                self.logged_in = False
                
                # Remove session file
                session_path = Path(self.session_file)
                if session_path.exists():
                    session_path.unlink()
            
            return {
                "success": True,
                "message": "Logged out successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to logout: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Factory function
def create_instagram_tools(config: Optional[Dict[str, Any]] = None) -> InstagramMCPTools:
    """Create Instagram MCP tools instance."""
    return InstagramMCPTools(config)