"""
Instagram Research Agent MCP (IRAM) - Scraping Module

This module handles Instagram content scraping using both Instagrapi (private API)
and Playwright (browser automation) for comprehensive data extraction.
"""

import os
import time
import random
import json
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import subprocess
import socket

from instagrapi import Client
from instagrapi.exceptions import LoginRequired, PleaseWaitFewMinutes, UserNotFound
import requests
from fake_useragent import UserAgent

from .evasion_manager import EvasionManager
from .utils import get_logger, validate_instagram_username

# Optional Playwright imports - graceful fallback if not available
try:
    from playwright.async_api import async_playwright
    from playwright_stealth import stealth
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available - browser-based scraping disabled")

logger = get_logger(__name__)


class ProxyManager:
    """Manages proxy rotation for Instagram scraping."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize proxy manager."""
        self.config = config or {}
        self.proxies: List[Dict[str, str]] = []
        self.current_proxy_index = 0
        self.failed_proxies: set = set()
        self.tor_enabled = False
        
        # Load proxies from config or environment
        self._load_proxies()
        
        # Check Tor availability
        self._check_tor()
        
        logger.info(f"Proxy manager initialized with {len(self.proxies)} proxies")
    
    def _load_proxies(self):
        """Load proxy list from configuration."""
        proxy_list = self.config.get("proxies", [])
        
        # Load from environment variable if available
        if not proxy_list and os.getenv("PROXY_LIST"):
            try:
                proxy_list = json.loads(os.getenv("PROXY_LIST"))
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in PROXY_LIST environment variable")
        
        # Load from file if specified
        proxy_file = self.config.get("proxy_file") or os.getenv("PROXY_FILE")
        if proxy_file and Path(proxy_file).exists():
            try:
                with open(proxy_file, 'r') as f:
                    file_proxies = json.load(f)
                proxy_list.extend(file_proxies)
            except Exception as e:
                logger.warning(f"Failed to load proxies from file {proxy_file}: {e}")
        
        # Parse proxy strings into dictionaries
        for proxy in proxy_list:
            if isinstance(proxy, str):
                # Format: "http://user:pass@host:port"
                self.proxies.append({"http": proxy, "https": proxy})
            elif isinstance(proxy, dict):
                self.proxies.append(proxy)
    
    def _check_tor(self):
        """Check if Tor is available and working."""
        try:
            # Check if Tor SOCKS proxy is running
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 9050))
            sock.close()
            
            if result == 0:
                self.tor_enabled = True
                # Add Tor proxy to the list
                tor_proxy = {
                    "http": "socks5://127.0.0.1:9050",
                    "https": "socks5://127.0.0.1:9050"
                }
                if tor_proxy not in self.proxies:
                    self.proxies.append(tor_proxy)
                logger.info("Tor proxy detected and available")
            else:
                logger.info("Tor proxy not available")
                
        except Exception as e:
            logger.warning(f"Failed to check Tor availability: {e}")
    
    def get_proxy(self) -> Optional[Dict[str, str]]:
        """Get the next available proxy."""
        if not self.proxies:
            return None
        
        # Skip failed proxies
        attempts = 0
        while attempts < len(self.proxies):
            proxy = self.proxies[self.current_proxy_index]
            proxy_key = json.dumps(proxy, sort_keys=True)
            
            if proxy_key not in self.failed_proxies:
                self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
                return proxy
            
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
            attempts += 1
        
        # If all proxies failed, clear failed list and try again
        logger.warning("All proxies marked as failed, clearing failed list")
        self.failed_proxies.clear()
        return self.proxies[0] if self.proxies else None
    
    def mark_proxy_failed(self, proxy: Dict[str, str]):
        """Mark a proxy as failed."""
        proxy_key = json.dumps(proxy, sort_keys=True)
        self.failed_proxies.add(proxy_key)
        logger.warning(f"Proxy marked as failed: {proxy.get('http', 'unknown')}")
    
    def get_new_tor_identity(self):
        """Request new Tor identity (circuit)."""
        if not self.tor_enabled:
            return False
        
        try:
            # Send NEWNYM signal to Tor control port
            import telnetlib
            tn = telnetlib.Telnet('127.0.0.1', 9051)
            tn.read_until(b"Escape character is '^]'.")
            tn.write(b"AUTHENTICATE\r\n")
            tn.read_until(b"250 OK\r\n")
            tn.write(b"SIGNAL NEWNYM\r\n")
            tn.read_until(b"250 OK\r\n")
            tn.close()
            
            # Wait for new circuit
            time.sleep(5)
            logger.info("New Tor identity requested")
            return True
        except Exception as e:
            logger.warning(f"Failed to get new Tor identity: {e}")
            return False


class SessionPool:
    """Manages multiple Instagram client sessions for load balancing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize session pool."""
        self.config = config or {}
        self.sessions: List[Dict[str, Any]] = []
        self.current_session_index = 0
        self.session_cooldowns: Dict[str, datetime] = {}
        
        # Load account credentials
        self.accounts = self._load_accounts()
        
        # Initialize sessions
        self._initialize_sessions()
        
        logger.info(f"Session pool initialized with {len(self.sessions)} sessions")
    
    def _load_accounts(self) -> List[Dict[str, str]]:
        """Load multiple account credentials."""
        accounts = []
        
        # Primary account from config
        primary_username = self.config.get("instagram_username") or os.getenv("INSTAGRAM_USERNAME")
        primary_password = self.config.get("instagram_password") or os.getenv("INSTAGRAM_PASSWORD")
        
        if primary_username and primary_password:
            accounts.append({
                "username": primary_username,
                "password": primary_password,
                "is_primary": True
            })
        
        # Additional accounts from environment or file
        additional_accounts = self.config.get("additional_accounts", [])
        
        accounts_env = os.getenv("INSTAGRAM_ADDITIONAL_ACCOUNTS")
        if accounts_env:
            try:
                additional_accounts.extend(json.loads(accounts_env))
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in INSTAGRAM_ADDITIONAL_ACCOUNTS")
        
        # Load from file if specified
        accounts_file = self.config.get("accounts_file") or os.getenv("INSTAGRAM_ACCOUNTS_FILE")
        if accounts_file and Path(accounts_file).exists():
            try:
                with open(accounts_file, 'r') as f:
                    file_accounts = json.load(f)
                additional_accounts.extend(file_accounts)
            except Exception as e:
                logger.warning(f"Failed to load accounts from {accounts_file}: {e}")
        
        # Add additional accounts
        for account in additional_accounts:
            if isinstance(account, dict) and "username" in account and "password" in account:
                account["is_primary"] = False
                accounts.append(account)
        
        return accounts
    
    def _initialize_sessions(self):
        """Initialize client sessions for all accounts."""
        for account in self.accounts:
            try:
                client = Client()
                client.delay_range = [1, 3]
                
                session_info = {
                    "client": client,
                    "username": account["username"],
                    "password": account["password"],
                    "is_primary": account.get("is_primary", False),
                    "authenticated": False,
                    "last_used": datetime.utcnow(),
                    "request_count": 0,
                    "session_file": f"session_{hashlib.md5(account['username'].encode()).hexdigest()}.json"
                }
                
                self.sessions.append(session_info)
                
            except Exception as e:
                logger.error(f"Failed to initialize session for {account['username']}: {e}")
    
    async def get_authenticated_session(self) -> Optional[Dict[str, Any]]:
        """Get an authenticated session, rotating if necessary."""
        if not self.sessions:
            return None
        
        # Try to find an already authenticated session
        for session in self.sessions:
            if session["authenticated"] and not self._is_session_on_cooldown(session):
                return session
        
        # Try to authenticate a session
        for _ in range(len(self.sessions)):
            session = self.sessions[self.current_session_index]
            self.current_session_index = (self.current_session_index + 1) % len(self.sessions)
            
            if not self._is_session_on_cooldown(session):
                if await self._authenticate_session(session):
                    return session
        
        # If no sessions available, return the primary one (even if on cooldown)
        primary_session = next((s for s in self.sessions if s.get("is_primary")), None)
        return primary_session or self.sessions[0] if self.sessions else None
    
    async def _authenticate_session(self, session: Dict[str, Any]) -> bool:
        """Authenticate a specific session."""
        try:
            client = session["client"]
            username = session["username"]
            password = session["password"]
            session_file = session["session_file"]
            
            # Try to load existing session
            if Path(session_file).exists():
                try:
                    client.load_settings(session_file)
                    client.login(username, password)
                    session["authenticated"] = True
                    logger.info(f"Session loaded for {username}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load session for {username}: {e}")
                    Path(session_file).unlink(missing_ok=True)
            
            # Fresh login
            if client.login(username, password):
                client.dump_settings(session_file)
                session["authenticated"] = True
                logger.info(f"Successfully authenticated {username}")
                return True
            else:
                logger.error(f"Authentication failed for {username}")
                return False
                
        except Exception as e:
            logger.error(f"Session authentication error for {session['username']}: {e}")
            # Set cooldown for failed authentication
            self.session_cooldowns[session["username"]] = datetime.utcnow() + timedelta(minutes=30)
            return False
    
    def _is_session_on_cooldown(self, session: Dict[str, Any]) -> bool:
        """Check if session is on cooldown."""
        username = session["username"]
        if username in self.session_cooldowns:
            return datetime.utcnow() < self.session_cooldowns[username]
        return False
    
    def mark_session_rate_limited(self, session: Dict[str, Any], cooldown_minutes: int = 60):
        """Mark session as rate limited with cooldown."""
        username = session["username"]
        self.session_cooldowns[username] = datetime.utcnow() + timedelta(minutes=cooldown_minutes)
        logger.warning(f"Session {username} marked as rate limited for {cooldown_minutes} minutes")
    
    def update_session_usage(self, session: Dict[str, Any]):
        """Update session usage statistics."""
        session["last_used"] = datetime.utcnow()
        session["request_count"] += 1


class UserAgentRotator:
    """Manages user agent rotation for web scraping."""
    
    def __init__(self):
        """Initialize user agent rotator."""
        self.ua = UserAgent()
        self.custom_agents = [
            "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent."""
        try:
            # Try to get a random user agent from fake_useragent
            if random.random() < 0.7:  # 70% chance to use fake_useragent
                return self.ua.random
            else:
                return random.choice(self.custom_agents)
        except Exception:
            # Fallback to custom agents if fake_useragent fails
            return random.choice(self.custom_agents)
    
    def get_mobile_user_agent(self) -> str:
        """Get a mobile user agent."""
        mobile_agents = [agent for agent in self.custom_agents if "Mobile" in agent]
        return random.choice(mobile_agents) if mobile_agents else self.custom_agents[0]


class InstagramScraper:
    """Main Instagram scraping class with multiple data sources."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Instagram scraper."""
        self.config = config or {}
        
        # Initialize enhanced components
        self.proxy_manager = ProxyManager(config)
        self.session_pool = SessionPool(config)
        self.user_agent_rotator = UserAgentRotator()
        self.evasion_manager = EvasionManager(config)
        
        # Legacy single client for backward compatibility
        self.client = None
        self.authenticated = False
        self.last_request_time = 0
        
        # Initialize browser context
        self.playwright = None
        self.browser = None
        self.context = None
        
        # Enhanced features flags
        self.enable_proxy_rotation = self.config.get("enable_proxy_rotation", True)
        self.enable_session_pooling = self.config.get("enable_session_pooling", True)
        self.enable_user_agent_rotation = self.config.get("enable_user_agent_rotation", True)
        
        logger.info("Enhanced Instagram scraper initialized")
    
    async def authenticate(self) -> bool:
        """Authenticate with Instagram using credentials."""
        try:
            if self.enable_session_pooling:
                # Use session pool for authentication
                session = await self.session_pool.get_authenticated_session()
                if session:
                    self.client = session["client"]
                    self.authenticated = True
                    logger.info("Successfully authenticated using session pool")
                    return True
                else:
                    logger.warning("No authenticated sessions available")
                    return False
            else:
                # Legacy single client authentication
                return await self._legacy_authenticate()
                
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    async def _legacy_authenticate(self) -> bool:
        """Legacy authentication method for backward compatibility."""
        username = self.config.get("instagram_username") or os.getenv("INSTAGRAM_USERNAME")
        password = self.config.get("instagram_password") or os.getenv("INSTAGRAM_PASSWORD")
        
        if not username or not password:
            logger.warning("Instagram credentials not provided, using public-only mode")
            return False
        
        if not self.client:
            self.client = Client()
            self.client.delay_range = [1, 3]
        
        # Apply evasion delay
        await self.evasion_manager.apply_delay()
        
        # Login with Instagrapi
        success = self.client.login(username, password)
        if success:
            self.authenticated = True
            logger.info("Successfully authenticated with Instagram (legacy)")
            return True
        else:
            logger.error("Failed to authenticate with Instagram")
            return False
    
    async def get_session(self) -> Optional[Dict[str, Any]]:
        """Get an authenticated session for making requests."""
        if self.enable_session_pooling:
            return await self.session_pool.get_authenticated_session()
        else:
            if not self.authenticated:
                await self.authenticate()
            return {"client": self.client, "authenticated": self.authenticated} if self.client else None
    
    async def init_browser(self) -> bool:
        """Initialize Playwright browser for fallback scraping."""
        try:
            self.playwright = await async_playwright().start()
            
            # Enhanced browser launch arguments for stealth
            browser_args = [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-extensions',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-first-run',
                '--disable-default-apps',
                '--disable-features=TranslateUI',
                '--disable-ipc-flooding-protection',
                '--enable-features=NetworkService,NetworkServiceLogging',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding'
            ]
            
            # Add proxy to browser args if available
            if self.enable_proxy_rotation:
                proxy = self.proxy_manager.get_proxy()
                if proxy and proxy.get("http"):
                    proxy_url = proxy["http"]
                    if proxy_url.startswith("socks5://"):
                        browser_args.append(f'--proxy-server={proxy_url}')
                    elif proxy_url.startswith("http://") or proxy_url.startswith("https://"):
                        browser_args.append(f'--proxy-server={proxy_url}')
            
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=browser_args
            )
            
            # Get random user agent
            user_agent = (
                self.user_agent_rotator.get_random_user_agent() 
                if self.enable_user_agent_rotation
                else "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            
            # Create context with enhanced stealth settings
            self.context = await self.browser.new_context(
                user_agent=user_agent,
                viewport={
                    "width": random.randint(1200, 1920),
                    "height": random.randint(800, 1080)
                },
                locale="en-US",
                timezone_id="America/New_York",
                geolocation={"longitude": -74.0060, "latitude": 40.7128},  # NYC
                permissions=["geolocation"],
                extra_http_headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Cache-Control": "max-age=0"
                }
            )
            
            # Apply stealth measures
            await stealth_async(self.context)
            
            # Add stealth JavaScript
            await self.context.add_init_script("""
                // Remove webdriver property
                delete Object.getPrototypeOf(navigator).webdriver;
                
                // Mock languages and plugins
                Object.defineProperty(navigator, 'languages', {
                    get: function() { return ['en-US', 'en']; }
                });
                
                Object.defineProperty(navigator, 'plugins', {
                    get: function() { return [1, 2, 3, 4, 5]; }
                });
                
                // Override the `plugins` property to use a custom getter.
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)
            
            logger.info(f"Enhanced browser initialized with user agent: {user_agent[:50]}...")
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