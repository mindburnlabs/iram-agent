"""
Firecrawl Discovery Tool for IRAM

This module implements a web research tool using Firecrawl with rate limiting,
usage safeguards, and a fallback search strategy.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional

from firecrawl import FirecrawlApp

from .utils import get_logger
from .config import get_config

logger = get_logger(__name__)


class FirecrawlDiscoveryTool:
    """Web research tool using Firecrawl."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Firecrawl discovery tool."""
        self.config = get_config()
        self.custom_config = config or {}
        
        # Initialize Firecrawl app
        self.app = None
        if self.config.firecrawl.api_key:
            self.app = FirecrawlApp(api_key=self.config.firecrawl.api_key)
        
        # Rate limiting and usage safeguards
        self.rate_limit_per_minute = 60
        self.request_timestamps = []
        
        logger.info("Firecrawl discovery tool initialized")

    async def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform a web search using Firecrawl."""
        try:
            if not self.app:
                return {"error": "Firecrawl API key not configured"}

            # Apply rate limiting
            if not await self._is_rate_limit_ok():
                return {"error": "Rate limit exceeded. Please wait."}

            # Perform search
            search_result = self.app.search(query, page_options={"limit": max_results})
            self.request_timestamps.append(asyncio.get_event_loop().time())

            return search_result

        except Exception as e:
            logger.error(f"Firecrawl search failed: {e}")
            return {"error": str(e)}

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape a URL using Firecrawl."""
        try:
            if not self.app:
                return {"error": "Firecrawl API key not configured"}

            # Apply rate limiting
            if not await self._is_rate_limit_ok():
                return {"error": "Rate limit exceeded. Please wait."}

            # Scrape URL
            scraped_data = self.app.scrape_url(url)
            self.request_timestamps.append(asyncio.get_event_loop().time())

            return scraped_data

        except Exception as e:
            logger.error(f"Firecrawl scrape failed: {e}")
            return {"error": str(e)}

    async def _is_rate_limit_ok(self) -> bool:
        """Check if the request is within the rate limit."""
        now = asyncio.get_event_loop().time()
        # Remove timestamps older than 60 seconds
        self.request_timestamps = [t for t in self.request_timestamps if now - t < 60]
        
        if len(self.request_timestamps) < self.rate_limit_per_minute:
            return True
        else:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about Firecrawl usage."""
        return {
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "requests_in_last_minute": len(self.request_timestamps)
        }


# Global Firecrawl tool instance
_firecrawl_tool: Optional[FirecrawlDiscoveryTool] = None

def get_firecrawl_tool() -> FirecrawlDiscoveryTool:
    """Get global Firecrawl tool instance."""
    global _firecrawl_tool
    if _firecrawl_tool is None:
        _firecrawl_tool = FirecrawlDiscoveryTool()
    return _firecrawl_tool
