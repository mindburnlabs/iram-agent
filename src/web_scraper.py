"""
Web Scraping Module using Firecrawl

This module provides web scraping capabilities using the Firecrawl Python SDK
for extracting content from websites in markdown and HTML formats.
"""

import os
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FirecrawlApp = None
    FIRECRAWL_AVAILABLE = False

from .utils import get_logger
from .config import get_config

logger = get_logger(__name__)


class WebScraper:
    """Web scraping class using Firecrawl for content extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the web scraper with Firecrawl."""
        self.config = config or {}
        self.app_config = get_config()
        
        # Initialize Firecrawl client
        self.firecrawl_client = None
        self._init_firecrawl()
        
        logger.info(f"Web scraper initialized - Firecrawl available: {FIRECRAWL_AVAILABLE}")
    
    def _init_firecrawl(self):
        """Initialize Firecrawl client with v2.2.0+ features."""
        try:
            if not FIRECRAWL_AVAILABLE:
                logger.warning("Firecrawl package not available. Install with: pip install firecrawl-py")
                return
            
            # Get API key from environment or config
            api_key = (
                self.config.get("firecrawl_api_key") or 
                os.getenv("FIRECRAWL_API_KEY")
            )
            
            if not api_key:
                logger.warning("Firecrawl API key not provided. Web scraping will be disabled.")
                return
            
            # Initialize with MCP integration parameter (v2.2.0)
            client_options = {
                'api_key': api_key,
                'integration': 'iram-agent'  # v2.2.0 MCP integration
            }
            
            self.firecrawl_client = FirecrawlApp(**client_options)
            logger.info("Firecrawl client initialized successfully with MCP integration")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firecrawl client: {e}")
            self.firecrawl_client = None
    
    async def scrape_url(
        self, 
        url: str, 
        formats: List[str] = None,
        include_tags: List[str] = None,
        exclude_tags: List[str] = None,
        only_main_content: bool = True,
        extract_images: bool = False,
        extract_data_attributes: bool = False,
        wait_for: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Scrape a single URL and return the content with v2.1.0+ features.
        
        Args:
            url: URL to scrape
            formats: List of formats to extract (e.g., ['markdown', 'html'])
            include_tags: HTML tags to include in extraction
            exclude_tags: HTML tags to exclude from extraction
            only_main_content: Whether to extract only main content
            extract_images: Extract images from the page (v2.1.0 feature)
            extract_data_attributes: Extract data-* attributes (v2.1.0 feature)
            wait_for: Time to wait before scraping (milliseconds)
        
        Returns:
            Dictionary containing scraped content and metadata
        """
        try:
            if not self.firecrawl_client:
                return {
                    "error": "Firecrawl client not initialized",
                    "url": url,
                    "scraped_at": datetime.utcnow().isoformat()
                }
            
            # Set default formats
            if formats is None:
                formats = ['markdown', 'html']
            
            # Prepare scrape options with v2.1.0+ features
            scrape_options = {
                'formats': formats,
                'onlyMainContent': only_main_content,
                'extractImages': extract_images,  # v2.1.0 feature
                'extractDataAttributes': extract_data_attributes  # v2.1.0 feature
            }
            
            if include_tags:
                scrape_options['includeTags'] = include_tags
            
            if exclude_tags:
                scrape_options['excludeTags'] = exclude_tags
            
            if wait_for:
                scrape_options['waitFor'] = wait_for
            
            # Execute scraping in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.firecrawl_client.scrape(url, **scrape_options)
            )
            
            return {
                "success": True,
                "url": url,
                "title": result.get("metadata", {}).get("title", ""),
                "description": result.get("metadata", {}).get("description", ""),
                "content": {
                    format_type: result.get(format_type, "") 
                    for format_type in formats
                },
                "metadata": result.get("metadata", {}),
                "scraped_at": datetime.utcnow().isoformat(),
                "method": "firecrawl"
            }
            
        except Exception as e:
            logger.error(f"Failed to scrape URL {url}: {e}")
            return {
                "error": str(e),
                "url": url,
                "scraped_at": datetime.utcnow().isoformat()
            }
    
    async def extract_structured_data(
        self,
        urls: Union[str, List[str]],
        schema: Dict[str, Any] = None,
        webhook_url: str = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from URLs using Firecrawl's extract endpoint (v2.1.0+).
        
        Args:
            urls: Single URL or list of URLs to extract from
            schema: Schema definition for structured extraction
            webhook_url: Webhook URL for result delivery (with signature support)
        
        Returns:
            Dictionary containing extracted structured data
        """
        try:
            if not self.firecrawl_client:
                return {
                    "error": "Firecrawl client not initialized",
                    "urls": urls,
                    "extracted_at": datetime.utcnow().isoformat()
                }
            
            # Prepare extract options
            extract_options = {}
            
            if schema:
                extract_options['schema'] = schema
            
            if webhook_url:
                extract_options['webhook'] = {
                    'url': webhook_url,
                    'signature': True  # v2.2.0 webhook signature support
                }
            
            # Convert single URL to list
            url_list = [urls] if isinstance(urls, str) else urls
            
            # Execute extraction in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.firecrawl_client.extract(url_list, **extract_options)
            )
            
            return {
                "success": True,
                "urls": url_list,
                "extracted_data": result.get("data", []),
                "schema": schema,
                "extracted_at": datetime.utcnow().isoformat(),
                "method": "firecrawl_extract"
            }
            
        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return {
                "error": str(e),
                "urls": urls,
                "extracted_at": datetime.utcnow().isoformat()
            }
    
    async def crawl_website(
        self,
        base_url: str,
        limit: int = 100,
        formats: List[str] = None,
        include_paths: List[str] = None,
        exclude_paths: List[str] = None,
        max_depth: int = None,
        allow_hash_routes: bool = True,
        proxy_location: str = None
    ) -> Dict[str, Any]:
        """
        Crawl an entire website and return content from multiple pages.
        
        Args:
            base_url: Base URL to start crawling from
            limit: Maximum number of pages to crawl (up to 100k in v2.1.0+)
            formats: List of formats to extract
            include_paths: URL paths to include in crawl
            exclude_paths: URL paths to exclude from crawl
            max_depth: Maximum crawl depth
            allow_hash_routes: Handle hash-based routes (v2.1.0 feature)
            proxy_location: Proxy location for crawling (v2.2.0 feature)
        
        Returns:
            Dictionary containing crawled content and metadata
        """
        try:
            if not self.firecrawl_client:
                return {
                    "error": "Firecrawl client not initialized",
                    "base_url": base_url,
                    "crawled_at": datetime.utcnow().isoformat()
                }
            
            # Set default formats
            if formats is None:
                formats = ['markdown', 'html']
            
            # Prepare crawl options with v2.1.0+ features
            crawl_options = {
                'limit': min(limit, 100000),  # v2.1.0+ supports up to 100k
                'scrapeOptions': {
                    'formats': formats,
                    'onlyMainContent': True
                },
                'allowHashRoutes': allow_hash_routes  # v2.1.0 feature
            }
            
            if include_paths:
                crawl_options['includePaths'] = include_paths
            
            if exclude_paths:
                crawl_options['excludePaths'] = exclude_paths
            
            if max_depth is not None:
                crawl_options['maxDepth'] = max_depth
            
            if proxy_location:
                crawl_options['proxyLocation'] = proxy_location  # v2.2.0 feature
            
            # Execute crawling
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.firecrawl_client.crawl(base_url, **crawl_options)
            )
            
            return {
                "success": True,
                "base_url": base_url,
                "pages_crawled": len(result.get("data", [])),
                "pages": result.get("data", []),
                "crawled_at": datetime.utcnow().isoformat(),
                "method": "firecrawl"
            }
            
        except Exception as e:
            logger.error(f"Failed to crawl website {base_url}: {e}")
            return {
                "error": str(e),
                "base_url": base_url,
                "crawled_at": datetime.utcnow().isoformat()
            }
    
    async def batch_scrape(
        self,
        urls: List[str],
        formats: List[str] = None,
        concurrent_limit: int = 5
    ) -> Dict[str, Any]:
        """
        Scrape multiple URLs concurrently.
        
        Args:
            urls: List of URLs to scrape
            formats: List of formats to extract
            concurrent_limit: Maximum concurrent scraping operations
        
        Returns:
            Dictionary containing results for all URLs
        """
        try:
            if not self.firecrawl_client:
                return {
                    "error": "Firecrawl client not initialized",
                    "urls": urls,
                    "scraped_at": datetime.utcnow().isoformat()
                }
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(concurrent_limit)
            
            async def scrape_single_url(url: str):
                async with semaphore:
                    return await self.scrape_url(url, formats)
            
            # Execute all scraping tasks concurrently
            tasks = [scrape_single_url(url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_scrapes = []
            failed_scrapes = []
            
            for i, result in enumerate(results):
                url = urls[i]
                if isinstance(result, Exception):
                    failed_scrapes.append({
                        "url": url,
                        "error": str(result)
                    })
                elif result.get("success"):
                    successful_scrapes.append(result)
                else:
                    failed_scrapes.append(result)
            
            return {
                "success": True,
                "total_urls": len(urls),
                "successful_scrapes": len(successful_scrapes),
                "failed_scrapes": len(failed_scrapes),
                "results": successful_scrapes,
                "failures": failed_scrapes,
                "scraped_at": datetime.utcnow().isoformat(),
                "method": "firecrawl_batch"
            }
            
        except Exception as e:
            logger.error(f"Batch scraping failed: {e}")
            return {
                "error": str(e),
                "urls": urls,
                "scraped_at": datetime.utcnow().isoformat()
            }
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        search_options: Dict[str, Any] = None,
        categories: List[str] = None
    ) -> Dict[str, Any]:
        """
        Search the web using Firecrawl's search functionality (v2.1.0+).
        
        Args:
            query: Search query
            max_results: Maximum number of search results
            search_options: Additional search options
            categories: Search categories (github, research) - v2.1.0 feature
        
        Returns:
            Dictionary containing search results
        """
        try:
            if not self.firecrawl_client:
                return {
                    "error": "Firecrawl client not initialized",
                    "query": query,
                    "searched_at": datetime.utcnow().isoformat()
                }
            
            # Prepare search options with v2.1.0+ features
            options = search_options or {}
            options.update({"limit": max_results})
            
            # Add categories filter (v2.1.0 feature)
            if categories:
                options["categories"] = categories
            
            # Execute search in thread pool to avoid blocking
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.firecrawl_client.search(query, **options)
            )
            
            return {
                "success": True,
                "query": query,
                "results": result.get("data", []),
                "total_results": len(result.get("data", [])),
                "searched_at": datetime.utcnow().isoformat(),
                "method": "firecrawl_search"
            }
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return {
                "error": str(e),
                "query": query,
                "searched_at": datetime.utcnow().isoformat()
            }
    
    async def map_website(
        self,
        url: str,
        search_query: str = None,
        ignore_sitemap: bool = False,
        include_subdomains: bool = False,
        limit: int = 5000
    ) -> Dict[str, Any]:
        """
        Map/discover all URLs on a website (15x faster in v2.2.0).
        
        Args:
            url: Website URL to map
            search_query: Optional search query to filter URLs
            ignore_sitemap: Whether to ignore sitemaps
            include_subdomains: Whether to include subdomains
            limit: Maximum number of URLs to discover (up to 100k)
        
        Returns:
            Dictionary containing discovered URLs
        """
        try:
            if not self.firecrawl_client:
                return {
                    "error": "Firecrawl client not initialized",
                    "url": url,
                    "mapped_at": datetime.utcnow().isoformat()
                }
            
            # Prepare map options
            map_options = {
                "limit": min(limit, 100000),  # v2.2.0 supports up to 100k
                "ignoreSitemap": ignore_sitemap,
                "includeSubdomains": include_subdomains
            }
            
            if search_query:
                map_options["search"] = search_query
            
            # Execute mapping in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.firecrawl_client.map_url(url, **map_options)
            )
            
            return {
                "success": True,
                "base_url": url,
                "urls_found": len(result.get("links", [])),
                "urls": result.get("links", []),
                "mapped_at": datetime.utcnow().isoformat(),
                "method": "firecrawl_map"
            }
            
        except Exception as e:
            logger.error(f"Website mapping failed for {url}: {e}")
            return {
                "error": str(e),
                "url": url,
                "mapped_at": datetime.utcnow().isoformat()
            }
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get queue status from Firecrawl API (v2.2.0 feature).
        
        Returns:
            Dictionary containing queue status information
        """
        try:
            if not self.firecrawl_client:
                return {
                    "error": "Firecrawl client not initialized",
                    "checked_at": datetime.utcnow().isoformat()
                }
            
            # Execute queue status check in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.firecrawl_client.get_queue_status()
            )
            
            return {
                "success": True,
                "queue_status": result,
                "checked_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Queue status check failed: {e}")
            return {
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }
    
    async def scrape_pdf(
        self,
        url: str,
        max_pages: int = None,
        formats: List[str] = None
    ) -> Dict[str, Any]:
        """
        Scrape PDF with maxPages parameter (v2.2.0 feature).
        
        Args:
            url: PDF URL to scrape
            max_pages: Maximum number of pages to parse
            formats: List of formats to extract
        
        Returns:
            Dictionary containing PDF content
        """
        try:
            if not self.firecrawl_client:
                return {
                    "error": "Firecrawl client not initialized",
                    "url": url,
                    "scraped_at": datetime.utcnow().isoformat()
                }
            
            # Set default formats
            if formats is None:
                formats = ['markdown', 'html']
            
            # Prepare scrape options with PDF-specific settings
            scrape_options = {
                'formats': formats,
                'onlyMainContent': True
            }
            
            # Add maxPages parameter for PDF parsing
            if max_pages is not None:
                scrape_options['maxPages'] = max_pages
            
            # Execute PDF scraping in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.firecrawl_client.scrape(url, **scrape_options)
            )
            
            return {
                "success": True,
                "url": url,
                "title": result.get("metadata", {}).get("title", ""),
                "pages_processed": max_pages,
                "content": {
                    format_type: result.get(format_type, "") 
                    for format_type in formats
                },
                "metadata": result.get("metadata", {}),
                "scraped_at": datetime.utcnow().isoformat(),
                "method": "firecrawl_pdf"
            }
            
        except Exception as e:
            logger.error(f"PDF scraping failed for {url}: {e}")
            return {
                "error": str(e),
                "url": url,
                "scraped_at": datetime.utcnow().isoformat()
            }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get scraper capabilities and status."""
        return {
            "firecrawl_available": FIRECRAWL_AVAILABLE,
            "client_initialized": self.firecrawl_client is not None,
            "supported_formats": ["markdown", "html"],
            "features": {
                "single_url_scraping": True,
                "website_crawling": True,
                "batch_scraping": True,
                "content_filtering": True,
                "metadata_extraction": True,
                "search_integration": True,      # v2.1.0+ feature
                "search_categories": True,       # v2.1.0 github, research categories
                "website_mapping": True,         # v2.1.0+ (15x faster in v2.2.0)
                "pdf_parsing": True,             # v2.2.0 with maxPages
                "queue_status": True,            # v2.2.0 feature
                "webhook_support": True,         # v2.2.0 with signatures
                "image_extraction": True,        # v2.1.0 feature
                "data_attributes": True,         # v2.1.0 feature
                "hash_based_routing": True,      # v2.1.0 feature
                "google_drive_support": True,    # v2.1.0 TXT, PDF, Sheets
                "structured_extraction": True,   # v2.1.0+ extract endpoint
                "proxy_location_support": True,  # v2.2.0 feature
                "mcp_integration": True,         # v2.2.0 MCP v3 support
                "static_ip_proxies": True,       # v2.2.0 feature
            },
            "version": "2.2.0",
            "timestamp": datetime.utcnow().isoformat()
        }


# Factory function
def create_web_scraper(config: Optional[Dict[str, Any]] = None) -> WebScraper:
    """Create and return a web scraper instance."""
    return WebScraper(config=config)


# Integration with agent tools
class WebScrapeTool:
    """Tool for web scraping that can be used with LangChain agents."""
    
    def __init__(self, web_scraper: WebScraper):
        self.web_scraper = web_scraper
        self.name = "web_scrape"
        self.description = """
        Advanced web scraping using Firecrawl v2.1.0+ with comprehensive capabilities:
        - scrape: Single URL scraping with 'url', 'formats', 'maxPages', 'extractImages', 'extractDataAttributes'
        - search: Web search with 'query', 'max_results', 'categories' (github, research)
        - map: Website mapping with 'url', 'limit', 'search_query', 'includeSubdomains'
        - crawl: Full website crawling with 'base_url', 'limit', 'allowHashRoutes', 'proxyLocation'
        - extract: Structured data extraction with 'urls', 'schema', 'webhook_url'
        - queue_status: Get Firecrawl queue status and performance metrics
        - batch: Batch URL processing with 'urls', 'formats', 'concurrent_limit'
        Input should be JSON with 'action' and relevant parameters.
        """
    
    def _run(self, query: str) -> str:
        """Execute web scraping operation with v2.2.0 capabilities."""
        try:
            import json
            
            # Parse input
            if isinstance(query, str):
                try:
                    parsed_query = json.loads(query)
                except json.JSONDecodeError:
                    # Assume it's a URL for backward compatibility
                    parsed_query = {"action": "scrape", "url": query, "formats": ["markdown"]}
            else:
                parsed_query = query
            
            action = parsed_query.get("action", "scrape")
            
            # Execute different actions based on v2.2.0 capabilities
            if action == "scrape":
                url = parsed_query.get("url", "")
                if not url:
                    return json.dumps({"error": "No URL provided for scrape action"})
                
                formats = parsed_query.get("formats", ["markdown"])
                max_pages = parsed_query.get("maxPages")  # v2.2.0 PDF feature
                
                if max_pages:
                    result = asyncio.run(
                        self.web_scraper.scrape_pdf(url, max_pages, formats)
                    )
                else:
                    result = asyncio.run(
                        self.web_scraper.scrape_url(
                            url, formats, 
                            parsed_query.get("include_tags"),
                            parsed_query.get("exclude_tags"),
                            parsed_query.get("only_main_content", True),
                            parsed_query.get("extractImages", False),  # v2.1.0
                            parsed_query.get("extractDataAttributes", False),  # v2.1.0
                            parsed_query.get("waitFor")  # v2.1.0+
                        )
                    )
                    
            elif action == "search":
                query_text = parsed_query.get("query", "")
                if not query_text:
                    return json.dumps({"error": "No query provided for search action"})
                
                max_results = parsed_query.get("max_results", 10)
                search_options = parsed_query.get("search_options", {})
                
                result = asyncio.run(
                    self.web_scraper.search(
                        query_text, 
                        max_results, 
                        search_options,
                        parsed_query.get("categories")  # v2.1.0 feature
                    )
                )
                
            elif action == "map":
                url = parsed_query.get("url", "")
                if not url:
                    return json.dumps({"error": "No URL provided for map action"})
                
                result = asyncio.run(
                    self.web_scraper.map_website(
                        url,
                        parsed_query.get("search_query"),
                        parsed_query.get("ignore_sitemap", False),
                        parsed_query.get("include_subdomains", False),
                        parsed_query.get("limit", 5000)
                    )
                )
                
            elif action == "crawl":
                base_url = parsed_query.get("base_url", "")
                if not base_url:
                    return json.dumps({"error": "No base_url provided for crawl action"})
                
                result = asyncio.run(
                    self.web_scraper.crawl_website(
                        base_url,
                        parsed_query.get("limit", 100),
                        parsed_query.get("formats", ["markdown"]),
                        parsed_query.get("include_paths"),
                        parsed_query.get("exclude_paths"),
                        parsed_query.get("max_depth"),
                        parsed_query.get("allowHashRoutes", True),  # v2.1.0
                        parsed_query.get("proxyLocation")  # v2.2.0
                    )
                )
                
            elif action == "queue_status":
                result = asyncio.run(
                    self.web_scraper.get_queue_status()
                )
                
            elif action == "extract":
                urls = parsed_query.get("urls", [])
                if not urls:
                    return json.dumps({"error": "No URLs provided for extract action"})
                
                result = asyncio.run(
                    self.web_scraper.extract_structured_data(
                        urls,
                        parsed_query.get("schema"),
                        parsed_query.get("webhook_url")
                    )
                )
                
            elif action == "batch":
                urls = parsed_query.get("urls", [])
                if not urls:
                    return json.dumps({"error": "No URLs provided for batch action"})
                
                result = asyncio.run(
                    self.web_scraper.batch_scrape(
                        urls,
                        parsed_query.get("formats", ["markdown"]),
                        parsed_query.get("concurrent_limit", 5)
                    )
                )
                
            else:
                return json.dumps({
                    "error": f"Unknown action: {action}",
                    "supported_actions": [
                        "scrape", "search", "map", "crawl", 
                        "extract", "queue_status", "batch"
                    ],
                    "version_features": {
                        "v2.1.0": ["image_extraction", "data_attributes", "search_categories", "hash_routes"],
                        "v2.2.0": ["pdf_max_pages", "queue_status", "webhook_signatures", "mcp_integration"]
                    }
                })
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Web scrape tool error: {e}")
            return json.dumps({"error": str(e)})
