"""
Instagram Research Agent MCP (IRAM) - Core Agent Orchestrator

This module implements the main agent orchestrator that uses LangChain ReAct agents
to decompose high-level tasks into executable sub-tasks.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import asyncio

from langchain import hub
from langchain_core.tools import tool, BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import load_tools

# Import our modules
from .scraping_module import InstagramScraper
from .analysis_module import ContentAnalyzer
from .evasion_manager import EvasionManager
from .web_scraper import WebScraper
from .vector_store import get_vector_store
from .firecrawl_tool import get_firecrawl_tool
from .budget_monitor import get_budget_monitor
from .utils import get_logger, validate_instagram_username
from .config import get_config, IRamConfig

logger = get_logger(__name__)


class ResearchTool(BaseTool):
    """Tool for researching Instagram accounts and discovering related content."""
    
    name: str = "instagram_research"
    description: str = """
    Research Instagram accounts, hashtags, and related content.
    Use this to discover profiles, analyze hashtags, and find related accounts.
    Input should be a JSON string with 'query' and 'type' fields.
    """
    scraper: InstagramScraper
    
    def __init__(self, scraper: InstagramScraper, **kwargs):
        super().__init__(scraper=scraper, **kwargs)
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute research on Instagram content."""
        try:
            # Parse the input
            if isinstance(query, str):
                try:
                    parsed_query = json.loads(query)
                except json.JSONDecodeError:
                    parsed_query = {"query": query, "type": "profile"}
            else:
                parsed_query = query
            
            search_query = parsed_query.get("query", "")
            search_type = parsed_query.get("type", "profile")
            
            if search_type == "profile":
                result = self.scraper.search_users(search_query)
            elif search_type == "hashtag":
                result = self.scraper.search_hashtags(search_query)
            else:
                result = self.scraper.search_general(search_query)
            
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error(f"Research tool error: {e}")
            return json.dumps({"error": str(e)})


class ScrapeTool(BaseTool):
    """Tool for scraping Instagram content."""
    
    name: str = "instagram_scrape"
    description: str = """
    Scrape Instagram profiles, posts, stories, or specific content.
    Input should be a JSON string with 'target', 'content_type', and optional 'limit' fields.
    """
    scraper: InstagramScraper
    
    def __init__(self, scraper: InstagramScraper, **kwargs):
        super().__init__(scraper=scraper, **kwargs)
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute scraping operation."""
        try:
            parsed_query = json.loads(query) if isinstance(query, str) else query
            
            target = parsed_query.get("target", "")
            content_type = parsed_query.get("content_type", "profile")
            limit = parsed_query.get("limit", 50)
            
            if content_type == "profile":
                result = self.scraper.get_profile_info(target)
            elif content_type == "posts":
                result = self.scraper.get_user_posts(target, limit)
            elif content_type == "stories":
                result = self.scraper.get_user_stories(target)
            elif content_type == "followers":
                result = self.scraper.get_followers(target, limit)
            else:
                result = {"error": f"Unknown content type: {content_type}"}
            
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error(f"Scrape tool error: {e}")
            return json.dumps({"error": str(e)})


class AnalyzeTool(BaseTool):
    """Tool for analyzing scraped Instagram content."""
    
    name: str = "content_analyze"
    description: str = """
    Analyze Instagram content for sentiment, topics, engagement patterns, etc.
    Input should be a JSON string with 'data' and 'analysis_type' fields.
    """
    analyzer: ContentAnalyzer
    
    def __init__(self, analyzer: ContentAnalyzer, **kwargs):
        super().__init__(analyzer=analyzer, **kwargs)
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute content analysis."""
        try:
            parsed_query = json.loads(query) if isinstance(query, str) else query
            
            data = parsed_query.get("data", {})
            analysis_type = parsed_query.get("analysis_type", "comprehensive")
            
            if analysis_type == "sentiment":
                result = self.analyzer.analyze_sentiment(data)
            elif analysis_type == "topics":
                result = self.analyzer.extract_topics(data)
            elif analysis_type == "engagement":
                result = self.analyzer.analyze_engagement(data)
            elif analysis_type == "comprehensive":
                result = self.analyzer.comprehensive_analysis(data)
            else:
                result = {"error": f"Unknown analysis type: {analysis_type}"}
            
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error(f"Analysis tool error: {e}")
            return json.dumps(result, default=str)


class WebScrapeTool(BaseTool):
    """Tool for web scraping using Firecrawl."""
    
    name: str = "web_scrape"
    description: str = """
    Advanced web scraping using Firecrawl with comprehensive capabilities:
    - scrape: Single URL scraping with 'url', 'formats', 'maxPages', 'extractImages'
    - search: Web search with 'query', 'max_results', 'categories'
    - map: Website mapping with 'url', 'limit', 'search_query', 'includeSubdomains'
    - crawl: Full website crawling with 'base_url', 'limit', 'allowHashRoutes'
    - extract: Structured data extraction with 'urls', 'schema', 'webhook_url'
    - batch: Batch URL processing with 'urls', 'formats', 'concurrent_limit'
    Input should be JSON with 'action' and relevant parameters.
    """
    web_scraper: WebScraper
    
    def __init__(self, web_scraper: WebScraper, **kwargs):
        super().__init__(web_scraper=web_scraper, **kwargs)
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute web scraping operation."""
        try:
            import json
            import asyncio
            
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
            
            # Execute different actions
            if action == "scrape":
                url = parsed_query.get("url", "")
                if not url:
                    return json.dumps({"error": "No URL provided for scrape action"})
                
                formats = parsed_query.get("formats", ["markdown"])
                result = asyncio.run(
                    self.web_scraper.scrape_url(
                        url, formats, 
                        parsed_query.get("include_tags"),
                        parsed_query.get("exclude_tags"),
                        parsed_query.get("only_main_content", True),
                        parsed_query.get("extractImages", False),
                        parsed_query.get("extractDataAttributes", False),
                        parsed_query.get("waitFor")
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
                        parsed_query.get("categories")
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
                    "supported_actions": ["scrape", "search", "batch"]
                })
            
            return json.dumps(result, default=str)
            
        except Exception as e:
            logger.error(f"Web scrape tool error: {e}")
            return json.dumps({"error": str(e)})


class SemanticSearchTool(BaseTool):
    """Tool for performing semantic search over collected data."""
    name: str = "semantic_search"
    description: str = """
    Perform semantic search over previously collected and indexed data.
    Use this to find relevant information from past analyses.
    Input should be a JSON string with a 'query' field.
    """
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute semantic search."""
        try:
            parsed_query = json.loads(query) if isinstance(query, str) else query
            search_query = parsed_query.get("query", "")
            
            if not search_query:
                return json.dumps({"error": "No query provided"})

            vector_store = get_vector_store()
            results = asyncio.run(vector_store.search(search_query))
            
            return json.dumps(results, default=str)
        except Exception as e:
            logger.error(f"Semantic search tool error: {e}")
            return json.dumps({"error": str(e)})


class FirecrawlTool(BaseTool):
    """Tool for performing web research using Firecrawl."""
    name: str = "firecrawl_research"
    description: str = """
    Perform web research, including web searches and scraping URLs, using Firecrawl.
    Use this to gather external information from the web.
    Input should be a JSON string with 'action' ('search' or 'scrape') and relevant parameters.
    """
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute Firecrawl research."""
        try:
            parsed_query = json.loads(query) if isinstance(query, str) else query
            action = parsed_query.get("action")
            
            firecrawl_tool = get_firecrawl_tool()

            if action == "search":
                search_query = parsed_query.get("query", "")
                if not search_query:
                    return json.dumps({"error": "No query provided for search"})
                results = asyncio.run(firecrawl_tool.search(search_query))
            elif action == "scrape":
                url = parsed_query.get("url")
                if not url:
                    return json.dumps({"error": "No URL provided for scrape"})
                results = asyncio.run(firecrawl_tool.scrape_url(url))
            else:
                return json.dumps({"error": "Invalid action specified"})
            
            return json.dumps(results, default=str)
        except Exception as e:
            logger.error(f"Firecrawl tool error: {e}")
            return json.dumps({"error": str(e)})


class InstagramAgentOrchestrator:
    """Main orchestrator for Instagram research agent operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent orchestrator."""
        # Use centralized configuration
        self.iram_config = get_config()
        self.config = config or {}
        
        # Initialize components
        self.scraper = InstagramScraper(config=self.config)
        self.analyzer = ContentAnalyzer(config=self.config)
        self.evasion_manager = EvasionManager(config=self.config)
        self.web_scraper = WebScraper(config={
            "firecrawl_api_key": self.iram_config.firecrawl.api_key,
            **self.config
        })
        self.budget_monitor = get_budget_monitor()
        
        # Initialize LLM with priority: Anthropic Claude > OpenAI > OpenRouter
        self.llm = self._initialize_llm()
        
        # Modern LangChain uses built-in conversation management in LangGraph
        # No need for explicit memory management
        
        # Create tools
        self.tools = self._create_tools()
        
        # Initialize agent
        self.agent = self._create_agent()
        
        logger.info("Instagram Agent Orchestrator initialized successfully")
    
    def _initialize_llm(self):
        """Initialize LLM with provider priority: Anthropic > OpenAI > OpenRouter."""
        llm_settings = self.iram_config.llm
        
        # Get API keys from settings and environment
        anthropic_key = llm_settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        openai_key = llm_settings.openai_api_key
        openrouter_key = llm_settings.openrouter_api_key
        
        # Determine primary provider from config
        primary_provider = self.iram_config.get_primary_llm_provider()
        
        # Try Anthropic Claude first (preferred)
        if anthropic_key:
            try:
                logger.info("Initializing Anthropic Claude LLM")
                return ChatAnthropic(
                    model=llm_settings.anthropic_model,
                    api_key=anthropic_key,
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic Claude: {e}")
        
        # Fallback to OpenRouter
        if openrouter_key and (primary_provider == "openrouter" or not openai_key):
            try:
                logger.info("Initializing OpenRouter LLM")
                return ChatOpenAI(
                    model=llm_settings.default_model,
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens,
                    openai_api_key=openrouter_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    model_kwargs={
                        "extra_headers": {
                            "HTTP-Referer": "https://github.com/mindburnlabs/iram-agent",
                            "X-Title": "IRAM Agent"
                        }
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter: {e}")
        
        # Fallback to OpenAI
        if openai_key:
            try:
                logger.info("Initializing OpenAI LLM")
                return ChatOpenAI(
                    model=llm_settings.openai_model,
                    api_key=openai_key,
                    temperature=llm_settings.temperature,
                    max_tokens=llm_settings.max_tokens
                )
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
        
        # If all else fails, raise an error
        raise ValueError(
            "No LLM provider available. Please set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY"
        )
    
    def _create_tools(self) -> List[BaseTool]:
        """Create the tools available to the agent."""
        return [
            ResearchTool(self.scraper),
            ScrapeTool(self.scraper),
            AnalyzeTool(self.analyzer),
            WebScrapeTool(self.web_scraper),
            SemanticSearchTool(),
            FirecrawlTool(),
        ]
    
    def _create_agent(self):
        """Create the ReAct agent using modern LangGraph patterns."""
        # Define system prompt for the Instagram research agent
        prompt = """You are IRAM (Instagram Research Agent MCP), an autonomous AI agent specialized in researching, analyzing, and reporting on Instagram content.

Your capabilities include:
1. Research: Discover Instagram profiles, hashtags, and related content
2. Scrape: Extract posts, stories, profiles, followers, and metadata
3. Analyze: Perform sentiment analysis, topic modeling, engagement analysis, and comprehensive content analysis

When given a high-level task, break it down into these steps:
1. Research phase: Discover relevant accounts, hashtags, or content
2. Scraping phase: Extract the actual data
3. Analysis phase: Process and analyze the collected data
4. Report phase: Synthesize findings into actionable insights

Always provide detailed, structured responses with clear insights and recommendations.
Follow ethical guidelines and only access publicly available content unless explicitly authorized for private content.

Available tools:
- instagram_research: For discovering and researching content
- instagram_scrape: For extracting specific content 
- content_analyze: For analyzing extracted content
- web_scrape: For scraping web content from URLs using Firecrawl
- semantic_search: For searching previously collected and indexed data
- firecrawl_research: For performing web research, including searches and scraping URLs

Remember to be thorough, accurate, and insightful in your analysis.
        
        # Create the ReAct agent using LangGraph
        return create_react_agent(
            self.llm,
            tools=self.tools,
            prompt=prompt
        )
    
    async def execute_task(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a high-level task using the agent."""
        try:
            logger.info(f"Executing task: {task}")
            
            # Add context to the task if provided
            if context:
                task_with_context = f"Task: {task}\nContext: {json.dumps(context, indent=2)}"
            else:
                task_with_context = task
            
            # Check budget before execution
            if not self.budget_monitor.is_within_budget():
                return {
                    "success": False,
                    "task": task,
                    "error": "Daily budget exceeded. Please try again tomorrow.",
                    "timestamp": datetime.utcnow().isoformat(),
                    "context": context
                }

            # Execute the task using LangGraph agent's invoke method
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.agent.invoke({"messages": [("user", task_with_context)]})
            )
            
            # Record the request in the budget monitor
            self.budget_monitor.record_request()
            
            # Extract the final response from LangGraph result
            if "messages" in result and result["messages"]:
                final_response = result["messages"][-1].content
            else:
                final_response = str(result)
            
            return {
                "success": True,
                "task": task,
                "result": final_response,
                "timestamp": datetime.utcnow().isoformat(),
                "context": context
            }
        
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "success": False,
                "task": task,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "context": context
            }
    
    def analyze_account_trends(self, username: str, days: int = 30) -> Dict[str, Any]:
        """Analyze trends for a specific Instagram account."""
        task = f"""
        Analyze Instagram account trends for @{username} over the last {days} days.
        
        Steps to complete:
        1. Research the account to get basic information
        2. Scrape recent posts (limit to posts from last {days} days)
        3. Analyze engagement patterns, sentiment trends, and topic evolution
        4. Generate a comprehensive trend report
        
        Focus on:
        - Engagement rate changes
        - Content theme evolution
        - Audience sentiment shifts
        - Posting frequency patterns
        - Peak engagement times
        """
        
        return asyncio.run(self.execute_task(task, {"username": username, "days": days}))
    
    def compare_accounts(self, usernames: List[str]) -> Dict[str, Any]:
        """Compare multiple Instagram accounts."""
        task = f"""
        Compare Instagram accounts: {', '.join(usernames)}
        
        Steps to complete:
        1. Research and scrape profile information for each account
        2. Scrape recent posts from each account (last 50 posts)
        3. Analyze each account's content strategy, engagement, and audience
        4. Generate a comparative analysis report
        
        Compare:
        - Follower counts and growth patterns
        - Engagement rates and patterns
        - Content themes and strategies
        - Posting frequencies
        - Audience sentiment
        """
        
        return asyncio.run(self.execute_task(task, {"usernames": usernames}))
    
    def research_hashtag_trends(self, hashtags: List[str]) -> Dict[str, Any]:
        """Research trends for specific hashtags."""
        task = f"""
        Research hashtag trends for: {', '.join(hashtags)}
        
        Steps to complete:
        1. Research each hashtag to understand its usage and context
        2. Scrape recent posts using these hashtags
        3. Analyze content themes, engagement patterns, and user demographics
        4. Generate trend insights and recommendations
        
        Focus on:
        - Hashtag popularity and growth
        - Associated content themes
        - Top performing posts
        - User engagement patterns
        - Related hashtag suggestions
        """
        
        return asyncio.run(self.execute_task(task, {"hashtags": hashtags}))


# Factory function for easy initialization
def create_instagram_agent(config: Optional[Dict[str, Any]] = None) -> InstagramAgentOrchestrator:
    """Create and return an Instagram agent orchestrator instance."""
    return InstagramAgentOrchestrator(config=config)