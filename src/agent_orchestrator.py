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

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.agents.react.base import ReActDocstoreAgent

# Import our modules
from .scraping_module import InstagramScraper
from .analysis_module import ContentAnalyzer
from .evasion_manager import EvasionManager
from .utils import get_logger, validate_instagram_username

logger = get_logger(__name__)


class ResearchTool(BaseTool):
    """Tool for researching Instagram accounts and discovering related content."""
    
    name: str = "instagram_research"
    description: str = """
    Research Instagram accounts, hashtags, and related content.
    Use this to discover profiles, analyze hashtags, and find related accounts.
    Input should be a JSON string with 'query' and 'type' fields.
    """
    
    def __init__(self, scraper: InstagramScraper, **kwargs):
        super().__init__(**kwargs)
        self.scraper = scraper
    
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
    
    def __init__(self, scraper: InstagramScraper, **kwargs):
        super().__init__(**kwargs)
        self.scraper = scraper
    
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
    
    def __init__(self, analyzer: ContentAnalyzer, **kwargs):
        super().__init__(**kwargs)
        self.analyzer = analyzer
    
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


class InstagramAgentOrchestrator:
    """Main orchestrator for Instagram research agent operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent orchestrator."""
        self.config = config or {}
        
        # Initialize components
        self.scraper = InstagramScraper(config=self.config)
        self.analyzer = ContentAnalyzer(config=self.config)
        self.evasion_manager = EvasionManager(config=self.config)
        
        # Initialize LLM
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if openrouter_key:
            # Use OpenRouter
            self.llm = ChatOpenAI(
                model=self.config.get("llm_model") or os.getenv("LLM_MODEL", "openrouter/sonoma-sky-alpha"),
                temperature=self.config.get("temperature", 0.1),
                openai_api_key=openrouter_key,
                openai_api_base="https://openrouter.ai/api/v1",
                model_kwargs={"extra_headers": {"HTTP-Referer": "https://github.com/mindburnlabs/iram-agent"}}
            )
        elif openai_key:
            # Fallback to OpenAI
            self.llm = ChatOpenAI(
                model=self.config.get("llm_model", "gpt-4"),
                temperature=self.config.get("temperature", 0.1),
                openai_api_key=openai_key
            )
        else:
            raise ValueError("Either OPENROUTER_API_KEY or OPENAI_API_KEY must be set")
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,
            return_messages=True
        )
        
        # Create tools
        self.tools = self._create_tools()
        
        # Initialize agent
        self.agent = self._create_agent()
        
        logger.info("Instagram Agent Orchestrator initialized successfully")
    
    def _create_tools(self) -> List[Tool]:
        """Create the tools available to the agent."""
        return [
            ResearchTool(self.scraper),
            ScrapeTool(self.scraper),
            AnalyzeTool(self.analyzer),
        ]
    
    def _create_agent(self):
        """Create the ReAct agent."""
        system_message = """You are IRAM (Instagram Research Agent MCP), an autonomous AI agent specialized in researching, analyzing, and reporting on Instagram content.

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

Remember to be thorough, accurate, and insightful in your analysis."""

        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={
                "system_message": system_message,
            }
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
            
            # Execute the task
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.agent.run, task_with_context
            )
            
            return {
                "success": True,
                "task": task,
                "result": result,
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