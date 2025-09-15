"""
Instagram Research Agent MCP (IRAM) - Unified MCP Server

This module implements a unified FastAPI-based MCP server that integrates
all Instagram research and analysis capabilities.
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Import our modules
from .agent_orchestrator import create_instagram_agent, InstagramAgentOrchestrator
from .scraping_module import InstagramScraper
from .analysis_module import ContentAnalyzer
from .evasion_manager import EvasionManager
from .utils import get_logger, validate_instagram_username

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Global variables
agent_orchestrator: Optional[InstagramAgentOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifespan of the application."""
    global agent_orchestrator

    # Startup
    logger.info("Starting IRAM MCP Server (lazy init mode)...")
    # Defer heavy initialization until the first request needing the agent
    agent_orchestrator = None

    yield

    # Shutdown
    logger.info("Shutting down IRAM MCP Server...")
    if agent_orchestrator:
        # Cleanup if needed
        pass


# Create FastAPI app
app = FastAPI(
    title="IRAM - Instagram Research Agent MCP",
    description="A comprehensive MCP server for Instagram research, analysis, and automation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class TaskRequest(BaseModel):
    task: str = Field(..., description="High-level task description")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the task")


class ProfileRequest(BaseModel):
    username: str = Field(..., description="Instagram username")
    include_posts: bool = Field(False, description="Include recent posts")
    include_stories: bool = Field(False, description="Include stories")
    post_limit: int = Field(50, description="Maximum number of posts to fetch")


class AnalysisRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Data to analyze")
    analysis_type: str = Field("comprehensive", description="Type of analysis to perform")


class CompareAccountsRequest(BaseModel):
    usernames: List[str] = Field(..., description="List of usernames to compare")


class HashtagTrendsRequest(BaseModel):
    hashtags: List[str] = Field(..., description="List of hashtags to analyze")


class ScrapeRequest(BaseModel):
    target: str = Field(..., description="Target username or hashtag")
    content_type: str = Field("profile", description="Type of content to scrape")
    limit: int = Field(50, description="Maximum number of items to fetch")


# Dependency to get agent orchestrator
def get_agent() -> InstagramAgentOrchestrator:
    """Get or lazily create the agent orchestrator instance."""
    global agent_orchestrator
    if agent_orchestrator is None:
        try:
            config = {
                "instagram_username": os.getenv("INSTAGRAM_USERNAME"),
                "instagram_password": os.getenv("INSTAGRAM_PASSWORD"),
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
                "llm_model": os.getenv("LLM_MODEL", "openrouter/sonoma-sky-alpha"),
                "debug": os.getenv("DEBUG", "false").lower() == "true",
            }
            agent_orchestrator = create_instagram_agent(config)
            logger.info("Agent orchestrator initialized successfully (lazy)")
        except Exception as e:
            logger.error(f"Failed to initialize agent orchestrator: {e}")
            raise HTTPException(status_code=500, detail="Agent initialization failed: " + str(e))
    return agent_orchestrator


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "IRAM MCP Server",
        "version": "1.0.0"
    }


# Main MCP endpoints
@app.post("/execute_task")
async def execute_task(
    request: TaskRequest,
    agent: InstagramAgentOrchestrator = Depends(get_agent)
):
    """Execute a high-level Instagram research task."""
    try:
        logger.info(f"Executing task: {request.task}")
        result = await agent.execute_task(request.task, request.context)
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fetch_profile")
async def fetch_profile(
    request: ProfileRequest,
    agent: InstagramAgentOrchestrator = Depends(get_agent)
):
    """Fetch comprehensive profile information."""
    try:
        if not validate_instagram_username(request.username):
            raise HTTPException(status_code=400, detail="Invalid Instagram username")
        
        # Create scraping task
        task_parts = [f"Fetch profile information for @{request.username}"]
        
        if request.include_posts:
            task_parts.append(f"Include last {request.post_limit} posts with engagement metrics")
        
        if request.include_stories:
            task_parts.append("Include current stories")
        
        task = ". ".join(task_parts)
        context = {
            "username": request.username,
            "include_posts": request.include_posts,
            "include_stories": request.include_stories,
            "post_limit": request.post_limit
        }
        
        result = await agent.execute_task(task, context)
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Profile fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scrape_content")
async def scrape_content(
    request: ScrapeRequest,
    agent: InstagramAgentOrchestrator = Depends(get_agent)
):
    """Scrape specific Instagram content."""
    try:
        scrape_query = {
            "target": request.target,
            "content_type": request.content_type,
            "limit": request.limit
        }
        
        result = agent.scraper.scrape_content(scrape_query)
        
        return JSONResponse(content={
            "success": True,
            "target": request.target,
            "content_type": request.content_type,
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Content scraping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_content")
async def analyze_content(
    request: AnalysisRequest,
    agent: InstagramAgentOrchestrator = Depends(get_agent)
):
    """Analyze Instagram content."""
    try:
        analysis_query = json.dumps({
            "data": request.data,
            "analysis_type": request.analysis_type
        })
        
        # Use the analysis tool from the agent
        analysis_tool = None
        for tool in agent.tools:
            if tool.name == "content_analyze":
                analysis_tool = tool
                break
        
        if not analysis_tool:
            raise HTTPException(status_code=500, detail="Analysis tool not available")
        
        result = analysis_tool._run(analysis_query)
        parsed_result = json.loads(result)
        
        return JSONResponse(content={
            "success": True,
            "analysis_type": request.analysis_type,
            "result": parsed_result,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_account_trends")
async def analyze_account_trends(
    username: str,
    days: int = 30,
    agent: InstagramAgentOrchestrator = Depends(get_agent)
):
    """Analyze trends for a specific Instagram account."""
    try:
        if not validate_instagram_username(username):
            raise HTTPException(status_code=400, detail="Invalid Instagram username")
        
        result = agent.analyze_account_trends(username, days)
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Account trends analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare_accounts")
async def compare_accounts(
    request: CompareAccountsRequest,
    agent: InstagramAgentOrchestrator = Depends(get_agent)
):
    """Compare multiple Instagram accounts."""
    try:
        # Validate all usernames
        for username in request.usernames:
            if not validate_instagram_username(username):
                raise HTTPException(status_code=400, detail=f"Invalid Instagram username: {username}")
        
        result = agent.compare_accounts(request.usernames)
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Account comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research_hashtag_trends")
async def research_hashtag_trends(
    request: HashtagTrendsRequest,
    agent: InstagramAgentOrchestrator = Depends(get_agent)
):
    """Research trends for specific hashtags."""
    try:
        result = agent.research_hashtag_trends(request.hashtags)
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Hashtag trends research failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search_users")
async def search_users(
    query: str,
    limit: int = 20,
    agent: InstagramAgentOrchestrator = Depends(get_agent)
):
    """Search for Instagram users."""
    try:
        result = agent.scraper.search_users(query, limit)
        return JSONResponse(content={
            "success": True,
            "query": query,
            "users": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"User search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search_hashtags")
async def search_hashtags(
    query: str,
    limit: int = 20,
    agent: InstagramAgentOrchestrator = Depends(get_agent)
):
    """Search for Instagram hashtags."""
    try:
        result = agent.scraper.search_hashtags(query, limit)
        return JSONResponse(content={
            "success": True,
            "query": query,
            "hashtags": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Hashtag search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task endpoints
@app.post("/schedule_analysis")
async def schedule_analysis(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    agent: InstagramAgentOrchestrator = Depends(get_agent)
):
    """Schedule a background analysis task."""
    try:
        def run_background_task():
            """Run the task in background."""
            try:
                result = asyncio.run(agent.execute_task(request.task, request.context))
                logger.info(f"Background task completed: {result}")
            except Exception as e:
                logger.error(f"Background task failed: {e}")
        
        background_tasks.add_task(run_background_task)
        
        return JSONResponse(content={
            "success": True,
            "message": "Task scheduled for background execution",
            "task": request.task,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Task scheduling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoints
@app.get("/config")
async def get_config():
    """Get current server configuration."""
    return {
        "server": "IRAM MCP Server",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "features": {
            "instagram_scraping": True,
            "content_analysis": True,
            "trend_analysis": True,
            "account_comparison": True,
            "hashtag_research": True,
            "background_tasks": True
        }
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Main execution
def main():
    """Main function to run the server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting IRAM MCP Server on {host}:{port}")
    
    uvicorn.run(
        "src.mcp_server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="debug" if debug else "info"
    )


if __name__ == "__main__":
    main()