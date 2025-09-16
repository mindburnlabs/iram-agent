"""
Instagram Research Agent MCP (IRAM) - Unified MCP Server

This module implements a unified FastAPI-based MCP server that integrates
all Instagram research and analysis capabilities with comprehensive
configuration, logging, middleware, and error handling.
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, APIRouter
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
from .instagram_tools import create_instagram_tools, InstagramMCPTools
from .utils import validate_instagram_username

# Import new configuration and middleware
from .config import get_config, IRamConfig
from .logging_config import setup_logging, get_logger
from .middleware import (
    ErrorHandlingMiddleware,
    RequestTracingMiddleware,
    RateLimitingMiddleware,
    SecurityHeadersMiddleware,
    create_error_handler
)

# Import API v1 router
from .api.v1 import router as v1_router

# Load environment variables and setup logging
load_dotenv()
setup_logging()

logger = get_logger(__name__)

# DB
from .db import create_all, get_engine

# Global variables
agent_orchestrator: Optional[InstagramAgentOrchestrator] = None
instagram_tools: Optional[InstagramMCPTools] = None
_db_ready: bool = False
_config: Optional[IRamConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifespan of the application."""
    global agent_orchestrator, _config
    
    # Load configuration
    _config = get_config()
    
    # Startup
    logger.info(
        "Starting IRAM MCP Server (lazy init mode)...", 
        extra={
            "environment": _config.environment,
            "version": _config.version,
            "has_database": _config.has_database(),
            "has_llm_provider": _config.has_llm_provider()
        }
    )
    
    # Defer heavy initialization until the first request needing the agent
    agent_orchestrator = None

    # Initialize DB schema if configured
    global _db_ready
    try:
        if _config.has_database():
            created = await create_all()
            _db_ready = bool(created)
            if _db_ready:
                logger.info("Database schema ensured")
            else:
                logger.warning("Database creation failed")
        else:
            logger.info("Database URL not set; running without DB")
            _db_ready = False
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        _db_ready = False

    yield

    # Shutdown
    logger.info("Shutting down IRAM MCP Server...")
    if agent_orchestrator:
        # Cleanup if needed
        pass


# Create FastAPI app with configuration
config = get_config()

app = FastAPI(
    title="IRAM - Instagram Research Agent MCP",
    description="""
    A comprehensive MCP server for Instagram research, analysis, and automation.
    
    ## Features
    
    * **Profile Analysis**: Comprehensive Instagram profile analysis with engagement metrics
    * **Content Analysis**: AI-powered content analysis including sentiment and topics
    * **Comparison Tools**: Side-by-side profile and performance comparisons
    * **Trend Analysis**: Hashtag and content trend identification
    * **Background Jobs**: Asynchronous processing for long-running tasks
    * **Real-time Updates**: Live job progress and status updates
    
    ## Authentication
    
    The API supports multiple authentication methods:
    * Bearer token authentication for API clients
    * Development mode with automatic mock user
    * External OAuth providers (Supabase, GitHub, Google)
    
    ## Rate Limits
    
    * 60 requests per minute per IP address
    * 10 requests burst limit
    * Higher limits available with authentication
    
    ## Support
    
    For support and documentation, visit our [GitHub repository](https://github.com/user/iram-agent).
    """,
    version=config.version,
    lifespan=lifespan,
    debug=config.is_development(),
    contact={
        "name": "IRAM Support",
        "url": "https://github.com/user/iram-agent",
        "email": "support@iram.dev"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    openapi_tags=[
        {
            "name": "Health",
            "description": "Health check and service status endpoints"
        },
        {
            "name": "API v1",
            "description": "Modern versioned API endpoints with comprehensive validation"
        },
        {
            "name": "Legacy API (Deprecated)",
            "description": "Legacy endpoints - please migrate to API v1"
        },
        {
            "name": "Jobs",
            "description": "Background job management and monitoring"
        },
        {
            "name": "Configuration",
            "description": "Service configuration and capabilities"
        },
        {
            "name": "Metrics",
            "description": "Performance metrics and monitoring data"
        }
    ],
    swagger_ui_parameters={
        "defaultModelsExpandDepth": 2,
        "defaultModelExpandDepth": 2,
        "displayOperationId": True,
        "displayRequestDuration": True
    }
)

# Add middleware in order (important: order matters)
# Security headers first
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware with configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=config.server.cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request tracing middleware
app.add_middleware(RequestTracingMiddleware)

# Rate limiting middleware (if enabled)
if config.features.enable_rate_limiting:
    app.add_middleware(
        RateLimitingMiddleware,
        config_name="api_per_ip"
    )

# Error handling middleware
app.add_middleware(
    ErrorHandlingMiddleware,
    include_debug_info=config.is_development()
)

# Add exception handlers
http_handler, general_handler = create_error_handler(config.is_development())
app.add_exception_handler(HTTPException, http_handler)
app.add_exception_handler(Exception, general_handler)

# Include API v1 router
app.include_router(v1_router, tags=["API v1"])

# Add backwards compatibility router for legacy endpoints
legacy_router = APIRouter(prefix="", tags=["Legacy API (Deprecated)"])

# Backwards compatibility: redirect old endpoints to v1
@legacy_router.get("/execute_task", deprecated=True)
async def legacy_execute_task_redirect():
    """Redirect to v1 API endpoint."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/v1/analysis/execute", status_code=301)

@legacy_router.get("/fetch_profile", deprecated=True) 
async def legacy_fetch_profile_redirect():
    """Redirect to v1 API endpoint."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/v1/profiles/analyze", status_code=301)

app.include_router(legacy_router)


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
            config = get_config()
            agent_config = {
                "instagram_username": config.instagram.username,
                "instagram_password": config.instagram.password,
                "openai_api_key": config.llm.openai_api_key,
                "openrouter_api_key": config.llm.openrouter_api_key,
                "llm_model": config.llm.default_model,
                "debug": config.is_development(),
                "requests_per_minute": config.instagram.requests_per_minute,
                "delay_min": config.instagram.delay_min,
                "delay_max": config.instagram.delay_max,
            }
            agent_orchestrator = create_instagram_agent(agent_config)
            logger.info(
                "Agent orchestrator initialized successfully (lazy)",
                extra={
                    "has_instagram_auth": config.has_instagram_auth(),
                    "primary_llm_provider": config.get_primary_llm_provider() if config.has_llm_provider() else None,
                    "features_enabled": {
                        "topic_modeling": config.features.enable_topic_modeling,
                        "computer_vision": config.features.enable_computer_vision,
                        "sentiment_analysis": config.features.enable_sentiment_analysis
                    }
                }
            )
        except Exception as e:
            logger.error(f"Failed to initialize agent orchestrator: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Agent initialization failed: {str(e)}"
            )
    return agent_orchestrator


# Dependency to get Instagram tools
def get_instagram_tools() -> InstagramMCPTools:
    """Get or lazily create the Instagram tools instance."""
    global instagram_tools
    if instagram_tools is None:
        try:
            config = get_config()
            tools_config = {
                "instagram_username": config.instagram.username,
                "instagram_password": config.instagram.password,
                "session_file": config.instagram.session_file,
                "requests_per_minute": config.instagram.requests_per_minute,
                "delay_min": config.instagram.delay_min,
                "delay_max": config.instagram.delay_max,
            }
            instagram_tools = create_instagram_tools(tools_config)
            logger.info("Instagram tools initialized successfully (lazy)")
        except Exception as e:
            logger.error(f"Failed to initialize Instagram tools: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Instagram tools initialization failed: {str(e)}"
            )
    return instagram_tools


# Health and readiness endpoints
@app.get("/health", tags=["Health"], summary="Health Check")
async def health_check():
    """Basic health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "iram-mcp-server",
        "version": get_config().version
    }

@app.get("/live", tags=["Health"], summary="Liveness Probe")
async def liveness():
    """Liveness probe - indicates if the service is alive."""
    import time
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
    }

@app.get("/ready", tags=["Health"], summary="Readiness Probe")
async def readiness():
    """Readiness probe - indicates if the service is ready to handle requests."""
    config = get_config()
    
    dependencies = {
        "database": _db_ready if config.has_database() else True,
        "llm_provider": config.has_llm_provider(),
        "instagram_auth": config.has_instagram_auth() or config.instagram.public_fallback,
    }
    
    # Check if all critical dependencies are ready
    ready = all(dependencies.values())
    
    return {
        "ready": ready,
        "dependencies": dependencies,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": config.environment
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


# Instagram MCP endpoints
@app.post("/instagram/send_dm")
async def instagram_send_dm(
    username: str,
    message: str,
    media_path: Optional[str] = None,
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Send a direct message via Instagram."""
    try:
        result = await tools.send_direct_message(username, message, media_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram DM failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/instagram/get_dms")
async def instagram_get_dms(
    username: Optional[str] = None,
    limit: int = 20,
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Get direct messages from Instagram."""
    try:
        result = await tools.get_direct_messages(username, limit)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram get DMs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/instagram/download_media")
async def instagram_download_media(
    media_url_or_shortcode: str,
    output_dir: Optional[str] = None,
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Download media from Instagram."""
    try:
        result = await tools.download_media(media_url_or_shortcode, output_dir)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram media download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/instagram/upload_photo")
async def instagram_upload_photo(
    image_path: str,
    caption: str = "",
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Upload a photo to Instagram."""
    try:
        result = await tools.upload_photo(image_path, caption)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram photo upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/instagram/upload_video")
async def instagram_upload_video(
    video_path: str,
    caption: str = "",
    thumbnail_path: Optional[str] = None,
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Upload a video to Instagram."""
    try:
        result = await tools.upload_video(video_path, caption, thumbnail_path)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram video upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/instagram/business_insights")
async def instagram_business_insights(
    username: Optional[str] = None,
    period: str = "week",
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Get Instagram business insights."""
    try:
        result = await tools.get_business_insights(username, period)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram business insights failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/instagram/stories/{username}")
async def instagram_get_stories(
    username: str,
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Get user stories from Instagram."""
    try:
        result = await tools.get_user_stories(username)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram get stories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/instagram/upload_story")
async def instagram_upload_story(
    image_path: str,
    mentions: Optional[List[str]] = None,
    hashtags: Optional[List[str]] = None,
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Upload a story to Instagram."""
    try:
        result = await tools.upload_story_photo(image_path, mentions, hashtags)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram story upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/instagram/like_media")
async def instagram_like_media(
    media_id_or_shortcode: str,
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Like a media post on Instagram."""
    try:
        result = await tools.like_media(media_id_or_shortcode)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram like media failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/instagram/comment_media")
async def instagram_comment_media(
    media_id_or_shortcode: str,
    comment: str,
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Comment on a media post on Instagram."""
    try:
        result = await tools.comment_media(media_id_or_shortcode, comment)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram comment media failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/instagram/follow_user")
async def instagram_follow_user(
    username: str,
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Follow a user on Instagram."""
    try:
        result = await tools.follow_user(username)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram follow user failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/instagram/media_info")
async def instagram_media_info(
    media_id_or_shortcode: str,
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Get detailed media information from Instagram."""
    try:
        result = await tools.get_media_info(media_id_or_shortcode)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram media info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/instagram/search_users")
async def instagram_search_users_advanced(
    query: str,
    limit: int = 20,
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Advanced user search on Instagram."""
    try:
        result = await tools.search_users_advanced(query, limit)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram search users failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/instagram/logout")
async def instagram_logout(
    tools: InstagramMCPTools = Depends(get_instagram_tools)
):
    """Logout from Instagram."""
    try:
        result = await tools.logout()
        global instagram_tools
        instagram_tools = None  # Reset global instance
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Instagram logout failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task endpoints
@app.post("/schedule_analysis")
async def schedule_analysis(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    agent: InstagramAgentOrchestrator = Depends(get_agent)
):
    """Schedule a background analysis task and persist a job record if DB is configured."""
    try:
        job_id = None
        # Persist job record if DB available
        if _db_ready:
            from .db import session_scope
            from .models import Job
            async with session_scope() as session:
                job = Job(task=request.task, status="queued", progress=0, payload=request.context or {})
                session.add(job)
                await session.flush()
                job_id = job.id

        def run_background_task(job_id_local: Optional[str]):
            try:
                # Execute task
                result = asyncio.run(agent.execute_task(request.task, request.context))
                logger.info(f"Background task completed: {result}")
                if _db_ready and job_id_local:
                    from .db import session_scope
                    from .models import Job
                    async def _update():
                        async with session_scope() as session:
                            db_job = await session.get(Job, job_id_local)
                            if db_job:
                                db_job.status = "completed" if result.get("success") else "failed"
                                db_job.progress = 100 if result.get("success") else db_job.progress
                                db_job.result = result
                    asyncio.run(_update())
            except Exception as e:
                logger.error(f"Background task failed: {e}")
                if _db_ready and job_id_local:
                    from .db import session_scope
                    from .models import Job
                    async def _fail():
                        async with session_scope() as session:
                            db_job = await session.get(Job, job_id_local)
                            if db_job:
                                db_job.status = "failed"
                                db_job.error = {"message": str(e)}
                    asyncio.run(_fail())

        background_tasks.add_task(run_background_task, job_id)

        return JSONResponse(content={
            "success": True,
            "message": "Task scheduled for background execution",
            "task": request.task,
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Task scheduling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoints
@app.get("/config", tags=["Configuration"], summary="Service Configuration")
async def get_server_config():
    """Get current server configuration (non-sensitive information only)."""
    config = get_config()
    
    return {
        "service": "IRAM MCP Server",
        "version": config.version,
        "environment": config.environment,
        "debug": config.is_development(),
        "capabilities": {
            "database": config.has_database(),
            "redis": config.has_redis(),
            "llm_provider": config.has_llm_provider(),
            "instagram_auth": config.has_instagram_auth(),
            "public_fallback": config.instagram.public_fallback
        },
        "features": {
            "topic_modeling": config.features.enable_topic_modeling,
            "computer_vision": config.features.enable_computer_vision,
            "sentiment_analysis": config.features.enable_sentiment_analysis,
            "playwright": config.features.enable_playwright,
            "instagrapi": config.features.enable_instagrapi,
            "background_jobs": config.features.enable_background_jobs,
            "scheduling": config.features.enable_scheduling,
            "webhooks": config.features.enable_webhooks,
            "rate_limiting": config.features.enable_rate_limiting
        },
        "limits": {
            "rate_limit_per_minute": config.server.rate_limit_per_minute,
            "rate_limit_burst": config.server.rate_limit_burst,
            "max_request_size": config.server.max_request_size,
            "request_timeout": config.server.request_timeout
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
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


# Add startup event to track server start time
@app.on_event("startup")
async def startup_event():
    """Initialize server state on startup."""
    import time
    app.state.start_time = time.time()
    logger.info("IRAM MCP Server startup completed")

# Add job inspection endpoints
@app.get("/jobs", tags=["Jobs"], summary="List Analysis Jobs")
async def list_jobs(limit: int = 50, offset: int = 0):
    """List analysis jobs with pagination."""
    if not _db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    
    from .db import session_scope
    from .models import Job
    from sqlalchemy import desc
    
    try:
        async with session_scope() as session:
            jobs = await session.execute(
                Job.__table__.select()
                .order_by(desc(Job.created_at))
                .limit(limit)
                .offset(offset)
            )
            job_list = [dict(job) for job in jobs.fetchall()]
            
            return {
                "jobs": job_list,
                "limit": limit,
                "offset": offset,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve jobs")

@app.get("/jobs/{job_id}", tags=["Jobs"], summary="Get Job Details")
async def get_job(job_id: int):
    """Get detailed information about a specific job."""
    if not _db_ready:
        raise HTTPException(status_code=503, detail="Database not available")
    
    from .db import session_scope
    from .models import Job
    
    try:
        async with session_scope() as session:
            job = await session.get(Job, job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            
            return {
                "id": job.id,
                "task": job.task,
                "status": job.status,
                "progress": job.progress,
                "payload": job.payload,
                "result": job.result,
                "error": job.error,
                "created_at": job.created_at.isoformat() + "Z" if job.created_at else None,
                "updated_at": job.updated_at.isoformat() + "Z" if job.updated_at else None,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve job")

# Add metrics endpoint
@app.get("/metrics", tags=["Metrics"], summary="Service Metrics")
async def get_metrics():
    """Get server metrics for monitoring."""
    import time
    config = get_config()
    
    # Basic metrics
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        "environment": config.environment,
        "version": config.version,
    }
    
    # Add job metrics if database is available
    if _db_ready:
        try:
            from .db import session_scope
            from .models import Job
            from sqlalchemy import func, text
            
            async with session_scope() as session:
                # Job status counts
                status_counts = await session.execute(
                    session.query(Job.status, func.count(Job.id))
                    .group_by(Job.status)
                )
                
                job_metrics = {
                    "jobs_by_status": dict(status_counts.fetchall()),
                    "total_jobs": sum(dict(status_counts.fetchall()).values())
                }
                
                metrics["jobs"] = job_metrics
        except Exception as e:
            logger.warning(f"Failed to get job metrics: {e}")
            metrics["jobs"] = {"error": "Failed to retrieve job metrics"}
    
    return metrics

# Main execution
def main():
    """Main function to run the server."""
    config = get_config()
    
    logger.info(
        f"Starting IRAM MCP Server on {config.server.host}:{config.server.port}",
        extra={
            "host": config.server.host,
            "port": config.server.port,
            "environment": config.environment,
            "debug": config.is_development(),
            "version": config.version
        }
    )
    
    uvicorn.run(
        "src.mcp_server:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload and config.is_development(),
        log_level="debug" if config.is_development() else "info",
        timeout_keep_alive=config.server.request_timeout,
        limit_max_requests=10000,
        limit_concurrency=1000
    )


if __name__ == "__main__":
    main()