"""
IRAM FastAPI Application

Main FastAPI application integrating all IRAM components including
cache, database, job queue, authentication, and API routes.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import uvicorn

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field

from .config import get_config, IRamConfig
from .logging_config import get_logger, setup_logging
from .cache import initialize_cache, close_cache, get_cache, cache_maintenance, warm_cache
from .db import init_database, close_database, get_db
from .scheduler import start_scheduler, stop_scheduler
from .middleware import setup_middleware, RateLimitMiddleware
from .repository import UserRepository, InstagramProfileRepository, JobRepository

logger = get_logger(__name__)


# Pydantic models for API
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    timestamp: str
    services: Dict[str, str]
    cache_stats: Optional[Dict[str, Any]] = None


class CacheStatsResponse(BaseModel):
    """Cache statistics response model."""
    backend: str
    stats: Dict[str, Any]
    maintenance: Optional[Dict[str, Any]] = None


class ConfigResponse(BaseModel):
    """Configuration response model."""
    environment: str
    version: str
    features: Dict[str, bool]
    providers: Dict[str, bool]


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting IRAM application...")
    
    # Initialize components
    try:
        # Load configuration
        config = get_config()
        
        # Initialize database
        if config.has_database():
            await init_database()
            logger.info("Database initialized")
        
        # Initialize cache
        await initialize_cache()
        logger.info("Cache initialized")
        
        # Start scheduler
        if config.features.enable_scheduling:
            await start_scheduler()
            logger.info("Scheduler started")
        
        # Warm cache with common data
        if config.environment == "production":
            asyncio.create_task(warm_cache())
        
        logger.info("IRAM application started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start IRAM application: {e}")
        raise
    
    # Application is running
    yield
    
    # Shutdown
    logger.info("Shutting down IRAM application...")
    
    try:
        # Stop scheduler
        if config.features.enable_scheduling:
            await stop_scheduler()
            logger.info("Scheduler stopped")
        
        # Close cache
        await close_cache()
        logger.info("Cache closed")
        
        # Close database
        if config.has_database():
            await close_database()
            logger.info("Database closed")
        
        logger.info("IRAM application shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()
    
    app = FastAPI(
        title="Instagram Research Agent MCP (IRAM)",
        description="""
        **IRAM** is an autonomous Instagram research and analysis platform that provides:
        
        - 🔍 **Profile Analysis**: Deep insights into Instagram profiles and engagement patterns
        - 📊 **Content Analytics**: Advanced analysis of posts, stories, and media
        - 🤖 **AI-Powered Insights**: LLM-driven trend analysis and content recommendations  
        - 📈 **Performance Metrics**: Track growth, engagement, and audience insights
        - ⚡ **Real-time Processing**: Async task processing with background jobs
        - 🔒 **Secure & Scalable**: Built with FastAPI, Redis caching, and PostgreSQL
        
        ## Features
        
        - Multi-method Instagram scraping with anti-detection
        - Comprehensive content analysis and sentiment tracking
        - Automated scheduling and background processing
        - RESTful API with OpenAPI documentation
        - Rate limiting and caching for optimal performance
        
        ## Getting Started
        
        1. Configure your Instagram credentials via environment variables
        2. Set up your preferred LLM provider (OpenRouter or OpenAI)
        3. Use the `/profiles/{username}/analyze` endpoint to start analysis
        4. Monitor job progress via `/jobs/{job_id}` endpoint
        
        For more information, visit the [IRAM Documentation](https://github.com/your-org/iram).
        """,
        version=config.version,
        debug=config.is_development(),
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "health",
                "description": "System health and monitoring endpoints"
            },
            {
                "name": "profiles", 
                "description": "Instagram profile analysis and management"
            },
            {
                "name": "analysis",
                "description": "Content analysis and insights generation"
            },
            {
                "name": "jobs",
                "description": "Background job management and monitoring"
            },
            {
                "name": "cache",
                "description": "Cache management and statistics"
            },
            {
                "name": "system",
                "description": "System configuration and administration"
            },
            {
                "name": "rate-limiting",
                "description": "Rate limiting management and monitoring"
            }
        ]
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Add CORS middleware
    if config.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_origins,
            allow_credentials=config.server.cors_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add trusted host middleware
    if config.server.allowed_hosts != ["*"]:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=config.server.allowed_hosts
        )
    
    # Setup routes
    setup_routes(app)
    
    # Include routers
    try:
        from .auth_routes import router as auth_router
        from .instagram_routes import router as instagram_router
        from .analysis_routes import router as analysis_router
        from .rate_limit_routes import router as rate_limit_router
        
        app.include_router(auth_router)
        app.include_router(instagram_router)
        app.include_router(analysis_router)
        app.include_router(rate_limit_router)
        
        logger.info("All route modules included successfully")
    except ImportError as e:
        logger.warning(f"Some route modules not available: {e}")
    
    return app


def setup_routes(app: FastAPI):
    """Setup all API routes."""
    
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint redirect to docs."""
        return {"message": "Welcome to IRAM API. Visit /docs for documentation."}
    
    @app.get("/health", 
             response_model=HealthResponse,
             tags=["health"],
             summary="Health Check",
             description="Get application health status and service availability")
    async def health_check():
        """Get application health status."""
        config = get_config()
        cache = await get_cache()
        
        # Check service status
        services = {
            "api": "healthy",
            "database": "healthy" if config.has_database() else "not_configured",
            "cache": "healthy",
            "llm_provider": "healthy" if config.has_llm_provider() else "not_configured",
            "instagram": "configured" if config.has_instagram_auth() else "not_configured"
        }
        
        # Get cache stats
        cache_stats = await cache.get_stats()
        
        return HealthResponse(
            status="healthy",
            version=config.version,
            timestamp=str(asyncio.get_event_loop().time()),
            services=services,
            cache_stats=cache_stats
        )
    
    @app.get("/config",
             response_model=ConfigResponse,
             tags=["system"],
             summary="Get Configuration",
             description="Get current application configuration and feature flags")
    async def get_app_config():
        """Get application configuration."""
        config = get_config()
        
        return ConfigResponse(
            environment=config.environment,
            version=config.version,
            features={
                "topic_modeling": config.features.enable_topic_modeling,
                "computer_vision": config.features.enable_computer_vision,
                "sentiment_analysis": config.features.enable_sentiment_analysis,
                "background_jobs": config.features.enable_background_jobs,
                "scheduling": config.features.enable_scheduling,
                "rate_limiting": config.features.enable_rate_limiting,
            },
            providers={
                "database": config.has_database(),
                "redis": config.has_redis(),
                "instagram": config.has_instagram_auth(),
                "llm": config.has_llm_provider(),
            }
        )
    
    @app.get("/cache/stats",
             response_model=CacheStatsResponse,
             tags=["cache"],
             summary="Get Cache Statistics",
             description="Get detailed cache performance statistics")
    async def get_cache_stats():
        """Get cache statistics."""
        cache = await get_cache()
        stats = await cache.get_stats()
        
        return CacheStatsResponse(
            backend=stats.get("backend", "unknown"),
            stats=stats
        )
    
    @app.post("/cache/maintenance",
              response_model=CacheStatsResponse,
              tags=["cache"],
              summary="Run Cache Maintenance",
              description="Perform cache maintenance including cleanup and optimization")
    async def run_cache_maintenance(background_tasks: BackgroundTasks):
        """Run cache maintenance tasks."""
        # Run maintenance in background
        background_tasks.add_task(cache_maintenance)
        
        cache = await get_cache()
        stats = await cache.get_stats()
        
        return CacheStatsResponse(
            backend=stats.get("backend", "unknown"),
            stats=stats,
            maintenance={"status": "scheduled"}
        )
    
    @app.delete("/cache/clear/{pattern}",
                tags=["cache"],
                summary="Clear Cache Pattern",
                description="Clear cache entries matching a specific pattern")
    async def clear_cache_pattern(pattern: str):
        """Clear cache entries matching pattern."""
        cache = await get_cache()
        cleared = await cache.clear_pattern(pattern)
        
        return {"message": f"Cleared {cleared} cache entries matching pattern: {pattern}"}
    
    # Placeholder routes for future implementation
    
    @app.get("/profiles/{username}",
             tags=["profiles"],
             summary="Get Profile",
             description="Get Instagram profile information")
    async def get_profile(username: str):
        """Get Instagram profile information."""
        # TODO: Implement profile fetching
        return {"message": f"Profile endpoint for {username} - Coming soon!"}
    
    @app.post("/profiles/{username}/analyze",
              tags=["profiles"],
              summary="Analyze Profile",
              description="Start comprehensive profile analysis")
    async def analyze_profile(username: str, background_tasks: BackgroundTasks):
        """Start profile analysis."""
        # TODO: Implement profile analysis job
        return {"message": f"Analysis started for {username} - Coming soon!"}
    
    @app.get("/jobs/{job_id}",
             tags=["jobs"],
             summary="Get Job Status",
             description="Get background job status and results")
    async def get_job_status(job_id: str):
        """Get job status and results."""
        # TODO: Implement job status tracking
        return {"message": f"Job status for {job_id} - Coming soon!"}
    
    @app.get("/jobs",
             tags=["jobs"],
             summary="List Jobs",
             description="List all background jobs with filtering options")
    async def list_jobs(
        status: Optional[str] = None,
        limit: int = Query(20, le=100),
        offset: int = Query(0, ge=0)
    ):
        """List background jobs."""
        # TODO: Implement job listing
        return {"message": "Job listing - Coming soon!"}


def custom_openapi():
    """Generate custom OpenAPI schema."""
    config = get_config()
    app = create_app()
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Instagram Research Agent MCP (IRAM)",
        version=config.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom info
    openapi_schema["info"].update({
        "contact": {
            "name": "IRAM Support",
            "url": "https://github.com/your-org/iram",
        },
        "license": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    })
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Create app instance
app = create_app()
app.openapi = custom_openapi


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url.path)
        }
    )


# Development server
def run_dev_server():
    """Run development server."""
    config = get_config()
    setup_logging()
    
    uvicorn.run(
        "src.app:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload or config.is_development(),
        log_level=config.logging.level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    run_dev_server()