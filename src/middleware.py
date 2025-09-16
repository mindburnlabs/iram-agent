"""
IRAM Middleware Components

Provides middleware for error handling, request tracing, rate limiting,
and other cross-cutting concerns.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import traceback

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import get_config
from .logging_config import get_logger, set_trace_id, set_correlation_id, get_trace_id


logger = get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for consistent error handling and response formatting."""
    
    def __init__(self, app, include_debug_info: bool = False):
        super().__init__(app)
        self.include_debug_info = include_debug_info
        
    async def dispatch(self, request: Request, call_next):
        """Handle request and format any errors consistently."""
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as exc:
            return await self._handle_http_exception(request, exc)
        
        except StarletteHTTPException as exc:
            return await self._handle_http_exception(request, exc)
            
        except Exception as exc:
            return await self._handle_general_exception(request, exc)
    
    async def _handle_http_exception(
        self, 
        request: Request, 
        exc: HTTPException
    ) -> JSONResponse:
        """Handle HTTP exceptions with consistent format."""
        error_data = {
            "error": True,
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail if isinstance(exc.detail, str) else "HTTP error",
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "path": str(request.url.path),
            "method": request.method
        }
        
        # Add trace ID if available
        trace_id = get_trace_id()
        if trace_id:
            error_data["trace_id"] = trace_id
        
        # Add remediation guidance for common errors
        error_data["remediation"] = self._get_remediation_guidance(exc.status_code)
        
        # Add debug info in development
        if self.include_debug_info and hasattr(exc, "__traceback__"):
            error_data["debug"] = {
                "traceback": traceback.format_exception(
                    type(exc), exc, exc.__traceback__
                )
            }
        
        logger.warning(
            f"HTTP exception: {exc.status_code} - {exc.detail}",
            extra={
                "status_code": exc.status_code,
                "path": str(request.url.path),
                "method": request.method
            }
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_data
        )
    
    async def _handle_general_exception(
        self, 
        request: Request, 
        exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        error_data = {
            "error": True,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "path": str(request.url.path),
            "method": request.method
        }
        
        # Add trace ID if available
        trace_id = get_trace_id()
        if trace_id:
            error_data["trace_id"] = trace_id
            
        error_data["remediation"] = (
            "This is an internal server error. Please try again later. "
            "If the problem persists, contact support with the trace ID."
        )
        
        # Add debug info in development
        if self.include_debug_info:
            error_data["debug"] = {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__)
            }
        
        logger.error(
            f"Unhandled exception: {type(exc).__name__}: {exc}",
            extra={
                "exception_type": type(exc).__name__,
                "path": str(request.url.path),
                "method": request.method
            },
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content=error_data
        )
    
    def _get_remediation_guidance(self, status_code: int) -> str:
        """Get user-friendly remediation guidance for common status codes."""
        guidance = {
            400: "Check your request parameters and ensure all required fields are provided.",
            401: "Authentication required. Please provide valid credentials.",
            403: "Access denied. Check your permissions for this resource.",
            404: "The requested resource was not found. Verify the URL and parameters.",
            405: "HTTP method not allowed for this endpoint.",
            422: "Request validation failed. Check the format of your input data.",
            429: "Rate limit exceeded. Please wait before making additional requests.",
            500: "Internal server error. Please try again later.",
            502: "Bad gateway. The service is temporarily unavailable.",
            503: "Service temporarily unavailable. Please try again later."
        }
        
        return guidance.get(
            status_code, 
            "Please check your request and try again."
        )


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware to add request tracing with trace and correlation IDs."""
    
    async def dispatch(self, request: Request, call_next):
        """Add tracing context to requests."""
        # Generate or extract trace ID
        trace_id = request.headers.get("X-Trace-ID")
        if not trace_id:
            trace_id = set_trace_id()
        else:
            set_trace_id(trace_id)
        
        # Generate or extract correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = set_correlation_id()
        else:
            set_correlation_id(correlation_id)
        
        # Log request start
        start_time = time.time()
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": str(request.url.path),
                "query_params": dict(request.query_params),
                "user_agent": request.headers.get("User-Agent"),
                "client_host": request.client.host if request.client else None
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add trace headers to response
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-Correlation-ID"] = correlation_id
            
            # Log request completion
            duration = time.time() - start_time
            logger.info(
                f"Request completed: {request.method} {request.url.path} - "
                f"{response.status_code} in {duration:.3f}s",
                extra={
                    "method": request.method,
                    "path": str(request.url.path),
                    "status_code": response.status_code,
                    "duration_ms": duration * 1000,
                    "response_size": response.headers.get("Content-Length", 0)
                }
            )
            
            return response
            
        except Exception as exc:
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} - "
                f"Exception after {duration:.3f}s",
                extra={
                    "method": request.method,
                    "path": str(request.url.path),
                    "duration_ms": duration * 1000,
                    "exception_type": type(exc).__name__
                },
                exc_info=True
            )
            raise


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Redis-backed distributed rate limiting middleware."""
    
    def __init__(self, app, config_name: str = "api_per_ip"):
        super().__init__(app)
        self.config_name = config_name
        self.rate_limiter = None
    
    async def _get_rate_limiter(self):
        """Get rate limiter instance lazily."""
        if not self.rate_limiter:
            from .rate_limiter import get_rate_limiter
            self.rate_limiter = await get_rate_limiter()
        return self.rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting based on client IP with Redis backend."""
        try:
            client_ip = self._get_client_ip(request)
            endpoint = f"{request.method}:{request.url.path}"
            
            # Get rate limiter
            limiter = await self._get_rate_limiter()
            
            # Check rate limits
            result = await limiter.check_rate_limit(
                config_name=self.config_name,
                identifier=client_ip,
                endpoint=endpoint,
                cost=1
            )
            
            if not result.allowed:
                return await self._rate_limit_exceeded_response(request, client_ip, result)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(result.limit)
            response.headers["X-RateLimit-Remaining"] = str(result.remaining)
            response.headers["X-RateLimit-Algorithm"] = result.algorithm or "unknown"
            response.headers["X-RateLimit-Scope"] = result.scope or "unknown"
            
            if result.reset_time:
                response.headers["X-RateLimit-Reset"] = str(int(result.reset_time.timestamp()))
            
            if result.retry_after:
                response.headers["Retry-After"] = str(result.retry_after)
            
            return response
            
        except Exception as e:
            # Fail open - if rate limiting fails, allow the request
            logger.error(f"Rate limiting middleware error: {e}")
            return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get the client IP address, considering proxies."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def _rate_limit_exceeded_response(
        self, 
        request: Request, 
        client_ip: str,
        result
    ) -> JSONResponse:
        """Return a rate limit exceeded response."""
        error_data = {
            "error": True,
            "error_code": "RATE_LIMIT_EXCEEDED",
            "message": f"Rate limit exceeded. Limit: {result.limit} requests per window.",
            "status_code": 429,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "remediation": "Please wait before making additional requests.",
            "rate_limit": {
                "limit": result.limit,
                "remaining": result.remaining,
                "algorithm": result.algorithm,
                "scope": result.scope,
                "retry_after": result.retry_after
            }
        }
        
        if result.retry_after:
            error_data["retry_after"] = result.retry_after
        
        # Add trace ID if available
        trace_id = get_trace_id()
        if trace_id:
            error_data["trace_id"] = trace_id
        
        logger.warning(
            f"Rate limit exceeded for {client_ip}: {request.method} {request.url.path}",
            extra={
                "client_ip": client_ip,
                "method": request.method,
                "path": str(request.url.path),
                "algorithm": result.algorithm,
                "scope": result.scope,
                "limit": result.limit,
                "remaining": result.remaining
            }
        )
        
        headers = {}
        if result.retry_after:
            headers["Retry-After"] = str(result.retry_after)
        
        return JSONResponse(
            status_code=429,
            content=error_data,
            headers=headers
        )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses."""
    
    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self'"
            )
        }
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response


def create_error_handler(include_debug: bool = False):
    """Create error handler function for FastAPI exception handlers."""
    
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        middleware = ErrorHandlingMiddleware(None, include_debug_info=include_debug)
        return await middleware._handle_http_exception(request, exc)
    
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        middleware = ErrorHandlingMiddleware(None, include_debug_info=include_debug)
        return await middleware._handle_general_exception(request, exc)
    
    return http_exception_handler, general_exception_handler


def setup_middleware(app):
    """Setup all middleware for the FastAPI app."""
    config = get_config()
    
    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add request tracing middleware
    app.add_middleware(RequestTracingMiddleware)
    
    # Add error handling middleware
    app.add_middleware(
        ErrorHandlingMiddleware, 
        include_debug_info=config.is_development()
    )
    
    # Add rate limiting middleware if enabled
    if config.features.enable_rate_limiting:
        app.add_middleware(
            RateLimitingMiddleware,
            config_name="api_per_ip"
        )


# Export the RateLimitMiddleware class alias for backwards compatibility
RateLimitMiddleware = RateLimitingMiddleware
