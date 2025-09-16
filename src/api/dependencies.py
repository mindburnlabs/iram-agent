"""
IRAM API Dependencies

Shared dependencies for authentication, rate limiting, and other
cross-cutting concerns used across API endpoints.
"""

import time
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..config import get_config
from ..logging_config import get_logger
from ..mcp_server import get_agent as _get_agent
from .models import User

logger = get_logger(__name__)

# Security scheme for JWT tokens
security = HTTPBearer(auto_error=False)

# In-memory store for demo purposes (use Redis in production)
_user_sessions: Dict[str, Dict[str, Any]] = {}
_rate_limit_store: Dict[str, list] = {}


class MockUser:
    """Mock user for development/testing without full auth system."""
    def __init__(self, user_id: int = 1, email: str = "demo@example.com", username: str = "demo"):
        self.id = user_id
        self.email = email
        self.username = username
        self.is_active = True
        self.created_at = "2024-01-01T00:00:00Z"


def get_agent():
    """Get the agent orchestrator dependency (wrapper for MCP server function)."""
    return _get_agent()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """
    Get the current authenticated user from JWT token.
    
    Returns None if no valid authentication is provided, allowing
    endpoints to handle anonymous access as needed.
    """
    config = get_config()
    
    # Skip authentication in development mode
    if config.is_development():
        logger.debug("Development mode: using mock user")
        mock_user = MockUser()
        return User(
            id=mock_user.id,
            email=mock_user.email,
            username=mock_user.username,
            is_active=mock_user.is_active,
            created_at=mock_user.created_at
        )
    
    if not credentials:
        logger.debug("No credentials provided")
        return None
    
    try:
        # For now, use a simple token-based system
        # In production, implement proper JWT validation
        token = credentials.credentials
        
        # Check if token exists in our mock session store
        if token in _user_sessions:
            user_data = _user_sessions[token]
            
            # Check if session is still valid
            if time.time() < user_data.get("expires_at", 0):
                return User(**user_data["user"])
            else:
                # Remove expired session
                del _user_sessions[token]
                logger.info(f"Expired session removed for token: {token[:8]}...")
        
        logger.warning(f"Invalid or expired token: {token[:8]}...")
        return None
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None


async def require_authentication(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """
    Require authentication for protected endpoints.
    
    Raises HTTPException if user is not authenticated.
    """
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not current_user.is_active:
        raise HTTPException(
            status_code=403,
            detail="User account is inactive"
        )
    
    return current_user


async def check_rate_limits(request: Request) -> None:
    """
    Check rate limits for the current request.
    
    Raises HTTPException if rate limit is exceeded.
    """
    config = get_config()
    
    # Skip rate limiting if disabled
    if not config.features.enable_rate_limiting:
        return
    
    # Get client IP
    client_ip = get_client_ip(request)
    current_time = time.time()
    
    # Clean up old requests (older than 1 minute)
    if client_ip in _rate_limit_store:
        _rate_limit_store[client_ip] = [
            req_time for req_time in _rate_limit_store[client_ip]
            if current_time - req_time < 60
        ]
    else:
        _rate_limit_store[client_ip] = []
    
    # Check rate limits
    recent_requests = len(_rate_limit_store[client_ip])
    
    if recent_requests >= config.server.rate_limit_per_minute:
        logger.warning(
            f"Rate limit exceeded for {client_ip}",
            extra={
                "client_ip": client_ip,
                "requests_count": recent_requests,
                "limit": config.server.rate_limit_per_minute
            }
        )
        
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {config.server.rate_limit_per_minute} requests per minute.",
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": str(config.server.rate_limit_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(current_time + 60))
            }
        )
    
    # Record this request
    _rate_limit_store[client_ip].append(current_time)


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request headers.
    
    Handles common proxy headers like X-Forwarded-For.
    """
    # Check for forwarded IP from proxy
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct client IP
    return request.client.host if request.client else "unknown"


async def check_permissions(
    resource: str,
    action: str,
    current_user: Optional[User] = Depends(get_current_user)
) -> None:
    """
    Check if the current user has permission to perform an action on a resource.
    
    For now, implements basic permission checks. In production, integrate
    with a proper RBAC system.
    """
    # Allow anonymous access to public resources
    public_resources = ["health", "status", "docs", "openapi"]
    if resource in public_resources:
        return
    
    # Require authentication for protected resources
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required for this resource",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Check if user is active
    if not current_user.is_active:
        raise HTTPException(
            status_code=403,
            detail="User account is inactive"
        )
    
    # For now, all authenticated users have full access
    # In production, implement role-based access control
    logger.debug(
        f"Permission check passed for user {current_user.id}",
        extra={
            "user_id": current_user.id,
            "resource": resource,
            "action": action
        }
    )


async def validate_content_length(request: Request) -> None:
    """
    Validate request content length against configured limits.
    """
    config = get_config()
    
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            content_size = int(content_length)
            if content_size > config.server.max_request_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request too large. Maximum size: {config.server.max_request_size} bytes"
                )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid content-length header"
            )


class RateLimiter:
    """
    Async context manager for rate limiting specific operations.
    
    Usage:
        async with RateLimiter("expensive_operation", limit=5):
            # Perform expensive operation
            pass
    """
    
    def __init__(self, operation: str, limit: int = 10, window: int = 60):
        self.operation = operation
        self.limit = limit
        self.window = window
        self._store_key = f"rate_limit:{operation}"
    
    async def __aenter__(self):
        current_time = time.time()
        
        # Get or create operation request history
        if self._store_key not in _rate_limit_store:
            _rate_limit_store[self._store_key] = []
        
        # Clean old requests
        _rate_limit_store[self._store_key] = [
            req_time for req_time in _rate_limit_store[self._store_key]
            if current_time - req_time < self.window
        ]
        
        # Check limit
        if len(_rate_limit_store[self._store_key]) >= self.limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded for operation '{self.operation}'. "
                       f"Limit: {self.limit} requests per {self.window} seconds."
            )
        
        # Record this request
        _rate_limit_store[self._store_key].append(current_time)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Nothing to clean up for now
        pass


# Dependency factories for common use cases
def require_role(required_role: str):
    """
    Create a dependency that requires a specific user role.
    
    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: User = Depends(require_role("admin"))):
            pass
    """
    async def role_checker(current_user: User = Depends(require_authentication)) -> User:
        # For now, all users are considered "user" role
        # In production, check actual user roles from database
        user_role = "user"  # Get from user.roles or similar
        
        if user_role != required_role and required_role != "user":
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return current_user
    
    return role_checker


def require_feature_flag(feature: str):
    """
    Create a dependency that requires a feature flag to be enabled.
    
    Usage:
        @app.get("/experimental")
        async def experimental_endpoint(_: None = Depends(require_feature_flag("experimental"))):
            pass
    """
    async def feature_checker() -> None:
        config = get_config()
        
        # Check if feature is enabled in configuration
        if not getattr(config.features, f"enable_{feature}", False):
            raise HTTPException(
                status_code=503,
                detail=f"Feature '{feature}' is currently disabled"
            )
    
    return feature_checker


# Mock authentication functions (replace with real auth in production)
def create_mock_session(user_id: int = 1, duration_hours: int = 24) -> str:
    """Create a mock authentication session for development/testing."""
    import secrets
    
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + (duration_hours * 3600)
    
    mock_user = MockUser(user_id=user_id)
    
    _user_sessions[token] = {
        "user": {
            "id": mock_user.id,
            "email": mock_user.email,
            "username": mock_user.username,
            "is_active": mock_user.is_active,
            "created_at": mock_user.created_at
        },
        "expires_at": expires_at,
        "created_at": time.time()
    }
    
    logger.info(f"Created mock session for user {user_id}: {token[:8]}...")
    return token


def invalidate_session(token: str) -> bool:
    """Invalidate an authentication session."""
    if token in _user_sessions:
        del _user_sessions[token]
        logger.info(f"Invalidated session: {token[:8]}...")
        return True
    return False


def list_active_sessions() -> Dict[str, Dict[str, Any]]:
    """List all active sessions (for debugging)."""
    current_time = time.time()
    active_sessions = {}
    
    for token, session in _user_sessions.items():
        if current_time < session.get("expires_at", 0):
            active_sessions[token] = {
                "user_id": session["user"]["id"],
                "expires_at": session["expires_at"],
                "created_at": session.get("created_at", 0)
            }
    
    return active_sessions