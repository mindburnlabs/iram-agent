"""
IRAM Authentication & Authorization System

Comprehensive authentication system providing:
- JWT token-based authentication
- Redis-backed session management
- Role-based access control (RBAC)
- API key authentication
- User management and registration
"""

import uuid
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Union, Tuple
from enum import Enum

from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, Field, EmailStr
import asyncio

from .config import get_config
from .cache import get_cache, user_session_key, CacheTransaction
from .logging_config import get_logger
from .models import User, APIKey
from .repository import UserRepository, APIKeyRepository

logger = get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"
    READONLY = "readonly"


class Permission(str, Enum):
    """System permissions."""
    # Profile operations
    VIEW_PROFILES = "view_profiles"
    ANALYZE_PROFILES = "analyze_profiles"
    DELETE_PROFILES = "delete_profiles"
    
    # Analysis operations
    VIEW_ANALYSIS = "view_analysis"
    CREATE_ANALYSIS = "create_analysis"
    DELETE_ANALYSIS = "delete_analysis"
    
    # System operations
    VIEW_SYSTEM = "view_system"
    MANAGE_SYSTEM = "manage_system"
    MANAGE_USERS = "manage_users"
    
    # Cache operations
    VIEW_CACHE = "view_cache"
    MANAGE_CACHE = "manage_cache"
    
    # Job operations
    VIEW_JOBS = "view_jobs"
    MANAGE_JOBS = "manage_jobs"


# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [p for p in Permission],  # All permissions
    UserRole.ANALYST: [
        Permission.VIEW_PROFILES,
        Permission.ANALYZE_PROFILES,
        Permission.VIEW_ANALYSIS,
        Permission.CREATE_ANALYSIS,
        Permission.VIEW_SYSTEM,
        Permission.VIEW_CACHE,
        Permission.VIEW_JOBS,
        Permission.MANAGE_JOBS,
    ],
    UserRole.USER: [
        Permission.VIEW_PROFILES,
        Permission.ANALYZE_PROFILES,
        Permission.VIEW_ANALYSIS,
        Permission.CREATE_ANALYSIS,
        Permission.VIEW_JOBS,
    ],
    UserRole.READONLY: [
        Permission.VIEW_PROFILES,
        Permission.VIEW_ANALYSIS,
        Permission.VIEW_JOBS,
    ],
}


# Pydantic models
class UserLogin(BaseModel):
    """User login request model."""
    email: EmailStr
    password: str
    remember_me: bool = False


class UserRegister(BaseModel):
    """User registration request model."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2, max_length=100)
    role: UserRole = UserRole.USER


class UserProfile(BaseModel):
    """User profile response model."""
    id: int
    email: str
    full_name: str
    role: UserRole
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserProfile
    permissions: List[Permission]


class SessionInfo(BaseModel):
    """Session information model."""
    user_id: int
    email: str
    role: UserRole
    permissions: List[Permission]
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class APIKeyCreate(BaseModel):
    """API key creation model."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    permissions: List[Permission] = []
    expires_at: Optional[datetime] = None


class APIKeyInfo(BaseModel):
    """API key information model."""
    id: int
    name: str
    description: Optional[str]
    permissions: List[Permission]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
    is_active: bool


class AuthService:
    """Authentication service with JWT and Redis session management."""
    
    def __init__(self):
        self.config = get_config()
        self.secret_key = self._get_secret_key()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 30
    
    def _get_secret_key(self) -> str:
        """Get or generate secret key."""
        # In production, use a strong secret key from environment
        secret = self.config.__dict__.get('secret_key') or secrets.token_urlsafe(32)
        return secret
    
    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(
        self, 
        user_id: int, 
        email: str, 
        role: UserRole,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token."""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode = {
            "sub": str(user_id),
            "email": email,
            "role": role.value,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, user_id: int) -> str:
        """Create a JWT refresh token."""
        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        
        to_encode = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
                return None
            
            return payload
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None
    
    async def create_session(
        self, 
        user: User, 
        request: Request,
        remember_me: bool = False
    ) -> SessionInfo:
        """Create a user session and store in Redis."""
        cache = await get_cache()
        
        # Calculate session expiration
        if remember_me:
            expires_at = datetime.utcnow() + timedelta(days=30)
            ttl = 30 * 24 * 3600  # 30 days
        else:
            expires_at = datetime.utcnow() + timedelta(hours=8)
            ttl = 8 * 3600  # 8 hours
        
        # Create session info
        session_info = SessionInfo(
            user_id=user.id,
            email=user.email,
            role=UserRole(user.role),
            permissions=ROLE_PERMISSIONS.get(UserRole(user.role), []),
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent")
        )
        
        # Store session in Redis
        session_key = user_session_key(user.id)
        await cache.set(session_key, session_info.dict(), ttl=ttl)
        
        logger.info(f"Session created for user {user.email}", extra={
            "user_id": user.id,
            "session_ttl": ttl,
            "remember_me": remember_me
        })
        
        return session_info
    
    async def get_session(self, user_id: int) -> Optional[SessionInfo]:
        """Get session information from Redis."""
        cache = await get_cache()
        session_key = user_session_key(user_id)
        
        session_data = await cache.get(session_key)
        if not session_data:
            return None
        
        try:
            return SessionInfo(**session_data)
        except Exception as e:
            logger.warning(f"Invalid session data for user {user_id}: {e}")
            await cache.delete(session_key)
            return None
    
    async def invalidate_session(self, user_id: int):
        """Invalidate a user session."""
        cache = await get_cache()
        session_key = user_session_key(user_id)
        await cache.delete(session_key)
        
        logger.info(f"Session invalidated for user ID {user_id}")
    
    async def refresh_session(self, user_id: int, extend_ttl: bool = True) -> Optional[SessionInfo]:
        """Refresh session TTL and return updated session info."""
        session = await self.get_session(user_id)
        if not session:
            return None
        
        if extend_ttl:
            cache = await get_cache()
            session_key = user_session_key(user_id)
            # Extend TTL by 8 hours
            await cache.expire(session_key, 8 * 3600)
        
        return session
    
    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return f"iram_{secrets.token_urlsafe(32)}"
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        try:
            user_repo = UserRepository()
            user = await user_repo.get_by_email(email)
            
            if not user or not user.is_active:
                return None
            
            if not self.verify_password(password, user.hashed_password):
                return None
            
            # Update last login
            await user_repo.update_last_login(user.id)
            
            return user
            
        except Exception as e:
            logger.error(f"User authentication failed: {e}")
            return None
    
    async def authenticate_api_key(self, api_key: str) -> Optional[Tuple[User, APIKey]]:
        """Authenticate using API key."""
        try:
            api_key_repo = APIKeyRepository()
            key_hash = self.hash_api_key(api_key)
            
            api_key_obj = await api_key_repo.get_by_hash(key_hash)
            if not api_key_obj or not api_key_obj.is_active:
                return None
            
            # Check expiration
            if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
                return None
            
            # Get associated user
            user_repo = UserRepository()
            user = await user_repo.get_by_id(api_key_obj.user_id)
            
            if not user or not user.is_active:
                return None
            
            # Update last used
            await api_key_repo.update_last_used(api_key_obj.id)
            
            return user, api_key_obj
            
        except Exception as e:
            logger.error(f"API key authentication failed: {e}")
            return None


# Global auth service instance
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get the global authentication service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service


# Dependency functions for FastAPI
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header)
) -> User:
    """Get current authenticated user from token or API key."""
    auth_service = get_auth_service()
    
    # Try JWT token first
    if credentials:
        token_data = await auth_service.verify_token(credentials.credentials)
        if token_data and token_data.get("type") == "access":
            user_id = int(token_data.get("sub"))
            
            # Check if session exists
            session = await auth_service.get_session(user_id)
            if session:
                # Refresh session
                await auth_service.refresh_session(user_id, extend_ttl=True)
                
                # Create user object from session
                user = User(
                    id=session.user_id,
                    email=session.email,
                    role=session.role.value,
                    is_active=True
                )
                return user
    
    # Try API key
    if api_key:
        auth_result = await auth_service.authenticate_api_key(api_key)
        if auth_result:
            user, _ = auth_result
            return user
    
    raise HTTPException(
        status_code=401,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_permission(permission: Permission):
    """Dependency factory to require specific permission."""
    async def permission_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        auth_service = get_auth_service()
        session = await auth_service.get_session(current_user.id)
        
        if not session:
            raise HTTPException(status_code=401, detail="Session not found")
        
        if permission not in session.permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied. Required permission: {permission.value}"
            )
        
        return current_user
    
    return permission_checker


def require_role(role: UserRole):
    """Dependency factory to require specific role."""
    async def role_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if UserRole(current_user.role) != role:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required role: {role.value}"
            )
        
        return current_user
    
    return role_checker


# Utility functions
def get_user_permissions(role: UserRole) -> List[Permission]:
    """Get permissions for a user role."""
    return ROLE_PERMISSIONS.get(role, [])


async def create_admin_user() -> Optional[User]:
    """Create default admin user if none exists."""
    try:
        user_repo = UserRepository()
        
        # Check if admin exists
        admin_count = await user_repo.count_by_role(UserRole.ADMIN.value)
        if admin_count > 0:
            return None
        
        # Create default admin
        auth_service = get_auth_service()
        admin_password = secrets.token_urlsafe(16)
        
        admin_user = await user_repo.create(
            email="admin@iram.local",
            full_name="IRAM Administrator",
            hashed_password=auth_service.hash_password(admin_password),
            role=UserRole.ADMIN.value
        )
        
        logger.info(
            f"Created default admin user: admin@iram.local with password: {admin_password}",
            extra={"user_id": admin_user.id}
        )
        
        return admin_user
        
    except Exception as e:
        logger.error(f"Failed to create admin user: {e}")
        return None