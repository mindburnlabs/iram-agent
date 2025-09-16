"""
Authentication Routes for IRAM

FastAPI routes for user authentication, registration, and management.
"""

from datetime import timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.security import HTTPAuthorizationCredentials

from .auth import (
    get_auth_service, get_current_user, get_current_active_user,
    require_permission, require_role,
    AuthService, UserLogin, UserRegister, UserProfile, TokenResponse,
    APIKeyCreate, APIKeyInfo, SessionInfo, UserRole, Permission
)
from .models import User, APIKey
from .repository import UserRepository, APIKeyRepository
from .logging_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(
    login_data: UserLogin,
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Authenticate user and return access token."""
    # Authenticate user
    user = await auth_service.authenticate_user(login_data.email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    
    # Create session
    session = await auth_service.create_session(user, request, login_data.remember_me)
    
    # Create access token
    expires_delta = timedelta(days=30) if login_data.remember_me else None
    access_token = auth_service.create_access_token(
        user_id=user.id,
        email=user.email,
        role=UserRole(user.role),
        expires_delta=expires_delta
    )
    
    # Create user profile response
    user_profile = UserProfile(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=UserRole(user.role),
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login
    )
    
    expires_in = (30 * 24 * 3600) if login_data.remember_me else (30 * 60)
    
    logger.info(f"User logged in: {user.email}", extra={
        "user_id": user.id,
        "remember_me": login_data.remember_me
    })
    
    return TokenResponse(
        access_token=access_token,
        expires_in=expires_in,
        user=user_profile,
        permissions=session.permissions
    )


@router.post("/register", response_model=UserProfile)
async def register(
    register_data: UserRegister,
    auth_service: AuthService = Depends(get_auth_service)
):
    """Register a new user."""
    user_repo = UserRepository()
    
    # Check if user already exists
    existing_user = await user_repo.get_by_email(register_data.email)
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Hash password
    hashed_password = auth_service.hash_password(register_data.password)
    
    # Create user
    user = await user_repo.create(
        email=register_data.email,
        full_name=register_data.full_name,
        hashed_password=hashed_password,
        role=register_data.role.value
    )
    
    logger.info(f"New user registered: {user.email}", extra={
        "user_id": user.id,
        "role": user.role
    })
    
    return UserProfile(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=UserRole(user.role),
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login
    )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Logout user and invalidate session."""
    await auth_service.invalidate_session(current_user.id)
    
    logger.info(f"User logged out: {current_user.email}", extra={
        "user_id": current_user.id
    })
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user profile."""
    return UserProfile(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=UserRole(current_user.role),
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@router.get("/session", response_model=SessionInfo)
async def get_session_info(
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Get current session information."""
    session = await auth_service.get_session(current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@router.post("/refresh")
async def refresh_token(
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Refresh authentication token."""
    session = await auth_service.refresh_session(current_user.id, extend_ttl=True)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Create new access token
    access_token = auth_service.create_access_token(
        user_id=current_user.id,
        email=current_user.email,
        role=UserRole(current_user.role)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 30 * 60  # 30 minutes
    }


# API Key Management Routes

@router.post("/api-keys", response_model=dict)
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """Create a new API key."""
    api_key_repo = APIKeyRepository()
    
    # Generate API key
    api_key = auth_service.generate_api_key()
    key_hash = auth_service.hash_api_key(api_key)
    
    # Create API key record
    api_key_obj = await api_key_repo.create(
        user_id=current_user.id,
        name=api_key_data.name,
        description=api_key_data.description,
        key_hash=key_hash,
        permissions=[p.value for p in api_key_data.permissions],
        expires_at=api_key_data.expires_at
    )
    
    logger.info(f"API key created: {api_key_data.name}", extra={
        "user_id": current_user.id,
        "api_key_id": api_key_obj.id
    })
    
    return {
        "api_key": api_key,  # Return the actual key only once
        "id": api_key_obj.id,
        "name": api_key_obj.name,
        "message": "API key created successfully. Save this key securely - it won't be shown again."
    }


@router.get("/api-keys", response_model=List[APIKeyInfo])
async def list_api_keys(
    current_user: User = Depends(get_current_active_user)
):
    """List user's API keys (without the actual keys)."""
    api_key_repo = APIKeyRepository()
    api_keys = await api_key_repo.get_by_user_id(current_user.id)
    
    return [
        APIKeyInfo(
            id=key.id,
            name=key.name,
            description=key.description,
            permissions=[Permission(p) for p in key.permissions],
            created_at=key.created_at,
            expires_at=key.expires_at,
            last_used=key.last_used,
            is_active=key.is_active
        )
        for key in api_keys
    ]


@router.delete("/api-keys/{api_key_id}")
async def delete_api_key(
    api_key_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """Delete an API key."""
    api_key_repo = APIKeyRepository()
    
    # Get API key
    api_key = await api_key_repo.get_by_id(api_key_id)
    if not api_key or api_key.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Delete API key
    await api_key_repo.delete(api_key_id)
    
    logger.info(f"API key deleted: {api_key.name}", extra={
        "user_id": current_user.id,
        "api_key_id": api_key_id
    })
    
    return {"message": "API key deleted successfully"}


@router.put("/api-keys/{api_key_id}/toggle")
async def toggle_api_key(
    api_key_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """Enable/disable an API key."""
    api_key_repo = APIKeyRepository()
    
    # Get API key
    api_key = await api_key_repo.get_by_id(api_key_id)
    if not api_key or api_key.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Toggle active status
    new_status = not api_key.is_active
    await api_key_repo.update_active_status(api_key_id, new_status)
    
    logger.info(f"API key {'enabled' if new_status else 'disabled'}: {api_key.name}", extra={
        "user_id": current_user.id,
        "api_key_id": api_key_id
    })
    
    return {
        "message": f"API key {'enabled' if new_status else 'disabled'}",
        "is_active": new_status
    }


# Admin-only routes

@router.get("/users", response_model=List[UserProfile])
async def list_users(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """List all users (admin only)."""
    user_repo = UserRepository()
    users = await user_repo.get_all(limit=limit, offset=offset)
    
    return [
        UserProfile(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=UserRole(user.role),
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
        for user in users
    ]


@router.get("/users/{user_id}", response_model=UserProfile)
async def get_user(
    user_id: int,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Get user by ID (admin only)."""
    user_repo = UserRepository()
    user = await user_repo.get_by_id(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserProfile(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=UserRole(user.role),
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login
    )


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    role: UserRole,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Update user role (admin only)."""
    user_repo = UserRepository()
    
    # Get user
    user = await user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Don't allow changing own role
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot change your own role")
    
    # Update role
    await user_repo.update_role(user_id, role.value)
    
    logger.info(f"User role updated: {user.email} -> {role.value}", extra={
        "user_id": user_id,
        "updated_by": current_user.id,
        "old_role": user.role,
        "new_role": role.value
    })
    
    return {"message": f"User role updated to {role.value}"}


@router.put("/users/{user_id}/status")
async def toggle_user_status(
    user_id: int,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Enable/disable user (admin only)."""
    user_repo = UserRepository()
    
    # Get user
    user = await user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Don't allow disabling own account
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot disable your own account")
    
    # Toggle active status
    new_status = not user.is_active
    await user_repo.update_active_status(user_id, new_status)
    
    # If disabling user, invalidate their session
    if not new_status:
        auth_service = get_auth_service()
        await auth_service.invalidate_session(user_id)
    
    logger.info(f"User {'enabled' if new_status else 'disabled'}: {user.email}", extra={
        "user_id": user_id,
        "updated_by": current_user.id,
        "new_status": new_status
    })
    
    return {
        "message": f"User {'enabled' if new_status else 'disabled'}",
        "is_active": new_status
    }