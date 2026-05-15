"""
Authentication routes: register, login, logout, get/update profile, change password.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field

# Import shared rate limiter
from backend.api.routes._limiter import limiter
from backend.config_loader import settings
from backend.models_peewee import User
from backend.services.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.services.database import run_db_operation
from backend.services.redis_client import get_redis_client
from backend.services.tenant_user_store import get_tenant_user_store
from backend.utils.log_sanitizer import sanitize_key

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Account lockout settings
LOGIN_ATTEMPTS_THRESHOLD = 5
LOGIN_LOCK_DURATION_SECONDS = 15 * 60  # 15 minutes


def _cookie_secure_enabled() -> bool:
    """Enable secure cookies by default outside local/test environments."""
    env = str(getattr(settings, "app_env", "development")).strip().lower()
    return env not in {"dev", "development", "local", "test"}


# ==================== Schemas ====================


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8)
    full_name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UpdateProfileRequest(BaseModel):
    full_name: str | None = None
    email: EmailStr | None = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


class AuthMeResponse(BaseModel):
    user_id: str
    email: str
    username: str
    full_name: str | None
    is_admin: bool
    email_verified: bool
    created_at: str | None
    last_login_at: str | None


# ==================== Helper Functions ====================


def _user_to_response(user: User) -> AuthMeResponse:
    """Convert a User model to the API response format."""
    return AuthMeResponse(
        user_id=user.id,
        email=user.email,
        username=user.username or "",
        full_name=user.full_name,
        is_admin=user.is_admin,
        email_verified=user.email_verified,
        created_at=user.created_at.isoformat() if user.created_at else None,
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )


async def get_current_user_from_token(
    access_token: str | None = Cookie(default=None, alias="access_token"),
) -> User:
    """
    Dependency that extracts and validates JWT from HttpOnly cookie.
    Returns the authenticated User object.
    """
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    payload = decode_access_token(access_token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user_store = get_tenant_user_store()
    user = await run_db_operation(user_store.get_user, payload["user_id"])
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
        )

    return user


# ==================== Routes ====================


@router.post("/register", response_model=dict, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/hour")
async def register(request: Request, data: RegisterRequest, response: Response) -> dict:
    """
    Register a new user account.

    - Checks for duplicate email/username
    - Hashes the password
    - Creates a JWT token and sets it as an HttpOnly cookie
    - Returns basic user info (no password)
    """
    user_store = get_tenant_user_store()

    # Check for existing user with same email or username
    existing_email = await run_db_operation(user_store.get_user_by_email, data.email)
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    if data.username:
        existing_username = await run_db_operation(
            user_store.get_user_by_username, data.username
        )
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken",
            )

    # Hash password
    hashed_password = hash_password(data.password)

    # Create user directly using model
    user = await run_db_operation(
        lambda: User.create_user(
            email=data.email,
            password=hashed_password,
            username=data.username,
            full_name=data.full_name,
            is_superuser=False,
            email_verified=False,
        )
    )

    _log.info("User registered: %s (%s)", user.id, user.email)

    # Create JWT token
    token = create_access_token(
        user_id=user.id, username=user.username or "", is_admin=user.is_admin
    )

    # Set HttpOnly cookie
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=_cookie_secure_enabled(),
        samesite="lax",
        max_age=60 * 60 * 24 * 7,  # 7 days (or use JWT exp)
    )

    return {
        "message": "User registered successfully",
        "user_id": user.id,
        "email": user.email,
        "username": user.username or "",
    }


@router.post("/login", response_model=AuthMeResponse)
@limiter.limit("20/minute")
async def login(
    request: Request, data: LoginRequest, response: Response
) -> AuthMeResponse:
    """
    Authenticate user with email and password.

    - Validates credentials
    - Creates JWT token and sets it as an HttpOnly cookie
    - Returns user profile data
    - Includes account lockout after 5 failed attempts (15 min lock)
    """
    user_store = get_tenant_user_store()
    user = await run_db_operation(user_store.get_user_by_email, data.email)
    if not user:
        # Still record attempt to prevent user enumeration
        await _record_failed_login(request, data.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    # Check if account is locked
    lock_key = f"login_lock:{user.email}"
    redis_client = get_redis_client()
    try:
        is_locked = await redis_client.exists(lock_key)
        if is_locked:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account temporarily locked due to too many failed attempts. Please try again later.",
            )
    except Exception:
        # Redis unavailable - continue without lockout
        pass

    # Verify password
    if not verify_password(data.password, user.password):
        await _record_failed_login(request, user.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    # On successful login, clear failed attempts
    await _clear_failed_logins(request, user.email)

    # Update last login
    now = datetime.utcnow()
    user.last_login_at = now
    user.save()

    _log.info("User logged in: %s (%s)", user.id, user.email)

    # Create JWT token
    token = create_access_token(
        user_id=user.id, username=user.username or "", is_admin=user.is_admin
    )

    # Set HttpOnly cookie
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=_cookie_secure_enabled(),
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
    )

    return _user_to_response(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response) -> None:
    """Log out by clearing the access token cookie."""
    response.delete_cookie(key="access_token")
    _log.info("User logged out")


@router.get("/me", response_model=AuthMeResponse)
async def get_me(user: User = Depends(get_current_user_from_token)) -> AuthMeResponse:
    """Get current authenticated user's profile."""
    return _user_to_response(user)


@router.put("/me", response_model=AuthMeResponse)
async def update_profile(
    data: UpdateProfileRequest,
    user: User = Depends(get_current_user_from_token),
) -> AuthMeResponse:
    """
    Update current user's profile (full name, email).

    - Email change requires re-verification (future)
    - Cannot change username (would require separate flow)
    """
    user_store = get_tenant_user_store()

    # Check email uniqueness if changing email
    if data.email and data.email != user.email:
        existing = await run_db_operation(user_store.get_user_by_email, data.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use",
            )
        user.email = data.email
        user.email_verified = False  # Require re-verification

    if data.full_name is not None:
        user.full_name = data.full_name

    user.save()

    _log.info("Profile updated for user: %s", user.id)

    return _user_to_response(user)


@router.put("/me/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    data: ChangePasswordRequest,
    user: User = Depends(get_current_user_from_token),
) -> None:
    """
    Change user's password.

    Validates:
    - Current password is correct
    - New password is at least 8 characters
    """
    # Verify current password
    if not verify_password(data.current_password, user.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Hash new password
    user.password = hash_password(data.new_password)
    user.save()

    _log.info("Password changed for user: %s", user.id)

    return None


# ==================== Account Lockout Helpers ====================


async def _record_failed_login(request: Request, email: str) -> None:
    """Record a failed login attempt and potentially lock the account."""
    redis_client = get_redis_client()
    client_ip = request.client.host if request.client else "unknown"

    # Increment attempt counter for this email
    attempts_key = f"login_attempts:{email}"
    lock_key = f"login_lock:{email}"

    try:
        client = await redis_client.get_client()

        # Increment and get current count
        count = await client.incr(attempts_key)
        # Set expiry on first attempt (24 hours)
        if count == 1:
            await client.expire(attempts_key, 24 * 60 * 60)

        # Also track by IP for brute force from same IP targeting different accounts
        ip_attempts_key = f"login_attempts_ip:{client_ip}"
        ip_count = await client.incr(ip_attempts_key)
        if ip_count == 1:
            await client.expire(ip_attempts_key, 24 * 60 * 60)

        _log.warning(
            "Failed login attempt #%d for email=%s from IP=%s", count, sanitize_key(email), sanitize_key(client_ip)
        )

        # If threshold reached, lock the account
        if count >= LOGIN_ATTEMPTS_THRESHOLD:
            await client.set(lock_key, "1", ex=LOGIN_LOCK_DURATION_SECONDS)
            _log.warning(
                "Account locked for email=%s due to %d failed attempts (IP=%s)",
                sanitize_key(email),
                count,
                sanitize_key(client_ip),
            )
    except Exception as e:
        _log.warning("Failed to record login attempt: %s", e)
        # Silently continue - Redis failures shouldn't block login


async def _clear_failed_logins(request: Request, email: str) -> None:
    """Clear failed login counters after successful authentication."""
    redis_client = get_redis_client()
    client_ip = request.client.host if request.client else "unknown"

    try:
        client = await redis_client.get_client()
        # Clear email-based counter
        await client.delete(f"login_attempts:{email}")
        await client.delete(f"login_lock:{email}")
        # Also clear IP-based counter
        await client.delete(f"login_attempts_ip:{client_ip}")
    except Exception as e:
        _log.warning("Failed to clear login attempts: %s", e)
