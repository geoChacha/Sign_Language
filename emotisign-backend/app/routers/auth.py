"""
Auth Router — /api/auth
Endpoints: register, login, logout, me, update-profile
"""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models import User
from app.schemas import (
    RegisterRequest, TokenResponse,
    UserResponse, UserUpdateRequest, SuccessResponse
)
from app.services.auth_service import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ─────────────────────────── Register ───────────────────────────

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.username == payload.username))
    if res.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already taken")

    res = await db.execute(select(User).where(User.email == payload.email))
    if res.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role=payload.role,
        preferred_sign_language=payload.preferred_sign_language,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


# ─────────────────────────── Login (form-data, Swagger compatible) ───────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    # Support login with either username or email
    from sqlalchemy import or_
    result = await db.execute(
        select(User).where(
            or_(User.username == username, User.email == username)
        )
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username/email or password"
        )

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    expire_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token({"sub": str(user.id)}, expires_delta=expire_delta)

    return TokenResponse(
        access_token=token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.model_validate(user)
    )


# ─────────────────────────── Me ───────────────────────────

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user


# ─────────────────────────── Search Users ───────────────────────────

@router.get("/users", response_model=list[UserResponse])
async def search_users(
    q: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Search users by username or email (excludes current user)"""
    query = select(User).where(User.id != current_user.id, User.is_active == True)
    
    if q:
        search_term = f"%{q}%"
        query = query.where(
            (User.username.ilike(search_term)) | (User.email.ilike(search_term))
        )
    
    query = query.limit(20)
    result = await db.execute(query)
    return result.scalars().all()


# ─────────────────────────── Update Profile ───────────────────────────

@router.patch("/me", response_model=UserResponse)
async def update_profile(
    payload: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if payload.full_name is not None:
        current_user.full_name = payload.full_name
    if payload.preferred_sign_language is not None:
        current_user.preferred_sign_language = payload.preferred_sign_language
    if payload.profile_picture_url is not None:
        current_user.profile_picture_url = payload.profile_picture_url

    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    return current_user


# ─────────────────────────── Logout ───────────────────────────

@router.post("/logout", response_model=SuccessResponse)
async def logout(current_user: User = Depends(get_current_user)):
    return SuccessResponse(message="Logged out successfully")
