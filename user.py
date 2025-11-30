from fastapi_users.db import SQLAlchemyBaseUserTable, SQLAlchemyUserDatabase
from fastapi_users import BaseUserManager, IntegerIDMixin
from fastapi import Depends, Request, HTTPException
from models import Base
from sqlalchemy import String, Column, Integer, Boolean, Text
from db import engine, SessionLocal
from fastapi_users import schemas
from fastapi_users.password import PasswordHelper
import os
import jwt
from typing import Optional

class User(SQLAlchemyBaseUserTable[int], Base):
    """
    User model with explicit field definitions
    """
    __tablename__ = "users"
    
    # Explicitly define all fields including the primary key
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Custom fields
    oauth_provider = Column(String, nullable=True)
    oauth_id = Column(String, nullable=True)
    # openai_api_key = Column(Text, nullable=True)
    tier = Column(String, default="free", nullable=False)
    stripe_customer_id = Column(String, nullable=True)

class UserRead(schemas.BaseUser[int]):
    oauth_provider: str | None = None
    oauth_id: str | None = None
    tier: str | None = None
    stripe_customer_id: str | None = None
    # openai_api_key: str | None = None

class UserCreate(schemas.BaseUserCreate):
    oauth_provider: str | None = None
    oauth_id: str | None = None
    tier: str | None = None
    stripe_customer_id: str | None = None
    # openai_api_key: str | None = None

class UserUpdate(schemas.BaseUserUpdate):
    oauth_provider: str | None = None
    oauth_id: str | None = None
    tier: str | None = None
    stripe_customer_id: str | None = None
    # openai_api_key: str | None = None

def init_db():
    Base.metadata.create_all(bind=engine)

# Custom sync user dependency that properly reads JWT token
def get_current_user_sync(request: Request) -> User:
    """
    Custom sync user dependency that reads JWT token and validates user
    """
    # Get token from cookie
    token = request.cookies.get("volatilx_cookie")
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        # Decode JWT token
        SECRET = os.getenv("JWT_SECRET", "SECRET")
        
        # Decode the token with proper options to handle FastAPI Users JWT format
        payload = jwt.decode(
            token, 
            SECRET, 
            algorithms=["HS256"],
            options={
                "verify_aud": False,  # Disable audience verification
                "verify_iss": False,  # Disable issuer verification
            }
        )
        
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get user from database
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == int(user_id)).first()
            if user is None:
                raise HTTPException(status_code=401, detail="User not found")
            
            if not user.is_active:
                raise HTTPException(status_code=401, detail="User inactive")
            
            return user
            
        finally:
            db.close()
            
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed")


def get_optional_user(request: Request) -> Optional[User]:
    """Return the current user when authenticated, otherwise None."""

    try:
        return get_current_user_sync(request)
    except HTTPException:
        return None

# Keep these for FastAPI Users routers (they won't be used for /analyze)
async def get_user_db():
    db = SessionLocal()
    try:
        yield SQLAlchemyUserDatabase(db, User)
    finally:
        db.close()

class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    reset_password_token_secret = os.getenv("RESET_PASSWORD_SECRET", "SECRET")
    verification_token_secret = os.getenv("VERIFICATION_SECRET", "SECRET")

async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    yield UserManager(user_db)