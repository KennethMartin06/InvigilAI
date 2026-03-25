"""
auth.py — JWT token creation/verification and password hashing utilities.
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db
from .models_db import User

# ── Password hashing ──────────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def hash_password(password: str) -> str:
    """
    Hash a plain-text password using bcrypt.

    Parameters
    ----------
    password : str

    Returns
    -------
    str — bcrypt hash.
    """
    return _pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a plain-text password against a stored bcrypt hash.

    Parameters
    ----------
    plain : str — password provided by user.
    hashed : str — stored hash from database.

    Returns
    -------
    bool
    """
    return _pwd_context.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a signed JWT access token.

    Parameters
    ----------
    data : dict — payload to encode (typically {"sub": str(user_id), "role": role}).
    expires_delta : timedelta — custom expiry; defaults to settings.jwt_expiry_hours.

    Returns
    -------
    str — encoded JWT string.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(hours=settings.jwt_expiry_hours)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    """
    Decode and verify a JWT token.

    Parameters
    ----------
    token : str

    Returns
    -------
    dict — decoded payload.

    Raises
    ------
    HTTPException 401 if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependencies ──────────────────────────────────────────────────────

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency that resolves the current authenticated user.

    Parameters
    ----------
    token : str — JWT from Authorization header.
    db : Session

    Returns
    -------
    User ORM instance.

    Raises
    ------
    HTTPException 401 if token invalid or user not found.
    """
    payload = decode_token(token)
    user_id: Optional[str] = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_role(role: str):
    """
    Dependency factory that enforces a specific user role.

    Parameters
    ----------
    role : str — required role ("admin" or "student").

    Returns
    -------
    Callable — FastAPI dependency.
    """
    def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access restricted to {role} accounts",
            )
        return current_user
    return _check


def get_user_from_token_string(token: str, db: Session) -> Optional[User]:
    """
    Resolve a user from a raw token string (used by WebSocket handler).

    Parameters
    ----------
    token : str — raw JWT (not via Authorization header).
    db : Session

    Returns
    -------
    User or None if invalid.
    """
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        if user_id is None:
            return None
        return db.query(User).filter(User.id == int(user_id)).first()
    except Exception:
        return None
