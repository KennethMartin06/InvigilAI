"""
flag_routes.py — Flag retrieval and acknowledgement endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..auth import get_current_user, require_role
from ..database import get_db
from ..models_db import ExamSession, Flag, User
from ..schemas import FlagResponse

router = APIRouter(prefix="/api/flags", tags=["flags"])


@router.get("/{session_id}", response_model=list[FlagResponse])
async def get_flags_for_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[FlagResponse]:
    """
    Return all flags for a session.

    Access rules:
    - Admin can view any session's flags.
    - Students can only view their own session's flags.
    """
    session = db.query(ExamSession).filter(ExamSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if current_user.role != "admin" and session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    flags = (
        db.query(Flag)
        .filter(Flag.session_id == session_id)
        .order_by(Flag.timestamp.asc())
        .all()
    )
    return [FlagResponse.model_validate(f) for f in flags]


@router.get("/student/{student_id}", response_model=list[FlagResponse])
async def get_flags_for_student(
    student_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[FlagResponse]:
    """
    Return all flags across all sessions for a student.

    Admin-only or the student themselves.
    """
    if current_user.role != "admin" and current_user.id != student_id:
        raise HTTPException(status_code=403, detail="Access denied")

    sessions = (
        db.query(ExamSession).filter(ExamSession.user_id == student_id).all()
    )
    session_ids = [s.id for s in sessions]
    if not session_ids:
        return []

    flags = (
        db.query(Flag)
        .filter(Flag.session_id.in_(session_ids))
        .order_by(Flag.timestamp.desc())
        .limit(100)
        .all()
    )
    return [FlagResponse.model_validate(f) for f in flags]


@router.patch("/{flag_id}/acknowledge", response_model=FlagResponse)
async def acknowledge_flag(
    flag_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
) -> FlagResponse:
    """Mark a flag as reviewed/acknowledged by an admin."""
    flag = db.query(Flag).filter(Flag.id == flag_id).first()
    if flag is None:
        raise HTTPException(status_code=404, detail="Flag not found")

    flag.acknowledged = True
    db.commit()
    db.refresh(flag)
    return FlagResponse.model_validate(flag)
