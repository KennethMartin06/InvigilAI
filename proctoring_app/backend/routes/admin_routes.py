"""
admin_routes.py — Admin-only dashboard and monitoring endpoints.
"""

from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..auth import require_role
from ..database import get_db
from ..models_db import ExamSession, Flag, User
from ..schemas import AdminStatsResponse, SessionResponse, StudentHistoryResponse, UserResponse

router = APIRouter(prefix="/api/admin", tags=["admin"])

_admin_required = require_role("admin")


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    status: Optional[str] = Query(None),
    student_id: Optional[int] = Query(None),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(_admin_required),
) -> list[SessionResponse]:
    """
    Return all exam sessions with optional filters.

    Query parameters
    ----------------
    status : str — filter by "active", "completed", or "terminated".
    student_id : int — filter by user id.
    from_date, to_date : date — filter by start_time range.
    """
    q = db.query(ExamSession)
    if status:
        q = q.filter(ExamSession.status == status)
    if student_id:
        q = q.filter(ExamSession.user_id == student_id)
    if from_date:
        q = q.filter(ExamSession.start_time >= datetime.combine(from_date, datetime.min.time()))
    if to_date:
        q = q.filter(ExamSession.start_time <= datetime.combine(to_date, datetime.max.time()))

    sessions = q.order_by(ExamSession.start_time.desc()).limit(200).all()

    result = []
    for s in sessions:
        user = db.query(User).filter(User.id == s.user_id).first()
        sr = SessionResponse(
            id=s.id,
            user_id=s.user_id,
            student_name=user.name if user else None,
            student_email=user.email if user else None,
            exam_title=s.exam_title,
            start_time=s.start_time,
            end_time=s.end_time,
            status=s.status,
            total_flags=s.total_flags or 0,
            max_cheating_score=s.max_cheating_score or 0.0,
        )
        result.append(sr)
    return result


@router.get("/sessions/active", response_model=list[SessionResponse])
async def active_sessions(
    db: Session = Depends(get_db),
    _: User = Depends(_admin_required),
) -> list[SessionResponse]:
    """Return all currently live (active) exam sessions."""
    sessions = (
        db.query(ExamSession)
        .filter(ExamSession.status == "active")
        .order_by(ExamSession.start_time.desc())
        .all()
    )
    result = []
    for s in sessions:
        user = db.query(User).filter(User.id == s.user_id).first()
        result.append(SessionResponse(
            id=s.id,
            user_id=s.user_id,
            student_name=user.name if user else None,
            student_email=user.email if user else None,
            exam_title=s.exam_title,
            start_time=s.start_time,
            end_time=s.end_time,
            status=s.status,
            total_flags=s.total_flags or 0,
            max_cheating_score=s.max_cheating_score or 0.0,
        ))
    return result


@router.get("/stats", response_model=AdminStatsResponse)
async def get_stats(
    db: Session = Depends(get_db),
    _: User = Depends(_admin_required),
) -> AdminStatsResponse:
    """
    Return aggregate statistics for the admin dashboard overview cards.
    """
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    total_sessions = db.query(func.count(ExamSession.id)).scalar() or 0
    active_sessions = (
        db.query(func.count(ExamSession.id))
        .filter(ExamSession.status == "active")
        .scalar() or 0
    )
    completed_sessions = (
        db.query(func.count(ExamSession.id))
        .filter(ExamSession.status == "completed")
        .scalar() or 0
    )
    flags_today = (
        db.query(func.count(Flag.id))
        .filter(Flag.timestamp >= today_start)
        .scalar() or 0
    )
    flags_all = db.query(func.count(Flag.id)).scalar() or 0
    avg_score = (
        db.query(func.avg(ExamSession.max_cheating_score)).scalar() or 0.0
    )
    high_risk = (
        db.query(func.count(ExamSession.user_id.distinct()))
        .filter(ExamSession.max_cheating_score >= 0.70)
        .scalar() or 0
    )

    # Flags by class name
    rows = db.query(Flag.class_name, func.count(Flag.id)).group_by(Flag.class_name).all()
    flags_by_class = {row[0]: row[1] for row in rows}

    return AdminStatsResponse(
        total_sessions=total_sessions,
        active_sessions=active_sessions,
        completed_sessions=completed_sessions,
        total_flags_today=flags_today,
        total_flags_all_time=flags_all,
        avg_cheating_score=round(float(avg_score), 4),
        high_risk_students=high_risk,
        flags_by_class=flags_by_class,
    )


@router.get("/student/{student_id}/history", response_model=StudentHistoryResponse)
async def student_history(
    student_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(_admin_required),
) -> StudentHistoryResponse:
    """Return all sessions and flags for a specific student."""
    from ..schemas import SessionResponse as SR
    user = db.query(User).filter(User.id == student_id).first()
    if user is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Student not found")

    sessions = (
        db.query(ExamSession)
        .filter(ExamSession.user_id == student_id)
        .order_by(ExamSession.start_time.desc())
        .all()
    )
    total_flags = sum(s.total_flags or 0 for s in sessions)
    scores = [s.max_cheating_score or 0.0 for s in sessions if s.max_cheating_score]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return StudentHistoryResponse(
        user=UserResponse.model_validate(user),
        sessions=[
            SR(
                id=s.id,
                user_id=s.user_id,
                student_name=user.name,
                student_email=user.email,
                exam_title=s.exam_title,
                start_time=s.start_time,
                end_time=s.end_time,
                status=s.status,
                total_flags=s.total_flags or 0,
                max_cheating_score=s.max_cheating_score or 0.0,
            )
            for s in sessions
        ],
        total_flags=total_flags,
        avg_cheating_score=round(avg_score, 4),
    )
