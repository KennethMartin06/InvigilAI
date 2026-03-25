"""
exam_routes.py — Student exam management endpoints.
"""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models_db import ExamQuestion, ExamSession, StudentAnswer, User
from ..schemas import (
    EndExamResponse,
    QuestionResponse,
    StartExamRequest,
    StartExamResponse,
    SubmitAnswerRequest,
)

router = APIRouter(prefix="/api/exam", tags=["exam"])


@router.post("/start", response_model=StartExamResponse, status_code=status.HTTP_201_CREATED)
async def start_exam(
    body: StartExamRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StartExamResponse:
    """
    Create a new ExamSession for the authenticated student and return
    the session ID together with all questions for the exam.

    Raises 404 if no questions exist for the given exam_title.
    """
    questions = (
        db.query(ExamQuestion)
        .filter(ExamQuestion.exam_title == body.exam_title)
        .order_by(ExamQuestion.question_number)
        .all()
    )
    if not questions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No questions found for exam: {body.exam_title}",
        )

    session = ExamSession(
        user_id=current_user.id,
        exam_title=body.exam_title,
        start_time=datetime.utcnow(),
        status="active",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return StartExamResponse(
        session_id=session.id,
        exam_title=session.exam_title,
        questions=[QuestionResponse.model_validate(q) for q in questions],
        start_time=session.start_time,
    )


@router.post("/answer", status_code=status.HTTP_200_OK)
async def submit_answer(
    body: SubmitAnswerRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Record or update a student's answer for a single question.

    Upserts: if an answer already exists for this (session, question) pair
    it is overwritten.
    """
    session = db.query(ExamSession).filter(
        ExamSession.id == body.session_id,
        ExamSession.user_id == current_user.id,
    ).first()
    if session is None or session.status != "active":
        raise HTTPException(status_code=404, detail="Active session not found")

    existing = db.query(StudentAnswer).filter(
        StudentAnswer.session_id == body.session_id,
        StudentAnswer.question_id == body.question_id,
    ).first()

    if existing:
        existing.selected_answer = body.selected_answer
        existing.answered_at = datetime.utcnow()
    else:
        answer = StudentAnswer(
            session_id=body.session_id,
            question_id=body.question_id,
            selected_answer=body.selected_answer,
        )
        db.add(answer)

    db.commit()
    return {"status": "saved"}


@router.post("/end/{session_id}", response_model=EndExamResponse)
async def end_exam(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EndExamResponse:
    """
    Mark an exam session as completed and return a summary.

    Idempotent: calling on an already-completed session returns the summary.
    """
    session = db.query(ExamSession).filter(
        ExamSession.id == session_id,
        ExamSession.user_id == current_user.id,
    ).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status == "active":
        session.status = "completed"
        session.end_time = datetime.utcnow()
        db.commit()
        db.refresh(session)

    total_questions = (
        db.query(ExamQuestion)
        .filter(ExamQuestion.exam_title == session.exam_title)
        .count()
    )
    answered = (
        db.query(StudentAnswer)
        .filter(StudentAnswer.session_id == session_id)
        .count()
    )
    duration = int(
        (session.end_time - session.start_time).total_seconds()
    ) if session.end_time else 0

    return EndExamResponse(
        session_id=session.id,
        exam_title=session.exam_title,
        duration_seconds=duration,
        total_questions=total_questions,
        answered_questions=answered,
        total_flags=session.total_flags or 0,
        max_cheating_score=session.max_cheating_score or 0.0,
        status=session.status,
    )


@router.get("/questions/{exam_title}", response_model=list[QuestionResponse])
async def get_questions(
    exam_title: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[QuestionResponse]:
    """Return all questions for a named exam in order."""
    questions = (
        db.query(ExamQuestion)
        .filter(ExamQuestion.exam_title == exam_title)
        .order_by(ExamQuestion.question_number)
        .all()
    )
    return [QuestionResponse.model_validate(q) for q in questions]
