"""
schemas.py — Pydantic request/response schemas for all API endpoints.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator


# ── Auth ──────────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: EmailStr
    name: str
    password: str
    role: str = "student"

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in ("student", "admin"):
            raise ValueError("role must be 'student' or 'admin'")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    name: str
    role: str


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    role: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Exam ──────────────────────────────────────────────────────────────────────

class StartExamRequest(BaseModel):
    exam_title: str


class QuestionResponse(BaseModel):
    id: int
    question_number: int
    question_text: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str

    model_config = {"from_attributes": True}


class StartExamResponse(BaseModel):
    session_id: int
    exam_title: str
    questions: list[QuestionResponse]
    start_time: datetime


class SubmitAnswerRequest(BaseModel):
    session_id: int
    question_id: int
    selected_answer: str

    @field_validator("selected_answer")
    @classmethod
    def answer_must_be_valid(cls, v: str) -> str:
        if v.upper() not in ("A", "B", "C", "D"):
            raise ValueError("selected_answer must be A, B, C, or D")
        return v.upper()


class EndExamResponse(BaseModel):
    session_id: int
    exam_title: str
    duration_seconds: int
    total_questions: int
    answered_questions: int
    total_flags: int
    max_cheating_score: float
    status: str


# ── Flags ─────────────────────────────────────────────────────────────────────

class FlagResponse(BaseModel):
    id: int
    session_id: int
    timestamp: datetime
    cheating_probability: float
    predicted_class: int
    class_name: str
    screenshot_path: Optional[str]
    visual_features: Optional[dict]
    behavioral_features: Optional[dict]
    acknowledged: bool

    model_config = {"from_attributes": True}


# ── Admin ─────────────────────────────────────────────────────────────────────

class SessionResponse(BaseModel):
    id: int
    user_id: int
    student_name: Optional[str] = None
    student_email: Optional[str] = None
    exam_title: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    total_flags: int
    max_cheating_score: float

    model_config = {"from_attributes": True}


class AdminStatsResponse(BaseModel):
    total_sessions: int
    active_sessions: int
    completed_sessions: int
    total_flags_today: int
    total_flags_all_time: int
    avg_cheating_score: float
    high_risk_students: int
    flags_by_class: dict[str, int]


class StudentHistoryResponse(BaseModel):
    user: UserResponse
    sessions: list[SessionResponse]
    total_flags: int
    avg_cheating_score: float


# ── Inference result (internal + WS message) ──────────────────────────────────

class PredictionResult(BaseModel):
    predicted_class: int
    class_name: str
    cheating_probability: float
    class_probabilities: dict[str, float]
    is_cheating: bool
