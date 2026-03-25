"""
models_db.py — SQLAlchemy ORM models for users, sessions, flags, and questions.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, ForeignKey, Text, JSON,
)
from sqlalchemy.orm import relationship
from .database import Base


class User(Base):
    """Represents a registered user (student or admin)."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="student")  # "student" | "admin"
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("ExamSession", back_populates="user", cascade="all, delete-orphan")


class ExamSession(Base):
    """Tracks a single exam sitting by a student."""

    __tablename__ = "exam_sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exam_title = Column(String(255), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String(20), default="active")  # "active" | "completed" | "terminated"
    total_flags = Column(Integer, default=0)
    max_cheating_score = Column(Float, default=0.0)

    user = relationship("User", back_populates="sessions")
    flags = relationship("Flag", back_populates="session", cascade="all, delete-orphan")
    answers = relationship("StudentAnswer", back_populates="session", cascade="all, delete-orphan")


class Flag(Base):
    """A cheating event detected during an exam session."""

    __tablename__ = "flags"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("exam_sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    cheating_probability = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False)
    class_name = Column(String(100), nullable=False)
    screenshot_path = Column(String(512), nullable=True)
    visual_features = Column(JSON, nullable=True)
    behavioral_features = Column(JSON, nullable=True)
    acknowledged = Column(Boolean, default=False)

    session = relationship("ExamSession", back_populates="flags")


class ExamQuestion(Base):
    """A multiple-choice question belonging to a named exam."""

    __tablename__ = "exam_questions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    exam_title = Column(String(255), nullable=False, index=True)
    question_number = Column(Integer, nullable=False)
    question_text = Column(Text, nullable=False)
    option_a = Column(String(512), nullable=False)
    option_b = Column(String(512), nullable=False)
    option_c = Column(String(512), nullable=False)
    option_d = Column(String(512), nullable=False)
    correct_answer = Column(String(1), nullable=False)  # "A" | "B" | "C" | "D"

    answers = relationship("StudentAnswer", back_populates="question")


class StudentAnswer(Base):
    """A student's answer to a single exam question."""

    __tablename__ = "student_answers"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("exam_sessions.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("exam_questions.id"), nullable=False)
    selected_answer = Column(String(1), nullable=True)
    answered_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ExamSession", back_populates="answers")
    question = relationship("ExamQuestion", back_populates="answers")
