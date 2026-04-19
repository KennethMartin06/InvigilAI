"""
seed_demo.py — Create demo accounts and sample session data.

Called on startup in app.py when SEED_DEMO_DATA=true.
Idempotent — safe to run on every startup (skips if already seeded).
"""

import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

DEMO_ADMIN    = {"email": "admin@proctor.ai",    "name": "Demo Admin",    "password": "admin123",    "role": "admin"}
DEMO_STUDENTS = [
    {"email": "student1@proctor.ai", "name": "Alice Johnson",  "password": "student123", "role": "student"},
    {"email": "student2@proctor.ai", "name": "Bob Smith",      "password": "student123", "role": "student"},
    {"email": "student3@proctor.ai", "name": "Carol Williams", "password": "student123", "role": "student"},
]

SAMPLE_QUESTIONS = [
    {
        "exam_title": "Demo Exam",
        "question_number": 1,
        "question_text": "What is the time complexity of binary search?",
        "option_a": "O(n)", "option_b": "O(log n)", "option_c": "O(n²)", "option_d": "O(1)",
        "correct_answer": "B",
    },
    {
        "exam_title": "Demo Exam",
        "question_number": 2,
        "question_text": "Which data structure uses LIFO ordering?",
        "option_a": "Queue", "option_b": "Heap", "option_c": "Stack", "option_d": "Tree",
        "correct_answer": "C",
    },
    {
        "exam_title": "Demo Exam",
        "question_number": 3,
        "question_text": "What does CPU stand for?",
        "option_a": "Central Processing Unit", "option_b": "Computer Personal Unit",
        "option_c": "Core Program Utility", "option_d": "Central Program Uploader",
        "correct_answer": "A",
    },
]


def seed(db_session) -> None:
    """Seed demo users, questions, and sample sessions. Idempotent."""
    from .models_db import User, ExamSession, ExamQuestion, Flag
    from .auth import hash_password

    # ── Demo users ─────────────────────────────────────────────────────────────
    for user_data in [DEMO_ADMIN, *DEMO_STUDENTS]:
        existing = db_session.query(User).filter_by(email=user_data["email"]).first()
        if not existing:
            user = User(
                email=user_data["email"],
                name=user_data["name"],
                password_hash=hash_password(user_data["password"]),
                role=user_data["role"],
            )
            db_session.add(user)
            logger.info("[Seed] Created user %s (%s)", user_data["email"], user_data["role"])

    db_session.flush()

    # ── Sample questions ───────────────────────────────────────────────────────
    for q in SAMPLE_QUESTIONS:
        exists = db_session.query(ExamQuestion).filter_by(
            exam_title=q["exam_title"], question_number=q["question_number"]
        ).first()
        if not exists:
            db_session.add(ExamQuestion(**q))

    db_session.flush()

    # ── Sample completed sessions with flags ───────────────────────────────────
    students = db_session.query(User).filter_by(role="student").all()
    for student in students:
        # Skip if this student already has sessions
        has_sessions = db_session.query(ExamSession).filter_by(user_id=student.id).first()
        if has_sessions:
            continue

        start = datetime.utcnow() - timedelta(hours=random.randint(1, 48))
        n_flags = random.randint(0, 5)
        max_score = round(random.uniform(0.1, 0.9), 3) if n_flags > 0 else round(random.uniform(0.0, 0.15), 3)

        session = ExamSession(
            user_id=student.id,
            exam_title="Demo Exam",
            start_time=start,
            end_time=start + timedelta(minutes=random.randint(20, 45)),
            status="completed",
            total_flags=n_flags,
            max_cheating_score=max_score,
        )
        db_session.add(session)
        db_session.flush()

        # Add some flags for realism
        class_choices = [1, 2, 3, 4]  # skip "Normal"
        class_names   = ["Gaze/Distraction", "External Device", "Multi-Person", "Abnormal Keystroke"]
        for i in range(n_flags):
            cls = random.choice(range(len(class_choices)))
            prob = round(random.uniform(0.72, 0.97), 3)
            flag = Flag(
                session_id=session.id,
                timestamp=start + timedelta(minutes=random.randint(2, 40)),
                cheating_probability=prob,
                predicted_class=class_choices[cls],
                class_name=class_names[cls],
                visual_features={},
                behavioral_features={},
                acknowledged=False,
            )
            db_session.add(flag)

        logger.info(
            "[Seed] Created session for %s (%d flags, max_score=%.2f)",
            student.email, n_flags, max_score,
        )

    db_session.commit()
    logger.info("[Seed] Demo data seeding complete ✓")
