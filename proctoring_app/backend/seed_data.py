"""
seed_data.py — Populate the database with demo users and exam questions.

Run from the backend/ directory:
    python seed_data.py

Creates:
  - 1 admin account: admin@proctor.ai / admin123
  - 3 student accounts: student{1,2,3}@proctor.ai / student123
  - 10 General Knowledge exam questions
"""

import sys
import os

# Allow running as a script from inside backend/ OR from repo root
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
for _p in [_root, os.path.dirname(_here)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from proctoring_app.backend.database import SessionLocal, create_tables
    from proctoring_app.backend.models_db import ExamQuestion, User
    from proctoring_app.backend.auth import hash_password
except ModuleNotFoundError:
    from database import SessionLocal, create_tables
    from models_db import ExamQuestion, User
    from auth import hash_password


QUESTIONS = [
    {
        "question_number": 1,
        "question_text": "What is the capital city of France?",
        "option_a": "Berlin",
        "option_b": "Madrid",
        "option_c": "Paris",
        "option_d": "Rome",
        "correct_answer": "C",
    },
    {
        "question_number": 2,
        "question_text": "Which element has the chemical symbol 'O'?",
        "option_a": "Gold",
        "option_b": "Oxygen",
        "option_c": "Osmium",
        "option_d": "Oganesson",
        "correct_answer": "B",
    },
    {
        "question_number": 3,
        "question_text": "What is the largest planet in our solar system?",
        "option_a": "Saturn",
        "option_b": "Uranus",
        "option_c": "Neptune",
        "option_d": "Jupiter",
        "correct_answer": "D",
    },
    {
        "question_number": 4,
        "question_text": "In what year did World War II end?",
        "option_a": "1943",
        "option_b": "1944",
        "option_c": "1945",
        "option_d": "1946",
        "correct_answer": "C",
    },
    {
        "question_number": 5,
        "question_text": "Who wrote the play 'Romeo and Juliet'?",
        "option_a": "Charles Dickens",
        "option_b": "William Shakespeare",
        "option_c": "Jane Austen",
        "option_d": "Mark Twain",
        "correct_answer": "B",
    },
    {
        "question_number": 6,
        "question_text": "What is the speed of light in a vacuum (approx.)?",
        "option_a": "150,000 km/s",
        "option_b": "200,000 km/s",
        "option_c": "300,000 km/s",
        "option_d": "400,000 km/s",
        "correct_answer": "C",
    },
    {
        "question_number": 7,
        "question_text": "Which country is the largest by land area?",
        "option_a": "Canada",
        "option_b": "China",
        "option_c": "United States",
        "option_d": "Russia",
        "correct_answer": "D",
    },
    {
        "question_number": 8,
        "question_text": "What is the powerhouse of the cell?",
        "option_a": "Nucleus",
        "option_b": "Ribosome",
        "option_c": "Mitochondria",
        "option_d": "Golgi apparatus",
        "correct_answer": "C",
    },
    {
        "question_number": 9,
        "question_text": "How many sides does a hexagon have?",
        "option_a": "5",
        "option_b": "6",
        "option_c": "7",
        "option_d": "8",
        "correct_answer": "B",
    },
    {
        "question_number": 10,
        "question_text": "Which ocean is the largest?",
        "option_a": "Atlantic Ocean",
        "option_b": "Indian Ocean",
        "option_c": "Arctic Ocean",
        "option_d": "Pacific Ocean",
        "correct_answer": "D",
    },
]

USERS = [
    {"email": "admin@proctor.ai",    "name": "Admin User",    "role": "admin",   "password": "admin123"},
    {"email": "student1@proctor.ai", "name": "Alice Johnson", "role": "student", "password": "student123"},
    {"email": "student2@proctor.ai", "name": "Bob Williams",  "role": "student", "password": "student123"},
    {"email": "student3@proctor.ai", "name": "Carol Smith",   "role": "student", "password": "student123"},
]


def seed() -> None:
    """Insert demo data if it doesn't already exist."""
    create_tables()
    db = SessionLocal()

    try:
        # Users
        for u in USERS:
            if not db.query(User).filter(User.email == u["email"]).first():
                user = User(
                    email=u["email"],
                    name=u["name"],
                    role=u["role"],
                    password_hash=hash_password(u["password"]),
                )
                db.add(user)
                print(f"  Created user: {u['email']}  ({u['role']})")
            else:
                print(f"  Skipped (exists): {u['email']}")

        db.commit()

        # Questions
        exam_title = "General Knowledge"
        existing_count = (
            db.query(ExamQuestion)
            .filter(ExamQuestion.exam_title == exam_title)
            .count()
        )
        if existing_count == 0:
            for q in QUESTIONS:
                question = ExamQuestion(exam_title=exam_title, **q)
                db.add(question)
            db.commit()
            print(f"  Inserted {len(QUESTIONS)} questions for '{exam_title}'")
        else:
            print(f"  Skipped questions (already {existing_count} exist for '{exam_title}')")

        print("\nSeed complete.")
        print("─" * 40)
        print("  Admin:    admin@proctor.ai  /  admin123")
        print("  Students: student{1,2,3}@proctor.ai  /  student123")
        print("  Exam:    'General Knowledge'  (10 questions)")
        print("─" * 40)

    finally:
        db.close()


if __name__ == "__main__":
    seed()
