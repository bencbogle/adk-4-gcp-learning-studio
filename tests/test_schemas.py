from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from adk_4_gcp_learning_studio.schemas import (
    Quiz,
    QuizQuestion,
    QuizQuestionAttempt,
)


def test_quiz_defaults_to_an_active_quiz() -> None:
    """Verify default quiz lifecycle and progress fields."""
    quiz = Quiz(
        id=1,
        user_id="learner-1",
        topic="Cloud Run",
        total_questions=3,
        started_at=datetime.now(UTC),
    )

    assert quiz.status == "active"
    assert quiz.answered_count == 0
    assert quiz.correct_count == 0
    assert quiz.question_types == []
    assert quiz.questions == []


def test_quiz_rejects_a_non_positive_question_count() -> None:
    """Verify that a quiz requires at least one question."""
    with pytest.raises(ValidationError):
        Quiz(
            id=1,
            user_id="learner-1",
            topic="Cloud Run",
            total_questions=0,
            started_at=datetime.now(UTC),
        )


def test_question_and_attempt_link_back_to_a_quiz() -> None:
    """Verify that attempts link to their questions and quiz."""
    question = QuizQuestion(
        id=10,
        quiz_id=1,
        question_number=1,
        prompt="When would you use a Cloud Run job?",
        question_type="application",
        expected_answer_or_rubric="Explain that jobs run tasks to completion rather than serving requests.",
    )
    attempt = QuizQuestionAttempt(
        question_id=question.id,
        learner_answer="For a task that runs to completion.",
        is_correct=True,
        points=1.0,
        feedback="Correct: a job is for finite work.",
        answered_at=datetime.now(UTC),
    )

    assert question.quiz_id == 1
    assert attempt.question_id == question.id


def test_quiz_contains_its_questions() -> None:
    """Verify that a quiz contains its questions."""
    question = QuizQuestion(
        id=10,
        quiz_id=1,
        question_number=1,
        prompt="What is Cloud Run?",
        question_type="recall",
        expected_answer_or_rubric="A managed platform for running containers.",
    )
    quiz = Quiz(
        id=1,
        user_id="learner-1",
        topic="Cloud Run",
        total_questions=1,
        started_at=datetime.now(UTC),
        questions=[question],
    )

    assert quiz.questions[0] == question
