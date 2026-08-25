from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class SourceChunk(BaseModel):
    chunk_id: str
    product: str
    title: str
    url: HttpUrl
    text: str
    indexed_at: datetime
    relevance_score: float | None = None


QuizStatus = Literal["active", "complete", "abandoned"]
Difficulty = Literal["introductory", "intermediate", "advanced"]
QuestionType = Literal["recall", "comparison", "prediction", "application"]


class QuizQuestionAttempt(BaseModel):
    question_id: int
    learner_answer: str
    is_correct: bool
    points: float = Field(default=0.0, ge=0.0)
    feedback: str
    misconception: str | None = None
    answered_at: datetime


class QuizQuestion(BaseModel):
    id: int
    quiz_id: int
    question_number: int = Field(ge=1)
    prompt: str
    question_type: QuestionType
    expected_answer_or_rubric: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_urls: list[HttpUrl] = Field(default_factory=list)
    attempts: list[QuizQuestionAttempt] = Field(default_factory=list)


class Quiz(BaseModel):
    id: int
    user_id: str
    topic: str
    total_questions: int = Field(gt=0)
    status: QuizStatus = "active"
    started_at: datetime
    completed_at: datetime | None = None
    answered_count: int = Field(default=0, ge=0)
    correct_count: int = Field(default=0, ge=0)
    last_active_at: datetime | None = None
    difficulty: Difficulty | None = None
    question_types: list[QuestionType] = Field(default_factory=list)
    questions: list[QuizQuestion] = Field(default_factory=list)
