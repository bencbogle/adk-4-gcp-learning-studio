"""ADK tools used by the learning studio agent."""

from datetime import UTC, datetime

from google.adk.tools.tool_context import ToolContext

from adk_4_gcp_learning_studio.config import settings
from adk_4_gcp_learning_studio.retrieval.vertex_ai_search import (
    VertexAiSearchDocumentationRetriever,
)
from adk_4_gcp_learning_studio.schemas import (
    Quiz,
    QuizQuestion,
    QuizQuestionAttempt,
    SourceChunk,
)

retriever = VertexAiSearchDocumentationRetriever(
    project_id=settings.google_cloud_project,
    data_store_id=settings.gcp_documentation_data_store_id,
)
ACTIVE_QUIZ_STATE_KEY = "active_quiz"


def _build_quiz_question(
    quiz_id: int,
    question_number: int,
    chunk: SourceChunk,
) -> QuizQuestion:
    """Create one free-response question from a documentation chunk.

    Args:
        quiz_id: ID of the containing quiz.
        question_number: Position of the question in the quiz.
        chunk: Documentation used as the question's evidence.

    Returns:
        A grounded quiz question.
    """
    return QuizQuestion(
        id=question_number,
        quiz_id=quiz_id,
        question_number=question_number,
        prompt=(
            f"What is the main purpose of {chunk.title}, and what problem does "
            "it solve?"
        ),
        question_type="recall",
        expected_answer_or_rubric=chunk.text,
        source_chunk_ids=[chunk.chunk_id],
        source_urls=[chunk.url],
    )


def _record_quiz_attempt(
    quiz: Quiz,
    question: QuizQuestion,
    learner_answer: str,
    is_correct: bool,
    feedback: str,
    points: float,
    misconception: str | None,
) -> None:
    """Append an attempt and update aggregate quiz progress.

    Args:
        quiz: Quiz being updated.
        question: Answered question.
        learner_answer: Original learner response.
        is_correct: Whether the response met the rubric.
        feedback: Feedback written by the evaluator.
        points: Points awarded.
        misconception: Identified misconception, if any.
    """
    question.attempts.append(
        QuizQuestionAttempt(
            question_id=question.id,
            learner_answer=learner_answer,
            is_correct=is_correct,
            points=points,
            feedback=feedback,
            misconception=misconception,
            answered_at=datetime.now(UTC),
        )
    )
    quiz.answered_count += 1
    if is_correct:
        quiz.correct_count += 1
    quiz.last_active_at = datetime.now(UTC)


def _load_active_quiz(tool_context: ToolContext) -> Quiz | None:
    """Load and validate the active quiz from ADK session state.

    Args:
        tool_context: Current ADK tool context.

    Returns:
        The active quiz, or ``None`` when the session has no quiz.
    """
    active_quiz = tool_context.state.get(ACTIVE_QUIZ_STATE_KEY)
    if not active_quiz:
        return None
    return Quiz.model_validate(active_quiz)


def _save_active_quiz(tool_context: ToolContext, quiz: Quiz) -> None:
    """Save a JSON-compatible quiz snapshot in ADK session state.

    Args:
        tool_context: Current ADK tool context.
        quiz: Quiz to store.
    """
    tool_context.state[ACTIVE_QUIZ_STATE_KEY] = quiz.model_dump(mode="json")


def _next_quiz_id(tool_context: ToolContext) -> int:
    """Return the next quiz ID for the current session.

    Args:
        tool_context: Current ADK tool context.

    Returns:
        The next session-local quiz ID.
    """
    active_quiz = _load_active_quiz(tool_context)
    return active_quiz.id + 1 if active_quiz else 1


def _find_unused_chunk(
    chunks: list[SourceChunk],
    quiz: Quiz,
) -> SourceChunk | None:
    """Return the first retrieved chunk not already used by a quiz.

    Args:
        chunks: Candidate documentation chunks.
        quiz: Quiz whose source chunks should be excluded.

    Returns:
        The first unused chunk, or ``None``.
    """
    used_chunk_ids = {
        chunk_id
        for question in quiz.questions
        for chunk_id in question.source_chunk_ids
    }
    return next(
        (chunk for chunk in chunks if chunk.chunk_id not in used_chunk_ids),
        None,
    )


async def search_gcp_documentation(query: str, product: str = "") -> dict:
    """Find citable official Google Cloud documentation for a learner's question.

    Args:
        query: The learner's question or relevant search terms.
        product: An optional product ID, such as ``cloud-run``.
    """
    chunks = await retriever.search(
        query=query,
        products=[product] if product else [],
        limit=5,
    )

    return {
        "sources": [
            {
                "title": chunk.title,
                "url": str(chunk.url),
                "text": chunk.text,
            }
            for chunk in chunks
        ]
    }


async def list_supported_products() -> dict:
    """List products represented by the tutor's indexed documentation.

    Use this when a learner asks what products or topics the tutor covers. This
    describes the corpus; it is not documentation about those products.
    """
    return {"products": await retriever.list_supported_products()}


async def start_quiz(
    topic: str,
    tool_context: ToolContext,
    question_count: int = 3,
) -> dict:
    """Start a grounded, free-response quiz in the current session.

    Args:
        topic: The GCP topic to quiz the learner about.
        question_count: The requested number of questions.
    """

    if question_count < 1:
        return {
            "status": "error",
            "message": "question_count must be at least 1.",
        }

    # Retrieve one chunk so the first question is grounded in the corpus.
    chunks = await retriever.search(query=topic, products=[], limit=1)
    if not chunks:
        return {
            "status": "error",
            "message": f"I could not find documentation for {topic}.",
        }

    chunk = chunks[0]

    # Keep quiz IDs unique within this session without introducing persistence yet.
    quiz_id = _next_quiz_id(tool_context)

    # Build the first question from the retrieved document's title and content.
    question = _build_quiz_question(quiz_id, 1, chunk)
    # Store JSON-compatible data because ADK session state is a dictionary.
    quiz = Quiz(
        id=quiz_id,
        user_id=tool_context.user_id,
        topic=topic,
        total_questions=question_count,
        started_at=datetime.now(UTC),
        last_active_at=datetime.now(UTC),
        question_types=["recall"],
        questions=[question],
    )
    _save_active_quiz(tool_context, quiz)

    return {
        "status": "success",
        "quiz_id": quiz.id,
        "topic": quiz.topic,
        "question_number": question.question_number,
        "total_questions": quiz.total_questions,
        "question": question.prompt,
        # The agent uses this internally for grading; it must not reveal it.
        "grading_guidance": question.expected_answer_or_rubric,
        "sources": [{"title": chunk.title, "url": str(chunk.url)}],
    }


async def submit_quiz_answer(
    question_id: int,
    learner_answer: str,
    is_correct: bool,
    feedback: str,
    tool_context: ToolContext,
    points: float = 0.0,
    misconception: str | None = None,
) -> dict:
    """Record an evaluated answer and return the next quiz question.

    Args:
        question_id: The ID of the question being answered.
        learner_answer: The learner's free-response answer.
        is_correct: Whether the answer meets the rubric.
        feedback: Concise feedback for the learner.
        points: Points awarded for the answer.
        misconception: The learner's misconception, if one was identified.
    """
    quiz = _load_active_quiz(tool_context)
    if quiz is None:
        return {"status": "error", "message": "There is no active quiz."}
    if quiz.status != "active":
        return {"status": "error", "message": "The active quiz is already complete."}

    question = next(
        (question for question in quiz.questions if question.id == question_id),
        None,
    )
    if question is None:
        return {
            "status": "error",
            "message": f"Question {question_id} is not part of this quiz.",
        }

    # Record the agent's evaluation alongside the learner's original answer.
    _record_quiz_attempt(
        quiz,
        question,
        learner_answer,
        is_correct,
        feedback,
        points,
        misconception,
    )

    if quiz.answered_count >= quiz.total_questions:
        quiz.status = "complete"
        quiz.completed_at = datetime.now(UTC)
        _save_active_quiz(tool_context, quiz)
        return {
            "status": "success",
            "quiz_status": quiz.status,
            "message": "Quiz complete.",
            "feedback": feedback,
            "answered_count": quiz.answered_count,
            "correct_count": quiz.correct_count,
        }

    # Retrieve more chunks and choose one not already used by this quiz.
    chunks = await retriever.search(
        query=quiz.topic,
        products=[],
        limit=quiz.total_questions,
    )
    next_chunk = _find_unused_chunk(chunks, quiz)
    if next_chunk is None:
        # Persist the recorded attempt even when the corpus cannot provide more.
        quiz.status = "complete"
        quiz.completed_at = datetime.now(UTC)
        _save_active_quiz(tool_context, quiz)
        return {
            "status": "success",
            "quiz_status": quiz.status,
            "message": "Quiz complete: there is no more unused documentation.",
            "feedback": feedback,
            "answered_count": quiz.answered_count,
            "correct_count": quiz.correct_count,
        }

    # Add the next grounded question before persisting the updated quiz.
    next_question = _build_quiz_question(
        quiz.id,
        len(quiz.questions) + 1,
        next_chunk,
    )
    quiz.questions.append(next_question)
    _save_active_quiz(tool_context, quiz)

    return {
        "status": "success",
        "quiz_status": quiz.status,
        "feedback": feedback,
        "answered_count": quiz.answered_count,
        "correct_count": quiz.correct_count,
        "question_number": next_question.question_number,
        "total_questions": quiz.total_questions,
        "question": next_question.prompt,
        # The agent uses this internally for grading; it must not reveal it.
        "grading_guidance": next_question.expected_answer_or_rubric,
        "sources": [{"title": next_chunk.title, "url": str(next_chunk.url)}],
    }
