"""Deterministic workflow nodes for creating grounded quiz questions."""

from datetime import UTC, datetime

from google.adk.agents import Agent
from google.adk.agents.context import Context
from google.adk.tools._node_tool import NodeTool
from google.adk.workflow import node
from pydantic import BaseModel, Field, HttpUrl

from adk_4_gcp_learning_studio.schemas import (
    QuestionType,
    Quiz,
    QuizQuestion,
    QuizQuestionAttempt,
    SourceChunk,
)
from adk_4_gcp_learning_studio.tools import retriever

ACTIVE_QUIZ_STATE_KEY = "active_quiz"
MAX_QUESTION_EVIDENCE_CHUNKS = 3
# Keep the learning progression predictable while varying question style.
QUESTION_TYPE_ROTATION: tuple[QuestionType, ...] = (
    "recall",
    "comparison",
    "application",
)


class StartQuizRequest(BaseModel):
    """Input for the workflow that starts and creates a quiz question."""

    topic: str = Field(min_length=1)
    question_count: int = Field(default=3, ge=1)


class SubmitQuizAnswerRequest(BaseModel):
    """Input for the workflow that records an answer and creates the next question."""

    question_id: int
    learner_answer: str
    is_correct: bool
    feedback: str
    points: float = Field(default=0.0, ge=0.0, le=1.0)
    misconception: str | None = None


class QuizQuestionEvidence(BaseModel):
    """One citable documentation chunk available to the question author."""

    chunk_id: str
    title: str
    text: str
    url: HttpUrl


class QuizQuestionAuthorInput(BaseModel):
    """Evidence and required style supplied to the question-author model."""

    evidence: list[QuizQuestionEvidence] = Field(
        min_length=1,
        max_length=MAX_QUESTION_EVIDENCE_CHUNKS,
    )
    question_type: QuestionType


class QuestionAuthorDecision(BaseModel):
    """The subjective judgement and draft returned by the author model."""

    is_answerable: bool
    prompt: str = ""
    grading_rubric: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)
    insufficiency_reason: str | None = None


class InsufficientEvidenceError(ValueError):
    """Raised when the supplied evidence cannot support another question."""


# This child agent judges supplied evidence and creates wording only; Python owns state.
quiz_question_author = Agent(
    name="quiz_question_author",
    model="gemini-2.5-flash",
    mode="single_turn",
    input_schema=QuizQuestionAuthorInput,
    output_schema=QuestionAuthorDecision,
    instruction=(
        "Assess whether the supplied documentation evidence can support one concise "
        "free-response Google Cloud quiz question. If it can, set is_answerable to "
        "true; follow the requested question_type (recall explains a concept, "
        "comparison contrasts ideas, and application uses a short practical "
        "scenario); return a prompt, a private grading rubric, and every supplied "
        "source_chunk_id materially used. Do not add unsupported facts. If the "
        "evidence is insufficient, set is_answerable to false, explain why in "
        "insufficiency_reason, and leave prompt, grading_rubric, and "
        "source_chunk_ids empty."
    ),
)


def _expected_question_type(question_number: int) -> QuestionType:
    """Return the question style scheduled for a quiz position.

    Args:
        question_number: One-based position of the question in its quiz.

    Returns:
        The scheduled question style.
    """
    return QUESTION_TYPE_ROTATION[(question_number - 1) % len(QUESTION_TYPE_ROTATION)]


def _load_active_quiz(ctx: Context) -> Quiz | None:
    """Load and validate the active quiz from ADK session state.

    Args:
        ctx: ADK context for the current workflow run.

    Returns:
        The active quiz, or ``None`` if this session has no quiz.
    """
    active_quiz = ctx.state.get(ACTIVE_QUIZ_STATE_KEY)
    return Quiz.model_validate(active_quiz) if active_quiz else None


def _save_active_quiz(ctx: Context, quiz: Quiz) -> None:
    """Save a JSON-compatible quiz snapshot in ADK session state.

    Args:
        ctx: ADK context for the current workflow run.
        quiz: Updated quiz to persist in the session.
    """
    ctx.state[ACTIVE_QUIZ_STATE_KEY] = quiz.model_dump(mode="json")


def _next_quiz_id(ctx: Context) -> int:
    """Return the next quiz ID for the current session.

    Args:
        ctx: ADK context for the current workflow run.

    Returns:
        A session-local quiz ID.
    """
    active_quiz = _load_active_quiz(ctx)
    return active_quiz.id + 1 if active_quiz else 1


def _unused_evidence_chunks(
    chunks: list[SourceChunk],
    quiz: Quiz | None = None,
) -> list[SourceChunk]:
    """Return distinct, non-empty chunks not cited by saved questions.

    Args:
        chunks: Candidate documentation chunks from retrieval.
        quiz: Optional quiz whose existing questions define used chunks.

    Returns:
        Up to ``MAX_QUESTION_EVIDENCE_CHUNKS`` usable chunks in retrieval order.
    """
    # Each question may cite several chunks, so exclude every prior citation.
    used_chunk_ids = {
        chunk_id
        for question in (quiz.questions if quiz else [])
        for chunk_id in question.source_chunk_ids
    }
    unused_chunks: list[SourceChunk] = []
    seen_chunk_ids: set[str] = set()
    for chunk in chunks:
        if (
            chunk.chunk_id in used_chunk_ids
            or chunk.chunk_id in seen_chunk_ids
            or not chunk.text.strip()
        ):
            continue
        unused_chunks.append(chunk)
        seen_chunk_ids.add(chunk.chunk_id)
        if len(unused_chunks) == MAX_QUESTION_EVIDENCE_CHUNKS:
            break
    return unused_chunks


def _record_quiz_attempt(
    quiz: Quiz,
    question: QuizQuestion,
    request: SubmitQuizAnswerRequest,
) -> None:
    """Append an answer attempt and update aggregate quiz progress.

    Args:
        quiz: Quiz being updated.
        question: Saved question receiving the learner's answer.
        request: Grading result supplied by the root tutor.

    Raises:
        ValueError: If the question was already answered or feedback is empty.
    """
    if question.attempts:
        raise ValueError(f"Question {question.id} has already been answered.")
    if not request.feedback.strip():
        raise ValueError("feedback must not be empty.")

    # Store the individual attempt before updating the quiz-wide totals.
    question.attempts.append(
        QuizQuestionAttempt(
            question_id=question.id,
            learner_answer=request.learner_answer,
            is_correct=request.is_correct,
            points=request.points,
            feedback=request.feedback,
            misconception=request.misconception,
            answered_at=datetime.now(UTC),
        )
    )
    quiz.answered_count += 1
    if request.is_correct:
        quiz.correct_count += 1
    quiz.last_active_at = datetime.now(UTC)


async def _ask_question_author(
    ctx: Context,
    evidence_chunks: list[SourceChunk],
    question_type: QuestionType,
) -> QuestionAuthorDecision:
    """Ask the model to make the only subjective quiz decision.

    Args:
        ctx: ADK context used to run the child author agent.
        evidence_chunks: Retrieved documentation that may ground the question.
        question_type: Style chosen by Python for this question position.

    Returns:
        The author's evidence judgement, wording, rubric, and cited chunk IDs.
    """
    author_input = QuizQuestionAuthorInput(
        evidence=[
            QuizQuestionEvidence(
                chunk_id=chunk.chunk_id,
                title=chunk.title,
                text=chunk.text,
                url=chunk.url,
            )
            for chunk in evidence_chunks
        ],
        question_type=question_type,
    )
    # The model receives evidence and a style, never session state or source search.
    draft_output = await ctx.run_node(quiz_question_author, author_input)
    return QuestionAuthorDecision.model_validate(draft_output)


def _selected_evidence(
    author_decision: QuestionAuthorDecision,
    candidates: list[SourceChunk],
) -> list[SourceChunk]:
    """Validate the author's decision and return its cited source chunks.

    Args:
        author_decision: Model output to validate as untrusted data.
        candidates: Source chunks supplied to the model.

    Returns:
        The candidate chunks selected by the author, in citation order.

    Raises:
        InsufficientEvidenceError: If the author judges the evidence insufficient.
        ValueError: If the draft or selected citations are invalid.
    """
    if not author_decision.is_answerable:
        reason = (
            author_decision.insufficiency_reason
            or "The retrieved evidence is insufficient."
        )
        raise InsufficientEvidenceError(reason)
    if not author_decision.prompt.strip():
        raise ValueError("The question author returned an empty prompt.")
    if not author_decision.grading_rubric.strip():
        raise ValueError("The question author returned an empty grading rubric.")

    selected_chunk_ids = author_decision.source_chunk_ids
    if not selected_chunk_ids:
        raise ValueError("The question author selected no supporting source chunks.")
    if len(selected_chunk_ids) != len(set(selected_chunk_ids)):
        raise ValueError("The question author selected duplicate source chunks.")

    evidence_by_id = {chunk.chunk_id: chunk for chunk in candidates}
    unknown_chunk_ids = [
        chunk_id for chunk_id in selected_chunk_ids if chunk_id not in evidence_by_id
    ]
    if unknown_chunk_ids:
        raise ValueError("The question author selected an unknown source chunk.")
    return [evidence_by_id[chunk_id] for chunk_id in selected_chunk_ids]


def _build_question(
    quiz: Quiz,
    author_decision: QuestionAuthorDecision,
    question_type: QuestionType,
    sources: list[SourceChunk],
) -> QuizQuestion:
    """Build a question from validated model output and deterministic metadata.

    Args:
        quiz: Quiz that will own the question.
        author_decision: Validated author decision.
        question_type: Style chosen by Python for this question position.
        sources: Validated source chunks selected by the author.

    Returns:
        A question that is ready to add to the quiz.
    """
    question_number = len(quiz.questions) + 1
    return QuizQuestion(
        id=question_number,
        quiz_id=quiz.id,
        question_number=question_number,
        prompt=author_decision.prompt.strip(),
        question_type=question_type,
        expected_answer_or_rubric=author_decision.grading_rubric.strip(),
        source_chunk_ids=[source.chunk_id for source in sources],
        source_urls=[source.url for source in sources],
    )


async def _create_question(
    ctx: Context,
    quiz: Quiz,
    evidence_chunks: list[SourceChunk],
) -> tuple[QuizQuestion, list[SourceChunk]]:
    """Create, validate, and save one grounded question.

    Args:
        ctx: ADK context used to run the child author and save state.
        quiz: Active quiz that will receive the question.
        evidence_chunks: Retrieved documentation available to the author.

    Returns:
        The saved question and the source chunks it cites.

    Raises:
        InsufficientEvidenceError: If there is no usable or sufficient evidence.
        ValueError: If the author returns invalid content or citations.
    """
    if not evidence_chunks:
        raise InsufficientEvidenceError("No usable documentation evidence was found.")

    # Python fixes the learning style from the question's position in the quiz.
    question_type = _expected_question_type(len(quiz.questions) + 1)

    # The model returns its sufficiency judgement, draft, and selected citations.
    author_decision = await _ask_question_author(ctx, evidence_chunks, question_type)

    # Python validates the model-selected IDs and resolves them back to real source chunks.
    sources = _selected_evidence(author_decision, evidence_chunks)

    # Python combines the validated decision with IDs, style, and URLs the model cannot control.
    question = _build_question(quiz, author_decision, question_type, sources)

    # Persist only after every deterministic check and model-output check has passed.
    quiz.questions.append(question)
    quiz.question_types.append(question.question_type)
    _save_active_quiz(ctx, quiz)

    return question, sources


def _question_result(
    quiz: Quiz,
    question: QuizQuestion,
    sources: list[SourceChunk],
) -> dict:
    """Return the private and learner-facing data for a saved question.

    Args:
        quiz: Quiz containing the saved question.
        question: Saved question to expose to the root tutor.
        sources: Documentation chunks cited by the question.

    Returns:
        Tool result containing the prompt, private rubric, and citation.
    """
    return {
        "quiz_id": quiz.id,
        "topic": quiz.topic,
        "question_number": question.question_number,
        "total_questions": quiz.total_questions,
        "question": question.prompt,
        "grading_guidance": question.expected_answer_or_rubric,
        "sources": [
            {"title": source.title, "url": str(source.url)} for source in sources
        ],
    }


async def _retrieve_question_evidence(
    topic: str,
    quiz: Quiz | None = None,
) -> list[SourceChunk]:
    """Retrieve a small, unused evidence set for one question.

    Args:
        topic: Quiz topic used as the retrieval query.
        quiz: Existing quiz whose cited chunks must not be reused.

    Returns:
        Up to three candidate chunks that are safe to show the author.
    """
    # Ask for extra candidates after question one because earlier citations are filtered out.
    limit = (
        MAX_QUESTION_EVIDENCE_CHUNKS
        if quiz is None
        else quiz.total_questions * MAX_QUESTION_EVIDENCE_CHUNKS
    )
    chunks = await retriever.search(query=topic, products=[], limit=limit)
    return _unused_evidence_chunks(chunks, quiz)


def _complete_quiz(
    ctx: Context,
    quiz: Quiz,
    feedback: str,
    message: str,
) -> dict:
    """Mark a quiz complete, save it, and return its final progress.

    Args:
        ctx: ADK context that owns the session state.
        quiz: Active quiz to complete.
        feedback: Feedback for the answer that ended the quiz.
        message: Learner-facing completion explanation.

    Returns:
        A standard completion result for the root tutor.
    """
    quiz.status = "complete"
    quiz.completed_at = datetime.now(UTC)
    _save_active_quiz(ctx, quiz)
    return {
        "status": "success",
        "quiz_status": quiz.status,
        "message": message,
        "feedback": feedback,
        "answered_count": quiz.answered_count,
        "correct_count": quiz.correct_count,
    }


async def start_quiz_turn(
    ctx: Context,
    node_input: StartQuizRequest,
) -> dict:
    """Create a quiz, retrieve evidence, author, and save its first question.

    Args:
        ctx: ADK context for session state and child-node execution.
        node_input: Topic and requested number of questions.

    Returns:
        A success result containing a saved first question, or an error result.
    """
    # Retrieve evidence before creating any persisted quiz state.
    evidence_chunks = await _retrieve_question_evidence(node_input.topic)
    if not evidence_chunks:
        return {
            "status": "error",
            "message": f"I could not find documentation for {node_input.topic}.",
        }

    # The quiz is persisted only after its first complete question is available.
    quiz = Quiz(
        id=_next_quiz_id(ctx),
        user_id=ctx.user_id,
        topic=node_input.topic,
        total_questions=node_input.question_count,
        started_at=datetime.now(UTC),
        last_active_at=datetime.now(UTC),
    )
    try:
        question, sources = await _create_question(ctx, quiz, evidence_chunks)
    except (TypeError, ValueError) as error:
        return {"status": "error", "message": str(error)}
    return {"status": "success", **_question_result(quiz, question, sources)}


async def submit_quiz_answer_turn(
    ctx: Context,
    node_input: SubmitQuizAnswerRequest,
) -> dict:
    """Record an answer, then create and save the next question when needed.

    Args:
        ctx: ADK context for session state and child-node execution.
        node_input: Learner answer and the root tutor's grading result.

    Returns:
        A completion result, next saved question, or error result.
    """
    quiz = _load_active_quiz(ctx)
    if quiz is None:
        return {"status": "error", "message": "There is no active quiz."}
    if quiz.status != "active":
        return {"status": "error", "message": "The active quiz is already complete."}

    question = next(
        (item for item in quiz.questions if item.id == node_input.question_id),
        None,
    )
    if question is None:
        return {
            "status": "error",
            "message": f"Question {node_input.question_id} is not part of this quiz.",
        }

    # Validate the grading result before changing the saved quiz snapshot.
    try:
        _record_quiz_attempt(quiz, question, node_input)
    except ValueError as error:
        return {"status": "error", "message": str(error)}

    # The final answer completes the quiz without retrieving another source.
    if quiz.answered_count >= quiz.total_questions:
        return _complete_quiz(ctx, quiz, node_input.feedback, "Quiz complete.")

    # Retrieve several candidates so the author can use a compact evidence set.
    evidence_chunks = await _retrieve_question_evidence(quiz.topic, quiz)
    if not evidence_chunks:
        return _complete_quiz(
            ctx,
            quiz,
            node_input.feedback,
            "Quiz complete: there is no more unused documentation.",
        )

    try:
        next_question, sources = await _create_question(ctx, quiz, evidence_chunks)
    except InsufficientEvidenceError as error:
        return _complete_quiz(ctx, quiz, node_input.feedback, f"Quiz complete: {error}")
    except (TypeError, ValueError) as error:
        return {"status": "error", "message": str(error)}
    return {
        "status": "success",
        "quiz_status": quiz.status,
        "feedback": node_input.feedback,
        "answered_count": quiz.answered_count,
        "correct_count": quiz.correct_count,
        **_question_result(quiz, next_question, sources),
    }


# Convert the Python turn functions into resumable ADK workflow nodes.
start_quiz_workflow = node(
    start_quiz_turn,
    name="start_quiz_workflow",
    parameter_binding="node_input",
    rerun_on_resume=True,
)
submit_quiz_answer_workflow = node(
    submit_quiz_answer_turn,
    name="submit_quiz_answer_workflow",
    parameter_binding="node_input",
    rerun_on_resume=True,
)

# Expose the workflow nodes as the two tools the root tutor can call.
start_quiz = NodeTool(
    start_quiz_workflow,
    name="start_quiz",
    description="Start a grounded quiz and return its first saved question.",
)
submit_quiz_answer = NodeTool(
    submit_quiz_answer_workflow,
    name="submit_quiz_answer",
    description="Record a graded quiz answer and return the next saved question.",
)
