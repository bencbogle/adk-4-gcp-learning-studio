from datetime import UTC, datetime
from typing import cast

import pytest
from google.adk.agents.context import Context

from adk_4_gcp_learning_studio import quiz_workflows, tools
from adk_4_gcp_learning_studio.agent import root_agent
from adk_4_gcp_learning_studio.schemas import SourceChunk


class FakeWorkflowContext:
    """Minimal workflow context that supplies deterministic author drafts."""

    def __init__(self, drafts: list[quiz_workflows.QuestionAuthorDecision]) -> None:
        self.state: dict = {}
        self.user_id = "learner-1"
        self.author_inputs: list[quiz_workflows.QuizQuestionAuthorInput] = []
        self._drafts = drafts

    async def run_node(
        self,
        _node: object,
        node_input: quiz_workflows.QuizQuestionAuthorInput,
    ) -> quiz_workflows.QuestionAuthorDecision:
        """Return the next pre-written author draft."""
        self.author_inputs.append(node_input)
        return self._drafts.pop(0)


def _chunk(number: int) -> SourceChunk:
    """Build one distinct documentation chunk for a quiz test."""
    return SourceChunk(
        chunk_id=f"chunk-{number}",
        product="cloud-run",
        title=f"Cloud Run topic {number}",
        url=f"https://cloud.google.com/run/docs/topic-{number}",
        text=f"Cloud Run concept {number}.",
        indexed_at=datetime.now(UTC),
    )


def _draft(
    number: int,
    source_chunk_ids: list[str] | None = None,
) -> quiz_workflows.QuestionAuthorDecision:
    """Build a valid structured draft returned by the author node."""
    return quiz_workflows.QuestionAuthorDecision(
        is_answerable=True,
        prompt=f"How would you explain Cloud Run concept {number}?",
        grading_rubric=f"Mentions Cloud Run concept {number}.",
        source_chunk_ids=(
            source_chunk_ids if source_chunk_ids is not None else [f"chunk-{number}"]
        ),
    )


def _as_context(context: FakeWorkflowContext) -> Context:
    """Cast the test double to the small Context surface the workflow uses."""
    return cast(Context, context)


@pytest.mark.asyncio
async def test_list_supported_products_returns_corpus_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that product listing exposes corpus capabilities."""

    async def fake_list_supported_products() -> list[str]:
        """Return fake corpus capabilities."""
        return ["cloud-run", "secret-manager"]

    monkeypatch.setattr(
        tools.retriever,
        "list_supported_products",
        fake_list_supported_products,
    )

    assert await tools.list_supported_products() == {
        "products": ["cloud-run", "secret-manager"]
    }


@pytest.mark.asyncio
async def test_search_documentation_preserves_citation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that the search tool exposes the complete source contract."""
    source_chunk = _chunk(1).model_copy(
        update={
            "indexed_at": datetime(2026, 8, 21, 12, 30, tzinfo=UTC),
            "relevance_score": 0.91,
        }
    )

    async def fake_search(
        query: str, products: list[str], limit: int
    ) -> list[SourceChunk]:
        """Return one source with all citation fields populated."""
        assert query == "What is Cloud Run?"
        assert products == ["cloud-run"]
        assert limit == 5
        return [source_chunk]

    monkeypatch.setattr(tools.retriever, "search", fake_search)

    assert await tools.search_gcp_documentation(
        query="What is Cloud Run?",
        product="Cloud Run",
    ) == {
        "sources": [
            {
                "chunk_id": "chunk-1",
                "product": "cloud-run",
                "title": "Cloud Run topic 1",
                "url": "https://cloud.google.com/run/docs/topic-1",
                "text": "Cloud Run concept 1.",
                "indexed_at": "2026-08-21T12:30:00+00:00",
                "relevance_score": 0.91,
            }
        ]
    }


@pytest.mark.asyncio
async def test_start_workflow_saves_the_first_question_in_one_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify retrieval, authoring, and saving happen in one workflow turn."""
    source_chunk = _chunk(1)

    async def fake_search(
        query: str, products: list[str], limit: int
    ) -> list[SourceChunk]:
        """Return the fixture source for the first question."""
        assert query == "Cloud Run"
        assert products == []
        assert limit == quiz_workflows.MAX_QUESTION_EVIDENCE_CHUNKS
        return [source_chunk]

    monkeypatch.setattr(tools.retriever, "search", fake_search)
    context = FakeWorkflowContext([_draft(1)])

    result = await quiz_workflows.start_quiz_turn(
        _as_context(context),
        quiz_workflows.StartQuizRequest(topic="Cloud Run", question_count=3),
    )

    assert result["status"] == "success"
    assert result["question"] == "How would you explain Cloud Run concept 1?"
    assert context.author_inputs[0].question_type == "recall"
    assert [item.chunk_id for item in context.author_inputs[0].evidence] == ["chunk-1"]
    saved_quiz = context.state[quiz_workflows.ACTIVE_QUIZ_STATE_KEY]
    assert saved_quiz["questions"][0]["source_chunk_ids"] == ["chunk-1"]
    assert "pending_question_source" not in saved_quiz


@pytest.mark.asyncio
async def test_submit_workflow_saves_the_next_question_and_rotates_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a submitted answer creates the next question in the same turn."""
    chunks = [_chunk(1), _chunk(2)]

    async def fake_search(
        query: str, products: list[str], limit: int
    ) -> list[SourceChunk]:
        """Return enough sources for both questions."""
        return chunks[:limit]

    monkeypatch.setattr(tools.retriever, "search", fake_search)
    context = FakeWorkflowContext([_draft(1), _draft(2)])
    await quiz_workflows.start_quiz_turn(
        _as_context(context),
        quiz_workflows.StartQuizRequest(topic="Cloud Run", question_count=2),
    )

    result = await quiz_workflows.submit_quiz_answer_turn(
        _as_context(context),
        quiz_workflows.SubmitQuizAnswerRequest(
            question_id=1,
            learner_answer="It runs containers.",
            is_correct=True,
            feedback="Correct.",
        ),
    )

    assert result["status"] == "success"
    assert result["question_number"] == 2
    assert result["question"] == "How would you explain Cloud Run concept 2?"
    assert context.author_inputs[1].question_type == "comparison"
    assert [item.chunk_id for item in context.author_inputs[1].evidence] == ["chunk-2"]
    saved_quiz = context.state[quiz_workflows.ACTIVE_QUIZ_STATE_KEY]
    assert saved_quiz["answered_count"] == 1
    assert saved_quiz["question_types"] == ["recall", "comparison"]
    assert len(saved_quiz["questions"]) == 2


@pytest.mark.asyncio
async def test_question_types_rotate_to_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the author receives the deterministic three-question rotation."""
    chunks = [_chunk(1), _chunk(2), _chunk(3)]

    async def fake_search(
        query: str, products: list[str], limit: int
    ) -> list[SourceChunk]:
        """Return enough distinct sources for every question."""
        return chunks[:limit]

    monkeypatch.setattr(tools.retriever, "search", fake_search)
    context = FakeWorkflowContext([_draft(1), _draft(2), _draft(3)])
    await quiz_workflows.start_quiz_turn(
        _as_context(context),
        quiz_workflows.StartQuizRequest(topic="Cloud Run", question_count=3),
    )
    for question_id in (1, 2):
        await quiz_workflows.submit_quiz_answer_turn(
            _as_context(context),
            quiz_workflows.SubmitQuizAnswerRequest(
                question_id=question_id,
                learner_answer="Answer.",
                is_correct=True,
                feedback="Correct.",
            ),
        )

    assert [item.question_type for item in context.author_inputs] == [
        "recall",
        "comparison",
        "application",
    ]


@pytest.mark.asyncio
async def test_submit_workflow_rejects_duplicate_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a saved question cannot be counted twice."""
    chunks = [_chunk(1), _chunk(2)]

    async def fake_search(
        query: str, products: list[str], limit: int
    ) -> list[SourceChunk]:
        """Return enough fixture sources to keep the quiz active."""
        return chunks[:limit]

    monkeypatch.setattr(tools.retriever, "search", fake_search)
    context = FakeWorkflowContext([_draft(1), _draft(2)])
    await quiz_workflows.start_quiz_turn(
        _as_context(context),
        quiz_workflows.StartQuizRequest(topic="Cloud Run", question_count=2),
    )
    request = quiz_workflows.SubmitQuizAnswerRequest(
        question_id=1,
        learner_answer="It runs containers.",
        is_correct=True,
        feedback="Correct.",
    )
    await quiz_workflows.submit_quiz_answer_turn(_as_context(context), request)

    assert await quiz_workflows.submit_quiz_answer_turn(
        _as_context(context), request
    ) == {
        "status": "error",
        "message": "Question 1 has already been answered.",
    }


def test_root_agent_exposes_workflow_tools_not_the_save_protocol() -> None:
    """Verify the conversational agent cannot call the obsolete save tool."""
    tool_names = [
        getattr(tool, "name", getattr(tool, "__name__", ""))
        for tool in root_agent.tools
    ]

    assert "start_quiz" in tool_names
    assert "submit_quiz_answer" in tool_names
    assert "save_quiz_question" not in tool_names


@pytest.mark.asyncio
async def test_start_workflow_saves_only_author_selected_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a question retains only the chunks selected by the author."""
    chunks = [_chunk(1), _chunk(2), _chunk(3)]

    async def fake_search(
        query: str, products: list[str], limit: int
    ) -> list[SourceChunk]:
        """Return the complete evidence candidate set."""
        assert limit == quiz_workflows.MAX_QUESTION_EVIDENCE_CHUNKS
        return chunks[:limit]

    monkeypatch.setattr(tools.retriever, "search", fake_search)
    context = FakeWorkflowContext([_draft(1, ["chunk-1", "chunk-3"])])

    result = await quiz_workflows.start_quiz_turn(
        _as_context(context),
        quiz_workflows.StartQuizRequest(topic="Cloud Run"),
    )

    assert [item.chunk_id for item in context.author_inputs[0].evidence] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert result["sources"] == [
        {
            "title": "Cloud Run topic 1",
            "url": "https://cloud.google.com/run/docs/topic-1",
        },
        {
            "title": "Cloud Run topic 3",
            "url": "https://cloud.google.com/run/docs/topic-3",
        },
    ]
    saved_quiz = context.state[quiz_workflows.ACTIVE_QUIZ_STATE_KEY]
    assert saved_quiz["questions"][0]["source_chunk_ids"] == ["chunk-1", "chunk-3"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_chunk_ids", "message"),
    [
        ([], "selected no supporting source chunks"),
        (["chunk-1", "chunk-1"], "selected duplicate source chunks"),
        (["missing"], "selected an unknown source chunk"),
    ],
)
async def test_start_workflow_rejects_invalid_author_citations(
    monkeypatch: pytest.MonkeyPatch,
    source_chunk_ids: list[str],
    message: str,
) -> None:
    """Verify invalid citations cannot persist a partial first question."""

    async def fake_search(
        query: str, products: list[str], limit: int
    ) -> list[SourceChunk]:
        """Return enough candidate chunks to exercise citation validation."""
        return [_chunk(1), _chunk(2)][:limit]

    monkeypatch.setattr(tools.retriever, "search", fake_search)
    context = FakeWorkflowContext([_draft(1, source_chunk_ids)])

    result = await quiz_workflows.start_quiz_turn(
        _as_context(context),
        quiz_workflows.StartQuizRequest(topic="Cloud Run"),
    )

    assert result == {"status": "error", "message": f"The question author {message}."}
    assert quiz_workflows.ACTIVE_QUIZ_STATE_KEY not in context.state


@pytest.mark.asyncio
async def test_submit_workflow_completes_when_evidence_is_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a recorded answer survives when no grounded next question is possible."""
    chunks = [_chunk(1), _chunk(2)]

    async def fake_search(
        query: str, products: list[str], limit: int
    ) -> list[SourceChunk]:
        """Return one saved and one candidate chunk."""
        return chunks[:limit]

    monkeypatch.setattr(tools.retriever, "search", fake_search)
    context = FakeWorkflowContext(
        [
            _draft(1),
            quiz_workflows.QuestionAuthorDecision(
                is_answerable=False,
                insufficiency_reason="The remaining chunk lacks enough detail.",
            ),
        ]
    )
    await quiz_workflows.start_quiz_turn(
        _as_context(context),
        quiz_workflows.StartQuizRequest(topic="Cloud Run", question_count=2),
    )

    result = await quiz_workflows.submit_quiz_answer_turn(
        _as_context(context),
        quiz_workflows.SubmitQuizAnswerRequest(
            question_id=1,
            learner_answer="It runs containers.",
            is_correct=True,
            feedback="Correct.",
        ),
    )

    assert result["quiz_status"] == "complete"
    assert result["message"] == (
        "Quiz complete: The remaining chunk lacks enough detail."
    )
    saved_quiz = context.state[quiz_workflows.ACTIVE_QUIZ_STATE_KEY]
    assert saved_quiz["answered_count"] == 1
    assert saved_quiz["questions"][0]["attempts"]
    assert len(saved_quiz["questions"]) == 1
