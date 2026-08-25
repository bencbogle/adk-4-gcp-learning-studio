from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest
from google.adk.tools.tool_context import ToolContext

from adk_4_gcp_learning_studio import agent, tools
from adk_4_gcp_learning_studio.schemas import SourceChunk


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

    assert await agent.list_supported_products() == {
        "products": ["cloud-run", "secret-manager"]
    }


@pytest.mark.asyncio
async def test_start_quiz_retrieves_context_and_saves_quiz_in_session_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that starting a quiz stores its first question in state."""
    source_chunk = SourceChunk(
        chunk_id="chunk-1",
        product="cloud-run",
        title="Cloud Run overview",
        url="https://cloud.google.com/run/docs/overview",
        text="Cloud Run runs containerized applications.",
        indexed_at=datetime.now(UTC),
    )

    async def fake_search(
        query: str, products: list[str], limit: int
    ) -> list:
        """Return the fixture chunk for the requested search."""
        assert query == "Cloud Run"
        assert products == []
        assert limit == 1
        return [source_chunk]

    monkeypatch.setattr(tools.retriever, "search", fake_search)
    tool_context = cast(
        ToolContext,
        SimpleNamespace(state={}, user_id="learner-1"),
    )

    result = await agent.start_quiz(
        topic="Cloud Run",
        question_count=3,
        tool_context=tool_context,
    )

    assert result["status"] == "success"
    assert result["quiz_id"] == 1
    assert result["question_number"] == 1
    assert result["total_questions"] == 3
    assert "question" in result
    assert tool_context.state[tools.ACTIVE_QUIZ_STATE_KEY]["topic"] == "Cloud Run"
    assert len(tool_context.state[tools.ACTIVE_QUIZ_STATE_KEY]["questions"]) == 1


@pytest.mark.asyncio
async def test_submit_quiz_answer_records_attempt_and_adds_next_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that an answer is recorded and the next question is added."""
    chunks = [
        SourceChunk(
            chunk_id="chunk-1",
            product="cloud-run",
            title="Cloud Run overview",
            url="https://cloud.google.com/run/docs/overview",
            text="Cloud Run runs containerized applications.",
            indexed_at=datetime.now(UTC),
        ),
        SourceChunk(
            chunk_id="chunk-2",
            product="cloud-run",
            title="Cloud Run services",
            url="https://cloud.google.com/run/docs/managing/services",
            text="A Cloud Run service serves requests.",
            indexed_at=datetime.now(UTC),
        ),
    ]

    async def fake_search(
        query: str, products: list[str], limit: int
    ) -> list[SourceChunk]:
        """Return the requested number of fixture chunks."""
        return chunks[:limit]

    monkeypatch.setattr(tools.retriever, "search", fake_search)
    tool_context = cast(
        ToolContext,
        SimpleNamespace(state={}, user_id="learner-1"),
    )
    await agent.start_quiz(
        topic="Cloud Run",
        tool_context=tool_context,
        question_count=2,
    )

    result = await agent.submit_quiz_answer(
        question_id=1,
        learner_answer="It runs containers.",
        is_correct=True,
        feedback="Correct.",
        tool_context=tool_context,
    )

    assert result["status"] == "success"
    assert result["question_number"] == 2
    saved_quiz = tool_context.state[tools.ACTIVE_QUIZ_STATE_KEY]
    assert saved_quiz["answered_count"] == 1
    assert saved_quiz["correct_count"] == 1
    assert len(saved_quiz["questions"][0]["attempts"]) == 1
    assert len(saved_quiz["questions"]) == 2
