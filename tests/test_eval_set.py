import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from adk_4_gcp_learning_studio.retrieval.fake import FakeDocumentationRetriever
from adk_4_gcp_learning_studio.schemas import SourceChunk


def _load_cases() -> list[dict[str, Any]]:
    """Load the small, repeatable retrieval evaluation set."""
    cases_path = Path(__file__).parents[1] / "evals" / "retrieval_cases.json"
    return json.loads(cases_path.read_text())


def test_retrieval_eval_set_has_unique_grounded_cases() -> None:
    """Verify retrieval cases are unique and use official documentation URLs."""
    cases = _load_cases()

    assert len(cases) == 5
    assert len({case["id"] for case in cases}) == len(cases)

    for case in cases:
        assert case["query"]
        assert case["product"]
        assert case["expected_source_urls"]
        assert all(
            url.startswith("https://docs.cloud.google.com/")
            for url in case["expected_source_urls"]
        )


@pytest.mark.asyncio
async def test_retrieval_eval_cases_find_their_expected_sources() -> None:
    """Verify that every evaluation case exercises the retriever contract."""
    cases = _load_cases()
    chunks = [
        SourceChunk(
            chunk_id=str(case["id"]),
            product=str(case["product"]),
            title=str(case["id"]),
            url=str(case["expected_source_urls"][0]),
            text=str(case["query"]),
            indexed_at=datetime(2026, 8, 21, tzinfo=UTC),
        )
        for case in cases
    ]
    retriever = FakeDocumentationRetriever(chunks)

    for case in cases:
        results = await retriever.search(
            query=str(case["query"]),
            products=[str(case["product"])],
            limit=5,
        )

        result_urls = {str(chunk.url) for chunk in results}
        expected_urls = {str(url) for url in case["expected_source_urls"]}
        assert expected_urls <= result_urls, case["id"]
