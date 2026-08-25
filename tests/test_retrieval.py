from datetime import UTC, datetime

import pytest
from pydantic import HttpUrl, TypeAdapter

from adk_4_gcp_learning_studio.retrieval.fake import FakeDocumentationRetriever
from adk_4_gcp_learning_studio.schemas import SourceChunk

http_url_adapter = TypeAdapter(HttpUrl)


@pytest.fixture
def cloud_run_chunks() -> list[SourceChunk]:
    """Return representative Cloud Run and Cloud Build chunks."""
    return [
        SourceChunk(
            chunk_id="cloud-run-service-overview",
            product="cloud-run",
            title="What is Cloud Run?",
            url=http_url_adapter.validate_python(
                "https://cloud.google.com/run/docs/overview/what-is-cloud-run"
            ),
            text="A Cloud Run service provides an endpoint for your application.",
            indexed_at=datetime(2026, 8, 21, tzinfo=UTC),
        ),
        SourceChunk(
            chunk_id="cloud-run-jobs-overview",
            product="cloud-run",
            title="What are Cloud Run jobs?",
            url=http_url_adapter.validate_python(
                "https://cloud.google.com/run/docs/create-jobs"
            ),
            text="A Cloud Run job runs tasks and exits when the work is complete.",
            indexed_at=datetime(2026, 8, 21, tzinfo=UTC),
        ),
        SourceChunk(
            chunk_id="cloud-build-overview",
            product="cloud-build",
            title="Cloud Build overview",
            url=http_url_adapter.validate_python(
                "https://cloud.google.com/build/docs/overview"
            ),
            text="Cloud Build executes your builds on Google Cloud infrastructure.",
            indexed_at=datetime(2026, 8, 21, tzinfo=UTC),
        ),
    ]


@pytest.mark.asyncio
async def test_search_returns_citable_cloud_run_chunks(
    cloud_run_chunks: list[SourceChunk],
) -> None:
    """Verify matching chunks retain their citation metadata."""
    retriever = FakeDocumentationRetriever(cloud_run_chunks)

    results = await retriever.search(
        query="endpoint application",
        products=["cloud-run"],
        limit=5,
    )

    assert [chunk.chunk_id for chunk in results] == ["cloud-run-service-overview"]
    assert results[0].title == "What is Cloud Run?"
    assert str(results[0].url) == "https://cloud.google.com/run/docs/overview/what-is-cloud-run"
    assert results[0].text
    assert results[0].indexed_at == datetime(2026, 8, 21, tzinfo=UTC)


@pytest.mark.asyncio
async def test_search_respects_product_filter_and_limit(
    cloud_run_chunks: list[SourceChunk],
) -> None:
    """Verify product filtering and result limits."""
    retriever = FakeDocumentationRetriever(cloud_run_chunks)

    results = await retriever.search(
        query="cloud",
        products=["cloud-run"],
        limit=1,
    )

    assert len(results) == 1
    assert results[0].product == "cloud-run"
