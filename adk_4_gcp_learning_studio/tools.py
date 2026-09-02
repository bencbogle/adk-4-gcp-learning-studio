"""ADK tools used by the learning studio agent."""

from adk_4_gcp_learning_studio.config import settings
from adk_4_gcp_learning_studio.retrieval.vertex_ai_search import (
    VertexAiSearchDocumentationRetriever,
)

retriever = VertexAiSearchDocumentationRetriever(
    project_id=settings.google_cloud_project,
    data_store_id=settings.gcp_documentation_data_store_id,
)


def _normalise_product(product: str) -> str:
    """Convert a learner-facing product name to its corpus identifier."""
    return "-".join(product.casefold().split())


async def search_gcp_documentation(query: str, product: str = "") -> dict:
    """Find citable official Google Cloud documentation for a learner's question.

    Args:
        query: The learner's question or relevant search terms.
        product: An optional product ID, such as ``cloud-run``.
    """
    product_id = _normalise_product(product)
    chunks = await retriever.search(
        query=query,
        products=[product_id] if product_id else [],
        limit=5,
    )
    return {
        "sources": [
            {
                "chunk_id": chunk.chunk_id,
                "product": chunk.product,
                "title": chunk.title,
                "url": str(chunk.url),
                "text": chunk.text,
                "indexed_at": chunk.indexed_at.isoformat(),
                "relevance_score": chunk.relevance_score,
            }
            for chunk in chunks
        ]
    }


async def list_supported_products() -> dict:
    """List products represented by the tutor's indexed documentation."""
    return {"products": await retriever.list_supported_products()}
