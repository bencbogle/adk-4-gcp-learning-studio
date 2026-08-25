"""The stable interface used to retrieve citable GCP documentation."""

from typing import Protocol

from adk_4_gcp_learning_studio.schemas import SourceChunk


class DocumentationRetriever(Protocol):
    """Find official documentation relevant to a learner's question.

    Implementations may use a local fixture, Vertex AI Search, or another
    retrieval backend. Callers only depend on this contract.
    """

    async def search(
        self,
        query: str,
        products: list[str],
        limit: int,
    ) -> list[SourceChunk]:
        """Return at most ``limit`` citable documentation chunks.

        Args:
            query: Search text.
            products: Product IDs used to filter results.
            limit: Maximum number of chunks to return.

        Returns:
            Matching documentation chunks.
        """
        ...
