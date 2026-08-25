"""Deterministic documentation retriever for tests and local experiments."""

from collections.abc import Iterable

from adk_4_gcp_learning_studio.schemas import SourceChunk


class FakeDocumentationRetriever:
    """Search a supplied in-memory collection of documentation chunks.

    This deliberately uses simple word matching. It verifies the application's
    retrieval boundary; it does not attempt to mimic production search ranking.
    """

    def __init__(self, chunks: Iterable[SourceChunk]) -> None:
        """Initialize the retriever with deterministic chunks.

        Args:
            chunks: Chunks available to search.
        """
        self._chunks: list[SourceChunk] = list(chunks)

    async def search(
        self,
        query: str,
        products: list[str],
        limit: int,
    ) -> list[SourceChunk]:
        """Return matching chunks in their supplied order.

        Args:
            query: Search text.
            products: Product IDs used to filter results.
            limit: Maximum number of chunks to return.

        Returns:
            Matching chunks.
        """
        if limit <= 0:
            return []

        query_terms = set(query.casefold().split())
        requested_products = {product.casefold() for product in products}

        matches: list[SourceChunk] = []
        for chunk in self._chunks:
            if (
                requested_products
                and chunk.product.casefold() not in requested_products
            ):
                continue

            searchable_text = f"{chunk.title} {chunk.text}".casefold()
            if query_terms and not query_terms.intersection(searchable_text.split()):
                continue

            matches.append(chunk)
            if len(matches) == limit:
                break

        return matches
