"""Vertex AI Search implementation of the documentation retrieval contract."""

import json
from datetime import UTC, datetime

from google.cloud import discoveryengine_v1
from pydantic import HttpUrl, TypeAdapter

from adk_4_gcp_learning_studio.schemas import SourceChunk

http_url_adapter = TypeAdapter(HttpUrl)


class VertexAiSearchDocumentationRetriever:
    """Retrieve citable documentation chunks from one Vertex AI Search data store."""

    def __init__(self, project_id: str, data_store_id: str) -> None:
        """Configure a retriever for a Vertex AI Search data store.

        Args:
            project_id: Google Cloud project containing the data store.
            data_store_id: Vertex AI Search data store ID.
        """
        # Create these lazily inside ``search`` so they attach to its event loop.
        self._search_client: discoveryengine_v1.SearchServiceAsyncClient | None = None
        self._document_client: discoveryengine_v1.DocumentServiceAsyncClient | None = (
            None
        )
        self._document_parent = (
            "projects/"
            f"{project_id}/locations/global/collections/default_collection/"
            f"dataStores/{data_store_id}/branches/0"
        )
        # This resource name identifies the exact data store that Google searches.
        self._serving_config = (
            "projects/"
            f"{project_id}/locations/global/collections/default_collection/"
            f"dataStores/{data_store_id}/servingConfigs/default_search"
        )

    async def search(
        self,
        query: str,
        products: list[str],
        limit: int,
    ) -> list[SourceChunk]:
        """Return relevant chunks with their citation metadata.

        Args:
            query: Search text.
            products: Product IDs used to filter results.
            limit: Maximum number of chunks to return.

        Returns:
            Relevant documentation chunks.
        """
        if limit <= 0:
            return []

        # Ask for text chunks rather than whole documents: they become SourceChunk.text.
        request = discoveryengine_v1.SearchRequest(
            serving_config=self._serving_config,
            query=query,
            page_size=min(limit, 100),
            content_search_spec=discoveryengine_v1.SearchRequest.ContentSearchSpec(
                search_result_mode=(
                    discoveryengine_v1.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS
                )
            ),
        )
        if products:
            # Apply the product filter in Google before results are returned.
            request.filter = (
                "product: ANY("
                + ", ".join(json.dumps(product) for product in products)
                + ")"
            )

        # Async gRPC clients bind to an event loop, which now exists because search runs.
        search_client = self._search_client
        if search_client is None:
            search_client = discoveryengine_v1.SearchServiceAsyncClient()
            self._search_client = search_client

        document_client = self._document_client
        if document_client is None:
            document_client = discoveryengine_v1.DocumentServiceAsyncClient()
            self._document_client = document_client

        results = await search_client.search(request=request)
        # A chunk result has no index time, so look it up once per parent document.
        indexed_at_by_document: dict[str, datetime] = {}
        unindexed_documents: set[str] = set()
        source_chunks: list[SourceChunk] = []

        async for result in results:
            chunk = result.chunk
            document_name = chunk.name.rsplit("/chunks/", maxsplit=1)[0]
            if document_name in unindexed_documents:
                continue

            indexed_at = indexed_at_by_document.get(document_name)
            if indexed_at is None:
                document = await document_client.get_document(name=document_name)
                # Proto-plus currently returns a datetime, despite its type stub.
                index_time = document.index_time
                if index_time is None:
                    # Segmentation can finish before indexing does. Do not expose
                    # a chunk that cannot yet carry the required citation time.
                    unindexed_documents.add(document_name)
                    continue
                if isinstance(index_time, datetime):
                    indexed_at = index_time.astimezone(UTC)
                else:
                    indexed_at = index_time.ToDatetime(tzinfo=UTC)
                indexed_at_by_document[document_name] = indexed_at

            # These citation fields were stored alongside the HTML when we uploaded it.
            metadata = dict(chunk.document_metadata.struct_data)
            source_chunks.append(
                # Convert Google's response object into our backend-independent model.
                SourceChunk(
                    chunk_id=chunk.name,
                    product=str(metadata["product"]),
                    title=str(metadata["title"]),
                    url=http_url_adapter.validate_python(str(metadata["source_url"])),
                    text=chunk.content,
                    indexed_at=indexed_at,
                    relevance_score=chunk.relevance_score,
                )
            )
            if len(source_chunks) == limit:
                break

        return source_chunks

    async def list_supported_products(self) -> list[str]:
        """Return product IDs present in indexed documents.

        Returns:
            Sorted product IDs.
        """
        document_client = self._document_client
        if document_client is None:
            document_client = discoveryengine_v1.DocumentServiceAsyncClient()
            self._document_client = document_client

        products: set[str] = set()
        documents = await document_client.list_documents(parent=self._document_parent)
        async for document in documents:
            if not document.index_status.index_time:
                continue

            metadata = dict(document.struct_data)
            product = metadata.get("product")
            if isinstance(product, str) and product:
                products.add(product)

        return sorted(products)
