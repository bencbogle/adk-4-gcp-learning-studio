"""Import a focused set of official Google Cloud pages into Vertex AI Search."""

from urllib.request import Request, urlopen

from google.api_core.exceptions import AlreadyExists
from google.cloud import discoveryengine_v1

from adk_4_gcp_learning_studio.config import settings

DOCUMENTS = (
    {
        "id": "cloud-run-deploying",
        "product": "cloud-run",
        "title": "Deploy container images to Cloud Run services",
        "url": "https://docs.cloud.google.com/run/docs/deploying",
    },
    {
        "id": "cloud-run-managing-services",
        "product": "cloud-run",
        "title": "Manage Cloud Run services",
        "url": "https://docs.cloud.google.com/run/docs/managing/services",
    },
    {
        "id": "cloud-run-create-jobs",
        "product": "cloud-run",
        "title": "Create jobs",
        "url": "https://docs.cloud.google.com/run/docs/create-jobs",
    },
    {
        "id": "artifact-registry-overview",
        "product": "artifact-registry",
        "title": "Artifact Registry overview",
        "url": "https://docs.cloud.google.com/artifact-registry/docs/overview",
    },
    {
        "id": "secret-manager-overview",
        "product": "secret-manager",
        "title": "Secret Manager overview",
        "url": "https://docs.cloud.google.com/secret-manager/docs/overview",
    },
    {
        "id": "cloud-sql-overview",
        "product": "cloud-sql",
        "title": "Cloud SQL overview",
        "url": "https://docs.cloud.google.com/sql/docs/introduction",
    },
    {
        "id": "cloud-build-overview",
        "product": "cloud-build",
        "title": "Overview of Cloud Build",
        "url": "https://docs.cloud.google.com/build/docs/overview",
    },
    {
        "id": "agent-platform-overview",
        "product": "agent-platform",
        "title": "Agent Platform overview",
        "url": "https://docs.cloud.google.com/gemini-enterprise-agent-platform/overview",
    },
    {
        "id": "agent-platform-adk-quickstart",
        "product": "agent-platform",
        "title": "Develop and deploy agents on Agent Runtime with Agent Development Kit",
        "url": "https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/runtime/quickstart-adk",
    },
    {
        "id": "iam-overview",
        "product": "iam",
        "title": "IAM overview",
        "url": "https://docs.cloud.google.com/iam/docs/overview",
    },
    {
        "id": "cloud-storage-overview",
        "product": "cloud-storage",
        "title": "Cloud Storage overview",
        "url": "https://docs.cloud.google.com/storage/docs/introduction",
    },
    {
        "id": "agent-search-overview",
        "product": "agent-search",
        "title": "About Vertex AI Search",
        "url": "https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/vertex-ai-search",
    },
    {
        "id": "pubsub-overview",
        "product": "pubsub",
        "title": "What is Pub/Sub?",
        "url": "https://docs.cloud.google.com/pubsub/docs/overview",
    },
    {
        "id": "workflows-overview",
        "product": "workflows",
        "title": "Workflows overview",
        "url": "https://docs.cloud.google.com/workflows/docs/overview",
    },
    {
        "id": "eventarc-overview",
        "product": "eventarc",
        "title": "Eventarc overview",
        "url": "https://docs.cloud.google.com/eventarc/docs/overview",
    },
    {
        "id": "cloud-logging-overview",
        "product": "cloud-logging",
        "title": "Cloud Logging overview",
        "url": "https://docs.cloud.google.com/logging/docs/overview",
    },
    {
        "id": "cloud-monitoring-overview",
        "product": "cloud-monitoring",
        "title": "Cloud Monitoring overview",
        "url": "https://docs.cloud.google.com/monitoring/docs/monitoring-overview",
    },
)


def download_page(url: str) -> bytes:
    """Download one official documentation page as HTML.

    Args:
        url: Documentation page URL.

    Returns:
        Raw HTML bytes.
    """
    request = Request(url, headers={"User-Agent": "gcp-learning-studio-doc-import/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def main() -> None:
    """Import the configured documentation pages into Vertex AI Search."""
    client = discoveryengine_v1.DocumentServiceClient()
    parent = (
        "projects/"
        f"{settings.google_cloud_project}/locations/global/collections/"
        f"default_collection/dataStores/{settings.gcp_documentation_data_store_id}/"
        "branches/0"
    )

    for item in DOCUMENTS:
        print(f"Downloading {item['title']}...")
        document = discoveryengine_v1.Document(
            id=item["id"],
            schema_id="default_schema",
            parent_document_id=item["id"],
            struct_data={
                "title": item["title"],
                "product": item["product"],
                "source_url": item["url"],
            },
            content=discoveryengine_v1.Document.Content(
                mime_type="text/html",
                raw_bytes=download_page(item["url"]),
            ),
        )
        try:
            client.create_document(
                parent=parent,
                document=document,
                document_id=item["id"],
            )
        except AlreadyExists:
            print(f"Already present: {item['id']}")
        else:
            print(f"Imported: {item['id']}")


if __name__ == "__main__":
    main()
