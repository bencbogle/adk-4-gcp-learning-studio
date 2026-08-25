## Import the focused Google Cloud corpus

The tutor searches the data store named by `GCP_DOCUMENTATION_DATA_STORE_ID`.
To add the focused product overviews and agent-workflow pages used for the
retrieval exercises, run:

```bash
uv run python scripts/import_cloud_run_docs.py
```

The importer downloads the official HTML pages and stores `title`, `product`,
and `source_url` alongside each document. Vertex AI Search then creates chunks
from the HTML using the data store's layout-based chunking configuration.

The starter set covers Cloud Run, Artifact Registry, Secret Manager, Cloud SQL,
Cloud Build, Agent Platform and ADK, IAM, Cloud Storage, Agent Search, Pub/Sub,
Workflows, Eventarc, Cloud Logging, and Cloud Monitoring.
