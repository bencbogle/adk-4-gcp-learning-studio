"""Local and deployed configuration for the learning studio."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Values shared by the agent and the documentation retriever."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_cloud_project: str
    google_cloud_location: str = "europe-west2"
    gcp_documentation_data_store_id: str


# BasedPyright cannot infer values supplied by BaseSettings from the environment.
settings = Settings()  # pyright: ignore[reportCallIssue]
