"""Configuration management for search index services."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # Azure OpenAI Configuration
    azure_openai_endpoint: str = Field(..., alias="AZURE_OPENAI_EMBEDDING_ENDPOINT")
    azure_openai_deployment: str = Field(..., alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    azure_openai_api_version: str = Field("2024-02-01", alias="AZURE_OPENAI_EMBEDDING_API_VERSION")

    # Azure Authentication
    azure_client_id: str = Field(..., alias="AZURE_CLIENT_ID")
    azure_client_secret: str = Field(..., alias="AZURE_CLIENT_SECRET")
    azure_tenant_id: str = Field(..., alias="AZURE_TENANT_ID")

    # Vena API Configuration
    vena_endpoint: str = Field(..., alias="VENA_ENDPOINT")

    # LanceDB Configuration
    lancedb_base_path: str = Field("./lancedb", alias="LANCEDB_BASE_PATH")
    storage_backend: str = Field("local", alias="STORAGE_BACKEND")  # "local" or "azure"

    # Azure Storage Configuration
    azure_storage_account_name: str = Field("", alias="AZURE_STORAGE_ACCOUNT_NAME")
    azure_storage_container_name: str = Field("lancedb", alias="AZURE_STORAGE_CONTAINER_NAME")

    # Service Configuration
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    max_embedding_batch_size: int = Field(1000, alias="MAX_EMBEDDING_BATCH_SIZE")


settings = Settings()
