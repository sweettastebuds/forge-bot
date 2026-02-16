"""Application configuration via environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All configuration for forge-bot, validated at startup."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # --- Gitea/Forgejo connection (required) ---
    forge_instance_url: str = Field(description="Base URL of Gitea/Forgejo instance")
    forge_api_token: str = Field(description="Bot account API token")
    forge_webhook_secret: str = Field(description="HMAC secret for webhook verification")

    # --- LLM connection ---
    llm_api_key: str = Field(description="API key for the LLM endpoint")
    llm_base_url: str = Field(default="https://api.openai.com/v1")
    llm_model: str = Field(default="gpt-4o")
    llm_temperature: float = Field(default=0.2)
    llm_max_tokens: int = Field(default=4096)
    llm_timeout: int = Field(default=120, description="Request timeout in seconds")
    llm_max_concurrent: int = Field(default=3, description="Max parallel LLM requests")
    llm_context_window: int = Field(
        default=8192,
        description="Model context window in tokens"
        " — controls conversation trimming and context budgets",
    )

    # --- Sandbox ---
    sandbox_enabled: bool = Field(default=True)
    sandbox_timeout: int = Field(default=60, description="Max execution time in seconds")
    sandbox_memory: str = Field(default="512m")
    sandbox_cpus: float = Field(default=1.0)
    sandbox_prepull_images: str = Field(
        default="python,node",
        description="Comma-separated image keys to pre-pull, or 'all'/'none'",
    )
    sandbox_images_file: str = Field(
        default="",
        description="Path to custom sandbox-images.json override",
    )

    # --- RAG (optional) ---
    rag_enabled: bool = Field(default=False)
    rag_embed_model: str = Field(default="all-MiniLM-L6-v2")
    rag_embed_url: str = Field(default="")
    rag_chunk_size: int = Field(default=1500)
    rag_chunk_overlap: int = Field(default=200)
    rag_top_k: int = Field(default=10)
    rag_max_context_tokens: int = Field(default=4000)
    rag_store_path: str = Field(default="/data/chromadb")

    # --- General ---
    webhook_port: int = Field(default=8080)
    log_level: str = Field(default="INFO")
    bot_command_prefix: str = Field(default="/")
