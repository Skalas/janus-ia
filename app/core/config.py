from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration for Janus IA orchestration layer."""

    PROJECT_NAME: str = "Janus IA"
    VERSION: str = "0.1.0"

    # API Security (optional for MVP - leave empty to skip validation)
    JANUS_API_KEY: str | None = None

    # GCP / Vertex AI
    GCP_PROJECT_ID: str | None = None
    GCP_LOCATION: str = "us-central1"
    # Gemini 3 preview (gemini-3-pro-preview, gemini-3-flash-preview) solo en endpoint global
    GCP_LOCATION_GEMINI: str = "global"
    # Claude en Vertex no está en us-central1; usar "global" o región específica
    GCP_LOCATION_CLAUDE: str = "global"

    # OpenAI
    OPENAI_API_KEY: str | None = None

    # Model aliases -> Vertex/OpenAI model IDs
    # Gemini: usar modelos STABLE (gemini-3-*-preview puede requerir allowlist)
    # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
    MODEL_ALIASES: dict[str, str] = {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "claude-sonnet": "claude-sonnet-4-5",
        "claude-opus": "claude-opus-4-5",
        "claude-haiku": "claude-haiku-4-5",
        "gemini-pro": "gemini-3-pro-preview",
        "gemini-flash": "gemini-3-flash-preview",
    }

    # Default provider models when no explicit alias (Tier 2 heuristic)
    VERTEX_CLAUDE_MODEL: str = "claude-sonnet-4-5"
    VERTEX_GEMINI_MODEL: str = "gemini-3-pro-preview"
    OPENAI_DEFAULT_MODEL: str = "gpt-4o"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
