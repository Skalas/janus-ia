from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Janus IA"
    VERSION: str = "0.1.0"
    
    # GCP
    GCP_PROJECT_ID: str | None = None
    GCP_LOCATION: str = "us-central1"
    
    # OpenAI
    OPENAI_API_KEY: str | None = None
    
    class Config:
        env_file = ".env"

settings = Settings()
