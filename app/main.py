import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Vertex AI generative_models está deprecado hasta jun 2026; migrar a google-genai
warnings.filterwarnings(
    "ignore",
    message=".*deprecated as of June 24, 2025.*",
    module="vertexai.generative_models",
)

from app.core.config import settings
from app.router import api


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # Cleanup if needed (e.g. close provider connections)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)

app.include_router(api.router, prefix="/v1")


@app.get("/health")
def health_check():
    return {"status": "ok", "service": settings.PROJECT_NAME}
