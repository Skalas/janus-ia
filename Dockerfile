# Janus IA - Meta Chat Orchestrator
# Python 3.11 slim for Cloud Run

FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml ./

# Install application and dependencies
RUN uv pip install --system --no-cache .

# Copy application
COPY app/ ./app/

# Non-root user
RUN useradd -m janus && chown -R janus:janus /app
USER janus

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
