"""Security - API key validation for incoming client requests."""

from fastapi import HTTPException, Request


def validate_api_key(request: Request) -> None:
    """
    Validate JANUS_API_KEY from Authorization header or X-API-Key.
    Raises HTTPException 401 if invalid or missing.
    """
    from app.core.config import settings

    key = settings.JANUS_API_KEY
    if not key:
        return  # No key configured, skip validation

    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        token = auth[7:].strip()
        if token == key:
            return

    api_key = request.headers.get("X-API-Key")
    if api_key and api_key.strip() == key:
        return

    raise HTTPException(status_code=401, detail="Invalid or missing API key")
