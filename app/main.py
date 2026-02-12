from fastapi import FastAPI
from app.router import api

app = FastAPI(title="Janus IA", version="0.1.0")

app.include_router(api.router, prefix="/v1")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Janus IA"}
