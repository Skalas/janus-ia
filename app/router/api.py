from fastapi import APIRouter
from app.models.schemas import ChatCompletionRequest

router = APIRouter()

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # TODO: Route request to appropriate provider
    return {"message": "Janus IA: Not implemented yet"}
