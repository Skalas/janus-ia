from app.providers.base import LLMProvider
from typing import List, Dict, Any

class VertexClaudeProvider(LLMProvider):
    async def chat_completion(self, messages: List[Dict[str, str]], stream: bool = False):
        # TODO: Implement Vertex Claude call
        pass
