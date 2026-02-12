from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any

class LLMProvider(ABC):
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        stream: bool = False
    ) -> Dict[str, Any] | AsyncGenerator[Any, None]:
        """
        Standard interface for all providers.
        Input: Standard list of messages [{'role': 'user', 'content': '...'}, ...]
        Output: Normalized OpenAI-compatible dict or async generator.
        """
        pass
