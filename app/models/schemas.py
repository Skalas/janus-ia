from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    stream: Optional[bool] = False

# TODO: Add Response schemas matching OpenAI
