from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    message: str


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="User message to send to the local LLM."
    )

    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to control the assistant behavior."
    )

    model: Optional[str] = Field(
        default=None,
        description="Optional model name. If not provided, DEFAULT_MODEL from .env is used."
    )


class ChatResponse(BaseModel):
    model: str
    response: str