from fastapi import APIRouter, HTTPException

from app.schemas import HealthResponse, ChatRequest, ChatResponse
from app.services.ollama_service import ollama_service


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Checks if the FastAPI app is running.
    """

    return {
        "status": "ok",
        "message": "Local LLM Gateway API is running."
    }


@router.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Sends a user message to the local Ollama model.
    """

    try:
        return ollama_service.chat(
            user_message=request.message,
            system_prompt=request.system_prompt,
            model=request.model
        )

    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))