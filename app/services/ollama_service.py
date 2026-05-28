import requests
from typing import Optional, Dict, Any, List

from app.config import OLLAMA_BASE_URL, DEFAULT_MODEL, REQUEST_TIMEOUT


class OllamaService:
    """
    Service layer for communicating with Ollama.

    This file contains the logic for:
    - sending chat requests to Ollama
    - summarization prompts
    - analysis prompts
    - extraction prompts
    - model listing
    - token estimation
    """

    def __init__(self):
        self.base_url = OLLAMA_BASE_URL.rstrip("/")
        self.default_model = DEFAULT_MODEL
        self.timeout = REQUEST_TIMEOUT

    def _resolve_model(self, model: Optional[str]) -> str:
        """
        If the user gives a model, use it.
        Otherwise use the default model from .env.
        """

        return model or self.default_model

    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation.

        1 token is approximately 4 characters.
        This is not exact, but useful for basic benchmarking.
        """

        return max(1, len(text) // 4)

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        num_predict: int = 220
    ) -> Dict[str, Any]:
        """
        Sends a chat request to Ollama's local API.
        """

        selected_model = self._resolve_model(model)

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": user_message
        })

        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_ctx": 2048,
                "num_thread": 4,
                "num_predict": num_predict
            }
        }

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()
        data = response.json()

        return {
            "model": selected_model,
            "response": data.get("message", {}).get("content", "")
        }

    def summarize(
        self,
        text: str,
        style: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a summarization prompt and sends it to Ollama.
        """

        prompt = f"""
Summarize the following text in this style: {style}

Text:
{text}

Return only the summary.
"""

        result = self.chat(
            user_message=prompt,
            system_prompt="You are a concise summarization assistant.",
            model=model,
            num_predict=220
        )

        return {
            "model": result["model"],
            "summary": result["response"]
        }

    def analyze(
        self,
        text: str,
        analysis_type: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates an analysis prompt and sends it to Ollama.
        """

        prompt = f"""
Analyze the following text.

Analysis type: {analysis_type}

Possible analysis types:
- sentiment
- risk
- keywords
- complaint analysis
- general

Text:
{text}

Return a clear structured analysis.
"""

        result = self.chat(
            user_message=prompt,
            system_prompt="You are a professional text analysis assistant.",
            model=model,
            num_predict=220
        )

        return {
            "model": result["model"],
            "analysis_type": analysis_type,
            "analysis": result["response"]
        }

    def extract(
        self,
        text: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates an information extraction prompt and sends it to Ollama.
        """

        prompt = f"""
Extract structured information from the following text.

Extract:
- names
- emails
- phone_numbers
- dates
- organizations
- locations
- keywords

Return JSON only.

Text:
{text}
"""

        result = self.chat(
            user_message=prompt,
            system_prompt="You extract structured data and return JSON only.",
            model=model,
            num_predict=220
        )

        return {
            "model": result["model"],
            "extracted_data": result["response"]
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lists locally available Ollama models.
        """

        response = requests.get(
            f"{self.base_url}/api/tags",
            timeout=self.timeout
        )

        response.raise_for_status()
        data = response.json()

        return data.get("models", [])


ollama_service = OllamaService()