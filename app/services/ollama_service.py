import requests
from typing import Optional, Dict, Any

from app.config import OLLAMA_BASE_URL, DEFAULT_MODEL, REQUEST_TIMEOUT


class OllamaService:
    """
    Service layer for communicating with Ollama.

    The FastAPI routes will call this service.
    This keeps Ollama logic separate from API routing logic.
    """

    def __init__(self):
        self.base_url = OLLAMA_BASE_URL.rstrip("/")
        self.default_model = DEFAULT_MODEL
        self.timeout = REQUEST_TIMEOUT

    def _resolve_model(self, model: Optional[str]) -> str:
        """
        Use requested model if provided.
        Otherwise use the default model from .env.
        """

        return model or self.default_model

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        num_predict:int=180
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
                "max_tokens": 2048,
                "temperature": 0.7,
                "num_ctx" : 2048,
                "num_thread" : 4,
                "num_predict": num_predict
                
            }
        }

        try:
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

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running."
            )

        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Ollama request timed out. Try a smaller prompt or faster model."
            )

        except requests.exceptions.HTTPError as error:
            raise RuntimeError(
                f"Ollama returned an HTTP error: {str(error)}"
            )

        except requests.exceptions.RequestException as error:
            raise RuntimeError(
                f"Unexpected error while calling Ollama: {str(error)}"
            )


ollama_service = OllamaService()