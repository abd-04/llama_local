import os
from dotenv import load_dotenv


# Load variables from .env file
load_dotenv()


# Ollama local server URL
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)


# Default local model
DEFAULT_MODEL = os.getenv(
    "DEFAULT_MODEL",
    "llama3.2"
)


# Request timeout in seconds
REQUEST_TIMEOUT = int(
    os.getenv("REQUEST_TIMEOUT", 120)
)