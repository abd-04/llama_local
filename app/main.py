from fastapi import FastAPI
from app.routers.ai_routes import router as ai_router

app= FastAPI(
    title="local llm gateway",
    description="a fastapi backend that wraps the local llama model from ollama",
)

@app.get("/")
def root():
    return {"message": "working!!!"}

app.include_router(ai_router, prefix="")

@app.get("/health")
def health():
    return {"status": "ok"}