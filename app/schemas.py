from typing import List, Optional
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


class SummarizeRequest(BaseModel):
    test:str = Field(..., min_length=10)
    style: str = "Bullet points"
    model: Optional[str]


class SummarizeResponse(BaseModel):
    model: str
    summary: str


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    analysis_type: str = "general"
    model: Optional[str] = None


class AnalyzeResponse(BaseModel):
    model: str
    analysis_type: str
    analysis: str

class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: Optional[str] = None


class ExtractResponse(BaseModel):
    model: str
    extracted_data: str

class BenchmarkRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: Optional[str] = None


class BenchmarkResponse(BaseModel):
    model: str
    latency_seconds: float
    output_characters: int
    estimated_tokens: int
    estimated_tokens_per_second: float
    response: str

class ThroughputTestRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    runs: int = Field(default=3, ge=1, le=10)
    model: Optional[str] = None


class SingleRunMetric(BaseModel):
    run_number: int
    latency_seconds: float
    output_characters: int
    estimated_tokens: int
    estimated_tokens_per_second: float


class ThroughputTestResponse(BaseModel):
    model: str
    runs: int
    total_latency_seconds: float
    average_latency_seconds: float
    total_estimated_tokens: int
    average_tokens_per_second: float
    results: List[SingleRunMetric]


class ConcurrencyTestRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    concurrent_requests: int = Field(default=2, ge=1, le=3)
    model: Optional[str] = None


class ConcurrencyResult(BaseModel):
    request_number: int
    success: bool
    latency_seconds: float
    estimated_tokens: int
    response_preview: Optional[str] = None
    error: Optional[str] = None


class ConcurrencyTestResponse(BaseModel):
    model: str
    concurrent_requests: int
    total_wall_time_seconds: float
    successful_requests: int
    failed_requests: int
    average_latency_seconds: float
    results: List[ConcurrencyResult]