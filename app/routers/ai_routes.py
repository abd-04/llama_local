from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import APIRouter

from app.schemas import (
    HealthResponse,
    ChatRequest,
    ChatResponse,
    SummarizeRequest,
    SummarizeResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    ExtractRequest,
    ExtractResponse,
    BenchmarkRequest,
    BenchmarkResponse,
    ThroughputTestRequest,
    ThroughputTestResponse,
    SingleRunMetric,
    ConcurrencyTestRequest,
    ConcurrencyTestResponse,
    ConcurrencyResult,
    ModelsResponse,
)

from app.services.ollama_service import ollama_service
from app.utils.timer import measure_time


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "message": "Local LLM Gateway API is running."
    }


@router.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return ollama_service.chat(
        user_message=request.message,
        system_prompt=request.system_prompt,
        model=request.model,
        num_predict=220
    )


@router.post("/api/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    return ollama_service.summarize(
        text=request.text,
        style=request.style,
        model=request.model
    )


@router.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    return ollama_service.analyze(
        text=request.text,
        analysis_type=request.analysis_type,
        model=request.model
    )


@router.post("/api/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest):
    return ollama_service.extract(
        text=request.text,
        model=request.model
    )


@router.post("/api/benchmark", response_model=BenchmarkResponse)
def benchmark(request: BenchmarkRequest):
    with measure_time() as timer:
        result = ollama_service.chat(
            user_message=request.prompt,
            system_prompt="You are a concise assistant. Answer clearly but briefly.",
            model=request.model,
            num_predict=120
        )

    latency_seconds = timer["elapsed"]
    response_text = result["response"]

    output_characters = len(response_text)
    estimated_tokens = ollama_service.estimate_tokens(response_text)

    estimated_tokens_per_second = round(
        estimated_tokens / latency_seconds,
        2
    ) if latency_seconds > 0 else 0.0

    return {
        "model": result["model"],
        "latency_seconds": latency_seconds,
        "output_characters": output_characters,
        "estimated_tokens": estimated_tokens,
        "estimated_tokens_per_second": estimated_tokens_per_second,
        "response": response_text
    }


@router.post("/api/throughput-test", response_model=ThroughputTestResponse)
def throughput_test(request: ThroughputTestRequest):
    results = []
    total_latency = 0.0
    total_tokens = 0

    for run_number in range(1, request.runs + 1):
        with measure_time() as timer:
            result = ollama_service.chat(
                user_message=request.prompt,
                system_prompt="You are a concise assistant. Keep answers short.",
                model=request.model,
                num_predict=100
            )

        latency_seconds = timer["elapsed"]
        response_text = result["response"]

        output_characters = len(response_text)
        estimated_tokens = ollama_service.estimate_tokens(response_text)

        tokens_per_second = round(
            estimated_tokens / latency_seconds,
            2
        ) if latency_seconds > 0 else 0.0

        total_latency += latency_seconds
        total_tokens += estimated_tokens

        results.append(
            SingleRunMetric(
                run_number=run_number,
                latency_seconds=latency_seconds,
                output_characters=output_characters,
                estimated_tokens=estimated_tokens,
                estimated_tokens_per_second=tokens_per_second
            )
        )

    average_latency = round(total_latency / request.runs, 4)

    average_tokens_per_second = round(
        total_tokens / total_latency,
        2
    ) if total_latency > 0 else 0.0

    return {
        "model": request.model or ollama_service.default_model,
        "runs": request.runs,
        "total_latency_seconds": round(total_latency, 4),
        "average_latency_seconds": average_latency,
        "total_estimated_tokens": total_tokens,
        "average_tokens_per_second": average_tokens_per_second,
        "results": results
    }


@router.post("/api/concurrency-test", response_model=ConcurrencyTestResponse)
def concurrency_test(request: ConcurrencyTestRequest):
    model_name = request.model or ollama_service.default_model

    def run_single_request(request_number: int):
        with measure_time() as timer:
            result = ollama_service.chat(
                user_message=request.prompt,
                system_prompt="You are a concise assistant. Keep answers very short.",
                model=request.model,
                num_predict=70
            )

        response_text = result["response"]
        estimated_tokens = ollama_service.estimate_tokens(response_text)

        return ConcurrencyResult(
            request_number=request_number,
            success=True,
            latency_seconds=timer["elapsed"],
            estimated_tokens=estimated_tokens,
            response_preview=response_text[:120],
            error=None
        )

    results = []

    with measure_time() as total_timer:
        with ThreadPoolExecutor(max_workers=request.concurrent_requests) as executor:
            futures = [
                executor.submit(run_single_request, i)
                for i in range(1, request.concurrent_requests + 1)
            ]

            for future in as_completed(futures):
                results.append(future.result())

    results = sorted(results, key=lambda item: item.request_number)

    successful_results = [result for result in results if result.success]
    failed_results = [result for result in results if not result.success]

    average_latency = round(
        sum(result.latency_seconds for result in successful_results) / len(successful_results),
        4
    ) if successful_results else 0.0

    return {
        "model": model_name,
        "concurrent_requests": request.concurrent_requests,
        "total_wall_time_seconds": total_timer["elapsed"],
        "successful_requests": len(successful_results),
        "failed_requests": len(failed_results),
        "average_latency_seconds": average_latency,
        "results": results
    }


@router.get("/api/models", response_model=ModelsResponse)
def list_models():
    models = ollama_service.list_models()
    return {"models": models}