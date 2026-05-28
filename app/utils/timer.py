import time
from contextlib import contextmanager


@contextmanager
def measure_time():
    """
    Measures how long a block of code takes to run.

    Used in benchmark, throughput, and concurrency endpoints
    to calculate LLM response latency.
    """

    start_time = time.perf_counter()

    timer = {
        "elapsed": 0.0
    }

    try:
        yield timer
    finally:
        end_time = time.perf_counter()
        timer["elapsed"] = round(end_time - start_time, 4)