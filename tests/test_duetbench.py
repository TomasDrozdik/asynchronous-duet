import logging

import pytest
from duet.bench import Benchmark
from duet.config import BenchmarkConfig, Pair

test_logger = logging.getLogger("test")


@pytest.mark.skip(reason="Runner mocking not implemented")
def test_benchmark_start_instances():
    benchmark_config = BenchmarkConfig(
        "duetname",
        {"image": "renaissance", "run": 1, "results": ["result.json"]},
        Pair.A,
    )
    benchmark = Benchmark(benchmark_config, test_logger)
    benchmark.start_instance()
