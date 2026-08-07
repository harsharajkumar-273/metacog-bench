"""
MetaCog-Bench: Metacognitive Evaluation Suite for Frontier AI Models.
"""

__version__ = "1.0.0"
__author__ = "Applied Cognitive Evaluation Group"

from metacog_bench.config import BENCHMARK_NAME, METRIC_WEIGHTS
from metacog_bench.evaluator import MetaCogEvaluator

__all__ = ["MetaCogEvaluator", "BENCHMARK_NAME", "METRIC_WEIGHTS"]
