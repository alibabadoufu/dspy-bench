"""dspy-bench: Open-source extensible Python project to evaluate DSPy strategies & optimizers."""

__version__ = "0.1.0"

from dspy_bench.config import Config
from dspy_bench.data.loader import JSONLDataset
from dspy_bench.optimizers.registry import OptimizerRegistry

__all__ = ["Config", "JSONLDataset", "OptimizerRegistry"]