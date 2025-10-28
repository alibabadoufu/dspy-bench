"""Optimizer implementations and registry for dspy-bench."""

from dspy_bench.optimizers.base import BaseOptimizer, OptimizationConfig, OptimizationResult
from dspy_bench.optimizers.registry import OptimizerRegistry, auto_register_optimizers, register_optimizer

# Auto-register all available optimizers
auto_register_optimizers()

__all__ = [
    "BaseOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "OptimizerRegistry",
    "auto_register_optimizers",
    "register_optimizer",
]