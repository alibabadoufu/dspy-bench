"""Base optimizer interface and result types."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from dspy_bench.dspy_integration.program_builder import DSPyProgramWrapper


class OptimizationResult(BaseModel):
    """Result of optimization process."""

    optimizer_name: str
    optimized_program: Optional[DSPyProgramWrapper] = None
    final_prompt: Optional[str] = None
    original_prompt: str
    metrics: Dict[str, float] = {}
    diagnostics: Dict[str, Any] = {}
    runtime_seconds: float = 0.0
    iterations_completed: int = 0
    success: bool = False
    error_message: Optional[str] = None
    checkpoints: List[str] = []  # Paths to checkpoint files
    metadata: Dict[str, Any] = {}


class OptimizationConfig(BaseModel):
    """Configuration for optimization process."""

    budget: Optional[int] = None  # Budget (e.g., number of LLM calls)
    max_iterations: int = 100
    early_stop: bool = True
    early_stop_patience: int = 10
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    val_metric: str = "accuracy"
    minimize_metric: bool = False
    random_seed: int = 42
    custom_params: Dict[str, Any] = {}


class BaseOptimizer(ABC):
    """Abstract base class for all optimizers."""

    name: str
    description: str
    required_config_keys: List[str] = []
    optional_config_keys: List[str] = []

    def __init__(self, config: OptimizationConfig):
        """Initialize optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self._checkpoints: List[OptimizationResult] = []
        self._best_result: Optional[OptimizationResult] = None

    @abstractmethod
    def optimize(
        self,
        program: DSPyProgramWrapper,
        train_data: List[Any],
        val_data: List[Any],
        metric_fn: callable,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Run optimization on the given program.

        Args:
            program: Initial DSPy program to optimize
            train_data: Training data examples
            val_data: Validation data examples
            metric_fn: Function to evaluate performance
            **kwargs: Additional optimizer-specific parameters

        Returns:
            Optimization result
        """
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this optimizer.

        Returns:
            Dictionary of default config values
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate optimizer configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid, raises ValueError if invalid
        """
        # Check required keys
        missing_keys = set(self.required_config_keys) - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        # Validate value types and ranges
        return True

    def save_checkpoint(self, result: OptimizationResult, checkpoint_path: str) -> None:
        """Save optimization checkpoint.

        Args:
            result: Current optimization result
            checkpoint_path: Path to save checkpoint
        """
        import pickle

        checkpoint_data = {
            "result": result,
            "optimizer_state": self._get_optimizer_state(),
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        self._checkpoints.append(result)
        result.checkpoints.append(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> OptimizationResult:
        """Load optimization checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded optimization result
        """
        import pickle

        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)

        result = checkpoint_data["result"]
        self._checkpoints.append(result)
        self._restore_optimizer_state(checkpoint_data["optimizer_state"])

        return result

    def _get_optimizer_state(self) -> Dict[str, Any]:
        """Get current optimizer state for checkpointing.

        Returns:
            Optimizer state dictionary
        """
        # Base implementation - subclasses can override
        return {
            "config": self.config.dict(),
            "checkpoints_count": len(self._checkpoints),
        }

    def _restore_optimizer_state(self, state: Dict[str, Any]) -> None:
        """Restore optimizer state from checkpoint.

        Args:
            state: Optimizer state dictionary
        """
        # Base implementation - subclasses can override
        pass

    def get_best_result(self) -> Optional[OptimizationResult]:
        """Get the best optimization result so far.

        Returns:
            Best result or None if no results yet
        """
        if self._best_result is None and self._checkpoints:
            # Determine best result from checkpoints
            best = None
            best_score = float('inf') if self.config.minimize_metric else float('-inf')

            for result in self._checkpoints:
                score = result.metrics.get(self.config.val_metric, 0.0)
                if self.config.minimize_metric:
                    if score < best_score:
                        best = result
                        best_score = score
                else:
                    if score > best_score:
                        best = result
                        best_score = score

            self._best_result = best

        return self._best_result

    def get_progress_info(self) -> Dict[str, Any]:
        """Get information about optimization progress.

        Returns:
            Progress information dictionary
        """
        return {
            "optimizer_name": self.name,
            "iterations_completed": len(self._checkpoints),
            "best_metric": self._best_result.metrics.get(self.config.val_metric, 0.0) if self._best_result else None,
            "current_iteration": len(self._checkpoints),
            "max_iterations": self.config.max_iterations,
        }