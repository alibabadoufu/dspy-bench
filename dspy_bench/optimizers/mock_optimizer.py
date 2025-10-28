"""Mock optimizer for testing purposes."""

import random
import time
from typing import Any, Dict, List

from dspy_bench.dspy_integration.program_builder import DSPyProgramWrapper
from dspy_bench.optimizers.base import BaseOptimizer, OptimizationConfig, OptimizationResult
from dspy_bench.optimizers.registry import register_optimizer


@register_optimizer("mock_optimizer")
class MockOptimizer(BaseOptimizer):
    """Mock optimizer for testing and demonstration."""

    name = "mock_optimizer"
    description = "Mock optimizer for testing - simulates optimization with random improvements"
    required_config_keys = []
    optional_config_keys = ["improvement_rate", "max_improvements", "random_seed"]

    def __init__(self, config: OptimizationConfig):
        """Initialize mock optimizer."""
        super().__init__(config)
        self._random_seed = getattr(config, "random_seed", 42)
        random.seed(self._random_seed)

    def optimize(
        self,
        program: DSPyProgramWrapper,
        train_data: List[Any],
        val_data: List[Any],
        metric_fn: callable,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Run mock optimization.

        Args:
            program: Initial DSPy program
            train_data: Training examples
            val_data: Validation examples
            metric_fn: Evaluation metric function
            **kwargs: Additional parameters

        Returns:
            Mock optimization result
        """
        start_time = time.time()
        original_prompt = program.seed_prompt

        try:
            # Extract mock parameters
            improvement_rate = kwargs.get("improvement_rate", 0.1)
            max_improvements = kwargs.get("max_improvements", 5)
            random_seed = kwargs.get("random_seed", self._random_seed)

            # Set random seed for reproducibility
            random.seed(random_seed)

            # Simulate optimization iterations
            improvements_made = 0
            current_prompt = original_prompt
            best_score = 0.5  # Starting with 50% accuracy
            optimization_history = []

            for iteration in range(min(max_improvements, getattr(self.config, 'max_iterations', 100))):
                # Simulate some "optimization work"
                time.sleep(0.1)  # Simulate processing time

                # Randomly decide if we make an improvement
                if random.random() < improvement_rate:
                    improvements_made += 1
                    # Simulate improvement in metric
                    improvement = random.uniform(0.05, 0.15)
                    best_score = min(1.0, best_score + improvement)

                    # Create a "better" prompt by adding iteration info
                    current_prompt = f"{original_prompt}\n\n[Optimized iteration {iteration + 1}]"

                optimization_history.append({
                    "iteration": iteration + 1,
                    "score": best_score,
                    "improvement_made": improvements_made > 0,
                })

                # Check early stopping condition
                if self.config.early_stop and best_score >= 0.95:
                    break

            # Create final metrics
            final_metrics = {
                "accuracy": best_score,
                "improvements_made": improvements_made,
                "iterations": len(optimization_history),
            }

            # Create wrapped program with "optimized" prompt
            wrapped_program = DSPyProgramWrapper(
                program=program.program,
                seed_prompt=current_prompt,
                instruction=program.instruction,
                signature=program.signature,
                metadata={
                    **program.metadata,
                    "optimized": True,
                    "optimizer": self.name,
                    "optimization_params": {
                        "improvement_rate": improvement_rate,
                        "max_improvements": max_improvements,
                        "random_seed": random_seed,
                    },
                    "optimization_history": optimization_history,
                }
            )

            runtime = time.time() - start_time

            result = OptimizationResult(
                optimizer_name=self.name,
                optimized_program=wrapped_program,
                final_prompt=current_prompt,
                original_prompt=original_prompt,
                metrics=final_metrics,
                diagnostics={
                    "train_size": len(train_data),
                    "val_size": len(val_data),
                    "optimization_history": optimization_history,
                    "improvement_rate": improvement_rate,
                },
                runtime_seconds=runtime,
                iterations_completed=len(optimization_history),
                success=True,
                metadata={
                    "mock_optimizer": True,
                    "config_used": kwargs,
                }
            )

            return result

        except Exception as e:
            runtime = time.time() - start_time
            return OptimizationResult(
                optimizer_name=self.name,
                original_prompt=original_prompt,
                metrics={},
                diagnostics={"error": str(e)},
                runtime_seconds=runtime,
                iterations_completed=0,
                success=False,
                error_message=str(e),
            )

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for mock optimizer.

        Returns:
            Default configuration dictionary
        """
        return {
            "improvement_rate": 0.2,
            "max_improvements": 10,
            "random_seed": 42,
        }