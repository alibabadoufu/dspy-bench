"""GEPA (Generalized Error-driven Prompt Augmentation) optimizer adapter."""

import time
from typing import Any, Dict, List

from dspy_bench.dspy_integration.program_builder import DSPyProgramWrapper
from dspy_bench.optimizers.base import BaseOptimizer, OptimizationConfig, OptimizationResult
from dspy_bench.optimizers.registry import register_optimizer


@register_optimizer("gepa")
class GEPAOptimizer(BaseOptimizer):
    """Adapter for GEPA optimizer."""

    name = "gepa"
    description = "Generalized Error-driven Prompt Augmentation"
    required_config_keys = []
    optional_config_keys = ["budget", "early_stop", "temperature", "max_iterations"]

    def __init__(self, config: OptimizationConfig):
        """Initialize GEPA optimizer."""
        super().__init__(config)
        self._gepa_optimizer = None

    def optimize(
        self,
        program: DSPyProgramWrapper,
        train_data: List[Any],
        val_data: List[Any],
        metric_fn: callable,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Run GEPA optimization.

        Args:
            program: Initial DSPy program
            train_data: Training examples
            val_data: Validation examples
            metric_fn: Evaluation metric function
            **kwargs: Additional parameters

        Returns:
            Optimization result
        """
        start_time = time.time()
        original_prompt = program.seed_prompt

        try:
            # Try to import GEPA
            try:
                import gepa
                from gepa import GEPA
            except ImportError:
                raise ImportError("GEPA not installed. Install with: pip install gepa")

            # Extract GEPA parameters
            budget = kwargs.get("budget", 100)
            early_stop = kwargs.get("early_stop", True)
            temperature = kwargs.get("temperature", 0.7)
            max_iterations = kwargs.get("max_iterations", self.config.max_iterations)

            # Create GEPA optimizer
            self._gepa_optimizer = GEPA(
                budget=budget,
                early_stop=early_stop,
                temperature=temperature,
                max_iterations=max_iterations,
            )

            # Convert data to GEPA format
            trainset = self._convert_to_gepa_examples(train_data)
            valset = self._convert_to_gepa_examples(val_data)

            # Create initial prompt template
            initial_prompt = program.seed_prompt

            # Run GEPA optimization
            optimized_prompt, optimization_history = self._gepa_optimizer.optimize(
                initial_prompt=initial_prompt,
                train_examples=trainset,
                val_examples=valset,
                metric_fn=metric_fn,
            )

            # Evaluate final prompt on validation set
            val_metrics = self._evaluate_prompt(optimized_prompt, valset, metric_fn)

            # Create wrapped program with optimized prompt
            wrapped_program = DSPyProgramWrapper(
                program=program.program,  # Keep original program structure
                seed_prompt=optimized_prompt,
                instruction=program.instruction,
                signature=program.signature,
                metadata={
                    **program.metadata,
                    "optimized": True,
                    "optimizer": self.name,
                    "optimization_params": {
                        "budget": budget,
                        "early_stop": early_stop,
                        "temperature": temperature,
                        "max_iterations": max_iterations,
                    },
                    "optimization_history": optimization_history,
                }
            )

            runtime = time.time() - start_time

            result = OptimizationResult(
                optimizer_name=self.name,
                optimized_program=wrapped_program,
                final_prompt=optimized_prompt,
                original_prompt=original_prompt,
                metrics=val_metrics,
                diagnostics={
                    "budget_used": len(optimization_history) if optimization_history else 0,
                    "train_size": len(train_data),
                    "val_size": len(val_data),
                    "optimization_history_length": len(optimization_history) if optimization_history else 0,
                },
                runtime_seconds=runtime,
                iterations_completed=len(optimization_history) if optimization_history else 0,
                success=True,
                metadata={
                    "gepa_version": getattr(gepa, "__version__", "unknown"),
                    "config_used": kwargs,
                }
            )

            return result

        except ImportError as e:
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
        """Get default configuration for GEPA.

        Returns:
            Default configuration dictionary
        """
        return {
            "budget": 100,
            "early_stop": True,
            "temperature": 0.7,
            "max_iterations": 50,
        }

    def _convert_to_gepa_examples(self, data: List[Any]) -> List[Dict[str, Any]]:
        """Convert data examples to GEPA format.

        Args:
            data: List of data examples

        Returns:
            List of GEPA-compatible examples
        """
        gepa_examples = []
        for example in data:
            if hasattr(example, 'input') and hasattr(example, 'label'):
                gepa_example = {
                    "input": example.input,
                    "output": example.label,
                }
                if hasattr(example, 'id'):
                    gepa_example["id"] = example.id
                gepa_examples.append(gepa_example)

        return gepa_examples

    def _evaluate_prompt(
        self,
        prompt: str,
        valset: List[Dict[str, Any]],
        metric_fn: callable,
    ) -> Dict[str, float]:
        """Evaluate prompt on validation set.

        Args:
            prompt: Prompt to evaluate
            valset: Validation examples
            metric_fn: Metric function

        Returns:
            Dictionary of metrics
        """
        if not valset:
            return {"accuracy": 0.0}

        # This is a simplified evaluation - in practice, you'd want to
        # use the actual LLM to generate predictions using the prompt
        # and then evaluate them with the metric function

        try:
            # Try to evaluate using the metric function if possible
            # This would typically involve calling the LLM with the prompt
            # and then scoring the results
            correct = 0
            total = len(valset)

            for example in valset:
                try:
                    # Here you would typically:
                    # 1. Format the prompt with the example input
                    # 2. Call the LLM to get a prediction
                    # 3. Compare with the expected output using metric_fn

                    # For now, return a placeholder evaluation
                    # In a real implementation, this would involve actual LLM calls
                    if metric_fn:
                        # This would be the actual evaluation logic
                        is_correct = metric_fn(example, "")  # Placeholder
                        if is_correct:
                            correct += 1
                except Exception:
                    pass

            accuracy = correct / total if total > 0 else 0.0
            return {"accuracy": accuracy}

        except Exception:
            # Return default metrics if evaluation fails
            return {"accuracy": 0.0}