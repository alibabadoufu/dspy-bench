"""BootstrapFewShot optimizer adapter."""

import time
from typing import Any, Dict, List

import dspy

from dspy_bench.dspy_integration.program_builder import DSPyProgramWrapper
from dspy_bench.optimizers.base import BaseOptimizer, OptimizationConfig, OptimizationResult
from dspy_bench.optimizers.registry import register_optimizer


@register_optimizer("bootstrap_fewshot")
class BootstrapFewShotOptimizer(BaseOptimizer):
    """Adapter for DSPy's BootstrapFewShot optimizer."""

    name = "bootstrap_fewshot"
    description = "Few-shot example selection and prompt augmentation"
    required_config_keys = []
    optional_config_keys = ["max_bootstrapped_demos", "max_labeled_demos", "max_rounds"]

    def __init__(self, config: OptimizationConfig):
        """Initialize BootstrapFewShot optimizer."""
        super().__init__(config)
        self._dspy_optimizer = None

    def optimize(
        self,
        program: DSPyProgramWrapper,
        train_data: List[Any],
        val_data: List[Any],
        metric_fn: callable,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Run BootstrapFewShot optimization.

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
            # Extract optimizer parameters
            max_bootstrapped_demos = kwargs.get("max_bootstrapped_demos", 8)
            max_labeled_demos = kwargs.get("max_labeled_demos", 8)
            max_rounds = kwargs.get("max_rounds", 1)

            # Create DSPy BootstrapFewShot optimizer
            self._dspy_optimizer = dspy.BootstrapFewShot(
                metric=metric_fn,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                max_rounds=max_rounds,
            )

            # Convert data to DSPy format
            trainset = self._convert_to_dspy_examples(train_data)
            valset = self._convert_to_dspy_examples(val_data)

            # Run optimization
            optimized_program = self._dspy_optimizer.compile(
                program.program,
                trainset=trainset,
                valset=valset,
            )

            # Evaluate optimized program
            val_metrics = self._evaluate_program(optimized_program, valset, metric_fn)

            # Create wrapped program
            wrapped_program = DSPyProgramWrapper(
                program=optimized_program,
                seed_prompt=original_prompt,
                instruction=program.instruction,
                signature=str(optimized_program.signature) if hasattr(optimized_program, 'signature') else None,
                metadata={
                    **program.metadata,
                    "optimized": True,
                    "optimizer": self.name,
                    "optimization_params": {
                        "max_bootstrapped_demos": max_bootstrapped_demos,
                        "max_labeled_demos": max_labeled_demos,
                        "max_rounds": max_rounds,
                    }
                }
            )

            # Extract optimized prompt if possible
            final_prompt = self._extract_prompt(optimized_program)

            runtime = time.time() - start_time

            result = OptimizationResult(
                optimizer_name=self.name,
                optimized_program=wrapped_program,
                final_prompt=final_prompt,
                original_prompt=original_prompt,
                metrics=val_metrics,
                diagnostics={
                    "demos_count": len(optimized_program.demos) if hasattr(optimized_program, 'demos') else 0,
                    "train_size": len(train_data),
                    "val_size": len(val_data),
                },
                runtime_seconds=runtime,
                iterations_completed=1,
                success=True,
                metadata={
                    "dspy_optimizer_class": "BootstrapFewShot",
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
        """Get default configuration for BootstrapFewShot.

        Returns:
            Default configuration dictionary
        """
        return {
            "max_bootstrapped_demos": 8,
            "max_labeled_demos": 8,
            "max_rounds": 1,
        }

    def _convert_to_dspy_examples(self, data: List[Any]) -> List[dspy.Example]:
        """Convert data examples to DSPy examples.

        Args:
            data: List of data examples

        Returns:
            List of DSPy examples
        """
        dspy_examples = []
        for example in data:
            if hasattr(example, 'input') and hasattr(example, 'label'):
                dspy_example = dspy.Example(
                    input=example.input,
                    output=example.label,
                )
                if hasattr(example, 'id'):
                    dspy_example.id = example.id
                dspy_examples.append(dspy_example)

        return dspy_examples

    def _evaluate_program(
        self,
        program: Any,
        valset: List[dspy.Example],
        metric_fn: callable,
    ) -> Dict[str, float]:
        """Evaluate program on validation set.

        Args:
            program: DSPy program to evaluate
            valset: Validation examples
            metric_fn: Metric function

        Returns:
            Dictionary of metrics
        """
        if not valset:
            return {"accuracy": 0.0}

        correct = 0
        total = len(valset)

        for example in valset:
            try:
                result = program(input=example.input)
                predicted = result.output if hasattr(result, 'output') else str(result)

                # Simple accuracy check - can be customized
                if metric_fn:
                    is_correct = metric_fn(example, predicted)
                else:
                    # Default: exact string match
                    is_correct = str(predicted).strip().lower() == str(example.output).strip().lower()

                if is_correct:
                    correct += 1
            except Exception:
                # Count as incorrect if prediction fails
                pass

        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy}

    def _extract_prompt(self, program: Any) -> str:
        """Extract the final prompt from optimized program.

        Args:
            program: Optimized DSPy program

        Returns:
            Final prompt string
        """
        # Try different ways to extract the prompt
        if hasattr(program, 'demos') and program.demos:
            # If the program has demonstrations, try to format them
            demos_text = []
            for demo in program.demos[:5]:  # Limit to first 5 demos
                if hasattr(demo, 'input') and hasattr(demo, 'output'):
                    demos_text.append(f"Input: {demo.input}\nOutput: {demo.output}")
            return "\n\n".join(demos_text)

        # Fallback to program string representation
        return str(program)