"""Main evaluator for calculating metrics on model predictions."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from rich.console import Console
from rich.table import Table

from dspy_bench.data.schema import DatasetExample


class MetricResult:
    """Container for metric calculation results."""

    def __init__(self, name: str, value: float, details: Optional[Dict[str, Any]] = None):
        """Initialize metric result.

        Args:
            name: Metric name
            value: Metric value
            details: Additional details about the metric
        """
        self.name = name
        self.value = value
        self.details = details or {}

    def __str__(self) -> str:
        """String representation of metric result."""
        return f"{self.name}: {self.value:.4f}"


class EvaluationResult:
    """Container for complete evaluation results."""

    def __init__(
        self,
        metrics: Dict[str, MetricResult],
        predictions: List[Dict[str, Any]],
        dataset_size: int,
        evaluation_time: float = 0.0,
    ):
        """Initialize evaluation result.

        Args:
            metrics: Dictionary of metric results
            predictions: List of predictions with details
            dataset_size: Size of evaluated dataset
            evaluation_time: Time taken for evaluation in seconds
        """
        self.metrics = metrics
        self.predictions = predictions
        self.dataset_size = dataset_size
        self.evaluation_time = evaluation_time

    def get_metric(self, name: str) -> Optional[MetricResult]:
        """Get a specific metric result.

        Args:
            name: Metric name

        Returns:
            Metric result or None if not found
        """
        return self.metrics.get(name)

    def summary_dict(self) -> Dict[str, Any]:
        """Get summary dictionary of results.

        Returns:
            Summary dictionary
        """
        return {
            "metrics": {name: result.value for name, result in self.metrics.items()},
            "dataset_size": self.dataset_size,
            "evaluation_time": self.evaluation_time,
            "predictions_count": len(self.predictions),
        }

    def print_summary(self) -> None:
        """Print a formatted summary of evaluation results."""
        console = Console()

        # Create metrics table
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for name, result in self.metrics.items():
            table.add_row(name, f"{result.value:.4f}")

        console.print(table)

        # Print additional info
        console.print(f"\nDataset size: {self.dataset_size}")
        console.print(f"Evaluation time: {self.evaluation_time:.2f}s")


class Evaluator:
    """Main evaluator for calculating metrics on model predictions."""

    def __init__(self, metrics: Union[List[str], Dict[str, Callable]]):
        """Initialize evaluator.

        Args:
            metrics: List of metric names or dict of metric functions
        """
        self.console = Console()
        self._metric_functions = self._prepare_metrics(metrics)

    def _prepare_metrics(self, metrics: Union[List[str], Dict[str, Callable]]) -> Dict[str, Callable]:
        """Prepare metric functions.

        Args:
            metrics: Metric names or functions

        Returns:
            Dictionary of metric functions
        """
        from dspy_bench.metrics.standard import (
            accuracy, f1, precision, recall, exact_match,
            classification_metrics, string_similarity,
        )

        # Standard metric mappings
        standard_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": exact_match,
            "classification": classification_metrics,
            "string_similarity": string_similarity,
        }

        metric_functions = {}

        if isinstance(metrics, list):
            for metric_name in metrics:
                if metric_name in standard_metrics:
                    metric_functions[metric_name] = standard_metrics[metric_name]
                else:
                    self.console.print(
                        f"Warning: Unknown metric '{metric_name}'. Available: {list(standard_metrics.keys())}",
                        style="yellow"
                    )
        elif isinstance(metrics, dict):
            metric_functions.update(metrics)
        else:
            raise ValueError("metrics must be a list of metric names or dict of metric functions")

        return metric_functions

    def evaluate(
        self,
        predictions: List[Any],
        ground_truth: List[Any],
        inputs: Optional[List[Any]] = None,
        task_type: str = "classification",
    ) -> EvaluationResult:
        """Evaluate predictions against ground truth.

        Args:
            predictions: Model predictions
            ground_truth: Ground truth labels
            inputs: Original inputs (optional, for some metrics)
            task_type: Type of task (classification, regression, generation)

        Returns:
            Evaluation result
        """
        import time

        start_time = time.time()

        if len(predictions) != len(ground_truth):
            raise ValueError(f"Predictions ({len(predictions)}) and ground truth ({len(ground_truth)}) must have same length")

        # Prepare prediction details
        prediction_details = []
        for i, (pred, true) in enumerate(zip(predictions, ground_truth)):
            detail = {
                "index": i,
                "prediction": pred,
                "ground_truth": true,
            }
            if inputs:
                detail["input"] = inputs[i]
            prediction_details.append(detail)

        # Calculate metrics
        calculated_metrics = {}

        for metric_name, metric_func in self._metric_functions.items():
            try:
                if metric_name == "classification":
                    # For classification metrics, use the comprehensive function
                    result_values = metric_func(ground_truth, predictions)
                    for sub_name, value in result_values.items():
                        calculated_metrics[sub_name] = MetricResult(sub_name, float(value))
                elif metric_name == "string_similarity":
                    # For string similarity, return multiple metrics
                    result_values = metric_func(ground_truth, predictions)
                    for sub_name, value in result_values.items():
                        calculated_metrics[sub_name] = MetricResult(sub_name, float(value))
                else:
                    # For simple metrics
                    value = metric_func(ground_truth, predictions)
                    calculated_metrics[metric_name] = MetricResult(metric_name, float(value))

            except Exception as e:
                self.console.print(f"Warning: Failed to calculate {metric_name}: {e}", style="yellow")
                calculated_metrics[metric_name] = MetricResult(metric_name, 0.0, {"error": str(e)})

        evaluation_time = time.time() - start_time

        return EvaluationResult(
            metrics=calculated_metrics,
            predictions=prediction_details,
            dataset_size=len(ground_truth),
            evaluation_time=evaluation_time,
        )

    def evaluate_dataset(
        self,
        dataset: List[DatasetExample],
        prediction_func: Callable[[str], Any],
        input_key: str = "input",
        label_key: str = "label",
    ) -> EvaluationResult:
        """Evaluate on a dataset using a prediction function.

        Args:
            dataset: List of dataset examples
            prediction_func: Function that takes input and returns prediction
            input_key: Key to extract input from examples
            label_key: Key to extract label from examples

        Returns:
            Evaluation result
        """
        self.console.print(f"Evaluating on {len(dataset)} examples...")

        inputs = []
        ground_truth = []
        predictions = []

        for example in dataset:
            input_text = getattr(example, input_key)
            label = getattr(example, label_key)

            try:
                prediction = prediction_func(input_text)
                predictions.append(prediction)
            except Exception as e:
                self.console.print(f"Warning: Prediction failed for example: {e}", style="yellow")
                predictions.append(None)

            inputs.append(input_text)
            ground_truth.append(label)

        return self.evaluate(predictions, ground_truth, inputs)

    def save_results(self, result: EvaluationResult, output_path: Union[str, Path]) -> None:
        """Save evaluation results to file.

        Args:
            result: Evaluation result to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare save data
        save_data = {
            "summary": result.summary_dict(),
            "metrics": {name: {"value": result.value, "details": result.details}
                       for name, result in result.metrics.items()},
            "predictions": result.predictions,
        }

        with output_path.open("w") as f:
            json.dump(save_data, f, indent=2, default=str)

        self.console.print(f"✓ Results saved to {output_path}", style="green")