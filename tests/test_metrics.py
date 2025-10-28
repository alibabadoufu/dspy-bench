"""Tests for metrics and evaluation system."""

import pytest

from dspy_bench.metrics import (
    Evaluator,
    EvaluationResult,
    MetricResult,
    accuracy,
    exact_match,
    precision,
    recall,
    f1,
)


class TestMetricResult:
    """Test metric result container."""

    def test_metric_result_creation(self):
        """Test creating metric result."""
        result = MetricResult("accuracy", 0.85, {"details": "test"})
        assert result.name == "accuracy"
        assert result.value == 0.85
        assert result.details["details"] == "test"

    def test_metric_result_str(self):
        """Test metric result string representation."""
        result = MetricResult("accuracy", 0.85)
        str_result = str(result)
        assert "accuracy" in str_result
        assert "0.8500" in str_result


class TestEvaluationResult:
    """Test evaluation result container."""

    def test_evaluation_result_creation(self):
        """Test creating evaluation result."""
        metrics = {
            "accuracy": MetricResult("accuracy", 0.85),
            "f1": MetricResult("f1", 0.82),
        }
        predictions = [{"index": 0, "prediction": "A", "ground_truth": "A"}]

        result = EvaluationResult(
            metrics=metrics,
            predictions=predictions,
            dataset_size=10,
            evaluation_time=5.2,
        )

        assert len(result.metrics) == 2
        assert result.dataset_size == 10
        assert result.evaluation_time == 5.2
        assert len(result.predictions) == 1

    def test_get_metric(self):
        """Test getting specific metric."""
        metrics = {
            "accuracy": MetricResult("accuracy", 0.85),
            "f1": MetricResult("f1", 0.82),
        }
        result = EvaluationResult(metrics, [], 10)

        accuracy_metric = result.get_metric("accuracy")
        assert accuracy_metric is not None
        assert accuracy_metric.value == 0.85

        missing_metric = result.get_metric("nonexistent")
        assert missing_metric is None

    def test_summary_dict(self):
        """Test summary dictionary generation."""
        metrics = {
            "accuracy": MetricResult("accuracy", 0.85),
            "f1": MetricResult("f1", 0.82),
        }
        result = EvaluationResult(metrics, [], 10, evaluation_time=5.2)

        summary = result.summary_dict()
        assert summary["metrics"]["accuracy"] == 0.85
        assert summary["metrics"]["f1"] == 0.82
        assert summary["dataset_size"] == 10
        assert summary["evaluation_time"] == 5.2
        assert summary["predictions_count"] == 0


class TestStandardMetrics:
    """Test standard metric functions."""

    def test_accuracy(self):
        """Test accuracy calculation."""
        y_true = [1, 2, 1, 2, 1]
        y_pred = [1, 2, 2, 2, 1]
        acc = accuracy(y_true, y_pred)
        assert acc == 0.8  # 4 out of 5 correct

    def test_perfect_accuracy(self):
        """Test perfect accuracy."""
        y_true = [1, 2, 3]
        y_pred = [1, 2, 3]
        acc = accuracy(y_true, y_pred)
        assert acc == 1.0

    def test_zero_accuracy(self):
        """Test zero accuracy."""
        y_true = [1, 2, 3]
        y_pred = [3, 2, 1]  # Only 1 correct (middle)
        acc = accuracy(y_true, y_pred)
        assert acc == 1/3

    def test_exact_match_strings(self):
        """Test exact match for strings."""
        y_true = ["cat", "dog", "bird"]
        y_pred = ["cat", "Dog", "bird"]  # Case difference in middle
        em = exact_match(y_true, y_pred)
        assert em == 2/3  # 2 out of 3 match exactly

    def test_exact_match_case_insensitive(self):
        """Test exact match is case insensitive."""
        y_true = ["Cat", "DOG", "Bird"]
        y_pred = ["cat", "dog", "bird"]
        em = exact_match(y_true, y_pred)
        assert em == 1.0  # All should match (case insensitive)

    def test_precision(self):
        """Test precision calculation."""
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 0, 1]
        prec = precision(y_true, y_pred)
        # True positives: 2 (indices 0, 4), False positives: 0
        assert prec == 1.0

    def test_recall(self):
        """Test recall calculation."""
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 0, 1]
        rec = recall(y_true, y_pred)
        # True positives: 2, False negatives: 1 (index 1)
        assert rec == 2/3

    def test_f1_score(self):
        """Test F1 score calculation."""
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 0, 1]
        f1_score = f1(y_true, y_pred)
        # Precision = 1.0, Recall = 2/3, so F1 = 2 * 1.0 * 2/3 / (1.0 + 2/3) = 4/5
        assert abs(f1_score - 0.8) < 0.01

    def test_different_length_inputs(self):
        """Test metrics with different length inputs."""
        y_true = [1, 2, 3]
        y_pred = [1, 2]  # Shorter
        with pytest.raises(ValueError):
            accuracy(y_true, y_pred)


class TestEvaluator:
    """Test evaluator class."""

    def test_evaluator_creation(self):
        """Test creating evaluator."""
        evaluator = Evaluator(["accuracy", "exact_match"])
        assert len(evaluator._metric_functions) == 2
        assert "accuracy" in evaluator._metric_functions
        assert "exact_match" in evaluator._metric_functions

    def test_evaluator_with_unknown_metric(self):
        """Test evaluator with unknown metric."""
        evaluator = Evaluator(["accuracy", "nonexistent_metric"])
        # Should only have the known metric
        assert len(evaluator._metric_functions) == 1
        assert "accuracy" in evaluator._metric_functions
        assert "nonexistent_metric" not in evaluator._metric_functions

    def test_evaluator_evaluate(self):
        """Test evaluation."""
        evaluator = Evaluator(["accuracy", "exact_match"])
        predictions = ["A", "B", "A", "C"]
        ground_truth = ["A", "C", "A", "C"]
        inputs = ["input1", "input2", "input3", "input4"]

        result = evaluator.evaluate(predictions, ground_truth, inputs)

        assert isinstance(result, EvaluationResult)
        assert len(result.metrics) == 2
        assert "accuracy" in result.metrics
        assert "exact_match" in result.metrics
        assert result.dataset_size == 4
        assert len(result.predictions) == 4

    def test_evaluator_evaluate_no_inputs(self):
        """Test evaluation without inputs."""
        evaluator = Evaluator(["accuracy"])
        predictions = ["A", "B", "A"]
        ground_truth = ["A", "C", "A"]

        result = evaluator.evaluate(predictions, ground_truth)

        assert result.dataset_size == 3
        assert len(result.predictions) == 3
        # Predictions should not have input field
        assert "input" not in result.predictions[0]

    def test_evaluator_mismatched_lengths(self):
        """Test evaluation with mismatched lengths."""
        evaluator = Evaluator(["accuracy"])
        predictions = ["A", "B"]
        ground_truth = ["A", "B", "C"]

        with pytest.raises(ValueError, match="must have same length"):
            evaluator.evaluate(predictions, ground_truth)

    def test_evaluator_save_results(self):
        """Test saving evaluation results."""
        evaluator = Evaluator(["accuracy"])
        predictions = ["A", "B"]
        ground_truth = ["A", "C"]

        result = evaluator.evaluate(predictions, ground_truth)

        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            evaluator.save_results(result, temp_path)

            # Load and verify
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)

            assert "summary" in saved_data
            assert "metrics" in saved_data
            assert "predictions" in saved_data
            assert saved_data["summary"]["dataset_size"] == 2

        finally:
            import os
            os.unlink(temp_path)