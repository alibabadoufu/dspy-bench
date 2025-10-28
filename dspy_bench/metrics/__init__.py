"""Metrics and evaluation system for dspy-bench."""

from dspy_bench.metrics.evaluator import Evaluator, EvaluationResult, MetricResult
from dspy_bench.metrics.standard import (
    accuracy,
    bleu_score,
    classification_metrics,
    exact_match,
    f1,
    multilabel_metrics,
    precision,
    recall,
    regression_metrics,
    rouge_scores,
    string_similarity,
)

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "MetricResult",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "exact_match",
    "classification_metrics",
    "regression_metrics",
    "multilabel_metrics",
    "string_similarity",
    "bleu_score",
    "rouge_scores",
]