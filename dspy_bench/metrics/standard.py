"""Standard evaluation metrics for dspy-bench."""

import re
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


def accuracy(y_true: List[Any], y_pred: List[Any]) -> float:
    """Calculate accuracy score.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def precision(y_true: List[Any], y_pred: List[Any], average: str = "binary") -> float:
    """Calculate precision score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('binary', 'micro', 'macro', 'weighted')

    Returns:
        Precision score
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def recall(y_true: List[Any], y_pred: List[Any], average: str = "binary") -> float:
    """Calculate recall score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('binary', 'micro', 'macro', 'weighted')

    Returns:
        Recall score
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def f1(y_true: List[Any], y_pred: List[Any], average: str = "binary") -> float:
    """Calculate F1 score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('binary', 'micro', 'macro', 'weighted')

    Returns:
        F1 score
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def precision_recall_f1(y_true: List[Any], y_pred: List[Any], average: str = "binary") -> Dict[str, float]:
    """Calculate precision, recall, and F1 scores.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def exact_match(y_true: List[str], y_pred: List[str]) -> float:
    """Calculate exact match accuracy for string outputs.

    Args:
        y_true: True strings
        y_pred: Predicted strings

    Returns:
        Exact match accuracy
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    matches = [1 if str(true).strip().lower() == str(pred).strip().lower() else 0
               for true, pred in zip(y_true, y_pred)]
    return sum(matches) / len(matches)


def bleu_score(y_true: List[str], y_pred: List[str]) -> float:
    """Calculate BLEU score for text generation.

    Args:
        y_true: List of reference texts
        y_pred: List of generated texts

    Returns:
        Average BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize
    except ImportError:
        raise ImportError("NLTK required for BLEU score. Install with: pip install nltk")

    smoothing = SmoothingFunction().method4
    scores = []

    for ref, pred in zip(y_true, y_pred):
        ref_tokens = [word_tokenize(str(ref).lower())]
        pred_tokens = word_tokenize(str(pred).lower())

        try:
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
            scores.append(score)
        except:
            scores.append(0.0)

    return np.mean(scores) if scores else 0.0


def rouge_scores(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores for text generation.

    Args:
        y_true: List of reference texts
        y_pred: List of generated texts

    Returns:
        Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L scores
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("rouge-score required for ROUGE scores. Install with: pip install rouge-score")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for ref, pred in zip(y_true, y_pred):
        scores = scorer.score(str(ref), str(pred))
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        "rouge1": np.mean(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": np.mean(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": np.mean(rougeL_scores) if rougeL_scores else 0.0,
    }


def classification_metrics(y_true: List[Any], y_pred: List[Any]) -> Dict[str, float]:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with various classification metrics
    """
    metrics = {
        "accuracy": accuracy(y_true, y_pred),
    }

    # Try to calculate precision, recall, F1
    try:
        prf_metrics = precision_recall_f1(y_true, y_pred, average="weighted")
        metrics.update(prf_metrics)
    except Exception:
        # If it fails (e.g., for multiclass issues), try macro averaging
        try:
            prf_metrics = precision_recall_f1(y_true, y_pred, average="macro")
            metrics.update(prf_metrics)
        except Exception:
            pass

    return metrics


def regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Calculate regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with regression metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def multilabel_metrics(y_true: List[List[Any]], y_pred: List[List[Any]]) -> Dict[str, float]:
    """Calculate multi-label classification metrics.

    Args:
        y_true: True labels (list of lists)
        y_pred: Predicted labels (list of lists)

    Returns:
        Dictionary with multi-label metrics
    """
    # Convert to binary format
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    return {
        "accuracy": accuracy(y_true_bin, y_pred_bin),
        "precision_micro": precision(y_true_bin, y_pred_bin, average="micro"),
        "recall_micro": recall(y_true_bin, y_pred_bin, average="micro"),
        "f1_micro": f1(y_true_bin, y_pred_bin, average="micro"),
        "precision_macro": precision(y_true_bin, y_pred_bin, average="macro"),
        "recall_macro": recall(y_true_bin, y_pred_bin, average="macro"),
        "f1_macro": f1(y_true_bin, y_pred_bin, average="macro"),
    }


def string_similarity(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Calculate various string similarity metrics.

    Args:
        y_true: True strings
        y_pred: Predicted strings

    Returns:
        Dictionary with similarity metrics
    """
    try:
        from difflib import SequenceMatcher
        from Levenshtein import distance as levenshtein_distance
    except ImportError:
        raise ImportError("python-Levenshtein required for string similarity. Install with: pip install python-Levenshtein")

    exact_matches = []
    sequence_similarities = []
    levenshtein_similarities = []

    for true, pred in zip(y_true, y_pred):
        true_str, pred_str = str(true), str(pred)

        # Exact match
        exact_matches.append(1 if true_str.strip().lower() == pred_str.strip().lower() else 0)

        # Sequence matcher similarity
        seq_sim = SequenceMatcher(None, true_str.lower(), pred_str.lower()).ratio()
        sequence_similarities.append(seq_sim)

        # Levenshtein similarity
        max_len = max(len(true_str), len(pred_str))
        if max_len == 0:
            lev_sim = 1.0
        else:
            lev_dist = levenshtein_distance(true_str, pred_str)
            lev_sim = 1 - (lev_dist / max_len)
        levenshtein_similarities.append(lev_sim)

    return {
        "exact_match": np.mean(exact_matches),
        "sequence_similarity": np.mean(sequence_similarities),
        "levenshtein_similarity": np.mean(levenshtein_similarities),
    }