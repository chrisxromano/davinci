"""
Classification metrics for GPR rebar detection models.

Metrics:
- Accuracy: Fraction of correct predictions
- Precision: TP / (TP + FP) — of predicted positives, how many are correct
- Recall: TP / (TP + FN) — of actual positives, how many were found
- F1: Harmonic mean of precision and recall

Scoring convention:
    Score = F1, so higher F1 = higher score.
    F1 is chosen as the primary score because it balances precision and recall,
    which is critical for rebar detection (missing rebar is dangerous,
    false positives waste inspection time).
"""

import numpy as np

from .errors import EmptyDatasetError, MetricsError
from .models import PredictionMetrics


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> PredictionMetrics:
    """
    Calculate classification metrics for binary rebar detection.

    Args:
        y_true: Ground truth labels (1D int array, values 0 or 1)
        y_pred: Predicted labels (1D int array, values 0 or 1)

    Returns:
        PredictionMetrics with all computed values

    Raises:
        MetricsError: If arrays have different lengths
        EmptyDatasetError: If arrays are empty
    """
    y_true = np.asarray(y_true, dtype=np.int32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.int32).flatten()

    if len(y_true) != len(y_pred):
        raise MetricsError(
            f"Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    if len(y_true) == 0:
        raise EmptyDatasetError("Empty input arrays")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return PredictionMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        n_samples=len(y_true),
    )


def validate_predictions(
    predictions: np.ndarray,
    expected_length: int | None = None,
) -> np.ndarray:
    """
    Validate and normalize prediction array for binary classification.

    Checks for:
    - Correct shape (1D or column vector)
    - No NaN or Inf values
    - Values are 0 or 1 (or thresholdable floats)
    - Correct length if expected_length provided

    Args:
        predictions: Raw predictions from model
        expected_length: Expected number of predictions (optional)

    Returns:
        Validated 1D numpy array of int32 (0 or 1)

    Raises:
        MetricsError: If predictions are invalid
    """
    predictions = np.asarray(predictions, dtype=np.float64)

    # Flatten if needed (handle (N,1) shape)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    elif predictions.ndim != 1:
        raise MetricsError(
            f"Invalid prediction shape: {predictions.shape}. Expected 1D or (N,1)."
        )

    # Check length
    if expected_length is not None and len(predictions) != expected_length:
        raise MetricsError(
            f"Prediction count mismatch: got {len(predictions)}, expected {expected_length}"
        )

    # Check for NaN/Inf
    if np.any(np.isnan(predictions)):
        nan_count = np.sum(np.isnan(predictions))
        raise MetricsError(f"Predictions contain {nan_count} NaN values")

    if np.any(np.isinf(predictions)):
        inf_count = np.sum(np.isinf(predictions))
        raise MetricsError(f"Predictions contain {inf_count} Inf values")

    # Threshold float predictions to binary (0/1)
    # Models may output probabilities — threshold at 0.5
    binary = (predictions >= 0.5).astype(np.int32)

    return binary
