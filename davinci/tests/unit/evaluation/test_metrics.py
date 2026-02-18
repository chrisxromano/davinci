"""Unit tests for classification metrics."""

import numpy as np
import pytest

from davinci.evaluation import (
    EmptyDatasetError,
    MetricsError,
    calculate_metrics,
    validate_predictions,
)


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should have all metrics at 1.0."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.score == 1.0

    def test_all_wrong(self) -> None:
        """All-wrong predictions should have 0 accuracy."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0

    def test_confusion_matrix_counts(self) -> None:
        """TP, TN, FP, FN should be counted correctly."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        # TP=2 (idx 0,4), TN=2 (idx 2,5), FP=1 (idx 3), FN=1 (idx 1)

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics.tp == 2
        assert metrics.tn == 2
        assert metrics.fp == 1
        assert metrics.fn == 1
        assert metrics.n_samples == 6

    def test_precision_calculation(self) -> None:
        """Precision = TP / (TP + FP)."""
        y_true = np.array([1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0])
        # TP=1, FP=2 -> precision = 1/3

        metrics = calculate_metrics(y_true, y_pred)

        assert pytest.approx(metrics.precision, rel=0.01) == 1 / 3

    def test_recall_calculation(self) -> None:
        """Recall = TP / (TP + FN)."""
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 0])
        # TP=1, FN=2 -> recall = 1/3

        metrics = calculate_metrics(y_true, y_pred)

        assert pytest.approx(metrics.recall, rel=0.01) == 1 / 3

    def test_f1_calculation(self) -> None:
        """F1 = 2 * precision * recall / (precision + recall)."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        # TP=2, FP=1, FN=1 -> P=2/3, R=2/3, F1=2/3

        metrics = calculate_metrics(y_true, y_pred)

        assert pytest.approx(metrics.f1, rel=0.01) == 2 / 3

    def test_accuracy_calculation(self) -> None:
        """Accuracy = (TP + TN) / N."""
        y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0])
        # Correct: idx 0,1,2,3,6,7,8 = 7/10

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics.accuracy == 0.7

    def test_no_positive_predictions(self) -> None:
        """When no positive predictions, precision and recall should be 0."""
        y_true = np.array([1, 1, 0])
        y_pred = np.array([0, 0, 0])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1 == 0.0

    def test_no_actual_positives(self) -> None:
        """When no actual positives, recall is 0."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 0, 0])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics.recall == 0.0

    def test_score_equals_f1(self) -> None:
        """Score should be equal to F1."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics.score == metrics.f1

    def test_single_sample(self) -> None:
        """Should handle single-sample input."""
        metrics = calculate_metrics(np.array([1]), np.array([1]))
        assert metrics.accuracy == 1.0
        assert metrics.n_samples == 1


class TestCalculateMetricsErrors:
    """Tests for error handling in calculate_metrics."""

    def test_length_mismatch_raises_error(self) -> None:
        """Mismatched array lengths should raise MetricsError."""
        y_true = np.array([1, 0])
        y_pred = np.array([1])

        with pytest.raises(MetricsError, match="length mismatch"):
            calculate_metrics(y_true, y_pred)

    def test_empty_arrays_raise_error(self) -> None:
        """Empty arrays should raise EmptyDatasetError."""
        y_true = np.array([])
        y_pred = np.array([])

        with pytest.raises(EmptyDatasetError):
            calculate_metrics(y_true, y_pred)


class TestValidatePredictions:
    """Tests for prediction validation."""

    def test_valid_binary_array(self) -> None:
        """Valid binary array should pass validation."""
        predictions = np.array([0, 1, 1, 0])
        result = validate_predictions(predictions)

        assert np.array_equal(result, predictions)

    def test_float_probabilities_thresholded(self) -> None:
        """Float predictions should be thresholded at 0.5."""
        predictions = np.array([0.1, 0.9, 0.4, 0.6])
        result = validate_predictions(predictions)

        assert np.array_equal(result, [0, 1, 0, 1])

    def test_column_vector_flattened(self) -> None:
        """Column vector (N,1) should be flattened."""
        predictions = np.array([[0.8], [0.2], [0.6]])
        result = validate_predictions(predictions)

        assert result.shape == (3,)
        assert np.array_equal(result, [1, 0, 1])

    def test_expected_length_match(self) -> None:
        """Correct length should pass validation."""
        predictions = np.array([0, 1, 1])
        result = validate_predictions(predictions, expected_length=3)

        assert len(result) == 3

    def test_expected_length_mismatch(self) -> None:
        """Wrong length should raise MetricsError."""
        predictions = np.array([0, 1])

        with pytest.raises(MetricsError, match="count mismatch"):
            validate_predictions(predictions, expected_length=5)

    def test_nan_values_raise_error(self) -> None:
        """NaN values should raise MetricsError."""
        predictions = np.array([0.5, np.nan, 0.8])

        with pytest.raises(MetricsError, match="NaN"):
            validate_predictions(predictions)

    def test_inf_values_raise_error(self) -> None:
        """Inf values should raise MetricsError."""
        predictions = np.array([0.5, np.inf, 0.8])

        with pytest.raises(MetricsError, match="Inf"):
            validate_predictions(predictions)

    def test_invalid_shape_raises_error(self) -> None:
        """2D array with multiple columns should raise MetricsError."""
        predictions = np.array([[0.5, 0.8], [0.3, 0.7]])

        with pytest.raises(MetricsError, match="Invalid prediction shape"):
            validate_predictions(predictions)

    def test_boundary_threshold(self) -> None:
        """Values exactly at 0.5 should be classified as positive."""
        predictions = np.array([0.5, 0.49999])
        result = validate_predictions(predictions)

        assert result[0] == 1
        assert result[1] == 0
