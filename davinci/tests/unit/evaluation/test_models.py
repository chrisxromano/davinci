"""Unit tests for evaluation models."""

import numpy as np
import pytest

from davinci.evaluation.models import (
    EvaluationBatch,
    EvaluationResult,
    PredictionMetrics,
)


class TestPredictionMetrics:
    """Tests for PredictionMetrics dataclass."""

    def test_score_equals_f1(self) -> None:
        """Score should equal F1."""
        metrics = PredictionMetrics(
            accuracy=0.8, precision=0.75, recall=0.9, f1=0.818,
            tp=9, tn=7, fp=3, fn=1, n_samples=20,
        )
        assert metrics.score == 0.818

    def test_to_dict_serialization(self) -> None:
        """to_dict returns correct keys and rounded values."""
        metrics = PredictionMetrics(
            accuracy=0.85, precision=0.80, recall=0.90, f1=0.848484,
            tp=9, tn=8, fp=2, fn=1, n_samples=20,
        )
        result = metrics.to_dict()

        assert result["accuracy"] == 0.85
        assert result["precision"] == 0.8
        assert result["recall"] == 0.9
        assert result["f1"] == 0.848484
        assert result["tp"] == 9
        assert result["tn"] == 8
        assert result["n_samples"] == 20


class TestEvaluationBatch:
    """Tests for EvaluationBatch dataclass."""

    @pytest.fixture
    def mixed_batch(self) -> EvaluationBatch:
        """Create batch with mixed success/failure results."""
        return EvaluationBatch(
            results=[
                EvaluationResult(
                    hotkey="hotkey1",
                    predictions=np.array([1, 0, 1]),
                    metrics=PredictionMetrics(
                        accuracy=0.9, precision=0.85, recall=0.95, f1=0.898,
                        tp=19, tn=17, fp=3, fn=1, n_samples=40,
                    ),
                ),
                EvaluationResult(
                    hotkey="hotkey2",
                    predictions=np.array([1, 1, 0]),
                    metrics=PredictionMetrics(
                        accuracy=0.8, precision=0.75, recall=0.85, f1=0.797,
                        tp=17, tn=15, fp=5, fn=3, n_samples=40,
                    ),
                ),
                EvaluationResult(
                    hotkey="hotkey3",
                    error=RuntimeError("Docker failed"),
                ),
            ],
            dataset_size=100,
            total_time_ms=500.0,
        )

    def test_get_ranking(self, mixed_batch: EvaluationBatch) -> None:
        """get_ranking returns sorted by score descending."""
        ranking = mixed_batch.get_ranking()

        assert len(ranking) == 2
        assert ranking[0] == ("hotkey1", 0.898)  # f1
        assert ranking[1] == ("hotkey2", 0.797)  # f1

    def test_get_best(self, mixed_batch: EvaluationBatch) -> None:
        """get_best returns highest scoring result."""
        best = mixed_batch.get_best()

        assert best is not None
        assert best.hotkey == "hotkey1"
        assert best.score == 0.898

    def test_get_best_empty_batch(self) -> None:
        """get_best returns None for empty batch."""
        batch = EvaluationBatch()
        assert batch.get_best() is None

    def test_get_best_all_failed(self) -> None:
        """get_best returns None when all failed."""
        batch = EvaluationBatch(
            results=[
                EvaluationResult(hotkey="h1", error=Exception("fail")),
                EvaluationResult(hotkey="h2", error=Exception("fail")),
            ]
        )
        assert batch.get_best() is None
