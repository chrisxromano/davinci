"""Tests for observability data models."""

import pytest
from datetime import datetime, UTC

from davinci.observability.models import (
    EvaluationLog,
    MinerResultLog,
    WindowPredictionLog,
)


class TestWindowPredictionLog:
    """Tests for WindowPredictionLog computed properties."""

    def test_correct_prediction(self) -> None:
        """Test correct property when prediction matches ground truth."""
        log = WindowPredictionLog(
            scan_id="scan-001",
            window_index=5,
            hotkey="5FTest",
            predicted_label=1,
            ground_truth_label=1,
        )

        assert log.correct is True

    def test_incorrect_prediction(self) -> None:
        """Test correct property when prediction doesn't match."""
        log = WindowPredictionLog(
            scan_id="scan-001",
            window_index=5,
            hotkey="5FTest",
            predicted_label=1,
            ground_truth_label=0,
        )

        assert log.correct is False

    def test_to_dict(self) -> None:
        """Test to_dict includes all fields."""
        log = WindowPredictionLog(
            scan_id="scan-001",
            window_index=3,
            hotkey="5FTest",
            predicted_label=0,
            ground_truth_label=0,
        )

        result = log.to_dict()

        assert result["scan_id"] == "scan-001"
        assert result["window_index"] == 3
        assert result["hotkey"] == "5FTest"
        assert result["predicted_label"] == 0
        assert result["ground_truth_label"] == 0
        assert result["correct"] is True


class TestMinerResultLog:
    """Tests for MinerResultLog serialization."""

    def test_to_dict_includes_all_metrics(self) -> None:
        """Test that to_dict includes all metric fields."""
        log = MinerResultLog(
            hotkey="5FTest",
            score=0.90,
            success=True,
            accuracy=0.85,
            precision=0.80,
            recall=0.90,
            f1=0.848,
            model_hash="abc123",
            inference_time_ms=1500.0,
            is_winner=True,
            is_copier=False,
        )

        result = log.to_dict()

        assert result["accuracy"] == 0.85
        assert result["precision"] == 0.8
        assert result["recall"] == 0.9
        assert result["f1"] == 0.848
        assert result["is_winner"] is True
        assert result["is_copier"] is False

    def test_to_dict_failed_miner_has_error(self) -> None:
        """Test that failed miner includes error message."""
        log = MinerResultLog(
            hotkey="5FTest",
            score=0.0,
            success=False,
            error="ModelCorruptedError: Invalid ONNX format",
        )

        result = log.to_dict()

        assert result["success"] is False
        assert result["error"] == "ModelCorruptedError: Invalid ONNX format"
        assert result["accuracy"] is None


class TestEvaluationLog:
    """Tests for EvaluationLog serialization."""

    def test_to_summary_dict_excludes_miner_results(self) -> None:
        """Test that miner_results are excluded from summary dict."""
        log = EvaluationLog(
            timestamp=datetime.now(UTC),
            evaluation_date="2025-01-15",
            validator_hotkey="5FValidator",
            netuid=1,
            dataset_size=1000,
            models_evaluated=50,
            models_succeeded=45,
            models_failed=5,
            winner_hotkey="5FWinner",
            winner_score=0.90,
            miner_results=[
                MinerResultLog(hotkey="5FA", score=0.90, success=True),
                MinerResultLog(hotkey="5FB", score=0.85, success=True),
            ],
        )

        result = log.to_summary_dict()

        assert "miner_results" not in result
        assert result["models_evaluated"] == 50
        assert result["winner_hotkey"] == "5FWinner"

    def test_to_summary_dict_formats_timestamp(self) -> None:
        """Test that timestamp is formatted as ISO string."""
        now = datetime(2025, 1, 15, 16, 0, 0, tzinfo=UTC)

        log = EvaluationLog(
            timestamp=now,
            evaluation_date="2025-01-15",
            validator_hotkey="5FValidator",
            netuid=1,
            dataset_size=1000,
            models_evaluated=50,
            models_succeeded=45,
            models_failed=5,
            winner_hotkey="5FWinner",
            winner_score=0.90,
        )

        result = log.to_summary_dict()

        assert result["timestamp"] == "2025-01-15T16:00:00+00:00"
