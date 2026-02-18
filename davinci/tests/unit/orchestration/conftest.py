"""Shared fixtures for orchestration unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from davinci.evaluation.models import EvaluationBatch, EvaluationResult, PredictionMetrics

if TYPE_CHECKING:
    from collections.abc import Sequence


def create_eval_result(
    hotkey: str,
    *,
    score: float = 0.9,
    success: bool = True,
    predictions: np.ndarray | None = None,
) -> EvaluationResult:
    """
    Create an EvaluationResult for testing.

    Args:
        hotkey: Model hotkey identifier.
        score: Model score (0-1). Used as F1 score directly.
        success: Whether evaluation succeeded.
        predictions: Optional predictions array. Defaults to [1, 0, 1].

    Returns:
        EvaluationResult configured for testing.
    """
    if predictions is None:
        predictions = np.array([1, 0, 1])

    metrics = PredictionMetrics(
        accuracy=score,
        precision=score,
        recall=score,
        f1=score,
        tp=int(score * 10),
        tn=int(score * 10),
        fp=int((1 - score) * 10),
        fn=int((1 - score) * 10),
        n_samples=len(predictions),
    ) if success else None

    return EvaluationResult(
        hotkey=hotkey,
        predictions=predictions if success else None,
        metrics=metrics,
        error=None if success else Exception("Eval failed"),
        inference_time_ms=100.0,
        model_hash="abc123",
    )


def create_eval_batch(
    results: Sequence[EvaluationResult],
    dataset_size: int = 10,
) -> EvaluationBatch:
    """Create an EvaluationBatch from results."""
    return EvaluationBatch(results=list(results), dataset_size=dataset_size)


def create_chain_metadata(
    hotkey: str,
    block_number: int = 1000,
    model_hash: str = "abc123",
) -> MagicMock:
    """
    Create mock ChainModelMetadata.

    Args:
        hotkey: Model hotkey identifier.
        block_number: Block number when model was committed.
        model_hash: Model file hash.

    Returns:
        MagicMock configured as ChainModelMetadata.
    """
    metadata = MagicMock()
    metadata.hotkey = hotkey
    metadata.block_number = block_number
    metadata.model_hash = model_hash
    return metadata


def create_dataset(size: int = 10) -> MagicMock:
    """
    Create mock EvaluationDataset.

    Args:
        size: Number of samples in dataset.

    Returns:
        MagicMock configured as EvaluationDataset.
    """
    dataset = MagicMock()
    dataset.input_batch = np.random.randn(size, 32, 64).astype(np.float32)
    dataset.ground_truth = np.random.randint(0, 2, size=size).astype(np.int32)
    dataset.__len__ = MagicMock(return_value=size)
    return dataset


def create_duplicate_result(
    copier_hotkeys: frozenset[str] | None = None,
    num_groups: int = 0,
) -> MagicMock:
    """
    Create mock DuplicateDetectionResult.

    Args:
        copier_hotkeys: Set of hotkeys identified as copiers.
        num_groups: Number of duplicate groups detected.

    Returns:
        MagicMock configured as DuplicateDetectionResult.
    """
    result = MagicMock()
    result.copier_hotkeys = copier_hotkeys or frozenset()
    result.groups = [MagicMock() for _ in range(num_groups)]
    return result


def create_winner_result(
    winner_hotkey: str,
    winner_score: float,
    winner_block: int = 100,
) -> MagicMock:
    """
    Create mock WinnerSelectionResult.

    Args:
        winner_hotkey: Hotkey of the winner.
        winner_score: Score of the winner.
        winner_block: Block number of winner's commitment.

    Returns:
        MagicMock configured as WinnerSelectionResult.
    """
    result = MagicMock()
    result.winner_hotkey = winner_hotkey
    result.winner_score = winner_score
    result.winner_block = winner_block
    return result


def create_weights(
    weights: dict[str, float],
    total: float = 1.0,
) -> MagicMock:
    """
    Create mock IncentiveWeights.

    Args:
        weights: Mapping of hotkey to weight.
        total: Total of all weights.

    Returns:
        MagicMock configured as IncentiveWeights.
    """
    result = MagicMock()
    result.weights = weights
    result.total = total
    # get_weight returns the weight for the given hotkey, defaulting to highest
    result.get_weight = lambda hk: weights.get(hk, max(weights.values()) if weights else 0)
    return result


# Pytest fixtures for common mock dependencies


