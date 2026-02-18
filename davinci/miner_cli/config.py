"""
Miner CLI configuration.

This module provides access to:
- Test data for local evaluation (GPR scan windows in .npz format)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .errors import ConfigurationError

# Default test data bundled with the package
_DEFAULT_TEST_DATA_PATH = Path(__file__).parent / "test_data.npz"

# Chain constraints
MAX_REPO_ID_BYTES = 51  # Maximum bytes for HF repo ID in commitment


def get_test_data(
    test_data_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load test GPR scan data for local model evaluation.

    Args:
        test_data_path: Path to .npz file with test data.
            If None, uses the bundled default test data.

    Returns:
        Tuple of (features, ground_truth) where:
        - features: (N, 1, depth, width) float32 array of GPR scan windows
        - ground_truth: (N,) int32 array of labels (0 or 1)

    Raises:
        ConfigurationError: If test data file is not found or invalid.
    """
    path = Path(test_data_path) if test_data_path else _DEFAULT_TEST_DATA_PATH

    if not path.exists():
        raise ConfigurationError(
            f"Test data not found: {path}. "
            "Generate test data with: python scripts/generate_mock_eval_data.py"
        )

    try:
        data = np.load(path)
    except Exception as e:
        raise ConfigurationError(f"Failed to load test data: {e}") from e

    if "windows" not in data or "labels" not in data:
        raise ConfigurationError(
            f"Invalid test data format in {path}. "
            "Expected .npz with 'windows' and 'labels' arrays."
        )

    windows = data["windows"].astype(np.float32)
    labels = data["labels"].astype(np.int32)

    # Reshape to (N, 1, depth, width) for ONNX model input
    if windows.ndim == 3:
        windows = windows[:, np.newaxis, :, :]

    return windows, labels
