"""File-based evaluation data loader.

The subnet owner drops pre-processed evaluation data as .npz files
into a configured directory. The validator watches this directory
and loads new data when it appears.

NPZ file format:
    - "windows": float32 array of shape (N, depth, width)
    - "labels": int32 array of shape (N,) with values 0 or 1
    - "scan_ids": array of strings, shape (N,)
    - "window_indices": int32 array of shape (N,)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .errors import EvaluationDataLoadError, EvaluationDataNotFoundError
from .models import EvaluationDataset, GPRScanWindow

logger = logging.getLogger(__name__)


class EvaluationDataLoader:
    """
    Loads evaluation datasets from .npz files in a directory.

    The subnet owner places .npz files in the data directory.
    The loader picks the most recently modified file.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load_latest(self) -> EvaluationDataset:
        """
        Load the most recent evaluation dataset from the data directory.

        Returns:
            EvaluationDataset with all windows from the latest .npz file.

        Raises:
            EvaluationDataNotFoundError: If no .npz files exist in data_dir.
            EvaluationDataLoadError: If the file is malformed.
        """
        if not self.data_dir.exists():
            raise EvaluationDataNotFoundError(
                f"Data directory does not exist: {self.data_dir}"
            )

        npz_files = sorted(
            self.data_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True
        )

        if not npz_files:
            raise EvaluationDataNotFoundError(
                f"No .npz files found in {self.data_dir}"
            )

        latest = npz_files[0]
        logger.info(f"Loading evaluation data from {latest.name}")

        return self._load_file(latest)

    def _load_file(self, path: Path) -> EvaluationDataset:
        """Load a single .npz file into an EvaluationDataset."""
        try:
            data = np.load(path, allow_pickle=False)
        except Exception as e:
            raise EvaluationDataLoadError(
                f"Failed to load {path.name}: {e}"
            ) from e

        required_keys = {"windows", "labels"}
        missing = required_keys - set(data.files)
        if missing:
            raise EvaluationDataLoadError(
                f"Missing required keys in {path.name}: {missing}. "
                f"Found: {data.files}"
            )

        windows_array = data["windows"]  # (N, depth, width)
        labels_array = data["labels"]  # (N,)

        if windows_array.ndim != 3:
            raise EvaluationDataLoadError(
                f"'windows' must be 3D (N, depth, width), got shape {windows_array.shape}"
            )

        if labels_array.ndim != 1:
            raise EvaluationDataLoadError(
                f"'labels' must be 1D (N,), got shape {labels_array.shape}"
            )

        if len(windows_array) != len(labels_array):
            raise EvaluationDataLoadError(
                f"Length mismatch: {len(windows_array)} windows vs {len(labels_array)} labels"
            )

        # Optional metadata arrays
        scan_ids = (
            data["scan_ids"] if "scan_ids" in data.files
            else np.array([path.stem] * len(labels_array))
        )
        window_indices = (
            data["window_indices"] if "window_indices" in data.files
            else np.arange(len(labels_array))
        )

        scan_windows = []
        for i in range(len(labels_array)):
            scan_windows.append(
                GPRScanWindow(
                    scan_id=str(scan_ids[i]),
                    window_index=int(window_indices[i]),
                    data=windows_array[i].astype(np.float32),
                    label=int(labels_array[i]),
                )
            )

        dataset = EvaluationDataset(windows=scan_windows)
        logger.info(f"Loaded {dataset}")
        return dataset
