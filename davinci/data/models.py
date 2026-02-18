"""Data models for GPR bridge scan evaluation datasets."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GPRScanWindow:
    """
    A single pre-processed GPR scan window for model evaluation.

    The data pipeline delivers pre-processed windows — models receive
    these directly as input. No raw DZT parsing happens in the subnet.

    Attributes:
        scan_id: Identifier for the source scan (e.g. filename stem).
        window_index: Index of this window within the source scan.
        data: Pre-processed 2D array (depth_samples x horizontal_samples), float32.
        label: Ground truth — 1 if rebar present in this window, 0 otherwise.
    """

    scan_id: str
    window_index: int
    data: np.ndarray  # shape: (depth, width), dtype float32
    label: int  # 0 or 1


@dataclass
class EvaluationDataset:
    """
    Evaluation dataset of pre-processed GPR scan windows with ground truth labels.

    Provided by the subnet owner as a set of files that the validator loads.
    """

    windows: list[GPRScanWindow] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.windows)

    def __repr__(self) -> str:
        n_pos = sum(1 for w in self.windows if w.label == 1)
        return (
            f"EvaluationDataset({len(self.windows)} windows, "
            f"{n_pos} positive, {len(self.windows) - n_pos} negative)"
        )

    @property
    def ground_truth(self) -> np.ndarray:
        """Extract ground truth labels as int array."""
        return np.array([w.label for w in self.windows], dtype=np.int32)

    @property
    def input_batch(self) -> np.ndarray:
        """
        Stack all window data into a batch tensor.

        Returns:
            np.ndarray of shape (N, depth, width), dtype float32
        """
        return np.stack([w.data for w in self.windows]).astype(np.float32)
