"""Tests for EvaluationDataLoader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from davinci.data import (
    EvaluationDataLoader,
    EvaluationDataLoadError,
    EvaluationDataNotFoundError,
)


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    return tmp_path / "eval_data"


def _make_npz(
    path: Path,
    num_windows: int = 10,
    depth: int = 32,
    width: int = 64,
    seed: int = 42,
) -> Path:
    """Create a valid .npz eval data file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    windows = rng.normal(size=(num_windows, depth, width)).astype(np.float32)
    labels = rng.integers(0, 2, size=num_windows).astype(np.int32)
    scan_ids = np.array([f"scan_{i}" for i in range(num_windows)])
    window_indices = np.arange(num_windows, dtype=np.int32)
    np.savez(path, windows=windows, labels=labels, scan_ids=scan_ids, window_indices=window_indices)
    return path


class TestEvaluationDataLoader:
    def test_load_latest_success(self, tmp_data_dir: Path):
        _make_npz(tmp_data_dir / "eval_001.npz", num_windows=10)
        loader = EvaluationDataLoader(tmp_data_dir)
        dataset = loader.load_latest()
        assert len(dataset) == 10
        assert dataset.ground_truth.shape == (10,)
        assert dataset.input_batch.shape == (10, 32, 64)

    def test_load_picks_most_recent(self, tmp_data_dir: Path):
        import time

        _make_npz(tmp_data_dir / "old.npz", num_windows=5, seed=1)
        time.sleep(0.05)
        _make_npz(tmp_data_dir / "new.npz", num_windows=15, seed=2)

        loader = EvaluationDataLoader(tmp_data_dir)
        dataset = loader.load_latest()
        assert len(dataset) == 15

    def test_no_data_dir_raises(self, tmp_data_dir: Path):
        loader = EvaluationDataLoader(tmp_data_dir)
        with pytest.raises(EvaluationDataNotFoundError, match="does not exist"):
            loader.load_latest()

    def test_empty_dir_raises(self, tmp_data_dir: Path):
        tmp_data_dir.mkdir(parents=True)
        loader = EvaluationDataLoader(tmp_data_dir)
        with pytest.raises(EvaluationDataNotFoundError, match="No .npz files"):
            loader.load_latest()

    def test_bad_npz_raises(self, tmp_data_dir: Path):
        tmp_data_dir.mkdir(parents=True)
        bad_file = tmp_data_dir / "bad.npz"
        bad_file.write_text("not a real npz")

        loader = EvaluationDataLoader(tmp_data_dir)
        with pytest.raises(EvaluationDataLoadError):
            loader.load_latest()

    def test_missing_keys_raises(self, tmp_data_dir: Path):
        tmp_data_dir.mkdir(parents=True)
        # Save npz with wrong keys
        np.savez(tmp_data_dir / "missing.npz", foo=np.array([1, 2, 3]))

        loader = EvaluationDataLoader(tmp_data_dir)
        with pytest.raises(EvaluationDataLoadError, match="Missing required keys"):
            loader.load_latest()

    def test_shape_mismatch_raises(self, tmp_data_dir: Path):
        tmp_data_dir.mkdir(parents=True)
        # 2D windows instead of 3D
        np.savez(
            tmp_data_dir / "flat.npz",
            windows=np.zeros((10, 64), dtype=np.float32),
            labels=np.zeros(10, dtype=np.int32),
        )

        loader = EvaluationDataLoader(tmp_data_dir)
        with pytest.raises(EvaluationDataLoadError, match="must be 3D"):
            loader.load_latest()

    def test_length_mismatch_raises(self, tmp_data_dir: Path):
        tmp_data_dir.mkdir(parents=True)
        np.savez(
            tmp_data_dir / "mismatch.npz",
            windows=np.zeros((10, 32, 64), dtype=np.float32),
            labels=np.zeros(5, dtype=np.int32),
        )

        loader = EvaluationDataLoader(tmp_data_dir)
        with pytest.raises(EvaluationDataLoadError, match="Length mismatch"):
            loader.load_latest()

    def test_optional_metadata(self, tmp_data_dir: Path):
        """Without scan_ids/window_indices, loader should use defaults."""
        tmp_data_dir.mkdir(parents=True)
        np.savez(
            tmp_data_dir / "minimal.npz",
            windows=np.zeros((5, 32, 64), dtype=np.float32),
            labels=np.array([0, 1, 1, 0, 1], dtype=np.int32),
        )

        loader = EvaluationDataLoader(tmp_data_dir)
        dataset = loader.load_latest()
        assert len(dataset) == 5
        assert dataset.ground_truth.sum() == 3

    def test_loads_fixture_data(self):
        """Load the pre-generated test fixture."""
        fixture_path = Path(__file__).parent.parent.parent / "fixtures" / "data"
        if not (fixture_path / "mock_eval_data.npz").exists():
            pytest.skip("Test fixture not generated")

        loader = EvaluationDataLoader(fixture_path)
        dataset = loader.load_latest()
        assert len(dataset) == 20
        assert dataset.input_batch.shape == (20, 32, 64)
