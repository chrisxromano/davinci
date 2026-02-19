#!/usr/bin/env python3
"""
Generate all test data needed for local validation testing.

Creates:
1. Evaluation dataset (.npz) — the validator input (GPR scan windows + labels)
2. Test ONNX model — a simple binary classifier for dry-run evaluation
3. Miner CLI test data — bundled test data for `davinci evaluate`

Usage:
    # Generate everything with defaults
    python scripts/generate_test_data.py

    # Custom output directory and dimensions
    python scripts/generate_test_data.py --output ./test_output --depth 32 --width 64 --num-windows 500

    # Generate only eval data (skip model)
    python scripts/generate_test_data.py --no-model

    # Generate with specific sample count and seed
    python scripts/generate_test_data.py --num-windows 500 --seed 123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# GPR Scan Window Generation
# ---------------------------------------------------------------------------

def generate_rebar_window(
    depth: int, width: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Synthetic GPR window with rebar present (label=1).

    Simulates a bright horizontal reflection band at a random depth
    with gaussian spread to mimic wave diffraction patterns.
    """
    window = rng.normal(0.0, 0.1, size=(depth, width)).astype(np.float32)

    # Bright horizontal band (rebar reflection)
    rebar_depth = rng.integers(depth // 4, 3 * depth // 4)
    band_intensity = rng.uniform(0.7, 1.0)
    window[rebar_depth, :] += band_intensity

    # Gaussian spread around band
    for offset in [-2, -1, 1, 2]:
        d = rebar_depth + offset
        if 0 <= d < depth:
            decay = 0.4 if abs(offset) == 1 else 0.15
            window[d, :] += band_intensity * decay

    # Add some vertical variation (rebar isn't perfectly uniform)
    col_noise = rng.normal(0.0, 0.05, size=width).astype(np.float32)
    window[rebar_depth, :] += col_noise

    return window


def generate_noise_window(
    depth: int, width: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Synthetic GPR window with no rebar (label=0).

    Just background noise + surface reflection typical of concrete scans.
    """
    window = rng.normal(0.0, 0.15, size=(depth, width)).astype(np.float32)

    # Surface reflection (always near the top)
    surface_depth = rng.integers(2, max(3, depth // 8))
    window[surface_depth, :] += rng.uniform(0.2, 0.4)

    # Occasional faint horizontal artifacts (not rebar)
    if rng.random() > 0.5:
        artifact_depth = rng.integers(depth // 3, 2 * depth // 3)
        window[artifact_depth, :] += rng.uniform(0.05, 0.15)

    return window


def generate_eval_dataset(
    num_windows: int,
    depth: int,
    width: int,
    positive_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a mock evaluation dataset matching the validator's expected format.

    Returns:
        Tuple of (windows, labels, scan_ids, window_indices) where:
        - windows: (N, depth, width) float32
        - labels: (N,) int32, values 0 or 1
        - scan_ids: (N,) string, scan identifiers
        - window_indices: (N,) int32, window index within scan
    """
    rng = np.random.default_rng(seed)

    num_positive = int(num_windows * positive_ratio)
    num_negative = num_windows - num_positive

    windows = []
    labels = []
    scan_ids = []
    window_indices = []

    windows_per_scan = 20

    for i in range(num_positive):
        windows.append(generate_rebar_window(depth, width, rng))
        labels.append(1)
        scan_ids.append(f"bridge_scan_{i // windows_per_scan:03d}")
        window_indices.append(i % windows_per_scan)

    for i in range(num_negative):
        windows.append(generate_noise_window(depth, width, rng))
        labels.append(0)
        idx = num_positive + i
        scan_ids.append(f"bridge_scan_{idx // windows_per_scan:03d}")
        window_indices.append(idx % windows_per_scan)

    # Shuffle to mix positives and negatives
    order = rng.permutation(num_windows)
    windows_arr = np.stack(windows)[order].astype(np.float32)
    labels_arr = np.array(labels, dtype=np.int32)[order]
    scan_ids_arr = np.array(scan_ids)[order]
    window_indices_arr = np.array(window_indices, dtype=np.int32)[order]

    return windows_arr, labels_arr, scan_ids_arr, window_indices_arr


def save_eval_npz(
    output_path: Path,
    windows: np.ndarray,
    labels: np.ndarray,
    scan_ids: np.ndarray,
    window_indices: np.ndarray,
) -> None:
    """Save evaluation dataset as .npz (validator-loadable format)."""
    np.savez(
        output_path,
        windows=windows,
        labels=labels,
        scan_ids=scan_ids,
        window_indices=window_indices,
    )


# ---------------------------------------------------------------------------
# Test ONNX Model Generation
# ---------------------------------------------------------------------------

def create_test_onnx_model(
    output_path: Path,
    depth: int,
    width: int,
    seed: int = 42,
) -> None:
    """
    Create a simple ONNX binary classifier for GPR scan windows.

    Architecture: Reshape(N,1,D,W) -> (N,D*W) -> MatMul -> Add -> Sigmoid -> (N,1)

    Input:  (batch, 1, depth, width)  float32
    Output: (batch, 1)                float32 (probabilities 0.0-1.0)

    Args:
        output_path: Where to save the .onnx file.
        depth: Expected depth dimension of input windows.
        width: Expected width dimension of input windows.
        seed: Random seed (different seeds -> different predictions).
    """
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        print("WARNING: onnx package not installed, skipping model generation.")
        print("  Install with: pip install onnx")
        return

    rng = np.random.default_rng(seed)
    n_flat = depth * width

    # Random weights (small init — sigmoid output will be ~0.5)
    weights = rng.standard_normal((n_flat, 1)).astype(np.float32) * 0.01
    bias = np.array([0.0], dtype=np.float32)

    # ONNX graph definition
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None, 1, depth, width]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 1]
    )

    shape_init = helper.make_tensor(
        "reshape_shape", TensorProto.INT64, [2], [-1, n_flat]
    )
    weight_init = helper.make_tensor(
        "weights", TensorProto.FLOAT, [n_flat, 1], weights.flatten().tolist()
    )
    bias_init = helper.make_tensor(
        "bias", TensorProto.FLOAT, [1], bias.tolist()
    )

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Reshape", ["input", "reshape_shape"], ["flat"]),
            helper.make_node("MatMul", ["flat", "weights"], ["matmul_out"]),
            helper.make_node("Add", ["matmul_out", "bias"], ["add_out"]),
            helper.make_node("Sigmoid", ["add_out"], ["output"]),
        ],
        name="gpr-test-classifier",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[shape_init, weight_init, bias_init],
    )

    model = helper.make_model(
        graph,
        producer_name="davinci-test-generator",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 9

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_eval_data(npz_path: Path) -> None:
    """Load and validate the generated .npz matches what the validator expects."""
    data = np.load(npz_path, allow_pickle=False)

    assert "windows" in data.files, "Missing 'windows' array"
    assert "labels" in data.files, "Missing 'labels' array"

    windows = data["windows"]
    labels = data["labels"]

    assert windows.ndim == 3, f"windows must be 3D (N,D,W), got {windows.ndim}D"
    assert labels.ndim == 1, f"labels must be 1D (N,), got {labels.ndim}D"
    assert len(windows) == len(labels), "windows/labels length mismatch"
    assert windows.dtype == np.float32, f"windows dtype must be float32, got {windows.dtype}"
    assert set(np.unique(labels)).issubset({0, 1}), "labels must be 0 or 1"

    print(f"  Validation passed: {npz_path.name}")


def verify_model(model_path: Path, depth: int, width: int) -> None:
    """Load and run a quick inference to verify the model works."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  WARNING: onnxruntime not installed, skipping model verification.")
        return

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Run a small test batch
    test_input = np.random.randn(5, 1, depth, width).astype(np.float32)
    outputs = session.run(None, {input_name: test_input})
    preds = outputs[0].flatten()

    assert len(preds) == 5, f"Expected 5 predictions, got {len(preds)}"
    assert not np.any(np.isnan(preds)), "Model produced NaN predictions"
    assert not np.any(np.isinf(preds)), "Model produced Inf predictions"

    print(f"  Verification passed: {model_path.name} (sample preds: {preds[:3].round(3)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate test data for local validation testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 200 windows, 32x64, eval data + model
  python scripts/generate_test_data.py

  # Larger dataset with custom dimensions
  python scripts/generate_test_data.py --num-windows 500 --depth 32 --width 64

  # Skip model generation
  python scripts/generate_test_data.py --no-model

  # Custom output directory
  python scripts/generate_test_data.py --output ./my_test_data
        """,
    )
    parser.add_argument(
        "--output", type=str, default="./eval_data",
        help="Output directory (default: ./eval_data)",
    )
    parser.add_argument(
        "--num-windows", type=int, default=200,
        help="Number of GPR scan windows (default: 200)",
    )
    parser.add_argument(
        "--depth", type=int, default=32,
        help="Depth dimension per window (default: 32)",
    )
    parser.add_argument(
        "--width", type=int, default=64,
        help="Width dimension per window (default: 64)",
    )
    parser.add_argument(
        "--positive-ratio", type=float, default=0.6,
        help="Fraction of windows with rebar present (default: 0.6)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-model", action="store_true",
        help="Skip ONNX test model generation",
    )
    parser.add_argument(
        "--update-miner-test-data", action="store_true",
        help="Also update davinci/miner_cli/test_data.npz",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Generate evaluation dataset (.npz)
    # -----------------------------------------------------------------------
    print(f"\n=== Generating evaluation dataset ===")
    print(f"  Windows: {args.num_windows} ({args.depth}x{args.width})")
    print(f"  Positive ratio: {args.positive_ratio:.0%}")
    print(f"  Seed: {args.seed}")

    windows, labels, scan_ids, window_indices = generate_eval_dataset(
        num_windows=args.num_windows,
        depth=args.depth,
        width=args.width,
        positive_ratio=args.positive_ratio,
        seed=args.seed,
    )

    eval_npz_path = output_dir / "mock_eval_data.npz"
    save_eval_npz(eval_npz_path, windows, labels, scan_ids, window_indices)

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    size_mb = eval_npz_path.stat().st_size / (1024 * 1024)

    print(f"  Saved: {eval_npz_path}")
    print(f"  Shape: ({len(labels)}, {args.depth}, {args.width})")
    print(f"  Labels: {n_pos} positive, {n_neg} negative")
    print(f"  Size: {size_mb:.2f} MB")

    verify_eval_data(eval_npz_path)

    # -----------------------------------------------------------------------
    # 2. Generate test ONNX model
    # -----------------------------------------------------------------------
    if not args.no_model:
        print(f"\n=== Generating test ONNX model ===")
        print(f"  Input shape: (batch, 1, {args.depth}, {args.width})")
        print(f"  Output shape: (batch, 1)")

        model_path = output_dir / "test_model.onnx"
        create_test_onnx_model(
            output_path=model_path,
            depth=args.depth,
            width=args.width,
            seed=args.seed,
        )

        if model_path.exists():
            model_size_kb = model_path.stat().st_size / 1024
            print(f"  Saved: {model_path} ({model_size_kb:.1f} KB)")
            verify_model(model_path, args.depth, args.width)

    # -----------------------------------------------------------------------
    # 3. Update miner CLI test data (optional)
    # -----------------------------------------------------------------------
    if args.update_miner_test_data:
        print(f"\n=== Updating miner CLI test data ===")
        miner_test_path = Path(__file__).parent.parent / "davinci" / "miner_cli" / "test_data.npz"
        save_eval_npz(miner_test_path, windows, labels, scan_ids, window_indices)
        print(f"  Saved: {miner_test_path}")
        verify_eval_data(miner_test_path)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n=== Done ===")
    print(f"  Eval data:  {eval_npz_path}")
    if not args.no_model:
        model_path = output_dir / "test_model.onnx"
        if model_path.exists():
            print(f"  Test model: {model_path}")

    print(f"\nTo test the validator locally:")
    print(f"  1. Set EVAL_DATA_DIR={output_dir} in your .env")
    print(f"  2. Start pylon + validator as normal")
    print(f"  3. The validator will pick up {eval_npz_path.name} on next eval cycle")

    if not args.no_model:
        print(f"\nTo test a model against this data:")
        print(f"  uv run python -m davinci.miner_cli evaluate {output_dir}/test_model.onnx")


if __name__ == "__main__":
    main()
