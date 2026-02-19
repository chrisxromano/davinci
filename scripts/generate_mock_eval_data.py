#!/usr/bin/env python3
"""
Generate mock GPR evaluation data for demo/testing.

Creates .npz files that simulate pre-processed GPR scan windows
with binary rebar presence labels.

Usage:
    python scripts/generate_mock_eval_data.py
    python scripts/generate_mock_eval_data.py --output ./eval_data --num-windows 200
    python scripts/generate_mock_eval_data.py --depth 32 --width 64 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np


def generate_rebar_window(
    depth: int, width: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate a synthetic GPR window that looks like it contains rebar.

    Rebar appears as a bright horizontal band at some depth, with
    hyperbolic reflections (simplified as a gaussian-blurred line).
    """
    window = rng.normal(0.0, 0.1, size=(depth, width)).astype(np.float32)

    # Add a bright horizontal band (rebar reflection) at a random depth
    rebar_depth = rng.integers(depth // 4, 3 * depth // 4)
    band_intensity = rng.uniform(0.7, 1.0)
    window[rebar_depth, :] += band_intensity

    # Add slight gaussian spread around the band (simulates wave diffraction)
    for offset in [-1, 1]:
        d = rebar_depth + offset
        if 0 <= d < depth:
            window[d, :] += band_intensity * 0.4

    return window


def generate_noise_window(
    depth: int, width: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate a synthetic GPR window with no rebar (just noise/concrete).
    """
    window = rng.normal(0.0, 0.15, size=(depth, width)).astype(np.float32)

    # Add some low-frequency background (surface reflection, etc.)
    surface_depth = rng.integers(2, depth // 8)
    window[surface_depth, :] += rng.uniform(0.2, 0.4)

    return window


def generate_mock_dataset(
    num_windows: int,
    depth: int,
    width: int,
    positive_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a full mock evaluation dataset.

    Returns:
        Tuple of (windows, labels, scan_ids, window_indices)
    """
    rng = np.random.default_rng(seed)

    num_positive = int(num_windows * positive_ratio)
    num_negative = num_windows - num_positive

    windows = []
    labels = []
    scan_ids = []
    window_indices = []

    # Generate positive windows (rebar present)
    for i in range(num_positive):
        windows.append(generate_rebar_window(depth, width, rng))
        labels.append(1)
        scan_ids.append(f"bridge_scan_{i // 20:03d}")
        window_indices.append(i % 20)

    # Generate negative windows (no rebar)
    for i in range(num_negative):
        windows.append(generate_noise_window(depth, width, rng))
        labels.append(0)
        scan_ids.append(f"bridge_scan_{(num_positive + i) // 20:03d}")
        window_indices.append((num_positive + i) % 20)

    # Shuffle
    order = rng.permutation(num_windows)
    windows_arr = np.stack(windows)[order].astype(np.float32)
    labels_arr = np.array(labels, dtype=np.int32)[order]
    scan_ids_arr = np.array(scan_ids)[order]
    window_indices_arr = np.array(window_indices, dtype=np.int32)[order]

    return windows_arr, labels_arr, scan_ids_arr, window_indices_arr


def main():
    parser = argparse.ArgumentParser(description="Generate mock GPR evaluation data")
    parser.add_argument(
        "--output", type=str, default="./eval_data",
        help="Output directory for .npz file",
    )
    parser.add_argument(
        "--num-windows", type=int, default=200,
        help="Number of GPR scan windows to generate",
    )
    parser.add_argument(
        "--depth", type=int, default=32,
        help="Depth dimension of each window (radio wave time samples)",
    )
    parser.add_argument(
        "--width", type=int, default=64,
        help="Width dimension of each window (horizontal samples)",
    )
    parser.add_argument(
        "--positive-ratio", type=float, default=0.6,
        help="Fraction of windows containing rebar (0.0 to 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_windows} mock GPR windows ({args.depth}x{args.width})...")

    windows, labels, scan_ids, window_indices = generate_mock_dataset(
        num_windows=args.num_windows,
        depth=args.depth,
        width=args.width,
        positive_ratio=args.positive_ratio,
        seed=args.seed,
    )

    output_path = output_dir / "mock_eval_data.npz"
    np.savez(
        output_path,
        windows=windows,
        labels=labels,
        scan_ids=scan_ids,
        window_indices=window_indices,
    )

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"Saved to {output_path}")
    print(f"  Windows: {len(labels)} ({n_pos} positive, {n_neg} negative)")
    print(f"  Shape: ({len(labels)}, {args.depth}, {args.width})")
    print(f"  File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
