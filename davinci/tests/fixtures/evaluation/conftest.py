"""Fixtures for evaluation integration tests."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

FIXTURES_DIR = Path(__file__).parent

# Default GPR scan dimensions for test models
DEFAULT_DEPTH = 32
DEFAULT_WIDTH = 64


def create_test_model(
    output_path: Path,
    depth: int = DEFAULT_DEPTH,
    width: int = DEFAULT_WIDTH,
    seed: int = 42,
) -> None:
    """
    Create a simple ONNX binary classifier for GPR scan windows.

    Architecture: Reshape(N,1,D,W) → (N,D*W) → MatMul → Sigmoid → (N,1)

    Different seeds produce different weights → different predictions.

    Args:
        output_path: Where to save the model
        depth: Expected input depth dimension
        width: Expected input width dimension
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    n_flat = depth * width
    weights = np.random.randn(n_flat, 1).astype(np.float32) * 0.01
    bias = np.array([0.0], dtype=np.float32)

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None, 1, depth, width]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 1]
    )

    # Reshape target shape: (N, D*W) — use -1 for batch dim
    shape_init = helper.make_tensor(
        "reshape_shape", TensorProto.INT64, [2], [-1, n_flat]
    )

    weight_init = helper.make_tensor(
        "weights", TensorProto.FLOAT, [n_flat, 1], weights.flatten().tolist()
    )
    bias_init = helper.make_tensor("bias", TensorProto.FLOAT, [1], bias.tolist())

    reshape_node = helper.make_node(
        "Reshape", inputs=["input", "reshape_shape"], outputs=["flat"]
    )
    matmul_node = helper.make_node(
        "MatMul", inputs=["flat", "weights"], outputs=["matmul_out"]
    )
    add_node = helper.make_node(
        "Add", inputs=["matmul_out", "bias"], outputs=["add_out"]
    )
    sigmoid_node = helper.make_node(
        "Sigmoid", inputs=["add_out"], outputs=["output"]
    )

    graph = helper.make_graph(
        nodes=[reshape_node, matmul_node, add_node, sigmoid_node],
        name="gpr-binary-classifier",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[shape_init, weight_init, bias_init],
    )

    model = helper.make_model(
        graph,
        producer_name="davinci-test",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 9

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))


def create_bad_model(output_path: Path) -> None:
    """
    Create a model that produces NaN outputs (for error testing).

    Takes GPR 4D input, produces NaN via 0/0.
    """
    depth, width = DEFAULT_DEPTH, DEFAULT_WIDTH
    n_flat = depth * width

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [None, 1, depth, width]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [None, 1]
    )

    shape_init = helper.make_tensor(
        "reshape_shape", TensorProto.INT64, [2], [-1, n_flat]
    )
    weights = np.ones((n_flat, 1), dtype=np.float32)
    weight_init = helper.make_tensor(
        "weights", TensorProto.FLOAT, [n_flat, 1], weights.flatten().tolist()
    )
    zero_init = helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])

    reshape_node = helper.make_node(
        "Reshape", inputs=["input", "reshape_shape"], outputs=["flat"]
    )
    matmul_node = helper.make_node(
        "MatMul", inputs=["flat", "weights"], outputs=["matmul_out"]
    )
    div_node = helper.make_node(
        "Div", inputs=["zero", "zero"], outputs=["nan_scalar"]
    )
    mul_node = helper.make_node(
        "Mul", inputs=["matmul_out", "nan_scalar"], outputs=["output"]
    )

    graph = helper.make_graph(
        nodes=[reshape_node, matmul_node, div_node, mul_node],
        name="nan-model",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[shape_init, weight_init, zero_init],
    )

    model = helper.make_model(
        graph,
        producer_name="davinci-test",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 9

    onnx.save(model, str(output_path))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the fixtures directory path."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def dummy_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create and return path to a dummy GPR classifier model."""
    model_dir = tmp_path_factory.mktemp("models")
    model_path = model_dir / "gpr_model.onnx"
    create_test_model(output_path=model_path)
    return model_path


@pytest.fixture(scope="session")
def dummy_input_data() -> np.ndarray:
    """Create sample GPR input (100 samples, 1 channel, 32 depth, 64 width)."""
    np.random.seed(123)
    return np.random.randn(100, 1, DEFAULT_DEPTH, DEFAULT_WIDTH).astype(np.float32)


@pytest.fixture(scope="session")
def dummy_ground_truth() -> np.ndarray:
    """Create sample binary ground truth labels."""
    np.random.seed(456)
    return np.random.randint(0, 2, size=100).astype(np.int32)
