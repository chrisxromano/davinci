"""
Evaluation module for running ONNX models and calculating metrics.

This module provides:
- Docker-based isolated model execution
- Binary classification metrics (Accuracy, Precision, Recall, F1)
- Evaluation orchestration for multiple models
- Result aggregation and ranking

Usage:
    from davinci.evaluation import (
        EvaluationOrchestrator,
        calculate_metrics,
        create_orchestrator,
    )

    # Quick metrics calculation
    metrics = calculate_metrics(y_true, y_pred)
    print(f"F1: {metrics.f1:.2%}, Accuracy: {metrics.accuracy:.2%}")

    # Full evaluation with Docker
    orchestrator = create_orchestrator()
    batch = await orchestrator.evaluate_all(models, features, ground_truth)
    best = batch.get_best()
"""

from .docker_runner import DockerConfig, DockerRunner, InferenceResult
from .errors import (
    DockerError,
    DockerExecutionError,
    DockerImageError,
    DockerNotAvailableError,
    EmptyDatasetError,
    EvaluationError,
    InferenceTimeoutError,
    InvalidPredictionError,
    MetricsError,
)
from .metrics import (
    calculate_metrics,
    validate_predictions,
)
from .models import EvaluationBatch, EvaluationResult, PredictionMetrics
from .orchestrator import (
    EvaluationOrchestrator,
    OrchestratorConfig,
    create_orchestrator,
)

__all__ = [
    # Factory (main entry points)
    "create_orchestrator",
    "calculate_metrics",
    # Orchestrator
    "EvaluationOrchestrator",
    "OrchestratorConfig",
    # Docker
    "DockerRunner",
    "DockerConfig",
    "InferenceResult",
    # Metrics
    "validate_predictions",
    # Models
    "PredictionMetrics",
    "EvaluationResult",
    "EvaluationBatch",
    # Errors
    "EvaluationError",
    "MetricsError",
    "EmptyDatasetError",
    "DockerError",
    "DockerNotAvailableError",
    "DockerImageError",
    "DockerExecutionError",
    "InferenceTimeoutError",
    "InvalidPredictionError",
]
