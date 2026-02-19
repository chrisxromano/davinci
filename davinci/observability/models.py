"""Data models for observability and WandB logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MinerResultLog:
    """
    Log entry for a single miner's evaluation result.

    Used for per-miner tracking in WandB tables.
    """

    hotkey: str
    score: float
    success: bool

    # Metrics (only if successful)
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None

    # Metadata
    model_hash: str | None = None
    hf_repo_id: str | None = None
    inference_time_ms: float | None = None
    is_winner: bool = False
    is_copier: bool = False

    # Error info (only if failed)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WandB logging."""
        return {
            "hotkey": self.hotkey,
            "score": round(self.score, 6) if self.score else 0.0,
            "success": self.success,
            "accuracy": round(self.accuracy, 6) if self.accuracy is not None else None,
            "precision": round(self.precision, 6) if self.precision is not None else None,
            "recall": round(self.recall, 6) if self.recall is not None else None,
            "f1": round(self.f1, 6) if self.f1 is not None else None,
            "model_hash": self.model_hash,
            "hf_repo_id": self.hf_repo_id,
            "inference_time_ms": (
                round(self.inference_time_ms, 2)
                if self.inference_time_ms is not None
                else None
            ),
            "is_winner": self.is_winner,
            "is_copier": self.is_copier,
            "error": self.error,
        }


@dataclass
class WindowPredictionLog:
    """
    Log entry for a single GPR window prediction.

    Used for dashboard joining - allows matching predictions
    to windows in the evaluation dataset.
    """

    # Window identifier
    scan_id: str
    window_index: int

    # Miner info
    hotkey: str

    # Prediction data
    predicted_label: int  # 0 or 1
    ground_truth_label: int  # 0 or 1

    @property
    def correct(self) -> bool:
        return self.predicted_label == self.ground_truth_label

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for WandB table row."""
        return {
            "scan_id": self.scan_id,
            "window_index": self.window_index,
            "hotkey": self.hotkey,
            "predicted_label": self.predicted_label,
            "ground_truth_label": self.ground_truth_label,
            "correct": self.correct,
        }


@dataclass
class EvaluationLog:
    """
    Complete log entry for a validation round.

    Contains summary metrics and references to detailed tables.
    """

    # Timestamp
    timestamp: datetime
    evaluation_date: str  # YYYY-MM-DD format for easy filtering

    # Validator info
    validator_hotkey: str
    netuid: int

    # Dataset info
    dataset_size: int

    # Evaluation summary
    models_evaluated: int
    models_succeeded: int
    models_failed: int

    # Winner info
    winner_hotkey: str
    winner_score: float
    winner_f1: float | None = None
    winner_block: int | None = None

    # Anti-cheat summary
    duplicate_groups_found: int = 0
    copiers_detected: int = 0

    # Timing
    total_evaluation_time_ms: float = 0.0

    # Detailed results (logged as WandB table)
    miner_results: list[MinerResultLog] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for WandB scalar logging.

        Does not include miner_results - those go to a separate table.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "evaluation_date": self.evaluation_date,
            "validator_hotkey": self.validator_hotkey,
            "netuid": self.netuid,
            "dataset_size": self.dataset_size,
            "models_evaluated": self.models_evaluated,
            "models_succeeded": self.models_succeeded,
            "models_failed": self.models_failed,
            "winner_hotkey": self.winner_hotkey,
            "winner_score": round(self.winner_score, 6),
            "winner_f1": (
                round(self.winner_f1, 6) if self.winner_f1 is not None else None
            ),
            "winner_block": self.winner_block,
            "duplicate_groups_found": self.duplicate_groups_found,
            "copiers_detected": self.copiers_detected,
            "total_evaluation_time_ms": round(self.total_evaluation_time_ms, 2),
        }


@dataclass
class WandbConfig:
    """Configuration for WandB logging."""

    # Project settings
    project: str = "davinci-evaluations"
    entity: str | None = None  # WandB team/user, None = default

    # Authentication
    api_key: str | None = None  # WandB API key, or set WANDB_API_KEY env var

    # Run settings
    run_name: str | None = None  # Auto-generated if None
    tags: list[str] = field(default_factory=list)

    # Feature flags
    enabled: bool = True
    offline: bool = False  # Run in offline mode

    # What to log
    log_miner_table: bool = True  # Log per-miner results table
    log_predictions_table: bool = (
        False  # Log per-window predictions (disabled by default)
    )

    # Prediction logging settings
    # Cap predictions to top N miners by score (to limit data volume).
    # None = log all miners (default).
    predictions_top_n_miners: int | None = None
