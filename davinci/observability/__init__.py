"""
Observability module for validator logging and monitoring.

This module provides WandB integration for logging evaluation results,
miner performance, and per-window predictions.

Usage:
    from davinci.observability import create_wandb_logger

    # Create logger
    logger = create_wandb_logger(
        project="davinci-subnet",
        validator_hotkey="5F...",
        enabled=True,
    )

    # Start run (once per validator session)
    logger.start_run()

    # Log each evaluation
    logger.log_evaluation(validation_result, dataset)

    # Finish when done
    logger.finish()

Logged Data:
    - Summary metrics (scalars): winner info, model counts, timing
    - Miner results table: per-miner scores, metrics, errors
    - Window predictions table: for dashboard joining with scan_id/window_index
"""

from .models import (
    EvaluationLog,
    MinerResultLog,
    WandbConfig,
    WindowPredictionLog,
)
from .wandb_logger import WandbLogger, create_wandb_logger

__all__ = [
    # Main entry point
    "create_wandb_logger",
    # Classes
    "WandbLogger",
    "WandbConfig",
    # Log models
    "EvaluationLog",
    "MinerResultLog",
    "WindowPredictionLog",
]
