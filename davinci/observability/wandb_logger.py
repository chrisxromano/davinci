"""WandB logging for validator evaluations."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .models import (
    EvaluationLog,
    MinerResultLog,
    WandbConfig,
    WindowPredictionLog,
)

if TYPE_CHECKING:
    import wandb
    from davinci.data.models import EvaluationDataset
    from davinci.orchestration.models import ValidationResult

logger = logging.getLogger(__name__)


class WandbLogger:
    """
    WandB logger for validator evaluation results.

    Handles:
    - Initialization of WandB run
    - Logging evaluation summaries as scalars
    - Logging per-miner results as tables
    - Logging per-window predictions as tables (for dashboard)

    Usage:
        config = WandbConfig(project="davinci-subnet")
        wandb_logger = WandbLogger(config, validator_hotkey="5F...")

        # Start a run (typically once per validator session)
        wandb_logger.start_run()

        # Log each evaluation round
        wandb_logger.log_evaluation(
            result=validation_result,
            dataset=validation_dataset,
        )

        # Finish when done
        wandb_logger.finish()
    """

    def __init__(
        self,
        config: WandbConfig,
        validator_hotkey: str,
        netuid: int = 0,
    ):
        """
        Initialize WandB logger.

        Args:
            config: WandB configuration
            validator_hotkey: Validator's SS58 address
            netuid: Subnet UID
        """
        self._config = config
        self._validator_hotkey = validator_hotkey
        self._netuid = netuid
        self._run: wandb.sdk.wandb_run.Run | None = None
        self._wandb: Any = None  # Lazy import

    def _import_wandb(self) -> Any:
        """Lazy import wandb to avoid dependency if disabled."""
        if self._wandb is None:
            try:
                import wandb

                self._wandb = wandb
            except ImportError as e:
                logger.error("wandb not installed. Install with: uv add wandb")
                raise ImportError(
                    "wandb is required for WandbLogger. Install with: uv add wandb"
                ) from e
        return self._wandb

    @property
    def is_enabled(self) -> bool:
        """Check if logging is enabled."""
        return self._config.enabled

    @property
    def is_running(self) -> bool:
        """Check if a run is currently active."""
        return self._run is not None

    def start_run(self, run_name: str | None = None) -> None:
        """
        Start a new WandB run.

        Args:
            run_name: Optional run name override
        """
        if not self._config.enabled:
            logger.info("WandB logging is disabled")
            return

        # Set API key if provided (wandb also checks WANDB_API_KEY env var)
        if self._config.api_key:
            import os

            os.environ["WANDB_API_KEY"] = self._config.api_key

        wandb = self._import_wandb()

        # Generate run name if not provided
        if run_name is None:
            run_name = self._config.run_name
        if run_name is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            run_name = f"validator-{self._validator_hotkey[:8]}-{timestamp}"

        # Prepare tags
        tags = list(self._config.tags)
        tags.append(f"netuid-{self._netuid}")
        tags.append(f"validator-{self._validator_hotkey[:8]}")

        # Initialize run
        mode = "offline" if self._config.offline else "online"

        self._run = wandb.init(
            project=self._config.project,
            entity=self._config.entity,
            name=run_name,
            tags=tags,
            config={
                "validator_hotkey": self._validator_hotkey,
                "netuid": self._netuid,
            },
            mode=mode,
            resume="allow",  # Allow resuming if run exists
        )

        logger.info(f"WandB run started: {self._run.name} ({self._run.url})")

    def finish(self) -> None:
        """Finish the current WandB run."""
        if self._run is not None:
            self._run.finish()
            logger.info("WandB run finished")
            self._run = None

    def log_evaluation(
        self,
        result: ValidationResult,
        dataset: EvaluationDataset,
        download_failures: dict[str, str] | None = None,
    ) -> None:
        """
        Log a complete evaluation round to WandB.

        Args:
            result: Validation result from orchestrator
            dataset: Evaluation dataset with GPR windows
            download_failures: Dict mapping hotkey to error message for miners
                              that failed during download (license, hash, etc.)
        """
        if not self._config.enabled:
            return

        if self._run is None:
            logger.warning("WandB run not started. Call start_run() first.")
            return

        try:
            # Build evaluation log
            eval_log = self._build_evaluation_log(result, download_failures)

            # Log summary metrics
            self._run.log(eval_log.to_summary_dict())

            # Log miner results table
            if self._config.log_miner_table:
                self._log_miner_table(eval_log.miner_results)

            # Log predictions table (for dashboard)
            if self._config.log_predictions_table:
                self._log_predictions_table(
                    result=result,
                    dataset=dataset,
                )

            logger.info(
                f"Logged evaluation: winner={eval_log.winner_hotkey[:8]}..., "
                f"score={eval_log.winner_score:.4f}, "
                f"models={eval_log.models_succeeded}/{eval_log.models_evaluated}"
            )

        except Exception as e:
            logger.error(f"Failed to log evaluation to WandB: {e}", exc_info=True)
            # Don't raise - logging failures shouldn't break validation

    def _build_evaluation_log(
        self,
        result: ValidationResult,
        download_failures: dict[str, str] | None = None,
    ) -> EvaluationLog:
        """Build EvaluationLog from ValidationResult."""
        now = datetime.now(UTC)

        # Get winner metrics
        winner_result = next(
            (
                r
                for r in result.eval_batch.results
                if r.hotkey == result.winner.winner_hotkey
            ),
            None,
        )
        winner_f1 = (
            winner_result.metrics.f1
            if winner_result and winner_result.metrics
            else None
        )

        # Build miner results
        miner_results = []
        copiers = result.duplicate_result.copier_hotkeys

        for eval_result in result.eval_batch.results:
            miner_log = MinerResultLog(
                hotkey=eval_result.hotkey,
                score=eval_result.score,
                success=eval_result.success,
                accuracy=eval_result.metrics.accuracy if eval_result.metrics else None,
                precision=eval_result.metrics.precision if eval_result.metrics else None,
                recall=eval_result.metrics.recall if eval_result.metrics else None,
                f1=eval_result.metrics.f1 if eval_result.metrics else None,
                model_hash=eval_result.model_hash,
                hf_repo_id=eval_result.hf_repo_id,
                inference_time_ms=eval_result.inference_time_ms,
                is_winner=(eval_result.hotkey == result.winner.winner_hotkey),
                is_copier=(eval_result.hotkey in copiers),
                error=eval_result.error_message if not eval_result.success else None,
            )
            miner_results.append(miner_log)

        # Add download failures (miners that never made it to evaluation)
        if download_failures:
            for hotkey, error_msg in download_failures.items():
                miner_log = MinerResultLog(
                    hotkey=hotkey,
                    score=0.0,
                    success=False,
                    error=error_msg,
                )
                miner_results.append(miner_log)

        return EvaluationLog(
            timestamp=now,
            evaluation_date=now.strftime("%Y-%m-%d"),
            validator_hotkey=self._validator_hotkey,
            netuid=self._netuid,
            dataset_size=result.eval_batch.dataset_size,
            models_evaluated=len(result.eval_batch.results),
            models_succeeded=result.eval_batch.successful_count,
            models_failed=result.eval_batch.failed_count,
            winner_hotkey=result.winner.winner_hotkey,
            winner_score=result.winner.winner_score,
            winner_f1=winner_f1,
            winner_block=result.winner.winner_block,
            duplicate_groups_found=len(result.duplicate_result.groups),
            copiers_detected=len(copiers),
            total_evaluation_time_ms=result.eval_batch.total_time_ms,
            miner_results=miner_results,
        )

    def _log_miner_table(self, miner_results: list[MinerResultLog]) -> None:
        """Log per-miner results as a WandB table."""
        wandb = self._import_wandb()

        columns = [
            "hotkey",
            "score",
            "success",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "inference_time_ms",
            "is_winner",
            "is_copier",
            "model_hash",
            "hf_repo_id",
            "error",
        ]

        table = wandb.Table(columns=columns)

        for miner in miner_results:
            table.add_data(
                miner.hotkey,
                miner.score,
                miner.success,
                miner.accuracy,
                miner.precision,
                miner.recall,
                miner.f1,
                miner.inference_time_ms,
                miner.is_winner,
                miner.is_copier,
                miner.model_hash,
                miner.hf_repo_id,
                miner.error,
            )

        self._run.log({"miner_results": table})

    def _log_predictions_table(
        self,
        result: ValidationResult,
        dataset: EvaluationDataset,
    ) -> None:
        """
        Log per-window predictions as a WandB table.

        For dashboard joining - allows matching predictions to scan windows.
        Logs all miners by default. When predictions_top_n_miners is set,
        limits to top N miners by score.
        """
        wandb = self._import_wandb()

        # Get top N miners by score
        successful = [r for r in result.eval_batch.results if r.success]
        sorted_by_score = sorted(successful, key=lambda r: r.score, reverse=True)
        top_miners = sorted_by_score[: self._config.predictions_top_n_miners]

        if not top_miners:
            logger.debug("No successful miners to log predictions for")
            return

        # Build table
        columns = [
            "scan_id",
            "window_index",
            "hotkey",
            "predicted_label",
            "ground_truth_label",
            "correct",
        ]
        table = wandb.Table(columns=columns)

        ground_truth = dataset.ground_truth

        for miner in top_miners:
            if miner.predictions is None:
                continue

            for i, window in enumerate(dataset.windows):
                if i < len(miner.predictions):
                    predicted = int(miner.predictions[i])
                    gt = int(ground_truth[i])

                    log_entry = WindowPredictionLog(
                        scan_id=window.scan_id,
                        window_index=window.window_index,
                        hotkey=miner.hotkey,
                        predicted_label=predicted,
                        ground_truth_label=gt,
                    )

                    table.add_data(
                        log_entry.scan_id,
                        log_entry.window_index,
                        log_entry.hotkey,
                        log_entry.predicted_label,
                        log_entry.ground_truth_label,
                        log_entry.correct,
                    )

        self._run.log({"window_predictions": table})
        logger.debug(
            f"Logged {len(top_miners)} miners x {len(dataset)} windows predictions"
        )


def create_wandb_logger(
    project: str = "davinci-evaluations",
    entity: str | None = None,
    api_key: str | None = None,
    validator_hotkey: str = "",
    netuid: int = 0,
    enabled: bool = True,
    offline: bool = False,
    log_predictions_table: bool = False,
    predictions_top_n_miners: int | None = None,
) -> WandbLogger:
    """
    Create a WandB logger with common configuration.

    Args:
        project: WandB project name
        entity: WandB entity (team/user)
        api_key: WandB API key (or set WANDB_API_KEY env var)
        validator_hotkey: Validator's SS58 address
        netuid: Subnet UID
        enabled: Whether logging is enabled
        offline: Run in offline mode
        log_predictions_table: Log per-window predictions table (disabled by default)
        predictions_top_n_miners: Limit to top N miners by score. None = all miners (default).

    Returns:
        Configured WandbLogger instance

    Example:
        logger = create_wandb_logger(
            project="davinci-subnet",
            validator_hotkey="5F...",
            api_key=config.wandb_api_key or None,
            enabled=not config.wandb_off,
        )
        logger.start_run()
        logger.log_evaluation(result, dataset)
        logger.finish()
    """
    config = WandbConfig(
        project=project,
        entity=entity,
        api_key=api_key or None,
        enabled=enabled,
        offline=offline,
        log_predictions_table=log_predictions_table,
        predictions_top_n_miners=predictions_top_n_miners,
    )
    return WandbLogger(config, validator_hotkey, netuid)
