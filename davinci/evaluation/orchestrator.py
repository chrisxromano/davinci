"""
Evaluation orchestrator for coordinating GPR model evaluations.

Manages the evaluation of multiple ONNX models against a validation dataset
of GPR scan windows with binary rebar detection labels.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .docker_runner import DockerConfig, DockerRunner
from .metrics import calculate_metrics
from .models import EvaluationBatch, EvaluationResult

if TYPE_CHECKING:
    from ..chain.models import ChainModelMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration for evaluation orchestrator."""

    max_concurrent: int = 4
    """Maximum number of concurrent model evaluations."""

    docker_config: DockerConfig = DockerConfig()
    """Docker configuration for model execution."""



class EvaluationOrchestrator:
    """
    Orchestrate evaluation of multiple models.

    Coordinates:
    - Docker-based model execution
    - Metrics calculation
    - Result aggregation
    - Error handling

    Usage:
        orchestrator = EvaluationOrchestrator(OrchestratorConfig())
        batch = await orchestrator.evaluate_all(
            models={hotkey: model_path},
            dataset=validation_dataset,
        )
        best = batch.get_best()
    """

    def __init__(self, config: OrchestratorConfig):
        """
        Initialize orchestrator.

        Args:
            config: Orchestrator configuration.
        """
        self._config = config
        self._docker_runner = DockerRunner(self._config.docker_config)

    async def evaluate_all(
        self,
        models: dict[str, Path],
        features: np.ndarray,
        ground_truth: np.ndarray,
        model_metadata: dict[str, ChainModelMetadata] | None = None,
    ) -> EvaluationBatch:
        """
        Evaluate all models on the validation dataset.

        Args:
            models: Mapping of hotkey -> model path
            features: GPR scan windows array (N, 1, depth, width)
            ground_truth: Binary rebar labels (N,) with values 0 or 1
            model_metadata: Optional chain metadata for each model

        Returns:
            EvaluationBatch with all results

        Example:
            batch = await orchestrator.evaluate_all(
                models={
                    "5F3sa...": Path("model_cache/5F3sa.../model.onnx"),
                    "5G7xb...": Path("model_cache/5G7xb.../model.onnx"),
                },
                features=dataset.input_batch,
                ground_truth=dataset.ground_truth,
            )
            best = batch.get_best()
        """
        start_time = time.time()
        results: list[EvaluationResult] = []
        model_metadata = model_metadata or {}

        logger.info(
            f"Starting evaluation of {len(models)} models on {len(ground_truth)} samples"
        )

        # Evaluate models with semaphore for concurrency control
        semaphore = asyncio.Semaphore(self._config.max_concurrent)

        async def evaluate_with_semaphore(
            hotkey: str, model_path: Path
        ) -> EvaluationResult:
            async with semaphore:
                return await self._evaluate_single_model(
                    hotkey=hotkey,
                    model_path=model_path,
                    features=features,
                    ground_truth=ground_truth,
                    metadata=model_metadata.get(hotkey),
                )

        # Create tasks for all models
        tasks = [
            evaluate_with_semaphore(hotkey, model_path)
            for hotkey, model_path in models.items()
        ]

        # Run all evaluations
        if tasks:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in completed_results:
                if isinstance(result, Exception):
                    # This shouldn't happen as _evaluate_single_model catches exceptions
                    logger.error(
                        f"Unexpected exception during evaluation: {result}",
                        exc_info=result,
                    )
                    continue
                results.append(result)

        elapsed_ms = (time.time() - start_time) * 1000

        batch = EvaluationBatch(
            results=results,
            dataset_size=len(ground_truth),
            total_time_ms=elapsed_ms,
        )

        logger.info(
            f"Evaluation complete: {batch.successful_count} succeeded, "
            f"{batch.failed_count} failed in {elapsed_ms:.1f}ms"
        )

        if batch.successful_count > 0:
            best = batch.get_best()
            if best:
                logger.info(f"Highest score: {best.hotkey} with {best.score:.2f}")

        return batch

    async def _evaluate_single_model(
        self,
        hotkey: str,
        model_path: Path,
        features: np.ndarray,
        ground_truth: np.ndarray,
        metadata: ChainModelMetadata | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a single model.

        Returns EvaluationResult (never raises).
        """
        logger.debug(f"Evaluating model for {hotkey}")

        try:
            # Run inference in Docker (blocking call wrapped in thread)
            inference_result = await asyncio.to_thread(
                self._docker_runner.run_inference,
                model_path,
                features,
            )

            predictions = inference_result.predictions

            # Calculate metrics
            metrics = calculate_metrics(
                y_true=ground_truth,
                y_pred=predictions,
            )

            logger.debug(
                f"Model {hotkey}: F1={metrics.f1:.2%}, accuracy={metrics.accuracy:.2%}"
            )

            return EvaluationResult(
                hotkey=hotkey,
                predictions=predictions,
                metrics=metrics,
                inference_time_ms=inference_result.inference_time_ms,
                model_hash=metadata.model_hash if metadata else None,
                hf_repo_id=metadata.hf_repo_id if metadata else None,
            )

        except Exception as e:
            logger.warning(f"Model {hotkey} evaluation failed: {e}")

            # Log container logs if available (for debugging)
            if hasattr(e, "logs") and e.logs:
                logger.warning(f"Container logs for {hotkey}:\n{e.logs}")

            return EvaluationResult(
                hotkey=hotkey,
                error=e,
                model_hash=metadata.model_hash if metadata else None,
                hf_repo_id=metadata.hf_repo_id if metadata else None,
            )


def create_orchestrator(
    max_concurrent: int = 4,
    docker_memory: str = "2g",
    docker_cpu: float = 1.0,
    docker_timeout: int = 300,
) -> EvaluationOrchestrator:
    """
    Create an evaluation orchestrator with custom configuration.

    Args:
        max_concurrent: Maximum concurrent evaluations
        docker_memory: Docker memory limit (e.g., '2g', '4g')
        docker_cpu: Docker CPU limit (1.0 = 1 core)
        docker_timeout: Inference timeout in seconds

    Returns:
        Configured EvaluationOrchestrator
    """
    config = OrchestratorConfig(
        max_concurrent=max_concurrent,
        docker_config=DockerConfig(
            memory_limit=docker_memory,
            cpu_limit=docker_cpu,
            timeout_seconds=docker_timeout,
        ),
    )

    return EvaluationOrchestrator(config)
