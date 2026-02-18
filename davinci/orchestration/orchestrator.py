"""Validation orchestrator - coordinates the evaluation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from davinci.duplicate_detector import create_duplicate_detector
from davinci.evaluation import create_orchestrator as create_eval_orchestrator
from davinci.incentives import (
    DistributorConfig,
    IncentiveDistributor,
    NoValidModelsError,
    WinnerSelector,
)

from .models import ValidationResult

if TYPE_CHECKING:
    from davinci.chain.models import ChainModelMetadata
    from davinci.data import EvaluationDataset
    from davinci.duplicate_detector import DuplicateDetector
    from davinci.evaluation import EvaluationOrchestrator

logger = logging.getLogger(__name__)


class ValidationOrchestrator:
    """
    Coordinates the full evaluation pipeline.

    Stateless - all inputs passed explicitly, returns result.
    All dependencies injected.

    Pipeline steps:
    1. Prepare input batch from evaluation dataset
    2. Run inference on all models
    3. Detect duplicate predictions, identify copiers
    4. Filter copiers, select winner (threshold + commit time)
    5. Calculate weight distribution (99/1/0 split)
    """

    def __init__(
        self,
        evaluator: EvaluationOrchestrator,
        detector: DuplicateDetector,
        selector: WinnerSelector,
        distributor: IncentiveDistributor,
    ):
        """
        Initialize orchestrator with all dependencies.

        Args:
            evaluator: Model evaluation orchestrator
            detector: Duplicate prediction detector
            selector: Winner selection logic
            distributor: Weight distribution logic
        """
        self._evaluator = evaluator
        self._duplicate_detector = detector
        self._selector = selector
        self._distributor = distributor

    @classmethod
    def create(
        cls,
        *,
        similarity_threshold: float = 1e-6,
        score_threshold: float = 0.005,
        winner_share: float = 0.99,
        docker_timeout: int = 300,
        docker_memory: str = "2g",
        docker_cpu: float = 1.0,
        docker_max_concurrent: int = 4,
    ) -> ValidationOrchestrator:
        """
        Create orchestrator with default dependencies.

        Args:
            similarity_threshold: Threshold for duplicate prediction detection.
            score_threshold: Score threshold for winner selection.
            winner_share: Share of emissions allocated to winner (default 99%).
            docker_timeout: Docker inference timeout in seconds.
            docker_memory: Docker memory limit (e.g., '2g').
            docker_cpu: Docker CPU limit (1.0 = 1 core).
            docker_max_concurrent: Maximum concurrent Docker evaluations.

        Returns:
            Configured ValidationOrchestrator ready to run evaluations.
        """
        evaluator = create_eval_orchestrator(
            max_concurrent=docker_max_concurrent,
            docker_memory=docker_memory,
            docker_cpu=docker_cpu,
            docker_timeout=docker_timeout,
        )

        return cls(
            evaluator=evaluator,
            detector=create_duplicate_detector(
                similarity_threshold=similarity_threshold
            ),
            selector=WinnerSelector(score_threshold),
            distributor=IncentiveDistributor(
                DistributorConfig(winner_share=winner_share)
            ),
        )

    async def run(
        self,
        dataset: EvaluationDataset,
        model_paths: dict[str, Path],
        chain_metadata: dict[str, ChainModelMetadata],
    ) -> ValidationResult:
        """
        Run full evaluation pipeline.

        Args:
            dataset: Evaluation dataset with GPR scan windows and ground truth
            model_paths: Mapping of hotkey -> path to ONNX model file
            chain_metadata: Mapping of hotkey -> chain commitment metadata

        Returns:
            ValidationResult with weights and evaluation details

        Raises:
            NoValidModelsError: If no models pass evaluation or all are copiers
        """
        logger.info(
            f"Starting evaluation: {len(model_paths)} models, {len(dataset)} samples"
        )

        # 1. Prepare input batch — add channel dimension (N, depth, width) → (N, 1, depth, width)
        features = dataset.input_batch[:, np.newaxis, :, :]
        ground_truth = dataset.ground_truth

        logger.debug(f"Input batch shape: {features.shape}")

        # 2. Run evaluation on all models
        logger.info("Running model evaluation...")
        eval_batch = await self._evaluator.evaluate_all(
            models=model_paths,
            features=features,
            ground_truth=ground_truth,
            model_metadata=chain_metadata,
        )

        successful_count = len(eval_batch.successful_results)
        failed_count = len(eval_batch.results) - successful_count
        logger.info(
            f"Evaluation complete: {successful_count} successful, {failed_count} failed"
        )

        if not eval_batch.successful_results:
            raise NoValidModelsError("All model evaluations failed")

        # 3. Detect duplicates
        logger.debug("Detecting duplicate predictions...")
        duplicates = self._duplicate_detector.detect(
            eval_batch.results,
            chain_metadata,
        )
        copiers = duplicates.copier_hotkeys

        if copiers:
            logger.info(
                f"Detected {len(duplicates.groups)} duplicate groups, "
                f"{len(copiers)} copiers"
            )

        # 4. Filter copiers and select winner
        valid_results = [
            r for r in eval_batch.results if r.success and r.hotkey not in copiers
        ]

        if not valid_results:
            raise NoValidModelsError(
                "No valid models identified for this validation round"
            )

        logger.debug(f"Selecting winner from {len(valid_results)} valid models...")
        winner = self._selector.select_winner(valid_results, chain_metadata)

        logger.info(
            f"Winner: {winner.winner_hotkey} "
            f"(score={winner.winner_score:.4f}, block={winner.winner_block})"
        )

        # 5. Calculate weight distribution
        weights = self._distributor.calculate_weights(
            results=eval_batch.results,
            winner_hotkey=winner.winner_hotkey,
            winner_score=winner.winner_score,
            cheater_hotkeys=copiers,
        )

        logger.info(
            f"Weights calculated: winner={weights.get_weight(winner.winner_hotkey):.2%}, "
            f"total={weights.total:.4f}"
        )

        return ValidationResult(
            weights=weights,
            winner=winner,
            eval_batch=eval_batch,
            duplicate_result=duplicates,
        )
