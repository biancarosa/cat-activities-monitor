"""
ML Training Pipeline that orchestrates multiple training processes.
"""

import logging
from typing import List, Dict
import asyncio
from datetime import datetime

from .base_trainer import BaseTrainer, TrainingData, TrainingResult

logger = logging.getLogger(__name__)


class MLTrainingPipeline:
    """
    Pipeline that runs multiple ML training processes in sequence or parallel.

    Each trainer receives the same training data and produces independent models
    that can be used for different aspects of cat identification and behavior analysis.
    """

    def __init__(self, trainers: List[BaseTrainer]):
        """
        Initialize the training pipeline with a list of trainers.

        Args:
            trainers: List of BaseTrainer instances to run
        """
        self.trainers = trainers
        self._is_initialized = False
        self.training_results = {}

    async def initialize(self) -> None:
        """
        Initialize all trainers in the pipeline.
        """
        logger.info(
            f"Initializing ML training pipeline with {len(self.trainers)} trainers"
        )

        for i, trainer in enumerate(self.trainers):
            if not trainer.is_initialized():
                trainer_name = trainer.get_trainer_name()
                logger.info(
                    f"Initializing trainer {i+1}/{len(self.trainers)}: {trainer_name}"
                )
                try:
                    await trainer.initialize()
                    logger.info(f"Successfully initialized {trainer_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize {trainer_name}: {e}")
                    raise

        self._is_initialized = True
        logger.info("ML training pipeline initialization complete")

    async def train_all(
        self, training_data: TrainingData, parallel: bool = False
    ) -> Dict[str, TrainingResult]:
        """
        Train all models in the pipeline using the provided data.

        Args:
            training_data: Training data to use for all trainers
            parallel: Whether to run trainers in parallel or sequentially

        Returns:
            Dictionary mapping trainer names to training results
        """
        if not self._is_initialized:
            await self.initialize()

        logger.info(
            f"Starting training pipeline with {len(training_data.features)} samples "
            f"({'parallel' if parallel else 'sequential'} execution)"
        )

        start_time = datetime.now()

        if parallel:
            results = await self._train_parallel(training_data)
        else:
            results = await self._train_sequential(training_data)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Store results and log summary
        self.training_results = results
        successful_trainers = sum(1 for result in results.values() if result.success)

        logger.info(
            f"Training pipeline completed in {total_time:.2f}s. "
            f"Success: {successful_trainers}/{len(self.trainers)} trainers"
        )

        return results

    async def _train_sequential(
        self, training_data: TrainingData
    ) -> Dict[str, TrainingResult]:
        """Train all models sequentially."""
        results = {}

        for i, trainer in enumerate(self.trainers):
            trainer_name = trainer.get_trainer_name()
            logger.info(f"Training model {i+1}/{len(self.trainers)}: {trainer_name}")

            try:
                # Validate training data for this specific trainer
                if not await trainer.validate_training_data(training_data):
                    results[trainer_name] = TrainingResult(
                        success=False, error_message="Training data validation failed"
                    )
                    continue

                # Check minimum samples requirement - let trainer determine sample count
                min_samples = await trainer.get_minimum_samples_required()
                # Get appropriate sample count based on trainer type
                if trainer_name == "YOLOTrainer":
                    sample_count = (
                        len(training_data.metadata) if training_data.metadata else 0
                    )
                else:
                    sample_count = (
                        len(training_data.features) if training_data.features else 0
                    )

                if sample_count < min_samples:
                    results[trainer_name] = TrainingResult(
                        success=False,
                        error_message=f"Insufficient training data: {sample_count} < {min_samples}",
                    )
                    continue

                # Train the model
                result = await trainer.train(training_data)
                results[trainer_name] = result

                if result.success:
                    logger.info(f"Successfully trained {trainer_name}")
                    if result.metrics:
                        logger.info(f"Training metrics: {result.metrics}")
                else:
                    logger.error(
                        f"Training failed for {trainer_name}: {result.error_message}"
                    )

            except Exception as e:
                logger.error(f"Unexpected error training {trainer_name}: {e}")
                results[trainer_name] = TrainingResult(
                    success=False, error_message=str(e)
                )

        return results

    async def _train_parallel(
        self, training_data: TrainingData
    ) -> Dict[str, TrainingResult]:
        """Train all models in parallel."""

        async def train_single(trainer: BaseTrainer) -> tuple[str, TrainingResult]:
            trainer_name = trainer.get_trainer_name()

            try:
                # Validate training data
                if not await trainer.validate_training_data(training_data):
                    return trainer_name, TrainingResult(
                        success=False, error_message="Training data validation failed"
                    )

                # Check minimum samples requirement - let trainer determine sample count
                min_samples = await trainer.get_minimum_samples_required()
                # Get appropriate sample count based on trainer type
                if trainer_name == "YOLOTrainer":
                    sample_count = (
                        len(training_data.metadata) if training_data.metadata else 0
                    )
                else:
                    sample_count = (
                        len(training_data.features) if training_data.features else 0
                    )

                if sample_count < min_samples:
                    return trainer_name, TrainingResult(
                        success=False,
                        error_message=f"Insufficient training data: {sample_count} < {min_samples}",
                    )

                # Train the model
                result = await trainer.train(training_data)

                if result.success:
                    logger.info(f"Successfully trained {trainer_name}")
                else:
                    logger.error(
                        f"Training failed for {trainer_name}: {result.error_message}"
                    )

                return trainer_name, result

            except Exception as e:
                logger.error(f"Unexpected error training {trainer_name}: {e}")
                return trainer_name, TrainingResult(success=False, error_message=str(e))

        # Execute all trainers in parallel
        tasks = [train_single(trainer) for trainer in self.trainers]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert results to dictionary
        results = {}
        for item in results_list:
            if isinstance(item, Exception):
                logger.error(f"Training task failed with exception: {item}")
                continue

            trainer_name, result = item
            results[trainer_name] = result

        return results

    async def cleanup(self) -> None:
        """
        Cleanup all trainers in the pipeline.
        """
        logger.info("Cleaning up ML training pipeline")

        cleanup_tasks = []
        for trainer in self.trainers:
            try:
                cleanup_tasks.append(trainer.cleanup())
            except Exception as e:
                logger.error(
                    f"Error setting up cleanup for {trainer.get_trainer_name()}: {e}"
                )

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    def is_initialized(self) -> bool:
        """Check if the pipeline has been initialized."""
        return self._is_initialized

    def get_trainer_names(self) -> List[str]:
        """Get names of all trainers in the pipeline."""
        return [trainer.get_trainer_name() for trainer in self.trainers]

    def get_last_training_results(self) -> Dict[str, TrainingResult]:
        """Get results from the last training run."""
        return self.training_results.copy()

    def get_successful_models(self) -> List[str]:
        """Get list of trainer names that successfully completed training."""
        return [
            name
            for name, result in self.training_results.items()
            if result.success and result.model_path
        ]
