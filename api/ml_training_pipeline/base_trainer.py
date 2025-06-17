"""
Base class for ML training processes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    """Container for training data."""
    features: List[List[float]]
    labels: List[str]
    metadata: List[Dict[str, Any]]


@dataclass
class TrainingResult:
    """Result of a training process."""
    success: bool
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    training_time_seconds: Optional[float] = None


class BaseTrainer(ABC):
    """
    Abstract base class for ML training processes.
    
    Each trainer represents a specific ML training task that can be run
    independently or as part of a training pipeline.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ML training process.
        
        Args:
            config: Configuration dictionary for the trainer
        """
        self.config = config or {}
        self._is_initialized = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the trainer and any required resources.
        This is called once before training begins.
        """
        pass
    
    @abstractmethod
    async def train(self, training_data: TrainingData) -> TrainingResult:
        """
        Train the model using the provided training data.
        
        Args:
            training_data: Container with features, labels, and metadata
            
        Returns:
            Training result with success status and model information
        """
        pass
    
    @abstractmethod
    def get_trainer_name(self) -> str:
        """Get the name of this trainer for logging/debugging."""
        pass
    
    async def validate_training_data(self, training_data: TrainingData) -> bool:
        """
        Validate that training data is suitable for this trainer.
        
        Args:
            training_data: Training data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if not training_data.features:
            self.logger.error("No features provided in training data")
            return False
        
        if not training_data.labels:
            self.logger.error("No labels provided in training data")
            return False
        
        if len(training_data.features) != len(training_data.labels):
            self.logger.error(
                f"Mismatch between features ({len(training_data.features)}) "
                f"and labels ({len(training_data.labels)})"
            )
            return False
        
        return True
    
    async def cleanup(self) -> None:
        """
        Cleanup resources when trainer is no longer needed.
        Override in subclasses if cleanup is required.
        """
        pass
    
    def is_initialized(self) -> bool:
        """Check if the trainer has been initialized."""
        return self._is_initialized
    
    def _set_initialized(self) -> None:
        """Mark the trainer as initialized."""
        self._is_initialized = True
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default fallback."""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    async def get_minimum_samples_required(self) -> int:
        """
        Get the minimum number of training samples required by this trainer.
        Override in subclasses to specify requirements.
        
        Returns:
            Minimum number of samples needed for training
        """
        return 10  # Default minimum