"""
Base class for ML detection processes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

from models import ImageDetections


class MLDetectionProcess(ABC):
    """
    Abstract base class for ML detection processes.

    Each process represents a single ML task that can be chained together
    in a pipeline (e.g., YOLO detection, feature extraction, classification).
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ML detection process.

        Args:
            config: Configuration dictionary for the process
        """
        self.config = config or {}
        self._is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the ML model and any required resources.
        This is called once before processing begins.
        """
        pass

    @abstractmethod
    async def process(
        self, image_array: np.ndarray, detections: ImageDetections
    ) -> ImageDetections:
        """
        Process the image and update detections.

        Args:
            image_array: Input image as numpy array (H, W, C)
            detections: Current detection results from previous processes

        Returns:
            Updated detection results
        """
        pass

    @abstractmethod
    def get_process_name(self) -> str:
        """Get the name of this process for logging/debugging."""
        pass

    async def cleanup(self) -> None:
        """
        Cleanup resources when process is no longer needed.
        Override in subclasses if cleanup is required.
        """
        pass

    def is_initialized(self) -> bool:
        """Check if the process has been initialized."""
        return self._is_initialized

    def _set_initialized(self) -> None:
        """Mark the process as initialized."""
        self._is_initialized = True
