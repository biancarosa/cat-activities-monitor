"""
ML Detection Pipeline that orchestrates multiple detection processes.
"""

import logging
from typing import List
import numpy as np

from models import ImageDetections
from .base_process import MLDetectionProcess

logger = logging.getLogger(__name__)


class MLDetectionPipeline:
    """
    Pipeline that runs multiple ML detection processes in sequence.
    
    Each process receives the image and the accumulated detection results
    from previous processes, allowing for progressive enhancement of detections.
    """
    
    def __init__(self, processes: List[MLDetectionProcess]):
        """
        Initialize the pipeline with a list of processes.
        
        Args:
            processes: List of MLDetectionProcess instances to run in sequence
        """
        self.processes = processes
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize all processes in the pipeline.
        """
        logger.info(f"Initializing ML pipeline with {len(self.processes)} processes")
        
        for i, process in enumerate(self.processes):
            if not process.is_initialized():
                logger.info(f"Initializing process {i+1}/{len(self.processes)}: {process.get_process_name()}")
                await process.initialize()
        
        self._is_initialized = True
        logger.info("ML pipeline initialization complete")
    
    async def process_image(self, image_array: np.ndarray) -> ImageDetections:
        """
        Process an image through all pipeline stages.
        
        Args:
            image_array: Input image as numpy array (H, W, C)
            
        Returns:
            Final detection results after all processes
        """
        if not self._is_initialized:
            await self.initialize()
        
        # Start with empty detections
        detections = ImageDetections(
            detected=False,
            cat_detected=False,
            cats_count=0,
            confidence=0.0,
            detections=[]
        )
        
        # Run each process in sequence
        for i, process in enumerate(self.processes):
            process_name = process.get_process_name()
            logger.debug(f"Running process {i+1}/{len(self.processes)}: {process_name}")
            
            try:
                detections = await process.process(image_array, detections)
                logger.debug(f"Process {process_name} completed. Cats detected: {detections.cats_count}")
            except Exception as e:
                logger.error(f"Error in process {process_name}: {e}")
                # Continue with other processes even if one fails
                continue
        
        return detections
    
    async def cleanup(self) -> None:
        """
        Cleanup all processes in the pipeline.
        """
        logger.info("Cleaning up ML pipeline")
        for process in self.processes:
            try:
                await process.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up process {process.get_process_name()}: {e}")
    
    def is_initialized(self) -> bool:
        """Check if the pipeline has been initialized."""
        return self._is_initialized
    
    def get_process_names(self) -> List[str]:
        """Get names of all processes in the pipeline."""
        return [process.get_process_name() for process in self.processes]