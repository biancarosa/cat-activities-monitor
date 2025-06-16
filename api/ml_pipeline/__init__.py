"""
ML Pipeline module for Cat Activities Monitor.

This module provides a flexible pipeline architecture for running multiple
machine learning detection processes in sequence.
"""

from .pipeline import MLDetectionPipeline
from .base_process import MLDetectionProcess
from .yolo_detection import YOLODetectionProcess
from .feature_extraction import FeatureExtractionProcess

__all__ = [
    'MLDetectionPipeline',
    'MLDetectionProcess', 
    'YOLODetectionProcess',
    'FeatureExtractionProcess'
]