"""
ML Training Pipeline module for Cat Activities Monitor.

This module provides a flexible pipeline architecture for training machine learning
models based on user feedback and cat identification data.
"""

from .pipeline import MLTrainingPipeline
from .base_trainer import BaseTrainer
from .cat_identification_trainer import CatIdentificationTrainer
from .feature_clustering_trainer import FeatureClusteringTrainer
from .yolo_trainer import YOLOTrainer

__all__ = [
    "MLTrainingPipeline",
    "BaseTrainer",
    "CatIdentificationTrainer",
    "FeatureClusteringTrainer",
    "YOLOTrainer",
]
