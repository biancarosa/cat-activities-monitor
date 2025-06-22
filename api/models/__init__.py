"""
Models package.
Contains all Pydantic models and data structures.
"""

from .config import (
    ImageConfig,
    AppConfig,
    ChangeDetectionConfig,
    ActivityDetectionConfig,
    YOLOConfig,
    GlobalConfig,
    Config,
)
from .detection import Detection, ImageDetections, CatActivity
from .feedback import (
    BoundingBox,
    FeedbackAnnotation,
    ImageFeedback,
    TrainingDataExport,
    ModelSaveRequest,
    ModelRetrainRequest,
)
from .cat import CatProfile, CreateCatProfileRequest, UpdateCatProfileRequest

__all__ = [
    # Config models
    "ImageConfig",
    "AppConfig",
    "ChangeDetectionConfig",
    "ActivityDetectionConfig",
    "YOLOConfig",
    "GlobalConfig",
    "Config",
    # Detection models
    "Detection",
    "ImageDetections",
    "CatActivity",
    # Feedback models
    "BoundingBox",
    "FeedbackAnnotation",
    "ImageFeedback",
    "TrainingDataExport",
    "ModelSaveRequest",
    "ModelRetrainRequest",
    # Cat models
    "CatProfile",
    "CreateCatProfileRequest",
    "UpdateCatProfileRequest",
]
