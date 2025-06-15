"""
Models package.
Contains all Pydantic models and data structures.
"""

from .config import (
    ImageConfig, AppConfig, ChangeDetectionConfig, YOLOConfig, 
    GlobalConfig, Config
)
from .detection import (
    Detection, CatDetectionWithActivity, CatActivity
)
from .feedback import (
    BoundingBox, FeedbackAnnotation, ImageFeedback, TrainingDataExport,
    ModelSaveRequest, ModelRetrainRequest
)
from .cat import CatProfile

__all__ = [
    # Config models
    "ImageConfig", "AppConfig", "ChangeDetectionConfig", "YOLOConfig", 
    "GlobalConfig", "Config",
    
    # Detection models  
    "Detection", "CatDetectionWithActivity", "CatActivity",
    
    # Feedback models
    "BoundingBox", "FeedbackAnnotation", "ImageFeedback", "TrainingDataExport",
    "ModelSaveRequest", "ModelRetrainRequest",
    
    # Cat models
    "CatProfile"
] 