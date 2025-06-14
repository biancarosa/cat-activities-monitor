"""
Models package.
Contains all Pydantic models and data structures.
"""

from .config import (
    ImageConfig, AppConfig, ChangeDetectionConfig, YOLOConfig, 
    GlobalConfig, Config
)
from .detection import (
    Detection, CatActivity, ActivityDetection, CatDetectionWithActivity
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
    "Detection", "CatActivity", "ActivityDetection", "CatDetectionWithActivity",
    
    # Feedback models
    "BoundingBox", "FeedbackAnnotation", "ImageFeedback", "TrainingDataExport",
    "ModelSaveRequest", "ModelRetrainRequest",
    
    # Cat models
    "CatProfile"
] 