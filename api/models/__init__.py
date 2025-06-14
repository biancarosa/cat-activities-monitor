"""
Models package.
Contains all Pydantic models and data structures.
"""

from .config import *
from .detection import *
from .feedback import *
from .cat import *
from .recognition import *

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
    "CatProfile",
    
    # Recognition models
    "CatFeature", "CatRecognitionModel", "CatRecognitionResult", "CatRecognitionRequest",
    "FeatureExtractionRequest", "FeatureExtractionResult", "CatSimilarityMatch",
    "RecognitionTrainingData", "ModelTrainingRequest", "ModelTrainingResult",
    "RecognitionSystemStatus", "ActivityEnhancedRecognition"
] 