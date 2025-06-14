"""
Configuration models.
"""

from typing import List
from pydantic import BaseModel, HttpUrl, ConfigDict


class ImageConfig(BaseModel):
    name: str
    url: HttpUrl
    interval_seconds: int = 60
    enabled: bool = True


class AppConfig(BaseModel):
    name: str = "Cat Activities Monitor API"
    log_level: str = "INFO"


class ChangeDetectionConfig(BaseModel):
    """Configuration for change detection."""
    enabled: bool = True
    similarity_threshold: float = 0.85  # 0.0-1.0, higher = more similar required to skip
    detection_change_threshold: float = 0.3  # Minimum change in detection confidence to trigger save
    position_change_threshold: float = 50  # Minimum pixel movement to trigger save
    activity_change_triggers: bool = True  # Save when activity changes


class YOLOConfig(BaseModel):
    model: str = "ml_models/yolo11l.pt"  # Using yolo11l for better performance
    confidence_threshold: float = 0.01   # Ultra-sensitive for detecting both cats
    iou_threshold: float = 0.1          # Very low IoU threshold to allow overlapping detections
    max_detections: int = 1000          # High max detections to catch multiple cats
    image_size: int = 1280              # Larger input size for more accuracy
    target_classes: List[int] = [15, 16]  # 15=cat, 16=dog (YOLO sometimes confuses cats/dogs)
    save_detection_images: bool = True   # Save images with bounding boxes for debugging
    detection_image_path: str = "./detections"
    change_detection: ChangeDetectionConfig = ChangeDetectionConfig()


class GlobalConfig(BaseModel):
    default_interval_seconds: int = 60
    max_concurrent_fetches: int = 3
    timeout_seconds: int = 30
    ml_model_config: YOLOConfig = YOLOConfig()


class Config(BaseModel):
    model_config = ConfigDict(fields={'global_': 'global'})
    
    app: AppConfig = AppConfig()        # Provide default
    images: List[ImageConfig] = []      # Default to empty list, will be populated from config file
    global_: GlobalConfig = GlobalConfig() 