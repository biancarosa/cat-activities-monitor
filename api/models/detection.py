"""
Detection and activity models.
"""

from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """
    Represents a single object detection result from ML model.
    
    Contains the detected object's class information, confidence score,
    and bounding box coordinates.
    """
    class_id: int = Field(
        ..., 
        description="YOLO class ID for the detected object",
        example=15,
        ge=0
    )
    class_name: str = Field(
        ..., 
        description="Human-readable name of the detected class",
        example="cat"
    )
    confidence: float = Field(
        ..., 
        description="Detection confidence score between 0.0 and 1.0",
        example=0.85,
        ge=0.0,
        le=1.0
    )
    bounding_box: Dict[str, float] = Field(
        ..., 
        description="Bounding box coordinates in pixels",
        example={
            "x1": 100.5,
            "y1": 150.2,
            "x2": 300.8,
            "y2": 400.1,
            "width": 200.3,
            "height": 249.9
        }
    )
    cat_uuid: Optional[str] = Field(
        None,
        description="Unique identifier for this cat detection",
        example="550e8400-e29b-41d4-a716-446655440000"
    )


class CatActivity(str, Enum):
    """
    Enumeration of detectable cat activities.
    
    These activities are recognized through pose analysis and behavioral patterns.
    """
    UNKNOWN = "unknown"
    SITTING = "sitting"
    LYING = "lying"
    STANDING = "standing"
    MOVING = "moving"
    EATING = "eating"
    PLAYING = "playing"
    SLEEPING = "sleeping"
    GROOMING = "grooming"


class ImageDetections(BaseModel):
    """
    List of detections for an image.
    """
    cat_detected: bool = Field(
        ..., 
        description="Whether any cats were detected in the image",
        example=True
    )
    confidence: float = Field(
        ..., 
        description="Overall detection confidence (highest among all detections)",
        example=0.89,
        ge=0.0,
        le=1.0
    )
    cats_count: int = Field(
        ..., 
        description="Number of cats detected",
        example=2,
        ge=0
    )
    detections: List[Detection] = Field(
        ..., 
        description="List of individual detection results"
    )