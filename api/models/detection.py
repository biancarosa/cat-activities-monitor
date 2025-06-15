"""
Detection models.
"""

from typing import Dict, List, Optional
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


class CatDetectionWithActivity(BaseModel):
    """
    Cat detection results with backwards compatibility for activities.
    
    This is the main response model for detection endpoints.
    Activities field is maintained for backwards compatibility but always empty.
    """
    detections: List[Detection] = Field(
        ..., 
        description="List of individual detection results"
    )
    cat_count: int = Field(
        ..., 
        description="Number of cats detected",
        example=2,
        ge=0
    )
    max_confidence: float = Field(
        ..., 
        description="Highest detection confidence among all detections",
        example=0.89,
        ge=0.0,
        le=1.0
    )
    activities: Dict[str, list] = Field(
        default_factory=dict, 
        description="Activities by cat (empty for backwards compatibility)"
    )
    primary_activity: Optional[str] = Field(
        None, 
        description="Primary activity (None for backwards compatibility)"
    )
    processing_time_ms: float = Field(
        ..., 
        description="Processing time in milliseconds",
        example=125.5,
        ge=0.0
    )
    image_size: Dict[str, int] = Field(
        ..., 
        description="Image dimensions",
        example={"width": 1920, "height": 1080}
    )