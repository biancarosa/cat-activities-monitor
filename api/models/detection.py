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


class ActivityDetection(BaseModel):
    """
    Model for activity detection results.
    
    Represents a detected activity with confidence, reasoning, and temporal information.
    """
    activity: CatActivity = Field(
        ..., 
        description="The detected cat activity"
    )
    confidence: float = Field(
        ..., 
        description="Confidence score for the activity detection",
        example=0.92,
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        ..., 
        description="Explanation of why this activity was detected",
        example="Cat is in a crouched position with head lowered towards ground, typical eating posture"
    )
    bounding_box: Dict[str, float] = Field(
        ..., 
        description="Bounding box coordinates for the activity region",
        example={
            "x1": 100.5,
            "y1": 150.2,
            "x2": 300.8,
            "y2": 400.1,
            "width": 200.3,
            "height": 249.9
        }
    )
    duration_seconds: Optional[float] = Field(
        None, 
        description="How long this activity has been detected in seconds",
        example=15.5,
        ge=0.0
    )
    cat_index: Optional[int] = Field(
        None, 
        description="Which cat this activity belongs to (0-based index)",
        example=0,
        ge=0
    )
    detection_id: Optional[str] = Field(
        None, 
        description="Unique ID to link to specific detection",
        example="det_20231201_143022_001"
    )


class CatDetectionWithActivity(BaseModel):
    """
    Enhanced cat detection that includes activity recognition.
    
    This is the main response model for detection endpoints, containing
    both object detection results and activity analysis.
    """
    detected: bool = Field(
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
    count: int = Field(
        ..., 
        description="Number of cats detected",
        example=2,
        ge=0
    )
    detections: List[Detection] = Field(
        ..., 
        description="List of individual detection results"
    )
    total_animals: int = Field(
        ..., 
        description="Total number of animals detected (cats + other animals)",
        example=2,
        ge=0
    )
    activities: List[ActivityDetection] = Field(
        default_factory=list, 
        description="List of detected activities for all cats"
    )
    primary_activity: Optional[CatActivity] = Field(
        None, 
        description="Most confident activity detected across all cats"
    )
    cat_activities: Optional[Dict[str, List[ActivityDetection]]] = Field(
        None, 
        description="Map of cat index to their detected activities",
        example={
            "0": [{"activity": "eating", "confidence": 0.92}],
            "1": [{"activity": "sitting", "confidence": 0.78}]
        }
    ) 