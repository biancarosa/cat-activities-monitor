"""
Detection and activity models.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """
    Represents a single object detection result from ML model.

    Contains the detected object's class information, confidence score,
    and bounding box coordinates.
    """

    class_id: int = Field(
        ..., description="YOLO class ID for the detected object", example=15, ge=0
    )
    class_name: str = Field(
        ..., description="Human-readable name of the detected class", example="cat"
    )
    confidence: float = Field(
        ...,
        description="Detection confidence score between 0.0 and 1.0",
        example=0.85,
        ge=0.0,
        le=1.0,
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
            "height": 249.9,
        },
    )
    cat_uuid: Optional[str] = Field(
        None,
        description="Unique identifier for this cat detection",
        example="550e8400-e29b-41d4-a716-446655440000",
    )
    cat_name: Optional[str] = Field(
        None, description="Identified name of the cat (if recognized)", example="Chico"
    )
    features: Optional[List[float]] = Field(
        None,
        description="Deep learning feature vector for cat recognition (2048 dimensions)",
        example=[0.123, -0.456, 0.789],
    )
    identification_suggestion: Optional[Dict[str, Any]] = Field(
        None,
        description="Cat identification suggestions based on feature matching",
        example={
            "suggested_profile": {
                "uuid": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Whiskers",
                "description": "Orange tabby with white paws",
            },
            "confidence": 0.85,
            "is_confident_match": True,
            "is_new_cat": False,
            "similarity_threshold": 0.75,
            "suggestion_threshold": 0.60,
            "top_matches": [],
        },
    )

    # Activity detection fields
    activity: Optional[str] = Field(
        None, description="Detected cat activity", example="sleeping"
    )

    activity_confidence: Optional[float] = Field(
        None,
        description="Confidence score for the detected activity",
        example=0.85,
        ge=0.0,
        le=1.0,
    )

    # Contextual object interaction fields
    nearby_objects: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Objects detected near this cat",
        example=[
            {
                "object_class": "bowl",
                "confidence": 0.89,
                "distance": 25.5,
                "relationship": "touching",
                "interaction_type": "eating",
            }
        ],
    )

    contextual_activity: Optional[str] = Field(
        None,
        description="Activity inferred from object interactions",
        example="eating_from_bowl",
    )

    interaction_confidence: Optional[float] = Field(
        None,
        description="Confidence in the object interaction detection",
        example=0.85,
        ge=0.0,
        le=1.0,
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

    detected: bool = Field(
        ..., description="Whether any objects were detected in the image", example=True
    )
    cat_detected: bool = Field(
        ..., description="Whether any cats were detected in the image", example=True
    )
    confidence: float = Field(
        ...,
        description="Overall detection confidence (highest among all detections)",
        example=0.89,
        ge=0.0,
        le=1.0,
    )
    cats_count: int = Field(..., description="Number of cats detected", example=2, ge=0)
    detections: List[Detection] = Field(
        ..., description="List of cat detection results"
    )
    contextual_objects: List[Detection] = Field(
        default_factory=list,
        description="List of contextual objects (bowls, furniture, etc.) detected for activity analysis",
    )
