"""
Cat profile models.
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field



class CatProfile(BaseModel):
    """
    Profile for a named cat with their characteristics.
    
    Contains identification information, physical characteristics,
    behavioral patterns, and detection statistics for individual cats.
    """
    cat_uuid: str = Field(
        ...,
        description="Unique UUID identifier for the cat profile",
        example="550e8400-e29b-41d4-a716-446655440000"
    )
    name: str = Field(
        ..., 
        description="Unique name identifier for the cat",
        example="Whiskers",
        min_length=1,
        max_length=50
    )
    description: Optional[str] = Field(
        None, 
        description="General description or notes about the cat",
        example="A friendly orange tabby who loves to play with string toys",
        max_length=500
    )
    color: Optional[str] = Field(
        None, 
        description="Primary color or color pattern of the cat",
        example="Orange tabby with white chest",
        max_length=100
    )
    breed: Optional[str] = Field(
        None, 
        description="Breed of the cat (if known)",
        example="Maine Coon",
        max_length=50
    )
    favorite_activities: List[str] = Field(
        default_factory=list, 
        description="List of activities this cat is commonly observed doing"
    )
    created_timestamp: datetime = Field(
        ..., 
        description="When this cat profile was first created",
        example="2023-12-01T14:30:22.123456"
    )
    last_seen_timestamp: Optional[datetime] = Field(
        None, 
        description="When this cat was last detected in any camera",
        example="2023-12-01T16:45:30.789012"
    )
    total_detections: int = Field(
        0, 
        description="Total number of times this cat has been detected",
        example=127,
        ge=0
    )
    average_confidence: float = Field(
        0.0, 
        description="Average detection confidence score for this cat",
        example=0.87,
        ge=0.0,
        le=1.0
    )
    preferred_locations: List[str] = Field(
        default_factory=list, 
        description="Camera/image sources where this cat is commonly seen",
        example=["living_room_camera", "kitchen_camera"]
    )
    bounding_box_color: str = Field(
        ...,
        description="Color of the bounding box for this cat",
        example="#FFA500"
    )


class CreateCatProfileRequest(BaseModel):
    """
    Request model for creating a new cat profile.
    """
    name: str = Field(
        ..., 
        description="Unique name identifier for the cat",
        example="Whiskers",
        min_length=1,
        max_length=50
    )
    description: Optional[str] = Field(
        None, 
        description="General description or notes about the cat",
        example="A friendly orange tabby who loves to play with string toys",
        max_length=500
    )
    color: Optional[str] = Field(
        None, 
        description="Primary color or color pattern of the cat",
        example="Orange tabby with white chest",
        max_length=100
    )
    breed: Optional[str] = Field(
        None, 
        description="Breed of the cat (if known)",
        example="Maine Coon",
        max_length=50
    )
    favorite_activities: List[str] = Field(
        default_factory=list, 
        description="List of activities this cat is commonly observed doing (as strings)"
    )


class UpdateCatProfileRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    breed: Optional[str] = None
    favorite_activities: Optional[List[str]] = None
    last_seen_timestamp: Optional[datetime] = None
    total_detections: Optional[int] = None
    average_confidence: Optional[float] = None
    preferred_locations: Optional[List[str]] = None
    bounding_box_color: Optional[str] = None 