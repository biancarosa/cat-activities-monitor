"""
Feedback and training models.
"""

from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .detection import Detection, CatActivity


class BoundingBox(BaseModel):
    """
    Standardized bounding box model for feedback.
    
    Represents rectangular coordinates in pixel space with additional
    width and height for convenience.
    """
    x1: float = Field(
        ..., 
        description="Left edge x-coordinate in pixels",
        example=100.5,
        ge=0.0
    )
    y1: float = Field(
        ..., 
        description="Top edge y-coordinate in pixels",
        example=150.2,
        ge=0.0
    )
    x2: float = Field(
        ..., 
        description="Right edge x-coordinate in pixels",
        example=300.8,
        ge=0.0
    )
    y2: float = Field(
        ..., 
        description="Bottom edge y-coordinate in pixels",
        example=400.1,
        ge=0.0
    )
    width: float = Field(
        ..., 
        description="Width of the bounding box in pixels",
        example=200.3,
        ge=0.0
    )
    height: float = Field(
        ..., 
        description="Height of the bounding box in pixels",
        example=249.9,
        ge=0.0
    )


class FeedbackAnnotation(BaseModel):
    """
    User-provided annotation for training data.
    
    Contains corrections, additions, or confirmations of detection results
    along with optional cat identification and activity feedback.
    """
    class_id: int = Field(
        ..., 
        description="YOLO class ID for the annotated object",
        example=15,
        ge=0
    )
    class_name: str = Field(
        ..., 
        description="Human-readable name of the annotated class",
        example="cat"
    )
    bounding_box: BoundingBox = Field(
        ..., 
        description="Corrected or confirmed bounding box coordinates"
    )
    confidence: Optional[float] = Field(
        None, 
        description="User-provided confidence rating for this annotation",
        example=0.95,
        ge=0.0,
        le=1.0
    )
    activity_feedback: Optional[str] = Field(
        None, 
        description="User's textual feedback about the cat's activity",
        example="The cat appears to be playing with a toy mouse",
        max_length=500
    )
    correct_activity: Optional[CatActivity] = Field(
        None, 
        description="User's correction of the detected activity"
    )
    activity_confidence: Optional[float] = Field(
        None, 
        description="User's confidence in their activity assessment",
        example=0.9,
        ge=0.0,
        le=1.0
    )
    cat_profile_uuid: Optional[str] = Field(
        None,
        description="UUID of the selected cat profile, if provided by the user",
        example="550e8400-e29b-41d4-a716-446655440000"
    )
    corrected_class_id: Optional[int] = Field(
        None,
        description="User-corrected YOLO class ID when the original detection was wrong",
        example=16,
        ge=0
    )
    corrected_class_name: Optional[str] = Field(
        None,
        description="Human-readable name of the corrected class",
        example="dog"
    )


class ImageFeedback(BaseModel):
    """
    Feedback data for a single detection image.
    
    Contains the original detection results along with user corrections,
    additions, and annotations for model improvement.
    """
    image_filename: str = Field(
        ..., 
        description="Name of the image file being annotated",
        example="camera1_20231201_143022_activity_detections.jpg"
    )
    image_path: str = Field(
        ..., 
        description="Full path to the image file",
        example="/detections/camera1_20231201_143022_activity_detections.jpg"
    )
    original_detections: List[Detection] = Field(
        ..., 
        description="Original detection results from the ML model"
    )
    user_annotations: List[FeedbackAnnotation] = Field(
        ..., 
        description="User-provided corrections and annotations"
    )
    feedback_type: str = Field(
        ..., 
        description="Type of feedback being provided",
        example="correction",
        pattern="^(correction|addition|confirmation|rejection)$"
    )
    notes: Optional[str] = Field(
        None, 
        description="Additional notes or comments from the user",
        example="The second cat was missed by the detector - it's partially hidden behind the first cat",
        max_length=1000
    )
    timestamp: datetime = Field(
        ..., 
        description="When this feedback was submitted",
        example="2023-12-01T14:30:22.123456"
    )
    user_id: Optional[str] = Field(
        "anonymous", 
        description="Identifier for the user providing feedback",
        example="user123",
        max_length=50
    )


class TrainingDataExport(BaseModel):
    """
    Structure for exporting training data.
    
    Contains paths to training images and labels along with metadata
    for ML model training.
    """
    images: List[str] = Field(
        ..., 
        description="Paths to training image files",
        example=[
            "/training_data/images/img001.jpg",
            "/training_data/images/img002.jpg"
        ]
    )
    labels: List[str] = Field(
        ..., 
        description="Paths to corresponding YOLO label files",
        example=[
            "/training_data/labels/img001.txt",
            "/training_data/labels/img002.txt"
        ]
    )
    classes: Dict[int, str] = Field(
        ..., 
        description="Class ID to name mapping for the training dataset",
        example={
            0: "cat",
            1: "dog"
        }
    )
    total_annotations: int = Field(
        ..., 
        description="Total number of annotations in the training dataset",
        example=1250,
        ge=0
    )
    export_timestamp: datetime = Field(
        ..., 
        description="When this training data was exported",
        example="2023-12-01T14:30:22.123456"
    )


class ModelSaveRequest(BaseModel):
    """
    Request model for saving current model with custom name.
    
    Used to save the currently loaded ML model with a custom name
    and optional description for future reference.
    """
    model_name: str = Field(
        ..., 
        description="Custom name for the saved model",
        example="my_custom_cat_model_v1",
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$"
    )
    description: Optional[str] = Field(
        None, 
        description="Optional description of the model",
        example="Custom trained model for indoor cats with high accuracy on tabby patterns",
        max_length=500
    )


class ModelRetrainRequest(BaseModel):
    """
    Request model for retraining with custom model name.
    
    Used to initiate model retraining with feedback data and specify
    a custom name for the resulting model.
    """
    custom_model_name: Optional[str] = Field(
        None, 
        description="Custom name for the retrained model",
        example="retrained_model_v2",
        max_length=100,
        pattern="^[a-zA-Z0-9_-]+$"
    )
    description: Optional[str] = Field(
        None, 
        description="Optional description of the retraining process",
        example="Retrained with 500 new annotations focusing on sleeping cats",
        max_length=500
    ) 