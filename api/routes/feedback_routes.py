"""
Feedback routes.
"""

import hashlib
import logging
from pathlib import Path
import numpy as np
from PIL import Image

from fastapi import APIRouter, Request, HTTPException

from models import ImageFeedback
from utils import convert_datetime_fields_to_strings
from ml_pipeline.feature_extraction import FeatureExtractionProcess

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback")


@router.post("")
async def submit_feedback(request: Request, feedback: ImageFeedback):
    """Submit feedback for a detection image to improve the model."""
    try:
        config_service = request.app.state.config_service
        database_service = request.app.state.database_service
        
        config = config_service.config
        
        # Convert web path to actual file system path
        image_path = feedback.image_path
        if image_path.startswith('/detections/'):
            # Remove the leading '/detections/' and use the configured detection path
            filename = image_path[12:]  # Remove '/detections/' (11 chars + 1 for the /)
            if config:
                actual_path = Path(config.global_.ml_model_config.detection_image_path) / filename
            else:
                actual_path = Path("./detections") / filename
        elif image_path.startswith('detections/'):
            # Handle case without leading slash
            filename = image_path[11:]  # Remove 'detections/' (11 chars)
            if config:
                actual_path = Path(config.global_.ml_model_config.detection_image_path) / filename
            else:
                actual_path = Path("./detections") / filename
        else:
            # Use the path as provided
            actual_path = Path(image_path)
        
        # Validate the image file exists
        if not actual_path.exists():
            raise HTTPException(status_code=404, detail=f"Image file not found: {actual_path}")
        
        # Generate feedback ID
        feedback_id = hashlib.md5(f"{feedback.image_filename}_{feedback.timestamp}".encode()).hexdigest()
        
        # Prepare feedback data for database
        feedback_data = feedback.model_dump()
        feedback_data['image_path'] = str(actual_path)  # Store the actual file system path
        
        # Convert datetime fields to strings for PostgreSQL
        feedback_data = convert_datetime_fields_to_strings(feedback_data)
        
        # Save feedback to database
        await database_service.save_feedback(feedback_id, feedback_data)
        
        # Initialize feature extraction for cat profile updates
        feature_extractor = None
        if any(ann.cat_profile_uuid for ann in feedback.user_annotations):
            try:
                feature_extractor = FeatureExtractionProcess()
                await feature_extractor.initialize()
                logger.info("Feature extractor initialized for profile updates")
            except Exception as e:
                logger.warning(f"Could not initialize feature extractor: {e}")
        
        # Load the detection image for feature extraction
        detection_image = None
        if feature_extractor:
            try:
                detection_image = np.array(Image.open(actual_path))
                logger.info(f"Loaded detection image for feature extraction: {actual_path}")
            except Exception as e:
                logger.warning(f"Could not load detection image: {e}")
        
        # Process cat naming and profile updates
        for i, annotation in enumerate(feedback.user_annotations):
            # Only update profile if cat_profile_uuid is provided
            if annotation.cat_profile_uuid:
                existing_profile = await database_service.get_cat_profile_by_uuid(annotation.cat_profile_uuid)
                if existing_profile is not None:
                    profile = existing_profile.copy()
                    profile['last_seen_timestamp'] = feedback.timestamp.isoformat()
                    profile['total_detections'] = profile.get('total_detections', 0) + 1
                    # Update average confidence
                    old_avg = profile.get('average_confidence', 0.0)
                    total_detections = profile['total_detections']
                    new_confidence = annotation.confidence or 0.0
                    profile['average_confidence'] = ((old_avg * (total_detections - 1)) + new_confidence) / total_detections
                    # Update preferred locations
                    source_location = feedback.image_filename.split('_')[0]
                    preferred_locations = profile.get('preferred_locations', [])
                    if source_location not in preferred_locations:
                        preferred_locations.append(source_location)
                        profile['preferred_locations'] = preferred_locations
                    # Update favorite activities if provided
                    if annotation.correct_activity:
                        favorite_activities = profile.get('favorite_activities', [])
                        activity_value = annotation.correct_activity.value if hasattr(annotation.correct_activity, 'value') else annotation.correct_activity
                        if activity_value not in favorite_activities:
                            favorite_activities.append(activity_value)
                            profile['favorite_activities'] = favorite_activities
                    
                    # Extract and update feature template if detection image and bounding box available
                    if (feature_extractor and detection_image is not None and 
                        i < len(feedback.original_detections)):
                        try:
                            detection = feedback.original_detections[i]
                            if hasattr(detection, 'bounding_box') and detection.bounding_box:
                                bbox = detection.bounding_box
                                
                                # Crop cat region from detection image
                                x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
                                x2, y2 = int(bbox.get('x2', detection_image.shape[1])), int(bbox.get('y2', detection_image.shape[0]))
                                
                                # Ensure valid bounding box
                                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                                    cat_crop = detection_image[y1:y2, x1:x2]
                                    
                                    # Extract features
                                    features = await feature_extractor._extract_features(cat_crop)
                                    
                                    if features is not None and len(features) > 0:
                                        # Update cat profile with features using ensemble averaging
                                        cat_identification_service = request.app.state.cat_identification_service
                                        if cat_identification_service:
                                            async with database_service.get_session() as session:
                                                await cat_identification_service.update_cat_profile_features(
                                                    annotation.cat_profile_uuid,
                                                    features.tolist(),
                                                    session
                                                )
                                            logger.info(
                                                f"🧠 Updated feature template for {profile.get('name')} "
                                                f"with {len(features)}-dimensional features"
                                            )
                                        else:
                                            logger.warning("Cat identification service not available for feature updates")
                                    else:
                                        logger.warning(f"Failed to extract features for {profile.get('name')}")
                                else:
                                    logger.warning(f"Invalid bounding box for {profile.get('name')}: {bbox}")
                            else:
                                logger.warning(f"No bounding box available for detection {i}")
                        except Exception as e:
                            logger.error(f"Error extracting features for {profile.get('name')}: {e}")
                    
                    await database_service.save_cat_profile(profile)
                    logger.info(f"🐱 Updated cat profile: {profile.get('name')} (Total detections: {profile['total_detections']})")
        
        # Cleanup feature extractor
        if feature_extractor:
            try:
                await feature_extractor.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up feature extractor: {e}")
        
        logger.info(f"📝 Feedback submitted for {feedback.image_filename}: {feedback.feedback_type}")
        logger.info(f"   Original detections: {len(feedback.original_detections)}")
        logger.info(f"   User annotations: {len(feedback.user_annotations)}")
        profiles_with_features = sum(1 for ann in feedback.user_annotations if ann.cat_profile_uuid)
        logger.info(f"   Cat profiles updated with features: {profiles_with_features}")
        logger.info(f"   Activity feedback: {[ann.activity_feedback for ann in feedback.user_annotations if ann.activity_feedback]}")
        logger.info(f"   Original path: {image_path}")
        logger.info(f"   Resolved path: {actual_path}")
        logger.info(f"   Saved to database with ID: {feedback_id}")
        
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_id,
            "image_filename": feedback.image_filename,
            "feedback_type": feedback.feedback_type,
            "annotations_count": len(feedback.user_annotations),
            # "named_cats": [ann.cat_profile_uuid for ann in feedback.user_annotations if ann.cat_profile_uuid],
            "activity_feedback_count": len([ann for ann in feedback.user_annotations if ann.activity_feedback]),
            "timestamp": feedback.timestamp.isoformat(),
            "persisted": True
        }
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.get("")
async def list_feedback(request: Request):
    """List all submitted feedback for review."""
    try:
        database_service = request.app.state.database_service
        feedback_database = await database_service.get_all_feedback()
        feedback_summary = []
        
        for feedback_id, feedback_data in feedback_database.items():
            feedback_summary.append({
                "feedback_id": feedback_id,
                "image_filename": feedback_data["image_filename"],
                "feedback_type": feedback_data["feedback_type"],
                "annotations_count": len(feedback_data["user_annotations"]),
                "timestamp": feedback_data["timestamp"],
                "user_id": feedback_data.get("user_id", "anonymous"),
                "notes": feedback_data.get("notes")
            })
        
        # Sort by timestamp (newest first)
        feedback_summary.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "total_feedback": len(feedback_summary),
            "feedback": feedback_summary,
            "persisted": True
        }
    except Exception as e:
        logger.error(f"Error listing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{feedback_id}")
async def get_feedback_details(request: Request, feedback_id: str):
    """Get detailed feedback information."""
    try:
        database_service = request.app.state.database_service
        feedback_data = await database_service.get_feedback_by_id(feedback_id)
        
        if feedback_data is None:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        return feedback_data
    except Exception as e:
        logger.error(f"Error getting feedback details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{feedback_id}")
async def delete_feedback(request: Request, feedback_id: str):
    """Delete feedback entry."""
    try:
        database_service = request.app.state.database_service
        
        # Get feedback details before deletion
        feedback_data = await database_service.get_feedback_by_id(feedback_id)
        
        if feedback_data is None:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        # Delete from database
        deleted = await database_service.delete_feedback(feedback_id)
        
        if not deleted:
            raise HTTPException(status_code=500, detail="Failed to delete feedback")
        
        logger.info(f"🗑️ Deleted feedback for {feedback_data['image_filename']} (ID: {feedback_id})")
        
        return {
            "message": "Feedback deleted successfully",
            "deleted_feedback_id": feedback_id
        }
    except Exception as e:
        logger.error(f"Error deleting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 