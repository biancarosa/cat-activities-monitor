"""
Cat profile routes.
"""

import logging
import uuid
from datetime import datetime
import random
from utils import BOUNDING_BOX_COLORS

from fastapi import APIRouter, Request, HTTPException

from models import CreateCatProfileRequest, UpdateCatProfileRequest
from utils import convert_datetime_fields_to_strings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cats")


@router.post("")
async def create_cat_profile(request: Request, create_request: CreateCatProfileRequest):
    """Create a new cat profile."""
    try:
        database_service = request.app.state.database_service

        # Check if cat name already exists
        existing_profile = await database_service.get_cat_profile_by_name(
            create_request.name
        )
        if existing_profile is not None:
            raise HTTPException(
                status_code=400, detail=f"Cat '{create_request.name}' already exists"
            )

        # Create full cat profile data dict
        profile_data = {
            "cat_uuid": str(uuid.uuid4()),
            "name": create_request.name,
            "description": create_request.description,
            "color": create_request.color,
            "bounding_box_color": random.choice(BOUNDING_BOX_COLORS),
            "breed": create_request.breed,
            "favorite_activities": create_request.favorite_activities,
            "created_timestamp": datetime.now().isoformat(),
            "total_detections": 0,
            "average_confidence": 0.0,
            "preferred_locations": [],
        }

        # Save to database (profile_data already has string timestamps)
        await database_service.save_cat_profile(profile_data)
        logger.info(
            f"🐱 Created new cat profile: {profile_data['name']} ({profile_data['cat_uuid']})"
        )

        return {
            "message": "Cat profile created successfully",
            "cat_uuid": profile_data["cat_uuid"],
            "cat_name": profile_data["name"],
            "created_timestamp": profile_data["created_timestamp"],
            "persisted": True,
        }

    except Exception as e:
        logger.error(f"Error creating cat profile: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create cat profile: {str(e)}"
        )


@router.get("")
async def list_cat_profiles(request: Request):
    """List all cat profiles."""
    try:
        database_service = request.app.state.database_service
        cat_profiles = await database_service.get_all_cat_profiles()

        return {
            "total_cats": len(cat_profiles),
            "cats": cat_profiles,
            "data_source": "postgresql_database",
        }
    except Exception as e:
        logger.error(f"Error listing cat profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{cat_uuid}")
async def get_cat_profile(request: Request, cat_uuid: str):
    """Get detailed information about a specific cat."""
    try:
        database_service = request.app.state.database_service
        profile = await database_service.get_cat_profile_by_uuid(cat_uuid)

        if profile is None:
            raise HTTPException(status_code=404, detail=f"Cat '{cat_uuid}' not found")

        return profile
    except Exception as e:
        logger.error(f"Error getting cat profile for {cat_uuid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{cat_uuid}")
async def update_cat_profile(
    request: Request, cat_uuid: str, updated_profile: UpdateCatProfileRequest
):
    """Update an existing cat profile."""
    try:
        database_service = request.app.state.database_service

        existing_profile = await database_service.get_cat_profile_by_uuid(cat_uuid)
        if existing_profile is None:
            raise HTTPException(status_code=404, detail=f"Cat '{cat_uuid}' not found")

        # Merge provided fields with existing profile
        updated_data = {
            **existing_profile,
            **{k: v for k, v in updated_profile.model_dump().items() if v is not None},
        }
        # Ensure required fields are present
        if not updated_data.get("bounding_box_color"):
            updated_data["bounding_box_color"] = random.choice(BOUNDING_BOX_COLORS)
        if not updated_data.get("cat_uuid"):
            updated_data["cat_uuid"] = existing_profile["cat_uuid"]
        if not updated_data.get("created_timestamp"):
            updated_data["created_timestamp"] = existing_profile["created_timestamp"]

        # Convert datetime fields to strings for PostgreSQL
        updated_data = convert_datetime_fields_to_strings(updated_data)

        await database_service.save_cat_profile(cat_uuid, updated_data)
        logger.info(f"🐱 Updated cat profile: {cat_uuid}")

        return {
            "message": "Cat profile updated successfully",
            "cat_uuid": cat_uuid,
            "persisted": True,
        }
    except Exception as e:
        logger.error(f"Error updating cat profile for {cat_uuid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{cat_uuid}")
async def delete_cat_profile(request: Request, cat_uuid: str):
    """Delete a cat profile."""
    try:
        database_service = request.app.state.database_service

        existing_profile = await database_service.get_cat_profile_by_uuid(cat_uuid)
        if existing_profile is None:
            raise HTTPException(status_code=404, detail=f"Cat '{cat_uuid}' not found")

        deleted = await database_service.delete_cat_profile(cat_uuid)
        if not deleted:
            raise HTTPException(status_code=500, detail="Failed to delete cat profile")

        logger.info(f"🗑️ Deleted cat profile: {cat_uuid}")

        return {"message": "Cat profile deleted successfully", "deleted_cat": cat_uuid}
    except Exception as e:
        logger.error(f"Error deleting cat profile for {cat_uuid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-name/{cat_name}/activity-history")
async def get_cat_activity_history(request: Request, cat_name: str):
    """Get activity history for a specific named cat."""
    try:
        database_service = request.app.state.database_service

        profile = await database_service.get_cat_profile_by_name(cat_name)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"Cat '{cat_name}' not found")

        # Find feedback entries for this cat from database
        feedback_database = await database_service.get_all_feedback()
        cat_feedback = []

        for feedback_id, feedback_data in feedback_database.items():
            for annotation in feedback_data.get("user_annotations", []):
                if annotation.get("cat_name") == cat_name:
                    cat_feedback.append(
                        {
                            "feedback_id": feedback_id,
                            "timestamp": feedback_data["timestamp"],
                            "image_filename": feedback_data["image_filename"],
                            "detected_activity": annotation.get("correct_activity"),
                            "activity_feedback": annotation.get("activity_feedback"),
                            "confidence": annotation.get("activity_confidence"),
                            "bounding_box": annotation["bounding_box"],
                        }
                    )

        # Sort by timestamp (newest first)
        cat_feedback.sort(key=lambda x: x["timestamp"], reverse=True)

        return {
            "cat_name": cat_name,
            "total_feedback_entries": len(cat_feedback),
            "activity_history": cat_feedback,
            "data_source": "postgresql_database",
        }
    except Exception as e:
        logger.error(f"Error getting cat activity history for {cat_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
