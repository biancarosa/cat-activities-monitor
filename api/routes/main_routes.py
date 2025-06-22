"""
Main routes for core application endpoints.
"""

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/api",
    summary="Get API Information",
    description="Returns basic information about the Cat Activities Monitor API including version and available endpoints.",
    response_description="API metadata and endpoint information",
    responses={
        200: {
            "description": "API information retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "service": "Cat Activities Monitor API",
                        "version": "2.0.0",
                        "description": "AI-powered cat monitoring system with YOLO object detection and activity analysis",
                        "endpoints": {
                            "system": "/system/status",
                            "cameras": "/cameras",
                            "detections": "/detections/images",
                            "activities": "/activities/summary",
                            "feedback": "/feedback",
                            "training": "/training/status",
                            "cats": "/cats",
                        },
                    }
                }
            },
        }
    },
)
async def api_root():
    """API information and version."""
    return {
        "service": "Cat Activities Monitor API",
        "version": "2.0.0",
        "description": "AI-powered cat monitoring system with YOLO object detection and activity analysis",
        "endpoints": {
            "system": "/system/status",
            "cameras": "/cameras",
            "detections": "/detections/images",
            "activities": "/activities/summary",
            "feedback": "/feedback",
            "training": "/training/status",
            "cats": "/cats",
        },
    }
