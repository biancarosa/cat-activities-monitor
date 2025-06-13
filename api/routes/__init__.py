"""
Routes package.
Contains all FastAPI route handlers organized by functionality.
"""

from . import (
    main_routes,
    system_routes,
    camera_routes,
    detection_routes,
    activity_routes,
    feedback_routes,
    training_routes,
    cat_routes
)

__all__ = [
    "main_routes",
    "system_routes",
    "camera_routes", 
    "detection_routes",
    "activity_routes",
    "feedback_routes",
    "training_routes",
    "cat_routes"
] 