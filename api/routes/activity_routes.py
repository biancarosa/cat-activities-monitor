"""
Activity analysis and history routes.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/activities")


@router.get("",
    summary="List Available Activities",
    description="Get a list of all available activity types that can be detected and tracked.",
    response_description="List of available activity types",
    responses={
        200: {
            "description": "Activity types retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "activities": [
                            {"id": "sleeping", "name": "Sleeping", "description": "Cat is resting or sleeping"},
                            {"id": "eating", "name": "Eating", "description": "Cat is eating food"},
                            {"id": "playing", "name": "Playing", "description": "Cat is playing or being active"},
                            {"id": "grooming", "name": "Grooming", "description": "Cat is grooming itself"}
                        ],
                        "total": 4
                    }
                }
            }
        },
        500: {"description": "Internal server error"}
    })
async def list_activities():
    """Get list of available activity types."""
    try:
        # This could be made configurable or stored in database
        activities = [
            {"id": "sleeping", "name": "Sleeping", "description": "Cat is resting or sleeping"},
            {"id": "eating", "name": "Eating", "description": "Cat is eating food"},
            {"id": "playing", "name": "Playing", "description": "Cat is playing or being active"},
            {"id": "grooming", "name": "Grooming", "description": "Cat is grooming itself"},
            {"id": "alert", "name": "Alert", "description": "Cat is alert and observing"},
            {"id": "walking", "name": "Walking", "description": "Cat is moving around"}
        ]
        
        return {
            "activities": activities,
            "total": len(activities)
        }
    except Exception as e:
        logger.error(f"Error listing activities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary",
    summary="Get Activity Summary",
    description="Get a summary of recent activity across all cameras and cats.",
    response_description="Activity summary statistics",
    responses={
        200: {
            "description": "Activity summary retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "summary": {
                            "total_detections_today": 45,
                            "active_cameras": 3,
                            "cats_detected": 2,
                            "most_active_camera": "living_room",
                            "most_common_activity": "sleeping",
                            "last_detection": "2024-01-15T14:30:00Z"
                        },
                        "by_camera": [
                            {
                                "camera_name": "living_room",
                                "detections_today": 25,
                                "last_detection": "2024-01-15T14:30:00Z",
                                "most_common_activity": "sleeping"
                            }
                        ],
                        "by_activity": [
                            {"activity": "sleeping", "count": 20, "percentage": 44.4},
                            {"activity": "eating", "count": 15, "percentage": 33.3},
                            {"activity": "playing", "count": 10, "percentage": 22.2}
                        ]
                    }
                }
            }
        },
        500: {"description": "Internal server error"}
    })
async def get_activity_summary(request: Request):
    """Get activity summary across all cameras."""
    try:
        database_service = request.app.state.database_service
        
        # Get today's date for filtering
        today = datetime.now().date()
        
        # Query database for activity summary
        with database_service.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total detections today
            cursor.execute("""
                SELECT COUNT(*) FROM detections 
                WHERE DATE(timestamp) = ?
            """, (today,))
            total_detections_today = cursor.fetchone()[0]
            
            # Active cameras (cameras with detections today)
            cursor.execute("""
                SELECT COUNT(DISTINCT image_name) FROM detections 
                WHERE DATE(timestamp) = ?
            """, (today,))
            active_cameras = cursor.fetchone()[0]
            
            # Most active camera
            cursor.execute("""
                SELECT image_name, COUNT(*) as count FROM detections 
                WHERE DATE(timestamp) = ?
                GROUP BY image_name 
                ORDER BY count DESC 
                LIMIT 1
            """, (today,))
            most_active_result = cursor.fetchone()
            most_active_camera = most_active_result[0] if most_active_result else None
            
            # Last detection
            cursor.execute("""
                SELECT timestamp FROM detections 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            last_detection_result = cursor.fetchone()
            last_detection = last_detection_result[0] if last_detection_result else None
            
            # Activity by camera
            cursor.execute("""
                SELECT image_name, COUNT(*) as count, MAX(timestamp) as last_detection
                FROM detections 
                WHERE DATE(timestamp) = ?
                GROUP BY image_name 
                ORDER BY count DESC
            """, (today,))
            by_camera = [
                {
                    "camera_name": row[0],
                    "detections_today": row[1],
                    "last_detection": row[2]
                }
                for row in cursor.fetchall()
            ]
        
        return {
            "summary": {
                "total_detections_today": total_detections_today,
                "active_cameras": active_cameras,
                "most_active_camera": most_active_camera,
                "last_detection": last_detection
            },
            "by_camera": by_camera,
            "by_activity": []  # This would need activity tracking implementation
        }
        
    except Exception as e:
        logger.error(f"Error getting activity summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{camera_name}",
    summary="Get Camera Activity History",
    description="Get activity history for a specific camera with optional date filtering.",
    response_description="Camera activity history",
    responses={
        200: {
            "description": "Activity history retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "camera_name": "living_room",
                        "total_detections": 25,
                        "date_range": {
                            "start": "2024-01-15T00:00:00Z",
                            "end": "2024-01-15T23:59:59Z"
                        },
                        "detections": [
                            {
                                "id": 123,
                                "timestamp": "2024-01-15T14:30:00Z",
                                "confidence": 0.89,
                                "detection_count": 2,
                                "bounding_boxes": [
                                    {
                                        "class_id": 15,
                                        "confidence": 0.89,
                                        "x1": 100.5, "y1": 150.2,
                                        "x2": 300.8, "y2": 400.1
                                    }
                                ]
                            }
                        ]
                    }
                }
            }
        },
        404: {"description": "Camera not found"},
        500: {"description": "Internal server error"}
    })
async def get_camera_activity_history(
    request: Request,
    camera_name: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, description="Maximum number of records to return")
):
    """Get activity history for a specific camera."""
    try:
        database_service = request.app.state.database_service
        
        # Set default date range if not provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        with database_service.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if camera exists in detections
            cursor.execute("""
                SELECT COUNT(*) FROM detections WHERE image_name = ?
            """, (camera_name,))
            
            if cursor.fetchone()[0] == 0:
                raise HTTPException(status_code=404, detail=f"No activity history found for camera '{camera_name}'")
            
            # Get detections for the camera within date range
            cursor.execute("""
                SELECT id, timestamp, confidence, detection_count, bounding_boxes
                FROM detections 
                WHERE image_name = ? 
                AND DATE(timestamp) BETWEEN ? AND ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (camera_name, start_date, end_date, limit))
            
            detections = []
            for row in cursor.fetchall():
                detection = {
                    "id": row[0],
                    "timestamp": row[1],
                    "confidence": row[2],
                    "detection_count": row[3],
                    "bounding_boxes": eval(row[4]) if row[4] else []  # Parse JSON string
                }
                detections.append(detection)
            
            return {
                "camera_name": camera_name,
                "total_detections": len(detections),
                "date_range": {
                    "start": f"{start_date}T00:00:00Z",
                    "end": f"{end_date}T23:59:59Z"
                },
                "detections": detections
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting activity history for camera '{camera_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e)) 