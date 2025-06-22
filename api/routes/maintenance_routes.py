"""
System maintenance and cleanup routes.
"""

import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/maintenance")


class CleanupConfigRequest(BaseModel):
    retention_days: int = 3


@router.get(
    "/cleanup/status",
    summary="Get Image Cleanup Status",
    description="Get current status of image cleanup configuration and statistics.",
    response_description="Image cleanup status and configuration",
)
async def get_cleanup_status(request: Request):
    """Get current image cleanup status and configuration."""
    try:
        image_cleanup_service = request.app.state.image_cleanup_service

        status = await image_cleanup_service.get_cleanup_status()
        
        return {
            "cleanup_service": {
                "retention_days": status["retention_days"],
                "detection_path": status["detection_path"],
                "path_exists": status.get("path_exists", False),
            },
            "current_images": {
                "total_images": status.get("total_images", 0),
                "old_images": status.get("old_images", 0),
                "cutoff_date": status.get("cutoff_date"),
            },
            "background_service": {
                "enabled": True,
                "interval_hours": 24,
                "description": "Automatic cleanup runs every 24 hours"
            }
        }

    except Exception as e:
        logger.error(f"Error getting cleanup status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cleanup status: {str(e)}")


@router.post(
    "/cleanup/run",
    summary="Run Image Cleanup Manually",
    description="Manually trigger image cleanup process to delete old images and update database records.",
    response_description="Cleanup operation results",
)
async def run_cleanup_manual(request: Request):
    """Manually trigger image cleanup process."""
    try:
        config_service = request.app.state.config_service
        image_cleanup_service = request.app.state.image_cleanup_service

        config = config_service.config
        if not config:
            raise HTTPException(status_code=404, detail="No configuration loaded")

        # Set the detection path from config
        image_cleanup_service.set_detection_path(
            config.global_.ml_model_config.detection_image_path
        )

        logger.info("🧹 Manual image cleanup triggered")
        cleanup_summary = await image_cleanup_service.cleanup_old_images()

        return {
            "message": "Manual cleanup completed",
            "summary": cleanup_summary,
            "retention_days": image_cleanup_service.retention_days,
            "detection_path": str(image_cleanup_service.detection_image_path),
        }

    except Exception as e:
        logger.error(f"Error running manual cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run cleanup: {str(e)}")


@router.put(
    "/cleanup/config",
    summary="Update Cleanup Configuration",
    description="Update image cleanup configuration such as retention days.",
    response_description="Updated cleanup configuration",
)
async def update_cleanup_config(request: Request, config_request: CleanupConfigRequest):
    """Update image cleanup configuration."""
    try:
        image_cleanup_service = request.app.state.image_cleanup_service

        # Validate retention days
        if config_request.retention_days < 1:
            raise HTTPException(status_code=400, detail="Retention days must be at least 1")
        if config_request.retention_days > 365:
            raise HTTPException(status_code=400, detail="Retention days cannot exceed 365")

        # Update configuration
        old_retention = image_cleanup_service.retention_days
        image_cleanup_service.set_retention_days(config_request.retention_days)

        logger.info(
            f"📋 Cleanup configuration updated: retention days changed from {old_retention} to {config_request.retention_days}"
        )

        return {
            "message": "Cleanup configuration updated successfully",
            "old_retention_days": old_retention,
            "new_retention_days": config_request.retention_days,
            "note": "Changes will take effect on the next scheduled cleanup (runs every 24 hours)"
        }

    except Exception as e:
        logger.error(f"Error updating cleanup config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@router.get(
    "/database/orphaned-records",
    summary="Get Orphaned Database Records",
    description="Get database records that reference missing image files (image_filename is null).",
    response_description="Statistics about orphaned database records",
)
async def get_orphaned_records(request: Request):
    """Get statistics about database records with null image_filename."""
    try:
        database_service = request.app.state.database_service

        # Query database for records with null image_filename
        async with database_service.get_session() as session:
            from persistence.models import DetectionResult
            from sqlalchemy import select, func

            # Count total records
            total_stmt = select(func.count(DetectionResult.id))
            total_result = await session.execute(total_stmt)
            total_records = total_result.scalar()

            # Count orphaned records (null image_filename)
            orphaned_stmt = select(func.count(DetectionResult.id)).where(
                DetectionResult.image_filename.is_(None)
            )
            orphaned_result = await session.execute(orphaned_stmt)
            orphaned_records = orphaned_result.scalar()

            # Count records with existing image_filename
            active_stmt = select(func.count(DetectionResult.id)).where(
                DetectionResult.image_filename.is_not(None)
            )
            active_result = await session.execute(active_stmt)
            active_records = active_result.scalar()

            # Get sample orphaned records for analysis
            sample_stmt = (
                select(DetectionResult.source_name, DetectionResult.timestamp, DetectionResult.cats_count)
                .where(DetectionResult.image_filename.is_(None))
                .order_by(DetectionResult.created_at.desc())
                .limit(5)
            )
            sample_result = await session.execute(sample_stmt)
            sample_records = [
                {
                    "source_name": row.source_name,
                    "timestamp": row.timestamp,
                    "cats_count": row.cats_count,
                }
                for row in sample_result
            ]

        return {
            "database_statistics": {
                "total_records": total_records,
                "active_records": active_records,
                "orphaned_records": orphaned_records,
                "orphaned_percentage": round((orphaned_records / total_records * 100), 2) if total_records > 0 else 0,
            },
            "sample_orphaned_records": sample_records,
            "note": "Orphaned records are detection records where the image file has been cleaned up but database data is preserved for historical analysis"
        }

    except Exception as e:
        logger.error(f"Error getting orphaned records info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get orphaned records: {str(e)}")