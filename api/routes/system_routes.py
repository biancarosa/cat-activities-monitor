"""
System and administration routes.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system")


@router.get(
    "/status",
    summary="Get System Status",
    description="Get comprehensive system status including configuration, services, and runtime information.",
    response_description="System status information",
    responses={
        200: {
            "description": "System status retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "running",
                        "version": "2.0.0",
                        "uptime": "2h 15m 30s",
                        "configuration_loaded": True,
                        "background_task_running": True,
                        "database_connected": True,
                        "ml_model_loaded": True,
                        "enabled_cameras": 3,
                        "total_cameras": 5,
                    }
                }
            },
        },
        500: {"description": "Internal server error"},
    },
)
async def get_system_status(request: Request):
    """Get comprehensive system status."""
    try:
        config_service = request.app.state.config_service
        database_service = request.app.state.database_service
        detection_service = request.app.state.detection_service

        config = config_service.config

        # Basic status information
        status_info = {
            "status": "running",
            "version": "2.0.0",
            "configuration_loaded": config is not None,
            "database_connected": database_service is not None,
            "ml_model_loaded": (
                detection_service.ml_pipeline is not None
                if detection_service
                else False
            ),
        }

        # Add configuration details if available
        if config:
            status_info.update(
                {
                    "enabled_cameras": len(
                        [img for img in config.images if img.enabled]
                    ),
                    "total_cameras": len(config.images),
                    "ml_model": config.global_.ml_model_config.model,
                    "confidence_threshold": config.global_.ml_model_config.confidence_threshold,
                    "default_interval": config.global_.default_interval_seconds,
                }
            )

        return status_info

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/health",
    summary="Health Check",
    description="Simple health check endpoint to verify the API is responding.",
    response_description="Health status",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-15T10:30:00Z",
                    }
                }
            },
        }
    },
)
async def health_check():
    """Simple health check endpoint."""
    from datetime import datetime

    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z"}


@router.get(
    "/config",
    summary="Get Current Configuration",
    description="Retrieve the current system configuration including camera sources, YOLO settings, and detection parameters.",
    response_description="Current system configuration",
    responses={
        200: {
            "description": "Configuration retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "global_": {
                            "ml_model_config": {
                                "model": "yolo11n.pt",
                                "confidence_threshold": 0.5,
                                "target_classes": [15, 16],
                            },
                            "default_interval_seconds": 60,
                            "max_concurrent_fetches": 3,
                            "timeout_seconds": 30,
                        },
                        "images": [
                            {
                                "name": "living_room",
                                "url": "http://camera1.local/snapshot.jpg",
                                "enabled": True,
                                "interval_seconds": 30,
                            }
                        ],
                    }
                }
            },
        },
        404: {"description": "No configuration loaded"},
        500: {"description": "Internal server error"},
    },
)
async def get_current_config(request: Request):
    """Get current configuration."""
    try:
        config_service = request.app.state.config_service
        config = config_service.config

        if not config:
            raise HTTPException(status_code=404, detail="No configuration loaded")

        return config.model_dump()
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/config/reload",
    summary="Reload Configuration",
    description="Reload the system configuration from the config file.",
    response_description="Configuration reload result",
    responses={
        200: {
            "description": "Configuration reloaded successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Configuration reloaded successfully",
                        "total_images": 5,
                        "enabled_images": 3,
                        "ml_model": "yolo11n.pt",
                    }
                }
            },
        },
        500: {"description": "Failed to reload configuration"},
    },
)
async def reload_configuration(request: Request):
    """Reload configuration from file."""
    try:
        config_service = request.app.state.config_service
        config = config_service.load_config()

        logger.info("Configuration reloaded successfully")

        return {
            "message": "Configuration reloaded successfully",
            "total_images": len(config.images),
            "enabled_images": len([img for img in config.images if img.enabled]),
            "ml_model": config.global_.ml_model_config.model,
        }
    except Exception as e:
        logger.error(f"Error reloading configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/logs",
    summary="Get Recent Logs",
    description="Retrieve recent log entries from the application log file.",
    response_description="Recent log entries",
    responses={
        200: {
            "description": "Log entries retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "logs": [
                            "2024-01-15 10:30:00 - INFO - Background image fetcher started",
                            "2024-01-15 10:29:45 - INFO - ML model loaded: yolo11n.pt",
                        ],
                        "total_lines": 2,
                    }
                }
            },
        },
        500: {"description": "Failed to read log file"},
    },
)
async def get_recent_logs(lines: int = 100):
    """Get recent log entries."""
    try:
        log_file = Path("api.log")
        if not log_file.exists():
            return {"logs": [], "total_lines": 0}

        with open(log_file, "r") as f:
            all_lines = f.readlines()

        # Get the last N lines
        recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        # Clean up the lines (remove newlines)
        clean_lines = [line.strip() for line in recent_lines if line.strip()]

        return {"logs": clean_lines, "total_lines": len(clean_lines)}
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
