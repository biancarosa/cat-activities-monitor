"""
Camera and image source routes.
"""

import logging

from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras")


@router.get("",
    summary="List Camera Sources",
    description="Get a list of all configured camera/image sources with their settings and status.",
    response_description="List of configured camera sources",
    responses={
        200: {
            "description": "Camera sources retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "cameras": [
                            {
                                "name": "living_room",
                                "url": "http://camera1.local/snapshot.jpg",
                                "interval_seconds": 30,
                                "enabled": True
                            },
                            {
                                "name": "kitchen",
                                "url": "http://camera2.local/snapshot.jpg", 
                                "interval_seconds": 60,
                                "enabled": False
                            }
                        ],
                        "total": 2,
                        "enabled": 1
                    }
                }
            }
        },
        404: {"description": "No configuration loaded"},
        500: {"description": "Internal server error"}
    })
async def list_cameras(request: Request):
    """List all configured camera sources."""
    try:
        config_service = request.app.state.config_service
        config = config_service.config
        
        if not config:
            raise HTTPException(status_code=404, detail="No configuration loaded")
        
        return {
            "cameras": [
                {
                    "name": img.name,
                    "url": str(img.url),
                    "interval_seconds": img.interval_seconds,
                    "enabled": img.enabled
                }
                for img in config.images
            ],
            "total": len(config.images),
            "enabled": len([img for img in config.images if img.enabled])
        }
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{camera_name}/fetch",
    summary="Fetch Specific Camera Image",
    description="Fetch and analyze a specific camera image by name. This will download the image, run YOLO detection, and return the results.",
    response_description="Camera image fetch and detection results",
    responses={
        200: {
            "description": "Camera image fetched and analyzed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Successfully fetched 'living_room'",
                        "camera_name": "living_room",
                        "url": "http://camera1.local/snapshot.jpg",
                        "success": True,
                        "detection": {
                            "detected": True,
                            "confidence": 0.89,
                            "count": 2,
                            "detections": [
                                {
                                    "class_id": 15,
                                    "confidence": 0.89,
                                    "bounding_box": {
                                        "x1": 100.5,
                                        "y1": 150.2,
                                        "x2": 300.8,
                                        "y2": 400.1,
                                        "width": 200.3,
                                        "height": 249.9
                                    }
                                }
                            ],
                            "total_animals": 2,
                            "activities": []
                        }
                    }
                }
            }
        },
        404: {"description": "Camera source not found or no configuration loaded"},
        500: {"description": "Failed to fetch or analyze camera image"}
    })
async def fetch_camera_image(request: Request, camera_name: str):
    """Fetch and analyze a specific camera image by name."""
    try:
        config_service = request.app.state.config_service
        image_service = request.app.state.image_service
        
        config = config_service.config
        if not config:
            raise HTTPException(status_code=404, detail="No configuration loaded")
        
        # Find the specific image configuration
        image_config = None
        for img in config.images:
            if img.name == camera_name:
                image_config = img
                break
        
        if not image_config:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_name}' not found")
        
        logger.info(f"🔍 Fetching image from '{camera_name}': {image_config.url}")
        
        # Fetch and analyze the image
        result = await image_service.fetch_single_image(
            image_config,
            config.global_.ml_model_config,
            config.global_.timeout_seconds
        )
        
        if result['success']:
            logger.info(f"✅ Successfully fetched '{camera_name}'")
            return {
                "message": f"Successfully fetched '{camera_name}'",
                "camera_name": camera_name,
                "url": str(image_config.url),
                **result
            }
        else:
            logger.error(f"❌ Failed to fetch '{camera_name}': {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch '{camera_name}': {result.get('error', 'Unknown error')}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching camera image '{camera_name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fetch-all",
    summary="Fetch All Camera Images",
    description="Fetch and analyze images from all enabled camera sources concurrently.",
    response_description="Results from fetching all camera images",
    responses={
        200: {
            "description": "Camera images fetch completed",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Fetch completed: 3/4 successful",
                        "total_cameras": 4,
                        "enabled_cameras": 4,
                        "successful": 3,
                        "failed": 1,
                        "results": [
                            {
                                "camera_name": "living_room",
                                "success": True,
                                "detection": {"detected": True, "count": 2}
                            },
                            {
                                "camera_name": "kitchen",
                                "success": False,
                                "error": "Connection timeout"
                            }
                        ]
                    }
                }
            }
        },
        404: {"description": "No configuration loaded"},
        500: {"description": "Internal server error"}
    })
async def fetch_all_cameras(request: Request):
    """Fetch and analyze images from all enabled camera sources."""
    try:
        config_service = request.app.state.config_service
        image_service = request.app.state.image_service
        
        config = config_service.config
        if not config:
            raise HTTPException(status_code=404, detail="No configuration loaded")
        
        enabled_cameras = config_service.get_enabled_images()
        
        if not enabled_cameras:
            return {
                "message": "No enabled cameras to fetch",
                "total_cameras": len(config.images),
                "enabled_cameras": 0,
                "successful": 0,
                "failed": 0,
                "results": []
            }
        
        logger.info(f"🔍 Fetching {len(enabled_cameras)} enabled camera images")
        
        # Fetch all images concurrently
        results = await image_service.fetch_all_images(
            enabled_cameras,
            config.global_.ml_model_config,
            config.global_.max_concurrent_fetches,
            config.global_.timeout_seconds
        )
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        logger.info(f"✅ Fetch completed: {successful}/{len(results)} successful")
        
        return {
            "message": f"Fetch completed: {successful}/{len(results)} successful",
            "total_cameras": len(config.images),
            "enabled_cameras": len(enabled_cameras),
            "successful": successful,
            "failed": failed,
            "results": [
                {
                    "camera_name": r.get('image_name', 'unknown'),
                    "success": r['success'],
                    **({"detection": r['detection']} if r['success'] and 'detection' in r else {}),
                    **({"error": r.get('error', 'Unknown error')} if not r['success'] else {})
                }
                for r in results
            ]
        }
        
    except Exception as e:
        logger.error(f"Error fetching all camera images: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 