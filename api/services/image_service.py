"""
Image service for Cat Activities Monitor API.
"""

import asyncio
import logging
from typing import Optional, Tuple
from datetime import datetime

import httpx

from models import ImageConfig, ImageDetections

logger = logging.getLogger(__name__)


class ImageService:
    """Service for fetching and processing images."""
    
    def __init__(self, detection_service, database_service):
        self.detection_service = detection_service
        self.database_service = database_service
    
    async def fetch_image_from_url(self, image_config: ImageConfig, yolo_config, timeout: int = 30) -> Tuple[bool, Optional[ImageDetections]]:
        """
        Fetch an image from the given URL, detect objects, and log results.
        Returns (success, detection_result).
        """
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.info(f"Fetching image '{image_config.name}' from URL: {image_config.url}")
                response = await client.get(str(image_config.url))
                response.raise_for_status()
                
                # Check if the response contains image data
                content_type = response.headers.get("content-type", "")
                image_data = None
                
                # Handle JSON responses (like dog.ceo API)
                if content_type.startswith("application/json"):
                    try:
                        json_data = response.json()
                        if "message" in json_data and json_data["message"].startswith("http"):
                            # This is likely a URL to an actual image, fetch that too
                            image_url = json_data["message"]
                            logger.info(f"Found image URL in JSON response: {image_url}")
                            image_response = await client.get(image_url)
                            image_response.raise_for_status()
                            content_type = image_response.headers.get("content-type", "")
                            image_data = image_response.content
                        else:
                            logger.warning("JSON response doesn't contain expected image URL structure")
                            return False, None
                    except Exception as e:
                        logger.error(f"Error processing JSON response: {e}")
                        return False, None
                elif content_type.startswith("image/"):
                    image_data = response.content
                else:
                    logger.warning(f"URL '{image_config.name}' did not return an image. Content-Type: {content_type}")
                    return False, None
                
                image_size = len(image_data)
                logger.info(f"Successfully fetched image '{image_config.name}'! Size: {image_size} bytes, Content-Type: {content_type}")
                
                # Perform object detection
                detection_result = None
                if self.detection_service.ml_model and yolo_config:
                    try:
                        detection_result = self.detection_service.detect_objects_with_activity(
                            image_data, 
                            yolo_config,
                            image_config.name
                        )
                        
                        if detection_result.cat_detected:
                            from services.detection_service import COCO_CLASSES
                            target_classes = [COCO_CLASSES.get(cls, f"class_{cls}") for cls in yolo_config.target_classes]
                            logger.info(f"🎯 TARGET OBJECTS DETECTED in '{image_config.name}'! "
                                      f"Count: {detection_result.cats_count}, "
                                      f"Max confidence: {detection_result.confidence:.2f}, "
                                      f"Classes: {target_classes}")
                            
                            # Log individual detections
                            for detection in detection_result.detections:
                                logger.info(f"  - {detection.class_name}: {detection.confidence:.2f}")
                        else:
                            logger.info(f"No target objects detected in '{image_config.name}' "
                                      f"(Total cats: {detection_result.cats_count})")
                        
                        # Save detection result to database for persistence
                        try:
                            import numpy as np
                            from PIL import Image
                            import io
                            
                            # Convert image data to numpy array for database storage
                            image = Image.open(io.BytesIO(image_data))
                            image_array = np.array(image)
                            
                            # Generate image filename for this detection
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            image_filename = f"{image_config.name}_{timestamp}_activity_detections.jpg"
                            
                            await self.database_service.save_detection_result(
                                image_config.name, 
                                detection_result, 
                                image_array, 
                                image_filename
                            )
                            logger.debug(f"💾 Saved detection result to database: {image_config.name} -> {image_filename}")
                        except Exception as e:
                            logger.warning(f"Failed to save detection result to database: {e}")
                            
                    except Exception as e:
                        logger.error(f"Error during object detection for '{image_config.name}': {e}")
                
                return True, detection_result
                
        except httpx.TimeoutException:
            logger.error(f"Timeout while fetching image '{image_config.name}' from {image_config.url}")
            return False, None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} while fetching image '{image_config.name}' from {image_config.url}")
            return False, None
        except Exception as e:
            logger.error(f"Unexpected error while fetching image '{image_config.name}' from {image_config.url}: {str(e)}")
            return False, None
    
    async def fetch_all_images(self, image_configs, yolo_config, max_concurrent: int = 5, timeout: int = 30):
        """Fetch all images concurrently with a semaphore limit."""
        results = []
        
        # Limit concurrent fetches
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(img_config):
            async with semaphore:
                success, detection_result = await self.fetch_image_from_url(img_config, yolo_config, timeout)
                result = {
                    "name": img_config.name,
                    "success": success,
                    "url": str(img_config.url)
                }
                if detection_result:
                    result["detection"] = detection_result.model_dump()
                return result
        
        # Create tasks for all images
        tasks = [fetch_with_semaphore(img) for img in image_configs]
        results = await asyncio.gather(*tasks)
        
        return results 