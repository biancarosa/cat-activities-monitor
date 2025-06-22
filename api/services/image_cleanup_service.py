"""
Image cleanup service for Cat Activities Monitor API.

This service handles the automated cleanup of old image files while preserving
database records for historical analysis.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from services.database_service import DatabaseService

logger = logging.getLogger(__name__)


class ImageCleanupService:
    """Service for managing automated cleanup of old detection images."""

    def __init__(self, database_service: DatabaseService, retention_days: int = 3):
        self.database_service = database_service
        self.retention_days = retention_days
        self.detection_image_path = Path("./detections")
        
    def set_detection_path(self, path: str):
        """Set the path where detection images are stored."""
        self.detection_image_path = Path(path)
        
    def set_retention_days(self, days: int):
        """Set the number of days to retain images."""
        self.retention_days = days

    async def cleanup_old_images(self) -> Dict[str, int]:
        """
        Clean up images older than retention_days.
        
        Returns a summary of the cleanup operation including:
        - images_found: Total images found on disk
        - images_deleted: Number of images physically deleted
        - database_records_updated: Number of database records updated to null filename
        - errors: Number of errors encountered
        """
        logger.info(f"🧹 Starting image cleanup process (retention: {self.retention_days} days)")
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        cutoff_timestamp = cutoff_date.strftime("%Y%m%d_%H%M%S")
        
        summary = {
            "images_found": 0,
            "images_deleted": 0,
            "database_records_updated": 0,
            "errors": 0
        }
        
        try:
            # Ensure detection directory exists
            if not self.detection_image_path.exists():
                logger.warning(f"Detection image path does not exist: {self.detection_image_path}")
                return summary
                
            # Find all image files in the detection directory
            image_files = list(self.detection_image_path.glob("*.jpg")) + \
                         list(self.detection_image_path.glob("*.jpeg")) + \
                         list(self.detection_image_path.glob("*.png"))
            
            summary["images_found"] = len(image_files)
            logger.info(f"Found {len(image_files)} image files in detection directory")
            
            old_images = []
            for image_file in image_files:
                try:
                    # Extract timestamp from filename (format: source_YYYYMMDD_HHMMSS_activity_detections.jpg)
                    filename = image_file.name
                    if self._is_image_old(filename, cutoff_timestamp):
                        old_images.append(image_file)
                except Exception as e:
                    logger.warning(f"Could not parse timestamp from filename {image_file.name}: {e}")
                    summary["errors"] += 1
                    
            logger.info(f"Found {len(old_images)} old images to clean up")
            
            # Process old images
            for image_file in old_images:
                try:
                    filename = image_file.name
                    
                    # Update database record to set image_filename to null
                    updated = await self._nullify_database_record(filename)
                    if updated:
                        summary["database_records_updated"] += 1
                        
                    # Delete the physical file
                    if image_file.exists():
                        image_file.unlink()
                        summary["images_deleted"] += 1
                        logger.debug(f"Deleted old image: {filename}")
                        
                except Exception as e:
                    logger.error(f"Error processing image {image_file.name}: {e}")
                    summary["errors"] += 1
                    
            logger.info(
                f"✅ Image cleanup completed: "
                f"{summary['images_deleted']} images deleted, "
                f"{summary['database_records_updated']} database records updated, "
                f"{summary['errors']} errors"
            )
            
        except Exception as e:
            logger.error(f"Error during image cleanup: {e}")
            summary["errors"] += 1
            
        return summary
        
    def _is_image_old(self, filename: str, cutoff_timestamp: str) -> bool:
        """
        Check if an image is older than the cutoff based on filename timestamp.
        
        Expected filename format: source_YYYYMMDD_HHMMSS_activity_detections.jpg
        """
        try:
            # Split filename and extract timestamp
            parts = filename.split('_')
            if len(parts) >= 3:
                # Extract date and time parts
                date_part = parts[1]  # YYYYMMDD
                time_part = parts[2]  # HHMMSS
                
                if len(date_part) == 8 and len(time_part) == 6:
                    timestamp = f"{date_part}_{time_part}"
                    return timestamp < cutoff_timestamp
                    
        except Exception as e:
            logger.debug(f"Could not parse timestamp from {filename}: {e}")
            
        # If we can't parse the timestamp, check file modification time as fallback
        try:
            file_path = self.detection_image_path / filename
            if file_path.exists():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                cutoff_date = datetime.now() - timedelta(days=self.retention_days)
                return file_mtime < cutoff_date
        except Exception as e:
            logger.debug(f"Could not check file mtime for {filename}: {e}")
            
        return False
        
    async def _nullify_database_record(self, filename: str) -> bool:
        """
        Update database record to set image_filename to null while preserving all other data.
        
        Returns True if a record was updated, False otherwise.
        """
        try:
            async with self.database_service.get_session() as session:
                from persistence.models import DetectionResult
                from sqlalchemy import update
                
                # Update the detection result to set image_filename to null
                stmt = (
                    update(DetectionResult)
                    .where(DetectionResult.image_filename == filename)
                    .values(image_filename=None)
                )
                
                result = await session.execute(stmt)
                await session.commit()
                
                rows_updated = result.rowcount
                if rows_updated > 0:
                    logger.debug(f"Nullified image_filename for {rows_updated} database records: {filename}")
                    return True
                else:
                    logger.debug(f"No database record found for image: {filename}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating database record for {filename}: {e}")
            return False
            
    async def get_cleanup_status(self) -> Dict[str, any]:
        """
        Get current status of images and cleanup information.
        
        Returns information about total images, old images, and cleanup configuration.
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            cutoff_timestamp = cutoff_date.strftime("%Y%m%d_%H%M%S")
            
            # Count current images
            if not self.detection_image_path.exists():
                return {
                    "detection_path": str(self.detection_image_path),
                    "retention_days": self.retention_days,
                    "total_images": 0,
                    "old_images": 0,
                    "cutoff_date": cutoff_date.isoformat(),
                    "path_exists": False
                }
                
            image_files = list(self.detection_image_path.glob("*.jpg")) + \
                         list(self.detection_image_path.glob("*.jpeg")) + \
                         list(self.detection_image_path.glob("*.png"))
            
            old_images_count = 0
            for image_file in image_files:
                try:
                    if self._is_image_old(image_file.name, cutoff_timestamp):
                        old_images_count += 1
                except Exception:
                    pass  # Skip files we can't parse
                    
            return {
                "detection_path": str(self.detection_image_path),
                "retention_days": self.retention_days,
                "total_images": len(image_files),
                "old_images": old_images_count,
                "cutoff_date": cutoff_date.isoformat(),
                "path_exists": True
            }
            
        except Exception as e:
            logger.error(f"Error getting cleanup status: {e}")
            return {
                "detection_path": str(self.detection_image_path),
                "retention_days": self.retention_days,
                "error": str(e),
                "path_exists": False
            }