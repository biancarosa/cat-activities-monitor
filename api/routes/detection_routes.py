"""
Detection results and image analysis routes.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detections")


@router.get("/database/status",
    summary="Get Detection Database Status",
    description="Get detection database status and statistics including recent activity and detection counts.",
    response_description="Detection database status and statistics")
async def get_detection_database_status(request: Request):
    """Get detection database status and statistics."""
    try:
        database_service = request.app.state.database_service
        
        # Get all detection results from database
        all_detection_results = await database_service.get_all_detection_results()
        
        # Count statistics
        total_sources = len(all_detection_results)
        sources_with_detections = len([r for r in all_detection_results.values() if r["detected"]])
        total_cats_detected = sum(r["count"] for r in all_detection_results.values())
        
        # Get recent detection activity using async PostgreSQL methods
        async with database_service.pool.acquire() as conn:
            # Count total detection records
            total_records = await conn.fetchval('SELECT COUNT(*) FROM detection_results')
            
            # Get most recent detections
            recent_detections = await conn.fetch('''
                SELECT source_name, timestamp, count, confidence 
                FROM detection_results 
                WHERE detected = true 
                ORDER BY created_at DESC 
                LIMIT 10
            ''')
        
        return {
            "database_url": database_service.database_url,
            "database_connected": database_service.pool is not None,
            "statistics": {
                "total_sources_tracked": total_sources,
                "sources_with_detections": sources_with_detections,
                "total_cats_detected": total_cats_detected,
                "total_detection_records": total_records
            },
            "current_detection_status": {
                source: {
                    "detected": data["detected"],
                    "count": data["count"],
                    "confidence": round(data["confidence"], 3),
                    "last_detection": data["timestamp"],
                    "activities": data.get("activities", [])
                }
                for source, data in all_detection_results.items()
            },
            "recent_activity": [
                {
                    "source": row['source_name'],
                    "timestamp": row['timestamp'],
                    "cats_detected": row['count'],
                    "confidence": round(row['confidence'], 3)
                }
                for row in recent_detections
            ],
            "data_persistence": {
                "database_records": total_records,
                "persistent": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting detection database status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detection database status: {str(e)}")


@router.post("/database/cleanup",
    summary="Clean Up Detection Database",
    description="Clean up old detection results from database, keeping only recent records.",
    response_description="Database cleanup results")
async def cleanup_detection_database(request: Request, keep_days: int = 7):
    """Clean up old detection results from database."""
    try:
        database_service = request.app.state.database_service
        deleted_count = await database_service.cleanup_old_detection_results(keep_days)
        return {
            "message": f"Database cleanup completed",
            "deleted_records": deleted_count,
            "kept_days": keep_days,
            "cleanup_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error cleaning up detection database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup database: {str(e)}")


@router.get("/images",
    summary="List Detection Images",
    description="Get list of detection images with metadata and annotation information.",
    response_description="List of detection images with metadata")
async def get_detection_images(request: Request):
    """Get list of detection images with metadata - returns database data as-is."""
    try:
        config_service = request.app.state.config_service
        database_service = request.app.state.database_service
        
        config = config_service.config
        if not config:
            raise HTTPException(status_code=404, detail="No configuration loaded")
        
        detection_path = Path(config.global_.ml_model_config.detection_image_path)
        if not detection_path.exists():
            return {"images": [], "total": 0}
        
        images = []
        feedback_database = await database_service.get_all_feedback()
        
        # Get all image files in the detection directory
        for image_file in detection_path.glob("*.jpg"):
            try:
                # Parse filename to extract metadata
                # Expected format: {source}_{timestamp}_activity_detections.jpg
                filename_parts = image_file.stem.split('_')
                
                if len(filename_parts) >= 3:
                    source = filename_parts[0]
                    timestamp_str = '_'.join(filename_parts[1:3])  # date_time
                    
                    # Try to parse timestamp
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    except ValueError:
                        timestamp = datetime.fromtimestamp(image_file.stat().st_mtime)
                else:
                    source = "unknown"
                    timestamp = datetime.fromtimestamp(image_file.stat().st_mtime)
                
                # Get file stats
                file_stats = image_file.stat()
                
                # Initialize default values
                cat_count = 0
                max_confidence = 0.0
                detections = []
                activities_by_cat = {}
                annotation_summary = []
                inference_method = "no_data"
                has_detailed_annotations = False
                
                # Check for feedback data first (highest priority)
                image_feedback = None
                for feedback_id, feedback_data in feedback_database.items():
                    if feedback_data["image_filename"] == image_file.name:
                        image_feedback = feedback_data
                        break
                
                if image_feedback:
                    # Use feedback data
                    original_detections = image_feedback.get("original_detections", [])
                    user_annotations = image_feedback.get("user_annotations", [])
                    
                    # Get detection info from feedback
                    cat_detections = [d for d in original_detections if d["class_name"] in ["cat"]]
                    cat_count = len(cat_detections)
                    
                    if original_detections:
                        max_confidence = max(d["confidence"] for d in original_detections)
                    
                    detections = original_detections
                    
                    # Organize activities from user annotations
                    for i, ann in enumerate(user_annotations):
                        if ann.get("correct_activity"):
                            cat_key = str(i)
                            if cat_key not in activities_by_cat:
                                activities_by_cat[cat_key] = []
                            activities_by_cat[cat_key].append({
                                "activity": ann["correct_activity"],
                                "confidence": ann.get("activity_confidence", 0.8),
                                "reasoning": ann.get("activity_feedback", "User feedback"),
                                "cat_index": i
                            })
                    
                    # Create annotation summary
                    for i, detection in enumerate(original_detections):
                        summary = f"Cat {i+1}: {detection['confidence']:.2f} confidence"
                        annotation_summary.append(summary)
                    
                    for i, ann in enumerate(user_annotations):
                        summary_parts = []
                        if ann.get("cat_name"):
                            summary_parts.append(f"Named: {ann['cat_name']}")
                        if ann.get("correct_activity"):
                            summary_parts.append(f"Activity: {ann['correct_activity']}")
                        if ann.get("activity_feedback"):
                            summary_parts.append(f"Notes: {ann['activity_feedback'][:30]}...")
                        
                        if summary_parts:
                            annotation_summary.append(f"User annotation {i+1}: {', '.join(summary_parts)}")
                    
                    inference_method = "user_feedback"
                    has_detailed_annotations = True
                
                else:
                    # Get database data for this specific image
                    db_detection_data = await database_service.get_detection_result_by_image(image_file.name)
                    
                    if db_detection_data and db_detection_data.get("detected"):
                        # Use database data as-is for this specific image
                        cat_count = db_detection_data.get("count", 0)
                        max_confidence = db_detection_data.get("confidence", 0.0)
                        detections = db_detection_data.get("detections", [])
                        
                        # Convert database activities to activities_by_cat structure
                        db_activities = db_detection_data.get("activities", [])
                        for act in db_activities:
                            cat_idx = act.get("cat_index")
                            if cat_idx is not None:
                                cat_key = str(cat_idx)
                                if cat_key not in activities_by_cat:
                                    activities_by_cat[cat_key] = []
                                activities_by_cat[cat_key].append({
                                    "activity": act.get("activity", "unknown"),
                                    "confidence": act.get("confidence", 0.0),
                                    "reasoning": act.get("reasoning", ""),
                                    "cat_index": cat_idx
                                })
                        
                        inference_method = "database_data"
                        has_detailed_annotations = False
                    else:
                        # No database data or no detections found - skip this image
                        logger.debug(f"Skipping image {image_file.name} - no detection data available")
                        continue
                
                # Create image info - return database data as-is
                image_info = {
                    "filename": image_file.name,
                    "source": source,
                    "timestamp": timestamp.isoformat(),
                    "timestamp_display": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "file_size": file_stats.st_size,
                    "file_size_mb": round(file_stats.st_size / (1024*1024), 2),
                    "cat_count": cat_count,
                    "max_confidence": round(max_confidence, 3) if max_confidence > 0 else None,
                    "activities_by_cat": activities_by_cat,
                    "has_feedback": image_feedback is not None,
                    "has_detailed_annotations": has_detailed_annotations,
                    "inference_method": inference_method,
                    "detections": detections[:3],  # Limit to first 3 detections for performance
                    "annotation_summary": annotation_summary[:5],  # Limit to 5 summary items
                    "detection_info": {
                        "cat_count": cat_count,
                        "max_confidence": round(max_confidence, 3) if max_confidence > 0 else None,
                        "activities_by_cat": activities_by_cat,
                        "has_feedback": image_feedback is not None,
                        "detections": detections[:3]
                    }
                }
                
                images.append(image_info)
                
            except Exception as e:
                logger.warning(f"Error processing image {image_file.name}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        images.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "images": images,
            "total": len(images),
            "detection_path": str(detection_path),
            "has_feedback_data": len(feedback_database) > 0
        }
        
    except Exception as e:
        logger.error(f"Error getting detection images: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detection images: {str(e)}")


@router.get("/images/{image_filename}/annotations",
    summary="Get Image Annotations",
    description="Get detailed annotations and detection data for a specific detection image.",
    response_description="Image annotations and detection details")
async def get_image_annotations(request: Request, image_filename: str):
    """Get detailed annotations for a specific detection image."""
    try:
        config_service = request.app.state.config_service
        database_service = request.app.state.database_service
        
        config = config_service.config
        if not config:
            raise HTTPException(status_code=404, detail="No configuration loaded")
        
        detection_path = Path(config.global_.ml_model_config.detection_image_path)
        image_file = detection_path / image_filename
        
        if not image_file.exists():
            raise HTTPException(status_code=404, detail=f"Image '{image_filename}' not found")
        
        # Get feedback data for this image
        feedback_database = await database_service.get_all_feedback()
        image_feedback = None
        
        for feedback_id, feedback_data in feedback_database.items():
            if feedback_data["image_filename"] == image_filename:
                image_feedback = feedback_data
                break
        
        if not image_feedback:
            return {
                "filename": image_filename,
                "has_annotations": False,
                "message": "No annotations available for this image"
            }
        
        return {
            "filename": image_filename,
            "has_annotations": True,
            "feedback_id": feedback_id,
            "original_detections": image_feedback["original_detections"],
            "user_annotations": image_feedback["user_annotations"],
            "feedback_type": image_feedback["feedback_type"],
            "notes": image_feedback.get("notes"),
            "timestamp": image_feedback["timestamp"],
            "user_id": image_feedback.get("user_id", "anonymous")
        }
        
    except Exception as e:
        logger.error(f"Error getting image annotations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get annotations: {str(e)}")



@router.get("/status",
    summary="Get Detection Status",
    description="Get current detection status across all camera sources.",
    response_description="Current detection status for all sources")
async def get_detection_status(request: Request):
    """Get current change detection configuration and status."""
    try:
        config_service = request.app.state.config_service
        
        config = config_service.config
        if not config:
            raise HTTPException(status_code=404, detail="No configuration loaded")
        
        change_config = config.global_.ml_model_config.change_detection
        
        return {
            "change_detection": {
                "enabled": change_config.enabled,
                "similarity_threshold": change_config.similarity_threshold,
                "detection_change_threshold": change_config.detection_change_threshold,
                "position_change_threshold": change_config.position_change_threshold,
                "activity_change_triggers": change_config.activity_change_triggers
            },
            "yolo_config": {
                "save_detection_images": config.global_.ml_model_config.save_detection_images,
                "detection_image_path": config.global_.ml_model_config.detection_image_path
            }
        }
    except Exception as e:
        logger.error(f"Error getting change detection status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/classes",
    summary="Get Available Detection Classes",
    description="Get list of all available YOLO detection classes that can be detected.",
    response_description="List of available detection classes")
async def get_available_classes():
    """Get available YOLO classes."""
    from services.detection_service import COCO_CLASSES
    
    return {
        "classes": COCO_CLASSES,
        "target_classes": {
            15: "cat",
            16: "dog"
        },
        "total_classes": len(COCO_CLASSES)
    }


@router.post("/images/{image_filename}/reprocess",
    summary="Reprocess Detection Image",
    description="Reprocess a detection image with current ML model and update database with new results.",
    response_description="Reprocessed detection results")
async def reprocess_detection_image(request: Request, image_filename: str):
    """Reprocess a detection image and update database with new results."""
    try:
        config_service = request.app.state.config_service
        database_service = request.app.state.database_service
        detection_service = request.app.state.detection_service
        
        config = config_service.config
        if not config:
            raise HTTPException(status_code=404, detail="No configuration loaded")
        
        detection_path = Path(config.global_.ml_model_config.detection_image_path)
        image_file = detection_path / image_filename
        
        if not image_file.exists():
            raise HTTPException(status_code=404, detail=f"Image '{image_filename}' not found")
        
        # Extract source name from filename
        # Expected format: {source}_{timestamp}_activity_detections.jpg
        filename_parts = image_file.stem.split('_')
        if len(filename_parts) >= 3:
            source_name = filename_parts[0]
        else:
            source_name = "unknown"
        
        # Read the image file
        with open(image_file, 'rb') as f:
            image_data = f.read()
        
        # Get current YOLO config
        yolo_config = config.global_.ml_model_config
        
        # Reprocess the image with current detection service
        logger.info(f"🔄 Reprocessing detection for image: {image_filename}")
        detection_result = detection_service.detect_objects_with_activity(
            image_data, 
            yolo_config,
            source_name
        )
        
        # Delete existing database record for this image
        async with database_service.pool.acquire() as conn:
            result = await conn.execute('''
                DELETE FROM detection_results 
                WHERE image_filename = $1
            ''', image_filename)
            deleted_count = int(result.split()[-1]) if result.startswith('DELETE') else 0
        
        # Save new detection result to database
        if detection_result.detected:
            import numpy as np
            from PIL import Image
            import io
            
            # Convert image data to numpy array for database storage
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            
            await database_service.save_detection_result(
                source_name, 
                detection_result, 
                image_array, 
                image_filename
            )
            logger.info(f"✅ Reprocessed and saved detection for {image_filename}: {detection_result.count} cats detected")
        else:
            logger.info(f"ℹ️ Reprocessed {image_filename}: No cats detected")
        
        # Return the new detection results
        return {
            "filename": image_filename,
            "source": source_name,
            "reprocessed": True,
            "previous_record_deleted": deleted_count > 0,
            "detection_result": {
                "detected": detection_result.detected,
                "count": detection_result.count,
                "confidence": round(detection_result.confidence, 3) if detection_result.confidence > 0 else None,
                "detections": [
                    {
                        "class_id": d.class_id,
                        "class_name": d.class_name,
                        "confidence": round(d.confidence, 3),
                        "bounding_box": d.bounding_box
                    } for d in detection_result.detections
                ],
                "activities": [
                    {
                        "activity": a.activity.value,
                        "confidence": round(a.confidence, 3),
                        "reasoning": a.reasoning,
                        "cat_index": a.cat_index
                    } for a in detection_result.activities
                ],
                "activities_by_cat": {
                    str(cat_idx): [
                        {
                            "activity": act.activity.value,
                            "confidence": round(act.confidence, 3),
                            "reasoning": act.reasoning,
                            "cat_index": act.cat_index
                        } for act in activities
                    ] for cat_idx, activities in detection_result.cat_activities.items()
                } if detection_result.cat_activities else {},
                "primary_activity": detection_result.primary_activity.value if detection_result.primary_activity else None
            },
            "reprocess_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error reprocessing detection image {image_filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reprocess detection: {str(e)}")


@router.post("/images/reprocess-all",
    summary="Reprocess All Detection Images",
    description="Reprocess all detection images with current ML model and update database with new results.",
    response_description="Bulk reprocessing results")
async def reprocess_all_detection_images(request: Request):
    """Reprocess all detection images and update database with new results."""
    try:
        config_service = request.app.state.config_service
        database_service = request.app.state.database_service
        detection_service = request.app.state.detection_service
        
        config = config_service.config
        if not config:
            raise HTTPException(status_code=404, detail="No configuration loaded")
        
        detection_path = Path(config.global_.ml_model_config.detection_image_path)
        if not detection_path.exists():
            return {"message": "No detection images directory found", "processed": 0, "errors": 0}
        
        # Get all image files
        image_files = list(detection_path.glob("*.jpg"))
        
        if not image_files:
            return {"message": "No detection images found", "processed": 0, "errors": 0}
        
        logger.info(f"🔄 Starting bulk reprocessing of {len(image_files)} detection images")
        
        processed_count = 0
        error_count = 0
        results = []
        
        # Get current YOLO config
        yolo_config = config.global_.ml_model_config
        
        for image_file in image_files:
            try:
                # Extract source name from filename
                filename_parts = image_file.stem.split('_')
                if len(filename_parts) >= 3:
                    source_name = filename_parts[0]
                else:
                    source_name = "unknown"
                
                # Read the image file
                with open(image_file, 'rb') as f:
                    image_data = f.read()
                
                # Reprocess the image
                detection_result = detection_service.detect_objects_with_activity(
                    image_data, 
                    yolo_config,
                    source_name
                )
                
                # Delete existing database record for this image
                async with database_service.pool.acquire() as conn:
                    result = await conn.execute('''
                        DELETE FROM detection_results 
                        WHERE image_filename = $1
                    ''', image_file.name)
                    deleted_count = int(result.split()[-1]) if result.startswith('DELETE') else 0
                
                # Save new detection result to database if cats detected
                if detection_result.detected:
                    import numpy as np
                    from PIL import Image
                    import io
                    
                    # Convert image data to numpy array for database storage
                    image = Image.open(io.BytesIO(image_data))
                    image_array = np.array(image)
                    
                    await database_service.save_detection_result(
                        source_name, 
                        detection_result, 
                        image_array, 
                        image_file.name
                    )
                
                results.append({
                    "filename": image_file.name,
                    "source": source_name,
                    "success": True,
                    "detected": detection_result.detected,
                    "count": detection_result.count,
                    "confidence": round(detection_result.confidence, 3) if detection_result.confidence > 0 else None,
                    "previous_record_deleted": deleted_count > 0
                })
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    logger.info(f"📊 Bulk reprocessing progress: {processed_count}/{len(image_files)} images processed")
                
            except Exception as e:
                logger.warning(f"Error reprocessing {image_file.name}: {e}")
                results.append({
                    "filename": image_file.name,
                    "success": False,
                    "error": str(e)
                })
                error_count += 1
        
        logger.info(f"✅ Bulk reprocessing completed: {processed_count} processed, {error_count} errors")
        
        return {
            "message": f"Bulk reprocessing completed",
            "total_images": len(image_files),
            "processed": processed_count,
            "errors": error_count,
            "results": results,
            "reprocess_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during bulk reprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk reprocess: {str(e)}")


 