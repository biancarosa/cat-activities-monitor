"""
Detection results and image analysis routes.
"""

import logging
from datetime import datetime
from pathlib import Path
import json

from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detections")


@router.get("/images",
    summary="List Detection Images",
    description="Get list of detection images with metadata and annotation information.",
    response_description="List of detection images with metadata")
async def get_detection_images(request: Request, page: int = 1, limit: int = 20):
    """Get list of detection images with metadata - returns database data as-is."""
    try:
        # Validate pagination parameters
        if page < 1:
            raise HTTPException(status_code=400, detail="Page must be >= 1")
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
        config_service = request.app.state.config_service
        database_service = request.app.state.database_service
        config = config_service.config
        if not config:
            raise HTTPException(status_code=404, detail="No configuration loaded")
        detection_imgs_path = Path(config.global_.ml_model_config.detection_image_path)
        if not detection_imgs_path.exists():
            return {"images": [], "total": 0}
        feedback_database = await database_service.get_all_feedback()
        # Query the database for detection results (paginated)
        paginated_results = await database_service.get_paginated_detection_results(page, limit)
        rows = paginated_results['results']
        total_images = paginated_results['total']
        total_pages = (total_images + limit - 1) // limit if total_images > 0 else 1
        
        images = []
        for row in rows:
            image_filename = row['image_filename']
            image_file = detection_imgs_path / image_filename if image_filename else None
            if not image_file or not image_file.exists():
                continue  # Skip if image file is missing
            # Get file stats
            file_stats = image_file.stat()
            # Feedback data
            image_feedback = None
            for feedback_id, feedback_data in feedback_database.items():
                if feedback_data["image_filename"] == image_filename:
                    image_feedback = feedback_data
                    break
            # Compose image info
            cat_count = row['cats_count']
            max_confidence = row['confidence']
            detections = row['detections'] if isinstance(row['detections'], list) else (json.loads(row['detections']) if row['detections'] else [])
            annotation_summary = []
            if image_feedback:
                original_detections = image_feedback.get("original_detections", [])
                user_annotations = image_feedback.get("user_annotations", [])
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
            image_info = {
                "filename": image_filename,
                "source": row['source_name'],
                "timestamp": row['timestamp'],
                "timestamp_display": row['timestamp'],
                "file_size": file_stats.st_size,
                "file_size_mb": round(file_stats.st_size / (1024*1024), 2),
                "cat_count": cat_count,
                "max_confidence": round(max_confidence, 3) if max_confidence is not None else None,
                "has_feedback": image_feedback is not None,
                "has_detailed_annotations": image_feedback is not None,
                "inference_method": "user_feedback" if image_feedback else "database_data",
                "detections": detections[:3],  # Limit to first 3 detections for performance
                "annotation_summary": annotation_summary[:5],  # Limit to 5 summary items
                "detection_info": {
                    "max_confidence": round(max_confidence, 3) if max_confidence is not None else None,
                    "has_feedback": image_feedback is not None,
                    "detections": detections[:3]
                }
            }
            images.append(image_info)
        return {
            "images": images,
            "total": total_images,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "detection_imgs_path": str(detection_imgs_path),
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
        
        detection_imgs_path = Path(config.global_.ml_model_config.detection_image_path)
        image_file = detection_imgs_path / image_filename
        
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
    from utils import COCO_CLASSES
    
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
        
        detection_imgs_path = Path(config.global_.ml_model_config.detection_image_path)
        image_file = detection_imgs_path / image_filename
        
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
        detection_result = await detection_service.detect_objects_with_activity(
            image_data, 
            yolo_config,
            source_name
        )
        
        # Delete existing database record for this image
        deleted_count = await database_service.delete_detection_result_by_filename(image_filename)
        
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
        
        detection_imgs_path = Path(config.global_.ml_model_config.detection_image_path)
        if not detection_imgs_path.exists():
            return {"message": "No detection images directory found", "processed": 0, "errors": 0}
        
        # Get all image files
        image_files = list(detection_imgs_path.glob("*.jpg"))
        
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
                detection_result = await detection_service.detect_objects_with_activity(
                    image_data, 
                    yolo_config,
                    source_name
                )
                
                # Delete existing database record for this image
                deleted_count = await database_service.delete_detection_result_by_filename(image_file.name)
                
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
            "message": "Bulk reprocessing completed",
            "total_images": len(image_files),
            "processed": processed_count,
            "errors": error_count,
            "results": results,
            "reprocess_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during bulk reprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk reprocess: {str(e)}")


 