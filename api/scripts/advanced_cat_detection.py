#!/usr/bin/env python3
"""
Advanced Cat Detection Script

This script uses multiple techniques to improve cat detection:
1. Image preprocessing (brightness, contrast, sharpening)
2. Multiple model ensembling
3. Region-based detection
4. Custom post-processing
"""

import logging
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# COCO class names
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella'
}

def preprocess_image(image, enhancement_type="balanced"):
    """
    Apply various preprocessing techniques to improve detection.
    """
    processed_images = []
    
    # Original image
    processed_images.append(("original", image))
    
    # Brightness adjustments
    enhancer = ImageEnhance.Brightness(image)
    bright_image = enhancer.enhance(1.3)  # 30% brighter
    processed_images.append(("bright", bright_image))
    
    dark_image = enhancer.enhance(0.7)  # 30% darker
    processed_images.append(("dark", dark_image))
    
    # Contrast adjustments
    enhancer = ImageEnhance.Contrast(image)
    high_contrast = enhancer.enhance(1.5)  # 50% more contrast
    processed_images.append(("high_contrast", high_contrast))
    
    low_contrast = enhancer.enhance(0.7)  # 30% less contrast
    processed_images.append(("low_contrast", low_contrast))
    
    # Sharpening
    enhancer = ImageEnhance.Sharpness(image)
    sharp_image = enhancer.enhance(2.0)  # Double sharpness
    processed_images.append(("sharp", sharp_image))
    
    # Color saturation
    enhancer = ImageEnhance.Color(image)
    saturated = enhancer.enhance(1.3)  # 30% more saturated
    processed_images.append(("saturated", saturated))
    
    desaturated = enhancer.enhance(0.5)  # 50% less saturated
    processed_images.append(("desaturated", desaturated))
    
    # Gaussian blur (sometimes helps with noisy images)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    processed_images.append(("slight_blur", blurred))
    
    # Edge enhancement
    edge_enhanced = image.filter(ImageFilter.EDGE_ENHANCE)
    processed_images.append(("edge_enhanced", edge_enhanced))
    
    return processed_images

def detect_with_multiple_models(image_array, models, config):
    """
    Run detection with multiple models and combine results.
    """
    all_detections = []
    
    for model_name, model in models.items():
        try:
            results = model(
                image_array,
                conf=config["conf"],
                iou=config["iou"],
                max_det=config["max_det"],
                imgsz=config["imgsz"],
                verbose=False
            )
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Focus on cats and dogs
                        if class_id in [15, 16]:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            detection = {
                                "model": model_name,
                                "class_id": class_id,
                                "class_name": COCO_CLASSES.get(class_id, f"class_{class_id}"),
                                "confidence": confidence,
                                "bbox": [x1, y1, x2, y2]
                            }
                            all_detections.append(detection)
        except Exception as e:
            logger.error(f"Error with model {model_name}: {e}")
    
    return all_detections

def non_max_suppression_custom(detections, iou_threshold=0.5):
    """
    Custom NMS that's more lenient for multiple cats.
    """
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    
    keep = []
    
    for detection in detections:
        should_keep = True
        x1, y1, x2, y2 = detection["bbox"]
        area1 = (x2 - x1) * (y2 - y1)
        
        for kept_detection in keep:
            kx1, ky1, kx2, ky2 = kept_detection["bbox"]
            
            # Calculate intersection
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                area2 = (kx2 - kx1) * (ky2 - ky1)
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > iou_threshold:
                    should_keep = False
                    break
        
        if should_keep:
            keep.append(detection)
    
    return keep

def detect_cats_advanced(image_path, output_dir="./advanced_results"):
    """
    Advanced cat detection using multiple techniques.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load image
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        return
    
    original_image = Image.open(image_path)
    logger.info(f"Loaded image: {image_path} ({original_image.size[0]}x{original_image.size[1]})")
    
    # Load multiple models
    models = {}
    model_names = ["models/yolo11s.pt", "models/yolo11m.pt", "models/yolo11l.pt"]
    
    for model_name in model_names:
        try:
            # Ensure models directory exists
            model_path = Path(model_name)
            if model_path.parent != Path('.'):
                model_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Loading model: {model_name}")
            models[model_name] = YOLO(model_name)
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
    
    if not models:
        logger.error("No models loaded successfully")
        return
    
    # Detection configurations to try
    configs = [
        {"conf": 0.01, "iou": 0.2, "max_det": 500, "imgsz": 1280, "name": "ultra_sensitive"},
        {"conf": 0.05, "iou": 0.3, "max_det": 300, "imgsz": 1280, "name": "very_sensitive"},
        {"conf": 0.1, "iou": 0.4, "max_det": 200, "imgsz": 1280, "name": "sensitive"},
    ]
    
    all_results = []
    
    # Try different image preprocessing
    processed_images = preprocess_image(original_image)
    
    for img_name, processed_img in processed_images:
        logger.info(f"\n🔄 Testing with {img_name} image preprocessing")
        img_array = np.array(processed_img)
        
        for config in configs:
            config_name = f"{img_name}_{config['name']}"
            logger.info(f"  Config: {config_name}")
            
            # Detect with multiple models
            detections = detect_with_multiple_models(img_array, models, config)
            
            # Apply custom NMS
            final_detections = non_max_suppression_custom(detections, iou_threshold=0.4)
            
            logger.info(f"    Found {len(final_detections)} cats/dogs")
            
            if final_detections:
                # Save annotated image
                annotated_image = processed_img.copy()
                draw = ImageDraw.Draw(annotated_image)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                # Draw detections
                for i, detection in enumerate(final_detections):
                    x1, y1, x2, y2 = detection["bbox"]
                    color = "lime" if detection["class_id"] == 15 else "orange"
                    
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Draw label
                    label = f"{detection['class_name']} {detection['confidence']:.3f} ({detection['model']})"
                    draw.text((x1, y1 - 20 - (i * 20)), label, fill=color, font=font)
                
                # Add config info
                config_text = f"Preprocessing: {img_name} | Config: {config['name']} | Detections: {len(final_detections)}"
                draw.text((10, 10), config_text, fill="white", font=font)
                
                # Save image
                output_file = output_path / f"{config_name}_detections.jpg"
                annotated_image.save(output_file)
                
                # Store results
                result_data = {
                    "preprocessing": img_name,
                    "config": config,
                    "config_name": config_name,
                    "detections": final_detections,
                    "detection_count": len(final_detections),
                    "output_file": str(output_file)
                }
                all_results.append(result_data)
    
    # Generate summary
    logger.info("\n📊 ADVANCED DETECTION SUMMARY")
    logger.info("=" * 80)
    
    # Sort by detection count
    all_results.sort(key=lambda x: x["detection_count"], reverse=True)
    
    best_results = [r for r in all_results if r["detection_count"] >= 2]
    
    if best_results:
        logger.info("🎉 CONFIGURATIONS THAT DETECTED MULTIPLE CATS:")
        for result in best_results:
            logger.info(f"✅ {result['config_name']:<30} | Cats: {result['detection_count']}")
            for detection in result["detections"]:
                logger.info(f"   - {detection['class_name']}: {detection['confidence']:.3f} (model: {detection['model']})")
        
        # Best configuration
        best = best_results[0]
        logger.info("\n🏆 BEST CONFIGURATION:")
        logger.info(f"   Preprocessing: {best['preprocessing']}")
        logger.info(f"   Config: {best['config']}")
        logger.info(f"   Detections: {best['detection_count']}")
        
    else:
        logger.info("📋 TOP SINGLE CAT DETECTIONS:")
        for result in all_results[:5]:
            logger.info(f"🎯 {result['config_name']:<30} | Cats: {result['detection_count']}")
            for detection in result["detections"]:
                logger.info(f"   - {detection['class_name']}: {detection['confidence']:.3f} (model: {detection['model']})")
        
        logger.warning("\n⚠️  Still only detecting one cat. Recommendations:")
        logger.warning("   1. The second cat might be too small/occluded")
        logger.warning("   2. Try manual annotation to create training data")
        logger.warning("   3. Consider using a different camera angle")
        logger.warning("   4. The second cat might be misclassified as another object")

def main():
    """Main function."""
    image_path = "../detections/bedroom_20250608_144015_activity_detections.jpg"
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        logger.info("Usage: python advanced_cat_detection.py [image_path]")
        sys.exit(1)
    
    logger.info("🚀 Advanced Multi-Cat Detection Script")
    logger.info(f"Testing image: {image_path}")
    
    detect_cats_advanced(image_path)

if __name__ == "__main__":
    main() 