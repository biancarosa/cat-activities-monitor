#!/usr/bin/env python3
"""
Multi-Cat Detection Fine-Tuning Script

This script helps you test different YOLO parameters to optimize detection
of multiple cats in your images, specifically for the bedroom scene.
"""

import logging
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
import yaml

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

def test_detection_parameters(image_path: str, output_dir: str = "./test_results"):
    """
    Test different YOLO parameters to find optimal settings for multi-cat detection.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load image
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        return
    
    image = Image.open(image_path)
    image_array = np.array(image)
    logger.info(f"Loaded image: {image_path} ({image.size[0]}x{image.size[1]})")
    
    # Test different model sizes
    models_to_test = [
        "models/yolo11n.pt",  # Nano - fastest
        "models/yolo11s.pt",  # Small - current
        "models/yolo11m.pt",  # Medium - better accuracy
        "models/yolo11l.pt",  # Large - high accuracy
    ]
    
    # Test different parameter combinations
    test_configs = [
        # Current settings (baseline)
        {"conf": 0.1, "iou": 0.45, "max_det": 100, "imgsz": 1280, "name": "current"},
        
        # More sensitive settings
        {"conf": 0.05, "iou": 0.3, "max_det": 300, "imgsz": 1280, "name": "sensitive"},
        
        # Very sensitive settings
        {"conf": 0.01, "iou": 0.2, "max_det": 500, "imgsz": 1280, "name": "very_sensitive"},
        
        # High resolution settings
        {"conf": 0.05, "iou": 0.3, "max_det": 300, "imgsz": 1920, "name": "high_res"},
        
        # Balanced settings
        {"conf": 0.08, "iou": 0.35, "max_det": 200, "imgsz": 1280, "name": "balanced"},
    ]
    
    results = []
    
    for model_name in models_to_test:
        logger.info(f"\n🔄 Testing model: {model_name}")
        
        try:
            # Ensure models directory exists
            model_path = Path(model_name)
            if model_path.parent != Path('.'):
                model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load model
            model = YOLO(model_name)
            
            for config in test_configs:
                config_name = f"{model_name.replace('.pt', '')}_{config['name']}"
                logger.info(f"  Testing config: {config_name}")
                
                # Run detection
                results_yolo = model(
                    image_array,
                    conf=config["conf"],
                    iou=config["iou"],
                    max_det=config["max_det"],
                    imgsz=config["imgsz"],
                    verbose=False
                )
                
                # Process results
                detections = []
                cat_detections = []
                
                for result in results_yolo:
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
                            
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            detection = {
                                "class_id": class_id,
                                "class_name": class_name,
                                "confidence": confidence,
                                "bbox": [x1, y1, x2, y2]
                            }
                            
                            detections.append(detection)
                            
                            # Focus on cats and dogs
                            if class_id in [15, 16]:
                                cat_detections.append(detection)
                
                # Save annotated image
                annotated_image = image.copy()
                draw = ImageDraw.Draw(annotated_image)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except (OSError, IOError):
                    font = ImageFont.load_default()
                
                # Draw all animal detections
                for detection in cat_detections:
                    x1, y1, x2, y2 = detection["bbox"]
                    color = "lime" if detection["class_id"] == 15 else "orange"
                    
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Draw label
                    label = f"{detection['class_name']} {detection['confidence']:.3f}"
                    draw.text((x1, y1 - 20), label, fill=color, font=font)
                
                # Add configuration info to image
                config_text = f"Model: {model_name} | Conf: {config['conf']} | IoU: {config['iou']} | Max: {config['max_det']} | Size: {config['imgsz']}"
                draw.text((10, 10), config_text, fill="white", font=font)
                draw.text((10, 30), f"Cats/Dogs detected: {len(cat_detections)}", fill="white", font=font)
                
                # Save image
                output_file = output_path / f"{config_name}_detections.jpg"
                annotated_image.save(output_file)
                
                # Store results
                result_data = {
                    "model": model_name,
                    "config": config,
                    "config_name": config_name,
                    "total_detections": len(detections),
                    "cat_detections": len(cat_detections),
                    "detections": cat_detections,
                    "output_file": str(output_file)
                }
                results.append(result_data)
                
                logger.info(f"    Found {len(cat_detections)} cats/dogs (total: {len(detections)} objects)")
                
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            continue
    
    # Generate summary report
    logger.info("\n📊 DETECTION SUMMARY REPORT")
    logger.info("=" * 80)
    
    # Sort by number of cat detections (descending)
    results.sort(key=lambda x: x["cat_detections"], reverse=True)
    
    for result in results:
        logger.info(f"🎯 {result['config_name']:<25} | Cats: {result['cat_detections']} | Total: {result['total_detections']}")
        
        # Show individual detections
        for detection in result["detections"]:
            logger.info(f"   - {detection['class_name']}: {detection['confidence']:.3f}")
    
    # Save detailed results to file
    report_file = output_path / "detection_report.yaml"
    with open(report_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger.info(f"\n📄 Detailed report saved to: {report_file}")
    logger.info(f"🖼️  Annotated images saved to: {output_path}")
    
    # Recommendations
    best_result = results[0] if results else None
    if best_result and best_result["cat_detections"] >= 2:
        logger.info(f"\n🎉 RECOMMENDED SETTINGS (detected {best_result['cat_detections']} cats):")
        logger.info(f"   Model: {best_result['model']}")
        logger.info(f"   Confidence: {best_result['config']['conf']}")
        logger.info(f"   IoU: {best_result['config']['iou']}")
        logger.info(f"   Max detections: {best_result['config']['max_det']}")
        logger.info(f"   Image size: {best_result['config']['imgsz']}")
        
        # Generate config snippet
        logger.info("\n📝 CONFIG.YAML SNIPPET:")
        logger.info("global:")
        logger.info("  ml_model_config:")
        logger.info(f"    model: \"{best_result['model']}\"")
        logger.info(f"    confidence_threshold: {best_result['config']['conf']}")
        logger.info(f"    iou_threshold: {best_result['config']['iou']}")
        logger.info(f"    max_detections: {best_result['config']['max_det']}")
        logger.info(f"    image_size: {best_result['config']['imgsz']}")
    else:
        logger.warning("⚠️  No configuration detected both cats reliably. Consider:")
        logger.warning("   1. Using a larger model (models/yolo11l.pt or models/yolo11x.pt)")
        logger.warning("   2. Custom training with your specific images")
        logger.warning("   3. Image preprocessing (brightness/contrast adjustment)")

def main():
    """Main function to run the detection tests."""
    # Default to the bedroom detection image
    image_path = "../detections/bedroom_20250608_144015_activity_detections.jpg"
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        logger.error(f"Image not found: {image_path}")
        logger.info("Usage: python test_multi_cat_detection.py [image_path]")
        logger.info(f"Default image: {image_path}")
        sys.exit(1)
    
    logger.info("🐱 Multi-Cat Detection Fine-Tuning Script")
    logger.info(f"Testing image: {image_path}")
    
    test_detection_parameters(image_path)

if __name__ == "__main__":
    main() 