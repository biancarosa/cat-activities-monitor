# Scripts Directory

This directory contains utility and testing scripts for the Cat Activities Monitor API project.

## Scripts

### `test_multi_cat_detection.py`
Comprehensive testing script that evaluates different YOLO models and parameters to optimize multi-cat detection. Tests various model sizes and configuration combinations.

**Usage:**
```bash
python scripts/test_multi_cat_detection.py [image_path]
```

### `advanced_cat_detection.py`
Advanced detection script that uses image preprocessing techniques and multiple model ensembling to improve cat detection accuracy. Includes brightness, contrast, and saturation adjustments.

**Usage:**
```bash
python scripts/advanced_cat_detection.py [image_path]
```

### `test_activity_detection.py`
Tests the cat activity detection functionality to verify that the system can properly identify different cat behaviors (sitting, lying, moving, etc.).

**Usage:**
```bash
python scripts/test_activity_detection.py
```

### `test_yolo.py`
Basic YOLO installation and functionality test script. Verifies that YOLO models can be loaded and run basic inference.

**Usage:**
```bash
python scripts/test_yolo.py
```

## Running Scripts

All scripts should be run from the project root directory:

```bash
# From the project root
python scripts/script_name.py
```

The scripts will automatically handle relative paths to models, images, and output directories. 