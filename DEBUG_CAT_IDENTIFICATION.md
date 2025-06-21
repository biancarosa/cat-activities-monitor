# Cat Identification Debug Guide

This guide provides tools and instructions for debugging cat identification issues in the Cat Activities Monitor system.

## Debug Script

**File:** `api/scripts/debug_cat_identification.py`

### Usage

Run the debug script inside the API container using `uv run`:

```bash
# Debug with expected cat name
docker-compose exec api uv run python scripts/debug_cat_identification.py <image_filename> <expected_cat_name>

# Debug without expected cat name
docker-compose exec api uv run python scripts/debug_cat_identification.py <image_filename>
```

### Examples

```bash
# Debug specific image expecting Chico
docker-compose exec api uv run python scripts/debug_cat_identification.py living-room_20250621_155350_activity_detections.jpg Chico

# Debug without expected cat name
docker-compose exec api uv run python scripts/debug_cat_identification.py living-room_20250621_155350_activity_detections.jpg

# Debug from inside container (if already exec'd in)
cd /app
uv run python scripts/debug_cat_identification.py living-room_20250621_155350_activity_detections.jpg Chico
```

## Debug Output Sections

### 🗄️ Database Check
- Verifies if the image exists in the detection_results table
- Shows cat count, confidence, and detection data
- Displays UUID and cat name assignments

### 📁 File System Check
- Confirms the image file exists on disk
- Shows file size and image properties
- Checks multiple possible file locations

### 🐱 Cat Profiles Check
- Lists all cat profiles in the database
- Shows which profiles have feature templates
- Indicates total detections and average confidence per cat

### 🧠 Feature Extraction
- Extracts features from the image using ResNet50
- Shows feature statistics (dimensions, mean, std, range)
- Simulates the ML pipeline feature extraction process

### 🔍 Identification Test
- Runs the cat identification algorithm
- Shows similarity scores and confidence levels
- Displays top matches with enhanced scores
- Compares results with expected cat name

### 🤖 Model Diagnostics
- Shows loaded model information
- Displays thresholds and configuration
- Lists trained cat names and metadata

## Common Issues and Solutions

### ❌ "Not found in database"
**Cause:** Image hasn't been processed by the detection pipeline
**Solution:** 
```bash
# Trigger reprocessing via API
curl -X POST http://localhost:8000/detections/reprocess/living-room_20250621_155350_activity_detections.jpg
```

### ❌ "Image file not found"
**Cause:** Image file missing from detections directory
**Solution:** Check if file exists in `api/detections/` or run fresh detection

### ❌ "No cat profiles found" or "No profiles have features"
**Cause:** Cat profiles haven't been created or don't have feature templates
**Solution:** 
1. Submit feedback for images with cat identifications
2. Ensure feedback includes cat profile assignments
3. Check that feature extraction completed successfully

### ❌ "Found 0 cat profiles with features"
**Cause:** Feature templates weren't properly saved during feedback
**Solution:**
1. Re-submit feedback for known cats
2. Check feedback processing logs for errors
3. Verify feature extraction pipeline is working

### ❌ Low similarity scores
**Cause:** Poor feature matching or lighting/pose differences
**Solutions:**
- Submit more diverse training examples
- Check image quality and crop regions
- Verify feature extraction is working correctly

### ❌ "Model not loaded"
**Cause:** Trained ML model file missing or corrupted
**Solution:**
1. Check `api/ml_models/cat_identification/` directory
2. Retrain model using Settings page
3. Verify training data requirements (minimum 10 annotations)

## Example Debug Session

```bash
$ docker-compose exec api uv run python scripts/debug_cat_identification.py living-room_20250621_155350_activity_detections.jpg Chico

================================================================================
🔍 DEBUGGING CAT IDENTIFICATION
📄 Image: living-room_20250621_155350_activity_detections.jpg
🐱 Expected Cat: Chico
================================================================================

🗄️ DATABASE CHECK
----------------------------------------
✅ Found in database
   📊 Cat count: 1
   🎯 Max confidence: 0.810
   📅 Timestamp: 2025-06-21 15:53:50.123456
   🔍 Detections: 1
      Detection 1:
         Class: cat
         Confidence: 0.810
         Cat UUID: e96b45ff-c7c2-4b88-906d-567b6954899c
         Cat Name: Chico
         Has Features: Yes
         Feature Length: 2048

🐱 CAT PROFILES CHECK
----------------------------------------
✅ Found 2 cat profiles:
   ✅ Chico
      UUID: e96b45ff-c7c2-4b88-906d-567b6954899c
      Features: Yes
      Feature length: 2048
      Total detections: 15
      Avg confidence: 0.785
      🎯 This is the expected cat!

   ✅ Pimenta
      UUID: f12a3456-b789-4c12-8def-123456789abc
      Features: Yes
      Feature length: 2048
      Total detections: 12
      Avg confidence: 0.772

📊 Summary: 2/2 profiles have features

🔍 IDENTIFICATION TEST
----------------------------------------
✅ Identification completed for 1 detections

   Detection 1 Results:
      Confidence: 0.847
      Is new cat: False
      Is confident match: True
      Similarity threshold: 0.750
      Suggestion threshold: 0.600
      Model enhanced: True
      Suggested cat: Chico
      Profile UUID: e96b45ff-c7c2-4b88-906d-567b6954899c
      Expected match: ✅ CORRECT
      Top matches:
         1. Chico: sim=0.834, enhanced=0.847, model=0.92
         2. Pimenta: sim=0.623, enhanced=0.635, model=0.08

🤖 MODEL DIAGNOSTICS
----------------------------------------
Model loaded: True
Model type: svm
Enhancement enabled: True
Enhancement weight: 0.3
Similarity threshold: 0.75
Suggestion threshold: 0.6
Trained cats: Chico, Pimenta

================================================================================
✅ Debug analysis completed
================================================================================

📋 Recent logs for this image:
=============================
2025-06-21T15:53:50.123456Z - services.cat_identification_service - INFO - High confidence match: Detection 0 -> Chico (similarity: 0.834)
2025-06-21T15:53:50.123456Z - services.detection_service - INFO - 🔄 UUID & Name assigned: Chico -> e96b45ff-c7c2-4b88-906d-567b6954899c
```

## Integration with CLAUDE.md

Add this to the troubleshooting section:

```markdown
### Cat Identification Issues
1. **Debug specific images**: Use `docker-compose exec api uv run python scripts/debug_cat_identification.py <filename> [expected_cat]`
2. **Check feature extraction**: Ensure cat profiles have feature templates
3. **Verify model loading**: Check if trained ML model is loaded properly
4. **Review similarity scores**: Minimum 0.75 for confident matches, 0.60 for suggestions
```

## When to Use Debug Scripts

1. **Cat not being recognized**: When a known cat isn't being identified
2. **Wrong cat identified**: When the system identifies the wrong cat
3. **Low confidence scores**: When similarity scores are unexpectedly low
4. **Feature extraction issues**: When features aren't being properly extracted
5. **Model performance**: When checking if the ML model enhancement is working
6. **After feedback submission**: To verify that new training data is being used

The debug scripts provide comprehensive diagnostics to identify and resolve cat identification issues quickly and systematically.