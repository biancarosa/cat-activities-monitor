# Activity Detection Implementation Roadmap

## Overview

This document outlines the complete implementation roadmap for contextual activity detection in the Cat Activities Monitor. The system detects cat activities by analyzing spatial relationships between cats and environmental objects (bowls, furniture, etc.).

## Phase 1: Rule-Based + Contextual Activity Detection ✅ **COMPLETED**

### Implementation Status: COMPLETE
**Completion Date**: June 22, 2025

### Implemented Features

#### Core Components
- ✅ **ContextualActivityDetectionProcess** (`api/ml_pipeline/contextual_activity_detection.py`)
  - Rule-based activity classification using spatial relationship analysis
  - Proximity detection between cats and contextual objects
  - IoU calculation and spatial positioning analysis
  - Confidence scoring for activity predictions

#### YOLO Model Extensions
- ✅ **Extended Object Detection** (`api/ml_pipeline/yolo_detection.py`)
  - Added contextual objects: bowl, chair, sofa, bed, dining table, toilet, potted plant, sink
  - COCO class IDs: [45, 56, 57, 59, 60, 61, 58, 71]
  - Dual detection mode: cats + environmental objects

#### Data Models
- ✅ **Enhanced Detection Model** (`api/models/detection.py`)
  ```python
  activity: Optional[str]                    # Detected cat activity
  activity_confidence: Optional[float]       # Confidence score
  nearby_objects: Optional[List[Dict]]       # Contextual objects
  contextual_activity: Optional[str]         # Activity inferred from context
  interaction_confidence: Optional[float]    # Object interaction confidence
  ```

#### Configuration
- ✅ **Activity Detection Config** (`api/config.yaml`)
  ```yaml
  activity_detection:
    enabled: true
    detection_mode: contextual
    interaction_thresholds:
      proximity_distance: 100.0
      overlap_threshold: 0.1
      eating_confidence: 0.8
      sleeping_confidence: 0.7
  ```

#### Database Integration
- ✅ **Migration 950793a1114b** - Activity detection fields documentation
- ✅ Activity data stored in existing JSON `detections` column (no schema changes needed)

#### Frontend UI
- ✅ **Enhanced ImageGallery** (`frontend/src/components/ImageGallery.tsx`)
  - Activity badges with emojis and confidence scores
  - Activity information in image details
  - Updated TypeScript interfaces for activity fields

### Current Capabilities

#### Supported Activities
1. **Eating/Drinking** 🍽️💧
   - Detection: Cat near/touching bowl
   - Analysis: Head position relative to bowl
   - Confidence: 0.8+

2. **Sleeping** 😴
   - Detection: Cat on furniture in horizontal position
   - Objects: bed, sofa, chair
   - Analysis: Aspect ratio > 2.0 (width/height)
   - Confidence: 0.7+

3. **Sitting/Perching** 🪑🏔️
   - Detection: Cat above furniture/tables
   - Objects: chair, dining table
   - Analysis: Vertical positioning
   - Confidence: 0.75+

4. **Exploring** 🔍
   - Detection: Cat near plants or investigating objects
   - Objects: potted plants
   - Confidence: 0.6+

5. **Grooming** 🧼
   - Detection: Cat near litter area
   - Objects: toilet (proxy for litter box)
   - Confidence: 0.7+

6. **Pose-Based Fallback**
   - Horizontal position → sleeping
   - Vertical position → sitting
   - Default → alert

### Technical Architecture

#### Pipeline Flow
```
Image → YOLODetectionProcess → FeatureExtractionProcess → ContextualActivityDetectionProcess → Cat Identification → Output
```

#### Key Algorithms
1. **Spatial Relationship Analysis**
   - Euclidean distance calculation
   - IoU (Intersection over Union) computation
   - Relative positioning (above, below, left, right)
   - Overlap detection

2. **Interaction Classification**
   - Proximity thresholds (100px default)
   - Overlap thresholds (0.1 IoU default)
   - Context-specific rules per object type

3. **Activity Confidence Scoring**
   - Object-based confidence weights
   - Spatial relationship quality
   - Pose analysis backup scoring

---

## Phase 2: Enhanced Classification

### Objective
Improve activity detection accuracy using pose estimation and advanced computer vision techniques.

### Components to Implement

#### 1. Pose Estimation Integration
- **YOLO11-Pose Model**: Add keypoint detection for cats
- **Keypoint Analysis**: Map body joints to activity patterns
- **Implementation**: New `PoseEstimationProcess` class

#### 2. Enhanced Activity Classification
- **Head Orientation**: Detect eating vs drinking vs grooming
- **Body Posture**: Sitting vs standing vs lying distinctions
- **Limb Positioning**: Playing vs resting analysis

#### 3. Temporal Analysis Setup
- **Frame Comparison**: Basic movement detection between frames
- **State Persistence**: Track activity continuity over time

### Technical Requirements
- YOLO11-Pose model integration
- Keypoint coordinate processing
- Pose-to-activity mapping algorithms
- Enhanced confidence scoring

### Expected Improvements
- 15-20% increase in activity detection accuracy
- Better distinction between similar activities
- Reduced false positives from spatial analysis alone

---

## Phase 3: Advanced Features

### Objective
Add sophisticated behavior analysis and custom object detection capabilities.

### Components to Implement

#### 1. Custom Object Training
- **Litter Box Detection**: Train custom YOLO model for litter boxes
- **Cat Furniture**: Cat towers, scratching posts, cat beds
- **Toys**: Interactive toy detection for play behavior

#### 2. Movement Tracking
- **Optical Flow**: Track cat movement between frames
- **Activity Transitions**: Detect when cats change activities
- **Speed Analysis**: Walking vs running vs pouncing

#### 3. Advanced Behavior Patterns
- **Sequence Analysis**: Multi-step behavior recognition
- **Environmental Context**: Time-of-day activity patterns
- **Multi-Cat Interactions**: Social behavior detection

### Technical Requirements
- Custom YOLO model training pipeline
- Optical flow implementation
- Temporal sequence processing
- Multi-object tracking system

---

## Phase 4: Activity Feedback System

### Objective
Enable users to provide feedback on activity detection accuracy to improve the system.

### Components to Implement

#### 1. Enhanced Feedback Modal
- **File**: `frontend/src/components/FeedbackModal.tsx`
- **Features**:
  - Per-cat activity correction interface
  - Activity confidence rating from users
  - Contextual object interaction validation
  - Alternative activity suggestions

#### 2. Activity-Specific Feedback Models
- **Backend Models**: Extend `FeedbackAnnotation` with activity fields
  ```python
  class ActivityFeedback(BaseModel):
      original_activity: Optional[str]
      corrected_activity: str
      user_confidence: float
      interaction_correct: bool
      alternative_activities: List[str]
      feedback_notes: Optional[str]
  ```

#### 3. Feedback Collection API
- **Routes**: Update `api/routes/feedback_routes.py`
- **Endpoints**:
  - `POST /feedback/activity` - Submit activity corrections
  - `GET /feedback/activity/stats` - Activity feedback analytics
  - `GET /feedback/activity/export` - Export activity training data

#### 4. UI Components
- Activity correction dropdowns
- Confidence sliders
- Object interaction checkboxes
- Bulk activity labeling tools

### Database Extensions
```sql
ALTER TABLE feedback ADD COLUMN activity_corrections JSON;
ALTER TABLE feedback ADD COLUMN activity_metadata JSON;
```

### Implementation Priority
1. Feedback UI components
2. API endpoint updates
3. Database schema extensions
4. Analytics dashboard

---

## Phase 5: ML Training with Activity Feedback

### Objective
Use collected feedback to train and improve activity detection models.

### Components to Implement

#### 1. Activity Training Data Export
- **Enhanced Export Pipeline**: 
  - Include spatial relationship features
  - Export object interaction patterns
  - Generate balanced activity datasets
  - Include temporal context when available

#### 2. Custom Activity Classification Model
- **Model Architecture**: Lightweight neural network for activity classification
  - Input: Spatial features + pose keypoints (if available)
  - Output: Activity probabilities
  - Training: Supervised learning with user feedback

#### 3. Model Integration
- **Pipeline Integration**: Add trained model to `ContextualActivityDetectionProcess`
- **Fallback System**: Use ML model + rule-based system for robustness
- **A/B Testing**: Compare rule-based vs ML-based predictions

#### 4. Continuous Learning System
- **Automated Retraining**: Weekly/monthly model updates
- **Performance Monitoring**: Track accuracy improvements
- **Feedback Loop**: Use new predictions to collect more feedback

### Technical Components

#### Training Pipeline
```python
class ActivityModelTrainer:
    def prepare_training_data(self, feedback_data)
    def train_classifier(self, features, labels)
    def evaluate_model(self, test_data)
    def deploy_model(self, model_path)
```

#### Model Architecture
- Input features: [spatial_distance, iou, object_type, pose_features]
- Hidden layers: 2-3 dense layers with dropout
- Output: Activity class probabilities
- Loss: Categorical crossentropy

#### Performance Metrics
- Activity classification accuracy
- Per-activity precision/recall
- User feedback incorporation rate
- Model confidence calibration

---

## Development Context & Pickup Instructions

### Current System State (June 2025)

#### Working Components
1. **ML Pipeline**: `api/ml_pipeline/pipeline.py` - 3-stage processing (YOLO → Features → Activity)
2. **Detection Service**: `api/services/detection_service.py` - Integrated with activity detection
3. **Frontend**: Activity badges and display working on image gallery
4. **Configuration**: Activity detection settings in `config.yaml`

#### Key Files Modified
- `api/ml_pipeline/contextual_activity_detection.py` - Main activity detection logic
- `api/models/detection.py` - Added activity fields to Detection model
- `api/models/config.py` - Added ActivityDetectionConfig
- `frontend/src/components/ImageGallery.tsx` - Activity UI display
- `frontend/src/lib/api.ts` - Updated TypeScript interfaces

#### Testing Commands
```bash
# Restart services
docker-compose restart api frontend

# Test activity detection
curl -X POST http://localhost:8000/cameras/fetch-all

# Check logs
docker-compose logs api --tail=20
```

### Known Issues & Considerations

#### Performance
- Current system processes all COCO objects - may need optimization for production
- Activity detection adds ~200ms per image processing time
- Consider caching spatial calculations for repeated object pairs

#### Accuracy Limitations
- Rule-based system has ~70% accuracy on current test images
- Needs more sophisticated pose analysis for eating vs drinking distinction
- Temporal context missing (activities across multiple frames)

#### Configuration Tuning
- Proximity thresholds may need camera-specific adjustments
- Confidence thresholds require validation with real user feedback
- Object detection confidence may need lowering for furniture detection

### Next Phase Priorities

1. **Phase 4 (Activity Feedback)** - High impact for system improvement
2. **Phase 2 (Pose Estimation)** - Significant accuracy gains
3. **Phase 5 (ML Training)** - Long-term learning capability
4. **Phase 3 (Advanced Features)** - Nice-to-have improvements

### Code Patterns & Architecture

#### Adding New Activity Types
1. Update activity emojis in `ImageGallery.tsx`
2. Add detection logic in `ContextualActivityDetectionProcess._classify_contextual_activity()`
3. Configure confidence thresholds in `config.yaml`

#### Adding New Object Types
1. Add COCO class ID to `contextual_objects` in config
2. Update object mapping in `ContextualActivityDetectionProcess.__init__()`
3. Add interaction logic in `_determine_interaction_type()`

#### Testing New Features
1. Use debug script: `docker-compose exec api uv run python scripts/debug_cat_identification.py`
2. Check activity detection logs: `docker-compose logs api | grep contextual_activity`
3. Verify frontend display with browser dev tools

---

## Success Metrics

### Phase 1 Achievements ✅
- ✅ 8 contextual object types detected
- ✅ 6 activity categories supported
- ✅ Real-time processing (<2s per image)
- ✅ Frontend activity visualization
- ✅ Zero database schema changes (JSON storage)

### Future Phase Targets
- **Phase 2**: 85%+ activity classification accuracy
- **Phase 4**: 100+ weekly user feedback submissions
- **Phase 5**: 90%+ accuracy with ML-enhanced detection
- **Phase 3**: 12+ object types, 10+ activity categories

---

*This roadmap serves as a comprehensive guide for continuing activity detection development. Each phase builds upon the previous implementation while maintaining backward compatibility and system stability.*