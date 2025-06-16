# Cat Activities Monitor - Claude Documentation

## Project Overview

This is a full-stack application that monitors cat activities using computer vision and machine learning. The system captures images from IP cameras, processes them using YOLO models to detect cats, and provides a web interface for reviewing and annotating the detections.

### Architecture
- **Frontend**: Next.js 15.3.3 with TypeScript, Tailwind CSS, and shadcn/ui components
- **Backend**: FastAPI with Python 3.11+, PostgreSQL database, YOLO ML models
- **Infrastructure**: Docker Compose for development, GitHub Actions for CI/CD

## Development Setup

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.11+ with uv (for local backend development)

### Quick Start
```bash
# Start all services
docker-compose up -d

# Access applications
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs

# Configure API URL (if not using localhost:8000)
# 1. Open http://localhost:3000/settings
# 2. Enter your API server URL
# 3. Click "Save & Test"
```

### Local Development

#### Frontend
```bash
cd frontend
npm install
npm run dev    # Starts on http://localhost:3000
npm run build  # Production build
npm run lint   # ESLint checking
```

#### Backend
```bash
cd api
uv sync
uv run uvicorn main:app --reload  # Starts on http://localhost:8000
uv run pytest                    # Run tests
uv run ruff check                # Linting
uv run black .                   # Code formatting
```

## Project Structure

```
cat-activities-monitor/
├── api/                        # FastAPI backend
│   ├── models/                 # Pydantic data models
│   ├── routes/                 # API route handlers
│   ├── services/               # Business logic services
│   ├── ml_models/              # YOLO model files
│   ├── detections/             # Generated detection images
│   ├── ml_models/training_data/ # ML training data
│   ├── config.yaml             # Application configuration
│   └── main.py                 # FastAPI application entry
├── frontend/                   # Next.js frontend
│   ├── src/app/                # Next.js App Router pages
│   ├── src/components/         # React components
│   │   └── ui/                 # shadcn/ui components
│   └── src/lib/                # Utilities and API client
└── docker-compose.yml          # Development environment
```

## Key Features

### Computer Vision
- **YOLO Integration**: Uses YOLO11 models for cat detection
- **Multi-Cat Detection**: Optimized for detecting multiple cats simultaneously
- **Activity Analysis**: Tracks cat movement and behavior patterns
- **Smart Image Saving**: Only saves images when significant changes are detected

### Web Interface
- **Image Gallery**: Browse and view detected cat activities
- **Per-Cat Feedback System**: Individual feedback for each cat detection (confirm, reject, correct, or skip)
- **Smart Annotation**: Only requires detailed annotations for confirmed/corrected cats
- **Progress Tracking**: Monitor annotation progress across all images
- **Training Management**: Export training data and retrain models with user feedback
- **Real-time Updates**: Live updates when new images are processed

### API Features
- **RESTful Design**: Clean, resource-based API endpoints
- **Real-time Processing**: Fetch new images from cameras on demand
- **Feedback Collection**: Collect human annotations for ML model training
- **Training Pipeline**: Export training data and retrain models automatically
- **System Monitoring**: Health checks and system status endpoints

## Coding Conventions

### Frontend Conventions
- **Components**: PascalCase files (e.g., `ImageGallery.tsx`)
- **Hooks**: Use `useCallback` for stable references to prevent infinite loops
- **API Calls**: Centralized in `/lib/api.ts` with full TypeScript typing
- **Styling**: Tailwind CSS with shadcn/ui components, dark theme by default
- **State Management**: Local state with `useState`, memoized callbacks for performance
- **Color System**: Consistent color utilities in `/lib/colors.ts` for cat identification
- **UI Patterns**: Color-coded indicators, responsive design, loading states

### Backend Conventions
- **Routes**: Modular route files by domain (`detection_routes.py`, `camera_routes.py`)
- **Services**: Business logic in service classes (`DetectionService`, `ImageService`)
- **Models**: Pydantic models with comprehensive validation and documentation
- **Error Handling**: HTTP exceptions with detailed error context
- **Async Operations**: Full async/await pattern throughout
- **Color System**: Consistent color assignment methods for visual coherence
- **Image Processing**: PIL-based drawing with standardized color palette

### File Naming
- **Frontend**: PascalCase for components, camelCase for utilities
- **Backend**: snake_case for all Python files
- **Configuration**: kebab-case for config files

## Configuration

### Camera Setup
Copy `api/config.example.yaml` to `api/config.yaml` and configure your camera URLs:

```yaml
images:
  - name: "living-room"
    url: "http://YOUR_CAMERA_IP:PORT/snapshot.jpg"
    interval_seconds: 30
    enabled: true
```

### ML Model Configuration
The YOLO model settings are optimized for multi-cat detection:

```yaml
global:
  ml_model_config:
    model: "ml_models/yolo11l.pt"
    confidence_threshold: 0.01  # Ultra-sensitive for detecting both cats
    iou_threshold: 0.1          # Low IoU for overlapping detections
    target_classes: [15, 16]    # Cats and dogs (YOLO sometimes confuses them)
```

## Per-Cat Feedback System

The application features an advanced per-cat feedback system that allows users to provide individual feedback for each detected cat in an image, rather than applying the same feedback to all cats.

### Feedback Types Per Cat
- **Confirm**: Mark a cat detection as correct (green highlight)
- **Reject**: Mark a cat detection as a false positive (red highlight)  
- **Correct**: Mark a cat detection as needing corrections (yellow highlight)
- **Skip**: Provide no feedback for this cat (neutral)

### Smart Overall Feedback
The system automatically determines the overall image feedback type based on individual cat actions:
- **Confirmation**: All cats confirmed, none rejected/corrected
- **Rejection**: All cats rejected, none confirmed/corrected
- **Correction**: Mixed feedback or any cats need correction

### Conditional Annotations
- **Detailed annotations** (name, activity, description) only appear for confirmed/corrected cats
- **Rejected cats** are marked as false positives without requiring detailed annotation
- **Skipped cats** are ignored in the feedback submission

### UI Features
- **Visual indicators**: Color-coded borders and backgrounds for each feedback type
- **Status messages**: Clear indication of what each feedback action means
- **Auto-determination**: Overall feedback type calculated automatically
- **Validation**: Ensures at least one cat has feedback before submission

## Training & Model Management

The application includes a comprehensive training pipeline accessible through the Settings page.

### Training Data Export
- **YOLO Format**: Exports user feedback as YOLO-compatible training dataset
- **Automatic Processing**: Converts annotations to proper label format
- **Metadata Preservation**: Includes cat names and activity information
- **Class Mapping**: Maintains COCO class compatibility

### Model Retraining
- **Fine-tuning**: Automatically fine-tunes existing models with user feedback
- **Custom Naming**: Generates timestamped model names for organization
- **Background Processing**: Training runs asynchronously with progress tracking
- **Minimum Requirements**: Requires at least 10 annotations to start training

### Training Status Monitoring
- **Real-time Progress**: Track current training jobs with progress indicators
- **Model History**: View all available models with creation dates and descriptions
- **Statistics Dashboard**: Monitor total feedback entries, annotations, and named cats
- **Job Management**: View current training job status and estimated completion

### Settings Page Features
- **Training Statistics**: Visual dashboard showing annotation progress
- **Export Button**: One-click training data export with progress indication
- **Retrain Button**: Automated model retraining with custom naming
- **Model Management**: View and manage available trained models
- **Status Monitoring**: Real-time training job progress and completion estimates
- **API Configuration**: Frontend API URL configuration with connection testing

## Common Tasks

### Adding New Components
1. Create component in `/frontend/src/components/`
2. Use TypeScript interfaces for props
3. Follow shadcn/ui patterns for styling
4. Export as default export

### Adding New API Endpoints
1. Create route in appropriate `/api/routes/` file
2. Add business logic to service class
3. Update API client in `/frontend/src/lib/api.ts`
4. Add TypeScript interfaces for request/response

### Debugging Infinite Loops
- Ensure callbacks passed to components are memoized with `useCallback`
- Check dependency arrays in `useEffect` hooks
- Avoid creating new object/function references in render cycles

### Testing
- **Frontend**: No specific test framework configured yet
- **Backend**: pytest with async support
- **Manual Testing**: Use `/docs` endpoint for API testing

## Performance Considerations

### Frontend
- Use `useCallback` for functions passed as props to prevent re-renders
- Memoize expensive computations with `useMemo`
- Avoid inline object/function creation in JSX

### Backend
- Database operations are async with connection pooling
- Image processing is optimized for batch operations
- Detection images are saved conditionally based on change detection

## Deployment

### Docker Production Build
```bash
# Build and push images (handled by GitHub Actions)
docker build -t cat-monitor-api ./api
docker build -t cat-monitor-frontend ./frontend
```

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string
- `POSTGRES_*`: Database configuration

### Frontend API Configuration
The frontend API URL is configured at runtime through the Settings page (`/settings`). Users can:
- Set a custom API URL that persists in localStorage
- Test the connection before saving
- Reset to the default `http://localhost:8000`
- No build-time environment variables are required

## Troubleshooting

### Common Issues
1. **API Connection Errors**: Use Settings page (`/settings`) to configure the correct API URL
2. **Infinite Loop in Image Fetching**: Ensure `onStatsUpdate` callback is memoized
3. **Camera Connection Errors**: Check camera URLs and network connectivity
4. **Model Loading Issues**: Ensure YOLO model files are in `ml_models/` directory
5. **Database Connection**: Verify PostgreSQL is running and accessible
6. **Color Inconsistencies**: Verify color palette matches between frontend and backend
7. **Training Data Issues**: Ensure minimum annotation requirements are met

### Debugging
- **Frontend**: Browser dev tools, React DevTools
- **Backend**: FastAPI interactive docs at `/docs`, application logs
- **Database**: PostgreSQL logs, connection status

## Contributing

### Development Workflow
When starting new work, I will ask if you'd like to work on a feature branch. If yes, I will:
1. Create a new feature branch with a descriptive name
2. Implement the requested changes
3. Commit the work with a descriptive message
4. Push the branch to the remote repository
5. Ask if you'd like me to create a pull request against the main branch

This ensures all changes are properly tracked and can be reviewed before merging. You have control over when pull requests are created.

### Before Submitting Changes
1. Run linting: `npm run lint` (frontend), `uv run ruff check` (backend)
2. Run type checking: `tsc --noEmit` (frontend)
3. Test your changes locally with `docker-compose up`
4. Ensure no infinite loops or performance regressions

### Code Review Checklist
- [ ] TypeScript interfaces for new data structures
- [ ] Error handling for API calls
- [ ] Memoized callbacks in React components
- [ ] Consistent naming conventions
- [ ] Documentation updates if needed

## API Reference

### Key Endpoints
- `GET /detections/images` - Fetch all detected images with metadata
- `POST /cameras/fetch-all` - Trigger new image capture from all cameras
- `POST /feedback` - Submit human annotations for detections
- `POST /training/export` - Export training data in YOLO format
- `POST /training/retrain` - Start model retraining with feedback data
- `GET /training/status` - Get training status and available models
- `GET /system/health` - System health check

### Response Types
All API responses are fully typed in `/frontend/src/lib/api.ts`:
- `ImageResponse` - Image metadata and detection results
- `FeedbackData` - Human annotation data
- `TrainingDataExportResult` - Training data export results
- `ModelRetrainResult` - Model retraining job information
- `TrainingStatus` - Training status and available models
- `SystemStatus` - System health information

## Color Consistency System

The application maintains visual consistency between backend-generated images and frontend UI through a coordinated color system.

### Color Palette
- **12 Predefined Colors**: Bright, distinctive colors for cat identification
- **Hash-based Assignment**: Consistent colors for named cats using string hashing
- **Index Fallback**: Anonymous cats use predictable index-based colors
- **Cross-platform Consistency**: Same colors in Python PIL drawing and TypeScript UI

### Implementation
- **Backend** (`api/services/detection_service.py`): 
  - `_get_cat_color()` method for consistent color assignment
  - Colored bounding boxes with cat index labels in detection images
- **Frontend** (`frontend/src/lib/colors.ts`):
  - `getCatColor()` and helper functions matching backend logic
  - Color utilities for backgrounds, borders, and CSS styling
- **UI Integration** (`frontend/src/components/ImageGallery.tsx`):
  - Color-coded activity indicators and cat identification
  - Colored dots in cat count badges matching detection image colors

### Color Palette
```
#FF6B6B (Red), #4ECDC4 (Teal), #45B7D1 (Blue), #96CEB4 (Green),
#FFEAA7 (Yellow), #DDA0DD (Plum), #98D8C8 (Mint), #F7DC6F (Gold),
#BB8FCE (Light Purple), #85C1E9 (Light Blue), #F8C471 (Peach), #82E0AA (Light Green)
```

## ML Model Information

### YOLO Configuration
- **Model**: YOLO11 Large (best accuracy for cat detection)
- **Classes**: Detects cats (class 15) and dogs (class 16)
- **Optimization**: Ultra-low confidence threshold for maximum sensitivity
- **Output**: Bounding boxes with confidence scores and class predictions
- **Visual Output**: Color-coded bounding boxes with cat index labels

### Activity Detection
- **Pose Analysis**: Analyzes bounding box characteristics to determine cat activities
- **Movement Tracking**: Compares positions across frames to detect movement patterns
- **Temporal Patterns**: Uses activity history to improve confidence and duration estimates
- **Activity Types**: sitting, lying, standing, moving, eating, playing, sleeping, grooming

### Training Data
- Enhanced training data stored in `/api/ml_models/training_data/`
- Metadata includes cat identification and behavioral annotations
- YOLO format export for model retraining with user feedback
- Minimum 10 annotations required for retraining initiation

## Workflow Memories

- After making changes on either the api or frontend, restart docker container of the changed service, and check their logs to see if things are still working.
```