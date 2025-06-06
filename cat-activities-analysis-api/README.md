# Cat Activities Analysis Service

A standalone FastAPI service for analyzing images to detect cats and their activities. This service can be used by the Home Assistant Cat Activities Monitor integration or as a standalone API.

## Features

- 🐱 **Cat Detection**: Detects presence of cats in images
- 🎯 **Activity Recognition**: Identifies cat activities (sleeping, playing, eating, etc.)
- 🧠 **Multiple AI Backends**: Supports local OpenCV, YOLO models, and cloud AI services
- 🚀 **Fast API**: RESTful API with automatic documentation
- 📊 **Detailed Analysis**: Returns confidence scores and bounding boxes

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the service**:
   ```bash
   python app.py
   ```

3. **Test the service**:
   - Visit http://localhost:8000 for service info
   - Visit http://localhost:8000/docs for interactive API documentation
   - POST images to http://localhost:8000/analyze

## API Usage

### Analyze Image

**POST /analyze**

Upload an image file to analyze for cat presence and activity.

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@your_cat_image.jpg"
```

**Response:**
```json
{
  "cat_detected": true,
  "activity": "sleeping",
  "confidence": 0.85,
  "bounding_box": {
    "x1": 100, "y1": 150,
    "x2": 300, "y2": 400,
    "width": 200, "height": 250
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "image_size": {"width": 1920, "height": 1080},
  "analysis_method": "basic_cv"
}
```

### Health Check

**GET /health**

Returns service health status.

## AI Backend Options

The service supports multiple analysis methods. Edit `app.py` to choose your preferred approach:

### 1. Basic OpenCV (Default)
Uses basic computer vision techniques for detection. Good for testing but limited accuracy.

### 2. YOLO Object Detection
Uncomment the YOLO sections in `app.py` and install:
```bash
pip install ultralytics
```

Then modify the code to use:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically
```

### 3. Cloud AI Services

#### AWS Rekognition
```bash
pip install boto3
```

Configure AWS credentials and use:
```python
import boto3
rekognition = boto3.client('rekognition', region_name='us-east-1')
```

#### Google Cloud Vision
```bash
pip install google-cloud-vision
```

Set up authentication and use:
```python
from google.cloud import vision
client = vision.ImageAnnotatorClient()
```

### 4. Custom Models
Replace the analysis logic with your own trained models for cat detection and activity recognition.

## Supported Activities

- `sleeping` - Cat is resting/sleeping
- `eating` - Cat is eating or drinking
- `playing` - Cat is actively playing
- `grooming` - Cat is grooming itself
- `exploring` - Cat is moving around/exploring
- `sitting` - Cat is in sitting position
- `lying_down` - Cat is lying down but awake
- `walking` - Cat is walking/moving
- `unknown` - Activity cannot be determined

## Configuration

### Environment Variables

You can configure the service using environment variables:

```bash
export CAT_SERVICE_HOST=0.0.0.0
export CAT_SERVICE_PORT=8000
export CAT_SERVICE_DEBUG=false
```

### Custom Model Paths

If using custom models, you can specify paths:

```bash
export YOLO_MODEL_PATH=/path/to/your/model.pt
export CUSTOM_MODEL_PATH=/path/to/your/custom_model
```

## Development

### Adding New Activities

1. Update the `CAT_ACTIVITIES` list in `app.py`
2. Modify the activity analysis logic in `analyze_cat_activity_basic()` or your custom function
3. Test with sample images

### Custom Analysis Methods

Create your own analysis function:

```python
async def your_custom_analysis(img: np.ndarray) -> Dict[str, Any]:
    # Your custom analysis logic here
    return {
        "cat_detected": bool,
        "activity": str,
        "confidence": float,
        "bounding_box": dict,  # optional
        "custom_data": any     # optional
    }
```

### Testing

Test the service with sample images:

```bash
# Test with a cat image
curl -X POST "http://localhost:8000/analyze" \
     -F "image=@test_images/cat_sleeping.jpg"

# Check health
curl http://localhost:8000/health
```

## Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
EXPOSE 8000

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t cat-analysis-service .
docker run -p 8000:8000 cat-analysis-service
```

## Production Deployment

For production use:

1. **Use a production WSGI server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
   ```

2. **Add authentication** if needed
3. **Configure logging** for production
4. **Set up monitoring** and health checks
5. **Use HTTPS** with proper SSL certificates

## Integration with Home Assistant

This service is designed to work with the Cat Activities Monitor Home Assistant integration. Configure the integration to use:

- **Analysis Service URL**: `http://your-server:8000`
- **Update Interval**: 30-60 seconds (adjust based on your needs)

## Troubleshooting

### Common Issues

**Service won't start:**
- Check that port 8000 is available
- Verify all dependencies are installed
- Check Python version (3.8+ required)

**Low detection accuracy:**
- Use a proper AI model (YOLO or cloud services)
- Ensure good image quality and lighting
- Fine-tune detection thresholds

**Performance issues:**
- Use GPU acceleration for ML models
- Reduce image resolution before analysis
- Consider caching mechanisms

### Logging

Enable debug logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## License

This project is licensed under the MIT License. 