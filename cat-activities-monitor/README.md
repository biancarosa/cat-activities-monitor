# Cat Activities Monitor - Home Assistant Integration

A Home Assistant custom integration that monitors your cats through camera feeds, detecting their presence and analyzing their activities using AI/ML services.

## Features

- 🐱 **Cat Detection**: Automatically detects when cats are visible in camera feeds
- 🎯 **Activity Recognition**: Identifies what your cat is doing (sleeping, playing, eating, etc.)
- 📊 **Real-time Monitoring**: Continuous monitoring with configurable update intervals
- 🔌 **Easy Integration**: Simple setup through Home Assistant UI
- 🌐 **Flexible Analysis**: Supports both local analysis and external AI services
- 📈 **Rich Sensors**: Provides detection status, activity type, and confidence levels

## Project Structure

```
catsitter-ai/
├── cat-activities-monitor/     # Home Assistant Integration
│   ├── __init__.py
│   ├── manifest.json
│   ├── config_flow.py
│   ├── const.py
│   ├── sensor.py
│   ├── analyzer.py
│   └── README.md
└── analysis-service/           # Standalone Analysis Service
    ├── app.py
    ├── requirements.txt
    └── README.md
```

## Supported Activities

- Sleeping
- Eating  
- Playing
- Grooming
- Exploring
- Sitting
- Lying down
- Walking
- Unknown (when activity cannot be determined)

## Installation

### Method 1: Manual Installation

1. Copy the `cat-activities-monitor` folder to your Home Assistant `custom_components` directory:
   ```
   config/
     custom_components/
       cat_activities_monitor/
         __init__.py
         manifest.json
         config_flow.py
         const.py
         sensor.py
         analyzer.py
   ```

2. Restart Home Assistant

3. Go to **Settings** → **Devices & Services** → **Add Integration**

4. Search for "Cat Activities Monitor" and add it

### Method 2: HACS (if published)

1. Add this repository to HACS
2. Install "Cat Activities Monitor"
3. Restart Home Assistant
4. Add the integration through the UI

## Configuration

### Basic Setup

1. **Select Cameras**: Choose which camera entities you want to monitor
2. **Analysis Service URL**: Configure how images should be analyzed:
   - Leave default (`http://localhost:8000`) for the external analysis service
   - Set to `local` for basic local analysis
   - Set to your custom AI service URL
3. **Update Interval**: How often to check cameras (10-300 seconds)

### Setting up the Analysis Service (Recommended)

For better cat detection and activity recognition, run the standalone analysis service:

1. **Navigate to the analysis service directory**:
   ```bash
   cd analysis-service
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the service**:
   ```bash
   python app.py
   ```

4. **Configure the integration** to use `http://localhost:8000` as the analysis service URL

For detailed setup instructions, see the [Analysis Service README](../analysis-service/README.md).

### Advanced AI Integration

The analysis service supports multiple AI backends:

#### Option 1: YOLO Object Detection
See the [Analysis Service documentation](../analysis-service/README.md#yolo-object-detection) for setup instructions.

#### Option 2: Cloud AI Services
Integrate with AWS Rekognition, Google Cloud Vision, or other cloud services. See the [Analysis Service documentation](../analysis-service/README.md#cloud-ai-services).

#### Option 3: Custom Models
Replace the analysis logic with your own trained models for cat detection and activity recognition.

## Usage

### Sensors Created

For each monitored camera, the integration creates:

1. **Cat Detected** (`binary_sensor`): Whether a cat is currently detected
2. **Cat Activity** (`sensor`): Current activity of the detected cat
3. **Detection Confidence** (`sensor`): Confidence level (0-100%)

### Automation Examples

**Notify when cat is detected:**
```yaml
automation:
  - alias: "Cat Spotted"
    trigger:
      - platform: state
        entity_id: binary_sensor.living_room_camera_cat_detected
        to: 'on'
    action:
      - service: notify.mobile_app_your_phone
        data:
          message: "Cat detected in living room!"
```

**Track cat activities:**
```yaml
automation:
  - alias: "Cat Activity Log"
    trigger:
      - platform: state
        entity_id: sensor.living_room_camera_cat_activity
    action:
      - service: logbook.log
        data:
          name: "Cat Activity"
          message: "Cat is {{ states('sensor.living_room_camera_cat_activity') }}"
```

### Dashboard Cards

**Entity Card:**
```yaml
type: entities
entities:
  - entity: binary_sensor.living_room_camera_cat_detected
  - entity: sensor.living_room_camera_cat_activity  
  - entity: sensor.living_room_camera_detection_confidence
```

**History Graph:**
```yaml
type: history-graph
entities:
  - entity: sensor.living_room_camera_cat_activity
hours_to_show: 24
```

## Troubleshooting

### Common Issues

**Integration not appearing:**
- Ensure files are in the correct `custom_components/cat_activities_monitor/` directory
- Check that `manifest.json` is valid
- Restart Home Assistant completely

**No cat detections:**
- Verify camera entities are working and accessible
- Check the analysis service is running (if using external service)
- Review Home Assistant logs for errors
- Ensure adequate lighting in camera view

**Low detection accuracy:**
- Consider using a proper AI model (YOLO, cloud services) in the analysis service
- Adjust camera positioning and lighting
- Fine-tune detection thresholds in the analysis service

### Logs

Enable debug logging to troubleshoot issues:

```yaml
# configuration.yaml
logger:
  default: info
  logs:
    custom_components.cat_activities_monitor: debug
```

## Development

### Adding New Activities

1. Update `CAT_ACTIVITIES` in `const.py`
2. Modify the analysis logic in the analysis service
3. Test with your specific use cases

### Custom Analysis Services

Create your own analysis service by implementing:
- POST `/analyze` endpoint that accepts image uploads
- Return JSON with `cat_detected`, `activity`, `confidence` fields
- Optional: `bounding_box` and additional metadata

See the [Analysis Service documentation](../analysis-service/README.md) for examples.

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## API Reference

### Analysis Service API

The integration communicates with the analysis service via HTTP API. For complete API documentation, see the [Analysis Service README](../analysis-service/README.md#api-usage).

**POST /analyze**
- **Input**: Multipart form with `image` file
- **Output**: JSON response with cat detection and activity data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

- Home Assistant community for integration patterns
- OpenCV for computer vision capabilities
- FastAPI for the analysis service framework 