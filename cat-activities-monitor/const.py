"""Constants for the Cat Activities Monitor integration."""

DOMAIN = "cat_activities_monitor"

# Configuration keys
CONF_CAMERA_ENTITIES = "camera_entities"
CONF_ANALYSIS_SERVICE_URL = "analysis_service_url"
CONF_UPDATE_INTERVAL = "update_interval"

# Default values
DEFAULT_UPDATE_INTERVAL = 30  # seconds
DEFAULT_ANALYSIS_SERVICE_URL = "http://localhost:8000/analyze"

# Cat activity states
CAT_ACTIVITIES = [
    "sleeping",
    "eating",
    "playing",
    "grooming",
    "exploring",
    "sitting",
    "lying_down",
    "walking",
    "unknown"
]

# Sensor types
SENSOR_TYPES = {
    "cat_detected": {
        "name": "Cat Detected",
        "icon": "mdi:cat",
        "device_class": "occupancy"
    },
    "cat_activity": {
        "name": "Cat Activity",
        "icon": "mdi:cat",
        "device_class": None
    },
    "confidence": {
        "name": "Detection Confidence",
        "icon": "mdi:percent",
        "unit": "%",
        "device_class": None
    }
} 