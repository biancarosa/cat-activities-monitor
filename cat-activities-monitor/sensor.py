"""Support for Cat Activities Monitor sensors."""
from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, SENSOR_TYPES

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Cat Activities Monitor sensors."""
    coordinator = hass.data[DOMAIN][config_entry.entry_id]["coordinator"]
    
    entities = []
    
    # Create sensors for each camera
    for camera_entity in coordinator.camera_entities:
        for sensor_type in SENSOR_TYPES:
            entities.append(
                CatActivitySensor(
                    coordinator=coordinator,
                    camera_entity=camera_entity,
                    sensor_type=sensor_type,
                )
            )
    
    async_add_entities(entities)


class CatActivitySensor(CoordinatorEntity, SensorEntity):
    """Representation of a Cat Activity sensor."""

    def __init__(
        self,
        coordinator,
        camera_entity: str,
        sensor_type: str,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self.camera_entity = camera_entity
        self.sensor_type = sensor_type
        
        # Extract camera name from entity_id
        camera_name = camera_entity.replace("camera.", "").replace("_", " ").title()
        
        self._attr_name = f"{camera_name} {SENSOR_TYPES[sensor_type]['name']}"
        self._attr_unique_id = f"{camera_entity}_{sensor_type}"
        self._attr_icon = SENSOR_TYPES[sensor_type]["icon"]
        
        if "unit" in SENSOR_TYPES[sensor_type]:
            self._attr_native_unit_of_measurement = SENSOR_TYPES[sensor_type]["unit"]
        
        if SENSOR_TYPES[sensor_type]["device_class"]:
            self._attr_device_class = SENSOR_TYPES[sensor_type]["device_class"]

    @property
    def native_value(self) -> Any:
        """Return the state of the sensor."""
        if not self.coordinator.data:
            return None
            
        camera_data = self.coordinator.data.get(self.camera_entity, {})
        
        if self.sensor_type == "cat_detected":
            return camera_data.get("cat_detected", False)
        elif self.sensor_type == "cat_activity":
            return camera_data.get("activity", "unknown")
        elif self.sensor_type == "confidence":
            confidence = camera_data.get("confidence", 0)
            return round(confidence * 100, 1) if confidence else 0
        
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return additional state attributes."""
        if not self.coordinator.data:
            return {}
            
        camera_data = self.coordinator.data.get(self.camera_entity, {})
        
        attributes = {
            "camera_entity": self.camera_entity,
            "last_detection": camera_data.get("timestamp"),
        }
        
        # Add bounding box information if available
        if "bounding_box" in camera_data:
            attributes["bounding_box"] = camera_data["bounding_box"]
        
        # Add activity details if available
        if "activity_details" in camera_data:
            attributes.update(camera_data["activity_details"])
            
        return attributes

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self.coordinator.last_update_success 