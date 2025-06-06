"""The Cat Activities Monitor integration."""
from __future__ import annotations

import logging
from datetime import timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.CAMERA]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Cat Activities Monitor from a config entry."""
    hass.data.setdefault(DOMAIN, {})
    
    # Create coordinator for managing data updates
    coordinator = CatActivitiesCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    
    hass.data[DOMAIN][entry.entry_id] = {
        "coordinator": coordinator,
    }
    
    # Forward the setup to the sensor and camera platforms
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data[DOMAIN].pop(entry.entry_id)
    
    return unload_ok


class CatActivitiesCoordinator(DataUpdateCoordinator):
    """Coordinator for cat activities monitoring."""
    
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=30),  # Check every 30 seconds
        )
        self.entry = entry
        self.camera_entities = entry.data.get("camera_entities", [])
        self.analysis_service_url = entry.data.get("analysis_service_url", "")
    
    async def _async_update_data(self):
        """Fetch data from cameras and analyze for cat activities."""
        try:
            from .analyzer import CatActivityAnalyzer
            
            analyzer = CatActivityAnalyzer(self.analysis_service_url)
            results = {}
            
            for camera_entity in self.camera_entities:
                # Get snapshot from camera
                snapshot = await self._get_camera_snapshot(camera_entity)
                if snapshot:
                    # Analyze the snapshot
                    analysis = await analyzer.analyze_image(snapshot)
                    results[camera_entity] = analysis
            
            return results
            
        except Exception as err:
            _LOGGER.error("Error updating cat activities data: %s", err)
            raise
    
    async def _get_camera_snapshot(self, camera_entity: str) -> bytes | None:
        """Get snapshot from camera entity."""
        try:
            camera_state = self.hass.states.get(camera_entity)
            if camera_state is None:
                _LOGGER.warning("Camera entity %s not found", camera_entity)
                return None
            
            # Use Home Assistant's camera service to get snapshot
            response = await self.hass.services.async_call(
                "camera",
                "snapshot",
                {"entity_id": camera_entity},
                blocking=True,
                return_response=True,
            )
            
            return response.get("content")
            
        except Exception as err:
            _LOGGER.error("Error getting snapshot from %s: %s", camera_entity, err)
            return None 