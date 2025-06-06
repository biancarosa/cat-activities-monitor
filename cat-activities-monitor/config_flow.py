"""Config flow for Cat Activities Monitor integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    CONF_CAMERA_ENTITIES,
    CONF_ANALYSIS_SERVICE_URL,
    CONF_UPDATE_INTERVAL,
    DEFAULT_UPDATE_INTERVAL,
    DEFAULT_ANALYSIS_SERVICE_URL,
)

_LOGGER = logging.getLogger(__name__)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the user input allows us to connect."""
    # Validate that selected cameras exist
    camera_entities = data[CONF_CAMERA_ENTITIES]
    for camera_entity in camera_entities:
        if hass.states.get(camera_entity) is None:
            raise InvalidCamera(f"Camera {camera_entity} not found")
    
    # Validate analysis service URL if provided
    analysis_url = data.get(CONF_ANALYSIS_SERVICE_URL)
    if analysis_url:
        # TODO: Add validation for the analysis service URL
        pass
    
    return {"title": f"Cat Monitor ({len(camera_entities)} cameras)"}


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Cat Activities Monitor."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                info = await validate_input(self.hass, user_input)
            except InvalidCamera:
                errors["base"] = "invalid_camera"
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return self.async_create_entry(title=info["title"], data=user_input)

        # Get all camera entities
        camera_entities = []
        for state in self.hass.states.async_all():
            if state.domain == "camera":
                camera_entities.append({
                    "value": state.entity_id,
                    "label": f"{state.attributes.get('friendly_name', state.entity_id)}"
                })

        data_schema = vol.Schema({
            vol.Required(CONF_CAMERA_ENTITIES): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=camera_entities,
                    multiple=True,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_ANALYSIS_SERVICE_URL, 
                default=DEFAULT_ANALYSIS_SERVICE_URL
            ): str,
            vol.Optional(
                CONF_UPDATE_INTERVAL, 
                default=DEFAULT_UPDATE_INTERVAL
            ): vol.All(int, vol.Range(min=10, max=300)),
        })

        return self.async_show_form(
            step_id="user", 
            data_schema=data_schema, 
            errors=errors
        )


class InvalidCamera(HomeAssistantError):
    """Error to indicate an invalid camera entity.""" 