"""
Utility functions for the Cat Activities Monitor API.
"""

from datetime import datetime
from typing import Any, Dict

BOUNDING_BOX_COLORS = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
    "#BB8FCE",  # Light Purple
    "#85C1E9",  # Light Blue
    "#F8C471",  # Peach
    "#82E0AA",  # Light Green
]

def convert_datetime_fields_to_strings(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert datetime objects in a dictionary to ISO format strings for PostgreSQL compatibility.
    
    Args:
        data: Dictionary that may contain datetime objects
        
    Returns:
        Dictionary with datetime objects converted to strings
    """
    converted_data = data.copy()
    
    for key, value in converted_data.items():
        if isinstance(value, datetime):
            converted_data[key] = value.isoformat()
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries
            converted_data[key] = convert_datetime_fields_to_strings(value)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # Handle lists of dictionaries
            converted_data[key] = [
                convert_datetime_fields_to_strings(item) if isinstance(item, dict) else item
                for item in value
            ]
    
    return converted_data