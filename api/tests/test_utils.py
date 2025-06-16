import datetime

import pytest

from api.utils import convert_datetime_fields_to_strings


def test_convert_datetime_fields_to_strings():
    """
    Tests the convert_datetime_fields_to_strings function.
    """
    now = datetime.datetime.now()
    data = {
        "name": "Test",
        "timestamp": now,
        "nested_data": {
            "event_time": now,
            "details": "Nested event"
        },
        "events": [
            {"id": 1, "time": now},
            {"id": 2, "time": now, "notes": "Second event"}
        ],
        "count": 5,
        "is_active": True
    }

    expected_iso_format = now.isoformat()
    expected_data = {
        "name": "Test",
        "timestamp": expected_iso_format,
        "nested_data": {
            "event_time": expected_iso_format,
            "details": "Nested event"
        },
        "events": [
            {"id": 1, "time": expected_iso_format},
            {"id": 2, "time": expected_iso_format, "notes": "Second event"}
        ],
        "count": 5,
        "is_active": True
    }

    result = convert_datetime_fields_to_strings(data)
    assert result == expected_data

    # Test with a list of dictionaries
    list_data = [
        {"id": 1, "time": now},
        {"id": 2, "time": now, "notes": "Second event"}
    ]
    expected_list_data = [
        {"id": 1, "time": expected_iso_format},
        {"id": 2, "time": expected_iso_format, "notes": "Second event"}
    ]
    list_result = convert_datetime_fields_to_strings(list_data)
    assert list_result == expected_list_data

    # Test with no datetime fields
    no_datetime_data = {"name": "Test", "count": 5}
    no_datetime_result = convert_datetime_fields_to_strings(no_datetime_data)
    assert no_datetime_result == no_datetime_data

    # Test with an empty dictionary
    empty_data = {}
    empty_result = convert_datetime_fields_to_strings(empty_data)
    assert empty_result == empty_data

    # Test with an empty list
    empty_list_data = []
    empty_list_result = convert_datetime_fields_to_strings(empty_list_data)
    assert empty_list_result == empty_list_data
