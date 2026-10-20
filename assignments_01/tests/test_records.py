from weatherkit import HourlyReading, WeatherResponse
from weatherkit.records import to_readings

def make_response() -> WeatherResponse:
    """ small valid response 3 hours with distinct values at each index"""
    return WeatherResponse.model_validate({
        "latitude": 35.2,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 200.0,
        "hourly": {
            "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
            "temperature_2m": [10.0, 11.0, 12.0],
            "precipitation": [0.0, 0.0, 0.1],
        }
    })
#test1 checks the count and that the order is preserved using the first and last timestamp
def test_one_reading_per_hour_in_order():
    readings = to_readings(make_response())
    assert len(readings) == 3 
    assert readings[0].timestamp == "2026-04-08T00:00" #first hour
    assert readings[-1].timestamp == "2026-04-08T02:00" #last hour

#test2 compares each reading to the same position in the three input lists
def test_values_match_input_at_each_index():
    response = make_response()
    readings = to_readings(response)
    
    for i, reading in enumerate(readings):
        assert reading.timestamp == response.hourly.time[i]
        assert reading.temperature_c == response.hourly.temperature_2m[i]
        assert reading.precipitation_mm == response.hourly.precipitation[i]

#test3 dataclasses compare their field automatically, so that the two identical fiels are ==
def test_identical_readings_are_equal(): 
    a = HourlyReading("2026-04-08T00:00", 10.0, 0.0)
    b = HourlyReading("2026-04-08T00:00", 10.0, 0.0)
    assert a == b