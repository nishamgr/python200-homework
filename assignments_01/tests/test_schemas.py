import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse

#a relative path like  "weather_raw.json" depends on directory pytest is run from,
#so it breaks elsewhere. __file__ is the location of this test file, so build the path from it works no matter where pytest is run from
data_file = Path(__file__).parent.parent /"weather_raw.json"

def make_raw() -> dict:
    """smallvalid response (3hours) that tests can copy and break"""
    return {
        "latitude": 35.2,
        "longitude": -80.0,
        "timezone": "GMT",
        "elevation": 200.0,
        "hourly": {
            "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
            "temperature_2m": [10.0, 11.0, 12.0],
            "precipitation": [0.0, 0.0, 0.1],
        },
    }
#test 1 
def test_valid_response_has_168_hours():
    #opend the real data file
    with open(data_file) as f:
        raw = json.load(f)  #reads the json 
    response =  WeatherResponse.model_validate(raw)  #validates and raises ValidationError
    assert len(response.hourly.time) == 168   #assert fails and the test if the num pf hourly isn't = 168

#test2
@pytest.mark.parametrize("bad_latitude", [200.0, -90.1, 90.1, -200.0])
def test_latitude_out_of_range_raises(bad_latitude):
    raw = make_raw() #starts with valid data, new dict
    raw["latitude"] = bad_latitude  #outside the allowed -90 to 90 range
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw) 
        
#test3
def test_mismatched_length_raises():
    raw = make_raw()
    raw["hourly"]["temperature_2m"] =  [10.0, 11.0] #2 entries, but has 3
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)    

#test4
def test_null_temperature_raises():
    raw = make_raw()
    raw["hourly"]["temperature_2m"][1]=  None #a null 
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)
