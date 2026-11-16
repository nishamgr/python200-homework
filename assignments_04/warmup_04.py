from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pytest
import joblib
import json 
import pandas as pd

#   --- Warmup Question 1 ---
#dict : created a json file to work with; features; lits of the input col names
json_file = {
    "model": "weather_classifier",
    "features" : [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max"
    ],
}
#creates  a new file and saves the dict into a json 
with open("model_info.json", "w") as f:
    json.dump(json_file, f, indent=2)

#ModelInfro clas: opens the file at path, loads it with json.load and stores the result in self.data
class ModelInfo:
    """ Open the Json file at path and stoe its contents in self.data    
    Args: 
        path: the location of the json file to read  
    """
    def __init__(self, path):
        """ open the json file oncec and keeps it contents in memoery
        arrti : 
            data: the dict loaded from the json file      
        """
        with open(path) as f:
            self.data = json.load(f)            #runs once, stores the contents
        
    def feature_count(self) -> int:
        """ Return the number of items inb the "feature" list """
        return len(self.data["features"])       #returns the number of items in features list
    

#created an object: info from the class and prints the feature count 
info = ModelInfo("model_info.json") 
print(info.feature_count())   

# Loading the file in __init__ is better than the laoding it inside feture_count()
#because loading in __init__ happens once, and every call to feature_count() reuses self.data in memory.

#   --- Warmup Question 2 ---
#dataclass with three type hinted filds
@dataclass
class DayForecast:
    """The weather forecst for one day.

    Attributes:
        date: forecast date as text, "2023-06-15
        high_c: the expected high temperature in degree celcius
        will_rain: True, if rain expected, False, if not
    """
    date: str
    high_c: float 
    will_rain: bool
    
    def summary(self) -> str:
        """Returns a reasable one sentence summary of this forecast"""
        rain_text =  "rain expected " if self.will_rain else "no rain"
        return f"{self.date}: high: {self.high_c:.1f}C, {rain_text}"
    
#created 2 DayForecast objects and prints each summary
day1 = DayForecast(date="2023-06-15", high_c=24.1, will_rain=True)
day2 = DayForecast(date="2023-11-25", high_c=22.1, will_rain=False)

print(day1.summary())
print(day2.summary())
    
#@dataclass decorator wrote the __init__ method. I didn't have to write it by hand to take date, high_c,and will rain and 
#store them on the object, that's why the two DayForecast objects works.

# --- Warmup Question 3 ---
def parse_temperature(text: str) -> float:
    """ Converts string like "24.1" to a float
    Args: 
        text: The temperature as a string
    Returns: 
        The temperature as a float
    Raises: 
        if any equired feature contain "temperature whenthe text is not a number raises ValueError"
    """
    try: 
        return float(text)
    except ValueError:
        raise ValueError(f"invalid temperature: {text!r} is not a number")
        
def test_parse_temperature_valid():
    result = parse_temperature(24.1)
    assert result == pytest.approx(24.1)
    
def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")
        
# without match, pytest.raises(ValueError) passes for any valuerror, even one whose message don't mention 
# "temperature". match="temperature" makes the test pass only when the error message contains that works "temperature"