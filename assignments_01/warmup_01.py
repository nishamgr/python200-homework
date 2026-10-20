from dataclasses import dataclass, field, FrozenInstanceError
from pydantic import BaseModel, Field, model_validator, ValidationError 
import pytest

# --- Classes --- 

# Q1)

class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location
        self.readings = list(readings) if readings is not None else []
# Q2)
    def __repr__(self):
        #when there is no __repr__, python will diplay a generic label with a memory address.
        # with it, python calls this method to get a readable string for an object
        avg = self.average()
        avg_text = "None" if avg is None else f"{avg:.1f}"
        return(
            f"Thermometer(location={self.location!r},"
            f"n_readings={len(self.readings)}, average={avg_text})"
        )   
    def add(self, reading):
        self.readings.append(reading) 
    
    def average(self):
        # the average() needs to handle the empty case becuase the average is sum/ count and with no reaings the count
        ## is 0, so the code will get error without the check. with check, it returns None.
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)
    
    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)
          
california = Thermometer("California")
for temp in [19.0, 23.5, 26.1, 20.3]:
    california.add(temp)
    
charlotte = Thermometer("Charlotte")
for temp in [18.0, 19.5, 20.9, 21.0]:
    charlotte.add(temp)
    
print(california.average())
print(california.hottest())
print(charlotte)
print(Thermometer("Nowhere"))

# Q3)
class TemperatureAlert:  
    #Threshold is stored in TemperatureAlert rather than of 
    # breaches because we want a fixed threshold so, when we call the breaches() it knows the rules.
    #the advantages that we get with twenty thermometers is that i create one alert and reuse it for every thermometer, so the threshold is written in one place,
    #if the rule changes, I only update the one line. 
    def __init__(self, threshold= 30.0):
        self.threshold = threshold
        
    def breaches(self, thermometer):
        result = []
        for reading in thermometer.readings:
            if reading > self.threshold:
                result.append(reading)
        return result
class Thermometer:
    def __init__(self, readings):
        self.readings = readings
    
thermo = Thermometer([23.5, 33.0, 35.2, 29.9, 40.1])
alert_low = TemperatureAlert(30.0)
alert_high = TemperatureAlert(35.0)
print(alert_low.breaches(thermo))
print(alert_high.breaches(thermo))

# --- Dataclasses, Type Hints, and Docstrings ---
# Q1)
@dataclass(frozen=True)
class Station:
    """ A weather station with an ID, a name, and location
    Attributes:
        station_id: station id number,
        name: station name,
        latitude: latitude of a station,
        longitude: longitude of a station,
        elevation: height of the station 
    """
    station_id: int
    name: str
    latitude: float
    longitude: float
    elevation: int

station_a = Station(12, "Jake", 12.05, 67.45, 33)
station_b = Station(12, "Jake", 12.05, 67.45, 33)
# Q2)
try: 
    station_a.name = "Sam"
except FrozenInstanceError as e:
    print("Can't Modify:", e)

print(station_a == station_b)

# Q3)
@dataclass 
class StationBatch:
    """ group of weather station in one region"""
    region: str
    Stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """add a station to the batch"""
        self.station.append(station) 
    def highest(self) -> Station | None:
        """ Returns the station with the highest elevation, or None if empty"""

#ErrorMessage with station: list[Station] = []
#ValueError: mutable default <class 'list'> for field Stations is not allowed: use default_factory
#when the class is defined, default value is created once, so every stationbatch would share the same list
#adding one batch to a station would change for all the others, which python refuses to allow that
#to fix it created a new fresh empty list for each new stationbatch by replaccing with (field(default_factory=list))

# --- Pydantic ---
# Q1)
class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=90)
    humidity: float = Field(ge=0, le=100)
    
# Q4) rejects any readigs where humidity = 0.0 & temp_c < -40
    @model_validator(mode="after")
    def reject_failed_sensor(self):
        """rejects readings with humidity  0.0 & temp below -40"""
        if self.humidity == 0 and self.temperature_c < -40:
            raise ValueError(
                f"humidity {self.humidity} while temperature_c is "
                f"{self.temperature_c} (below -40)"
            )
        return self
#valid reading still construct
valid_reading = Reading(station_id= "B12", timestamp="2026-10-15T22:24:00", temperature_c=-10.1, humidity=0.0)
print("Valid reading accepted:", valid_reading)

## the reason why the Field constraints can't express this rule:
# Field check one value at the time. Here humidity=0.0 and temp_c =-50.0
#which they are valid on their own. The error shows up when two are combined. Each field is valid on it's own.
#and a Field can't see the other Field, so a mdoel_val is needed.
#model_validator runs after all Fields are set, so it can compare
try:
    Reading(station_id= "B12", timestamp="2026-10-15T22:24:00", temperature_c=-50.0, humidity=0.0)
except ValidationError as e:
    print("Bad combiantion Rejected")
    print(e)    
    
#good reading
reading = Reading(
    station_id= "STN-12", 
    timestamp="2026-10-15T22:24:00",
    temperature_c=24.1,
    humidity=56.0
)
print(reading)

# Q2) Failure 1 missing require field
try: 
    Reading(timestamp="2026-10-15T22:24:00", temperature_c=25.5, humidity=56.0)
except ValidationError as e:
    print("Failure 1: missing field")
    print(e)

# Failure 2 a temp_c of 150.0
try:
    Reading(station_id="B12", timestamp="2026-10-15T22:24:00", temperature_c=150.0, humidity=56.0)
except ValidationError as e:
    print("Failure 2: Temp is too high")
    print(e)
    
#Failure 3 humidity is text that cant be int
try:
    Reading(station_id="B12", timestamp="2026-10-15T22:24:00", temperature_c=25.0, humidity="very humid")
except ValidationError as e:
    print("Failure 3: can't be a number")
    print(e)

# Q3) several errors at once
try: 
    Reading(station_id="NP", temperature_c="Twenty-five", humidity=56.0)
except ValidationError as e:
    print(f"{e.error_count()} problems found")
    for err in e.errors():
        print(f"field={err['loc']} type={err['type']} msg={err['msg']}")

# --- pytest ---
# Q3)

def celsius_to_fahrenheit(celsius: float) -> float:
    """convert a temperature from celsius to frahrenhiet"""
    return celsius * 9/5 + 32  
    
def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    
    #floats are stored imprefictly in binary, so when i do == the value like 98.6 may be off a tiny amount. approx
    #compares within small tolerance instead of requiring a excat match.
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)
    
# Q2)
def mean(values: list[float]) -> float:
    """ calculate the average of a list if numbers.
    Args:
        values: the nunmber to average
    Returns:
        the arithemtic mean of the vlues
    Raises:
        ValueError: If 'values' is empty
    """
    if not values:
        raise ValueError("cannot calculate the mean of an empty list")
    return sum(values)/ len(values)

def test_mean_of_empty_raises():
    #pytest alone would fail without match = because without it the test passes on any ValueError, each one caused by a typo in the test
    #match checks that it's the error i meant.
    with pytest.raises(ValueError, match="empty"):
        mean([])

# Q3)
@pytest.mark.parametrize(
    "values, expected, reason", 
    [
        ([1, 2, 3], 2, "whole nums"),
        ([5], 5, "single element"),
        ([-1, -2, -3], -2, "all negative nums"),
        ([-5, 5], 0, "negative and positive cancles out"),
        ([1.5, 3.5], 2.5, "decimals"),
    ],
)
def test_mean_values(values, expected, reason):
    #one parametrized test with four cases better than four nearly identical test func because
    #the logic is written once; less repetition, fewer places for typos
    #each casse still runs and reports pass and fail seperatly, and adding new case is just one new line in the table.
    assert mean(values) == pytest.approx(expected), reason

# Q4)
#def test_celsius_to_fahrenheit():
#        assert celsius_to_fahrenheit(0) == 32
#>       assert celsius_to_fahrenheit(100) == 212
#E       assert 257.0 == 212
#E        +  where 257.0 = celsius_to_fahrenheit(100)

#warmup_01.py:203: AssertionError
#FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212

#the expected value is (212), when i changed the math pytest showed me the actual value the func returned(257),
#the exact unput that caused it(100), and which assert line failed. That is more useful than 'a bare assertion failed'
#because i can see what went wrong without adding the print() statements or using the a debugger.

