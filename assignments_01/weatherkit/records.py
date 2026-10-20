from dataclasses import dataclass
from .schemas import WeatherResponse

#WeatherResponse is a Pydantic model because it sits ON the boundry: it is build from a raw JSON that comes
#from outside the API. which can't trust. It needs runtime validation, types, the longtitude/latitude, range and matching list lengths.

#HourlyReading is a dataclass because it lives INSIDE the boundry. it is every created by to_readings()
#from a WeatherResponse that had already validated, so rechecking every work would be redundant. 
#dataclass holds typed fields and gives us __init__, __repr__, and == for free without validating anything.

@dataclass
class HourlyReading:
    """one hour of weather observations
    units:
    timestamps: time string: eg. "2026-04-08T00:00
    temperature_c: air temp at 2m, in degree celsius
    precipitation_mm: during the hour in millimeters
    """
    timestamp: str
    temperature_c: float
    precipitation_mm: float
    
def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """converts the columnar hourly block into one Reading per hour.
    Args:
        response: A validated response whose hourly lists all have the same lengths
    Returns:
        a list of HourlyReading objects, once per hour, in the same order as input lists
    
    """
    hourly = response.hourly
    readings = []
    for t, temp, rain in zip(hourly.time, hourly.temperature_2m, hourly.precipitation):
        readings.append(
            HourlyReading(timestamp=t, temperature_c=temp, precipitation_mm=rain)
            )
    return readings 