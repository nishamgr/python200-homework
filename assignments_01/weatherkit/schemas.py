from pydantic import BaseModel, Field, model_validator
from typing import Self


class HourlyBlock(BaseModel):
    """Hourly Weather Data as a lists, with one entry per hour"""
    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]
    
    @model_validator(mode="after") 
    def check_same_length(self) -> Self:
        if not len(self.time) == len(self.temperature_2m) == len(self.precipitation):
            raise ValueError(f"Three lists must be the same length; time={len(self.time)}, temperature_2m={len(self.temperature_2m)}, precipitation={len(self.precipitation)}")
        return self
  
class WeatherResponse(BaseModel):
    """Top-Level weather API response; location details and the hourly data"""
    latitude: float = Field(ge=-90, le=90, description="latitude in degrees")
    longitude: float = Field(ge=-180, le=180, description="longitude in degrees")
    timezone: str
    elevation: float
    hourly: HourlyBlock
    

    