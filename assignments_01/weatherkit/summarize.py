from dataclasses import dataclass
from .records import HourlyReading
 
@dataclass
class DailySummary: 
    """ summary of one calendar day of weather observations
    units:
        date: calendar date: "2026-04-09"
        temp_max: highest temp of the day, in celsius
        temp_min: lowest temp of the day, in celsius
        precipitation_sum: total precipitation for the day in millimeter
        hours_observed: no. of hourly readings that contributed
        """
    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int
    
    def temp_range(self) -> float:
        """ returns the difference between the temp_max - temp_min in degree celsius"""
        return self.temp_max - self.temp_min



class DailyAggregator:
    """ groups hourly readings into daily summaries"""
    def __init__ (self, min_hours: int = 24) -> None:
        """ create an aggregator with a minimum observation rule.
        Args: 
            min_hours: the minimum numbers of hourly readings a day must before it is reported
            days with fewer readings are dropped because their  max and min can't be trusted. 
            defaults to a full day
        """
        self.min_hours = min_hours
    
    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """group hourly readings by date and compute summary for each day
        Days with fewer than min_hours observations are dropped, because a daily max or min 
        from a partial day can't be trusted.
        Args: 
            readings: hourly readings, in any order. Each date is taken from the first 10 characters of the timestamp.
        Returns:
            A list of DailySummary objects, one per complete day, sorted by date. 
        """
        groups = self._group_by_date(readings)
            
        summaries = []
        for date in sorted(groups):   #goes thru the days in date order
            items = groups[date]  #this day's pile
            if len(items) < self.min_hours:     #if few cards
                continue                        #skips this day
            temps = [r.temperature_c for r in items]  
            summaries.append(DailySummary(
                date=date,
                temp_max=max(temps),
                temp_min=min(temps),
                precipitation_sum=sum(r.precipitation_mm for r in items),
                hours_observed=len(items),
            ))
        return summaries
    
    def _group_by_date(self, readings: list[HourlyReading]) -> dict[str, list[HourlyReading]]:
        """Group readings by calendar date, first 10 characters of the timestamp
        Args:
            readings: hourly readings, in any order
        Returns: 
            a dict mapping each date string to the list its readings
        """
        groups: dict[str, list[HourlyReading]] = {} #empty dict
        for reading in readings: #one hourly reading at a time
            date = reading.timestamp[:10] #keeps just "YYY-MM-DD"
            groups.setdefault(date, []).append(reading) #gets this date list or starts new one, then adds the reading
        return groups
    
    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """ returns the dates dropped for having too few observations
        a date is dropped when it has fewer than min_hours readings. 
        this lets callers see what summarize() dropped instead of days disapearing silently.
        Args:
            readings: hourly readings, in any order
        Returns:
            the dropped dates as string, sorted
        """
        groups = self._group_by_date(readings)   #makes the piles
        dropped = []
        
        for date in sorted(groups):
            # same test as summarize(), except that the short days are the ones kept
            if len(groups[date]) < self.min_hours:  # few readings
                dropped.append(date)  #records a date that's too short
        return dropped
    
