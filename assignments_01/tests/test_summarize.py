import pytest

from weatherkit import HourlyReading
from weatherkit.summarize import DailyAggregator

#fixture is a func that supplies shared test data. any test that lists
#readings as a parameter automatically recieves this lits
@pytest.fixture
def readings() -> list[HourlyReading]:
    """two dates: 2026-04-08 has 3 readings, 2026-04-09 has only 2 """
    return[
        HourlyReading("2026-04-08T00:00", 10.0, 0.1),
        HourlyReading("2026-04-08T01:00", 14.0, 0.2),
        HourlyReading("2026-04-08T02:00", 12.0, 0.3),
        HourlyReading("2026-04-09T00:00", 5.0, 1.0),
        HourlyReading("2026-04-09T01:00", 8.0, 2.0),    
    ]
   
#test1: grouping, min_hours=2 lets both days thru, so there will be two summaries
def test_grouping_produces_one_summary_per_date(readings):
    summaries = DailyAggregator(min_hours=2).summarize(readings)
    assert len(summaries) == 2
    assert [s.date for s in summaries] == ["2026-04-08", "2026-04-09"]
   
#test2: nax and min, summaries() finds the right high and lowbfor a day, high 14.0 and low 10.0
def test_temp_max_and_min_are_correct(readings):
    day = DailyAggregator(min_hours=2).summarize(readings)[0]
    assert day.temp_max == 14.0
    assert day.temp_min == 10.0

#test3: tests if the rain for the day added up correcly so output = 0.6, used pytest.approx for arounding diff
def test_precipitation_sum_adds_up(readings):
    day = DailyAggregator(min_hours=2).summarize(readings)[0]
    assert day.precipitation_sum == pytest.approx(0.6)

#after breaking one function in weatherkit/ the relevant test failed. i had gone to summarize.py (min_hours) and had flipped "<" to ">"
#test_short_days_is_dropped_and_reported failed also the ohter summarize tests. 

#test4: test short days and reports days with few readings left out 
def test_short_day_is_dropped_and_reported(readings):
    aggregator = DailyAggregator(min_hours=3)
    summaries = aggregator.summarize(readings)
    assert [s.date for s in summaries] == ["2026-04-08"]
    assert aggregator.incomplete_days(readings) == ["2026-04-09"]   
    
#test5: tests min hours =2, so it qualifies 
def test_lowering_min_hours_keeps_the_day(readings):
    aggregator = DailyAggregator(min_hours=2)
    summarize = aggregator.summarize(readings)
    assert[s.date for s in summarize] == ["2026-04-08", "2026-04-09"]
    assert aggregator.incomplete_days(readings) == []  