import json
from weatherkit import WeatherResponse
from weatherkit.summarize import DailyAggregator
from weatherkit.records import to_readings

def main() -> None:
    #load and validate
    with open("weather_raw.json") as f:
        raw = json.load(f)
    #convert and aggregate   
    response = WeatherResponse.model_validate(raw)
    readings = to_readings(response)
    aggregator = DailyAggregator()
    summaries =  aggregator.summarize(readings)
    dropped = aggregator.incomplete_days(readings)

    #print the table
    print(f"{'Date':<12}{'High':>8}{'Low':>8}{'Precip':>8}{'Range':>8}")
    for s in summaries:
        print(f"{s.date:<12}{s.temp_max:>8.1f}{s.temp_min:>8.1f}{s.precipitation_sum:>8.1f}{s.temp_range():>8.1f}")
    if dropped:
        #warning about dropped days
        print(f"WARNING: dropped incomplete days: {','.join(dropped)}") 
#without guard, importing report.py would run main() right away, loading the file and printing the table, even if the  importer
#only wanted a helper func.           
if __name__ == "__main__":
    main()
    
#1 -> Rejecting the whole file is the not a right behavior, skippping is better because one broken hour doesn't throw away a whole week.
#to allow blanks, would tell the schema that a temperature can be a number or None, and the code after it has to ignore the None value.

#2 -> min_hours24 prevents that by leaving the date out.Then incomplete_days() tewlls which day was left out, so i know its pending not lost.

#3 ->  because the code is a package, someone esle's script, like a week 10 pipeline, can write from weatherkit import DailyAggregator and use it
#without copying my files or knowing which file each class is in.

