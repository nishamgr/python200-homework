"""Classify a few sample days using the saved weather model."""

# --- Task 5: The Script --- 
from weather_model import WeatherClassifier

hypothetical_days = [
    {"temperature_2m_max": 18, "temperature_2m_min": 10,
     "precipitation_sum": 0.0, "wind_speed_10m_max": 12},       #clearly good: mild, dry, calm
    {"temperature_2m_max": 3, "temperature_2m_min": -4,
     "precipitation_sum": 12.0, "wind_speed_10m_max": 40},      #clearly bad: cold, rainy, windy
    {"temperature_2m_max": 22, "temperature_2m_min": 13,
     "precipitation_sum": 9.0, "wind_speed_10m_max": 14},       #borderline: warm, but rainy
    {"temperature_2m_max": 25, "temperature_2m_min": 15,
     "precipitation_sum": 0.0, "wind_speed_10m_max": 33},       #borderline: mild and dry but windy
]

def main() -> None:
    """Load the compenent and print a prediction for each hypothetical days"""
    classifier = WeatherClassifier("models/weather_classifier.pkl")
    for day, result in zip(hypothetical_days, classifier.predict(hypothetical_days)):
        #Prints a readable line per day with the inputs, the label, and the probability
        print(f"{day['temperature_2m_max']:.0f}C, {day['precipitation_sum']:.0f}mm rain, "
              f"{day['wind_speed_10m_max']:.0f} km/h wind -> {result.label} ({result.probability:.2f})")
#This guard runs main() only when the file is executed directly    
if __name__ == "__main__":
    main()
    
#If I omitted the guard and another script imported predict_weather.py to reuse a helper, main() would
#run as soon as another script imported predict_weather.py just to reuse a helper.

# ---Task 6: Reflection --
#1. Without a fixed order, a caller who passes the dict key in different order would feed themodel values in 
#the rong columns, and it can predict wrong output. For example, predict "good" for a cold, windy and rainy day.
#Selecting self.features in a fixed order prevents this.
#2. Building a tiny model in a fixture makes the test self_control and stabale.
#they test the compenent's behavior rather than the model's accuracy.
#3. Before production, I would validation for bad values like missing or non-numeric data, since real
#database records are messier than my test data. 