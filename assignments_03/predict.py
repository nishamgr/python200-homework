#   --- Task 8: Reflect ---

#1: My boaderline day is Day 2, max temp is 22.0 C, min temp is 12 C, no rain and wind is 28km/h
#the model gives it P(good) = 0.50, which shows uncertenty: Which is basically 50/50 chance
#This aligns with the day being positioned at the limit of my classification rule. The model 
#is very sure about the clear days (scoring 1.00 for the good day and just 0.06 for the bad windy day). 
#If the model had predicted 0.52, I wouldn't place much trust in the "good" label. Instead, 
#I would present the user with the probability directly or introduce a "maybe" category 
# (for instance, 0.4 to 0.6), that tells the user, “borderline day,
# check the forecast yourself.” Because a false positive (sending someone out on a bad day) is the costlier mistake, 
# I could also raise the threshold. so the app only says “good” when P(good) is above about 0.7.

#2: Running predicted.py file before running a train_classifier.py
#It should crash with a FileNotFoundError at joblib.load, because the files "models/weather_classifier.pkl" and metadata json
#wouldn't exist yet for it to run.
#To make the error message  more helpful, I would include the try:, Except where the load file is.
#and most important would be to run train_classifier.py first to create models/weather_classifier.pk1 

#3: To suppport this mdoel so that the model runs inside a cloud pipline that classifies each day's
#weather automatically. One thing about the predict.py that would need to change to support that is
#the four hard-coded hypothetical days. In pipeline it would need to take in real weather data automatically(from like API, satabase each day)
#instead of hand-coded values.

import joblib
import json
import pandas as pd
#try:
clf = joblib.load("models/weather_classifier.pkl")
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
    
#except FileNotFoundError:
#raise SystemExit("Model files not found. Run train_classifier.py first to create them in models/.")

#prints the key metadata
print("City:", metadata["city"])
print("Features:", metadata["features"])
print(f"Test accuracy: {metadata['test_accuracy']:.3f}")
print()

    
#building new days using the exact features name and order from the metadata
new_days = pd.DataFrame(
    [
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12},   #GOOD(ABOUT1.00)
        {"temperature_2m_max": 22, "temperature_2m_min": 12, "precipitation_sum": 0.0, "wind_speed_10m_max": 28},   #in the borderline about (.50)
        {"temperature_2m_max": 25, "temperature_2m_min": 15, "precipitation_sum": 0.0, "wind_speed_10m_max": 33},   #skip, about(0.06)
        {"temperature_2m_max": 5,  "temperature_2m_min": -2, "precipitation_sum": 15.0, "wind_speed_10m_max": 45},  #skip with a low P(good)
    ]
)[metadata["features"]]

#predicts class (1=good, 0=skip) and p(good) for each new day
preds = clf.predict(new_days)
probs = clf.predict_proba(new_days)[:, 1] 

#loops over each new day
for i, (pred, prob) in enumerate(zip(preds, probs)):            
    label = "good for running" if pred == 1 else "skip"         #turn the 0/1 prediction into a readable label
    print(f"Day {i + 1}: {new_days.iloc[i].to_dict()}")     #prints the day's 4 inputs
    print(f"    Prediction: {label} (Confidence, P(good): {prob:.2f})")
    print()

