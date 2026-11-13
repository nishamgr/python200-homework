import json
import sys
import os
import sklearn
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import joblib


#   --- Task 1: Fetch the Data ---

#request to the OpenMeteo API
url = "https://archive-api.open-meteo.com/v1/archive"
#parameters
params = {
    "latitude": 37.77,
    "longitude": -122.42,
    "start_date": "2020-01-01",
    "end_date": "2020-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}
#sends requests and gets response back
response = requests.get(url, params=params)
response.raise_for_status()         #stops with an error message if something goes wrong

#tuns the data into table
df = pd.DataFrame(response.json()["daily"]) 
df["date"] = pd.to_datetime(df["time"]) 
df = df.drop("time", axis=1)            #drops the old time col since we now have date

#prints shape and first 5 rows
print(df.shape)
print(df.head())

# I chose San Francisco City

#   --- Task 2: Engineer the Label ---

#creates a new col: 1- good for running, 0- not good
df["good_for_running"] =(
    (df["temperature_2m_max"]  >= 7) &  #high atleast 7
    (df["temperature_2m_max"]  <= 26) & #most highest is 26
    (df["temperature_2m_min"] >= 0) &   #low is at least 0
    (df["precipitation_sum" ] < 3) &    #under 3mm of precipation
    (df["wind_speed_10m_max"] < 30) #wind under 30km/h
).astype(int)   #1(True)/0(False) 

#checks class balance. print no.days good(1). and no.days not good(0)
counts = df["good_for_running"].value_counts()
print("Good days (1):", counts[1])
print("Bad days (0):", counts[0])
print(df["good_for_running"].value_counts(normalize=True))

# About 58.7% of days (215/366) are good for running in SF, which seems reasonable for SF's weather.
#SF rarely gets very hot or freezing, it's pretty try  most of the year.
#If my data is very imbalanced, accuracy would be misleading and recall, precision and F1 would matter most tolook at metrics

#   --- Task 3: Build and Evaluate a KNN Classifier ---

#features: 4 weather vars the model will learn from
features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[features]
y = df["good_for_running"]

#train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y        #stratify=y keeps the same good/ skip ratio in both sets   
)

print("X_train/test shape: ", X_train.shape, X_test.shape)

#knn on raw data, built as a pipeline 
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])

#trains KNN on the scaled data
knn_pipeline.fit(X_train, y_train)
print("Scaled Accuracy: ", accuracy_score(y_test, knn_pipeline.predict(X_test)))

#choosing k: using 5-fold cross-val in the trraining data
k_values = [1, 6, 13, 17, 25]
mean_scores = {}
#cross-validating: loop over k_vales, bilds fresh pipeline each time & scoring it with 5-folds
for k in k_values:
    pipe = Pipeline([
        ("scaler",  StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)
    mean_scores[k] = cv_scores.mean()
    print(f"k={k:2d}: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")
#best cross-validation score is around k = 1
best_k = 1
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])
final_model.fit(X_train, y_train)       #training: fit the final model
y_pred = final_model.predict(X_test)        #evaluation: predicts on unseen test data
print("Final test accuracy:", accuracy_score(y_test, final_model.predict(X_test)))
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=3))

#   --- Task 4: Build and Evaluate a Logistic Regression Classifier ---

#logestic Regression model
logreg_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
logreg_model.fit(X_train, y_train)      #fit
#Evaluating the model
y_pred_lr = logreg_model.predict(X_test)
print("logreg test accuracy:", accuracy_score(y_test, logreg_model.predict(X_test)))
print("Logreg Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("logreg Classification Report:")
print(classification_report(y_test, y_pred_lr, target_names=["skip", "good"]))

#prints coefficients
logreg = logreg_model.named_steps["logreg"]
print("Reading the Coefficients:")
for name, coef in zip(features, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")       #1 = good or favors good, negative values push towards "skip"
    
#wind_speed_10m_max with(-3.42) most strongly push a day towards "skip" and on second negative coeff; precipitation_sum   : -2.53
#which means more rain and more wind both  push a day strongly toward "skip". 
#also, temperature_2m_max  : -0.87 pushes towards "skip" which fits hot days being bad for a run. 
#temperature_2m_min  : +0.27 is small and positive, so warmer lows slightly favor "good"


#   --- Task 5: Compare and Choose ---

#1 - Logistic Regression accuracy on my data. The acccuracy is 0.973(97.3%), compared to 0.932(93.2%) for KNN model.
#Logestic Regression model scored higher, but the difference is small.

#2: In the running app, Accuracy alone isn't enough because those two kinda of false positive and false negative have different cost.
#False Positive - the app will say it's a good day to run, but it's actually is a bad day for a run. The user would to sent out in a bad weather conditon.
#False Neegative - The app will say it's a badday to run, but it's actually a good day for a run. So, the user would miss a good day for a run.
#I would weigh precision more heavily, because telling user to go for a run in the bad weather condition is the worst mistake. 
#I would still check recall so the app doesn't say bad day too often.

#3: I would deply Logistic Regression because:
# - it can gives us why the model made a decision and
# - Logreg learns on weight per feature, where we can read those weights to see what the model learned
# - It is also small and makes predictions very fast
# - Logreg gives a smooth probabilty for every prediction which are useful confidence score. So, the app can show how confidence where good or bad day.

#   Task 6: Save the Model ---
#saves the fitted pipeline
os.makedirs("models", exist_ok=True) 
#saves the fitted pipeline
joblib.dump(logreg_model, "models/weather_classifier.pkl")
print("Saved.")

#computing test scores from the saved model
y_pred_jb = logreg_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_jb)
test_f1 = f1_score(y_test, y_pred_jb)

#Saving Metadata Alongside the Model
FEATURES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]

metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "features": FEATURES,
    "label": "good_for_running",
    "city": "San Francisco, CA",  
    "test_accuracy": float(test_accuracy),
    "test_f1": float(test_f1),
    "label_rule": {
        "temperature_2m_max": "7 to 26 C",
        "temperature_2m_min": ">= 0 C",
        "precipitation_sum": "< 3 mm",
        "wind_speed_10m_max": "< 30 km/h",
    },
}
#saving the metadata
with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
    
print("Model and Metadata Saved.")


