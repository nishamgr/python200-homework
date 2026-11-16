import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- Preprocessing --- 


# Q1)
#fit, transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#prints the mean of each column of the scaled training data
print("Train mean:", X_train_scaled.mean(axis=0).round(2)) 

#Fit the scaler on X_train only so no data from the test set leaks into the preprocessing.
#The scaler must not learn anything from the test set.

# Q2)
#OneHotEncoder 
encoder = OneHotEncoder(sparse_output=False)
encoded_season = encoder.fit_transform(df[["season"]])
#prints resulting shapes and category anmes
print("Season shape : ", encoded_season.shape)
print("Category name: ", encoder.get_feature_names_out())

#The single season became 4 colums, because there are 4 unique seasons: Fall, Spring, Summer, Winter
#OneHotEncoding makes one new colums per cateogry, so a nfeature with N categories becomes N columns

# Q3)

#new split that includes "season"
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]

features = numeric + ["season"] 
X = df[features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#prepare the data: numeric cols getrs scaled, and season get's oneHot encoded
preprocess = ColumnTransformer(
    transformers = [
        ("numeric", StandardScaler(), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore"), ["season"])
    ] 
)

#chains the step together: prepares the dta first, then predicts
model = Pipeline(
    steps = [
        ("preprocessing", preprocess),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

#train the model using train data only
model.fit(X_train, y_train)
#predict the answer for the test data using the trained model
prediction = model.predict(X_test)
#checks how many prediction are right; 1.0=allcorrect
print("Test Accuracy: ", accuracy_score(y_test, prediction))

#pipeline: By hand I will have to scale, oneHotEncode, join the cols and repeat the same, steps on the test set in the right order before predicting
#The pipeline does all fo that automatically, I only have to deine the steps once, and fit and predict runs all of them automatically.
#And Only learns from the training data with no leakeage.


# --- KNN --- 

# Q1) 
#build pipeline of SS and KHC
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])

#fit on training data
knn_scaled.fit(X_train[numeric], y_train)
y_pred = knn_scaled.predict(X_test[numeric])

#print the test accuracy & full classification report
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification report: ", classification_report(y_test, y_pred, target_names=["skip", "good"]))

# Q2)
#looped over each k list, build a scaled pipeline for that k, 
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),        
    ])
    #ran 5 fold cv and prtints mean cv accuracy for each k
    scores = cross_val_score(knn, X_train[numeric], y_train, cv=5)
    print(f"k={k:2d}: mean CV accuracy = {scores.mean():.3f}")
    
    #I would chose, k=11: mean CV accuracy = 0.942 because it is the highest mean cv accuracy among all the list
    #very small k overfits to noise, while very large k underfits, so a middle value worked best
    
# Q3)
#fit knn on unscaled, raw features
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train[numeric], y_train)

#prints unscaled and scaled accuracy
print("Unscaled accuracy: ", accuracy_score(y_test, knn_unscaled.predict(X_test[numeric])))
print("Scaled accuracy: ", accuracy_score(y_test, knn_scaled.predict(X_test[numeric])))

#Yes, scaling helped (about 0.925 unscaled vs. about 0.958 scaled).
#scaling matters for distance based models like KNN because distance is calculated across all features at once. 
#without scaling, large range features (temperature, wind) dominate the distance, while small range features like precipitation barely gets count.
#scaling puts all features on a comparable scale so each one contributes fairly.


# --- Logistic Regression --- 

# Q1)
#build a pipeline
lg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logReg", LogisticRegression(max_iter=1000)),
])
#fit
lg_pipeline.fit(X_train[numeric], y_train)
#predict on the test set
y_pred = lg_pipeline.predict(X_test[numeric])

# report
print(classification_report(y_test, y_pred, target_names=["skip", "good"]))

# Q2)
#interpreting the coefficients: prints name alongside its coefficient
coefs = lg_pipeline.named_steps["logReg"].coef_[0]

for name, coef in zip(numeric, coefs):
    print(f"{name}: {coef:.3f}")
    
    #wind_speed_10m_max pushes strongly towards "skip" because it has the large negative voefficient.
    #yes it does match my intution about running weather because stong wind makes running unpleasant

# Q3)
# prints first 5 test days for the probability of "good"
probabilities = lg_pipeline.predict_proba(X_test[numeric])[:, 1]
print("Probabilities", probabilities[:5].round(2))

#probability near 0.5 tells me that the model is not confident about that day because it sits right between 0 and 1.
#THe closer a probability is to 0 and 1, the more confident the mdoel is.

# ---  Evaluation --- 

#Q 1)
#logesticRegression Pipeline
labels = ["skip", "good"]
#confusion matrix on the test set
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(colorbar=False)

plt.title("Logestic Regression Confusion Matrix")
plt.savefig("outputs/logreg_confusion_matrix.png")
plt.show()

# The model made: False Positive: 9 (bad days predicted "good"), and False Negative: 7 ("good days predict "skip")
# For the running app, the False Positive matters the most which predicts bad days as good
#This can predict a rainy or windy is a good day to go  for run on such a bad condition,
#whereas a Flase Negative would only cost them to miss the good days 


# ---  joblib --- 
# Q 1)
#saves the fitted pipeline
joblib.dump(lg_pipeline, "models/warmup_model.pkl")
print("Saved.")

#loads it back and confirms identical predictions
loaded = joblib.load("models/warmup_model.pkl")
print("Loaded.")
assert (loaded.predict(X_test[numeric]) == lg_pipeline.predict(X_test[numeric])).all()
print("Predictions match.")

#If i had saved the the LogisticRegression step without the scaler, then called .predict() on raw data
#the mdoel would get unscaled features.
#it learned its weights on standardized values, so the predictions would be wrong or unreliable, with no error to warn me.
#saving the whole pipeline keeps the scaler and the model together.