import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#--- Task 1: Fetch the Data ---

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 37.78,
    "longitude": -122.19,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

#printsthe shape of dataset (rows, col)
print(df.shape)
#prints the first 5 rows
print(df.head())

#I chose Oakland, CA

#--- Task 2: Look at the Features ---

# checked for missing value and non found
print(df.describe())

#checked for missing values: non found, no cleaning needed
print(df.isnull().sum())

#correlation between each numeric col with temperature_2m_max
print(df.corr(numeric_only=True)["temperature_2m_max"].sort_values())

#temperature_2m_min is strongly related to the temperature_2m_max
#warmer days tends to have warmer nights

#scatter plot of (temperature_2m_min vs temperature_2m_max)
plt.figure(figsize=(7,5))
plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"], alpha=0.6)
plt.xlabel("Daily Low Temp")
plt.ylabel("Daily High Temp")
plt.title("Daily Low vs. Daily High")


plt.savefig("outputs/project_02_low_vs_high.png")
plt.show()

#I see that the scatter plot shows an upward trend, 
#the days with warmer lows also have warmer highs, which mactches the corr of 0.77
#The dots are somewhat spread out, the realtionship is strong but not perfect.

# --- Task 3: Engineer a Feature ---

#adding a binary feature is_summer col: 1 if the date is summer(june,july,august), 0 otherwise
df["is_summer"] = df["date"].dt.month.isin([6,7,8]).astype(int)
#corr for is_summer
print("Correlation of is_summer with temperature_2m_max:", df["is_summer"].corr(df["temperature_2m_max"]))

print("Summer days:", (df["is_summer"] == 1).sum())
print("Non-Summer days:", (df["is_summer"] == 0).sum())

#Yes, I expect "is_summer" to help predict the daily high because it will help us separates summer days vs. 
#non_summer days. Summer days tend to be warmer, and non of the original cols capture the season directly.

#  --- Task 4: Baseline Model --  
#feature and target
features = ["temperature_2m_min", "temperature_2m_max", "wind_speed_10m_max", "is_summer" ]
X = df[["temperature_2m_min"]]
y = df["temperature_2m_max"]

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=42
)
#create model, fit on training data, predict on the test data
model = LinearRegression() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test)

#evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Slope (temperature_2m_min):", model.coef_[0])
print("RMSE:", rmse)
print("R2:", r2)

#The RMSE is 3.3 degree celsius. In plain English,
# prediciton is off by about 3.3 degree celsius from the actual daily high

# --- Task 5: Full Model ---
#feature and target
feature_cols = ["temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "is_summer" ]
X = df[feature_cols].values
y = df["temperature_2m_max"].values

#Train/test split
X_train_m2, X_test_m2, y_train_m2, y_test_m2 = train_test_split(
    X, y, test_size = 0.2, random_state=42
)

#create model, fit on training data, predict on the test data
model_m2 = LinearRegression() 
model_m2.fit(X_train_m2, y_train_m2) 
y_pred_m2 = model_m2.predict(X_test_m2)

#evaluate
train_r2_m2 = model_m2.score(X_train_m2, y_train_m2)
r2_m2 = r2_score(y_test_m2, y_pred_m2)
rmse_m2 = np.sqrt(mean_squared_error(y_test_m2, y_pred_m2))

print(f"Full Model Train R2: {train_r2_m2:.4f}")
print(f"Full Model Test R2: {r2_m2:.4f}")
print(f"Baseline Test RMSE: {rmse:.4f}")
print(f"Full Model Test RMSE: {rmse_m2:.4f}")
print(f"Reduction: {rmse - rmse_m2:.4f}")

#comparing the baseline
print(f"Baseline Test R2: {r2:.4f}")
print(f"Full Model Test R2: {r2_m2:.4f}")
print(f"Improvement: {r2_m2 - r2:.4f}")

#baseline used one feature(temperature_2m_min) to predict (temperature_2m_max)
#Full model added precipitation_sum, wind_speed_10m_max, and is_summer
#Compare to my R2 baseline to full model R2, 
#My baseline R2 rose from 0.54 to 0.72 an improvement is about 0.1767. 
#Test RMSE on the other hand fell from 3.27 to 2.57, Therefore adding features helps substantially on mdoel's performance.

#Prints each feature name with its coefficient
for name, coef in zip(feature_cols, model_m2.coef_):
    print(f"{name:22s}: {coef:+.3f}")
print(f"{'intercept':22s}: {model_m2.intercept_:+.3f}")
    
#coefficient interpetation
#temperature_2m_min(+0.99) and is_summer(+0.57) raise the predicted high 
#and wind_speed_10m_max(-0.34) lower it which is because windier days are cooler.
#precipitation_sum(+0.004) is basically 0, which adds nothing
#non of the sings are very suprising because warmer nights go with warmer days, summer is bit warmer
#and wind has a cooling effect.
#train R2(0.74) and Test R2(0.72) are close, the gap is about 0.04
#this means the model is not overfitting and generalizes well to new data. 

# --- Task 6: Evaluate and Summarize ---

#predicted vs actual plot full model on the test set
plt.figure(figsize=(7,7))
plt.scatter(y_test_m2, y_pred_m2, alpha=0.5)

#diagonal reference line, predicted =actual
low = min(y_pred_m2.min(), y_test_m2.min())
high = max(y_pred_m2.max(), y_test_m2.max())
plt.plot([low, high], [low, high], "r--", label="Perfect Prediction")

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Actual Temperature_2m_max")
plt.ylabel("Predicted Temperature_2m_max")
plt.legend()

plt.savefig("outputs/predicted_vs_actual_high.png")
plt.show()

#The model struggles most at the high end: the hotest days(actual 25-31) sitting well below thee diagonal.
#The low end has a few scattered misses, and the middle is most closet to the line
#the model is pulled towards the middle, model's mistakes aren't the same size evreywhere

#plain-language summary in comments covering:

#The size of the dataset and the test set:
# 65 rows(one year), split(80/20) ( days held out for tesing)
#Adding features raised trest R2 from 0.54 to 0.72 and lowered RMSE to 3.27 to 2.57 degree.
#Train/Test are close, no overfitting

#The RMSE and R² of your full model, and what a typical error of that size means on a temperature scale:
#The Full Model RMSE is 2.57 and R2 is 0.72,
#RMSE typical prediction is off by about 2.57 degree c. 
#R2, model explains about 72% of variation in daiily max temperature
#comparing to the baseline which had RMSE of 3.27 and R2 of 0.54, which shows that the full model is better

#Which feature has the largest effect on the predicted high, and in which direction:
#Temperature_2m_min(+0.992) is the largest effect on the prediction hiigh, and the direction is ppositive
#warmer lows go with warmer highs. 

#One result that surprised you: 
#is_summer's effect (+0.57 degrees) on the predicted high,
#compared to other features is-summer have small effects, when my expectation was that summer matter alot
#but then temperature_2m_min captures most of the deasonal signal
