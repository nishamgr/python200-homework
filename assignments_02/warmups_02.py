
# --- scikit-learn API ---

# Q1) 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

#two days i want predictions for
new_lows = np.array([[6], [18]])

model = LinearRegression() #create model
model.fit(temp_min, temp_max) #fit
predictions = model.predict(new_lows) #predict

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for a low of 6:", predictions[0])
print("Predicted high for a low of 18:", predictions[1])

# for a day with a low of 6, the model expects a high of about 12.6.
# for a day with a low of 18, the model expects a high of about 24.8.

# Q2) 
x = np.array([10, 20, 30, 40, 50])
print("Original shape:", x.shape)

x_2d = x.reshape(-1,1)
print("Reshaped shape:", x_2d.shape)

#Scikit-learn needs X to be 2D array because scikit-learn expects X to be 2D, with rows as samples and column as features.
#rehaping helps make it clear, so we know that its 5 samples, 1 features (5,1), which 1D array won't be able to say whether
#it's 5 samples with 1 feature or 1 sample with 5 features.

# --- Linear Regression ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Q1) 
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm" )
plt.title("Daily High vs Low")
plt.xlabel("Daily low (temp_min)")
plt.ylabel("Daily high (temp_max)")
plt.legend()

 #saves the plt as an img in the outputs folder
plt.savefig("outputs/high_vs_low.png")   
plt.show()

#Yes, I see two visible groups in red and blue color, both bands slope upwards at about the same angle; they look like two parallel lines. 
#It suggests that is_rainy affects the daily high, rainy days(red) have lower highs than the dry days(blue) at the same temp_min.

# Q2) 
#Define feature, reshaped to 2d and target
X = temp_min.reshape(-1,1)
y = temp_max

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#prints shapes of all four arrays
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3)
#fit to LR
model = LinearRegression()
model.fit(X_train, y_train)  #uses only training data

#print slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

#predict
y_pred = model.predict(X_test)
print("Predict:", y_pred)

#Evaluate RMSE AND R2
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("R2:", r2)

#The slope output I got is 0.94, it means that for every 1 degree increase in the daily low,
# the model predicts the daily high increase by 1 degree. AS showen in the high_vs_low scatter plot; 
#high rise steadily as low rise.

# Q4)
#added x_rainy as second feature
X_full = np.column_stack([temp_min, is_rainy])

X_train_m2, X_test_m2, y_train_m2, y_test_m2 = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)
#fit the new model
model2 = LinearRegression()
model2.fit(X_train_m2, y_train_m2)

#print r2; compare R2 for both models
R2_model1 = model.score(X_test, y_test)
R2_model2 = model2.score(X_test_m2, y_test_m2)

print("R2 (temp_min only):", R2_model1)
print("R2 (temp_min + is_rainy):", R2_model2)
print("temp_min coefficient:", model2.coef_[0])
print("is_rainy coefficient:", model2.coef_[1])

#is_rainy coefficient is -2.919, which means that the model predicts a rainy days high will be about 2.9 degree lower
#than a dry day's high.
#in pratical term, rain lowers the daily high by about 3 degree at a given daily low.

# Q4)
#model2 predcict 
y_pred_m2 = model2.predict(X_test_m2)

plt.figure()
plt.scatter(y_pred_m2, y_test_m2, color="blue", alpha=0.5, label="Test days")

#diagonal reference line, predicted =actual
low = min(y_pred_m2.min(), y_test_m2.min())
high = max(y_pred_m2.max(), y_test_m2.max())

plt.plot([low, high], [low, high], color="red", linestyle="--", label="Perfect prediction")
plt.xlabel("Model's prediction daily high (temp_max)")
plt.ylabel("Actual daily high (temp_max)")
plt.title("Predicted vs Actual")
plt.legend()

plt.savefig("outputs/predicted_vs_actual_warmup.png")
plt.show()

#If the point falls above the diagonal, it means that the actual daily is greater than the model predicted, so the model
#underpredicted or gussed too low for that day.
#if the points falls below the diagonal. it means that the acutal daily is lowert than the model predicted, so the model
#overpredicted or gussed too high for that day. 
