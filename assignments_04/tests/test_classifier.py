import pytest
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from weather_model import WeatherClassifier, Prediction, FEATURES

# --- Task 4: The Test Suite --
#fixtures - setup code that the pytest hands to tests
@pytest.fixture
def model_path(tmp_path):
    """ Train a tiny model on obvius days and save it to a temp file"""
    #15 copues of clearly good days
    good = [{"temperature_2m_max": 19, "temperature_2m_min": 8,
             "precipitation_sum": 0.0, "wind_speed_10m_max": 10}] * 15
    # 15 copes of a clearly bad day
    bad = [{"temperature_2m_max": 3, "temperature_2m_min": - 7,
            "precipitation_sum": 13.0, "wind_speed_10m_max": 40.0}] * 15
    #features table in the fixed orderm with labels 1 = good and 0=skip
    X = pd.DataFrame(good + bad)[FEATURES]
    y = [1] * 15 + [0] * 15
    #pipline mpodel: scaler, and logistic regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X, y)      #trains the tiny model
    #clean temporary folder that pytest provides for each test
    path = tmp_path /"weather_classifier.pkl"
    joblib.dump(pipe, path)
    return path

@pytest.fixture
def clf(model_path):
    """A WeatherClassifier loaded from the savedb tiny model"""
    return WeatherClassifier(model_path)

#runs once per (day, expected_label) pair. 
#A clearly good day is labeled "good". and A clearly bad day (cold, rainy, windy) is labeled "skip" using  @pytest.mark.parametrize 
@pytest.mark.parametrize(
    "day, expected_label",
    [
        # 1: clearly good day -> "good"
        ({"temperature_2m_max": 18, "temperature_2m_min": 9,
          "precipitation_sum": 0.0, "wind_speed_10m_max": 10}, "good"),
        # 2: cold, rainy, windy day -> "skip"
        ({"temperature_2m_max": 3, "temperature_2m_min": -5,
          "precipitation_sum": 10.0, "wind_speed_10m_max": 38}, "skip"),
    ],
)
def test_labels_for_good_and_bad_days(clf, day, expected_label):
    result = clf.predict([day])
    assert result[0].label == expected_label
    
#The returned probability is between 0 and 1.
def test_probability_is_between_zero_and_one(clf):
    result = clf.predict([
        {"temperature_2m_max": 15, "temperature_2m_min": 8,
            "precipitation_sum": 1.0, "wind_speed_10m_max": 12.0}
    ])
    assert 0.0 <= result[0].probability <= 1.0  

#Passing three days returns three Prediction objects, in order.
def test_one_prediction_per_day_in_order(clf):
    days = [
        {"temperature_2m_max": 18, "temperature_2m_min": 9,
         "precipitation_sum": 0.0, "wind_speed_10m_max": 10},       #good
        {"temperature_2m_max": 2, "temperature_2m_min": -6,
         "precipitation_sum": 12.0, "wind_speed_10m_max": 40},      #skip
        {"temperature_2m_max": 18, "temperature_2m_min": 9,
         "precipitation_sum": 0.0, "wind_speed_10m_max": 10},       #good
    ]
    results = clf.predict(days)
    assert len(results) == 3
    assert all(isinstance(r, Prediction) for r in results)
    
#Passing an empty list returns an empty list.  
def test_empty_list_returns_empty(clf):
    assert clf.predict([]) == []
    
#A day missing a required feature raises ValueError (use pytest.raises(ValueError, match=...)).
def test_missing_features_raises(clf):
    with pytest.raises(ValueError, match="missing required features"):
        clf.predict([{"temperature_2m_max": 18, "temperature_2m_min": 9}])
    
#Constructing a WeatherClassifier with a path to a missing file raises FileNotFoundError (use match).
def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model file"):
        WeatherClassifier(tmp_path/ "does_not_exist.pk1")
        
#