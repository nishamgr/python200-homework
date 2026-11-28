from __future__ import annotations
import joblib
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Prediction:
    """ The model's verdict for a single day
    
    Attributes:
    label: (str) either "good" or "skip"
    probability: (float) the probability, from 0 to 1, that the day is good for running
    """
    label: str
    probability: float

# --- Task 2: The WeatherClassifier Class ---
FEATURES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]

class WeatherClassifier:
    """ Loads a saved weather pipeline once and predicts on many days."""
    
    def __init__(self,model_path: str | Path):
        """ Loas the saved pipeline from disk.
        
        Args: 
            model_path: path to a .pkl file saved with joblib
        Raises:
            FileNotFooundError: if no file exists at model_path"""
        model_path = Path(model_path)       #accepts a str or a path
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model file at {model_path}, Train and save a model first."
            )
        self.model = joblib.load(model_path)  #load once, store on self
        self.features = FEATURES
    
    def predict(self, days: list[dict]) -> list[Prediction]:
        """ predicts whether each day is good for running
        Args:
            days: List of dictionaries, each with the keys temperature_2m_max, temperature_2m_min
            precipitation_sum, and wind_speed_10m_max 
        Returns: 
            One prediction per input a day, in the same order
        
        Raise: 
            ValueError: if any required feature is missing
        """
        if not days:
            return []  # returns empty list when nothing to predict
        #otherwise bild a df from days
        frame = pd.DataFrame(days)
        #finds any required features that are not in the input
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise ValueError(f"missing required features: {missing}")
        
        X = frame[self.features]        #selets the features in the fixed order
        labels = self.model.predict(X)      #0 or 1 for each day
        probabilities = self.model.predict_proba(X)[:, 1]   #probability of good
        
        #returns the results
        return[
            Prediction("good" if int(label)==1 else "skip", float(prob))
            #Prediction("skip", float(prob))
            for label, prob in zip(labels, probabilities)
        ]

