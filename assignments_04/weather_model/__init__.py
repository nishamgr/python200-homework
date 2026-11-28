# --- Task 3: The Package Entry Point ---

"""weather_model: a reusable weather classifier component."""

from weather_model.classifier import WeatherClassifier, Prediction, FEATURES

__all__ = ["WeatherClassifier", "Prediction", "FEATURES"]