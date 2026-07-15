import numpy as np
from typing import Tuple, Optional, List, Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb  # type: ignore
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    KERAS_AVAILABLE = True
except:
    KERAS_AVAILABLE = False


class RandomForestModel:
    def __init__(self, n_trees: int = 100, max_depth: int = 10, min_samples_split: int = 5):
        self.model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        X_scaled = self.scaler.transform(X_test)
        return self.model.score(X_scaled, y_test)


class XGBoostModel:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 6):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        self.model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, verbosity=0)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    
    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        X_scaled = self.scaler.transform(X_test)
        return self.model.score(X_scaled, y_test)


class NeuralNetworkModel:
    def __init__(self, input_size: int, hidden_layers: List[int] = None, dropout_rate: float = 0.2):
        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras not installed")
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]
        self.input_size = input_size
        self.scaler = StandardScaler()
        self.model = self._build_model(input_size, hidden_layers, dropout_rate)
        self.is_fitted = False
    
    def _build_model(self, input_size: int, hidden_layers: List[int], dropout_rate: float):
        model = keras.Sequential()
        model.add(keras.layers.Dense(hidden_layers[0], activation="relu", input_dim=input_size))
        model.add(keras.layers.Dropout(dropout_rate))
        for units in hidden_layers[1:]:
            model.add(keras.layers.Dense(units, activation="relu"))
            model.add(keras.layers.Dropout(dropout_rate))
        model.add(keras.layers.Dense(1, activation="linear"))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32, verbose: int = 0) -> None:
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.is_fitted = True
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        X_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()


class MLModels:
    AVAILABLE_MODELS = {
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "neural_network": NeuralNetworkModel
    }
    
    @staticmethod
    def create_model(model_type: str, **kwargs):
        if model_type not in MLModels.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        ModelClass = MLModels.AVAILABLE_MODELS[model_type]
        return ModelClass(**kwargs)
    
    @staticmethod
    def list_available_models() -> List[str]:
        return list(MLModels.AVAILABLE_MODELS.keys())
