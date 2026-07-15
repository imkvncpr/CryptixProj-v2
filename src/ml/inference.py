import numpy as np
import pickle
import joblib
from typing import Tuple, Optional, Dict, List
from datetime import datetime


class ModelInference:
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler
        self.prediction_history = []
    
    @staticmethod
    def batch_predict(model, X_data: np.ndarray, batch_size: int = 32) -> np.ndarray:
        predictions = []
        
        for i in range(0, len(X_data), batch_size):
            batch = X_data[i:i+batch_size]
            batch_pred = model.predict(batch)
            if isinstance(batch_pred, np.ndarray):
                predictions.append(batch_pred)
            else:
                predictions.append(np.array([batch_pred]))
        
        if len(predictions) == 0:
            return np.array([])
        
        return np.concatenate(predictions)
    
    @staticmethod
    def prediction_to_signal(prediction: float, threshold: float = 0.5, confidence_mapping: bool = True) -> Dict:
        if confidence_mapping:
            confidence = min(int(abs(prediction) * 100), 100)
        else:
            confidence = 50
        
        if prediction > threshold:
            return {
                'signal_type': 'BUY',
                'strength': 'STRONG' if abs(prediction) > 1.0 else 'MODERATE',
                'confidence': confidence,
                'prediction_value': float(prediction)
            }
        elif prediction < -threshold:
            return {
                'signal_type': 'SELL',
                'strength': 'STRONG' if abs(prediction) > 1.0 else 'MODERATE',
                'confidence': confidence,
                'prediction_value': float(prediction)
            }
        else:
            return {
                'signal_type': 'HOLD',
                'strength': 'WEAK',
                'confidence': 50,
                'prediction_value': float(prediction)
            }
    
    @staticmethod
    def save_model(model, model_path: str, method: str = 'joblib') -> str:
        try:
            if method == 'joblib':
                joblib.dump(model, model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            return model_path
        except Exception as e:
            raise Exception(f"Failed to save model: {e}")
    
    @staticmethod
    def load_model(model_path: str, method: str = 'joblib'):
        try:
            if method == 'joblib':
                model = joblib.load(model_path)
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            return model
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def predict_live(self, latest_data: np.ndarray, scale: bool = True) -> float:
        if self.model is None:
            raise ValueError("Model not set.")
        
        if len(latest_data.shape) == 1:
            latest_data = latest_data.reshape(1, -1)
        
        if scale and self.scaler is not None:
            latest_data = self.scaler.transform(latest_data)
        
        prediction = self.model.predict(latest_data)[0]
        
        self.prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'prediction': float(prediction)
        })
        
        return float(prediction)
    
    def predict_sequence(self, X_sequence: np.ndarray, scale: bool = True) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not set.")
        
        if scale and self.scaler is not None:
            X_sequence = self.scaler.transform(X_sequence)
        
        predictions = self.model.predict(X_sequence)
        return predictions
    
    def get_prediction_history(self) -> List[Dict]:
        return self.prediction_history
    
    def clear_history(self) -> None:
        self.prediction_history = []
