import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelTrainer:
    def __init__(self):
        self.training_history = {}
    
    @staticmethod
    def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[Tuple, Tuple, Tuple]:
        total = len(X)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    @staticmethod
    def train_model(model, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        metrics = {'train_score': train_score}
        
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            metrics['val_score'] = val_score
        
        return metrics
    
    @staticmethod
    def walk_forward_validation(X: np.ndarray, y: np.ndarray, model, window_size: int = 100, step: int = 20) -> List[float]:
        scores = []
        
        for i in range(0, len(X) - window_size - step, step):
            X_train = X[i:i+window_size]
            y_train = y[i:i+window_size]
            X_test = X[i+window_size:i+window_size+step]
            y_test = y[i+window_size:i+window_size+step]
            
            try:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                scores.append(score)
            except:
                scores.append(0)
        
        return scores
    
    @staticmethod
    def evaluate_model(predictions: np.ndarray, actual: np.ndarray, metrics: List[str] = None) -> Dict[str, float]:
        if metrics is None:
            metrics = ['mse', 'mae', 'r2']
        
        results = {}
        
        for metric in metrics:
            if metric == 'mse':
                results['mse'] = mean_squared_error(actual, predictions)
            elif metric == 'rmse':
                results['rmse'] = np.sqrt(mean_squared_error(actual, predictions))
            elif metric == 'mae':
                results['mae'] = mean_absolute_error(actual, predictions)
            elif metric == 'r2':
                results['r2'] = r2_score(actual, predictions)
            elif metric == 'mape':
                mape = np.mean(np.abs((actual - predictions) / (actual + 1e-10))) * 100
                results['mape'] = mape
        
        return results
    
    @staticmethod
    def calculate_directional_accuracy(predictions: np.ndarray, actual: np.ndarray) -> float:
        pred_direction = (np.diff(predictions) > 0).astype(int)
        actual_direction = (np.diff(actual) > 0).astype(int)
        
        correct = np.sum(pred_direction == actual_direction)
        accuracy = correct / len(pred_direction)
        
        return accuracy
