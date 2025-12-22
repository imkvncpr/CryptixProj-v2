"""
Enhanced Cryptocurrency Predictor Base Class

Provides foundation for LSTM-based crypto price prediction with:
- Flexible LSTM architecture
- Comprehensive evaluation metrics
- Cross-validation support
- Model persistence
- Metadata tracking
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Any, Dict, Tuple, List, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import json
from datetime import datetime, timezone
from loguru import logger


class CryptoPredictor(ABC):
    """
    Abstract base class for cryptocurrency price prediction.
    
    Subclasses must implement download_data() method.
    """
    
    def __init__(
        self,
        ticker: str,
        look_back: int = 60,
        units: List[int] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        model_version: str = "v1.0"
    ):
        """
        Initialize cryptocurrency predictor.
        
        Args:
            ticker: Cryptocurrency ticker (e.g., 'BTC-USD')
            look_back: Number of previous days to use for prediction
            units: List of LSTM units for each layer
            dropout: Dropout rate for regularization
            learning_rate: Adam optimizer learning rate
            model_version: Model version string
        """
        self.ticker = ticker
        self.look_back = look_back
        self.units = units if units is not None else [50, 50]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model_version = model_version
        
        # Initialize scaler and model
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None
        
        # Metadata
        self.metadata = {
            'ticker': ticker,
            'look_back': look_back,
            'units': units,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'model_version': model_version,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Initialized {self.__class__.__name__} for {ticker}")
    
    @abstractmethod
    def download_data(self, **kwargs) -> Optional[pd.DataFrame]:
        """
        Abstract method to download cryptocurrency data.
        
        Must be implemented by subclasses.
        """
        pass
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'close'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for LSTM training.
        
        Args:
            data: DataFrame with price data
            target_column: Column to predict
            
        Returns:
            (X, y) tuple of features and targets
        """
        # Extract target column
        df = data[[target_column]].copy()
        
        # Scale data to [0, 1]
        scaled_data = self.scaler.fit_transform(df.values)
        
        # Create sequences
        x, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            x.append(scaled_data[i - self.look_back:i, 0])
            y.append(scaled_data[i, 0])
        
        x = np.array(x)
        y = np.array(y)
        
        # Reshape for LSTM: [samples, time_steps, features]
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        
        logger.debug(f"Preprocessed data - X shape: {x.shape}, y shape: {y.shape}")
        return x, y
    
    def build_model(self) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential(name=f"{self.ticker}_predictor")
        
        # First LSTM layer
        model.add(keras.layers.LSTM(
            self.units[0],
            return_sequences=len(self.units) > 1,
            input_shape=(self.look_back, 1),
            name="lstm_1"
        ))
        model.add(keras.layers.Dropout(self.dropout, name="dropout_1"))
        
        # Additional LSTM layers
        for i, num_units in enumerate(self.units[1:], start=2):
            return_seq = i < len(self.units)
            model.add(keras.layers.LSTM(
                num_units,
                return_sequences=return_seq,
                name=f"lstm_{i}"
            ))
            model.add(keras.layers.Dropout(self.dropout, name=f"dropout_{i}"))
        
        # Dense layers
        model.add(keras.layers.Dense(25, activation='relu', name="dense_1"))
        model.add(keras.layers.Dense(1, name="output"))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)  # Fixed typo
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        logger.info(f"Built model for {self.ticker}")  # Fixed typo
        model.summary(print_fn=logger.debug)
        
        return model
    
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: Optional[List[keras.callbacks.Callback]] = None
    ) -> keras.callbacks.History:
        """
        Train the model with training data and optional validation data.
        
        Args:
            x_train: Training features
            y_train: Training targets
            x_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            self.model = self.build_model()
        
        # Prepare validation data
        validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None
        
        # Use default callbacks if none provided
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        
        # Train
        self.history = self.model.fit(
            x_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.success(f"Training completed for {self.ticker}")
        return self.history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict on new data.
        
        Args:
            x: Input features
            
        Returns:
            Predicted prices
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        
        predictions_scaled = self.model.predict(x, verbose=0)
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def predict_future(
        self,
        last_sequence: np.ndarray,
        days: int = 7
    ) -> np.ndarray:
        """
        Predict future prices iteratively.
        
        Args:
            last_sequence: Last sequence of prices (scaled)
            days: Number of days to predict
            
        Returns:
            Array of predicted prices
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict next day
            next_pred_scaled = self.model.predict(
                current_sequence.reshape(1, self.look_back, 1),
                verbose=0
            )[0, 0]
            
            predictions.append(next_pred_scaled)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred_scaled)
        
        # Convert to numpy array and inverse transform
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            x_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        
        # Predictions
        y_pred_scaled = self.model.predict(x_test, verbose=0).flatten()
        
        # Inverse transform
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inv = self.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = np.mean((y_test_inv - y_pred_inv) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_inv - y_pred_inv))
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / (np.abs(y_test_inv) + 1e-8))) * 100
        
        # Directional accuracy
        y_test_direction = np.diff(y_test_inv) > 0
        y_pred_direction = np.diff(y_pred_inv) > 0
        directional_accuracy = np.mean(y_test_direction == y_pred_direction) * 100
        
        # R² score
        ss_res = np.sum((y_test_inv - y_pred_inv) ** 2)
        ss_tot = np.sum((y_test_inv - np.mean(y_test_inv)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Directional Accuracy (%)': directional_accuracy,
            'R2 Score': r2_score
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def cross_validate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Perform time series cross-validation.
        
        Args:
            x: Features
            y: Targets
            n_splits: Number of folds
            
        Returns:
            Dictionary of cross-validation results
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'mae': [],
            'mse': [],
            'directional_accuracy': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(x), 1):
            logger.info(f"Starting fold {fold}/{n_splits}")
            
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]
            
            # Reset model for each fold
            self.model = None
            
            # Train
            self.fit(x_train, y_train, epochs=50, batch_size=32)
            
            # Evaluate
            metrics = self.evaluate(x_test, y_test)
            
            cv_scores['mae'].append(metrics['MAE'])
            cv_scores['mse'].append(metrics['MSE'])
            cv_scores['directional_accuracy'].append(metrics['Directional Accuracy (%)'])
        
        # Aggregate results
        results = {
            metric: {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
            for metric, scores in cv_scores.items()
        }
        
        logger.info(f"Cross-validation results: {results}")
        return results
    
    def save_model(self, path: str):
        """
        Save model, scaler, and metadata to disk.
        
        Args:
            path: Directory path to save to
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path_obj / "model.keras"  # Use .keras format (recommended)
        self.model.save(model_path)
        
        # Save scaler
        import pickle
        scaler_path = path_obj / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        self.metadata['saved_at'] = datetime.now(timezone.utc).isoformat()
        metadata_path = path_obj / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.success(f"Model, scaler, and metadata saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model, scaler, and metadata from disk.
        
        Args:
            path: Directory path to load from
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        
        # Load model
        model_path = path_obj / "model.keras"
        if not model_path.exists():
            model_path = path_obj / "model.h5"  # Fallback to old format
        self.model = keras.models.load_model(str(model_path))
        
        # Load scaler
        import pickle
        scaler_path = path_obj / "scaler.pkl"
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load metadata
        metadata_path = path_obj / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Update instance attributes
        self.ticker = self.metadata.get('ticker', self.ticker)
        self.look_back = self.metadata.get('look_back', self.look_back)
        self.units = self.metadata.get('units', self.units)
        self.dropout = self.metadata.get('dropout', self.dropout)
        self.learning_rate = self.metadata.get('learning_rate', self.learning_rate)
        self.model_version = self.metadata.get('model_version', self.model_version)
        
        logger.success(f"Model, scaler, and metadata loaded from {path}")
    
    def _get_default_callbacks(self) -> List[keras.callbacks.Callback]:
        """Get default callbacks for training."""
        return [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as a dictionary."""
        return {
            'ticker': self.ticker,
            'look_back': self.look_back,
            'units': self.units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'model_version': self.model_version
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ticker='{self.ticker}', version='{self.model_version}')"