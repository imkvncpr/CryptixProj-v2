import pytest
import numpy as np
from src.ml.features import FeatureEngineer
from src.ml.models import MLModels
from src.ml.training import ModelTrainer
from src.ml.inference import ModelInference


class TestFeatureEngineer:
    def test_price_features(self):
        prices = np.array(list(range(100, 150)), dtype=float)
        feats = FeatureEngineer.create_price_features(prices, window=10)
        assert 'returns' in feats
        assert 'volatility' in feats
        assert 'momentum' in feats
    
    def test_indicator_features(self):
        prices = np.array(list(range(100, 150)), dtype=float)
        indicators = {'rsi': np.random.rand(50) * 100}
        feats = FeatureEngineer.create_indicator_features(indicators)
        assert 'rsi_norm' in feats
    
    def test_statistical_features(self):
        prices = np.array(list(range(100, 150)), dtype=float)
        feats = FeatureEngineer.create_statistical_features(prices, window=10)
        assert 'skewness' in feats
        assert 'kurtosis' in feats
    
    def test_all_features(self):
        prices = np.array(list(range(100, 150)), dtype=float)
        indicators = {'rsi': np.random.rand(50) * 100}
        df = FeatureEngineer.create_all_features(prices, indicators, window=10)
        assert len(df) == 50


class TestMLModels:
    def test_random_forest_create(self):
        model = MLModels.create_model('random_forest', n_trees=5)
        assert model is not None
    
    def test_model_training(self):
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        model = MLModels.create_model('random_forest', n_trees=5)
        model.fit(X[:80], y[:80])
        score = model.score(X[80:], y[80:])
        assert isinstance(score, (float, np.floating))
    
    def test_xgboost_available(self):
        models = MLModels.list_available_models()
        assert 'xgboost' in models
    
    def test_neural_network_available(self):
        models = MLModels.list_available_models()
        assert 'neural_network' in models


class TestModelTrainer:
    def test_split_data(self):
        X = np.random.randn(200, 10)
        y = np.random.randn(200)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = ModelTrainer.split_data(X, y)
        assert len(X_train) == 140
        assert len(X_val) == 30
        assert len(X_test) == 30
    
    def test_train_model(self):
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        model = MLModels.create_model('random_forest', n_trees=5)
        metrics = ModelTrainer.train_model(model, X[:80], y[:80], X[80:], y[80:])
        assert 'train_score' in metrics
    
    def test_evaluate_model(self):
        pred = np.random.randn(50)
        actual = np.random.randn(50)
        metrics = ModelTrainer.evaluate_model(pred, actual)
        assert 'mse' in metrics
        assert 'mae' in metrics
    
    def test_walk_forward_validation(self):
        X = np.random.randn(200, 10)
        y = np.random.randn(200)
        model = MLModels.create_model('random_forest', n_trees=3)
        scores = ModelTrainer.walk_forward_validation(X, y, model, window_size=50, step=10)
        assert len(scores) > 0


class TestModelInference:
    def test_batch_predict(self):
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        model = MLModels.create_model('random_forest', n_trees=5)
        model.fit(X[:80], y[:80])
        pred = ModelInference.batch_predict(model, X[80:], batch_size=5)
        assert len(pred) == 20
    
    def test_prediction_to_signal_buy(self):
        signal = ModelInference.prediction_to_signal(0.8)
        assert signal['signal_type'] == 'BUY'
    
    def test_prediction_to_signal_sell(self):
        signal = ModelInference.prediction_to_signal(-0.8)
        assert signal['signal_type'] == 'SELL'
    
    def test_prediction_to_signal_hold(self):
        signal = ModelInference.prediction_to_signal(0.1)
        assert signal['signal_type'] == 'HOLD'
    
    def test_predict_live(self):
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        model = MLModels.create_model('random_forest', n_trees=5)
        model.fit(X[:80], y[:80])
        inference = ModelInference(model=model)
        pred = inference.predict_live(X[80])
        assert isinstance(pred, (float, np.floating))
    
    def test_predict_sequence(self):
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        model = MLModels.create_model('random_forest', n_trees=5)
        model.fit(X[:80], y[:80])
        inference = ModelInference(model=model)
        preds = inference.predict_sequence(X[80:90])
        assert len(preds) == 10
