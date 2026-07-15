import numpy as np
import pandas as pd
from typing import Dict, Optional, List


class FeatureEngineer:
    def __init__(self):
        pass
    
    @staticmethod
    def create_price_features(prices: np.ndarray, window: int = 20) -> Dict[str, np.ndarray]:
        features = {}
        n = len(prices)
        
        returns = np.diff(prices) / prices[:-1]
        features['returns'] = np.concatenate(([0], returns))
        
        volatility = pd.Series(prices).rolling(window).std().values
        features['volatility'] = np.nan_to_num(volatility)
        
        sma = pd.Series(prices).rolling(window).mean().values
        momentum = prices - sma
        features['momentum'] = np.nan_to_num(momentum)
        
        rolling_high = pd.Series(prices).rolling(window).max().values
        rolling_low = pd.Series(prices).rolling(window).min().values
        high_low_ratio = (rolling_high - rolling_low) / (rolling_low + 1e-10)
        features['high_low_ratio'] = np.nan_to_num(high_low_ratio)
        
        close_position = (prices - rolling_low) / (rolling_high - rolling_low + 1e-10)
        features['close_position'] = np.nan_to_num(close_position)
        
        return features
    
    @staticmethod
    def create_indicator_features(indicators: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        features = {}
        
        for indicator_name, indicator_values in indicators.items():
            if len(indicator_values) > 0:
                min_val = np.nanmin(indicator_values)
                max_val = np.nanmax(indicator_values)
                
                if max_val > min_val:
                    normalized = (indicator_values - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(indicator_values)
                
                features[f'{indicator_name}_norm'] = np.nan_to_num(normalized)
        
        return features
    
    @staticmethod
    def create_statistical_features(prices: np.ndarray, window: int = 20) -> Dict[str, np.ndarray]:
        features = {}
        n = len(prices)
        returns = np.diff(prices) / prices[:-1]
        
        price_series = pd.Series(prices)
        skewness = price_series.rolling(window).skew().values
        features['skewness'] = np.nan_to_num(skewness)
        
        kurtosis = price_series.rolling(window).kurt().values
        features['kurtosis'] = np.nan_to_num(kurtosis)
        
        autocorr = np.zeros(n)
        for i in range(window, len(returns)):
            corr = np.corrcoef(returns[i-window:i], returns[i-window+1:i+1])[0, 1]
            autocorr[i] = np.nan_to_num(corr)
        features['autocorr'] = autocorr
        
        price_range = np.max(prices) - np.min(prices)
        if price_range > 0:
            features['price_range_norm'] = (prices - np.min(prices)) / price_range
        else:
            features['price_range_norm'] = np.zeros_like(prices)
        
        return features
    
    @staticmethod
    def create_all_features(prices: np.ndarray, indicators: Optional[Dict[str, np.ndarray]] = None, window: int = 20) -> pd.DataFrame:
        all_features = {}
        
        price_features = FeatureEngineer.create_price_features(prices, window)
        all_features.update(price_features)
        
        if indicators is not None:
            indicator_features = FeatureEngineer.create_indicator_features(indicators)
            all_features.update(indicator_features)
        
        stat_features = FeatureEngineer.create_statistical_features(prices, window)
        all_features.update(stat_features)
        
        df = pd.DataFrame(all_features)
        
        return df
