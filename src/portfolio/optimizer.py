import numpy as np
from typing import Dict, List, Tuple


class PortfolioOptimizer:
    def __init__(self):
        pass
    
    @staticmethod
    def min_variance_portfolio(returns: np.ndarray) -> np.ndarray:
        n_assets = returns.shape[1]
        cov_matrix = np.cov(returns.T)
        
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except:
            cov_inv = np.linalg.pinv(cov_matrix)
        
        ones = np.ones(n_assets)
        weights = np.dot(cov_inv, ones)
        weights = weights / np.sum(weights)
        
        return np.maximum(weights, 0)
    
    @staticmethod
    def max_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> np.ndarray:
        n_assets = returns.shape[1]
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)
        
        excess_returns = mean_returns - risk_free_rate
        
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except:
            cov_inv = np.linalg.pinv(cov_matrix)
        
        weights = np.dot(cov_inv, excess_returns)
        
        if np.sum(weights) != 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n_assets) / n_assets
        
        return np.maximum(weights, 0)
    
    @staticmethod
    def equal_weight_portfolio(num_assets: int) -> np.ndarray:
        weights = np.ones(num_assets) / num_assets
        return weights
    
    @staticmethod
    def risk_parity_portfolio(returns: np.ndarray) -> np.ndarray:
        n_assets = returns.shape[1]
        volatilities = np.std(returns, axis=0)
        inv_vols = 1.0 / volatilities
        weights = inv_vols / np.sum(inv_vols)
        return weights
