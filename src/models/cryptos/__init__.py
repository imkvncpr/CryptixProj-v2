"""
Cryptocurrency Predictors

Provides LSTM-based predictors for major cryptocurrencies.
"""

from .cryptocoin_predictor import CryptoCoinPredictor
from .bitcoin_predictor import BitcoinPredictor
from .tether_predictor import TetherPredictor
from .ethereum_predictor import EthereumPredictor
from .usdcoin_predictor import USDCoinPredictor

__all__ = [
    'CryptoCoinPredictor',
    'BitcoinPredictor',
    'TetherPredictor',
    'EthereumPredictor',
    'USDCoinPredictor'
]