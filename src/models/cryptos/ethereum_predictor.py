"""
Ethereum Cryptocurrency Predictor

Optimized LSTM parameters for Ethereum price prediction.
Inherits YFinance functionality from CryptoCoinPredictor.
"""

from typing import List
from src.models.cryptos.cryptocoin_predictor import CryptoCoinPredictor


class EthereumPredictor(CryptoCoinPredictor):
    """
    Ethereum price predictor using LSTM.
    
    Optimized parameters for Ethereum's characteristics:
    - 3-layer LSTM: [100, 50, 25] units (slightly simpler than Bitcoin)
    - Moderate dropout: 0.2
    - Standard learning rate: 0.001
    - Look back: 60 days
    
    Ethereum tends to follow Bitcoin trends but with its own dynamics
    related to DeFi, NFT markets, and network upgrades.
    """
    
    def __init__(
        self,
        ticker: str = "ETH-USD",
        look_back: int = 60,
        units: List[int] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        model_version: str = "v1.0"
    ):
        """
        Initialize Ethereum predictor with optimized defaults.
        
        Args:
            ticker: Ethereum ticker (default: 'ETH-USD')
            look_back: Days to look back (default: 60)
            units: LSTM units per layer (default: [100, 50, 25])
            dropout: Dropout rate (default: 0.2)
            learning_rate: Learning rate (default: 0.001)
            model_version: Model version (default: 'v1.0')
        """
        # Ethereum-optimized architecture
        if units is None:
            units = [100, 50, 25]  # Slightly less complex than Bitcoin
        
        super().__init__(
            ticker=ticker,
            name="Ethereum",
            look_back=look_back,
            units=units,
            dropout=dropout,
            learning_rate=learning_rate,
            model_version=model_version
        )


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("ETHEREUM PREDICTOR - EXAMPLE USAGE")
    print("=" * 70)
    
    # Initialize predictor
    predictor = EthereumPredictor()
    print(f"\n1. Initialized: {predictor}")
    print(f"   Architecture: {predictor.units}")
    print(f"   Look back: {predictor.look_back} days")
    
    # Download data
    print("\n2. Downloading Ethereum data...")
    data = predictor.download_data(period="1y")
    
    if data is None:
        print("❌ Failed to download data")
        exit(1)
    
    print(f"✓ Downloaded {len(data)} days of data")
    print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Price summary
    summary = predictor.get_price_summary(data)
    print(f"\n3. Price Statistics:")
    print(f"   Latest: ${summary['current']:,.2f}")
    print(f"   Average: ${summary['mean']:,.2f}")
    print(f"   Range: ${summary['min']:,.2f} - ${summary['max']:,.2f}")
    print(f"   Change: {summary['change_pct']:+.2f}%")
    print(f"   Volatility: {summary['volatility']:.2f}%")
    
    # Market info
    info = predictor.get_crypto_info()
    print(f"\n4. Market Info:")
    print(f"   Current Price: ${info['current_price']:,.2f}")
    print(f"   Market Cap: ${info['market_cap']:,.0f}")
    print(f"   24h Volume: {info['volume_24h']:,.0f}")
    
    # Preprocess data
    print("\n5. Preprocessing data...")
    X, y = predictor.preprocess_data(data, target_column='close')
    print(f"✓ Created {len(X)} sequences")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    print("\n" + "=" * 70)
    print("ETHEREUM PREDICTOR READY!")
    print("=" * 70)