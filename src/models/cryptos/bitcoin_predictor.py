"""Bitcoin Cryptocurrency Predictor"""

import sys
from pathlib import Path

# Add project root to path - MUST BE BEFORE OTHER IMPORTS
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import List
from src.models.cryptos.cryptocoin_predictor import CryptoCoinPredictor


class BitcoinPredictor(CryptoCoinPredictor):
    """
    Bitcoin price predictor using LSTM.
    
    Optimized for Bitcoin's high volatility:
    - 3-layer LSTM: [128, 64, 32] units
    - 60-day lookback period
    """
    
    def __init__(
        self,
        ticker: str = "BTC-USD",
        look_back: int = 60,
        units: List[int] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        model_version: str = "v1.0"
    ):
        if units is None:
            units = [128, 64, 32]
        
        super().__init__(
            ticker=ticker,
            name="Bitcoin",
            look_back=look_back,
            units=units,
            dropout=dropout,
            learning_rate=learning_rate,
            model_version=model_version
        )


def main():
    """Test the Bitcoin predictor"""
    print("=" * 70)
    print("TESTING BITCOIN PREDICTOR")
    print("=" * 70)
    
    # Create predictor
    print("\n1. Creating Bitcoin predictor...")
    btc = BitcoinPredictor()
    print(f"   ✓ Created: {btc}")
    print(f"   ✓ Architecture: {btc.units}")
    print(f"   ✓ Look back: {btc.look_back} days")
    
    # Download data
    print("\n2. Downloading Bitcoin data...")
    data = btc.download_data(period="5d")
    
    if data is not None:
        print(f"   ✓ Downloaded {len(data)} days of data")
        print(f"   ✓ Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Get summary
        print("\n3. Price Statistics:")
        summary = btc.get_price_summary(data)
        print(f"   Current: ${summary['current']:,.2f}")
        print(f"   Average: ${summary['mean']:,.2f}")
        print(f"   Range: ${summary['min']:,.2f} - ${summary['max']:,.2f}")
        print(f"   Change: {summary['change_pct']:+.2f}%")
        print(f"   Volatility: {summary['volatility']:.2f}%")
        
        # Get market info
        print("\n4. Market Info:")
        info = btc.get_crypto_info()
        print(f"   Market Cap: ${info['market_cap']:,.0f}")
        print(f"   24h Volume: {info['volume_24h']:,.0f}")
        
        print("\n" + "=" * 70)
        print("✅ BITCOIN PREDICTOR TEST COMPLETE!")
        print("=" * 70)
    else:
        print("   ✗ Failed to download data")


if __name__ == "__main__":
    print("🚀 Script starting...")
    main()
    print("🏁 Script finished")