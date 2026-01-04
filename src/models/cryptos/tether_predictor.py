"""Tether (USDT) Stablecoin Predictor"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import List
from src.models.cryptos.cryptocoin_predictor import CryptoCoinPredictor


class TetherPredictor(CryptoCoinPredictor):
    """
    Tether stablecoin predictor using LSTM.
    
    Optimized for stablecoin (pegged to $1):
    - 2-layer LSTM: [50, 25] units (simple)
    - 30-day lookback (shorter period)
    - Lower dropout: 0.15
    - Higher learning rate: 0.002
    """
    
    def __init__(
        self,
        ticker: str = "USDT-USD",
        look_back: int = 30,
        units: List[int] = None,
        dropout: float = 0.15,
        learning_rate: float = 0.002,
        model_version: str = "v1.0"
    ):
        if units is None:
            units = [50, 25]
        
        super().__init__(
            ticker=ticker,
            name="Tether",
            look_back=look_back,
            units=units,
            dropout=dropout,
            learning_rate=learning_rate,
            model_version=model_version
        )


def main():
    """Test the Tether predictor"""
    print("=" * 70)
    print("TESTING TETHER (USDT) PREDICTOR")
    print("=" * 70)
    
    print("\n1. Creating Tether predictor...")
    usdt = TetherPredictor()
    print(f"   ✓ Created: {usdt}")
    print(f"   ✓ Architecture: {usdt.units} (simple for stablecoin)")
    print(f"   ✓ Look back: {usdt.look_back} days (shorter)")
    
    print("\n2. Downloading Tether data...")
    data = usdt.download_data(period="5d")
    
    if data is not None:
        print(f"   ✓ Downloaded {len(data)} days of data")
        print(f"   ✓ Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        print("\n3. Price Statistics (Stablecoin - should be near $1):")
        summary = usdt.get_price_summary(data)
        print(f"   Current: ${summary['current']:.6f}")
        print(f"   Average: ${summary['mean']:.6f}")
        print(f"   Range: ${summary['min']:.6f} - ${summary['max']:.6f}")
        print(f"   Deviation from $1: ${abs(summary['current'] - 1.0):.6f}")
        print(f"   Volatility: {summary['volatility']:.4f}% (should be very low)")
        
        print("\n4. Market Info:")
        info = usdt.get_crypto_info()
        print(f"   Market Cap: ${info['market_cap']:,.0f}")
        print(f"   24h Volume: {info['volume_24h']:,.0f}")
        
        print("\n" + "=" * 70)
        print("✅ TETHER PREDICTOR TEST COMPLETE!")
        print("=" * 70)
    else:
        print("   ✗ Failed to download data")


if __name__ == "__main__":
    print("🚀 Script starting...")
    main()
    print("🏁 Script finished")