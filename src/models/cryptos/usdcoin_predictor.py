"""USD Coin (USDC) Stablecoin Predictor"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import List
from src.models.cryptos.cryptocoin_predictor import CryptoCoinPredictor


class USDCoinPredictor(CryptoCoinPredictor):
    """
    USD Coin stablecoin predictor using LSTM.
    
    Optimized for regulated stablecoin (pegged to $1):
    - 2-layer LSTM: [50, 25] units
    - 30-day lookback
    """
    
    def __init__(
        self,
        ticker: str = "USDC-USD",
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
            name="USD Coin",
            look_back=look_back,
            units=units,
            dropout=dropout,
            learning_rate=learning_rate,
            model_version=model_version
        )


def main():
    """Test the USD Coin predictor"""
    print("=" * 70)
    print("TESTING USD COIN (USDC) PREDICTOR")
    print("=" * 70)
    
    print("\n1. Creating USD Coin predictor...")
    usdc = USDCoinPredictor()
    print(f"   ✓ Created: {usdc}")
    print(f"   ✓ Architecture: {usdc.units}")
    
    print("\n2. Downloading USDC data...")
    data = usdc.download_data(period="5d")
    
    if data is not None:
        print(f"   ✓ Downloaded {len(data)} days")
        
        summary = usdc.get_price_summary(data)
        print("\n3. Price Statistics:")
        print(f"   Current: ${summary['current']:.6f}")
        print(f"   Deviation from $1: ${abs(summary['current'] - 1.0):.6f}")
        print(f"   Volatility: {summary['volatility']:.4f}%")
        
        print("\n" + "=" * 70)
        print("✅ USDC PREDICTOR TEST COMPLETE!")
        print("=" * 70)
    else:
        print("   ✗ Failed to download data")


if __name__ == "__main__":
    print("🚀 Script starting...")
    main()
    print("🏁 Script finished")