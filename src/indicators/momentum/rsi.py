import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from typing import Dict, Any

from src.indicators.base_indicator import BaseIndicator

class RSI(BaseIndicator):
    def __init__(self, period: int = 14):
        super().__init__(name = 'RSI')
        self.period = period
        
    def calculate(self, data: pd.DataFrame)-> pd.Series:
        self.validate_data(data)
        
        close = data['close']
        
        delta = close.diff()
        
        gain = delta.where(delta > 0, 0)
        
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window = self.period).mean()
        
        avg_loss = loss.rolling(window = self.period).mean()
        
        rs = avg_gain / avg_loss
        
        rsi = 100 - (100 / (1 + rs))
        
        self.values = rsi
        
        return rsi
    
    def interpret(self, current_value: float)-> Dict[str, Any]:
        if current_value > 70:
            signal = "SELL"
            
            strength = "STRONG" if current_value > 80 else "MODERATE"
            
            confidence = 85 if current_value > 80 else 70
            
            reasoning = f"RSI at {current_value:.1f} - Overbought Condition"
            
        elif current_value < 30:
            signal = "BUY"
            
            strength = "STRONG" if current_value < 20 else "MODERATE"
            
            confidence = 85 if current_value < 20 else 70
            
            reasoning = f"RSI at {current_value: .1f} - Oversold Condition"
            
        else:
            signal = "HOLD"
            
            strength = "NEUTRAL"
            
            confidence = 50
            
            reasoning = f"RSI at {current_value: .1f} - Neutral Zone"
            
        return{
            'indicator': 'RSI',
            'value': round(current_value, 2),
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'reasoning' : reasoning,
            }
        
if __name__ == "__main__":
# ↑ Special Python idiom - only run code when file executed directly
# __name__ = special variable automatically set by Python
# Two cases:
#   1. File run directly: python rsi.py
#      → Python sets __name__ = "__main__"
#      → Condition is True, code runs
#   2. File imported: from rsi import RSI
#      → Python sets __name__ = "src.indicators.momentum.rsi"
#      → Condition is False, code skipped
# Purpose: Allow file to be both:
#   - Runnable script (with tests)
#   - Importable module (without running tests)
# Common pattern in Python for test code
# Without this: Importing would run all tests (bad!)

    print("=" * 60)
    # ↑ Print 60 equal signs as separator line
    # "=" * 60 = string repetition
    #   - Takes string "=" and repeats it 60 times
    #   - Result: "============...========" (60 chars)
    # print() = built-in function that outputs to console
    # Purpose: Visual separator for readability
    # Output:
    # ============================================================
    
    print("TESTING RSI INDICATOR")
    # ↑ Print header text for test section
    # Simple string output
    # Tells user what's being tested
    # Makes output clear when running multiple tests
    
    print("=" * 60)
    # ↑ Another separator line (matches the top one)
    # Creates a "box" around the header:
    # ============================================================
    # TESTING RSI INDICATOR
    # ============================================================
    # Professional-looking output formatting
    
    from src.models.cryptos.bitcoin_predictor import BitcoinPredictor
    # ↑ Import BitcoinPredictor class
    # This is INSIDE the if block, so only imports when testing
    # Why here and not at top?
    #   - Only needed for testing, not for RSI class itself
    #   - Keeps imports clean when RSI is used elsewhere
    # BitcoinPredictor = class from Sprint 1
    # Located at: src/models/cryptos/bitcoin_predictor.py
    # Purpose: Download real Bitcoin data from Yahoo Finance
    # Used for: Testing RSI with real market data
    
    # Download Bitcoin data
    print("\n1. Downloading Bitcoin data...")
    # ↑ Print status message for step 1
    # \n = newline character (blank line before message)
    # Creates spacing between sections
    # Output:
    # 
    # 1. Downloading Bitcoin data...
    # Numbered steps make output easy to follow
    # "..." indicates action in progress
    
    btc = BitcoinPredictor()
    # ↑ Create instance of BitcoinPredictor class
    # Syntax: ClassName() = call constructor (__init__)
    # btc = variable holding the BitcoinPredictor object
    # No parameters needed (uses defaults)
    # After this line: btc object ready to use
    # Can call methods: btc.download_data(), etc.
    # Think of it as: btc = your Bitcoin data downloader
    
    data = btc.download_data(period="3mo")
    # ↑ Download 3 months of Bitcoin historical data
    # btc.download_data() = method call on btc object
    # period="3mo" = parameter (3 months of history)
    # Other options: "1mo", "6mo", "1y", "5y", etc.
    # What it does:
    #   1. Connects to Yahoo Finance API
    #   2. Downloads BTC-USD price data
    #   3. Returns pandas DataFrame with OHLCV columns
    # data = DataFrame with columns:
    #   - open: Opening price each day
    #   - high: Highest price each day
    #   - low: Lowest price each day
    #   - close: Closing price each day
    #   - volume: Trading volume each day
    # Rows: One row per day (about 90 rows for 3 months)
    # Example row: 
    #   Date: 2024-01-15
    #   open: 66500, high: 67200, low: 66300, 
    #   close: 66900, volume: 25000000
    
    print(f"   Data points: {len(data)}")
    # ↑ Print how many days of data we downloaded
    # f-string = f"..." with {variable} placeholders
    # len(data) = number of rows in DataFrame
    #   - len() = built-in Python function for length
    #   - data = DataFrame with ~90 rows
    #   - Returns: 90 (or similar number)
    # {len(data)} = placeholder that gets replaced with actual number
    # "   " = indentation (3 spaces) for nice formatting
    # Output example: "   Data points: 90"
    # Purpose: Verify data downloaded successfully
    # If shows 0 or error → download failed
    
    print(f"   Current price: ${data['close'].iloc[-1]:,.2f}")
    # ↑ Print the most recent Bitcoin closing price
    # Breaking down the f-string:
    #   data['close'] = extract 'close' column (Series of prices)
    #   .iloc[-1] = get last item using integer location
    #     - iloc = integer location indexer
    #     - -1 = last position (Python negative indexing)
    #     - Like: prices[last_index]
    #   :,.2f = format specifier
    #     - : = start formatting
    #     - , = use comma as thousands separator
    #     - .2 = 2 decimal places
    #     - f = float (decimal number)
    # Example: 66898.19234 → "66,898.19"
    # Output: "   Current price: $66,898.19"
    # Purpose: Show current Bitcoin price for context
    # Helps understand RSI in relation to actual price
    
    # Create and calculate RSI
    print("\n2. Calculating RSI(14)...")
    # ↑ Status message for step 2
    # \n = blank line before message
    # "RSI(14)" = RSI with 14-day period
    # Indicates what's happening next
    
    rsi = RSI(period=14)
    # ↑ Create RSI indicator instance
    # RSI = the class we defined above
    # period=14 = use 14-day period (Wilder's default)
    # Calls: __init__(self, period=14)
    # After this: rsi object ready to use
    # Has attributes: rsi.period = 14, rsi.name = "RSI"
    # Can call methods: rsi.calculate(), rsi.interpret()
    
    rsi_values = rsi.calculate(data)
    # ↑ Calculate RSI for all historical data
    # rsi.calculate(data) = call calculate method
    # Passes data (DataFrame) as parameter
    # What happens inside:
    #   1. Validates data has correct columns
    #   2. Extracts close prices
    #   3. Calculates price changes (delta)
    #   4. Separates gains and losses
    #   5. Calculates rolling averages
    #   6. Calculates RS and RSI
    #   7. Stores in rsi.values
    #   8. Returns RSI Series
    # rsi_values = pandas Series with RSI for each day
    # Length: Same as data (90 rows)
    # Example values: [NaN, NaN, ..., 45.2, 46.8, 45.5]
    #   - First ~14 values are NaN (not enough data yet)
    #   - Rest are RSI values (0-100)
    
    current_rsi = rsi.get_latest()
    # ↑ Get the most recent RSI value
    # rsi.get_latest() = method inherited from BaseIndicator
    # What it does: return self.values.iloc[-1]
    #   - self.values = the RSI Series we just calculated
    #   - .iloc[-1] = last value (most recent)
    # current_rsi = single float number (e.g., 45.67)
    # This is the RSI value for today/most recent day
    # Used for: Generating current signal
    
    print(f"   Current RSI: {current_rsi:.2f}")
    # ↑ Print the current RSI value
    # {current_rsi:.2f} = format to 2 decimal places
    # Example: 45.6723 → "45.67"
    # Output: "   Current RSI: 45.67"
    # Purpose: Show RSI value before interpreting
    # User can see: "Is it overbought (>70)? Oversold (<30)? Neutral?"
    
    # Interpret signal
    print("\n3. Interpreting RSI signal...")
    # ↑ Status message for step 3
    # Indicates: About to generate trading signal
    # "Interpreting" = converting number to actionable recommendation
    
    result = rsi.interpret(current_rsi)
    # ↑ Generate trading signal from current RSI value
    # rsi.interpret(current_rsi) = call interpret method we defined
    # Passes current_rsi (e.g., 45.67) as parameter
    # What happens inside:
    #   1. Checks if RSI > 70 (overbought)
    #   2. Checks if RSI < 30 (oversold)
    #   3. Otherwise, neutral
    #   4. Determines signal, strength, confidence
    #   5. Creates reasoning string
    #   6. Returns dictionary with all info
    # result = dictionary like:
    #   {
    #     'indicator': 'RSI',
    #     'value': 45.67,
    #     'signal': 'HOLD',
    #     'strength': 'NEUTRAL',
    #     'confidence': 50,
    #     'reasoning': 'RSI at 45.67 - Neutral zone'
    #   }
    # This is the final output - actionable trading signal
    
    print(f"   Signal: {result['signal']}")
    # ↑ Print the trading signal
    # result['signal'] = access 'signal' key from dictionary
    # Dictionary access: dict[key] returns value
    # Example: result['signal'] = "HOLD"
    # Output: "   Signal: HOLD"
    # This tells trader: What to do (BUY/SELL/HOLD)
    
    print(f"   Strength: {result['strength']}")
    # ↑ Print the signal strength
    # result['strength'] = "STRONG", "MODERATE", or "NEUTRAL"
    # Example: "NEUTRAL"
    # Output: "   Strength: NEUTRAL"
    # Tells trader: How strong is the signal?
    
    print(f"   Confidence: {result['confidence']}%")
    # ↑ Print confidence percentage
    # result['confidence'] = 85, 70, or 50 (integer)
    # Add % after number for clarity
    # Example: 50
    # Output: "   Confidence: 50%"
    # Tells trader: How confident are we?
    
    print(f"   Reasoning: {result['reasoning']}")
    # ↑ Print explanation of the signal
    # result['reasoning'] = descriptive string
    # Example: "RSI at 45.67 - Neutral zone"
    # Output: "   Reasoning: RSI at 45.67 - Neutral zone"
    # Tells trader: WHY this signal was generated
    # Most important for understanding and trust
    
    # Show recent RSI values
    print("\n4. Recent RSI values:")
    # ↑ Header for recent values section
    # Shows historical RSI values leading up to current
    # Purpose: See trend (rising? falling? stable?)
    
    for i in range(-5, 0):
    # ↑ Loop through last 5 RSI values
    # for = loop keyword in Python
    # i = loop variable (takes each value in sequence)
    # range(-5, 0) = sequence of numbers
    #   - Start: -5
    #   - Stop: 0 (exclusive, so stops at -1)
    #   - Values: [-5, -4, -3, -2, -1]
    # Why negative numbers?
    #   - Used with .iloc[i] to index from end
    #   - .iloc[-5] = 5th from last
    #   - .iloc[-1] = last (most recent)
    # Loop executes 5 times, once for each value
    
        print(f"   {i} days ago: {rsi_values.iloc[i]:.2f}")
        # ↑ Print RSI value from i days ago
        # This line runs 5 times (once per loop iteration)
        # Breaking down:
        #   {i} = the loop variable (-5, -4, -3, -2, -1)
        #   rsi_values = Series of all RSI values
        #   .iloc[i] = get value at position i
        #     - iloc[-5] = 5th from last
        #     - iloc[-1] = last
        #   :.2f = format to 2 decimals
        # Example outputs (5 lines):
        #   "   -5 days ago: 48.23"
        #   "   -4 days ago: 46.15"
        #   "   -3 days ago: 47.89"
        #   "   -2 days ago: 46.34"
        #   "   -1 days ago: 45.67"  ← This is current_rsi
        # Purpose: Show RSI trend over last 5 days
        # Trader can see: Is RSI rising, falling, or stable?
        # Pattern recognition: Divergences, trends
    
    print("\n" + "=" * 60)
    # ↑ Closing separator line
    # "\n" = blank line before separator
    # "=" * 60 = 60 equal signs
    # Creates visual closure for test output
    
    print("✅ RSI TEST COMPLETE!")
    # ↑ Success message with checkmark emoji
    # ✅ = Unicode checkmark symbol
    # Indicates: Test ran successfully, no errors
    # Visual confirmation everything worked
    
    print("=" * 60)
    # ↑ Final separator line
    # Completes the "box" around success message:
    # 
    # ============================================================
    # ✅ RSI TEST COMPLETE!
    # ============================================================
    # Professional, clean output formatting
