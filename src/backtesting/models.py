from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# CANDLE CLASS
# ═══════════════════════════════════════════════════════════

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    
    def __post_init__(self):
        """Validate candle data after initialization"""
        if not self.timestamp:
            raise ValueError("Timestamp cannot be None")
        
        if not self.symbol or len(self.symbol) == 0:
            raise ValueError("Symbol cannot be empty")
        
        # Validate prices are positive
        prices = [self.open, self.high, self.low, self.close]
        if any(p <= 0 for p in prices):
            raise ValueError(f"All prices must be positive. Got: O={self.open}, H={self.high}, L={self.low}, C={self.close}")
        
        # Validate OHLC logic: High >= all prices, Low <= all prices
        if self.high < max(self.open, self.close):
            raise ValueError(f"High (${self.high}) must be >= Open (${self.open}) and Close (${self.close})")
        
        if self.low > min(self.open, self.close):
            raise ValueError(f"Low (${self.low}) must be <= Open (${self.open}) and Close (${self.close})")
        
        # Validate volume
        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative. Got: {self.volume}")
    
    # ─────────────────────────────────────────────────────────
    # PROPERTIES
    # ─────────────────────────────────────────────────────────
    
    @property
    def mid(self) -> float:
        """Midpoint between high and low"""
        return (self.high + self.low) / 2
    
    @property
    def hl_range(self) -> float:
        """High-Low range (volatility indicator)"""
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        """True if close > open (green candle)"""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """True if close < open (red candle)"""
        return self.close < self.open
    
    @property
    def is_doji(self) -> bool:
        """True if open ≈ close (within 1% of HL range)"""
        return abs(self.close - self.open) < (self.hl_range * 0.01)
    
    @property
    def body_size(self) -> float:
        """Size of candle body (absolute difference between open and close)"""
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        """Distance from high to max(open, close)"""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        """Distance from low to min(open, close)"""
        return min(self.open, self.close) - self.low
    
    @property
    def wick_ratio(self) -> float:
        """Ratio of wicks to body (0 = no wicks, high = long wicks)"""
        if self.body_size == 0:
            return 0.0
        return (self.upper_wick + self.lower_wick) / self.body_size
    
    # ─────────────────────────────────────────────────────────
    # SPECIAL METHODS
    # ─────────────────────────────────────────────────────────
    
    def __repr__(self) -> str:
        """String representation of candle"""
        direction = "↑" if self.is_bullish else "↓"
        return (
            f"{direction} {self.symbol} | {self.timestamp.strftime('%Y-%m-%d %H:%M')} | "
            f"O:{self.open:.2f} H:{self.high:.2f} L:{self.low:.2f} C:{self.close:.2f}"
        )
    
    def __lt__(self, other: 'Candle') -> bool:
        """Compare candles by timestamp (for sorting)"""
        return self.timestamp < other.timestamp
    
    def __eq__(self, other: 'Candle') -> bool:
        """Check equality by timestamp"""
        return self.timestamp == other.timestamp
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'symbol': self.symbol
        }


# ═══════════════════════════════════════════════════════════
# BACKTEST TRADE CLASS
# ═══════════════════════════════════════════════════════════

@dataclass
class BacktestTrade:
    trade_id: int
    symbol: str
    entry_date: datetime
    entry_price: float
    entry_quantity: float
    exit_date: datetime
    exit_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def __post_init__(self):
        """Validate trade data after initialization"""
        # Validate dates
        if self.exit_date <= self.entry_date:
            raise ValueError(f"Exit date ({self.exit_date}) must be after entry date ({self.entry_date})")
        
        # Validate prices are positive
        if self.entry_price <= 0 or self.exit_price <= 0:
            raise ValueError(f"Prices must be positive. Entry: {self.entry_price}, Exit: {self.exit_price}")
        
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError(f"Stop loss must be positive. Got: {self.stop_loss}")
        
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError(f"Take profit must be positive. Got: {self.take_profit}")
        
        # Validate quantity
        if self.entry_quantity <= 0:
            raise ValueError(f"Quantity must be positive. Got: {self.entry_quantity}")
        
        # Validate trade ID
        if self.trade_id <= 0:
            raise ValueError(f"Trade ID must be positive. Got: {self.trade_id}")
        
        # Validate symbol
        if not self.symbol or len(self.symbol) == 0:
            raise ValueError("Symbol cannot be empty")
        
        logger.debug(f"Trade #{self.trade_id} initialized: {self.symbol} @ ${self.entry_price}")
    
    # ─────────────────────────────────────────────────────────
    # VALUE PROPERTIES
    # ─────────────────────────────────────────────────────────
    
    @property
    def entry_value(self) -> float:
        """Total cost of entry (price × quantity)"""
        return self.entry_price * self.entry_quantity
    
    @property
    def exit_value(self) -> float:
        """Total proceeds from exit (price × quantity)"""
        return self.exit_price * self.entry_quantity
    
    @property
    def gross_pnl(self) -> float:
        """Profit/Loss in dollars (before commissions)"""
        return self.exit_value - self.entry_value
    
    @property
    def pnl_percent(self) -> float:
        """Profit/Loss as percentage"""
        if self.entry_price == 0:
            return 0.0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100
    
    # ─────────────────────────────────────────────────────────
    # OUTCOME PROPERTIES
    # ─────────────────────────────────────────────────────────
    
    @property
    def is_winning_trade(self) -> bool:
        """True if trade made profit"""
        return self.gross_pnl > 0
    
    @property
    def is_losing_trade(self) -> bool:
        """True if trade made loss"""
        return self.gross_pnl < 0
    
    @property
    def is_breakeven(self) -> bool:
        """True if trade broke even"""
        return self.gross_pnl == 0
    
    @property
    def was_stopped_out(self) -> bool:
        """True if exit price touched stop loss"""
        if self.stop_loss is None:
            return False
        return self.exit_price <= self.stop_loss
    
    @property
    def hit_take_profit(self) -> bool:
        """True if exit price reached take profit"""
        if self.take_profit is None:
            return False
        return self.exit_price >= self.take_profit
    
    # ─────────────────────────────────────────────────────────
    # DURATION PROPERTIES
    # ─────────────────────────────────────────────────────────
    
    @property
    def duration(self) -> timedelta:
        """Time held (as timedelta)"""
        return self.exit_date - self.entry_date
    
    @property
    def duration_days(self) -> int:
        """Time held in days"""
        return self.duration.days
    
    @property
    def duration_hours(self) -> float:
        """Time held in hours"""
        return self.duration.total_seconds() / 3600
    
    # ─────────────────────────────────────────────────────────
    # RISK/REWARD PROPERTIES
    # ─────────────────────────────────────────────────────────
    
    @property
    def risk_distance(self) -> float:
        """Distance from entry to stop loss"""
        if self.stop_loss is None:
            return 0.0
        return abs(self.entry_price - self.stop_loss)
    
    @property
    def reward_distance(self) -> float:
        """Distance from entry to take profit"""
        if self.take_profit is None:
            return 0.0
        return abs(self.take_profit - self.entry_price)
    
    @property
    def risk_reward_ratio(self) -> float:
        """Risk/Reward ratio (higher is better)"""
        if self.risk_distance == 0:
            return 0.0
        return self.reward_distance / self.risk_distance
    
    @property
    def actual_risk_taken(self) -> float:
        """Actual risk taken (difference from entry to exit)"""
        return abs(self.entry_price - self.exit_price)
    
    # ─────────────────────────────────────────────────────────
    # SPECIAL METHODS
    # ─────────────────────────────────────────────────────────
    
    def __repr__(self) -> str:
        """String representation of trade"""
        direction = "BUY" if self.entry_price < self.exit_price else "SELL"
        pnl_str = f"+${abs(self.gross_pnl):.2f}" if self.is_winning_trade else f"-${abs(self.gross_pnl):.2f}"
        
        return (
            f"Trade #{self.trade_id} | {self.symbol} {direction} | "
            f"Entry: ${self.entry_price:.2f} | Exit: ${self.exit_price:.2f} | "
            f"P&L: {pnl_str} ({self.pnl_percent:+.2f}%) | Duration: {self.duration_days}d"
        )
    
    def to_dict(self) -> dict:
        """Convert trade to dictionary (useful for CSV export)"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'entry_date': self.entry_date.isoformat(),
            'entry_price': round(self.entry_price, 2),
            'entry_quantity': round(self.entry_quantity, 8),
            'entry_value': round(self.entry_value, 2),
            'exit_date': self.exit_date.isoformat(),
            'exit_price': round(self.exit_price, 2),
            'exit_value': round(self.exit_value, 2),
            'gross_pnl': round(self.gross_pnl, 2),
            'pnl_percent': round(self.pnl_percent, 2),
            'duration_days': self.duration_days,
            'stop_loss': round(self.stop_loss, 2) if self.stop_loss else None,
            'take_profit': round(self.take_profit, 2) if self.take_profit else None,
            'was_stopped_out': self.was_stopped_out,
            'hit_take_profit': self.hit_take_profit,
            'risk_reward_ratio': round(self.risk_reward_ratio, 2)
        }


# ═══════════════════════════════════════════════════════════
# BACKTEST RESULTS CLASS
# ═══════════════════════════════════════════════════════════

@dataclass
class BacktestResults:
    trades: List[BacktestTrade]
    portfolio_values: Dict[datetime, float]
    daily_returns: Dict[datetime, float]
    metrics: Dict[str, float]
    
    def __post_init__(self):
        """Validate results data after initialization"""
        if not isinstance(self.trades, list):
            raise TypeError("trades must be a list")
        
        if not isinstance(self.portfolio_values, dict):
            raise TypeError("portfolio_values must be a dict")
        
        if not isinstance(self.daily_returns, dict):
            raise TypeError("daily_returns must be a dict")
        
        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dict")
        
        logger.debug(f"BacktestResults created with {len(self.trades)} trades")
    
    # ─────────────────────────────────────────────────────────
    # COUNT PROPERTIES
    # ─────────────────────────────────────────────────────────
    
    @property
    def total_trades(self) -> int:
        """Total number of trades"""
        return len(self.trades)
    
    @property
    def winning_trades(self) -> int:
        """Number of winning trades"""
        return sum(1 for t in self.trades if t.is_winning_trade)
    
    @property
    def losing_trades(self) -> int:
        """Number of losing trades"""
        return sum(1 for t in self.trades if t.is_losing_trade)
    
    @property
    def breakeven_trades(self) -> int:
        """Number of breakeven trades"""
        return sum(1 for t in self.trades if t.is_breakeven)
    
    # ─────────────────────────────────────────────────────────
    # P&L PROPERTIES
    # ─────────────────────────────────────────────────────────
    
    @property
    def total_pnl(self) -> float:
        """Total profit/loss in dollars"""
        return sum(t.gross_pnl for t in self.trades)
    
    @property
    def total_wins(self) -> float:
        """Sum of all winning trades"""
        return sum(t.gross_pnl for t in self.trades if t.is_winning_trade)
    
    @property
    def total_losses(self) -> float:
        """Sum of all losing trades (as positive)"""
        return abs(sum(t.gross_pnl for t in self.trades if t.is_losing_trade))
    
    # ─────────────────────────────────────────────────────────
    # RATE & RATIO PROPERTIES
    # ─────────────────────────────────────────────────────────
    
    @property
    def win_rate(self) -> float:
        """Percentage of winning trades"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def loss_rate(self) -> float:
        """Percentage of losing trades"""
        if self.total_trades == 0:
            return 0.0
        return (self.losing_trades / self.total_trades) * 100
    
    @property
    def average_win(self) -> float:
        """Average size of winning trade"""
        if self.winning_trades == 0:
            return 0.0
        return self.total_wins / self.winning_trades
    
    @property
    def average_loss(self) -> float:
        """Average size of losing trade (as positive)"""
        if self.losing_trades == 0:
            return 0.0
        return self.total_losses / self.losing_trades
    
    @property
    def profit_factor(self) -> float:
        """Ratio of wins to losses (higher is better, >1.0 is profitable)"""
        if self.total_losses == 0:
            return 0.0 if self.total_wins == 0 else float('inf')
        return self.total_wins / self.total_losses
    
    @property
    def payoff_ratio(self) -> float:
        """Ratio of average win to average loss"""
        if self.average_loss == 0:
            return 0.0
        return self.average_win / self.average_loss
    
    # ─────────────────────────────────────────────────────────
    # SUMMARY METHODS
    # ─────────────────────────────────────────────────────────
    
    def get_summary(self) -> dict:
        return {
            # Trade Counts
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'breakeven_trades': self.breakeven_trades,
            
            # P&L
            'total_pnl': round(self.total_pnl, 2),
            'total_wins': round(self.total_wins, 2),
            'total_losses': round(self.total_losses, 2),
            
            # Rates
            'win_rate': round(self.win_rate, 2),
            'loss_rate': round(self.loss_rate, 2),
            
            # Averages
            'average_win': round(self.average_win, 2),
            'average_loss': round(self.average_loss, 2),
            
            # Ratios
            'profit_factor': round(self.profit_factor, 2),
            'payoff_ratio': round(self.payoff_ratio, 2),
            
            # From metrics dict
            'total_return_pct': self.metrics.get('total_return_pct', 0.0),
            'sharpe_ratio': self.metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': self.metrics.get('max_drawdown', 0.0),
            'cagr': self.metrics.get('cagr', 0.0)
        }
    
    def get_detailed_report(self) -> str:
        """
        Get formatted text report of results
        """
        summary = self.get_summary()
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║                   BACKTEST RESULTS                       ║
╚══════════════════════════════════════════════════════════╝

📊 TRADE SUMMARY
├─ Total Trades:        {summary['total_trades']}
├─ Winning Trades:      {summary['winning_trades']}
├─ Losing Trades:       {summary['losing_trades']}
└─ Breakeven Trades:    {summary['breakeven_trades']}

💰 PROFITABILITY
├─ Total P&L:           ${summary['total_pnl']:,.2f}
├─ Total Wins:          ${summary['total_wins']:,.2f}
├─ Total Losses:        ${summary['total_losses']:,.2f}
├─ Average Win:         ${summary['average_win']:,.2f}
└─ Average Loss:        ${summary['average_loss']:,.2f}

📈 METRICS
├─ Win Rate:            {summary['win_rate']:.2f}%
├─ Profit Factor:       {summary['profit_factor']:.2f}
├─ Payoff Ratio:        {summary['payoff_ratio']:.2f}
├─ Total Return:        {summary['total_return_pct']:.2f}%
├─ Sharpe Ratio:        {summary['sharpe_ratio']:.2f}
└─ Max Drawdown:        {summary['max_drawdown']:.2f}%

═══════════════════════════════════════════════════════════
        """
        return report
    
    def __repr__(self) -> str:
        """String representation of results"""
        return (
            f"BacktestResults | {self.total_trades} trades | "
            f"{self.win_rate:.1f}% win rate | "
            f"P&L: ${self.total_pnl:,.2f} | "
            f"Sharpe: {self.metrics.get('sharpe_ratio', 0):.2f}"
        )


# ═══════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════

def validate_candle(candle: Candle) -> bool:
    try:
        if not isinstance(candle, Candle):
            raise TypeError(f"Expected Candle, got {type(candle)}")
        
        # __post_init__ already validates, but we can add extra checks here
        if candle.volume < 0:
            raise ValueError(f"Volume cannot be negative: {candle.volume}")
        
        return True
    except Exception as e:
        logger.error(f"Candle validation failed: {e}")
        raise


def validate_trade(trade: BacktestTrade) -> bool:
    try:
        if not isinstance(trade, BacktestTrade):
            raise TypeError(f"Expected BacktestTrade, got {type(trade)}")
        
        # __post_init__ already validates
        return True
    except Exception as e:
        logger.error(f"Trade validation failed: {e}")
        raise


def validate_results(results: BacktestResults) -> bool:
    try:
        if not isinstance(results, BacktestResults):
            raise TypeError(f"Expected BacktestResults, got {type(results)}")
        
        # __post_init__ already validates
        return True
    except Exception as e:
        logger.error(f"Results validation failed: {e}")
        raise