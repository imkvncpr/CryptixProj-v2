import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .models import Candle, BacktestTrade, BacktestResults
from .data_handler import HistoricalDataHandler

logger = logging.getLogger(__name__)

class Backtest:
    def __init__(self,
                 initial_capital: float,
                 data_handler: HistoricalDataHandler,
                 start_date: datetime,
                 end_date: datetime,
                 commission: float = 0.001):
        
        self.initial_capital = initial_capital
        self.data_handler = data_handler
        self.start_date = start_date
        self.end_date = end_date
        self.commission = commission
        
        self.current_cash = initial_capital
        self.positions: Dict[str, float] = {}
        
        self.closed_trades: List[BacktestTrade] = []
        self.trade_id_counter = 0
        
        self.portfolio_values: Dict[datetime, float] = {}
        self.daily_pnl: Dict[datetime, float] = {}
        
        # Risk management
        self.max_risk_per_trade = 0.02  # 2%
        self.max_position_size = 0.30   # 30%
        
        logger.info(
            f"Backtest initialized: "
            f"Capital=${initial_capital:,.2f}, "
            f"Period: {start_date.date()} to {end_date.date()}"
        )
    
    # ─────────────────────────────────────────────────────────
    # TRADE MANAGEMENT
    # ─────────────────────────────────────────────────────────
    
    def add_trade(self, 
                  symbol: str,
                  entry_date: datetime,
                  entry_price: float,
                  entry_quantity: float,
                  exit_date: datetime,
                  exit_price: float,
                  stop_loss: Optional[float] = None,
                  take_profit: Optional[float] = None) -> bool:
        
        if entry_quantity <= 0:
            logger.warning(f"Invalid quantity {entry_quantity}, skipping trade")
            return False
        
        if exit_date <= entry_date:
            logger.warning(f"Invalid dates: exit before entry, skipping trade")
            return False
        
        entry_cost = entry_price * entry_quantity    
        commission_cost = entry_cost * self.commission
        total_cost = entry_cost + commission_cost
        
        if total_cost > self.current_cash:
            logger.warning(
                f"Insufficient cash: need ${total_cost:,.2f}, "
                f"have ${self.current_cash:,.2f}, skipping trade"
            )
            return False
        
        self.current_cash -= total_cost
        
        self.trade_id_counter += 1
        trade = BacktestTrade(
            trade_id=self.trade_id_counter,
            symbol=symbol,
            entry_date=entry_date,
            entry_price=entry_price,
            entry_quantity=entry_quantity,
            exit_date=exit_date,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit   
        )
        
        exit_proceeds = exit_price * entry_quantity
        exit_commission = exit_proceeds * self.commission
        net_proceeds = exit_proceeds - exit_commission
        
        self.current_cash += net_proceeds
        
        self.closed_trades.append(trade)
        
        logger.info(
            f"Trade #{trade.trade_id}: {symbol} | "
            f"Entry: ${entry_price:.2f} x {entry_quantity} | "
            f"Exit: ${exit_price:.2f} | "
            f"P&L: ${trade.gross_pnl:,.2f} ({trade.pnl_percent:+.2f}%)"
        )
        
        return True
    
    def run(self) -> BacktestResults:
        logger.info("Starting backtest run...")
        
        try:
            candles = self.data_handler.get_data(
                symbol='BTC/USD',
                start_date=self.start_date,
                end_date=self.end_date
            )
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
        
        if not candles:
            logger.error("No candle data found for backtest period")
            raise ValueError("No candle data available")
        
        logger.info(f"Loaded {len(candles)} candles for backtest")
        
        for candle in candles:
            signal = self._generate_signal(candle)
            
            if signal:
                self._execute_signal(signal, candle)
            
            portfolio_value = self._calculate_portfolio_value(candle)
            self.portfolio_values[candle.timestamp] = portfolio_value
            
            daily_pnl = portfolio_value - self.initial_capital
            self.daily_pnl[candle.timestamp] = daily_pnl
        
        # Calculate results AFTER loop completes
        results = self._calculate_results()
        
        logger.info("Backtest run complete")
        
        return results
    
    def _generate_signal(self, candle: Candle) -> Optional[Dict]:
        import random
        
        if random.random() > 0.98:  # FIXED: Added parentheses ()
            return {
                'type': 'BUY',
                'entry_price': candle.close,
                'exit_price': candle.close * 1.05,
                'duration': 5,
                'stop_loss': candle.close * 0.98,
                'take_profit': candle.close * 1.10
            }
        return None
    
    def _execute_signal(self, signal: Dict, candle: Candle) -> None:
        if signal['type'] != 'BUY':
            return
        
        position_size = self._calculate_position_size(signal['entry_price'])
        
        self.add_trade(
            symbol=candle.symbol,
            entry_date=candle.timestamp,
            entry_price=signal['entry_price'], 
            entry_quantity=position_size,
            exit_date=candle.timestamp + timedelta(days=signal.get('duration', 1)),
            exit_price=signal.get('exit_price', candle.close * 1.05),
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit')
        )
    
    def _calculate_position_size(self, current_price: float) -> float: 
        risk_amount = self.initial_capital * self.max_risk_per_trade
        stop_loss_pct = 0.02
        price_risk = current_price * stop_loss_pct
        position_size = risk_amount / price_risk
        
        max_position_value = self.initial_capital * self.max_position_size
        max_position = max_position_value / current_price
        
        position_size = min(position_size, max_position)
        
        return position_size
    
    def _calculate_portfolio_value(self, candle: Candle) -> float:
        value = self.current_cash
        
        for symbol, quantity in self.positions.items():
            value += quantity * candle.close
        
        return value
    
    def _calculate_results(self) -> BacktestResults:
        logger.info("Calculating backtest metrics...")
        
        metrics = {
            'total_return_pct': self.calculate_total_return(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.calculate_max_drawdown(),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor()
        }
        
        results = BacktestResults(
            trades=self.closed_trades,
            portfolio_values=self.portfolio_values,
            daily_returns=self.daily_pnl,
            metrics=metrics
        )
        
        return results
    
    # ─────────────────────────────────────────────────────────
    # METRIC CALCULATIONS
    # ─────────────────────────────────────────────────────────
    
    def calculate_total_return(self) -> float:
        final_value = self.current_cash
        return ((final_value - self.initial_capital) / self.initial_capital) * 100
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        if not self.daily_pnl:
            return 0.0
        
        returns = list(self.daily_pnl.values())
        
        if len(returns) < 2:
            return 0.0
        
        volatility = np.std(returns)
        if volatility == 0:
            return 0.0
        
        avg_return = np.mean(returns)
        
        sharpe = (avg_return - (risk_free_rate / 252)) / volatility
        
        return sharpe * np.sqrt(252)
    
    def calculate_max_drawdown(self) -> float:
        if not self.portfolio_values:
            return 0.0
        
        values = list(self.portfolio_values.values())
        max_dd = 0.0
        peak = values[0]
        
        for value in values:
            if value > peak:
                peak = value
            
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd * 100  # FIXED: Moved outside loop
    
    def calculate_win_rate(self) -> float:
        if len(self.closed_trades) == 0:
            return 0.0
        
        winning = sum(1 for t in self.closed_trades if t.is_winning_trade)
        return (winning / len(self.closed_trades)) * 100
    
    def calculate_profit_factor(self) -> float:
        wins = sum(t.gross_pnl for t in self.closed_trades if t.gross_pnl > 0)
        losses = abs(sum(t.gross_pnl for t in self.closed_trades if t.gross_pnl < 0))
        
        if losses == 0:
            return 0.0 if wins == 0 else float('inf')
        
        return wins / losses
    
    def plot_results(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")
            return
        
        if not self.portfolio_values:
            logger.warning("No portfolio values to plot")
            return
        
        dates = sorted(self.portfolio_values.keys())
        values = [self.portfolio_values[d] for d in dates]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curve
        ax1.plot(dates, values, linewidth=2, color='blue', label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Equity Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Daily returns
        daily_returns = [self.daily_pnl[d] for d in dates]
        colors = ['green' if v > 0 else 'red' for v in daily_returns]
        ax2.bar(dates, daily_returns, color=colors, alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Daily P&L ($)')
        ax2.set_xlabel('Date')
        ax2.set_title('Daily Profit/Loss')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_results_summary(self) -> str:
        if not self.closed_trades:
            return "No trades executed"
        
        results = self._calculate_results()
        summary = results.get_detailed_report()
        
        return summary
    
    def __repr__(self) -> str:
        return (
            f"Backtest | ${self.initial_capital} capital | "
            f"{len(self.closed_trades)} trades | "
            f"Current cash: ${self.current_cash}"
        )