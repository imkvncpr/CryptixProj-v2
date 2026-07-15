from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np


class BaseStrategy(ABC):
    def __init__(self, name: str, symbol: str = "BTC/USD",
                 stop_loss_pct: float = 5.0, take_profit_pct: float = 10.0):
        self.name = name
        self.symbol = symbol
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trades = []
        self.signals = []
        
    @abstractmethod
    def calculate_signal(self, prices: np.ndarray, **kwargs):
        raise NotImplementedError("Subclasses must implement calculate_signal()")
    
    def validate_entry(self, signal) -> bool:
        if signal.strength < 50.0:
            return False
        if signal.direction == 0:
            return False
        return True
    
    def validate_exit(self, position_pnl_pct: float) -> bool:
        if position_pnl_pct < -self.stop_loss_pct:  # FIX: Added negative sign
            return True
        if position_pnl_pct > self.take_profit_pct:
            return True  # FIX: Was just "return" - added True
        return False
    
    def record_trade(self, entry_price: float, exit_price: float,  # FIX: Was record_table
                     quantity: float, direction: int) -> None:
        if direction == 1:
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
            
        trade = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'direction': direction,
            'pnl': pnl,
            'pnl_pct': (pnl / (entry_price * quantity) * 100) if entry_price > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self.trades.append(trade)
        
    def get_performance_metrics(self) -> Dict[str, Any]:  # FIX: Added space after ->
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        winning = [t for t in self.trades if t['pnl'] > 0]
        
        return{
            'total_trades': len(self.trades),
            'winning_trades': len(winning),
            'losing_trades': len(self.trades) - len(winning),
            'win_rate': (len(winning) / len(self.trades) * 100) if self.trades else 0.0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(self.trades) if self.trades else 0.0
        }
        
        
    def __repr__(self) -> str:
        metrics = self.get_performance_metrics()
        return f"{self.name}(trades={metrics['total_trades']}, win_rate={metrics['win_rate']:.1f}%)"