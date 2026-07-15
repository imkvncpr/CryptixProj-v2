"""
Portfolio Manager - Track crypto holdings and performance
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class Position:
    """Represents a single crypto position"""
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        entry_date: datetime = None
    ):
        """Initialize position"""
        self.symbol = symbol
        self.quantity = quantity  # FIXED: was quatity
        self.entry_price = entry_price
        self.entry_date = entry_date or datetime.now(timezone.utc)
        self.current_price = entry_price
    
    def update_price(self, current_price: float):
        """Update current market price"""
        self.current_price = current_price
    
    @property
    def entry_value(self) -> float:
        """Total value at entry"""
        return self.quantity * self.entry_price
    
    @property
    def current_value(self) -> float:
        """Current total value"""
        return self.quantity * self.current_price  # FIXED: was entry_price
    
    @property
    def pnl(self) -> float:
        """Profit/Loss in dollars"""
        return self.current_value - self.entry_value  # FIXED: was entry_price
    
    @property
    def pnl_percent(self) -> float:
        """Profit/Loss percentage"""
        if self.entry_value == 0:
            return 0.0
        return (self.pnl / self.entry_value) * 100
    
    def __repr__(self) -> str:
        return (
            f"Position({self.symbol}: {self.quantity} @ "
            f"${self.entry_price:.2f}, P&L: ${self.pnl:.2f} "
            f"({self.pnl_percent:.2f}%))"
        )


# FIXED: Moved to module level (was nested inside Position)
class PortfolioManager:
    """Manages multiple crypto positions"""
    
    def __init__(self, initial_cash: float = 10000.0):
        """Initialize portfolio"""
        self.initial_cash = initial_cash
        self.current_cash = initial_cash  # FIXED: was current_funding
        self.positions: Dict[str, Position] = {}
        
        logger.info(f"Portfolio initialized with ${initial_cash:,.2f}")
    
    def add_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float
    ) -> bool:
        """Add new position"""
        cost = quantity * entry_price
        
        if cost > self.current_cash:  # FIXED: now uses current_cash
            logger.error(
                f"Insufficient cash: need ${cost:,.2f}, "
                f"have ${self.current_cash:,.2f}"
            )
            raise ValueError(f"Insufficient cash for position")
        
        position = Position(symbol, quantity, entry_price)
        self.positions[symbol] = position
        self.current_cash -= cost
        
        logger.info(
            f"Added position: {quantity} {symbol} @ ${entry_price:.2f} "
            f"(cost: ${cost:,.2f})"
        )
        return True
    
    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close existing position"""
        if symbol not in self.positions:
            raise ValueError(f"No position in {symbol}")
        
        position = self.positions[symbol]
        exit_value = position.quantity * exit_price
        pnl = exit_value - position.entry_value
        
        self.current_cash += exit_value
        logger.info(
            f"Closed {symbol}: exit=${exit_price:.2f}, "
            f"P&L ${pnl:,.2f}"
        )
        
        del self.positions[symbol]
        return pnl
    
    def update_prices(self, prices: Dict[str, float]):  # FIXED: Dict[str, float] not Dict[str, Dict]
        """Update prices for all positions"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all positions"""
        return list(self.positions.values())
    
    # ADDED: Missing properties
    @property
    def total_position_value(self) -> float:
        """Total value of all positions"""
        return sum(pos.current_value for pos in self.positions.values())
    
    @property
    def portfolio_value(self) -> float:
        """Total portfolio value (positions + cash)"""
        return self.total_position_value + self.current_cash
    
    @property
    def total_pnl(self) -> float:
        """Total P&L across all positions"""
        return sum(pos.pnl for pos in self.positions.values())
    
    @property
    def total_pnl_percent(self) -> float:
        """Total P&L percentage"""
        if self.initial_cash == 0:
            return 0.0
        return (self.total_pnl / self.initial_cash) * 100
    
    @property
    def cash_utilization(self) -> float:
        """Percentage of capital deployed"""
        if self.initial_cash == 0:
            return 0.0
        invested = self.initial_cash - self.current_cash
        return (invested / self.initial_cash) * 100
    
    # ADDED: Missing methods
    def get_allocation(self) -> Dict[str, float]:
        """Get portfolio allocation percentages"""
        total = self.portfolio_value
        if total == 0:
            return {}
        
        allocation = {}
        for symbol, position in self.positions.items():
            allocation[symbol] = (position.current_value / total) * 100
        
        allocation['cash'] = (self.current_cash / total) * 100
        return allocation
    
    def get_summary(self) -> Dict:
        """Get complete portfolio summary"""
        return {
            'portfolio_value': round(self.portfolio_value, 2),
            'initial_cash': round(self.initial_cash, 2),
            'current_cash': round(self.current_cash, 2),
            'total_invested': round(self.initial_cash - self.current_cash, 2),
            'total_pnl': round(self.total_pnl, 2),
            'total_pnl_percent': round(self.total_pnl_percent, 2),
            'cash_utilization': round(self.cash_utilization, 2),
            'num_positions': len(self.positions),
            'allocation': {
                k: round(v, 2) for k, v in self.get_allocation().items()
            },
            'positions': {
                symbol: {
                    'quantity': round(pos.quantity, 8),
                    'entry_price': round(pos.entry_price, 2),
                    'current_price': round(pos.current_price, 2),
                    'entry_value': round(pos.entry_value, 2),
                    'current_value': round(pos.current_value, 2),
                    'pnl': round(pos.pnl, 2),
                    'pnl_percent': round(pos.pnl_percent, 2)
                }
                for symbol, pos in self.positions.items()
            }
        }


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PORTFOLIO MANAGER")
    print("=" * 60)
    
    portfolio = PortfolioManager(initial_cash=50000)  # $50k
    
    print("\n1. Adding positions...")
    portfolio.add_position('bitcoin', 0.1, 45000)     # $4,500
    portfolio.add_position('ethereum', 5, 2500)       # $12,500
    print(f"   Cash remaining: ${portfolio.current_cash:,.2f}")
    
    print("\n2. Updating prices...")
    portfolio.update_prices({'bitcoin': 46000, 'ethereum': 2600})
    
    print("\n3. Portfolio summary:")
    summary = portfolio.get_summary()
    print(f"   Portfolio value: ${summary['portfolio_value']:,.2f}")
    print(f"   Total P&L: ${summary['total_pnl']:,.2f} ({summary['total_pnl_percent']:.2f}%)")
    
    print("\n4. Closing Bitcoin position...")
    pnl = portfolio.close_position('bitcoin', 47000)
    print(f"   P&L from close: ${pnl:,.2f}")
    
    print("\n" + "=" * 60)
    print("✅ PORTFOLIO MANAGER TEST COMPLETE!")
    print("=" * 60)
    
