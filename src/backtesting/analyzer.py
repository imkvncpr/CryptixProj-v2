import logging
from typing import Dict, List, Optional
from datetime import datetime

from .models import BacktestResults, BacktestTrade
from . import metrics

logger = logging.getLogger(__name__)

class BacktestAnalyzer:
    def __init__(self, results: BacktestResults):
        self.results = results
        self.trades = results.trades
        self.portfolio_values = results.portfolio_values
        self.daily_pnl = results.daily_returns
        self.initial_metrics = results.metrics
        self._calculated_metrics = {}
        logger.info(f"BacktestAnalyzer initialized with {len(self.trades)} trades")
        
    def calculate_all_metrics(self)-> Dict[str, float]:
        logger.info("Calculating all metrics...")
        
        portfolio_values_list = list(self.portfolio_values.values())
        
        if not portfolio_values_list:
            logger.warning("No portfolio values available")
            return {}
        
        initial_value = portfolio_values_list[0] if portfolio_values_list else 0
        final_value = portfolio_values_list[-1] if portfolio_values_list else 0
        num_days = len(self.portfolio_values)
        
        total_return = metrics.calculate_total_return(initial_value, final_value)
        annual_return = metrics.calculate_annual_return(total_return, num_days)
        
        daily_returns_list = list(self.daily_pnl.values())
        
        self._calculated_metrics = {
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'sharpe_ratio': metrics.calculate_sharpe_ratio(daily_returns_list),
            'sortino_ratio': metrics.calculate_sortino_ratio(daily_returns_list),
            'calmar_ratio': metrics.calculate_calmar_ratio(annual_return, self.initial_metrics.get('max_drawdown', 1.0)),
            'recovery_factor': metrics.calculate_recovery_factor(final_value - initial_value, (initial_value * self.initial_metrics.get('max_drawdown', 1.0) / 100)),
            'total_trades': len(self.trades),
            'winning_trades': sum(1 for t in self.trades if t.is_winning_trade),
            'losing_trades': sum(1 for t in self.trades if t.is_losing_trade),
            'win_rate': self.initial_metrics.get('win_rate', 0.0),
            'profit_factor': self.initial_metrics.get('profit_factor', 0.0),
            'avg_win': metrics.calculate_avg_win(self.trades),
            'avg_loss': metrics.calculate_avg_loss(self.trades),
            'avg_trade_duration': metrics.calculate_avg_trade_duration(self.trades),
            'risk_reward_ratio': metrics.calculate_risk_reward_ratio(self.trades),
            'payoff_ratio': metrics.calculate_payoff_ratio(self.trades),
            'win_loss_ratio': metrics.calculate_win_loss_ratio(self.trades),
            'max_consecutive_wins': metrics.calculate_max_consecutive_wins(self.trades),
            'max_consecutive_losses': metrics.calculate_max_consecutive_losses(self.trades),
            'max_drawdown_pct': self.initial_metrics.get('max_drawdown', 0.0),
        }
        
        logger.info("Metrics calculation complete")
        return self._calculated_metrics
    
    def get_trade_summary(self)-> Dict:
        if not self.trades:
            return {'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0, 'breakeven_trades': 0, 'win_rate': 0.0, 
                    'largest_win': 0.0, 'largest_loss': 0.0, 'avg_duration_days': 0.0}
            
        winning = [t for t in self.trades if t.is_winning_trade]
        losing = [t for t in self.trades if t.is_losing_trade]
        
        largest_win = max([t.gross_pnl for t in winning], default=0.0)
        largest_loss = min([t.gross_pnl for t in losing], default=0.0)
    
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'breakeven_trades': sum(1 for t in self.trades if t.gross_pnl == 0),
            'win_rate': (len(winning) / len(self.trades) * 100) if self.trades else 0.0,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_duration_days': metrics.calculate_avg_trade_duration(self.trades)
        }
        
    def get_monthly_returns(self)-> Dict[str, float]:
        return metrics.calculate_monthly_returns(self.daily_pnl)
    
    def get_yearly_returns(self)-> Dict[str, float]:
        return metrics.calculate_yearly_returns(self.daily_pnl)
    
    def get_best_trade(self) -> Optional[BacktestTrade]:
        if not self.trades:
            return None
        return max(self.trades, key=lambda t: t.gross_pnl)
    
    def get_worst_trade(self)-> Optional[BacktestTrade]:
        if not self.trades:
            return None
        return min(self.trades, key=lambda t: t.gross_pnl)
    
    def get_average_loss(self)-> float:
        return metrics.calculate_avg_loss(self.trades)
    
    def get_risk_reward_ratio(self)-> float:
        return metrics.calculate_risk_reward_ratio(self.trades)

    def print_summary(self) -> None:
        summary = self.get_summary_text()
        print(summary)
        
    def get_summary_text(self)-> str:
        self.calculate_all_metrics()
        trade_summary = self.get_trade_summary()
        best_trade = self.get_best_trade()
        worst_trade = self.get_worst_trade()
        
        lines = []
        lines.append("=" * 70)
        lines.append("BACKTEST RESULTS SUMMARY".center(70))
        lines.append("=" * 70)
        lines.append("")
        
        lines.append("TRADE STATISTICS".upper())
        lines.append("─" * 70)
        lines.append(f"Total Trades:           {trade_summary['total_trades']}")
        lines.append(f"Winning Trades:         {trade_summary['winning_trades']} ({trade_summary['win_rate']:.1f}%)")
        lines.append(f"Losing Trades:          {trade_summary['losing_trades']}")
        lines.append(f"Breakeven Trades:       {trade_summary['breakeven_trades']}")
        lines.append(f"Average Duration:       {trade_summary['avg_duration_days']:.1f} days")
        lines.append("")
        
        lines.append("PERFORMANCE METRICS".upper())
        lines.append("─" * 70)
        lines.append(f"Total Return:           {self._calculated_metrics.get('total_return_pct', 0.0):.2f}%")
        lines.append(f"Annual Return:          {self._calculated_metrics.get('annual_return_pct', 0.0):.2f}%")
        lines.append(f"Sharpe Ratio:           {self._calculated_metrics.get('sharpe_ratio', 0.0):.2f}")
        lines.append(f"Sortino Ratio:          {self._calculated_metrics.get('sortino_ratio', 0.0):.2f}")
        lines.append(f"Calmar Ratio:           {self._calculated_metrics.get('calmar_ratio', 0.0):.2f}")
        lines.append(f"Max Drawdown:           {self._calculated_metrics.get('max_drawdown_pct', 0.0):.2f}%")
        lines.append("")
        
        lines.append("TRADE ANALYSIS".upper())
        lines.append("─" * 70)
        lines.append(f"Largest Win:            ${trade_summary['largest_win']:.2f}")
        lines.append(f"Largest Loss:           ${trade_summary['largest_loss']:.2f}")
        lines.append(f"Average Win:            ${self.get_average_win():.2f}")
        lines.append(f"Average Loss:           ${self.get_average_loss():.2f}")
        lines.append(f"Risk/Reward Ratio:      {self.get_risk_reward_ratio():.2f}")
        lines.append(f"Profit Factor:          {self._calculated_metrics.get('profit_factor', 0.0):.2f}")
        lines.append("")
        
        lines.append("STREAKS".upper())
        lines.append("─" * 70)
        lines.append(f"Longest Win Streak:     {self._calculated_metrics.get('max_consecutive_wins', 0)}")
        lines.append(f"Longest Loss Streak:    {self._calculated_metrics.get('max_consecutive_losses', 0)}")
        lines.append("")
        
        if best_trade:
            lines.append("BEST TRADE".upper())
            lines.append("─" * 70)
            lines.append(f"Symbol:                 {best_trade.symbol}")
            lines.append(f"Entry:                  ${best_trade.entry_price:.2f}")
            lines.append(f"Exit:                   ${best_trade.exit_price:.2f}")
            lines.append(f"Profit:                 ${best_trade.gross_pnl:.2f} ({best_trade.pnl_percent:.2f}%)")
            lines.append("")
        
        if worst_trade:
            lines.append("WORST TRADE".upper())
            lines.append("─" * 70)
            lines.append(f"Symbol:                 {worst_trade.symbol}")
            lines.append(f"Entry:                  ${worst_trade.entry_price:.2f}")
            lines.append(f"Exit:                   ${worst_trade.exit_price:.2f}")
            lines.append(f"Loss:                   ${worst_trade.gross_pnl:.2f} ({worst_trade.pnl_percent:.2f}%)")
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def get_detailed_report(self) -> str:
        """Get full detailed report with all trades"""
        lines = []
        lines.append(self.get_summary_text())
        lines.append("")
        lines.append("ALL TRADES".upper())
        lines.append("─" * 70)
        lines.append(f"{'ID':<4} {'Symbol':<10} {'Entry Date':<12} {'Entry Price':<12} {'Exit Price':<12} {'PnL':<12} {'%Return':<10}")
        lines.append("─" * 70)
        
        for trade in self.trades:
            entry_date = trade.entry_date.strftime('%Y-%m-%d')
            lines.append(f"{trade.trade_id:<4} {trade.symbol:<10} {entry_date:<12} ${trade.entry_price:<11.2f} ${trade.exit_price:<11.2f} ${trade.gross_pnl:<11.2f} {trade.pnl_percent:>8.2f}%")
        
        lines.append("─" * 70)
        
        monthly_returns = self.get_monthly_returns()
        if monthly_returns:
            lines.append("")
            lines.append("MONTHLY RETURNS".upper())
            lines.append("─" * 70)
            lines.append(f"{'Month':<12} {'P&L':<15} {'Return %':<15}")
            lines.append("─" * 70)
            
            for month in sorted(monthly_returns.keys()):
                pnl = monthly_returns[month]
                lines.append(f"{month:<12} ${pnl:<14.2f}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation"""
        return f"BacktestAnalyzer | {len(self.trades)} trades | Win rate: {self.get_trade_summary()['win_rate']:.1f}%"
        
            
    
    def get_average_win(self) -> float:
        return metrics.calculate_avg_win(self.trades)
    
    def get_average_loss(self) -> float:
        return metrics.calculate_avg_loss(self.trades)
