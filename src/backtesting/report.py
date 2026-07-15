import logging
from pathlib import Path
from .analyzer import BacktestAnalyzer

logger = logging.getLogger(__name__)

class BacktestReport:
    """Generate backtest reports in multiple formats"""
    
    def __init__(self, analyzer: BacktestAnalyzer):
        """Initialize report generator"""
        self.analyzer = analyzer
        self.trades = analyzer.trades
        logger.info("BacktestReport initialized")
    
    def to_text(self) -> str:
        """Generate text report"""
        logger.info("Generating text report...")
        return self.analyzer.get_detailed_report()
    
    def to_html(self) -> str:
        """Generate HTML report"""
        logger.info("Generating HTML report...")
        metrics = self.analyzer._calculated_metrics
        trade_summary = self.analyzer.get_trade_summary()
        monthly_returns = self.analyzer.get_monthly_returns()
        
        html_lines = []
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html>")
        html_lines.append("<head>")
        html_lines.append("    <title>Backtest Report</title>")
        html_lines.append("    <meta charset='utf-8'>")
        html_lines.append("    <style>")
        html_lines.append("        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }")
        html_lines.append("        h1 { color: #333; text-align: center; }")
        html_lines.append("        h2 { color: #555; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; margin-top: 30px; }")
        html_lines.append("        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; background-color: white; }")
        html_lines.append("        th { background-color: #4CAF50; color: white; padding: 12px; text-align: left; }")
        html_lines.append("        td { padding: 10px; border-bottom: 1px solid #ddd; }")
        html_lines.append("        .positive { color: green; font-weight: bold; }")
        html_lines.append("        .negative { color: red; font-weight: bold; }")
        html_lines.append("    </style>")
        html_lines.append("</head>")
        html_lines.append("<body>")
        html_lines.append("    <h1>Backtest Analysis Report</h1>")
        html_lines.append("    <h2>Performance Metrics</h2>")
        html_lines.append("    <table>")
        html_lines.append("        <tr><th>Metric</th><th>Value</th></tr>")
        
        metrics_list = [
            ('Total Return', f"{metrics.get('total_return_pct', 0.0):.2f}%"),
            ('Annual Return', f"{metrics.get('annual_return_pct', 0.0):.2f}%"),
            ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0.0):.2f}"),
            ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0.0):.2f}"),
            ('Calmar Ratio', f"{metrics.get('calmar_ratio', 0.0):.2f}"),
            ('Max Drawdown', f"{metrics.get('max_drawdown_pct', 0.0):.2f}%"),
        ]
        
        for name, value in metrics_list:
            html_lines.append(f"        <tr><td>{name}</td><td>{value}</td></tr>")
        
        html_lines.append("    </table>")
        html_lines.append("    <h2>Trade Statistics</h2>")
        html_lines.append("    <table>")
        html_lines.append("        <tr><th>Statistic</th><th>Value</th></tr>")
        
        trade_stats = [
            ('Total Trades', str(trade_summary['total_trades'])),
            ('Winning Trades', str(trade_summary['winning_trades'])),
            ('Losing Trades', str(trade_summary['losing_trades'])),
            ('Win Rate', f"{trade_summary['win_rate']:.1f}%"),
            ('Avg Win', f"${self.analyzer.get_average_win():.2f}"),
            ('Avg Loss', f"${self.analyzer.get_average_loss():.2f}"),
            ('Risk/Reward Ratio', f"{self.analyzer.get_risk_reward_ratio():.2f}"),
            ('Profit Factor', f"{metrics.get('profit_factor', 0.0):.2f}"),
        ]
        
        for name, value in trade_stats:
            html_lines.append(f"        <tr><td>{name}</td><td>{value}</td></tr>")
        
        html_lines.append("    </table>")
        
        if self.trades:
            html_lines.append("    <h2>All Trades</h2>")
            html_lines.append("    <table>")
            html_lines.append("        <tr><th>ID</th><th>Symbol</th><th>Entry Date</th><th>Entry Price</th><th>Exit Price</th><th>P&L</th><th>Return %</th></tr>")
            
            for trade in self.trades:
                pnl_class = 'positive' if trade.gross_pnl > 0 else 'negative' if trade.gross_pnl < 0 else ''
                html_lines.append(f"        <tr><td>{trade.trade_id}</td><td>{trade.symbol}</td><td>{trade.entry_date.strftime('%Y-%m-%d')}</td><td>${trade.entry_price:.2f}</td><td>${trade.exit_price:.2f}</td><td class='{pnl_class}'>${trade.gross_pnl:.2f}</td><td class='{pnl_class}'>{trade.pnl_percent:.2f}%</td></tr>")
            
            html_lines.append("    </table>")
        
        if monthly_returns:
            html_lines.append("    <h2>Monthly Returns</h2>")
            html_lines.append("    <table>")
            html_lines.append("        <tr><th>Month</th><th>P&L</th></tr>")
            
            for month in sorted(monthly_returns.keys()):
                pnl = monthly_returns[month]
                pnl_class = 'positive' if pnl > 0 else 'negative' if pnl < 0 else ''
                html_lines.append(f"        <tr><td>{month}</td><td class='{pnl_class}'>${pnl:.2f}</td></tr>")
            
            html_lines.append("    </table>")
        
        html_lines.append("</body>")
        html_lines.append("</html>")
        
        return "\n".join(html_lines)
    
    def to_csv(self) -> str:
        """Generate CSV report of trades"""
        logger.info("Generating CSV report...")
        lines = []
        lines.append('TradeID,Symbol,EntryDate,EntryPrice,EntryQuantity,ExitDate,ExitPrice,EntryValue,ExitValue,GrossPnL,PnLPercent')
        
        for trade in self.trades:
            row = f"{trade.trade_id},{trade.symbol},{trade.entry_date.strftime('%Y-%m-%d %H:%M:%S')},{trade.entry_price:.2f},{trade.entry_quantity:.4f},{trade.exit_date.strftime('%Y-%m-%d %H:%M:%S')},{trade.exit_price:.2f},{trade.entry_value:.2f},{trade.exit_value:.2f},{trade.gross_pnl:.2f},{trade.pnl_percent:.2f}"
            lines.append(row)
        
        return '\n'.join(lines)
    
    def save_html(self, file_path: str) -> None:
        """Save HTML report to file"""
        logger.info(f"Saving HTML report to {file_path}")
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(self.to_html())
        logger.info(f"HTML report saved to {file_path}")
    
    def save_text(self, file_path: str) -> None:
        """Save text report to file"""
        logger.info(f"Saving text report to {file_path}")
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(self.to_text())
        logger.info(f"Text report saved to {file_path}")
    
    def save_csv(self, file_path: str) -> None:
        """Save CSV report to file"""
        logger.info(f"Saving CSV report to {file_path}")
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(self.to_csv())
        logger.info(f"CSV report saved to {file_path}")
    
    def __repr__(self) -> str:
        """String representation"""
        return f"BacktestReport | {len(self.trades)} trades"