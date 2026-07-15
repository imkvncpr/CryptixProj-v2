import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.backtesting.backtest import Backtest
from src.backtesting.data_handler import HistoricalDataHandler, create_sample_csv
from src.backtesting.models import Candle, BacktestTrade, BacktestResults
from src.backtesting.analyzer import BacktestAnalyzer
from src.backtesting.report import BacktestReport
from src.backtesting import metrics


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Create test data once for all tests"""
    Path('./data').mkdir(exist_ok=True)
    create_sample_csv('./data/test_bitcoin.csv', 'BTC/USD', 100, 28500)
    yield


@pytest.fixture
def sample_trades():
    """Create sample trades for testing"""
    trades = [
        BacktestTrade(
            trade_id=1,
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500
        ),
        BacktestTrade(
            trade_id=2,
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 6),
            entry_price=29500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 10),
            exit_price=28500
        ),
        BacktestTrade(
            trade_id=3,
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 11),
            entry_price=28500,
            entry_quantity=0.2,
            exit_date=datetime(2023, 1, 15),
            exit_price=30500
        ),
    ]
    return trades


@pytest.fixture
def sample_results(sample_trades):
    """Create sample BacktestResults"""
    portfolio_values = {
        datetime(2023, 1, 1): 10000,
        datetime(2023, 1, 2): 10100,
        datetime(2023, 1, 3): 10050,
        datetime(2023, 1, 4): 10200,
        datetime(2023, 1, 5): 10100,
    }
    
    daily_returns = {
        datetime(2023, 1, 1): 0,
        datetime(2023, 1, 2): 100,
        datetime(2023, 1, 3): -50,
        datetime(2023, 1, 4): 150,
        datetime(2023, 1, 5): -100,
    }
    
    metrics_dict = {
        'total_return_pct': 1.0,
        'sharpe_ratio': 0.5,
        'max_drawdown': 2.0,
        'win_rate': 66.67,
        'profit_factor': 2.5
    }
    
    return BacktestResults(
        trades=sample_trades,
        portfolio_values=portfolio_values,
        daily_returns=daily_returns,
        metrics=metrics_dict
    )


class TestMetricsCalculations:
    """Test metrics calculation functions"""
    
    def test_calculate_total_return(self):
        total_return = metrics.calculate_total_return(10000, 11500)
        assert total_return == pytest.approx(15.0)
    
    def test_calculate_total_return_loss(self):
        total_return = metrics.calculate_total_return(10000, 9000)
        assert total_return == pytest.approx(-10.0)
    
    def test_calculate_annual_return(self):
        annual = metrics.calculate_annual_return(15.0, 365)
        assert annual == pytest.approx(15.0, 0.01)
    
    def test_calculate_annual_return_half_year(self):
        annual = metrics.calculate_annual_return(10.0, 182)
        assert annual == pytest.approx(20.0, rel=0.01)
    
    def test_calculate_sharpe_ratio_positive(self):
        returns = [100, 50, -25, 75, 150]
        sharpe = metrics.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_calculate_sharpe_ratio_empty(self):
        sharpe = metrics.calculate_sharpe_ratio([])
        assert sharpe == 0.0
    
    def test_calculate_sharpe_ratio_single(self):
        sharpe = metrics.calculate_sharpe_ratio([100])
        assert sharpe == 0.0
    
    def test_calculate_sortino_ratio_positive(self):
        returns = [100, 50, -25, 75, 150]
        sortino = metrics.calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
    
    def test_calculate_sortino_ratio_all_positive(self):
        returns = [100, 50, 75, 150]
        sortino = metrics.calculate_sortino_ratio(returns)
        assert sortino == float('inf')
    
    def test_calculate_calmar_ratio(self):
        calmar = metrics.calculate_calmar_ratio(15.0, 5.0)
        assert calmar == pytest.approx(3.0)
    
    def test_calculate_calmar_ratio_zero_drawdown(self):
        calmar = metrics.calculate_calmar_ratio(15.0, 0.0)
        assert calmar == 0.0
    
    def test_calculate_recovery_factor(self):
        recovery = metrics.calculate_recovery_factor(1500, 500)
        assert recovery == pytest.approx(3.0)
    
    def test_calculate_monthly_returns(self):
        daily_pnl = {
            datetime(2023, 1, 1): 100,
            datetime(2023, 1, 5): 200,
            datetime(2023, 2, 1): -50,
            datetime(2023, 2, 15): 150,
        }
        monthly = metrics.calculate_monthly_returns(daily_pnl)
        assert monthly['2023-01'] == 300
        assert monthly['2023-02'] == 100
    
    def test_calculate_yearly_returns(self):
        daily_pnl = {
            datetime(2023, 1, 1): 100,
            datetime(2023, 12, 31): 200,
            datetime(2024, 1, 1): 150,
        }
        yearly = metrics.calculate_yearly_returns(daily_pnl)
        assert yearly['2023'] == 300
        assert yearly['2024'] == 150


class TestBacktestAnalyzer:
    """Test BacktestAnalyzer class"""
    
    def test_analyzer_creation(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        assert analyzer.results == sample_results
        assert len(analyzer.trades) == 3
    
    def test_calculate_all_metrics(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        metrics_dict = analyzer.calculate_all_metrics()
        assert 'total_return_pct' in metrics_dict
        assert 'annual_return_pct' in metrics_dict
        assert 'sharpe_ratio' in metrics_dict
        assert 'total_trades' in metrics_dict
    
    def test_get_trade_summary(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        summary = analyzer.get_trade_summary()
        assert summary['total_trades'] == 3
        assert summary['winning_trades'] == 2
        assert summary['losing_trades'] == 1
        assert summary['win_rate'] == pytest.approx(66.67, 0.01)
    
    def test_get_monthly_returns(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        monthly = analyzer.get_monthly_returns()
        assert '2023-01' in monthly
        assert monthly['2023-01'] == 100
    
    def test_get_yearly_returns(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        yearly = analyzer.get_yearly_returns()
        assert '2023' in yearly
        assert yearly['2023'] == 100
    
    def test_get_best_trade(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        best = analyzer.get_best_trade()
        assert best is not None
        assert best.gross_pnl > 0
    
    def test_get_worst_trade(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        worst = analyzer.get_worst_trade()
        assert worst is not None
        assert worst.trade_id == 2
    
    def test_get_average_win(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        avg_win = analyzer.get_average_win()
        assert avg_win > 0
    
    def test_get_average_loss(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        avg_loss = analyzer.get_average_loss()
        assert avg_loss < 0
    
    def test_get_risk_reward_ratio(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        ratio = analyzer.get_risk_reward_ratio()
        assert ratio > 0
    
    def test_analyzer_repr(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        repr_str = repr(analyzer)
        assert 'BacktestAnalyzer' in repr_str
        assert '3 trades' in repr_str


class TestBacktestReport:
    """Test BacktestReport class"""
    
    def test_report_creation(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        report = BacktestReport(analyzer)
        assert report.analyzer == analyzer
        assert len(report.trades) == 3
    
    def test_to_text(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        report = BacktestReport(analyzer)
        text = report.to_text()
        assert isinstance(text, str)
        assert 'BACKTEST RESULTS SUMMARY' in text
    
    def test_to_html(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        report = BacktestReport(analyzer)
        html = report.to_html()
        assert isinstance(html, str)
        assert '<html>' in html
        assert '<table>' in html
        assert '</html>' in html
    
    def test_to_html_contains_metrics(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        report = BacktestReport(analyzer)
        html = report.to_html()
        assert 'Performance Metrics' in html
        assert 'Trade Statistics' in html
    
    def test_to_html_contains_trades(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        report = BacktestReport(analyzer)
        html = report.to_html()
        assert 'All Trades' in html
    
    def test_to_csv(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        report = BacktestReport(analyzer)
        csv = report.to_csv()
        assert isinstance(csv, str)
        assert 'TradeID' in csv
        assert 'Symbol' in csv
        assert '1,BTC/USD' in csv
    
    def test_save_html(self, sample_results, tmp_path):
        analyzer = BacktestAnalyzer(sample_results)
        report = BacktestReport(analyzer)
        file_path = tmp_path / 'report.html'
        report.save_html(str(file_path))
        assert file_path.exists()
        content = file_path.read_text()
        assert '<html>' in content
    
    def test_report_repr(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        report = BacktestReport(analyzer)
        repr_str = repr(report)
        assert 'BacktestReport' in repr_str
        assert '3 trades' in repr_str


class TestAnalyzerIntegration:
    """Integration tests"""
    
    def test_analyzer_with_real_backtest(self):
        data_handler = HistoricalDataHandler('./data')
        data_handler.load_csv('BTC/USD', './data/test_bitcoin.csv')
        
        backtest = Backtest(
            initial_capital=10000,
            data_handler=data_handler,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 15),
            commission=0.001
        )
        
        success = backtest.add_trade(
            symbol='BTC/USD',
            entry_date=datetime(2023, 1, 1),
            entry_price=28500,
            entry_quantity=0.1,
            exit_date=datetime(2023, 1, 5),
            exit_price=29500
        )
        assert success
        
        backtest.portfolio_values = {datetime(2023, 1, 1): 10000, datetime(2023, 1, 5): 10100}
        results = backtest._calculate_results()
        analyzer = BacktestAnalyzer(results)
        metrics_dict = analyzer.calculate_all_metrics()
        assert 'total_trades' in metrics_dict
        assert metrics_dict['total_trades'] == 1
    
    def test_full_workflow(self):
        data_handler = HistoricalDataHandler('./data')
        data_handler.load_csv('BTC/USD', './data/test_bitcoin.csv')
        
        backtest = Backtest(
            initial_capital=10000,
            data_handler=data_handler,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 15)
        )
        
        for i in range(3):
            backtest.add_trade(
                symbol='BTC/USD',
                entry_date=datetime(2023, 1, 1) + timedelta(days=i*10),
                entry_price=28500 + (i * 100),
                entry_quantity=0.1,
                exit_date=datetime(2023, 1, 5) + timedelta(days=i*10),
                exit_price=29500 + (i * 100)
            )
        
        backtest.portfolio_values = {datetime(2023, 1, 1): 10000, datetime(2023, 1, 5): 10100}
        results = backtest._calculate_results()
        analyzer = BacktestAnalyzer(results)
        analyzer.print_summary()
        
        report = BacktestReport(analyzer)
        html = report.to_html()
        csv = report.to_csv()
        
        assert '<html>' in html
        assert 'TradeID' in csv
    
    def test_monthly_returns_breakdown(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        monthly = analyzer.get_monthly_returns()
        assert len(monthly) > 0
        for month_key in monthly.keys():
            assert len(month_key) == 7  # 'YYYY-MM' format
    
    def test_complete_analysis_chain(self, sample_results):
        analyzer = BacktestAnalyzer(sample_results)
        analyzer.calculate_all_metrics()
        analyzer.get_trade_summary()
        analyzer.get_monthly_returns()
        analyzer.get_best_trade()
        analyzer.get_worst_trade()
        
        report = BacktestReport(analyzer)
        html = report.to_html()
        csv = report.to_csv()
        text = report.to_text()
        
        assert len(html) > 0
        assert len(csv) > 0
        assert len(text) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

