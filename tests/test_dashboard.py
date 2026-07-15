import pytest
import os

class TestReportingPackage:
    def test_reporting_dir_exists(self):
        assert os.path.exists('src/reporting')
    
    def test_charts_file_exists(self):
        assert os.path.exists('src/reporting/charts.py')
    
    def test_dashboard_file_exists(self):
        assert os.path.exists('src/reporting/dashboard.py')
    
    def test_init_file_exists(self):
        assert os.path.exists('src/reporting/__init__.py')

class TestChartsModule:
    def test_charts_import(self):
        from src.reporting import charts
        assert charts is not None
    
    def test_plot_functions_exist(self):
        from src.reporting.charts import (
            plot_portfolio_value, 
            plot_pnl,
            plot_monthly_returns,
            plot_drawdown,
            plot_asset_allocation,
            plot_returns_distribution
        )
        assert all([
            callable(plot_portfolio_value),
            callable(plot_pnl),
            callable(plot_monthly_returns),
            callable(plot_drawdown),
            callable(plot_asset_allocation),
            callable(plot_returns_distribution)
        ])

class TestDashboard:
    def test_dashboard_file_valid(self):
        with open('src/reporting/dashboard.py', 'r') as f:
            content = f.read()
            assert 'streamlit' in content
            assert 'st.title' in content or 'title' in content
    
    def test_dashboard_has_tabs(self):
        with open('src/reporting/dashboard.py', 'r') as f:
            content = f.read()
            assert 'st.tabs' in content or 'tabs' in content
    
    def test_dashboard_has_metrics(self):
        with open('src/reporting/dashboard.py', 'r') as f:
            content = f.read()
            assert 'metric' in content or 'Metric' in content

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
