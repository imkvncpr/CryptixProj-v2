import matplotlib.pyplot as plt
import pandas as pd

def plot_portfolio_value(dates, values):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, values, linewidth=2, color='#1f77b4')
    ax.fill_between(dates, values, alpha=0.3, color='#1f77b4')
    ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (USD)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_pnl(trade_dates, pnl_values):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if x > 0 else 'red' for x in pnl_values]
    ax.bar(trade_dates, pnl_values, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('Trade P&L', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trade Date')
    ax.set_ylabel('P&L (USD)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_monthly_returns(returns_dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    months = list(returns_dict.keys())
    values = list(returns_dict.values())
    colors = ['green' if x > 0 else 'red' for x in values]
    ax.bar(months, values, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('Monthly Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_drawdown(dates, peak_values, current_values):
    fig, ax = plt.subplots(figsize=(12, 6))
    drawdown = [(peak - curr) / peak * 100 for peak, curr in zip(peak_values, current_values)]
    ax.fill_between(dates, drawdown, alpha=0.5, color='red')
    ax.plot(dates, drawdown, linewidth=2, color='darkred')
    ax.set_title('Drawdown Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_asset_allocation(symbols, values):
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(range(len(symbols)))
    wedges, texts, autotexts = ax.pie(values, labels=symbols, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Asset Allocation', fontsize=14, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    plt.tight_layout()
    return fig

def plot_returns_distribution(returns):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(returns, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax.set_title('Returns Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Return (USD)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    plt.tight_layout()
    return fig
