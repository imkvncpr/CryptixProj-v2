import numpy as np
from typing import List 
from datetime import datetime

def calculate_total_return(initial_value: float, final_value: float)-> float:
    return ((final_value - initial_value) / initial_value) * 100

def calculate_annual_return(total_return: float, num_days: int)-> float:
    years = num_days / 365.25
    if years == 0:
        return 0.0
    return (total_return / 100) / years * 100

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02)-> float:
    if not returns or len(returns) < 2:
        return 0.0
    returns_array = np.array(returns)
    volatility = np.std(returns_array)
    if volatility == 0:
        return 0.0
    
    mean_return = np.mean(returns_array)
    sharpe = (mean_return - (risk_free_rate / 252)) / volatility
    return sharpe * np.sqrt(252)

def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02)-> float:
    if not returns or len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    downside_returns = returns_array[returns_array < 0]
    if len(downside_returns) == 0:
        return float('inf')
    downside_volatility = np.std(downside_returns)
    if downside_volatility == 0:
        return 0.0
    mean_return = np.mean(returns_array)
    sortino = (mean_return - (risk_free_rate / 252)) / downside_volatility
    return sortino * np.sqrt(252)

def calculate_calmar_ratio(annual_return: float, max_drawdown: float)-> float:
    if max_drawdown == 0:
        return 0.0
    return annual_return / max_drawdown

def calculate_recovery_factor(total_pnl: float, max_drawndown_amount: float)-> float:
    if max_drawndown_amount == 0:
        return 0.0
    return abs(total_pnl) / abs(max_drawndown_amount)

def calculate_max_consecutive_wins(trades: List)-> int:
    if not trades:
        return 0
    max_streak = 0
    current_streak = 0
    for trade in trades:
        if trade.is_winning_trade:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak

def calculate_max_consecutive_losses(trades: List)-> int:
    if not trades:
        return 0 
    max_streak = 0
    current_streak = 0
    for trade in trades:
        if trade.is_losing_trade:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak

def calculate_avg_win(trades: List)-> float:
    winning_trades = [t for t in trades if t.is_winning_trade]
    if not winning_trades:
        return 0.0
    total_wins = sum(t.gross_pnl for t in winning_trades)
    return total_wins / len(winning_trades)

def calculate_avg_loss(trades: List)-> float:
    losing_trades = [t for t in trades if t.is_losing_trade]
    if not losing_trades:
        return 0.0
    total_losses = sum(t.gross_pnl for t in losing_trades)
    return total_losses / len(losing_trades)

def calculate_avg_trade_duration(trades: List)-> float:
    if not trades:
        return 0.0
    total_duration = sum(t.duration_days for t in trades)
    return total_duration / len(trades)

def calculate_risk_reward_ratio(trades: List)-> float:
    avg_win = calculate_avg_win(trades)
    avg_loss = abs(calculate_avg_loss(trades))
    if avg_loss == 0:
        return 0.0
    return avg_win / avg_loss

def calculate_payoff_ratio(trades: List)-> float:
    return calculate_risk_reward_ratio(trades)

def calculate_monthly_returns(daily_pnl: dict)-> dict:
    monthly = {}
    for date, pnl in daily_pnl.items():
        month_key = date.strftime('%Y-%m')
        if month_key not in monthly:
            monthly[month_key] = 0.0
        monthly[month_key] += pnl
    return monthly

def calculate_yearly_returns(daily_pnl: dict) -> dict:
    yearly = {}
    for date, pnl in daily_pnl.items():
        year_key = date.strftime('%Y')
        if year_key not in yearly:
            yearly[year_key] = 0.0
        yearly[year_key] += pnl
    return yearly

def calculate_win_loss_ratio(trades: List) -> float:
    total_trades = len(trades)
    if total_trades == 0:
        return 0.0
    winning = sum(1 for t in trades if t.is_winning_trade)
    losing = sum(1 for t in trades if t.is_losing_trade)
    if losing == 0:
        return float('inf') if winning > 0 else 0.0
    return winning / losing
