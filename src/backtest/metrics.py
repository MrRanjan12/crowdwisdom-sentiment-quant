"""
Quantitative Performance & Risk Metrics for CrowdWisdomTrading.
Calculates Annualized Sharpe, Sortino, Max Drawdown, Calmar, and Win Rate.
"""

from typing import Any

import numpy as np
import pandas as pd


def calculate_portfolio_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: int = 8760  # 8760 for hourly crypto, 252 for daily
) -> dict[str, Any]:
    """
    Computes standard quantitative performance metrics from a returns series.
    """
    if returns.empty or returns.std() == 0 or np.isnan(returns.std()):
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar_ratio": 0.0,
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "annualized_volatility_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0
        }

    # Per-period risk free rate
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = returns - rf_per_period

    # 1. Total & Annualized Return
    cum_returns = (1.0 + returns).cumprod()
    total_return = float(cum_returns.iloc[-1] - 1.0)
    n_periods = len(returns)
    years = max(n_periods / periods_per_year, 1e-4)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)

    # 2. Annualized Volatility
    ann_vol = float(returns.std() * np.sqrt(periods_per_year))

    # 3. Sharpe Ratio
    mean_excess = excess_returns.mean()
    std_returns = returns.std()
    sharpe = float((mean_excess / (std_returns + 1e-9)) * np.sqrt(periods_per_year))

    # 4. Sortino Ratio (Downside deviation only)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        sortino = float((mean_excess / (downside_std + 1e-9)) * np.sqrt(periods_per_year))
    else:
        sortino = sharpe * 1.5

    # 5. Maximum Drawdown (MDD)
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = float(abs(drawdown.min())) * 100.0

    # 6. Calmar Ratio
    calmar = float(cagr / (max_dd / 100.0 + 1e-6))

    # 7. Win Rate & Profit Factor
    pos_returns = returns[returns > 0]
    neg_returns = returns[returns < 0]
    win_rate = float(len(pos_returns) / (len(returns[returns != 0]) + 1e-6)) * 100.0
    gross_profit = float(pos_returns.sum())
    gross_loss = float(abs(neg_returns.sum()))
    profit_factor = float(gross_profit / (gross_loss + 1e-6))

    return {
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar_ratio": round(calmar, 3),
        "total_return_pct": round(total_return * 100.0, 2),
        "annualized_return_pct": round(cagr * 100.0, 2),
        "annualized_volatility_pct": round(ann_vol * 100.0, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2)
    }
