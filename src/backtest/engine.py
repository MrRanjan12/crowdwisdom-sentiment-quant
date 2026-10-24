"""
Quantitative Backtest Engine for CrowdWisdomTrading.
Simulates execution of multi-horizon sentiment & price alpha strategies
with realistic transaction costs, slippage, and portfolio attribution.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from .metrics import calculate_portfolio_metrics

class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        fee_bps: float = 5.0,
        risk_free_rate: float = 0.04,
        allow_short: bool = False
    ):
        self.initial_capital = initial_capital
        self.fee = fee_bps / 10000.0
        self.risk_free_rate = risk_free_rate
        self.allow_short = allow_short

    def run_multi_horizon_backtest(
        self,
        df_oos: pd.DataFrame,
        horizon_weights: Dict[int, float] = None,
        threshold: float = 0.0005
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Backtests strategy driven by multi-horizon (n+0 to n+5) forecasts.
        """
        if horizon_weights is None:
            # Forward horizons only (n+1 to n+5). n+0 is nowcasting current bar.
            horizon_weights = {1: 0.35, 2: 0.30, 3: 0.20, 4: 0.10, 5: 0.05}

        logger.info(f"Running multi-horizon backtest across {len(df_oos)} out-of-sample periods...")
        df = df_oos.copy().sort_values("timestamp_utc").reset_index(drop=True)

        # 1. Composite forward alpha signal
        composite_signal = np.zeros(len(df))
        for h, weight in horizon_weights.items():
            col = f"pred_h{h}"
            if col in df.columns:
                composite_signal += weight * df[col].fillna(0.0).values

        # 2. Smooth signal and normalize to z-score to prevent high-frequency fee drag
        s_series = pd.Series(composite_signal)
        smooth_alpha = s_series.ewm(span=8, adjust=False).mean()
        roll_std = s_series.rolling(48, min_periods=12).std().bfill().fillna(1e-4)
        z_score = (smooth_alpha / (roll_std + 1e-6)).fillna(0.0).values

        # Trend & social momentum confirmation filter
        trend = df["price_to_ema_24"].fillna(0.0).values if "price_to_ema_24" in df.columns else np.zeros(len(df))
        alpha = z_score + 0.3 * np.sign(trend)
        df["composite_alpha"] = alpha

        # 3. Position allocation with hysteresis band (reduces excessive turnover & slippage)
        positions = np.zeros(len(df))
        curr_pos = 0.0
        enter_thresh = 0.50
        exit_thresh = 0.00

        for i in range(len(df)):
            a = alpha[i]
            if curr_pos == 0.0:
                if a > enter_thresh:
                    curr_pos = 1.0
                elif a < -enter_thresh and self.allow_short:
                    curr_pos = -1.0
            elif curr_pos > 0.0:
                if a < exit_thresh:
                    curr_pos = 0.0
            elif curr_pos < 0.0:
                if a > -exit_thresh:
                    curr_pos = 0.0

            positions[i] = curr_pos

        df["target_position"] = positions
        # Shift position by 1 bar to simulate entering at next open/close (realistic execution)
        df["position"] = df["target_position"].shift(1).fillna(0.0)

        # 3. Turnover and transaction fees
        df["turnover"] = df["position"].diff().abs().fillna(0.0)
        df["tx_cost"] = df["turnover"] * self.fee

        # 4. Strategy Returns vs Benchmark (Buy & Hold)
        df["benchmark_return"] = df["log_return"]
        df["strategy_raw_return"] = df["position"] * df["benchmark_return"]
        df["strategy_net_return"] = df["strategy_raw_return"] - df["tx_cost"]

        # 5. Equity Curves
        df["strategy_equity"] = self.initial_capital * (1.0 + df["strategy_net_return"]).cumprod()
        df["benchmark_equity"] = self.initial_capital * (1.0 + df["benchmark_return"]).cumprod()

        # 6. Drawdowns
        strat_peak = df["strategy_equity"].cummax()
        df["strategy_drawdown"] = (df["strategy_equity"] - strat_peak) / strat_peak

        bench_peak = df["benchmark_equity"].cummax()
        df["benchmark_drawdown"] = (df["benchmark_equity"] - bench_peak) / bench_peak

        # 7. Compute institutional performance metrics
        strat_metrics = calculate_portfolio_metrics(df["strategy_net_return"], self.risk_free_rate)
        bench_metrics = calculate_portfolio_metrics(df["benchmark_return"], self.risk_free_rate)

        summary = {
            "strategy": strat_metrics,
            "benchmark": bench_metrics,
            "alpha_spread": round(strat_metrics["annualized_return_pct"] - bench_metrics["annualized_return_pct"], 2),
            "sharpe_improvement": round(strat_metrics["sharpe_ratio"] - bench_metrics["sharpe_ratio"], 3),
            "max_dd_reduction": round(bench_metrics["max_drawdown_pct"] - strat_metrics["max_drawdown_pct"], 2)
        }

        logger.info(
            f"Backtest Complete | Strategy Sharpe: {strat_metrics['sharpe_ratio']} (vs Bench: {bench_metrics['sharpe_ratio']}) | "
            f"Strategy Max DD: {strat_metrics['max_drawdown_pct']}% (vs Bench: {bench_metrics['max_drawdown_pct']}%)"
        )
        return df, summary
