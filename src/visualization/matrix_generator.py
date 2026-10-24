"""
Visual Matrix Output Generator for CrowdWisdomTrading Quant System.
Generates institutional publication-grade charts visualizing X sentiment vs price dynamics,
multi-horizon prediction matrices, equity curves, and risk profiles.
"""

import os
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from loguru import logger


class VisualMatrixGenerator:
    def __init__(self, output_dir: str = "outputs/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_style()

    def _setup_style(self):
        """Applies institutional dark quantitative palette."""
        plt.style.use("dark_background")
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["axes.edgecolor"] = "#444444"
        plt.rcParams["axes.linewidth"] = 0.8
        plt.rcParams["grid.color"] = "#2A2A2A"
        plt.rcParams["grid.linestyle"] = "--"
        plt.rcParams["grid.alpha"] = 0.6

    def generate_all_matrices(
        self,
        df_backtest: pd.DataFrame,
        metrics_summary: dict[str, Any],
        feature_importances: dict[int, pd.Series] | None = None
    ) -> list[str]:
        """Runs the full visual matrix suite and saves PNG plots."""
        saved_plots = []
        p1 = self.plot_sentiment_price_matrix(df_backtest)
        p2 = self.plot_multi_horizon_accuracy_matrix(metrics_summary)
        p3 = self.plot_backtest_performance_matrix(df_backtest)
        saved_plots.extend([p1, p2, p3])

        if feature_importances:
            p4 = self.plot_feature_importance_matrix(feature_importances)
            saved_plots.append(p4)

        return saved_plots

    def plot_sentiment_price_matrix(self, df: pd.DataFrame) -> str:
        """
        Visualizes X sentiment changes directly against price action over time.
        Top panel: Price path with bullish/bearish sentiment regimes.
        Bottom panel: Engagement-weighted sentiment and sentiment momentum.
        """
        file_path = os.path.join(self.output_dir, "sentiment_price_matrix.png")
        logger.info(f"Generating Sentiment vs Price Matrix Chart at {file_path}...")

        _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        dates = df["timestamp_utc"]

        # Top Panel: Price + Regime Shading
        ax1.plot(dates, df["close"], color="#00E5FF", linewidth=1.5, label="Asset Close Price ($)")
        if "price_ema_24" in df.columns:
            ax1.plot(dates, df["price_ema_24"], color="#FFAB00", linestyle="--", linewidth=1.0, label="24h EMA")

        # Shading for extreme sentiment zones
        sent = df["engagement_weighted_sentiment"]
        ax1.fill_between(dates, df["close"].min() * 0.98, df["close"].max() * 1.02,
                         where=(sent > 0.15), color="#00E676", alpha=0.12, label="Extreme Bull Sentiment")
        ax1.fill_between(dates, df["close"].min() * 0.98, df["close"].max() * 1.02,
                         where=(sent < -0.15), color="#FF1744", alpha=0.12, label="Extreme Bear Sentiment")

        ax1.set_title("CrowdWisdomTrading: X (Twitter) Sentiment Dynamics vs Market Price Action", fontsize=14, fontweight="bold", pad=12)
        ax1.set_ylabel("Price ($)", fontsize=11)
        ax1.legend(loc="upper left", framealpha=0.4)
        ax1.grid(True)

        # Bottom Panel: Sentiment & Momentum
        ax2.bar(dates, df["engagement_weighted_sentiment"], color=np.where(sent >= 0, "#00E676", "#FF1744"),
                alpha=0.6, width=0.03, label="Engagement-Weighted Sentiment")
        if "sentiment_momentum" in df.columns:
            ax2.plot(dates, df["sentiment_momentum"], color="#FFD600", linewidth=1.2, label="Sentiment Momentum (EMA Diff)")

        ax2.axhline(0, color="#FFFFFF", linestyle=":", alpha=0.5)
        ax2.set_ylabel("Sentiment Score", fontsize=11)
        ax2.set_xlabel("Time (UTC)", fontsize=11)
        ax2.legend(loc="upper left", framealpha=0.4)
        ax2.grid(True)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.tight_layout()
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close()
        return file_path

    def plot_multi_horizon_accuracy_matrix(self, metrics: dict[str, Any]) -> str:
        """
        Visualizes prediction performance across horizons n+0 to n+5.
        Plots Directional Hit Ratio (%) and Information Coefficient (Rank IC).
        """
        file_path = os.path.join(self.output_dir, "multi_horizon_matrix.png")
        logger.info(f"Generating Multi-Horizon Accuracy Matrix Chart at {file_path}...")

        horizons = []
        hit_ratios = []
        rank_ics = []
        rmses = []

        for key, val in metrics.items():
            if key.startswith("horizon_"):
                horizons.append(val["horizon"])
                hit_ratios.append(val["hit_ratio_pct"])
                rank_ics.append(val["rank_ic"])
                rmses.append(val["rmse"] * 100) # Convert to bps / %

        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

        # Subplot 1: Directional Accuracy (Hit Ratio)
        x = np.arange(len(horizons))
        bars = ax1.bar(x, hit_ratios, color="#7C4DFF", alpha=0.85, width=0.55)
        ax1.axhline(50.0, color="#FF5252", linestyle="--", linewidth=1.5, label="Random Benchmark (50%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(horizons, fontsize=11)
        ax1.set_ylabel("Directional Hit Ratio (%)", fontsize=11)
        ax1.set_ylim(40, max(max(hit_ratios) + 5, 60))
        ax1.set_title("Out-of-Sample Directional Hit Ratio (n+0 to n+5)", fontsize=12, fontweight="bold")
        ax1.legend(loc="lower right")
        ax1.grid(True)

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f"{height:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Subplot 2: Information Coefficient (Rank IC)
        bars2 = ax2.bar(x, rank_ics, color="#00B0FF", alpha=0.85, width=0.55)
        ax2.axhline(0.0, color="#FFFFFF", linestyle=":", linewidth=1.0)
        ax2.set_xticks(x)
        ax2.set_xticklabels(horizons, fontsize=11)
        ax2.set_ylabel("Spearman Rank IC", fontsize=11)
        ax2.set_title("Information Coefficient Across Horizons", fontsize=12, fontweight="bold")
        ax2.grid(True)

        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f"{height:.3f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3 if height >= 0 else -12), textcoords="offset points",
                         ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.suptitle("Multi-Horizon Walk-Forward Predictive Accuracy Matrix", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close()
        return file_path

    def plot_backtest_performance_matrix(self, df: pd.DataFrame) -> str:
        """
        Visualizes Cumulative Strategy Equity vs Buy & Hold Benchmark and Underwater Drawdowns.
        """
        file_path = os.path.join(self.output_dir, "backtest_performance_matrix.png")
        logger.info(f"Generating Backtest Performance Matrix at {file_path}...")

        _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

        dates = df["timestamp_utc"]

        # Top Panel: Equity Curves
        ax1.plot(dates, df["strategy_equity"], color="#00E676", linewidth=2.0, label="Multi-Horizon Sentiment Strategy (Net of Costs)")
        ax1.plot(dates, df["benchmark_equity"], color="#9E9E9E", linestyle="--", linewidth=1.5, label="Passive Buy & Hold Benchmark")
        ax1.set_ylabel("Portfolio Value ($ USD)", fontsize=11)
        ax1.set_title("Quantitative Strategy Equity Growth vs Benchmark", fontsize=13, fontweight="bold")
        ax1.legend(loc="upper left", framealpha=0.5)
        ax1.grid(True)

        # Bottom Panel: Underwater Drawdown
        ax2.plot(dates, df["strategy_drawdown"] * 100.0, color="#00E676", linewidth=1.2, label="Strategy Drawdown")
        ax2.plot(dates, df["benchmark_drawdown"] * 100.0, color="#FF1744", linestyle=":", linewidth=1.2, label="Benchmark Drawdown")
        ax2.fill_between(dates, df["strategy_drawdown"] * 100.0, 0, color="#00E676", alpha=0.15)
        ax2.fill_between(dates, df["benchmark_drawdown"] * 100.0, 0, color="#FF1744", alpha=0.1)
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Time (UTC)", fontsize=11)
        ax2.legend(loc="lower left", framealpha=0.5)
        ax2.grid(True)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.tight_layout()
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close()
        return file_path

    def plot_feature_importance_matrix(self, feature_importances: dict[int, pd.Series]) -> str:
        """Visualizes top sentiment & price feature importances."""
        file_path = os.path.join(self.output_dir, "feature_importance_matrix.png")
        logger.info(f"Generating Feature Importance Matrix at {file_path}...")

        # Average importance across horizons
        combined = pd.DataFrame(feature_importances).fillna(0.0).mean(axis=1).sort_values(ascending=True)
        top_features = combined.tail(15)

        _fig, ax = plt.subplots(figsize=(10, 6.5))
        y_pos = np.arange(len(top_features))

        # Color sentiment features differently from price features
        colors = ["#00E5FF" if any(s in f for s in ["sent", "bull", "bear", "post", "dispersion"]) else "#FFAB00" for f in top_features.index]

        ax.barh(y_pos, top_features.values, color=colors, alpha=0.85, height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features.index, fontsize=10)
        ax.set_xlabel("Average Split Importance (LightGBM)", fontsize=11)
        ax.set_title("Top 15 Alpha Features (Cyan: Sentiment-derived, Orange: Price-derived)", fontsize=12, fontweight="bold")
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close()
        return file_path
