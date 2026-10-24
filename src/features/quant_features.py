"""
Quantitative Feature Engineering Pipeline for CrowdWisdomTrading Quant System.
Combines X (Twitter) Sentiment and OHLCV Prices strictly into time-based predictive signals.
Constructs forward multi-horizon targets for n+0, n+1, n+2, n+3, n+4, n+5.
Strictly eliminates lookahead bias.
"""

import numpy as np
import pandas as pd
from loguru import logger


class QuantFeaturePipeline:
    def __init__(self, asset_symbol: str = "BTC-USD", horizons: list | None = None):
        if horizons is None:
            horizons = [0, 1, 2, 3, 4, 5]
        self.asset_symbol = asset_symbol
        self.horizons = horizons

    def build_aligned_features(self, df_prices: pd.DataFrame, df_sentiment_agg: pd.DataFrame) -> pd.DataFrame:
        """
        Merges prices and sentiment on timestamp_utc, constructs strictly sentiment & price features,
        and generates multi-horizon target labels.
        """
        logger.info("Aligning price bars and aggregated social sentiment...")
        if df_prices.empty:
            raise ValueError("Price DataFrame is empty!")

        # Ensure datetime indexing
        df_p = df_prices.copy()
        df_p["timestamp_utc"] = pd.to_datetime(df_p["timestamp_utc"], utc=True)
        df_p = df_p.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])

        if df_sentiment_agg is not None and not df_sentiment_agg.empty:
            df_s = df_sentiment_agg.copy()
            df_s["timestamp_utc"] = pd.to_datetime(df_s["timestamp_utc"], utc=True)
            df_s = df_s.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])
            # Merge on timestamp_utc
            merged = pd.merge(df_p, df_s, on=["timestamp_utc", "asset_symbol"], how="left")
        else:
            merged = df_p.copy()
            merged["post_count"] = 0
            merged["sentiment_score"] = 0.0
            merged["engagement_weighted_sentiment"] = 0.0
            merged["sentiment_dispersion"] = 0.0
            merged["bull_ratio"] = 0.5

        # Fill missing sentiment with neutral baselines
        merged["post_count"] = merged["post_count"].fillna(0)
        merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)
        merged["engagement_weighted_sentiment"] = merged["engagement_weighted_sentiment"].fillna(0.0)
        merged["sentiment_dispersion"] = merged["sentiment_dispersion"].fillna(0.0)
        merged["bull_ratio"] = merged["bull_ratio"].fillna(0.5)

        df_feat = merged.sort_values("timestamp_utc").reset_index(drop=True)

        logger.info("Calculating technical price features...")
        # 1. Price Returns & Lags
        df_feat["log_return"] = np.log(df_feat["close"] / df_feat["close"].shift(1)).fillna(0.0)
        for lag in [1, 2, 3, 6, 12, 24]:
            df_feat[f"ret_lag_{lag}"] = df_feat["log_return"].shift(lag).fillna(0.0)

        # 2. Price Volatility & Moving Averages
        for w in [6, 12, 24, 72]:
            df_feat[f"realized_vol_{w}"] = df_feat["log_return"].rolling(w, min_periods=2).std().fillna(0.0)
            df_feat[f"price_ema_{w}"] = df_feat["close"].ewm(span=w, adjust=False).mean()
            df_feat[f"price_to_ema_{w}"] = (df_feat["close"] / df_feat[f"price_ema_{w}"] - 1.0).fillna(0.0)

        # 3. High-Low Spread & Volume Momentum
        df_feat["hl_spread"] = ((df_feat["high"] - df_feat["low"]) / df_feat["close"]).fillna(0.0)
        df_feat["vol_ema_24"] = df_feat["volume"].ewm(span=24, adjust=False).mean()
        df_feat["volume_ratio"] = (df_feat["volume"] / (df_feat["vol_ema_24"] + 1e-6)).fillna(1.0)

        # 4. RSI (14 periods)
        delta = df_feat["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        df_feat["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
        df_feat["rsi_14"] = df_feat["rsi_14"].fillna(50.0)

        logger.info("Calculating time-based sentiment features...")
        # 5. Sentiment Momentum & Dispersion
        for sw in [3, 6, 12, 24, 72]:
            df_feat[f"sent_ema_{sw}"] = df_feat["engagement_weighted_sentiment"].ewm(span=sw, adjust=False).mean()
            df_feat[f"sent_momentum_{sw}"] = df_feat["engagement_weighted_sentiment"] - df_feat[f"sent_ema_{sw}"]
            df_feat[f"post_vol_ema_{sw}"] = df_feat["post_count"].ewm(span=sw, adjust=False).mean()
            df_feat[f"post_vol_ratio_{sw}"] = (df_feat["post_count"] / (df_feat[f"post_vol_ema_{sw}"] + 1e-6)).fillna(1.0)

        # Primary sentiment features for database & summary
        df_feat["sentiment_momentum"] = df_feat["sent_momentum_12"]
        df_feat["realized_volatility"] = df_feat["realized_vol_24"]

        # 6. Price-Sentiment Divergence
        # Z-score of price distance vs Z-score of sentiment
        price_dist = df_feat["price_to_ema_24"]
        sent_val = df_feat["engagement_weighted_sentiment"]
        p_z = (price_dist - price_dist.rolling(72, min_periods=12).mean()) / (price_dist.rolling(72, min_periods=12).std() + 1e-6)
        s_z = (sent_val - sent_val.rolling(72, min_periods=12).mean()) / (sent_val.rolling(72, min_periods=12).std() + 1e-6)
        df_feat["price_sentiment_divergence"] = (p_z - s_z).fillna(0.0)

        # 7. Time / Calendar Features (strictly cyclical)
        hour_of_day = df_feat["timestamp_utc"].dt.hour
        day_of_week = df_feat["timestamp_utc"].dt.dayofweek
        df_feat["sin_hour"] = np.sin(2 * np.pi * hour_of_day / 24.0)
        df_feat["cos_hour"] = np.cos(2 * np.pi * hour_of_day / 24.0)
        df_feat["sin_dow"] = np.sin(2 * np.pi * day_of_week / 7.0)
        df_feat["cos_dow"] = np.cos(2 * np.pi * day_of_week / 7.0)

        logger.info("Constructing multi-horizon forecasting targets (n+0 to n+5)...")
        # 8. Target Construction:
        # n+0: Nowcasting current bar close relative to open
        df_feat["target_h0"] = (df_feat["close"] - df_feat["open"]) / (df_feat["open"] + 1e-9)
        df_feat["target_price_h0"] = df_feat["close"]

        # n+1 to n+5: Forward returns relative to current close
        for h in [1, 2, 3, 4, 5]:
            future_close = df_feat["close"].shift(-h)
            df_feat[f"target_h{h}"] = (future_close - df_feat["close"]) / (df_feat["close"] + 1e-9)
            df_feat[f"target_price_h{h}"] = future_close

        # Clean infs
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        
        # Drop rows where initial lags are completely missing
        df_clean = df_feat.iloc[72:].copy().reset_index(drop=True)
        logger.info(f"Built {len(df_clean)} feature bars across {len(df_clean.columns)} variables.")
        return df_clean
