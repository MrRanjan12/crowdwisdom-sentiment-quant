"""
Sentiment Analysis Engine for CrowdWisdomTrading Quant System.
Scores tweets using an enhanced financial VADER lexicon and calculates
engagement-weighted sentiment, dispersion, and bull/bear ratios.
"""

import numpy as np
import pandas as pd
from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self._inject_financial_crypto_lexicon()

    def _inject_financial_crypto_lexicon(self):
        """Augments VADER with specialized crypto/financial market slang & signals."""
        financial_lexicon = {
            "bullish": 2.5,
            "bearish": -2.5,
            "breakout": 2.2,
            "breakdown": -2.2,
            "ath": 2.8,
            "all-time high": 2.8,
            "liquidation": -2.0,
            "squeeze": 1.8,
            "short squeeze": 2.4,
            "long squeeze": -2.4,
            "dump": -2.6,
            "dumping": -2.6,
            "pump": 2.0,
            "pumping": 2.0,
            "fud": -2.0,
            "rekt": -3.0,
            "hodl": 1.5,
            "hodling": 1.5,
            "accumulation": 2.0,
            "distribution": -2.0,
            "inflows": 1.8,
            "outflows": -1.8,
            "whales buying": 2.5,
            "whales selling": -2.5,
            "oversold": 1.2,
            "overbought": -1.2,
            "dovish": 1.5,
            "hawkish": -1.5,
            "moon": 2.0,
            "support held": 1.8,
            "resistance broken": 2.0,
            "scam": -3.0
        }
        self.analyzer.lexicon.update(financial_lexicon)

    def score_text(self, text: str) -> float:
        """Computes normalized compound sentiment in [-1.0, 1.0]."""
        if not text or not isinstance(text, str):
            return 0.0
        return self.analyzer.polarity_scores(text)["compound"]

    def compute_engagement_weight(
        self,
        likes: int,
        retweets: int,
        views: int,
        followers: int,
        alpha: float = 0.1,
        beta: float = 2.0
    ) -> float:
        """
        Calculates logarithmic engagement weight:
        w_i = ln(1 + likes + beta*retweets + alpha*views) * ln(10 + followers)
        """
        interaction_score = 1.0 + max(0, likes) + beta * max(0, retweets) + alpha * max(0, views)
        follower_score = max(10, followers)
        return float(np.log1p(interaction_score) * np.log1p(follower_score))

    def aggregate_hourly_sentiment(self, df_tweets: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
        """
        Aggregates individual posts into time bars aligned to UTC.
        Computes:
        - raw sentiment
        - engagement-weighted sentiment
        - sentiment dispersion (polarization)
        - post volume
        - bull/bear counts and ratios
        """
        if df_tweets.empty:
            logger.warning("No tweets provided for aggregation.")
            return pd.DataFrame()

        logger.info(f"Scoring {len(df_tweets)} tweets with Financial VADER...")
        df = df_tweets.copy()
        df["sentiment_raw"] = df["text"].apply(self.score_text)
        
        # Calculate engagement weights
        weights = [
            self.compute_engagement_weight(l, r, v, f)
            for l, r, v, f in zip(
                df["likes"].fillna(0),
                df["retweets"].fillna(0),
                df["views"].fillna(0),
                df["author_followers"].fillna(0)
            )
        ]
        df["weight"] = weights
        df["weighted_sentiment_val"] = df["sentiment_raw"] * df["weight"]

        # Timestamp floor to hourly bar
        df["bar_timestamp"] = df["timestamp_utc"].dt.floor(freq)
        
        # Groupby bar_timestamp
        grouped = df.groupby(["bar_timestamp", "asset_symbol"])

        records = []
        for (ts, symbol), group in grouped:
            total_posts = len(group)
            total_weight = group["weight"].sum()
            mean_sent = group["sentiment_raw"].mean()
            weighted_sent = group["weighted_sentiment_val"].sum() / (total_weight + 1e-6)
            dispersion = group["sentiment_raw"].std()
            if np.isnan(dispersion):
                dispersion = 0.0

            bulls = (group["sentiment_raw"] > 0.05).sum()
            bears = (group["sentiment_raw"] < -0.05).sum()
            bull_ratio = bulls / (bulls + bears + 1e-6)

            records.append({
                "timestamp_utc": ts,
                "asset_symbol": symbol,
                "post_count": total_posts,
                "sentiment_score": float(mean_sent),
                "engagement_weighted_sentiment": float(weighted_sent),
                "sentiment_dispersion": float(dispersion),
                "bull_count": int(bulls),
                "bear_count": int(bears),
                "bull_ratio": float(bull_ratio)
            })

        df_agg = pd.DataFrame(records)
        df_agg = df_agg.sort_values("timestamp_utc").reset_index(drop=True)
        logger.info(f"Aggregated sentiment into {len(df_agg)} hourly bars.")
        return df_agg
