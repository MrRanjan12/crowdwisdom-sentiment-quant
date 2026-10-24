"""
Multi-Horizon Predictive Model (n+0, n+1, n+2, n+3, n+4, n+5) for CrowdWisdomTrading.
Uses LightGBM / Gradient Boosting Regressors trained strictly on Sentiment and Price features.
Executes Purged Walk-Forward Cross-Validation to generate out-of-sample forecasts.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .walk_forward import PurgedWalkForwardCV


class MultiHorizonForecaster:
    def __init__(
        self,
        horizons: list[int] | None = None,
        n_splits: int = 5,
        embargo_bars: int = 6
    ):
        if horizons is None:
            horizons = [0, 1, 2, 3, 4, 5]
        self.horizons = horizons
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
        self.models: dict[int, Any] = {}
        self.feature_names: list[str] = []
        self.feature_importances: dict[int, pd.Series] = {}

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Filters strictly sentiment and price features."""
        exclude_cols = {
            "timestamp_utc", "asset_symbol", "bar_timestamp", "open", "high", "low", "close", "volume",
            "bull_count", "bear_count", "sentiment_raw", "weight", "weighted_sentiment_val"
        }
        for h in range(10):
            exclude_cols.add(f"target_h{h}")
            exclude_cols.add(f"target_price_h{h}")

        feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        return feature_cols

    def _create_regressor(self):
        """Initializes LightGBM Regressor with scikit-learn fallback."""
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=120,
                learning_rate=0.03,
                num_leaves=25,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        except Exception:  # noqa: BLE001
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.03,
                max_leaf_nodes=25,
                random_state=42
            )

    def train_walk_forward(self, df_features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Runs Walk-Forward Validation across all prediction horizons (n+0 to n+5).
        Returns out-of-sample predictions DataFrame and comprehensive evaluation metrics.
        """
        logger.info(f"Starting Multi-Horizon Purged Walk-Forward Training for horizons: {self.horizons}...")
        self.feature_names = self._get_feature_columns(df_features)
        logger.info(f"Using {len(self.feature_names)} features strictly derived from sentiment and price.")

        cv = PurgedWalkForwardCV(n_splits=self.n_splits, min_train_ratio=0.5, embargo_bars=self.embargo_bars)
        
        # DataFrame to store out-of-sample predictions
        df_oos = df_features.copy()
        for h in self.horizons:
            df_oos[f"pred_h{h}"] = np.nan
            df_oos[f"pred_price_h{h}"] = np.nan

        metrics_summary = {}

        for h in self.horizons:
            target_col = f"target_h{h}"
            # Valid rows for this horizon
            valid_mask = df_features[target_col].notna() & (~np.isinf(df_features[target_col]))
            df_h = df_features[valid_mask].copy().reset_index(drop=True)

            y_actual_all = []
            y_pred_all = []
            importances_list = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df_h)):
                X_train = df_h.loc[train_idx, self.feature_names]
                y_train = df_h.loc[train_idx, target_col]
                X_test = df_h.loc[test_idx, self.feature_names]
                y_test = df_h.loc[test_idx, target_col]

                model = self._create_regressor()
                model.fit(X_train, y_train)

                preds = model.predict(X_test)
                
                # Map back to df_oos by timestamp
                test_timestamps = df_h.loc[test_idx, "timestamp_utc"]
                df_oos.loc[df_oos["timestamp_utc"].isin(test_timestamps), f"pred_h{h}"] = preds
                
                # Calculate predicted price: P_t * (1 + pred_return)
                current_prices = df_oos.loc[df_oos["timestamp_utc"].isin(test_timestamps), "close"]
                df_oos.loc[df_oos["timestamp_utc"].isin(test_timestamps), f"pred_price_h{h}"] = current_prices * (1.0 + preds)

                y_actual_all.extend(y_test)
                y_pred_all.extend(preds)

                if hasattr(model, "feature_importances_"):
                    importances_list.append(model.feature_importances_)

            y_actual_arr = np.array(y_actual_all)
            y_pred_arr = np.array(y_pred_all)

            if len(y_actual_arr) > 0:
                rmse = float(np.sqrt(mean_squared_error(y_actual_arr, y_pred_arr)))
                mae = float(mean_absolute_error(y_actual_arr, y_pred_arr))
                # Directional hit ratio: % where sign(pred) == sign(actual)
                hit_ratio = float(np.mean(np.sign(y_pred_arr) == np.sign(y_actual_arr))) * 100.0
                # Rank IC (Spearman correlation)
                rank_ic, p_val = spearmanr(y_actual_arr, y_pred_arr)
                if np.isnan(rank_ic):
                    rank_ic = 0.0

                metrics_summary[f"horizon_n+{h}"] = {
                    "horizon": f"n+{h}",
                    "rmse": rmse,
                    "mae": mae,
                    "hit_ratio_pct": hit_ratio,
                    "rank_ic": float(rank_ic),
                    "ic_pvalue": float(p_val) if not np.isnan(p_val) else 1.0,
                    "test_samples": len(y_actual_arr)
                }

                logger.info(
                    f"Horizon n+{h} Results | Hit Ratio: {hit_ratio:.2f}% | Rank IC: {rank_ic:.4f} | RMSE: {rmse:.5f}"
                )

            # Fit full model for feature importance reporting
            full_model = self._create_regressor()
            full_model.fit(df_h[self.feature_names], df_h[target_col])
            self.models[h] = full_model
            if hasattr(full_model, "feature_importances_"):
                self.feature_importances[h] = pd.Series(full_model.feature_importances_, index=self.feature_names).sort_values(ascending=False)

        # Filter out rows that have no predictions
        df_oos_clean = df_oos.dropna(subset=[f"pred_h{h}" for h in self.horizons]).reset_index(drop=True)
        return df_oos_clean, metrics_summary
