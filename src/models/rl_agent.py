"""
Reinforcement Learning Policy Agent for CrowdWisdomTrading Quant System.
Learns dynamic asset allocation policy (Long / Flat / Short) directly from
social sentiment states, momentum, and price signals using Q-Learning.
"""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


class SentimentRLTrader:
    def __init__(
        self,
        alpha: float = 0.1,       # Learning rate
        gamma: float = 0.95,      # Discount factor
        epsilon: float = 0.1,     # Exploration rate
        fee_bps: float = 5.0      # Transaction fee penalty
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.fee = fee_bps / 10000.0
        self.q_table: dict[tuple, np.ndarray] = {}
        # Actions: 0 = Short (-1.0), 1 = Flat (0.0), 2 = Long (+1.0)
        self.action_map = {0: -1.0, 1: 0.0, 2: 1.0}

    def _discretize_state(self, row: pd.Series) -> tuple[int, int, int, int]:
        """
        Discretizes continuous sentiment and price features into a compact state tuple:
        (sentiment_regime, momentum_regime, rsi_regime, divergence_regime)
        """
        # 1. Sentiment: -1 = Bearish, 0 = Neutral, 1 = Bullish
        s = row.get("engagement_weighted_sentiment", 0.0)
        s_state = 0 if abs(s) < 0.05 else (1 if s > 0 else -1)

        # 2. Sentiment Momentum
        sm = row.get("sentiment_momentum", 0.0)
        sm_state = 0 if abs(sm) < 0.02 else (1 if sm > 0 else -1)

        # 3. Price RSI: 0 = Oversold (<35), 1 = Neutral (35-65), 2 = Overbought (>65)
        rsi = row.get("rsi_14", 50.0)
        rsi_state = 0 if rsi < 35 else (2 if rsi > 65 else 1)

        # 4. Price-Sentiment Divergence
        div = row.get("price_sentiment_divergence", 0.0)
        div_state = 0 if abs(div) < 0.5 else (1 if div > 0 else -1)

        return (s_state, sm_state, rsi_state, div_state)

    def _get_q_values(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)
        return self.q_table[state]

    def train_policy(self, df_train: pd.DataFrame, episodes: int = 10) -> dict[str, Any]:
        """Trains the Q-learning policy on the training partition."""
        logger.info(f"Training Sentiment RL Agent across {episodes} episodes ({len(df_train)} steps)...")
        
        for ep in range(episodes):
            prev_action = 1 # Start flat
            for i in range(len(df_train) - 1):
                curr_row = df_train.iloc[i]
                next_row = df_train.iloc[i + 1]
                
                curr_state = self._discretize_state(curr_row)
                next_state = self._discretize_state(next_row)
                
                q_vals = self._get_q_values(curr_state)
                
                # Epsilon-greedy action selection
                if np.random.rand() < self.epsilon:
                    action = np.random.choice([0, 1, 2])
                else:
                    action = np.argmax(q_vals)

                # Realized step return
                fwd_ret = next_row["log_return"]
                pos = self.action_map[action]
                prev_pos = self.action_map[prev_action]
                
                turnover = abs(pos - prev_pos)
                cost = turnover * self.fee
                # Sortino downside penalty
                trade_ret = (pos * fwd_ret) - cost
                downside_penalty = 1.5 * max(0.0, -trade_ret) ** 2
                reward = trade_ret - downside_penalty

                # Bellman update
                next_q = self._get_q_values(next_state)
                target = reward + self.gamma * np.max(next_q)
                self.q_table[curr_state][action] += self.alpha * (target - self.q_table[curr_state][action])
                prev_action = action

        logger.info(f"RL Agent training completed. Q-table contains {len(self.q_table)} discovered states.")
        return {"states_learned": len(self.q_table)}

    def evaluate_policy(self, df_test: pd.DataFrame) -> pd.Series:
        """Evaluates learned policy greedily out-of-sample, returning target positions."""
        actions = []
        for i in range(len(df_test)):
            row = df_test.iloc[i]
            state = self._discretize_state(row)
            q_vals = self._get_q_values(state)
            best_action = np.argmax(q_vals)
            actions.append(self.action_map[best_action])
        return pd.Series(actions, index=df_test.index)
