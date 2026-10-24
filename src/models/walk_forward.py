"""
Walk-Forward Time-Series Splitter with Purging and Embargo.
Prevents information leakage across overlapping multi-horizon forward targets.
"""

from typing import Generator, Tuple
import numpy as np
import pandas as pd

class PurgedWalkForwardCV:
    """
    Time-Series Walk-Forward Splitter with Embargo Window.
    Ensures test sets strictly follow training sets and an embargo buffer
    is placed between them to eliminate auto-correlation and target overlap leakage.
    """
    def __init__(self, n_splits: int = 5, min_train_ratio: float = 0.5, embargo_bars: int = 6):
        self.n_splits = n_splits
        self.min_train_ratio = min_train_ratio
        self.embargo_bars = embargo_bars

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(df)
        min_train_size = int(n_samples * self.min_train_ratio)
        remaining_samples = n_samples - min_train_size
        test_size = remaining_samples // self.n_splits

        for split_idx in range(self.n_splits):
            train_end = min_train_size + split_idx * test_size
            test_start = train_end + self.embargo_bars
            test_end = min(n_samples, test_start + test_size)

            if test_start >= n_samples:
                break

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            yield train_indices, test_indices
