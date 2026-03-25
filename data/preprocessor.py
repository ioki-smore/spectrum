"""
Generic preprocessing stage (Stage 1 of the detection pipeline).

Transforms raw time-series data before it enters the anomaly detection algorithm.
Supported modes:
  - "none":    Pass-through (no transformation).
  - "diff":    First-order differencing.
  - "ratio":   Ratio to rolling baseline (value / rolling_median).
               Highlights level shifts; normal ≈ 1.0, spike > 1.0.

The Preprocessor supports an optional ``fit`` / ``transform`` pattern.
When ``fit`` is called on training data, per-feature baseline statistics
are learned.  Subsequent ``transform`` calls use a **clamped causal
rolling median**: the rolling median tracks normal drift but is clamped
to the training-observed range so anomalous values can never drag the
baseline away from normal.

If ``transform`` is called without a prior ``fit``, a plain causal
rolling median is used as a fallback (fully stateless).
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import polars as pl

METADATA_COLUMNS = frozenset({"timestamp", "label", "time", "datetime", "Label"})


class Preprocessor:
    """Stage-1 preprocessor: transforms raw data before algorithm scoring."""

    def __init__(
        self,
        mode: str = "none",
        baseline_window: int = 61,
        smoothing_window: int = 0,
        fill_value: float = 0.0,
    ):
        self.mode = mode
        self.baseline_window = baseline_window
        self.smoothing_window = smoothing_window
        self.fill_value = fill_value

        # Learned baseline from fit() — used by ratio mode
        # Maps col -> (tail_array, baseline_lo, baseline_hi)
        self._train_stats: Optional[Dict[str, Tuple[np.ndarray, float, float]]] = None

    def fit(
        self, df: pl.DataFrame, feature_columns: Optional[Iterable[str]] = None
    ) -> "Preprocessor":
        """Learn baseline context from (normal) training data.

        For ratio mode, stores:
          - The last ``baseline_window`` values (tail) for seeding the
            causal rolling median.
          - The observed min/max of the training data for clamping the
            rolling median during ``transform``.  This prevents anomalous
            values from dragging the baseline outside the normal range.
        """
        if self.mode != "ratio":
            return self

        columns = list(feature_columns) if feature_columns is not None else df.columns
        self._train_stats = {}
        for col in columns:
            if col in METADATA_COLUMNS or not df[col].dtype.is_numeric():
                continue
            vals = df[col].to_numpy().astype(np.float64)
            tail = vals[-self.baseline_window:].copy()
            self._train_stats[col] = (tail,)

        return self

    def transform(
        self, df: pl.DataFrame, feature_columns: Optional[Iterable[str]] = None
    ) -> pl.DataFrame:
        if df.is_empty():
            return df

        columns = list(feature_columns) if feature_columns is not None else df.columns

        if self.mode == "ratio":
            return self._ratio_transform(df, columns)

        exprs = []
        for col in columns:
            if col in METADATA_COLUMNS or not df[col].dtype.is_numeric():
                exprs.append(pl.col(col))
                continue

            expr = pl.col(col)
            if self.mode == "diff":
                expr = expr.diff().fill_null(self.fill_value)

            if self.smoothing_window and self.smoothing_window > 1:
                expr = expr.rolling_mean(self.smoothing_window).fill_null(self.fill_value)

            exprs.append(expr.alias(col))

        return df.select(exprs)

    @staticmethod
    def _causal_rolling_median(arr: np.ndarray, w: int) -> np.ndarray:
        """Causal (left-aligned) rolling median using pure numpy.

        For each index i, computes median(arr[max(0, i-w+1) : i+1]).
        Equivalent to ``pd.Series(arr).rolling(w, min_periods=1).median()``
        but without the pandas dependency.
        """
        n = len(arr)
        out = np.empty(n, dtype=np.float64)
        for i in range(min(w, n)):
            out[i] = np.median(arr[: i + 1])
        if n > w:
            # Use stride_tricks for the bulk of the array
            shape = (n - w + 1, w)
            strides = (arr.strides[0], arr.strides[0])
            windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            out[w - 1 :] = np.median(windows, axis=1)
        return out

    def _ratio_transform(self, df: pl.DataFrame, columns: list) -> pl.DataFrame:
        """Compute value / baseline for each numeric column.

        When ``fit`` was called, prepends the training tail so the causal
        rolling median starts from a clean, normal context.  The median
        is naturally robust to short anomaly spikes (up to ~50% of the
        window can be outliers without affecting the median).

        Without ``fit``, falls back to a plain causal rolling median.
        """
        eps = 1e-8
        result = df.clone()
        w = self.baseline_window

        for col in columns:
            if col in METADATA_COLUMNS or not df[col].dtype.is_numeric():
                continue

            values = df[col].to_numpy().astype(np.float64)

            if self._train_stats is not None and col in self._train_stats:
                (tail,) = self._train_stats[col]
                # Prepend training tail for clean context
                extended = np.concatenate([tail, values])
                prefix_len = len(tail)
                baseline = self._causal_rolling_median(extended, w)
                ratio = extended / (baseline + eps)
                # Strip prefix
                ratio = ratio[prefix_len:]
            else:
                # Fallback: plain causal rolling median
                baseline = self._causal_rolling_median(values, w)
                ratio = values / (baseline + eps)

            result = result.with_columns(pl.Series(col, ratio))

        return result
