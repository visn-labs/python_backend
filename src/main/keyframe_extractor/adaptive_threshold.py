"""Adaptive (online) threshold management for keyframe extraction.

Maintains a sliding window of recent motion scores and derives low/high
thresholds as mean + k*std. Falls back to default thresholds until
sufficient history is accumulated.
"""
from __future__ import annotations
from collections import deque
from math import sqrt
from typing import Deque, Tuple, Optional

class AdaptiveThresholdManager:
    def __init__(
        self,
        window_size: int = 300,
        k_low: float = 1.0,
        k_high: float = 3.0,
        min_history: int = 30,
        default_low: int = 5000,
        default_high: int = 10000,
        smooth_factor: float = 0.0,
    ) -> None:
        """
        window_size: number of most recent motion scores to retain.
        k_low / k_high: multipliers for std dev above mean.
        min_history: minimum samples before adaptive thresholds activate.
        default_low / default_high: used until enough history collected.
        smooth_factor: optional EMA smoothing (0 disables, else 0..1).
        """
        self.window_size = window_size
        self.k_low = k_low
        self.k_high = k_high
        self.min_history = min_history
        self.default_low = default_low
        self.default_high = default_high
        self.smooth_factor = smooth_factor
        self._scores: Deque[int] = deque(maxlen=window_size)
        self._ema_low: Optional[float] = None
        self._ema_high: Optional[float] = None

    def update(self, motion_score: int) -> Tuple[int, int]:
        """Add a new motion score and return (low_threshold, high_threshold)."""
        self._scores.append(motion_score)
        if len(self._scores) < self.min_history:
            # Not enough data, stick with defaults
            return self.default_low, self.default_high

        mean, std = self._mean_std()
        low = int(mean + self.k_low * std)
        high = int(mean + self.k_high * std)
        if high <= low:
            high = low + 1  # ensure ordering

        if self.smooth_factor > 0.0:
            if self._ema_low is None or self._ema_high is None:
                # Initialize both EMAs
                self._ema_low = float(low)
                self._ema_high = float(high)
            else:
                alpha = self.smooth_factor
                self._ema_low = (1 - alpha) * self._ema_low + alpha * low
                self._ema_high = (1 - alpha) * self._ema_high + alpha * high
            # _ema_low/high now guaranteed not None
            return int(self._ema_low), int(self._ema_high)
        return low, high

    def _mean_std(self) -> Tuple[float, float]:
        n = len(self._scores)
        if n == 0:
            return 0.0, 0.0
        s = sum(self._scores)
        mean = s / n
        # population std
        var = sum((x - mean) ** 2 for x in self._scores) / n
        return mean, sqrt(var)

    @property
    def history_length(self) -> int:
        return len(self._scores)

    def current_thresholds(self) -> Tuple[int, int]:
        if len(self._scores) < self.min_history:
            return self.default_low, self.default_high
        mean, std = self._mean_std()
        low = int(mean + self.k_low * std)
        high = int(mean + self.k_high * std)
        if high <= low:
            high = low + 1
        if self.smooth_factor > 0.0 and self._ema_low is not None and self._ema_high is not None:
            return int(self._ema_low), int(self._ema_high)
        return low, high
