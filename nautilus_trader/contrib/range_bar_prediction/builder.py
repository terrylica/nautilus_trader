# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2023 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

"""
Range bar construction and feature engineering components.

This module provides:
- RangeBarBuilder: Constructs percentage range bars from a stream of trades
- FeatureEngine: Computes features for directional prediction from completed bars
"""

from collections import deque
from typing import Optional

import numpy as np

from nautilus_trader.contrib.range_bar_prediction.data import RangeBar, BarFeatures


class RangeBarBuilder:
    """
    Constructs percentage range bars from a stream of aggTrades.

    Maintains state for one r value. For cross-scale features, instantiate
    multiple builders with different r values.

    Design decisions (per specification):
    - First breach wins (no waiting for bar to "settle")
    - Overshoot absorbed (no synthetic splits)
    - Next bar opens at first trade after previous close (gaps permitted)

    Parameters
    ----------
    r_pct : float
        Percentage threshold for bar completion (e.g., 0.001 for 0.1%).

    Examples
    --------
    >>> builder = RangeBarBuilder(r_pct=0.001)  # 0.1% bars
    >>> for trade in trades:
    ...     bar = builder.process_trade(trade)
    ...     if bar is not None:
    ...         print(f"Bar completed: {bar.direction}")
    """

    def __init__(self, r_pct: float):
        if r_pct <= 0:
            raise ValueError(f"r_pct must be positive, got {r_pct}")

        self.r_pct = r_pct
        self._reset_bar()

    def _reset_bar(self) -> None:
        """Clear state for a new bar. Next trade will establish the open."""
        self.bar_open: Optional[float] = None
        self.bar_high: Optional[float] = None
        self.bar_low: Optional[float] = None
        self.bar_volume: float = 0.0
        self.bar_buy_volume: float = 0.0
        self.bar_trade_count: int = 0
        self.bar_open_time: Optional[int] = None

    def process_trade(self, trade: dict) -> Optional[RangeBar]:
        """
        Process a single aggTrade. Returns a completed RangeBar if the
        threshold was breached, otherwise returns None.

        Parameters
        ----------
        trade : dict
            Expected format (Binance aggTrades):
            {
                'p': price (string or float),
                'q': quantity (string or float),
                'T': timestamp in milliseconds,
                'm': is_buyer_maker (bool) - True means sell aggressor
            }

        Returns
        -------
        Optional[RangeBar]
            A completed bar if threshold was breached, None otherwise.
        """
        price = float(trade["p"])
        qty = float(trade["q"])
        timestamp = trade["T"]
        is_sell_aggressor = trade["m"]

        # If no bar in progress, this trade opens a new bar
        if self.bar_open is None:
            self.bar_open = price
            self.bar_high = price
            self.bar_low = price
            self.bar_open_time = timestamp

        # Update running statistics
        self.bar_high = max(self.bar_high, price)
        self.bar_low = min(self.bar_low, price)
        self.bar_volume += qty
        if not is_sell_aggressor:
            self.bar_buy_volume += qty
        self.bar_trade_count += 1

        # Check if threshold breached
        pct_move = abs(price - self.bar_open) / self.bar_open

        if pct_move >= self.r_pct:
            # Bar completes - construct the bar object
            direction = 1 if price > self.bar_open else -1

            completed_bar = RangeBar(
                open_price=self.bar_open,
                high_price=self.bar_high,
                low_price=self.bar_low,
                close_price=price,
                volume=self.bar_volume,
                buy_volume=self.bar_buy_volume,
                trade_count=self.bar_trade_count,
                open_time_ms=self.bar_open_time,
                close_time_ms=timestamp,
                direction=direction,
                r_pct=self.r_pct,
            )

            # Reset for next bar (will open on next trade)
            self._reset_bar()

            return completed_bar

        return None

    def process_trade_tick(
        self,
        price: float,
        size: float,
        timestamp_ns: int,
        aggressor_side: int,
    ) -> Optional[RangeBar]:
        """
        Process a NautilusTrader TradeTick format.

        Parameters
        ----------
        price : float
            Trade price.
        size : float
            Trade size/quantity.
        timestamp_ns : int
            Timestamp in nanoseconds.
        aggressor_side : int
            1 for buyer aggressor, 2 for seller aggressor.

        Returns
        -------
        Optional[RangeBar]
            A completed bar if threshold was breached, None otherwise.
        """
        # Convert to aggTrades format
        trade = {
            "p": price,
            "q": size,
            "T": timestamp_ns // 1_000_000,  # Convert ns to ms
            "m": aggressor_side == 2,  # m=True means sell aggressor
        }
        return self.process_trade(trade)

    @property
    def is_bar_in_progress(self) -> bool:
        """Whether a bar is currently being built."""
        return self.bar_open is not None

    @property
    def current_bar_progress(self) -> Optional[float]:
        """
        Current progress toward bar completion as a fraction of r_pct.

        Returns None if no bar is in progress.
        """
        if self.bar_open is None or self.bar_high is None or self.bar_low is None:
            return None

        # Check distance from open to current extremes
        up_move = (self.bar_high - self.bar_open) / self.bar_open
        down_move = (self.bar_open - self.bar_low) / self.bar_open

        max_move = max(up_move, down_move)
        return max_move / self.r_pct


class FeatureEngine:
    """
    Computes features for directional prediction from completed range bars.

    Maintains rolling history needed for lagged features and percentile
    calculations. Also tracks a coarser-scale bar builder for cross-scale
    features.

    Parameters
    ----------
    primary_r : float
        The r value for primary bar construction (e.g., 0.001 for 0.1%).
    coarse_r : float
        A larger r value (typically 2-3x primary) for cross-scale features.
    duration_lookback : int, default 100
        Number of bars for duration percentile calculation.

    Examples
    --------
    >>> engine = FeatureEngine(primary_r=0.001, coarse_r=0.002)
    >>> for trade in trades:
    ...     result = engine.process_trade(trade)
    ...     if result is not None:
    ...         bar, features = result
    ...         # Use bar and features for prediction
    """

    def __init__(
        self,
        primary_r: float,
        coarse_r: float,
        duration_lookback: int = 100,
    ):
        if coarse_r <= primary_r:
            raise ValueError(
                f"coarse_r ({coarse_r}) must be greater than primary_r ({primary_r})"
            )

        self.primary_builder = RangeBarBuilder(primary_r)
        self.coarse_builder = RangeBarBuilder(coarse_r)

        # Rolling history for features
        self.recent_bars: deque[RangeBar] = deque(maxlen=10)
        self.duration_history: deque[float] = deque(maxlen=duration_lookback)

        # Track coarse bar state
        self.last_coarse_bar: Optional[RangeBar] = None
        self.bars_since_coarse: int = 0

        # Track total bars processed
        self.total_bars: int = 0

    def process_trade(
        self, trade: dict
    ) -> Optional[tuple[RangeBar, BarFeatures]]:
        """
        Process a trade through both builders.

        Parameters
        ----------
        trade : dict
            Trade data in Binance aggTrades format.

        Returns
        -------
        Optional[tuple[RangeBar, BarFeatures]]
            If a primary bar completes, returns (bar, features).
            Otherwise returns None.
        """
        # Always update coarse builder (may complete independently)
        coarse_bar = self.coarse_builder.process_trade(trade)
        if coarse_bar is not None:
            self.last_coarse_bar = coarse_bar
            self.bars_since_coarse = 0

        # Check primary builder
        primary_bar = self.primary_builder.process_trade(trade)
        if primary_bar is None:
            return None

        # Primary bar completed - compute features
        features = self._compute_features(primary_bar)

        # Update state for next bar
        self.recent_bars.append(primary_bar)
        self.duration_history.append(primary_bar.duration_seconds)
        self.bars_since_coarse += 1
        self.total_bars += 1

        return (primary_bar, features)

    def process_trade_tick(
        self,
        price: float,
        size: float,
        timestamp_ns: int,
        aggressor_side: int,
    ) -> Optional[tuple[RangeBar, BarFeatures]]:
        """
        Process a NautilusTrader TradeTick.

        Parameters
        ----------
        price : float
            Trade price.
        size : float
            Trade size/quantity.
        timestamp_ns : int
            Timestamp in nanoseconds.
        aggressor_side : int
            1 for buyer aggressor, 2 for seller aggressor.

        Returns
        -------
        Optional[tuple[RangeBar, BarFeatures]]
            If a primary bar completes, returns (bar, features).
            Otherwise returns None.
        """
        trade = {
            "p": price,
            "q": size,
            "T": timestamp_ns // 1_000_000,
            "m": aggressor_side == 2,
        }
        return self.process_trade(trade)

    def _compute_features(self, bar: RangeBar) -> BarFeatures:
        """
        Extract feature vector from a completed bar plus context.

        Parameters
        ----------
        bar : RangeBar
            The completed bar.

        Returns
        -------
        BarFeatures
            Feature vector for this bar.
        """
        # Lagged directions
        prev_dir = self.recent_bars[-1].direction if len(self.recent_bars) >= 1 else 0
        prev_2_dir = self.recent_bars[-2].direction if len(self.recent_bars) >= 2 else 0

        # Direction streak: count consecutive same-direction bars
        streak = 0
        if len(self.recent_bars) > 0:
            streak_dir = self.recent_bars[-1].direction
            for old_bar in reversed(self.recent_bars):
                if old_bar.direction == streak_dir:
                    streak += 1
                else:
                    break

        # Coarse bar features
        coarse_dir = self.last_coarse_bar.direction if self.last_coarse_bar else 0
        coarse_age = self.bars_since_coarse

        # Duration percentile (how unusual is this bar's duration?)
        if len(self.duration_history) >= 10:
            sorted_durations = sorted(self.duration_history)
            percentile = (
                np.searchsorted(sorted_durations, bar.duration_seconds)
                / len(sorted_durations)
            )
        else:
            percentile = 0.5  # Insufficient history, assume median

        return BarFeatures(
            duration_seconds=bar.duration_seconds,
            volume_imbalance=bar.volume_imbalance,
            price_efficiency=bar.price_efficiency,
            trade_count=bar.trade_count,
            prev_direction=prev_dir,
            prev_2_direction=prev_2_dir,
            direction_streak=streak,
            coarse_bar_direction=coarse_dir,
            coarse_bar_age=coarse_age,
            duration_percentile=percentile,
            bar_direction=bar.direction,
            bar_close_time_ms=bar.close_time_ms,
        )

    def reset(self) -> None:
        """Reset all state for a fresh start."""
        self.primary_builder._reset_bar()
        self.coarse_builder._reset_bar()
        self.recent_bars.clear()
        self.duration_history.clear()
        self.last_coarse_bar = None
        self.bars_since_coarse = 0
        self.total_bars = 0
