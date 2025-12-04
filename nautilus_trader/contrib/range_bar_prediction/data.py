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
Core data structures for the Range Bar Directional Prediction system.

This module contains the fundamental data classes used throughout the system:
- RangeBar: A percentage range bar constructed from trade data
- BarFeatures: Feature vector for directional prediction
- TradeResult: Record of a single executed trade
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RangeBar:
    """
    A single percentage range bar constructed from aggTrades.

    The bar closes when price moves >= r percent from the opening price.
    Direction is +1 for up bars (close > open) and -1 for down bars.

    Parameters
    ----------
    open_price : float
        The opening price of the bar.
    high_price : float
        The highest price during the bar.
    low_price : float
        The lowest price during the bar.
    close_price : float
        The closing price of the bar (the price that breached the threshold).
    volume : float
        Total volume traded during the bar.
    buy_volume : float
        Volume from buy aggressors (taker buys).
    trade_count : int
        Number of trades in this bar.
    open_time_ms : int
        Timestamp of first trade in milliseconds.
    close_time_ms : int
        Timestamp of closing trade in milliseconds.
    direction : int
        Bar direction: +1 for up, -1 for down.
    r_pct : float
        The r value (percentage threshold) used to construct this bar.
    """

    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    buy_volume: float
    trade_count: int
    open_time_ms: int
    close_time_ms: int
    direction: int
    r_pct: float

    @property
    def duration_seconds(self) -> float:
        """Time elapsed to complete this bar in seconds."""
        return (self.close_time_ms - self.open_time_ms) / 1000.0

    @property
    def volume_imbalance(self) -> float:
        """
        Normalized buy/sell imbalance.

        Returns
        -------
        float
            +1 means all buy aggression, -1 means all sell aggression,
            0 means balanced.
        """
        if self.volume == 0:
            return 0.0
        return (2 * self.buy_volume - self.volume) / self.volume

    @property
    def price_efficiency(self) -> float:
        """
        How directly price moved from open to close.

        Returns
        -------
        float
            1.0 means no wicks (price moved directly), lower values mean
            more back-and-forth before the bar closed.
        """
        total_range = self.high_price - self.low_price
        if total_range == 0:
            return 1.0
        directional_move = abs(self.close_price - self.open_price)
        return directional_move / total_range

    @property
    def sell_volume(self) -> float:
        """Volume from sell aggressors (taker sells)."""
        return self.volume - self.buy_volume


@dataclass
class BarFeatures:
    """
    Feature vector for directional prediction, computed at bar completion.

    These features are available before the next bar's direction is known,
    making them valid for prediction. The target (next_direction) is filled
    in after the next bar completes.

    Parameters
    ----------
    duration_seconds : float
        Time to complete the bar.
    volume_imbalance : float
        Normalized buy/sell volume imbalance [-1, 1].
    price_efficiency : float
        How directly price moved from open to close [0, 1].
    trade_count : int
        Number of trades in the bar.
    prev_direction : int
        Direction of bar N-1.
    prev_2_direction : int
        Direction of bar N-2.
    direction_streak : int
        Consecutive bars in same direction.
    coarse_bar_direction : int
        Direction of most recent 2x r bar.
    coarse_bar_age : int
        How many primary bars since coarse bar closed.
    duration_percentile : float
        Percentile of duration vs rolling window [0, 1].
    bar_direction : int
        Direction of the current bar (for training data).
    bar_close_time_ms : int
        Close time of the bar for timestamp reference.
    next_direction : Optional[int]
        Target: direction of the next bar. None until known.
    """

    duration_seconds: float
    volume_imbalance: float
    price_efficiency: float
    trade_count: int
    prev_direction: int
    prev_2_direction: int
    direction_streak: int
    coarse_bar_direction: int
    coarse_bar_age: int
    duration_percentile: float
    bar_direction: int = 0
    bar_close_time_ms: int = 0
    next_direction: Optional[int] = None

    # Feature names for the ML model (excluding bar metadata and target)
    FEATURE_NAMES: list[str] = field(default_factory=lambda: [
        "duration_seconds",
        "volume_imbalance",
        "price_efficiency",
        "trade_count",
        "prev_direction",
        "prev_2_direction",
        "direction_streak",
        "coarse_bar_direction",
        "coarse_bar_age",
        "duration_percentile",
    ])

    def to_array(self) -> np.ndarray:
        """
        Convert features to numpy array for sklearn, excluding target.

        Returns
        -------
        np.ndarray
            Array of feature values in consistent order.
        """
        return np.array([
            self.duration_seconds,
            self.volume_imbalance,
            self.price_efficiency,
            self.trade_count,
            self.prev_direction,
            self.prev_2_direction,
            self.direction_streak,
            self.coarse_bar_direction,
            self.coarse_bar_age,
            self.duration_percentile,
        ])

    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Return the list of feature names in array order."""
        return [
            "duration_seconds",
            "volume_imbalance",
            "price_efficiency",
            "trade_count",
            "prev_direction",
            "prev_2_direction",
            "direction_streak",
            "coarse_bar_direction",
            "coarse_bar_age",
            "duration_percentile",
        ]


@dataclass
class TradeResult:
    """
    Record of a single executed trade for backtest analysis.

    Parameters
    ----------
    entry_time_ms : int
        Entry timestamp in milliseconds.
    exit_time_ms : int
        Exit timestamp in milliseconds.
    direction : int
        Trade direction: +1 long, -1 short.
    entry_price : float
        Price at entry.
    exit_price : float
        Price at exit.
    predicted_prob : float
        Predicted probability for the direction we bet on.
    actual_outcome : int
        +1 if prediction was correct, -1 if wrong.
    gross_pnl_pct : float
        PnL before transaction costs as percentage.
    net_pnl_pct : float
        PnL after transaction costs as percentage.
    """

    entry_time_ms: int
    exit_time_ms: int
    direction: int
    entry_price: float
    exit_price: float
    predicted_prob: float
    actual_outcome: int
    gross_pnl_pct: float
    net_pnl_pct: float

    @property
    def is_winner(self) -> bool:
        """Whether this trade was profitable after costs."""
        return self.net_pnl_pct > 0

    @property
    def hold_time_seconds(self) -> float:
        """Duration of the trade in seconds."""
        return (self.exit_time_ms - self.entry_time_ms) / 1000.0
