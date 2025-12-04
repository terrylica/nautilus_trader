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
MVP Backtest engine for range bar directional prediction.

This module provides:
- MVPBacktestEngine: Simulates trading based on directional predictions
- run_backtest: Complete backtest orchestration function
"""

from typing import Generator, Optional

import numpy as np

from nautilus_trader.contrib.range_bar_prediction.data import (
    RangeBar,
    BarFeatures,
    TradeResult,
)
from nautilus_trader.contrib.range_bar_prediction.builder import FeatureEngine
from nautilus_trader.contrib.range_bar_prediction.classifier import DirectionalClassifier


class MVPBacktestEngine:
    """
    Simulates trading based on directional predictions.

    Key design choices:
    - Fixed fractional position sizing (not Kelly)
    - Explicit transaction cost modeling
    - Entry only when probability exceeds threshold
    - Exit on next bar completion (hold for exactly one bar)

    Parameters
    ----------
    probability_threshold : float, default 0.55
        Minimum P(predicted direction) to enter a trade.
    transaction_cost_pct : float, default 0.0005
        Round-trip cost as fraction (0.0005 = 0.05% = 5 bps).
    position_size_pct : float, default 0.01
        Fraction of capital to risk per trade.

    Examples
    --------
    >>> engine = MVPBacktestEngine(probability_threshold=0.55, transaction_cost_pct=0.0005)
    >>> for bar, features in bar_stream:
    ...     result = engine.on_bar_complete(bar, features, classifier)
    ...     if result:
    ...         print(f"Closed trade: {result.net_pnl_pct:.4%}")
    >>> summary = engine.get_summary()
    """

    def __init__(
        self,
        probability_threshold: float = 0.55,
        transaction_cost_pct: float = 0.0005,
        position_size_pct: float = 0.01,
    ):
        if not 0.5 <= probability_threshold <= 1.0:
            raise ValueError(
                f"probability_threshold must be in [0.5, 1.0], got {probability_threshold}"
            )
        if transaction_cost_pct < 0:
            raise ValueError(
                f"transaction_cost_pct must be non-negative, got {transaction_cost_pct}"
            )

        self.prob_threshold = probability_threshold
        self.tx_cost = transaction_cost_pct
        self.position_size = position_size_pct

        self.trades: list[TradeResult] = []
        self.current_position: Optional[dict] = None

        # Track statistics
        self.total_signals: int = 0
        self.signals_above_threshold: int = 0

    def on_bar_complete(
        self,
        bar: RangeBar,
        features: BarFeatures,
        classifier: DirectionalClassifier,
    ) -> Optional[TradeResult]:
        """
        Called when a bar completes. Manages position entry/exit.

        The strategy is:
        1. If we have a position, close it at this bar's open
        2. Get prediction for next bar
        3. If confidence >= threshold, enter new position

        Parameters
        ----------
        bar : RangeBar
            The completed bar.
        features : BarFeatures
            Features computed from this bar.
        classifier : DirectionalClassifier
            Trained classifier for predictions.

        Returns
        -------
        Optional[TradeResult]
            TradeResult if a trade was closed, None otherwise.
        """
        result = None

        # If we have a position, close it at this bar's open
        # (which is the first trade after previous bar closed)
        if self.current_position is not None:
            result = self._close_position(bar)

        # Get prediction for next bar
        p_up = classifier.predict_proba(features)
        self.total_signals += 1

        # Determine predicted direction and confidence
        if p_up > 0.5:
            predicted_dir = 1
            confidence = p_up
        else:
            predicted_dir = -1
            confidence = 1 - p_up

        # Only enter if confidence exceeds threshold
        if confidence >= self.prob_threshold:
            self.signals_above_threshold += 1
            self.current_position = {
                "direction": predicted_dir,
                "entry_price": bar.close_price,  # Enter at bar close
                "entry_time": bar.close_time_ms,
                "predicted_prob": confidence,
            }

        return result

    def _close_position(self, bar: RangeBar) -> TradeResult:
        """
        Close current position at bar's open price.

        Parameters
        ----------
        bar : RangeBar
            The bar that just completed (position exits at its open).

        Returns
        -------
        TradeResult
            Record of the closed trade.
        """
        pos = self.current_position

        # Exit at bar's open (first price of new bar)
        exit_price = bar.open_price

        # Calculate PnL
        if pos["direction"] == 1:  # Long
            gross_pnl = (exit_price - pos["entry_price"]) / pos["entry_price"]
        else:  # Short
            gross_pnl = (pos["entry_price"] - exit_price) / pos["entry_price"]

        net_pnl = gross_pnl - self.tx_cost

        # Did we predict correctly?
        actual_move = exit_price - pos["entry_price"]
        if pos["direction"] == 1:
            correct = actual_move > 0
        else:
            correct = actual_move < 0

        result = TradeResult(
            entry_time_ms=pos["entry_time"],
            exit_time_ms=bar.open_time_ms,
            direction=pos["direction"],
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            predicted_prob=pos["predicted_prob"],
            actual_outcome=1 if correct else -1,
            gross_pnl_pct=gross_pnl,
            net_pnl_pct=net_pnl,
        )

        self.trades.append(result)
        self.current_position = None

        return result

    def get_summary(self) -> dict:
        """
        Compute aggregate backtest statistics.

        Returns
        -------
        dict
            Dictionary containing:
            - n_trades: Number of completed trades
            - win_rate: Fraction of correct predictions
            - mean_gross_pnl_pct: Average gross PnL per trade
            - mean_net_pnl_pct: Average net PnL per trade
            - total_net_pnl_pct: Cumulative net PnL
            - sharpe_per_trade: Risk-adjusted return per trade
            - max_drawdown_trade: Worst single trade
            - best_trade: Best single trade
            - signal_rate: Fraction of bars generating trades
        """
        if len(self.trades) == 0:
            return {
                "n_trades": 0,
                "win_rate": None,
                "mean_gross_pnl_pct": None,
                "mean_net_pnl_pct": None,
                "total_net_pnl_pct": 0.0,
                "sharpe_per_trade": None,
                "max_drawdown_trade": None,
                "best_trade": None,
                "signal_rate": 0.0,
            }

        net_pnls = np.array([t.net_pnl_pct for t in self.trades])
        gross_pnls = np.array([t.gross_pnl_pct for t in self.trades])
        outcomes = np.array([t.actual_outcome for t in self.trades])

        win_rate = float(np.mean(outcomes == 1))

        # Sharpe-like ratio (mean / std of per-trade returns)
        mean_return = float(np.mean(net_pnls))
        std_return = float(np.std(net_pnls)) if len(net_pnls) > 1 else 1.0
        sharpe_per_trade = mean_return / std_return if std_return > 0 else 0.0

        # Calculate drawdown
        cumulative_pnl = np.cumsum(net_pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        signal_rate = (
            self.signals_above_threshold / self.total_signals
            if self.total_signals > 0
            else 0.0
        )

        return {
            "n_trades": len(self.trades),
            "win_rate": win_rate,
            "mean_gross_pnl_pct": float(np.mean(gross_pnls)),
            "mean_net_pnl_pct": mean_return,
            "total_net_pnl_pct": float(np.sum(net_pnls)),
            "total_gross_pnl_pct": float(np.sum(gross_pnls)),
            "sharpe_per_trade": sharpe_per_trade,
            "max_drawdown_trade": float(np.min(net_pnls)),
            "best_trade": float(np.max(net_pnls)),
            "max_cumulative_drawdown": max_drawdown,
            "signal_rate": signal_rate,
            "total_signals": self.total_signals,
        }

    def get_equity_curve(self) -> np.ndarray:
        """
        Get the cumulative PnL over time.

        Returns
        -------
        np.ndarray
            Cumulative net PnL after each trade.
        """
        if len(self.trades) == 0:
            return np.array([0.0])

        net_pnls = [t.net_pnl_pct for t in self.trades]
        return np.cumsum(net_pnls)

    def reset(self) -> None:
        """Reset the backtest engine for a new run."""
        self.trades = []
        self.current_position = None
        self.total_signals = 0
        self.signals_above_threshold = 0


def run_backtest(
    trades_iterator: Generator[dict, None, None],
    primary_r: float = 0.001,
    coarse_r: float = 0.002,
    train_bars: int = 2000,
    validation_bars: int = 500,
    prob_threshold: float = 0.55,
    tx_cost: float = 0.0005,
    min_train_samples: int = 500,
    verbose: bool = True,
) -> dict:
    """
    Run complete backtest with train/validation split.

    This function orchestrates the full MVP workflow:
    1. Build range bars from trades
    2. Compute features for each bar
    3. Split into training and validation sets
    4. Train classifier on training data
    5. Evaluate and backtest on validation data

    Parameters
    ----------
    trades_iterator : Generator[dict, None, None]
        Generator yielding Binance aggTrade dicts.
    primary_r : float, default 0.001
        Primary range bar threshold (0.001 = 0.1%).
    coarse_r : float, default 0.002
        Coarser threshold for cross-scale features.
    train_bars : int, default 2000
        Number of bars for training.
    validation_bars : int, default 500
        Number of bars for validation.
    prob_threshold : float, default 0.55
        Minimum probability to enter trade.
    tx_cost : float, default 0.0005
        Round-trip transaction cost.
    min_train_samples : int, default 500
        Minimum samples required to fit classifier.
    verbose : bool, default True
        Whether to print progress updates.

    Returns
    -------
    dict
        Dictionary containing:
        - train_metrics: Training set evaluation
        - validation_metrics: Validation set evaluation
        - backtest_summary: Trading statistics
        - n_total_bars: Total bars built
        - primary_r: Primary r value used
        - coarse_r: Coarse r value used
        - error: Error message if failed
    """
    # Initialize components
    feature_engine = FeatureEngine(primary_r, coarse_r)
    classifier = DirectionalClassifier()
    backtest = MVPBacktestEngine(prob_threshold, tx_cost)

    # Collect bars and features
    all_bars: list[RangeBar] = []
    all_features: list[BarFeatures] = []

    if verbose:
        print("Processing trades and building bars...")

    trade_count = 0
    for trade in trades_iterator:
        result = feature_engine.process_trade(trade)
        trade_count += 1

        if result is not None:
            bar, features = result

            # Link previous features to this bar's direction
            if len(all_features) > 0:
                all_features[-1].next_direction = bar.direction

            all_bars.append(bar)
            all_features.append(features)

            # Progress indicator
            if verbose and len(all_bars) % 1000 == 0:
                print(f"  Built {len(all_bars)} bars from {trade_count:,} trades...")

    if verbose:
        print(f"Total bars built: {len(all_bars)} from {trade_count:,} trades")

    # Verify we have enough data
    total_needed = train_bars + validation_bars
    if len(all_features) < total_needed:
        return {
            "error": f"Insufficient data: {len(all_features)} bars, need {total_needed}",
            "n_total_bars": len(all_bars),
            "n_trades_processed": trade_count,
        }

    # Split data: train on first chunk, validate on second
    train_features = all_features[:train_bars]
    val_features = all_features[train_bars : train_bars + validation_bars]

    # Filter to features with known targets
    train_features_labeled = [f for f in train_features if f.next_direction is not None]
    val_features_labeled = [f for f in val_features if f.next_direction is not None]

    if verbose:
        print(f"\nTraining on {len(train_features_labeled)} samples...")

    # Train classifier
    fit_success = classifier.fit(train_features_labeled, min_samples=min_train_samples)

    if not fit_success:
        return {
            "error": f"Failed to fit classifier - only {len(train_features_labeled)} labeled samples",
            "n_total_bars": len(all_bars),
            "n_trades_processed": trade_count,
        }

    # Evaluate on training data (expect overfitting)
    train_metrics = classifier.evaluate(train_features_labeled)
    if verbose:
        print(f"Training metrics: accuracy={train_metrics['accuracy']:.4f}, "
              f"brier={train_metrics['brier_score']:.4f}")

    # Evaluate on validation data (honest assessment)
    val_metrics = classifier.evaluate(val_features_labeled)
    if verbose:
        print(f"Validation metrics: accuracy={val_metrics['accuracy']:.4f}, "
              f"brier={val_metrics['brier_score']:.4f}")

    # Run backtest on validation period
    if verbose:
        print(f"\nRunning backtest on validation period...")

    val_bars = all_bars[train_bars : train_bars + validation_bars]
    val_feat = all_features[train_bars : train_bars + validation_bars]

    for bar, features in zip(val_bars, val_feat):
        backtest.on_bar_complete(bar, features, classifier)

    backtest_summary = backtest.get_summary()

    if verbose:
        print(f"Backtest: {backtest_summary['n_trades']} trades, "
              f"win_rate={backtest_summary['win_rate']:.4f}, "
              f"total_net_pnl={backtest_summary['total_net_pnl_pct']:.4%}")

    # Key question: does edge survive transaction costs?
    edge_after_costs = backtest_summary.get("mean_net_pnl_pct", 0) or 0

    if verbose:
        print("\n" + "=" * 60)
        if edge_after_costs > 0:
            print(f"POSITIVE EDGE DETECTED: {edge_after_costs * 100:.4f}% per trade")
            print("Proceed with caution - validate on additional out-of-sample data")
        else:
            print(f"NO EDGE AFTER COSTS: {edge_after_costs * 100:.4f}% per trade")
            print("Hypothesis not supported for this r value and feature set")
        print("=" * 60)

    # Get feature importance
    feature_importance = classifier.get_feature_importance()

    return {
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "backtest_summary": backtest_summary,
        "feature_importance": feature_importance,
        "n_total_bars": len(all_bars),
        "n_trades_processed": trade_count,
        "primary_r": primary_r,
        "coarse_r": coarse_r,
        "equity_curve": backtest.get_equity_curve().tolist(),
    }
