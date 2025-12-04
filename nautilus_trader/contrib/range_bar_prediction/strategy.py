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
NautilusTrader strategy integration for range bar directional prediction.

This module provides a complete strategy implementation that:
- Builds percentage range bars from trade ticks
- Extracts features for ML prediction
- Trains a classifier on historical data
- Executes trades based on predictions
"""

from decimal import Decimal
from typing import Optional

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide, AggressorSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy

from nautilus_trader.contrib.range_bar_prediction.data import (
    RangeBar,
    BarFeatures,
)
from nautilus_trader.contrib.range_bar_prediction.builder import FeatureEngine
from nautilus_trader.contrib.range_bar_prediction.classifier import DirectionalClassifier


class RangeBarPredictionStrategyConfig(StrategyConfig, frozen=True):
    """
    Configuration for RangeBarPredictionStrategy instances.

    Parameters
    ----------
    instrument_id : str
        The instrument ID for the strategy (e.g., "ETHUSDT.BINANCE").
    trade_size : Decimal
        The position size per trade.
    primary_r_pct : float, default 0.001
        Primary range bar threshold as percentage (0.001 = 0.1%).
    coarse_r_pct : float, default 0.002
        Coarser range bar threshold for cross-scale features.
    warmup_bars : int, default 500
        Number of bars needed before making predictions.
    retrain_interval : int, default 1000
        Retrain classifier every N bars.
    probability_threshold : float, default 0.55
        Minimum probability to enter a trade.
    min_training_samples : int, default 300
        Minimum samples required to train classifier.
    close_positions_on_stop : bool, default True
        If all open positions should be closed on strategy stop.
    """

    instrument_id: str
    trade_size: Decimal
    primary_r_pct: float = 0.001
    coarse_r_pct: float = 0.002
    warmup_bars: int = 500
    retrain_interval: int = 1000
    probability_threshold: float = 0.55
    min_training_samples: int = 300
    close_positions_on_stop: bool = True


class RangeBarPredictionStrategy(Strategy):
    """
    Range bar directional prediction trading strategy.

    This strategy:
    1. Subscribes to trade ticks for the configured instrument
    2. Builds percentage range bars from incoming ticks
    3. Extracts features from completed bars
    4. Trains a directional classifier periodically
    5. Executes trades when predictions exceed confidence threshold

    The strategy holds positions for exactly one bar (enter at bar close,
    exit at next bar open).

    Parameters
    ----------
    config : RangeBarPredictionStrategyConfig
        The configuration for the strategy instance.

    Notes
    -----
    THIS IS AN EXPERIMENTAL STRATEGY FOR TESTING THE RANGE BAR
    DIRECTIONAL PREDICTION HYPOTHESIS. IT IS NOT INTENDED FOR
    LIVE TRADING WITHOUT EXTENSIVE FURTHER VALIDATION.
    """

    def __init__(self, config: RangeBarPredictionStrategyConfig) -> None:
        PyCondition.true(
            config.coarse_r_pct > config.primary_r_pct,
            f"coarse_r_pct ({config.coarse_r_pct}) must be > primary_r_pct ({config.primary_r_pct})",
        )
        PyCondition.true(
            0.5 <= config.probability_threshold <= 1.0,
            f"probability_threshold must be in [0.5, 1.0], got {config.probability_threshold}",
        )
        super().__init__(config)

        # Configuration
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.trade_size = Decimal(config.trade_size)
        self.primary_r_pct = config.primary_r_pct
        self.coarse_r_pct = config.coarse_r_pct
        self.warmup_bars = config.warmup_bars
        self.retrain_interval = config.retrain_interval
        self.probability_threshold = config.probability_threshold
        self.min_training_samples = config.min_training_samples
        self.close_positions_on_stop = config.close_positions_on_stop

        # Internal state
        self.instrument: Optional[Instrument] = None
        self.feature_engine: Optional[FeatureEngine] = None
        self.classifier: DirectionalClassifier = DirectionalClassifier()

        # Track bars and features for training
        self.all_bars: list[RangeBar] = []
        self.all_features: list[BarFeatures] = []
        self.bars_since_last_train: int = 0

        # Track current prediction state
        self.current_prediction: Optional[dict] = None
        self.pending_entry: bool = False

        # Statistics
        self.total_ticks_processed: int = 0
        self.total_bars_completed: int = 0
        self.total_trades: int = 0
        self.correct_predictions: int = 0

    def on_start(self) -> None:
        """Actions to be performed on strategy start."""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.instrument_id}")
            self.stop()
            return

        # Initialize feature engine
        self.feature_engine = FeatureEngine(
            primary_r=self.primary_r_pct,
            coarse_r=self.coarse_r_pct,
        )

        # Subscribe to trade ticks
        self.subscribe_trade_ticks(self.instrument_id)

        self.log.info(
            f"Started RangeBarPredictionStrategy for {self.instrument_id} "
            f"with primary_r={self.primary_r_pct:.4%}, coarse_r={self.coarse_r_pct:.4%}",
            LogColor.GREEN,
        )

    def on_trade_tick(self, tick: TradeTick) -> None:
        """
        Actions to be performed when the strategy receives a trade tick.

        This is where the main logic happens:
        1. Feed tick to feature engine
        2. If bar completes, handle bar completion
        """
        self.total_ticks_processed += 1

        # Convert TradeTick to format expected by feature engine
        aggressor_side = 1 if tick.aggressor_side == AggressorSide.BUYER else 2

        result = self.feature_engine.process_trade_tick(
            price=float(tick.price),
            size=float(tick.size),
            timestamp_ns=tick.ts_event,
            aggressor_side=aggressor_side,
        )

        if result is not None:
            bar, features = result
            self._on_range_bar_complete(bar, features, tick)

    def _on_range_bar_complete(
        self,
        bar: RangeBar,
        features: BarFeatures,
        tick: TradeTick,
    ) -> None:
        """
        Handle completion of a range bar.

        Parameters
        ----------
        bar : RangeBar
            The completed range bar.
        features : BarFeatures
            Features extracted from the bar.
        tick : TradeTick
            The tick that closed the bar.
        """
        self.total_bars_completed += 1
        self.bars_since_last_train += 1

        # Link previous features to this bar's direction (for training)
        if len(self.all_features) > 0:
            self.all_features[-1].next_direction = bar.direction

        # Store bar and features
        self.all_bars.append(bar)
        self.all_features.append(features)

        # Log bar completion periodically
        if self.total_bars_completed % 100 == 0:
            self.log.info(
                f"Bar #{self.total_bars_completed}: direction={bar.direction}, "
                f"duration={bar.duration_seconds:.1f}s, "
                f"efficiency={bar.price_efficiency:.2f}",
                LogColor.CYAN,
            )

        # Handle any pending entry first
        if self.pending_entry and self.current_prediction is not None:
            self._handle_pending_entry(bar)

        # Check if we have a position to close
        if self.portfolio.is_net_long(self.instrument_id):
            self._check_exit_long(bar)
        elif self.portfolio.is_net_short(self.instrument_id):
            self._check_exit_short(bar)

        # Retrain classifier if needed
        if (
            self.bars_since_last_train >= self.retrain_interval
            and len(self.all_features) >= self.warmup_bars
        ):
            self._retrain_classifier()

        # Make prediction for next bar if classifier is trained
        if self.classifier.is_fitted:
            self._make_prediction(bar, features)

    def _retrain_classifier(self) -> None:
        """Retrain the classifier on accumulated data."""
        # Use all but the last 20% for training
        train_cutoff = int(len(self.all_features) * 0.8)
        train_features = [
            f for f in self.all_features[:train_cutoff] if f.next_direction is not None
        ]

        if len(train_features) < self.min_training_samples:
            self.log.warning(
                f"Insufficient training samples: {len(train_features)} < {self.min_training_samples}"
            )
            return

        success = self.classifier.fit(train_features, min_samples=self.min_training_samples)

        if success:
            # Evaluate on recent data
            eval_features = [
                f for f in self.all_features[train_cutoff:] if f.next_direction is not None
            ]
            if len(eval_features) > 0:
                metrics = self.classifier.evaluate(eval_features)
                self.log.info(
                    f"Classifier retrained: accuracy={metrics['accuracy']:.4f}, "
                    f"brier={metrics['brier_score']:.4f}, n_samples={len(train_features)}",
                    LogColor.GREEN,
                )

            self.bars_since_last_train = 0
        else:
            self.log.warning("Failed to retrain classifier")

    def _make_prediction(self, bar: RangeBar, features: BarFeatures) -> None:
        """Make prediction for next bar direction."""
        p_up = self.classifier.predict_proba(features)

        if p_up > 0.5:
            predicted_dir = 1
            confidence = p_up
        else:
            predicted_dir = -1
            confidence = 1 - p_up

        self.current_prediction = {
            "direction": predicted_dir,
            "confidence": confidence,
            "bar_close_price": bar.close_price,
        }

        # Log high-confidence predictions
        if confidence >= self.probability_threshold:
            direction_str = "LONG" if predicted_dir == 1 else "SHORT"
            self.log.info(
                f"Signal: {direction_str} with confidence {confidence:.4f}",
                LogColor.BLUE,
            )

            # Set pending entry for next bar open
            self.pending_entry = True

    def _handle_pending_entry(self, bar: RangeBar) -> None:
        """Handle entry at bar open after getting signal on previous bar close."""
        if self.current_prediction is None or not self.pending_entry:
            return

        pred = self.current_prediction
        if pred["confidence"] < self.probability_threshold:
            self.pending_entry = False
            return

        # Only enter if flat
        if not self.portfolio.is_flat(self.instrument_id):
            self.pending_entry = False
            return

        # Enter position
        if pred["direction"] == 1:
            self._enter_long()
        else:
            self._enter_short()

        self.pending_entry = False
        self.total_trades += 1

    def _check_exit_long(self, bar: RangeBar) -> None:
        """Check if we should exit long position."""
        # Exit at bar open (we hold for exactly one bar)
        self.close_all_positions(self.instrument_id)

        # Track if prediction was correct
        if self.current_prediction is not None:
            if self.current_prediction["direction"] == 1:
                if bar.direction == 1:
                    self.correct_predictions += 1
                    self.log.info("Prediction CORRECT (long)", LogColor.GREEN)
                else:
                    self.log.info("Prediction WRONG (long)", LogColor.RED)

    def _check_exit_short(self, bar: RangeBar) -> None:
        """Check if we should exit short position."""
        # Exit at bar open (we hold for exactly one bar)
        self.close_all_positions(self.instrument_id)

        # Track if prediction was correct
        if self.current_prediction is not None:
            if self.current_prediction["direction"] == -1:
                if bar.direction == -1:
                    self.correct_predictions += 1
                    self.log.info("Prediction CORRECT (short)", LogColor.GREEN)
                else:
                    self.log.info("Prediction WRONG (short)", LogColor.RED)

    def _enter_long(self) -> None:
        """Enter a long position."""
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)

    def _enter_short(self) -> None:
        """Enter a short position."""
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.trade_size),
        )
        self.submit_order(order)

    def on_data(self, data: Data) -> None:
        """Actions to be performed when the strategy receives generic data."""
        pass

    def on_event(self, event: Event) -> None:
        """Actions to be performed when the strategy receives an event."""
        pass

    def on_stop(self) -> None:
        """Actions to be performed when the strategy is stopped."""
        self.cancel_all_orders(self.instrument_id)
        if self.close_positions_on_stop:
            self.close_all_positions(self.instrument_id)

        # Unsubscribe from data
        self.unsubscribe_trade_ticks(self.instrument_id)

        # Log final statistics
        accuracy = (
            self.correct_predictions / self.total_trades
            if self.total_trades > 0
            else 0.0
        )
        self.log.info(
            f"Strategy stopped. Stats: "
            f"ticks={self.total_ticks_processed:,}, "
            f"bars={self.total_bars_completed}, "
            f"trades={self.total_trades}, "
            f"accuracy={accuracy:.4f}",
            LogColor.YELLOW,
        )

    def on_reset(self) -> None:
        """Actions to be performed when the strategy is reset."""
        if self.feature_engine is not None:
            self.feature_engine.reset()
        self.classifier.reset()
        self.all_bars.clear()
        self.all_features.clear()
        self.bars_since_last_train = 0
        self.current_prediction = None
        self.pending_entry = False
        self.total_ticks_processed = 0
        self.total_bars_completed = 0
        self.total_trades = 0
        self.correct_predictions = 0

    def on_save(self) -> dict[str, bytes]:
        """Actions to be performed when the strategy is saved."""
        # Could serialize classifier state here if needed
        return {}

    def on_load(self, state: dict[str, bytes]) -> None:
        """Actions to be performed when the strategy is loaded."""
        pass

    def on_dispose(self) -> None:
        """Actions to be performed when the strategy is disposed."""
        pass
