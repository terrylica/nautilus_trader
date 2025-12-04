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
Minimal Viable Implementation: Percentage Range Bar Directional Prediction

This module provides a complete system for testing whether directional prediction
on percentage range bars yields edge after transaction costs.

Key Components:
- RangeBar: Data structure for percentage range bars
- RangeBarBuilder: Constructs range bars from trade ticks
- FeatureEngine: Extracts features for ML prediction
- DirectionalClassifier: Logistic regression with calibration
- MVPBacktestEngine: Simulates trading with transaction costs
- RangeBarPredictionStrategy: NautilusTrader strategy integration
"""

from nautilus_trader.contrib.range_bar_prediction.data import (
    RangeBar,
    BarFeatures,
    TradeResult,
)
from nautilus_trader.contrib.range_bar_prediction.builder import (
    RangeBarBuilder,
    FeatureEngine,
)
from nautilus_trader.contrib.range_bar_prediction.classifier import (
    DirectionalClassifier,
)
from nautilus_trader.contrib.range_bar_prediction.backtest import (
    MVPBacktestEngine,
    run_backtest,
)
from nautilus_trader.contrib.range_bar_prediction.strategy import (
    RangeBarPredictionStrategy,
    RangeBarPredictionStrategyConfig,
)
from nautilus_trader.contrib.range_bar_prediction.data_utils import (
    load_aggtrades_from_file,
    fetch_aggtrades_binance,
)


__all__ = [
    # Data structures
    "RangeBar",
    "BarFeatures",
    "TradeResult",
    # Builders
    "RangeBarBuilder",
    "FeatureEngine",
    # Classifier
    "DirectionalClassifier",
    # Backtest
    "MVPBacktestEngine",
    "run_backtest",
    # Strategy
    "RangeBarPredictionStrategy",
    "RangeBarPredictionStrategyConfig",
    # Data utilities
    "load_aggtrades_from_file",
    "fetch_aggtrades_binance",
]
