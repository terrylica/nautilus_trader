#!/usr/bin/env python3
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
NautilusTrader Backtest Example: Range Bar Directional Prediction Strategy

This script demonstrates how to run the RangeBarPredictionStrategy within
the NautilusTrader backtesting framework.

Requirements:
- Historical trade tick data (can be obtained from Binance data portal)
- NautilusTrader properly installed

Usage:
    python nautilus_backtest_example.py

Notes:
    This example uses test data provided by NautilusTrader. For real backtests,
    you'll need to provide your own historical trade tick data.

    THIS IS AN EXPERIMENTAL STRATEGY FOR TESTING PURPOSES ONLY.
    IT IS NOT INTENDED FOR LIVE TRADING WITHOUT EXTENSIVE VALIDATION.
"""

import time
from decimal import Decimal

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.model.currencies import ETH
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import TradeTickDataWrangler
from nautilus_trader.test_kit.providers import TestDataProvider
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from nautilus_trader.contrib.range_bar_prediction.strategy import (
    RangeBarPredictionStrategy,
    RangeBarPredictionStrategyConfig,
)


def main():
    print("=" * 60)
    print("NautilusTrader Range Bar Prediction Backtest")
    print("=" * 60)
    print()

    # Configure backtest engine
    config = BacktestEngineConfig(
        trader_id="BACKTESTER-001",
        logging="INFO",
    )

    # Build the backtest engine
    engine = BacktestEngine(config=config)

    # Add a trading venue
    BINANCE = Venue("BINANCE")
    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,  # Allow both long and short
        base_currency=None,  # Multi-currency account
        starting_balances=[Money(1_000_000.0, USDT), Money(100.0, ETH)],
    )

    # Add instruments
    ETHUSDT_BINANCE = TestInstrumentProvider.ethusdt_binance()
    engine.add_instrument(ETHUSDT_BINANCE)

    # Add data - using NautilusTrader test data
    print("Loading trade tick data...")
    provider = TestDataProvider()
    wrangler = TradeTickDataWrangler(instrument=ETHUSDT_BINANCE)

    try:
        ticks = wrangler.process(provider.read_csv_ticks("binance-ethusdt-trades.csv"))
        print(f"Loaded {len(ticks):,} trade ticks")
    except FileNotFoundError:
        print("Test data not found. Please ensure NautilusTrader test data is available.")
        print("You can download trade tick data from Binance data portal:")
        print("  https://data.binance.vision/")
        return

    engine.add_data(ticks)

    # Configure the Range Bar Prediction strategy
    strategy_config = RangeBarPredictionStrategyConfig(
        instrument_id=str(ETHUSDT_BINANCE.id),
        trade_size=Decimal("0.1"),  # Trade 0.1 ETH per signal
        primary_r_pct=0.001,  # 0.1% range bars
        coarse_r_pct=0.002,  # 0.2% for cross-scale features
        warmup_bars=200,  # Reduced for smaller test dataset
        retrain_interval=500,  # Retrain every 500 bars
        probability_threshold=0.55,  # Only trade when >= 55% confident
        min_training_samples=150,  # Minimum samples to train
        close_positions_on_stop=True,
    )

    # Instantiate and add the strategy
    strategy = RangeBarPredictionStrategy(config=strategy_config)
    engine.add_strategy(strategy=strategy)

    print()
    print("Strategy Configuration:")
    print(f"  Instrument: {strategy_config.instrument_id}")
    print(f"  Trade Size: {strategy_config.trade_size} ETH")
    print(f"  Primary r: {strategy_config.primary_r_pct:.2%}")
    print(f"  Coarse r: {strategy_config.coarse_r_pct:.2%}")
    print(f"  Warmup Bars: {strategy_config.warmup_bars}")
    print(f"  Probability Threshold: {strategy_config.probability_threshold:.2%}")
    print()

    input("Press Enter to start backtest...")

    # Run the backtest
    start_time = time.time()
    engine.run()
    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)
    print(f"Runtime: {elapsed:.2f} seconds")
    print()

    # Generate and display reports
    with pd.option_context(
        "display.max_rows",
        100,
        "display.max_columns",
        None,
        "display.width",
        300,
    ):
        print("Account Report:")
        print("-" * 40)
        print(engine.trader.generate_account_report(BINANCE))
        print()

        print("Order Fills Report:")
        print("-" * 40)
        fills_report = engine.trader.generate_order_fills_report()
        if not fills_report.empty:
            print(fills_report)
        else:
            print("No fills (strategy may not have generated signals)")
        print()

        print("Positions Report:")
        print("-" * 40)
        positions_report = engine.trader.generate_positions_report()
        if not positions_report.empty:
            print(positions_report)
        else:
            print("No positions")
        print()

    # Strategy statistics
    print("Strategy Statistics:")
    print("-" * 40)
    print(f"  Total Ticks Processed: {strategy.total_ticks_processed:,}")
    print(f"  Total Bars Completed: {strategy.total_bars_completed}")
    print(f"  Total Trades: {strategy.total_trades}")
    if strategy.total_trades > 0:
        accuracy = strategy.correct_predictions / strategy.total_trades
        print(f"  Prediction Accuracy: {accuracy:.2%}")
    print()

    # Cleanup
    engine.reset()
    engine.dispose()

    print("Done.")


if __name__ == "__main__":
    main()
