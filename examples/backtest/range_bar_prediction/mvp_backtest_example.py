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
MVP Backtest Example: Percentage Range Bar Directional Prediction

This script demonstrates the complete MVP backtest workflow for testing
whether directional prediction on percentage range bars yields edge
after transaction costs.

Usage:
    # With synthetic data (for testing):
    python mvp_backtest_example.py --synthetic

    # With local data file:
    python mvp_backtest_example.py --file /path/to/aggtrades.jsonl

    # Fetch from Binance API (slow, use for small date ranges):
    python mvp_backtest_example.py --symbol BTCUSDT --start 2024-11-01 --end 2024-11-03

Example Output:
    Processing trades and building bars...
      Built 1000 bars from 150,000 trades...
      Built 2000 bars from 300,000 trades...
    Total bars built: 2500 from 375,000 trades

    Training on 1600 samples...
    Training metrics: accuracy=0.5312, brier=0.2498
    Validation metrics: accuracy=0.5180, brier=0.2501

    Running backtest on validation period...
    Backtest: 127 trades, win_rate=0.5118, total_net_pnl=-0.0032%

    ============================================================
    NO EDGE AFTER COSTS: -0.0025% per trade
    Hypothesis not supported for this r value and feature set
    ============================================================
"""

import argparse
from datetime import datetime
from pprint import pprint

from nautilus_trader.contrib.range_bar_prediction import (
    run_backtest,
    load_aggtrades_from_file,
    fetch_aggtrades_binance,
)
from nautilus_trader.contrib.range_bar_prediction.data_utils import (
    generate_synthetic_trades,
)


def main():
    parser = argparse.ArgumentParser(
        description="MVP Backtest: Range Bar Directional Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source options (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )
    data_group.add_argument(
        "--file",
        type=str,
        help="Path to aggTrades data file (JSON, CSV, or ZIP)",
    )
    data_group.add_argument(
        "--symbol",
        type=str,
        help="Binance symbol to fetch (e.g., BTCUSDT)",
    )

    # API fetch options
    parser.add_argument(
        "--start",
        type=str,
        help="Start date for API fetch (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date for API fetch (YYYY-MM-DD)",
    )

    # Backtest parameters
    parser.add_argument(
        "--primary-r",
        type=float,
        default=0.001,
        help="Primary range bar threshold (default: 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--coarse-r",
        type=float,
        default=0.002,
        help="Coarse range bar threshold (default: 0.002 = 0.2%%)",
    )
    parser.add_argument(
        "--train-bars",
        type=int,
        default=2000,
        help="Number of bars for training (default: 2000)",
    )
    parser.add_argument(
        "--val-bars",
        type=int,
        default=500,
        help="Number of bars for validation (default: 500)",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.55,
        help="Probability threshold for trading (default: 0.55)",
    )
    parser.add_argument(
        "--tx-cost",
        type=float,
        default=0.0005,
        help="Transaction cost (default: 0.0005 = 5 bps)",
    )

    # Synthetic data options
    parser.add_argument(
        "--n-trades",
        type=int,
        default=500000,
        help="Number of synthetic trades (default: 500000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data (default: 42)",
    )

    # Output options
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Get data source
    if args.synthetic:
        print("=" * 60)
        print("Using SYNTHETIC data for testing")
        print("This tests the system mechanics, not real market edge")
        print("=" * 60)
        print()

        trades = generate_synthetic_trades(
            n_trades=args.n_trades,
            seed=args.seed,
        )

    elif args.file:
        print(f"Loading data from: {args.file}")
        trades = load_aggtrades_from_file(args.file)

    else:
        # API fetch
        if not args.start or not args.end:
            parser.error("--start and --end required when using --symbol")

        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")

        print(f"Fetching {args.symbol} from {args.start} to {args.end}")
        print("Note: This may take a while for large date ranges")
        print()

        trades = fetch_aggtrades_binance(
            symbol=args.symbol,
            start_time_ms=int(start_dt.timestamp() * 1000),
            end_time_ms=int(end_dt.timestamp() * 1000),
            verbose=not args.quiet,
        )

    # Run backtest
    results = run_backtest(
        trades_iterator=trades,
        primary_r=args.primary_r,
        coarse_r=args.coarse_r,
        train_bars=args.train_bars,
        validation_bars=args.val_bars,
        prob_threshold=args.prob_threshold,
        tx_cost=args.tx_cost,
        verbose=not args.quiet,
    )

    # Print detailed results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)

    if "error" in results:
        print(f"\nError: {results['error']}")
        return 1

    print("\nConfiguration:")
    print(f"  Primary r: {results['primary_r']:.4%}")
    print(f"  Coarse r: {results['coarse_r']:.4%}")
    print(f"  Total bars: {results['n_total_bars']}")
    print(f"  Trades processed: {results['n_trades_processed']:,}")

    print("\nTraining Metrics:")
    train = results["train_metrics"]
    print(f"  Samples: {train['n_samples']}")
    print(f"  Accuracy: {train['accuracy']:.4f}")
    print(f"  Edge: {train['edge']:.4f}")
    print(f"  Brier Score: {train['brier_score']:.4f}")
    print(f"  Class Balance: {train['class_balance']:.4f}")

    print("\nValidation Metrics:")
    val = results["validation_metrics"]
    print(f"  Samples: {val['n_samples']}")
    print(f"  Accuracy: {val['accuracy']:.4f}")
    print(f"  Edge: {val['edge']:.4f}")
    print(f"  Brier Score: {val['brier_score']:.4f}")
    print(f"  Class Balance: {val['class_balance']:.4f}")

    print("\nBacktest Summary:")
    bt = results["backtest_summary"]
    if bt["n_trades"] > 0:
        print(f"  Total Trades: {bt['n_trades']}")
        print(f"  Win Rate: {bt['win_rate']:.4f}")
        print(f"  Mean Gross PnL: {bt['mean_gross_pnl_pct']:.4%}")
        print(f"  Mean Net PnL: {bt['mean_net_pnl_pct']:.4%}")
        print(f"  Total Net PnL: {bt['total_net_pnl_pct']:.4%}")
        print(f"  Sharpe (per trade): {bt['sharpe_per_trade']:.4f}")
        print(f"  Max Drawdown (trade): {bt['max_drawdown_trade']:.4%}")
        print(f"  Best Trade: {bt['best_trade']:.4%}")
        print(f"  Signal Rate: {bt['signal_rate']:.4f}")
    else:
        print("  No trades executed (probability threshold too high?)")

    if results.get("feature_importance"):
        print("\nFeature Importance:")
        importance = results["feature_importance"]
        for name, score in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {name}: {score:.4f}")

    # Final verdict
    print("\n" + "=" * 60)
    mean_net = bt.get("mean_net_pnl_pct", 0) or 0
    if mean_net > 0:
        print("VERDICT: Preliminary positive edge detected")
        print("Next steps:")
        print("  1. Test on additional out-of-sample periods")
        print("  2. Vary r values and probability thresholds")
        print("  3. Consider paper trading before live capital")
    else:
        print("VERDICT: No edge after transaction costs")
        print("Next steps:")
        print("  1. Try different r values (0.15%, 0.2%)")
        print("  2. Increase probability threshold (0.58%, 0.60%)")
        print("  3. Consider additional features")
        print("  4. Test on different market conditions")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
