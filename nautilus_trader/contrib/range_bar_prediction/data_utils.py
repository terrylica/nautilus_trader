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
Data loading utilities for range bar prediction.

This module provides functions for loading and fetching aggTrades data
from various sources:
- Local JSON/CSV files
- Binance API (historical data)
- Binance data portal downloads
"""

import gzip
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional
from zipfile import ZipFile

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def load_aggtrades_from_file(filepath: str) -> Generator[dict, None, None]:
    """
    Load aggTrades from a newline-delimited JSON file.

    Parameters
    ----------
    filepath : str
        Path to the data file. Supports:
        - .json/.jsonl - newline-delimited JSON
        - .json.gz/.jsonl.gz - gzip compressed JSON
        - .csv - CSV format

    Yields
    ------
    dict
        Trade data with keys: 'p' (price), 'q' (quantity),
        'T' (timestamp), 'm' (is_buyer_maker).

    Examples
    --------
    >>> for trade in load_aggtrades_from_file("btcusdt_aggtrades.jsonl"):
    ...     print(f"Price: {trade['p']}, Qty: {trade['q']}")
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = "".join(path.suffixes).lower()

    if suffix in (".json.gz", ".jsonl.gz"):
        yield from _load_gzip_json(path)
    elif suffix in (".json", ".jsonl"):
        yield from _load_json(path)
    elif suffix == ".csv":
        yield from _load_csv(path)
    elif suffix == ".csv.gz":
        yield from _load_gzip_csv(path)
    elif suffix == ".zip":
        yield from _load_zip(path)
    else:
        # Try JSON by default
        yield from _load_json(path)


def _load_json(path: Path) -> Generator[dict, None, None]:
    """Load newline-delimited JSON file."""
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_gzip_json(path: Path) -> Generator[dict, None, None]:
    """Load gzip-compressed JSON file."""
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_csv(path: Path) -> Generator[dict, None, None]:
    """Load CSV file (Binance format)."""
    import csv

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        # Handle different column naming conventions
        for row in reader:
            yield _normalize_csv_row(row)


def _load_gzip_csv(path: Path) -> Generator[dict, None, None]:
    """Load gzip-compressed CSV file."""
    import csv
    import io

    with gzip.open(path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield _normalize_csv_row(row)


def _load_zip(path: Path) -> Generator[dict, None, None]:
    """Load from zip archive (Binance data portal format)."""
    import csv
    import io

    with ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".csv"):
                with zf.open(name) as f:
                    reader = csv.reader(io.TextIOWrapper(f))
                    # Skip header if present
                    first_row = next(reader, None)
                    if first_row and not first_row[0].replace(".", "").isdigit():
                        pass  # Was header, skip
                    else:
                        # Process first row
                        if first_row:
                            yield _parse_binance_csv_row(first_row)

                    for row in reader:
                        yield _parse_binance_csv_row(row)


def _normalize_csv_row(row: dict) -> dict:
    """Normalize CSV row to standard format."""
    # Handle different column names
    price = row.get("p") or row.get("price") or row.get("Price")
    qty = row.get("q") or row.get("qty") or row.get("quantity") or row.get("Quantity")
    timestamp = row.get("T") or row.get("time") or row.get("timestamp") or row.get("Timestamp")
    is_buyer_maker = row.get("m") or row.get("is_buyer_maker") or row.get("isBuyerMaker")

    # Convert is_buyer_maker to bool
    if isinstance(is_buyer_maker, str):
        is_buyer_maker = is_buyer_maker.lower() in ("true", "1", "yes")

    return {
        "p": price,
        "q": qty,
        "T": int(timestamp),
        "m": bool(is_buyer_maker),
    }


def _parse_binance_csv_row(row: list) -> dict:
    """
    Parse Binance data portal CSV row.

    Binance aggTrades CSV format:
    agg_trade_id, price, quantity, first_trade_id, last_trade_id,
    transact_time, is_buyer_maker
    """
    return {
        "p": row[1],  # price
        "q": row[2],  # quantity
        "T": int(row[5]),  # transact_time
        "m": row[6].lower() == "true",  # is_buyer_maker
    }


def fetch_aggtrades_binance(
    symbol: str,
    start_time_ms: int,
    end_time_ms: int,
    batch_size: int = 1000,
    rate_limit_sleep: float = 0.1,
    max_retries: int = 3,
    verbose: bool = True,
) -> Generator[dict, None, None]:
    """
    Fetch historical aggTrades from Binance API.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., "BTCUSDT").
    start_time_ms : int
        Start timestamp in milliseconds.
    end_time_ms : int
        End timestamp in milliseconds.
    batch_size : int, default 1000
        Number of trades per API request (max 1000).
    rate_limit_sleep : float, default 0.1
        Sleep time between requests to avoid rate limiting.
    max_retries : int, default 3
        Maximum retry attempts for failed requests.
    verbose : bool, default True
        Whether to print progress updates.

    Yields
    ------
    dict
        Trade data with keys: 'p', 'q', 'T', 'm'.

    Notes
    -----
    For large date ranges, consider using Binance data portal downloads
    instead of the API to avoid rate limiting.

    Examples
    --------
    >>> from datetime import datetime
    >>> start = int(datetime(2024, 11, 1).timestamp() * 1000)
    >>> end = int(datetime(2024, 11, 2).timestamp() * 1000)
    >>> for trade in fetch_aggtrades_binance("BTCUSDT", start, end):
    ...     print(f"Price: {trade['p']}")
    """
    if not HAS_REQUESTS:
        raise ImportError(
            "requests library required for API fetch. "
            "Install with: pip install requests"
        )

    base_url = "https://api.binance.com/api/v3/aggTrades"
    current_start = start_time_ms
    total_trades = 0

    while current_start < end_time_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_time_ms,
            "limit": batch_size,
        }

        # Retry logic
        for attempt in range(max_retries):
            try:
                response = requests.get(base_url, params=params, timeout=30)

                if response.status_code == 429:
                    # Rate limited - wait and retry
                    wait_time = int(response.headers.get("Retry-After", 60))
                    if verbose:
                        print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    if verbose:
                        print(f"API error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        return

                trades = response.json()
                break

            except requests.RequestException as e:
                if verbose:
                    print(f"Request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return
        else:
            # All retries failed
            return

        if len(trades) == 0:
            break

        for trade in trades:
            yield {
                "p": trade["p"],
                "q": trade["q"],
                "T": trade["T"],
                "m": trade["m"],
            }

        total_trades += len(trades)

        # Move start time past last trade
        current_start = trades[-1]["T"] + 1

        # Progress update
        if verbose and total_trades % 10000 == 0:
            progress_pct = (current_start - start_time_ms) / (end_time_ms - start_time_ms) * 100
            print(f"  Fetched {total_trades:,} trades ({progress_pct:.1f}%)")

        # Rate limiting
        time.sleep(rate_limit_sleep)

    if verbose:
        print(f"Completed: fetched {total_trades:,} trades")


def download_binance_aggtrades(
    symbol: str,
    date: datetime,
    output_dir: str = "./data",
    verbose: bool = True,
) -> Optional[str]:
    """
    Download aggTrades from Binance data portal for a specific date.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., "BTCUSDT").
    date : datetime
        Date to download.
    output_dir : str, default "./data"
        Directory to save downloaded file.
    verbose : bool, default True
        Whether to print progress updates.

    Returns
    -------
    Optional[str]
        Path to downloaded file, or None if download failed.

    Notes
    -----
    Downloads from https://data.binance.vision/
    Files are typically available with a 1-2 day delay.

    Examples
    --------
    >>> from datetime import datetime
    >>> path = download_binance_aggtrades("BTCUSDT", datetime(2024, 11, 1))
    >>> if path:
    ...     for trade in load_aggtrades_from_file(path):
    ...         print(trade)
    """
    if not HAS_REQUESTS:
        raise ImportError(
            "requests library required for download. "
            "Install with: pip install requests"
        )

    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-aggTrades-{date_str}.zip"
    url = f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{filename}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / filename

    if output_file.exists():
        if verbose:
            print(f"File already exists: {output_file}")
        return str(output_file)

    if verbose:
        print(f"Downloading: {url}")

    try:
        response = requests.get(url, stream=True, timeout=300)

        if response.status_code == 404:
            if verbose:
                print(f"File not found (may not be available yet): {filename}")
            return None

        if response.status_code != 200:
            if verbose:
                print(f"Download failed: {response.status_code}")
            return None

        # Write to file
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if verbose and total_size > 0:
                    progress = downloaded / total_size * 100
                    print(f"\r  Progress: {progress:.1f}%", end="", flush=True)

        if verbose:
            print(f"\nSaved to: {output_file}")

        return str(output_file)

    except requests.RequestException as e:
        if verbose:
            print(f"Download error: {e}")
        return None


def generate_synthetic_trades(
    n_trades: int = 100000,
    initial_price: float = 50000.0,
    volatility: float = 0.0001,
    seed: int = 42,
) -> Generator[dict, None, None]:
    """
    Generate synthetic aggTrades for testing.

    Parameters
    ----------
    n_trades : int, default 100000
        Number of trades to generate.
    initial_price : float, default 50000.0
        Starting price.
    volatility : float, default 0.0001
        Per-trade price volatility (std dev of returns).
    seed : int, default 42
        Random seed for reproducibility.

    Yields
    ------
    dict
        Synthetic trade data.

    Examples
    --------
    >>> from nautilus_trader.contrib.range_bar_prediction import run_backtest
    >>> trades = generate_synthetic_trades(n_trades=50000)
    >>> results = run_backtest(trades)
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    price = initial_price
    timestamp = int(datetime(2024, 1, 1).timestamp() * 1000)

    for _ in range(n_trades):
        # Random walk for price
        returns = rng.normal(0, volatility)
        price *= (1 + returns)

        # Random quantity
        qty = rng.exponential(0.1)

        # Random side (slightly biased based on price direction)
        is_buyer_maker = rng.random() < 0.5

        yield {
            "p": str(round(price, 2)),
            "q": str(round(qty, 8)),
            "T": timestamp,
            "m": is_buyer_maker,
        }

        # Increment timestamp (random interval 50-500ms)
        timestamp += rng.integers(50, 500)
