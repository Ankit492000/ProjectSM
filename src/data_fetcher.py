"""Fetch historical candle data and instrument masters from Upstox + fallback sources."""

import gzip
import json
import time
from datetime import datetime, timedelta

import httpx
import pandas as pd

from src.auth import get_access_token
from src import storage

UPSTOX_BASE = "https://api.upstox.com"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange"

# Rate limiting: simple token bucket (50/sec)
_last_request_time = 0.0
_MIN_INTERVAL = 0.025  # 40 req/sec to stay under 50/sec limit


def _rate_limit():
    global _last_request_time
    now = time.time()
    wait = _MIN_INTERVAL - (now - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.time()


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Accept": "application/json",
    }


# ── Instrument Master ─────────────────────────────────────────────────────────

def fetch_instruments(exchange: str = "NSE") -> pd.DataFrame:
    """Download instrument master from Upstox and cache it.
    
    exchange: NSE, BSE, MCX, or complete
    """
    url = f"{INSTRUMENTS_URL}/{exchange}.json.gz"
    print(f"Downloading instruments for {exchange}...")
    resp = httpx.get(url, timeout=60)
    resp.raise_for_status()

    data = json.loads(gzip.decompress(resp.content))
    storage.save_instruments(exchange, data)
    print(f"Cached {len(data)} instruments for {exchange}.")
    return pd.DataFrame(data)


# ── Historical Candles (Upstox v3) ────────────────────────────────────────────

def fetch_historical_candles(
    instrument_key: str,
    interval: str = "day",
    unit: str = "days",
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """Fetch historical OHLCV candles from Upstox v3 API.
    
    instrument_key: e.g. "NSE_EQ|INE669E01016" (from instrument master)
    interval: 1, 5, 15, 30 (for intraday) or day/week/month
    unit: minutes, hours, days, weeks, months
    from_date / to_date: YYYY-MM-DD strings
    """
    _rate_limit()

    if to_date is None:
        to_date = datetime.now().strftime("%Y-%m-%d")
    if from_date is None:
        from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    url = (
        f"{UPSTOX_BASE}/v3/historical-candle"
        f"/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"
    )

    resp = httpx.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    body = resp.json()

    candles = body.get("data", {}).get("candles", [])
    if not candles:
        print(f"No candle data returned for {instrument_key}")
        return pd.DataFrame()

    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "open_interest"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def fetch_and_store(
    symbol: str,
    exchange: str = "NSE",
    interval: str = "day",
    unit: str = "days",
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """Fetch candles and persist to Parquet. Resolves symbol to instrument_key automatically."""
    # Ensure instruments are cached
    instruments = storage.load_instruments(exchange)
    if instruments.empty:
        fetch_instruments(exchange)

    instrument_key = storage.get_instrument_key(symbol, exchange)
    if not instrument_key:
        raise ValueError(f"Could not resolve instrument_key for {symbol} on {exchange}. "
                         "Check symbol name or refresh instruments.")

    print(f"Fetching {interval} candles for {symbol} ({instrument_key})...")
    df = fetch_historical_candles(instrument_key, interval, unit, from_date, to_date)

    if not df.empty:
        path = storage.save_candles(symbol, exchange, interval, df)
        print(f"Saved {len(df)} candles → {path}")

    return df


# ── Fallback: yfinance ────────────────────────────────────────────────────────

def fetch_yfinance(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fallback: fetch data via yfinance (for quick prototyping).
    
    For NSE stocks, append .NS (e.g., RELIANCE.NS). For BSE, append .BO.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval).reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Rename 'date' to 'timestamp' for consistency
    if "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    return df
