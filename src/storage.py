"""Parquet-based file storage for historical candles, live ticks, and instrument data."""

from pathlib import Path
from datetime import date

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# Load storage paths from config
_cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f)["storage"]

BASE_DIR = Path(_cfg["base_dir"])
HISTORICAL_DIR = Path(_cfg["historical_dir"])
TICKS_DIR = Path(_cfg["ticks_dir"])
INSTRUMENTS_DIR = Path(_cfg["instruments_dir"])


# ── Historical Candles ────────────────────────────────────────────────────────

def save_candles(symbol: str, exchange: str, interval: str, df: pd.DataFrame) -> Path:
    """Write OHLCV candle data to Parquet.
    
    Path: data/historical/{exchange}/{symbol}/{interval}.parquet
    """
    path = HISTORICAL_DIR / exchange / symbol / f"{interval}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    # If file exists, merge and deduplicate by timestamp
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def load_candles(
    symbol: str,
    exchange: str,
    interval: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Read candle data from Parquet with optional date range filter."""
    path = HISTORICAL_DIR / exchange / symbol / f"{interval}.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if start_date:
            df = df[df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.to_datetime(end_date)]
    return df


# ── Live Tick Data ────────────────────────────────────────────────────────────

def append_ticks(symbol: str, tick_date: date, df: pd.DataFrame) -> Path:
    """Append tick data to daily Parquet file.
    
    Path: data/ticks/{YYYY-MM-DD}/{symbol}.parquet
    """
    date_str = tick_date.isoformat()
    path = TICKS_DIR / date_str / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def load_ticks(symbol: str, tick_date: date) -> pd.DataFrame:
    """Load tick data for a symbol on a given date."""
    path = TICKS_DIR / tick_date.isoformat() / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ── Instrument Master ─────────────────────────────────────────────────────────

def save_instruments(exchange: str, data: list[dict]) -> Path:
    """Cache instrument master as Parquet."""
    path = INSTRUMENTS_DIR / f"{exchange}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def load_instruments(exchange: str) -> pd.DataFrame:
    """Load cached instrument master."""
    path = INSTRUMENTS_DIR / f"{exchange}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def get_instrument_key(symbol: str, exchange: str) -> str | None:
    """Resolve a trading symbol to its Upstox instrument_key from cached data."""
    df = load_instruments(exchange)
    if df.empty:
        return None
    # Try exact match on trading_symbol or name
    for col in ["trading_symbol", "tradingsymbol", "name"]:
        if col in df.columns:
            match = df[df[col].str.upper() == symbol.upper()]
            if not match.empty:
                key_col = "instrument_key" if "instrument_key" in match.columns else "key"
                if key_col in match.columns:
                    return match.iloc[0][key_col]
    return None
