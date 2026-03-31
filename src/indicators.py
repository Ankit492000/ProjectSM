"""Compatibility shim: provides pandas_ta-style function API using the `ta` library.

The `pandas_ta` package is no longer available on PyPI. This module wraps the `ta`
library (pip install ta) to expose the same function signatures used throughout
the codebase: ta.rsi(), ta.ema(), ta.sma(), ta.macd(), ta.bbands(), ta.atr(),
ta.adx(), ta.vwap(), ta.kc().
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    return RSIIndicator(close=close, window=length).rsi()


def ema(close: pd.Series, length: int = 20) -> pd.Series:
    return EMAIndicator(close=close, window=length).ema_indicator()


def sma(close: pd.Series, length: int = 20) -> pd.Series:
    return SMAIndicator(close=close, window=length).sma_indicator()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    m = MACD(close=close, window_fast=fast, window_slow=slow, window_sign=signal)
    return pd.DataFrame({
        f"MACD_{fast}_{slow}_{signal}": m.macd(),
        f"MACDs_{fast}_{slow}_{signal}": m.macd_signal(),
        f"MACDh_{fast}_{slow}_{signal}": m.macd_diff(),
    })


def bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    bb = BollingerBands(close=close, window=length, window_dev=std)
    return pd.DataFrame({
        f"BBU_{length}_{std}": bb.bollinger_hband(),
        f"BBM_{length}_{std}": bb.bollinger_mavg(),
        f"BBL_{length}_{std}": bb.bollinger_lband(),
        f"BBB_{length}_{std}": bb.bollinger_wband(),
        f"BBP_{length}_{std}": bb.bollinger_pband(),
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    return AverageTrueRange(high=high, low=low, close=close, window=length).average_true_range()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    a = ADXIndicator(high=high, low=low, close=close, window=length)
    return pd.DataFrame({
        f"ADX_{length}": a.adx(),
        f"DMP_{length}": a.adx_pos(),
        f"DMN_{length}": a.adx_neg(),
    })


def kc(high: pd.Series, low: pd.Series, close: pd.Series,
       length: int = 20, scalar: float = 1.5) -> pd.DataFrame:
    k = KeltnerChannel(high=high, low=low, close=close, window=length, window_atr=length,
                        multiplier=scalar)
    return pd.DataFrame({
        f"KCU_{length}_{scalar}": k.keltner_channel_hband(),
        f"KCM_{length}_{scalar}": k.keltner_channel_mband(),
        f"KCL_{length}_{scalar}": k.keltner_channel_lband(),
    })


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Simple VWAP approximation (rolling cumulative)."""
    typical_price = (high + low + close) / 3
    cumvol = volume.cumsum()
    cumtp = (typical_price * volume).cumsum()
    result = cumtp / cumvol
    result[cumvol == 0] = np.nan
    return result
