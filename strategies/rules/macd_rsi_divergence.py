"""MACD + RSI Divergence — best for catching reversals at swing highs/lows.

Scenario: Price makes a new high but MACD/RSI doesn't (bearish divergence) → top
forming. Price makes a new low but MACD/RSI doesn't (bullish divergence) → bottom
forming.

Works well on: Large-cap reversals (RELIANCE, HDFC Bank, INFY), index tops/bottoms.
Timeframe: daily / weekly (more reliable on higher TFs)
Indian market edge: NIFTY divergences at round numbers (18000, 20000, 22000) are
historically strong reversal signals with high win rates.
"""

from typing import Any

import pandas as pd
import src.indicators as ta
import numpy as np

from src.strategy import Strategy, Signal


class MACDRSIDivergence(Strategy):
    """Detect bullish and bearish divergences between price and MACD/RSI.

    Bullish divergence: Price makes lower low, RSI/MACD makes higher low → BUY
    Bearish divergence: Price makes higher high, RSI/MACD makes lower high → SELL

    Dual confirmation: Both MACD histogram and RSI must show divergence.
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        super().__init__(name, params)
        self.rsi_period = self.params.get("rsi_period", 14)
        self.macd_fast = self.params.get("macd_fast", 12)
        self.macd_slow = self.params.get("macd_slow", 26)
        self.macd_signal = self.params.get("macd_signal", 9)
        self.lookback = self.params.get("divergence_lookback", 20)  # bars to look back for swing points
        self.require_both = self.params.get("require_both", False)  # require both RSI + MACD divergence
        self.min_swing_pct = self.params.get("min_swing_pct", 1.0)  # min % move between swings

    def on_tick(self, symbol: str, tick: dict) -> Signal | None:
        return None

    def _find_swing_lows(self, series: pd.Series, order: int = 5) -> list[tuple[int, float]]:
        """Find local minima (swing lows) in a series."""
        swings = []
        for i in range(order, len(series) - 1):
            if pd.isna(series.iloc[i]):
                continue
            window = series.iloc[max(0, i - order):i + order + 1]
            if series.iloc[i] == window.min():
                swings.append((i, series.iloc[i]))
        return swings

    def _find_swing_highs(self, series: pd.Series, order: int = 5) -> list[tuple[int, float]]:
        """Find local maxima (swing highs) in a series."""
        swings = []
        for i in range(order, len(series) - 1):
            if pd.isna(series.iloc[i]):
                continue
            window = series.iloc[max(0, i - order):i + order + 1]
            if series.iloc[i] == window.max():
                swings.append((i, series.iloc[i]))
        return swings

    def on_candle(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        needed = self.macd_slow + self.macd_signal + self.lookback + 10
        if len(df) < needed:
            return None

        df = df.copy()
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)

        macd_df = ta.macd(df["close"], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        if macd_df is None:
            return None
        df = pd.concat([df, macd_df], axis=1)

        hist_col = [c for c in df.columns if "MACDh" in c or "MACD_hist" in c.replace("_", "")]
        if not hist_col:
            hist_col = [c for c in df.columns if "h_" in c and "MACD" in c]
        if not hist_col:
            return None
        hist_col = hist_col[0]

        # Work on recent window
        window = df.iloc[-self.lookback - 10:]
        price = df["close"].iloc[-1]

        # Find swing points in the lookback window
        price_lows = self._find_swing_lows(window["low"], order=3)
        price_highs = self._find_swing_highs(window["high"], order=3)
        rsi_lows = self._find_swing_lows(window["rsi"], order=3)
        rsi_highs = self._find_swing_highs(window["rsi"], order=3)
        macd_lows = self._find_swing_lows(window[hist_col], order=3)
        macd_highs = self._find_swing_highs(window[hist_col], order=3)

        bull_rsi = self._check_bullish_divergence(price_lows, rsi_lows)
        bull_macd = self._check_bullish_divergence(price_lows, macd_lows)
        bear_rsi = self._check_bearish_divergence(price_highs, rsi_highs)
        bear_macd = self._check_bearish_divergence(price_highs, macd_highs)

        # ── Bullish divergence ──
        if self.require_both:
            bull_signal = bull_rsi and bull_macd
        else:
            bull_signal = bull_rsi or bull_macd

        if bull_signal:
            parts = []
            if bull_rsi:
                parts.append("RSI")
            if bull_macd:
                parts.append("MACD")
            confidence = 0.65 if len(parts) == 1 else 0.80
            return Signal(
                symbol=symbol, action="BUY", confidence=confidence,
                price=price, strategy_name=self.name,
                reason=f"Bullish divergence ({'+'.join(parts)}) — "
                       f"price making lower lows, indicator making higher lows",
            )

        # ── Bearish divergence ──
        if self.require_both:
            bear_signal = bear_rsi and bear_macd
        else:
            bear_signal = bear_rsi or bear_macd

        if bear_signal:
            parts = []
            if bear_rsi:
                parts.append("RSI")
            if bear_macd:
                parts.append("MACD")
            confidence = 0.65 if len(parts) == 1 else 0.80
            return Signal(
                symbol=symbol, action="SELL", confidence=confidence,
                price=price, strategy_name=self.name,
                reason=f"Bearish divergence ({'+'.join(parts)}) — "
                       f"price making higher highs, indicator making lower highs",
            )

        return None

    def _check_bullish_divergence(
        self, price_swings: list[tuple[int, float]], indicator_swings: list[tuple[int, float]]
    ) -> bool:
        """Bullish: price lower low + indicator higher low."""
        if len(price_swings) < 2 or len(indicator_swings) < 2:
            return False

        # Compare last two swing lows
        p1_idx, p1_val = price_swings[-2]
        p2_idx, p2_val = price_swings[-1]
        i1_idx, i1_val = indicator_swings[-2]
        i2_idx, i2_val = indicator_swings[-1]

        # Price: lower low
        price_lower = p2_val < p1_val
        # Indicator: higher low
        indicator_higher = i2_val > i1_val

        # Min swing distance
        if p1_val > 0:
            swing_pct = abs(p2_val - p1_val) / p1_val * 100
            if swing_pct < self.min_swing_pct:
                return False

        return price_lower and indicator_higher

    def _check_bearish_divergence(
        self, price_swings: list[tuple[int, float]], indicator_swings: list[tuple[int, float]]
    ) -> bool:
        """Bearish: price higher high + indicator lower high."""
        if len(price_swings) < 2 or len(indicator_swings) < 2:
            return False

        p1_idx, p1_val = price_swings[-2]
        p2_idx, p2_val = price_swings[-1]
        i1_idx, i1_val = indicator_swings[-2]
        i2_idx, i2_val = indicator_swings[-1]

        price_higher = p2_val > p1_val
        indicator_lower = i2_val < i1_val

        if p1_val > 0:
            swing_pct = abs(p2_val - p1_val) / p1_val * 100
            if swing_pct < self.min_swing_pct:
                return False

        return price_higher and indicator_lower
