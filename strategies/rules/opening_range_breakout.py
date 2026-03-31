"""Opening Range Breakout (ORB) — best for intraday / first-hour trading.

Scenario: Define the high and low of the first N candles (opening range).
Buy when price breaks above the range high, sell when below range low.
Classic intraday strategy used heavily on Indian markets.

Works well on: BANKNIFTY, NIFTY 50, liquid F&O stocks (RELIANCE, ICICI, HDFC).
Timeframe: 5min / 15min candles (intraday)

Indian market edge: 9:15-9:30 AM range (first 15 min) is the most traded ORB window.
FII/DII activity creates strong opening ranges that tend to define the day's direction.

NOTE: For daily data backtesting, this strategy simulates ORB using the first candle's
open/high/low as the "opening range" — useful for swing-level ORB on daily bars.
For true intraday ORB, use 5min/15min candle data.
"""

from typing import Any

import pandas as pd
import src.indicators as ta

from src.strategy import Strategy, Signal


class OpeningRangeBreakout(Strategy):
    """Opening Range Breakout with ATR-based targets and invalidation.

    Setup: First N candles define range_high and range_low.
    BUY:  Close > range_high with volume (breakout above opening range).
    SELL: Close < range_low with volume (breakdown below opening range).

    Target: range_width * reward_multiplier from breakout point.
    Invalidation: Close back inside the range → SELL (failed breakout).

    For daily data: Uses previous day's high/low as "range" and current day's
    close for breakout detection (swing ORB).
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        super().__init__(name, params)
        self.range_bars = self.params.get("range_bars", 1)  # # of bars that define the range
        self.breakout_confirm_bars = self.params.get("breakout_confirm_bars", 1)
        self.volume_factor = self.params.get("volume_factor", 1.2)
        self.atr_period = self.params.get("atr_period", 14)
        self.use_atr_filter = self.params.get("use_atr_filter", True)
        self.min_range_atr_ratio = self.params.get("min_range_atr_ratio", 0.5)
        self.max_range_atr_ratio = self.params.get("max_range_atr_ratio", 2.0)
        # For daily swing ORB: track if we're in a breakout
        self._position_state: dict[str, str] = {}  # symbol → "long" | "short" | None

    def on_tick(self, symbol: str, tick: dict) -> Signal | None:
        return None

    def on_candle(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        needed = self.atr_period + self.range_bars + 5
        if len(df) < needed:
            return None

        df = df.copy()

        # Define opening range from the range_bars
        # For daily: the "range" is the previous bar(s) high/low
        range_slice = df.iloc[-(self.range_bars + self.breakout_confirm_bars):-self.breakout_confirm_bars]
        if len(range_slice) < self.range_bars:
            return None

        range_high = range_slice["high"].max()
        range_low = range_slice["low"].min()
        range_width = range_high - range_low

        if range_width <= 0:
            return None

        # ATR filter: skip if range is abnormally small or large vs ATR
        if self.use_atr_filter:
            atr = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
            if atr is not None and not pd.isna(atr.iloc[-1]):
                atr_val = atr.iloc[-1]
                ratio = range_width / atr_val if atr_val > 0 else 0
                if ratio < self.min_range_atr_ratio or ratio > self.max_range_atr_ratio:
                    return None

        # Current bar (breakout candle)
        price = df["close"].iloc[-1]
        bar_high = df["high"].iloc[-1]
        bar_low = df["low"].iloc[-1]
        prev_close = df["close"].iloc[-2]

        # Volume check
        vol_ok = True
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(20).mean().iloc[-1]
            if not pd.isna(vol_ma) and vol_ma > 0:
                vol_ok = df["volume"].iloc[-1] > vol_ma * self.volume_factor

        current_state = self._position_state.get(symbol)

        # ── Failed breakout detection (if already in position) ──
        if current_state == "long" and price < range_low:
            self._position_state[symbol] = None
            return Signal(
                symbol=symbol, action="SELL", confidence=0.75,
                price=price, strategy_name=self.name,
                reason=f"ORB failed breakout — price fell back below range low ₹{range_low:.1f}",
            )

        if current_state == "short" and price > range_high:
            self._position_state[symbol] = None
            return Signal(
                symbol=symbol, action="BUY", confidence=0.75,
                price=price, strategy_name=self.name,
                reason=f"ORB short invalidated — price rose above range high ₹{range_high:.1f}",
            )

        # ── Breakout above range high ──
        if current_state is None and price > range_high and prev_close <= range_high:
            if vol_ok:
                breakout_strength = (price - range_high) / range_width
                confidence = min(0.60 + breakout_strength * 0.2 + (0.1 if vol_ok else 0), 0.92)
                self._position_state[symbol] = "long"
                return Signal(
                    symbol=symbol, action="BUY", confidence=confidence,
                    price=price, strategy_name=self.name,
                    reason=f"ORB breakout UP — range [{range_low:.1f}-{range_high:.1f}], "
                           f"close ₹{price:.1f}" + (", volume confirmed" if vol_ok else ""),
                )

        # ── Breakdown below range low ──
        if current_state is None and price < range_low and prev_close >= range_low:
            if vol_ok:
                breakdown_strength = (range_low - price) / range_width
                confidence = min(0.60 + breakdown_strength * 0.2, 0.90)
                self._position_state[symbol] = "short"
                return Signal(
                    symbol=symbol, action="SELL", confidence=confidence,
                    price=price, strategy_name=self.name,
                    reason=f"ORB breakdown — range [{range_low:.1f}-{range_high:.1f}], "
                           f"close ₹{price:.1f}",
                )

        return None
