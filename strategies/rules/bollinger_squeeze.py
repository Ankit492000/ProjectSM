"""Bollinger Band Squeeze Breakout — best for pre-breakout compression setups.

Scenario: Stock has been consolidating in a tight range (low volatility). Bollinger
Bands squeeze (bandwidth contracts). When the squeeze releases with a directional
candle + volume, a strong move follows.

Works well on: Nifty IT, Nifty Bank during earnings season, stocks in triangles/flags.
Timeframe: daily / 4hr
Indian market edge: Consolidation before result season → explosive move post earnings.
"""

from typing import Any

import pandas as pd
import pandas_ta as ta

from src.strategy import Strategy, Signal


class BollingerSqueezeBreakout(Strategy):
    """Detect Bollinger Band squeeze (low bandwidth) → breakout on expansion.

    Squeeze detection: BB bandwidth < percentile threshold (tight bands).
    Breakout: Close breaks above upper band (bullish) or below lower band (bearish)
    with volume surge.

    Keltner Channel filter (optional): True squeeze = BB inside KC (like TTM Squeeze).
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        super().__init__(name, params)
        self.bb_period = self.params.get("bb_period", 20)
        self.bb_std = self.params.get("bb_std", 2.0)
        self.squeeze_lookback = self.params.get("squeeze_lookback", 120)
        self.squeeze_percentile = self.params.get("squeeze_percentile", 20)  # BW < 20th percentile = squeeze
        self.keltner_filter = self.params.get("keltner_filter", True)
        self.kc_period = self.params.get("kc_period", 20)
        self.kc_multiplier = self.params.get("kc_multiplier", 1.5)
        self.volume_surge = self.params.get("volume_surge_factor", 1.3)
        self.min_squeeze_bars = self.params.get("min_squeeze_bars", 3)  # squeeze must last N bars

    def on_tick(self, symbol: str, tick: dict) -> Signal | None:
        return None

    def on_candle(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        needed = max(self.bb_period, self.kc_period, self.squeeze_lookback) + 10
        if len(df) < needed:
            return None

        df = df.copy()

        # Bollinger Bands
        bb = ta.bbands(df["close"], length=self.bb_period, std=self.bb_std)
        if bb is None:
            return None
        df = pd.concat([df, bb], axis=1)

        # Find BB columns dynamically
        upper_col = [c for c in df.columns if "BBU" in c]
        lower_col = [c for c in df.columns if "BBL" in c]
        mid_col = [c for c in df.columns if "BBM" in c]
        bw_col = [c for c in df.columns if "BBB" in c]

        if not (upper_col and lower_col and mid_col and bw_col):
            return None

        upper_col, lower_col, mid_col, bw_col = upper_col[0], lower_col[0], mid_col[0], bw_col[0]

        # Bandwidth and squeeze detection
        bandwidth = df[bw_col]
        bw_threshold = bandwidth.rolling(self.squeeze_lookback).quantile(self.squeeze_percentile / 100).iloc[-1]

        if pd.isna(bw_threshold):
            return None

        # Check if we were in squeeze for min_squeeze_bars
        recent_bw = bandwidth.iloc[-(self.min_squeeze_bars + 1):-1]
        was_in_squeeze = (recent_bw < bw_threshold).all()

        if not was_in_squeeze:
            return None

        # Current bar: squeeze releasing (bandwidth expanding)
        bw_now = bandwidth.iloc[-1]
        bw_prev = bandwidth.iloc[-2]
        squeeze_releasing = bw_now > bw_prev

        if not squeeze_releasing:
            return None

        # Keltner Channel filter: true squeeze means BB was inside KC
        if self.keltner_filter:
            kc = ta.kc(df["high"], df["low"], df["close"],
                       length=self.kc_period, scalar=self.kc_multiplier)
            if kc is not None:
                kc_upper = [c for c in kc.columns if "KCU" in c]
                kc_lower = [c for c in kc.columns if "KCL" in c]
                if kc_upper and kc_lower:
                    # BB was inside KC = true squeeze
                    bb_upper_prev = df[upper_col].iloc[-2]
                    bb_lower_prev = df[lower_col].iloc[-2]
                    kc_u = kc[kc_upper[0]].iloc[-2]
                    kc_l = kc[kc_lower[0]].iloc[-2]
                    if not pd.isna(kc_u) and not pd.isna(kc_l):
                        if bb_upper_prev > kc_u or bb_lower_prev < kc_l:
                            return None  # Not a true squeeze (BB was outside KC)

        price = df["close"].iloc[-1]
        upper = df[upper_col].iloc[-1]
        lower = df[lower_col].iloc[-1]

        # Volume confirmation
        vol_ok = True
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(20).mean().iloc[-1]
            if not pd.isna(vol_ma) and vol_ma > 0:
                vol_ok = df["volume"].iloc[-1] > vol_ma * self.volume_surge

        # ── Bullish breakout: close above upper band ──
        if price > upper and vol_ok:
            confidence = min(0.65 + (0.1 if vol_ok else 0) + (bw_now - bw_prev) / bw_now * 0.2, 0.95)
            return Signal(
                symbol=symbol, action="BUY", confidence=confidence,
                price=price, strategy_name=self.name,
                reason=f"BB Squeeze breakout UP — price ₹{price:.1f} broke above ₹{upper:.1f}"
                       + (", volume surge" if vol_ok else ""),
            )

        # ── Bearish breakdown: close below lower band ──
        if price < lower and vol_ok:
            confidence = min(0.65 + (0.1 if vol_ok else 0), 0.90)
            return Signal(
                symbol=symbol, action="SELL", confidence=confidence,
                price=price, strategy_name=self.name,
                reason=f"BB Squeeze breakdown — price ₹{price:.1f} broke below ₹{lower:.1f}",
            )

        return None
