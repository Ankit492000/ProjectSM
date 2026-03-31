"""Supertrend Momentum — best for trending markets / breakout continuation.

Scenario: Market has picked a direction (post-budget rally, sector rotation,
earnings breakout). Ride the trend using Supertrend as a dynamic trailing stop.

Works well on: NIFTY/BANKNIFTY during directional moves, mid-cap momentum names.
Timeframe: daily / 1hr
Indian market edge: Very popular among Indian retail traders — price respects Supertrend
levels on liquid names due to collective behavior. Nifty trends 60% of the time on daily TF.
"""

from typing import Any

import pandas as pd
import pandas_ta as ta
import numpy as np

from src.strategy import Strategy, Signal


class SupertrendMomentum(Strategy):
    """Trend-following using Supertrend indicator + ADX filter for trend strength.

    BUY:  Supertrend flips bullish (price crosses above ST) AND ADX > threshold
    SELL: Supertrend flips bearish (price crosses below ST)

    Dual timeframe confirmation: Optional long-term SMA (200) filter — only BUY
    when price is above SMA 200 (structural uptrend).
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        super().__init__(name, params)
        self.atr_period = self.params.get("atr_period", 10)
        self.multiplier = self.params.get("multiplier", 3.0)
        self.adx_period = self.params.get("adx_period", 14)
        self.adx_threshold = self.params.get("adx_threshold", 20)
        self.use_sma_filter = self.params.get("use_sma_filter", True)
        self.sma_period = self.params.get("sma_period", 200)

    def on_tick(self, symbol: str, tick: dict) -> Signal | None:
        return None

    def _compute_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Supertrend manually for reliability."""
        hl2 = (df["high"] + df["low"]) / 2
        atr = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        if atr is None:
            return df

        upper_band = hl2 + self.multiplier * atr
        lower_band = hl2 - self.multiplier * atr

        supertrend = pd.Series(np.nan, index=df.index)
        direction = pd.Series(1, index=df.index)  # 1 = bullish, -1 = bearish

        for i in range(1, len(df)):
            if pd.isna(upper_band.iloc[i]):
                continue

            # Adjust bands based on previous values
            if not pd.isna(lower_band.iloc[i - 1]):
                if lower_band.iloc[i] < lower_band.iloc[i - 1] and df["close"].iloc[i - 1] > lower_band.iloc[i - 1]:
                    lower_band.iloc[i] = lower_band.iloc[i - 1]

            if not pd.isna(upper_band.iloc[i - 1]):
                if upper_band.iloc[i] > upper_band.iloc[i - 1] and df["close"].iloc[i - 1] < upper_band.iloc[i - 1]:
                    upper_band.iloc[i] = upper_band.iloc[i - 1]

            # Direction logic
            if not pd.isna(supertrend.iloc[i - 1]):
                if supertrend.iloc[i - 1] == upper_band.iloc[i - 1]:
                    direction.iloc[i] = -1 if df["close"].iloc[i] > upper_band.iloc[i] else 1
                    if direction.iloc[i] == -1:
                        direction.iloc[i] = 1  # flip to bullish
                else:
                    direction.iloc[i] = 1 if df["close"].iloc[i] < lower_band.iloc[i] else -1
                    if direction.iloc[i] == 1:
                        direction.iloc[i] = -1  # flip to bearish

                # Simplified: use close vs bands
                if df["close"].iloc[i] > upper_band.iloc[i]:
                    direction.iloc[i] = 1
                elif df["close"].iloc[i] < lower_band.iloc[i]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = direction.iloc[i - 1]

            supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

        df["supertrend"] = supertrend
        df["st_direction"] = direction
        return df

    def on_candle(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        needed = max(self.atr_period, self.adx_period, self.sma_period if self.use_sma_filter else 0) + 10
        if len(df) < needed:
            return None

        df = df.copy()
        df = self._compute_supertrend(df)

        # ADX for trend strength
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.adx_period)
        if adx_df is not None:
            df = pd.concat([df, adx_df], axis=1)

        adx_col = f"ADX_{self.adx_period}"
        if adx_col not in df.columns:
            return None

        price = df["close"].iloc[-1]
        adx = df[adx_col].iloc[-1]
        dir_now = df["st_direction"].iloc[-1]
        dir_prev = df["st_direction"].iloc[-2]
        st = df["supertrend"].iloc[-1]

        if pd.isna(adx) or pd.isna(dir_now) or pd.isna(dir_prev):
            return None

        # SMA 200 structural filter
        if self.use_sma_filter:
            sma = ta.sma(df["close"], length=self.sma_period)
            if sma is not None and not pd.isna(sma.iloc[-1]):
                if price < sma.iloc[-1]:
                    return None  # don't buy below SMA 200

        # ── BUY: Supertrend flips bullish + ADX confirms trend ──
        if dir_prev == -1 and dir_now == 1 and adx > self.adx_threshold:
            trend_strength = min((adx - self.adx_threshold) / 30, 0.3)
            confidence = min(0.6 + trend_strength, 0.95)
            return Signal(
                symbol=symbol, action="BUY", confidence=confidence,
                price=price, strategy_name=self.name,
                reason=f"Supertrend flipped bullish at ₹{st:.1f}, ADX {adx:.0f} (strong trend)",
            )

        # ── SELL: Supertrend flips bearish ──
        if dir_prev == 1 and dir_now == -1:
            confidence = min(0.6 + (adx / 100) * 0.2, 0.9)
            return Signal(
                symbol=symbol, action="SELL", confidence=confidence,
                price=price, strategy_name=self.name,
                reason=f"Supertrend flipped bearish at ₹{st:.1f}, ADX {adx:.0f}",
            )

        return None
