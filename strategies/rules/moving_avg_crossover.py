"""EMA Crossover strategy — rule-based, configurable fast/slow periods."""

from typing import Any

import pandas as pd
import pandas_ta as ta

from src.strategy import Strategy, Signal


class EMACrossover(Strategy):
    """Buy when fast EMA crosses above slow EMA (with optional volume confirmation).
       Sell when fast EMA crosses below slow EMA."""

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        super().__init__(name, params)
        self.fast = self.params.get("fast_period", 9)
        self.slow = self.params.get("slow_period", 21)
        self.volume_confirm = self.params.get("volume_confirm", True)
        self.volume_ma_period = self.params.get("volume_ma_period", 20)

    def on_tick(self, symbol: str, tick: dict) -> Signal | None:
        # This rule-based strategy operates on candles, not individual ticks.
        return None

    def on_candle(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        if len(df) < self.slow + 2:
            return None

        df = df.copy()
        df[f"ema_{self.fast}"] = ta.ema(df["close"], length=self.fast)
        df[f"ema_{self.slow}"] = ta.ema(df["close"], length=self.slow)

        fast_now = df[f"ema_{self.fast}"].iloc[-1]
        fast_prev = df[f"ema_{self.fast}"].iloc[-2]
        slow_now = df[f"ema_{self.slow}"].iloc[-1]
        slow_prev = df[f"ema_{self.slow}"].iloc[-2]

        if pd.isna(fast_now) or pd.isna(slow_now):
            return None

        # Volume confirmation
        vol_ok = True
        if self.volume_confirm and "volume" in df.columns:
            vol_ma = df["volume"].rolling(self.volume_ma_period).mean().iloc[-1]
            vol_ok = df["volume"].iloc[-1] > vol_ma if not pd.isna(vol_ma) else True

        price = df["close"].iloc[-1]
        ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else None

        # Bullish crossover
        if fast_prev <= slow_prev and fast_now > slow_now and vol_ok:
            confidence = min(0.5 + abs(fast_now - slow_now) / price * 100, 0.95)
            return Signal(
                symbol=symbol, action="BUY", confidence=confidence,
                price=price, strategy_name=self.name,
                reason=f"EMA {self.fast} crossed above EMA {self.slow}"
                       + (" with volume confirmation" if self.volume_confirm and vol_ok else ""),
            )

        # Bearish crossover
        if fast_prev >= slow_prev and fast_now < slow_now:
            confidence = min(0.5 + abs(slow_now - fast_now) / price * 100, 0.95)
            return Signal(
                symbol=symbol, action="SELL", confidence=confidence,
                price=price, strategy_name=self.name,
                reason=f"EMA {self.fast} crossed below EMA {self.slow}",
            )

        return None
