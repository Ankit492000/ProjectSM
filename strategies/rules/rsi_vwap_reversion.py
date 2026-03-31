"""RSI + VWAP Mean Reversion — best for range-bound / sideways markets.

Scenario: Stock is trading in a channel. When price gets oversold near VWAP support,
buy the dip. When overbought above VWAP resistance, sell the rip.

Works well on: NIFTY 50 stocks during consolidation phases, large-cap liquid names.
Timeframe: 15min / 1hr / daily
"""

from typing import Any

import pandas as pd
import pandas_ta as ta

from src.strategy import Strategy, Signal


class RSIVWAPMeanReversion(Strategy):
    """Mean reversion using RSI extremes + VWAP as dynamic fair value.

    BUY:  RSI < oversold threshold AND price < VWAP (dip below fair value)
    SELL: RSI > overbought threshold AND price > VWAP (rip above fair value)

    Extra confirmation: Volume spike on reversal candle, ATR filter to avoid
    chopping in dead markets.
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        super().__init__(name, params)
        self.rsi_period = self.params.get("rsi_period", 14)
        self.oversold = self.params.get("oversold", 30)
        self.overbought = self.params.get("overbought", 70)
        self.rsi_exit_mid = self.params.get("rsi_exit_mid", 50)
        self.volume_spike = self.params.get("volume_spike_factor", 1.5)
        self.atr_period = self.params.get("atr_period", 14)
        self.min_atr_pct = self.params.get("min_atr_pct", 0.5)  # skip if ATR < 0.5% of price

    def on_tick(self, symbol: str, tick: dict) -> Signal | None:
        return None

    def on_candle(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        if len(df) < max(self.rsi_period, self.atr_period, 20) + 5:
            return None

        df = df.copy()
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)

        # VWAP: use pandas_ta if volume exists, else fall back to SMA(20) as proxy
        if "volume" in df.columns and df["volume"].sum() > 0:
            vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            if vwap is not None:
                df["vwap"] = vwap
            else:
                df["vwap"] = ta.sma(df["close"], length=20)
        else:
            df["vwap"] = ta.sma(df["close"], length=20)

        df["vol_ma"] = df["volume"].rolling(20).mean() if "volume" in df.columns else 0

        rsi = df["rsi"].iloc[-1]
        price = df["close"].iloc[-1]
        vwap_val = df["vwap"].iloc[-1]
        atr = df["atr"].iloc[-1]

        if pd.isna(rsi) or pd.isna(vwap_val) or pd.isna(atr):
            return None

        # Skip dead/low-volatility markets
        atr_pct = atr / price * 100
        if atr_pct < self.min_atr_pct:
            return None

        # Volume spike check
        vol_ok = True
        if "volume" in df.columns and df["vol_ma"].iloc[-1] > 0:
            vol_ok = df["volume"].iloc[-1] > df["vol_ma"].iloc[-1] * self.volume_spike

        # ── BUY: Oversold + below VWAP ──
        if rsi < self.oversold and price < vwap_val:
            rsi_depth = (self.oversold - rsi) / self.oversold  # deeper oversold = higher confidence
            confidence = min(0.55 + rsi_depth * 0.3 + (0.1 if vol_ok else 0), 0.95)
            return Signal(
                symbol=symbol, action="BUY", confidence=confidence,
                price=price, strategy_name=self.name,
                reason=f"RSI {rsi:.0f} oversold, price ₹{price:.1f} below VWAP ₹{vwap_val:.1f}"
                       + (", volume spike" if vol_ok else ""),
            )

        # ── SELL: Overbought + above VWAP ──
        if rsi > self.overbought and price > vwap_val:
            rsi_height = (rsi - self.overbought) / (100 - self.overbought)
            confidence = min(0.55 + rsi_height * 0.3 + (0.1 if vol_ok else 0), 0.95)
            return Signal(
                symbol=symbol, action="SELL", confidence=confidence,
                price=price, strategy_name=self.name,
                reason=f"RSI {rsi:.0f} overbought, price ₹{price:.1f} above VWAP ₹{vwap_val:.1f}",
            )

        # ── Exit signal: RSI returning to midline (if you're already in a position) ──
        rsi_prev = df["rsi"].iloc[-2] if len(df) > 1 else rsi
        if not pd.isna(rsi_prev):
            # Was oversold, now crossing above mid → exit long
            if rsi_prev < self.rsi_exit_mid and rsi >= self.rsi_exit_mid:
                return Signal(
                    symbol=symbol, action="SELL", confidence=0.6,
                    price=price, strategy_name=self.name,
                    reason=f"RSI reverted to {rsi:.0f} — mean reversion target hit",
                )

        return None
