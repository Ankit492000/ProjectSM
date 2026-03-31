"""Fundamental + Technical Combo strategy — the "CANSLIM-lite" approach.

Scenario: Only buy technically strong setups in fundamentally sound stocks.
This is the "quality filter + timing" approach. Fundamentals decide WHAT to buy,
technicals decide WHEN to buy.

Works well on: Portfolio-level stock selection, swing trading quality stocks.
Timeframe: daily / weekly
Indian market edge: Filter out F&O pump-and-dump scripts. Focus on Nifty 500
quality names with clean balance sheets. The Indian market has many low-quality
small-caps — fundamental filtering is essential.

Fundamental gate:
  - Overall fundamental score > threshold → stock is tradeable
  - Below threshold → skip entirely, no matter what technicals say

Technical entry (once fundamentals pass):
  - Price above SMA 200 (structural uptrend)
  - RSI between 40-65 (not overbought, with momentum room)
  - EMA 21 > EMA 50 (intermediate trend bullish)
  - Volume above average (institutional participation)

This is a conservative, higher-probability strategy with fewer trades but better
risk-adjusted returns.
"""

from typing import Any

import pandas as pd
import src.indicators as ta

from src.strategy import Strategy, Signal


class FundaTechCombo(Strategy):
    """Fundamental filter + Technical timing combo strategy.

    Step 1: Check fundamental score (cached). If below threshold → skip.
    Step 2: Check technical conditions for entry/exit timing.
    """

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        super().__init__(name, params)
        self.min_fundamental_score = self.params.get("min_fundamental_score", 0.5)
        self.sector_pe = self.params.get("sector_median_pe", 25.0)
        self.ticker_suffix = self.params.get("ticker_suffix", ".NS")  # .NS for NSE, .BO for BSE
        self.sma_long = self.params.get("sma_long", 200)
        self.ema_mid = self.params.get("ema_mid", 21)
        self.ema_slow = self.params.get("ema_slow", 50)
        self.rsi_period = self.params.get("rsi_period", 14)
        self.rsi_buy_low = self.params.get("rsi_buy_low", 40)
        self.rsi_buy_high = self.params.get("rsi_buy_high", 65)
        self.rsi_sell_threshold = self.params.get("rsi_sell_threshold", 75)
        self.volume_ma_period = self.params.get("volume_ma_period", 20)

        # Cache fundamental scores to avoid re-fetching every candle
        self._fund_cache: dict[str, float | None] = {}

    def _get_fundamental_score(self, symbol: str) -> float | None:
        """Get the fundamental score for a symbol. Cached after first call."""
        if symbol in self._fund_cache:
            return self._fund_cache[symbol]

        try:
            from src.fundamentals import get_fundamentals, FundamentalScore
            ticker = symbol + self.ticker_suffix
            data = get_fundamentals(ticker)
            score = FundamentalScore.compute(data, sector_median_pe=self.sector_pe)
            self._fund_cache[symbol] = score.overall
            return score.overall
        except Exception as e:
            # If fundamentals unavailable, don't block — return None (skip fundamental gate)
            print(f"  FundaTech: Could not fetch fundamentals for {symbol}: {e}")
            self._fund_cache[symbol] = None
            return None

    def on_tick(self, symbol: str, tick: dict) -> Signal | None:
        return None

    def on_candle(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        needed = max(self.sma_long, self.ema_slow, self.volume_ma_period) + 10
        if len(df) < needed:
            return None

        # ── Step 1: Fundamental gate ──
        fund_score = self._get_fundamental_score(symbol)
        if fund_score is not None and fund_score < self.min_fundamental_score:
            return None  # Fundamentally weak → skip

        fundamental_tag = f"Funda: {fund_score:.2f}" if fund_score is not None else "Funda: N/A"

        # ── Step 2: Technical analysis ──
        df = df.copy()
        df["sma_long"] = ta.sma(df["close"], length=self.sma_long)
        df[f"ema_{self.ema_mid}"] = ta.ema(df["close"], length=self.ema_mid)
        df[f"ema_{self.ema_slow}"] = ta.ema(df["close"], length=self.ema_slow)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)

        price = df["close"].iloc[-1]
        sma = df["sma_long"].iloc[-1]
        ema_mid = df[f"ema_{self.ema_mid}"].iloc[-1]
        ema_slow = df[f"ema_{self.ema_slow}"].iloc[-1]
        rsi = df["rsi"].iloc[-1]

        if any(pd.isna(v) for v in [sma, ema_mid, ema_slow, rsi]):
            return None

        # Volume check
        vol_ok = True
        if "volume" in df.columns:
            vol_ma = df["volume"].rolling(self.volume_ma_period).mean().iloc[-1]
            if not pd.isna(vol_ma) and vol_ma > 0:
                vol_ok = df["volume"].iloc[-1] > vol_ma

        # ── BUY conditions (all must be true) ──
        above_sma200 = price > sma
        ema_bullish = ema_mid > ema_slow
        rsi_in_range = self.rsi_buy_low <= rsi <= self.rsi_buy_high
        # EMA crossover (just happened)
        ema_mid_prev = df[f"ema_{self.ema_mid}"].iloc[-2]
        ema_slow_prev = df[f"ema_{self.ema_slow}"].iloc[-2]
        ema_cross = ema_mid_prev <= ema_slow_prev and ema_mid > ema_slow

        if above_sma200 and ema_bullish and rsi_in_range and vol_ok:
            reasons = [fundamental_tag]
            confidence = 0.55

            if fund_score and fund_score >= 0.7:
                confidence += 0.10
                reasons.append("strong fundamentals")
            if ema_cross:
                confidence += 0.10
                reasons.append("fresh EMA crossover")
            if vol_ok:
                confidence += 0.05
                reasons.append("volume confirmed")

            confidence = min(confidence, 0.95)
            return Signal(
                symbol=symbol, action="BUY", confidence=confidence,
                price=price, strategy_name=self.name,
                reason=f"FundaTech BUY — {', '.join(reasons)} | RSI {rsi:.0f}",
            )

        # ── SELL conditions ──
        # Sell if price drops below SMA 200 OR RSI overbought OR EMA bearish cross
        below_sma200 = price < sma
        rsi_overbought = rsi > self.rsi_sell_threshold
        ema_bearish_cross = ema_mid_prev >= ema_slow_prev and ema_mid < ema_slow

        if below_sma200 or ema_bearish_cross:
            reasons = [fundamental_tag]
            if below_sma200:
                reasons.append("broke below SMA 200")
            if ema_bearish_cross:
                reasons.append("EMA bearish crossover")
            if rsi_overbought:
                reasons.append(f"RSI overbought {rsi:.0f}")

            return Signal(
                symbol=symbol, action="SELL", confidence=0.70,
                price=price, strategy_name=self.name,
                reason=f"FundaTech SELL — {', '.join(reasons)}",
            )

        return None
