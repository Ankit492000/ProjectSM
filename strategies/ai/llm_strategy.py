"""LLM-based strategy — sends market context to OpenAI/Claude/Ollama for signal generation."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.analyzer import add_indicators
from src.strategy import Strategy, Signal

load_dotenv()

SYSTEM_PROMPT = """You are a professional Indian stock market analyst. 
Given recent OHLCV data with technical indicators for a stock, output a JSON object with exactly these fields:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation (1-2 sentences)"
}
Only output the JSON, nothing else. Be conservative — only recommend BUY/SELL when the signal is clear."""


class LLMStrategy(Strategy):
    """Strategy that uses an LLM to analyze price data and generate signals."""

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        super().__init__(name, params)
        self.lookback = self.params.get("lookback_candles", 50)
        self.indicators = self.params.get("indicators", ["rsi", "macd", "bbands", "ema_9", "ema_21"])
        self.provider = self.params.get("provider", os.getenv("AI_DEFAULT_PROVIDER", "ollama"))

    def on_tick(self, symbol: str, tick: dict) -> Signal | None:
        # LLM strategies run on candle close, not on every tick.
        return None

    def on_candle(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        if len(df) < self.lookback:
            return None

        recent = df.tail(self.lookback).copy()
        recent = add_indicators(recent, self.indicators)

        # Build context string for the LLM
        context = self._build_context(symbol, recent)

        try:
            response = self._call_llm(context)
            return self._parse_response(symbol, response, df["close"].iloc[-1])
        except Exception as e:
            print(f"LLM strategy error for {symbol}: {e}")
            return None

    def _build_context(self, symbol: str, df: pd.DataFrame) -> str:
        """Format recent data + indicators as text for LLM consumption."""
        # Last 10 candles with all columns (concise summary)
        last_n = df.tail(10)
        cols = [c for c in last_n.columns if c != "timestamp"]
        summary = last_n[cols].to_string(index=False, float_format="%.2f")

        latest = df.iloc[-1]
        indicator_summary = []
        for col in df.columns:
            if col not in ["timestamp", "open", "high", "low", "close", "volume", "open_interest"]:
                val = latest[col]
                if pd.notna(val):
                    indicator_summary.append(f"{col}: {val:.2f}")

        return (
            f"Stock: {symbol}\n"
            f"Last {len(last_n)} candles:\n{summary}\n\n"
            f"Current indicators:\n" + "\n".join(indicator_summary)
        )

    def _call_llm(self, context: str) -> str:
        """Call the configured LLM provider and return raw response text."""
        if self.provider == "ollama":
            return self._call_ollama(context)
        elif self.provider == "openai":
            return self._call_openai(context)
        elif self.provider == "anthropic":
            return self._call_anthropic(context)
        else:
            raise ValueError(f"Unknown AI provider: {self.provider}")

    def _call_ollama(self, context: str) -> str:
        import httpx
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3")
        resp = httpx.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                "stream": False,
                "format": "json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _call_openai(self, context: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return resp.choices[0].message.content

    def _call_anthropic(self, context: str) -> str:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": context}],
        )
        return resp.content[0].text

    def _parse_response(self, symbol: str, raw: str, current_price: float) -> Signal | None:
        """Parse LLM JSON response into a Signal."""
        try:
            # Strip markdown code fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(text)
        except json.JSONDecodeError:
            print(f"  LLM returned non-JSON: {raw[:200]}")
            return None

        action = data.get("action", "HOLD").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            return None

        confidence = float(data.get("confidence", 0))
        reason = data.get("reason", "LLM signal")

        return Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reason=f"[AI/{self.provider}] {reason}",
            strategy_name=self.name,
            price=current_price,
        )
