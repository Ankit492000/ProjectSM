"""Strategy engine — base class, Signal dataclass, and StrategyRunner."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass
class Signal:
    """A buy/sell/hold signal emitted by a strategy."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 to 1.0
    reason: str
    strategy_name: str
    price: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def passes_threshold(self, threshold: float) -> bool:
        return self.action != "HOLD" and self.confidence >= threshold


class Strategy(ABC):
    """Base class for all strategies (rule-based and AI-based)."""

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def on_tick(self, symbol: str, tick: dict) -> Signal | None:
        """Process a single real-time tick. Return a Signal or None."""
        ...

    @abstractmethod
    def on_candle(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        """Process updated candle DataFrame. Return a Signal or None."""
        ...


class StrategyRunner:
    """Loads strategies from config, dispatches ticks/candles, collects signals."""

    def __init__(self, config_path: str | Path | None = None):
        self.strategies: list[Strategy] = []
        self._config_path = Path(config_path or
                                  Path(__file__).resolve().parent.parent / "config" / "strategies.yaml")
        self._load_settings()
        self._load_strategies()

    def _load_settings(self):
        settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
        with open(settings_path) as f:
            cfg = yaml.safe_load(f)
        self.confidence_threshold = cfg.get("strategy", {}).get("signal_confidence_threshold", 0.5)

    def _load_strategies(self):
        with open(self._config_path) as f:
            cfg = yaml.safe_load(f)

        for entry in cfg.get("strategies", []):
            if not entry.get("enabled", True):
                continue
            try:
                mod = importlib.import_module(entry["module"])
                cls = getattr(mod, entry["class"])
                strategy = cls(name=entry["name"], params=entry.get("params", {}))
                self.strategies.append(strategy)
                print(f"  Loaded strategy: {entry['name']} ({entry['class']})")
            except Exception as e:
                print(f"  Failed to load strategy {entry['name']}: {e}")

    def process_tick(self, symbol: str, tick: dict) -> list[Signal]:
        """Run all strategies on a tick, return signals that pass threshold."""
        signals = []
        for strategy in self.strategies:
            try:
                sig = strategy.on_tick(symbol, tick)
                if sig and sig.passes_threshold(self.confidence_threshold):
                    signals.append(sig)
            except Exception as e:
                print(f"Strategy {strategy.name} error on tick: {e}")
        return signals

    def process_candle(self, symbol: str, df: pd.DataFrame) -> list[Signal]:
        """Run all strategies on candle data, return signals that pass threshold."""
        signals = []
        for strategy in self.strategies:
            try:
                sig = strategy.on_candle(symbol, df)
                if sig and sig.passes_threshold(self.confidence_threshold):
                    signals.append(sig)
            except Exception as e:
                print(f"Strategy {strategy.name} error on candle: {e}")
        return signals
