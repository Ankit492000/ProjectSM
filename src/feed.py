"""Upstox Market Data Feed v3 — WebSocket client with protobuf decoding."""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import httpx
import pandas as pd
import websockets

from src.auth import get_access_token
from src import storage

FEED_AUTH_URL = "https://api.upstox.com/v3/feed/market-data-feed/authorize"

# Try to import compiled protobuf; fall back to JSON parsing if unavailable
_use_protobuf = False
try:
    from proto import MarketDataFeed_pb2
    _use_protobuf = True
except ImportError:
    pass


class MarketFeed:
    """Async WebSocket client for Upstox v3 market data feed."""

    def __init__(
        self,
        instruments: list[str],
        mode: str = "ltpc",
        on_tick: Callable[[str, dict], Any] | None = None,
        store_ticks: bool = True,
        tick_flush_interval: int = 5,
    ):
        """
        instruments: list of instrument_keys (e.g. ["NSE_EQ|INE669E01016"])
        mode: ltpc | full | option_greeks
        on_tick: async or sync callback(symbol, tick_dict) called for each tick
        """
        self.instruments = instruments
        self.mode = mode
        self.on_tick = on_tick
        self.store_ticks = store_ticks
        self.tick_flush_interval = tick_flush_interval

        self._ws = None
        self._running = False
        self._tick_buffer: dict[str, list[dict]] = {}

    async def _get_ws_url(self) -> str:
        """Get authorized WebSocket URL from Upstox."""
        token = get_access_token()
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                FEED_AUTH_URL,
                headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"]["authorized_redirect_uri"]

    def _build_subscription_msg(self, method: str = "sub") -> bytes | str:
        """Build subscription request (binary for protobuf, JSON fallback)."""
        if _use_protobuf:
            req = MarketDataFeed_pb2.FeedRequest()
            req.method = method
            req.mode = self.mode
            req.instrumentKeys.extend(self.instruments)
            return req.SerializeToString()
        else:
            return json.dumps({
                "guid": "projectsm",
                "method": method,
                "data": {
                    "mode": self.mode,
                    "instrumentKeys": self.instruments,
                },
            })

    def _decode_message(self, raw: bytes | str) -> list[dict]:
        """Decode a feed message into a list of tick dicts."""
        if isinstance(raw, bytes) and _use_protobuf:
            return self._decode_protobuf(raw)
        elif isinstance(raw, str):
            return self._decode_json(raw)
        else:
            # Binary but no protobuf compiled — try JSON decode
            try:
                return self._decode_json(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return []

    def _decode_protobuf(self, raw: bytes) -> list[dict]:
        """Decode protobuf feed message."""
        feed = MarketDataFeed_pb2.FeedResponse()
        feed.ParseFromString(raw)
        ticks = []
        for key, ff in feed.feeds.items():
            tick = {"instrument_key": key, "timestamp": datetime.now().isoformat()}
            if ff.HasField("ltpc"):
                tick.update({
                    "ltp": ff.ltpc.ltp,
                    "ltt": ff.ltpc.ltt,
                    "ltq": ff.ltpc.ltq,
                    "cp": ff.ltpc.cp,
                })
            if ff.HasField("marketFF"):
                mff = ff.marketFF
                tick.update({
                    "ltp": mff.ltpc.ltp if mff.HasField("ltpc") else tick.get("ltp"),
                    "open": mff.marketOHLC[0].ohlc.open if mff.marketOHLC else None,
                    "high": mff.marketOHLC[0].ohlc.high if mff.marketOHLC else None,
                    "low": mff.marketOHLC[0].ohlc.low if mff.marketOHLC else None,
                    "close": mff.marketOHLC[0].ohlc.close if mff.marketOHLC else None,
                    "volume": mff.marketOHLC[0].ohlc.volume if mff.marketOHLC else None,
                })
            ticks.append(tick)
        return ticks

    def _decode_json(self, raw: str) -> list[dict]:
        """Decode JSON feed message (fallback)."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        ticks = []
        feeds = data.get("feeds", {})
        for key, feed_data in feeds.items():
            tick = {"instrument_key": key, "timestamp": datetime.now().isoformat()}
            ff = feed_data.get("ff", feed_data.get("ltpc", {}))
            if isinstance(ff, dict):
                ltpc = ff.get("ltpc", ff)
                tick["ltp"] = ltpc.get("ltp") or ltpc.get("ltP")
                tick["cp"] = ltpc.get("cp") or ltpc.get("cP")
            ticks.append(tick)
        return ticks

    def _buffer_tick(self, tick: dict) -> None:
        """Add tick to in-memory buffer for periodic flush."""
        key = tick.get("instrument_key", "unknown")
        if key not in self._tick_buffer:
            self._tick_buffer[key] = []
        self._tick_buffer[key].append(tick)

    async def _flush_ticks(self) -> None:
        """Flush tick buffers to Parquet storage."""
        today = date.today()
        for symbol, ticks in self._tick_buffer.items():
            if ticks:
                df = pd.DataFrame(ticks)
                storage.append_ticks(symbol, today, df)
        self._tick_buffer.clear()

    async def _flush_loop(self) -> None:
        """Periodically flush ticks to storage."""
        while self._running:
            await asyncio.sleep(self.tick_flush_interval)
            await self._flush_ticks()

    async def connect(self) -> None:
        """Connect to WebSocket and start receiving ticks."""
        self._running = True
        reconnect_delay = 5
        max_attempts = 10
        attempt = 0

        while self._running and attempt < max_attempts:
            try:
                ws_url = await self._get_ws_url()
                print(f"Connecting to Upstox WebSocket...")

                async with websockets.connect(ws_url, ping_interval=25) as ws:
                    self._ws = ws
                    attempt = 0  # reset on successful connect
                    print("Connected. Subscribing to instruments...")

                    sub_msg = self._build_subscription_msg("sub")
                    await ws.send(sub_msg)

                    # Start tick flush loop
                    flush_task = asyncio.create_task(self._flush_loop())

                    try:
                        async for message in ws:
                            ticks = self._decode_message(message)
                            for tick in ticks:
                                if self.store_ticks:
                                    self._buffer_tick(tick)
                                if self.on_tick:
                                    if asyncio.iscoroutinefunction(self.on_tick):
                                        await self.on_tick(tick.get("instrument_key", ""), tick)
                                    else:
                                        self.on_tick(tick.get("instrument_key", ""), tick)
                    finally:
                        flush_task.cancel()
                        await self._flush_ticks()  # final flush

            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                attempt += 1
                print(f"WebSocket disconnected ({e}). Reconnecting in {reconnect_delay}s... "
                      f"(attempt {attempt}/{max_attempts})")
                await asyncio.sleep(reconnect_delay)

        if attempt >= max_attempts:
            print("Max reconnect attempts reached. Feed stopped.")
        self._running = False

    async def disconnect(self) -> None:
        """Gracefully disconnect."""
        self._running = False
        if self._ws:
            await self._ws.close()
