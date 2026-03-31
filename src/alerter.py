"""Telegram bot alerter — sends buy/sell signals to configured chat(s)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from telegram import Bot
from telegram.constants import ParseMode

from src.strategy import Signal

load_dotenv()

_cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f).get("telegram", {})

ENABLED = _cfg.get("enabled", True)
ALERT_FORMAT = _cfg.get("alert_format", "{action} — {symbol} | {strategy} | {confidence:.0%} | ₹{price}")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_IDS = [cid.strip() for cid in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if cid.strip()]


class Alerter:
    """Sends trading signals to Telegram."""

    def __init__(self):
        if not BOT_TOKEN:
            print("Warning: TELEGRAM_BOT_TOKEN not set. Alerts will be printed to console only.")
        self._bot = Bot(token=BOT_TOKEN) if BOT_TOKEN else None

    def format_signal(self, signal: Signal) -> str:
        """Format a Signal into a readable Telegram message."""
        return ALERT_FORMAT.format(
            action=signal.action,
            symbol=signal.symbol,
            strategy=signal.strategy_name,
            confidence=signal.confidence,
            price=signal.price,
            reason=signal.reason,
            timestamp=signal.timestamp.strftime("%H:%M:%S"),
        )

    async def send_alert(self, signal: Signal) -> None:
        """Send alert to all configured Telegram chats."""
        text = self.format_signal(signal)

        if not ENABLED:
            print(f"[ALERT DISABLED] {text}")
            return

        if not self._bot or not CHAT_IDS:
            print(f"[ALERT] {text}")
            return

        for chat_id in CHAT_IDS:
            try:
                await self._bot.send_message(chat_id=chat_id, text=text)
            except Exception as e:
                print(f"Failed to send Telegram alert to {chat_id}: {e}")
                # Respect Telegram rate limits
                await asyncio.sleep(1)

    async def send_text(self, text: str) -> None:
        """Send arbitrary text message to all chats."""
        if not self._bot or not CHAT_IDS:
            print(f"[MSG] {text}")
            return

        for chat_id in CHAT_IDS:
            try:
                await self._bot.send_message(chat_id=chat_id, text=text)
            except Exception as e:
                print(f"Failed to send Telegram message to {chat_id}: {e}")
                await asyncio.sleep(1)

    async def send_startup_message(self, instruments: list[str], strategies: list[str]) -> None:
        """Send a startup notification."""
        text = (
            "🟢 ProjectSM Live Mode Started\n"
            f"Instruments: {', '.join(instruments)}\n"
            f"Strategies: {', '.join(strategies)}\n"
        )
        await self.send_text(text)
