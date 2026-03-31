"""Historical data analysis — technical indicators, pattern detection, backtesting helpers."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import src.indicators as ta


# ── Technical Indicators ──────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame, indicators: list[str] | None = None) -> pd.DataFrame:
    """Add technical indicators to OHLCV DataFrame (in-place columns).
    
    Supported: rsi, macd, bbands, ema_9, ema_21, sma_50, sma_200, atr, vwap
    If indicators is None, adds a default set.
    """
    if df.empty:
        return df

    if indicators is None:
        indicators = ["rsi", "macd", "bbands", "ema_9", "ema_21"]

    for ind in indicators:
        if ind == "rsi":
            df["rsi"] = ta.rsi(df["close"], length=14)
        elif ind == "macd":
            macd = ta.macd(df["close"])
            if macd is not None:
                df = pd.concat([df, macd], axis=1)
        elif ind == "bbands":
            bb = ta.bbands(df["close"], length=20)
            if bb is not None:
                df = pd.concat([df, bb], axis=1)
        elif ind.startswith("ema_"):
            period = int(ind.split("_")[1])
            df[f"ema_{period}"] = ta.ema(df["close"], length=period)
        elif ind.startswith("sma_"):
            period = int(ind.split("_")[1])
            df[f"sma_{period}"] = ta.sma(df["close"], length=period)
        elif ind == "atr":
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        elif ind == "vwap" and "volume" in df.columns:
            df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

    return df


# ── Backtesting Helper ────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    trades: pd.DataFrame


def backtest(
    df: pd.DataFrame,
    buy_signals: pd.Series,
    sell_signals: pd.Series,
    initial_capital: float = 100_000,
    position_size: float = 1.0,
) -> BacktestResult:
    """Simple long-only walk-forward backtest.
    
    buy_signals / sell_signals: boolean Series aligned with df index.
    Returns BacktestResult with trade log and metrics.
    """
    trades = []
    position = None
    capital = initial_capital
    peak_capital = initial_capital

    for i in range(len(df)):
        if buy_signals.iloc[i] and position is None:
            entry_price = df["close"].iloc[i]
            qty = int((capital * position_size) / entry_price)
            if qty > 0:
                position = {"entry_idx": i, "entry_price": entry_price, "qty": qty,
                            "entry_time": df["timestamp"].iloc[i]}

        elif sell_signals.iloc[i] and position is not None:
            exit_price = df["close"].iloc[i]
            pnl = (exit_price - position["entry_price"]) * position["qty"]
            capital += pnl
            trades.append({
                "entry_time": position["entry_time"],
                "exit_time": df["timestamp"].iloc[i],
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "qty": position["qty"],
                "pnl": pnl,
            })
            position = None

        peak_capital = max(peak_capital, capital)

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "entry_price", "exit_price", "qty", "pnl"]
    )

    total_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0.0
    winning = trades_df[trades_df["pnl"] > 0] if not trades_df.empty else trades_df
    losing = trades_df[trades_df["pnl"] <= 0] if not trades_df.empty else trades_df

    # Sharpe ratio (annualized, assuming daily returns)
    if not trades_df.empty and len(trades_df) > 1:
        returns = trades_df["pnl"] / initial_capital
        sharpe = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown from equity curve
    if trades_df.empty:
        max_dd = 0.0
    else:
        equity = [initial_capital]
        for _, t in trades_df.iterrows():
            equity.append(equity[-1] + t["pnl"])
        eq_series = pd.Series(equity)
        peak = eq_series.cummax()
        drawdown = (eq_series - peak) / peak
        max_dd = abs(drawdown.min())

    return BacktestResult(
        total_trades=len(trades_df),
        winning_trades=len(winning),
        losing_trades=len(losing),
        total_pnl=total_pnl,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        win_rate=len(winning) / len(trades_df) if len(trades_df) > 0 else 0.0,
        trades=trades_df,
    )


def print_backtest_report(result: BacktestResult, strategy_name: str = "Strategy") -> None:
    """Print a readable backtest summary."""
    print(f"\n{'='*50}")
    print(f"  Backtest Report: {strategy_name}")
    print(f"{'='*50}")
    print(f"  Total Trades:   {result.total_trades}")
    print(f"  Win Rate:       {result.win_rate:.1%}")
    print(f"  Total PnL:      ₹{result.total_pnl:,.2f}")
    print(f"  Max Drawdown:   {result.max_drawdown:.2%}")
    print(f"  Sharpe Ratio:   {result.sharpe_ratio:.2f}")
    print(f"  Winners:        {result.winning_trades}")
    print(f"  Losers:         {result.losing_trades}")
    print(f"{'='*50}\n")
