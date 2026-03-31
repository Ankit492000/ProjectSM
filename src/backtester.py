"""Comprehensive backtesting engine — walk-forward simulation with Indian market costs."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.strategy import Strategy, Signal


# ── Indian Market Transaction Costs ───────────────────────────────────────────
# Based on discount broker (Zerodha/Upstox) fee structure for NSE equity.

@dataclass
class CostModel:
    """Transaction cost model for Indian equity markets."""

    # Brokerage: flat per-order or percentage (whichever applies)
    brokerage_per_order: float = 20.0          # ₹20 flat per executed order (delivery)
    brokerage_pct: float = 0.0                 # % of turnover (0 for flat-fee brokers)
    brokerage_intraday_pct: float = 0.03       # 0.03% for intraday (if applicable)

    # STT (Securities Transaction Tax)
    stt_delivery_pct: float = 0.1              # 0.1% on both buy & sell for delivery
    stt_intraday_pct: float = 0.025            # 0.025% on sell side only for intraday

    # Exchange transaction charges (NSE)
    exchange_txn_pct: float = 0.00345          # ~0.00345% of turnover

    # GST: 18% on (brokerage + exchange txn charges)
    gst_pct: float = 18.0

    # SEBI turnover charges
    sebi_per_crore: float = 10.0               # ₹10 per crore of turnover

    # Stamp duty (buy-side, varies by state — using average)
    stamp_duty_pct: float = 0.015              # 0.015% on buy side

    # Slippage: simulated price impact
    slippage_pct: float = 0.05                 # 0.05% default slippage

    # Trade type
    is_intraday: bool = False

    def calculate(self, side: str, price: float, qty: int) -> float:
        """Calculate total transaction cost for one leg (buy or sell).

        Returns total cost in ₹.
        """
        turnover = price * qty

        # Brokerage
        if self.brokerage_pct > 0:
            brokerage = turnover * self.brokerage_pct / 100
        else:
            brokerage = self.brokerage_per_order

        # STT
        if self.is_intraday:
            stt = turnover * self.stt_intraday_pct / 100 if side == "SELL" else 0
        else:
            stt = turnover * self.stt_delivery_pct / 100

        # Exchange charges
        exchange = turnover * self.exchange_txn_pct / 100

        # GST on brokerage + exchange
        gst = (brokerage + exchange) * self.gst_pct / 100

        # SEBI
        sebi = turnover * self.sebi_per_crore / 1e7

        # Stamp duty (buy side only)
        stamp = turnover * self.stamp_duty_pct / 100 if side == "BUY" else 0

        # Slippage
        slippage = turnover * self.slippage_pct / 100

        return brokerage + stt + exchange + gst + sebi + stamp + slippage

    def total_round_trip(self, entry_price: float, exit_price: float, qty: int) -> float:
        """Total cost for a complete buy→sell round trip."""
        return (self.calculate("BUY", entry_price, qty)
                + self.calculate("SELL", exit_price, qty))


# ── Position Sizing ───────────────────────────────────────────────────────────

@dataclass
class PositionSizer:
    """Determines how many shares to buy given capital and risk parameters."""

    mode: str = "fixed_pct"          # fixed_pct | fixed_amount | risk_pct
    allocation_pct: float = 90.0     # % of capital to use (keeps 10% cash buffer)
    fixed_amount: float = 50_000     # ₹ amount per trade (for fixed_amount mode)
    risk_pct: float = 2.0            # % of capital risked per trade (for risk_pct mode)
    max_position_pct: float = 100.0  # max % of capital in a single position

    def calculate_qty(self, capital: float, price: float, stop_loss_pct: float = 0) -> int:
        """Return number of shares to buy."""
        if price <= 0:
            return 0

        if self.mode == "fixed_pct":
            amount = capital * min(self.allocation_pct, self.max_position_pct) / 100
        elif self.mode == "fixed_amount":
            amount = min(self.fixed_amount, capital * self.max_position_pct / 100)
        elif self.mode == "risk_pct" and stop_loss_pct > 0:
            # Risk = capital * risk_pct/100 = qty * price * stop_loss_pct/100
            risk_amount = capital * self.risk_pct / 100
            amount = risk_amount / (stop_loss_pct / 100) if stop_loss_pct > 0 else 0
            amount = min(amount, capital * self.max_position_pct / 100)
        else:
            amount = capital * min(self.allocation_pct, self.max_position_pct) / 100

        return max(int(amount / price), 0)


# ── Risk Management ──────────────────────────────────────────────────────────

@dataclass
class RiskManager:
    """Risk controls applied during backtest execution."""

    stop_loss_pct: float = 0.0           # % below entry to trigger stop-loss (0 = disabled)
    trailing_stop_pct: float = 0.0       # trailing stop as % from peak (0 = disabled)
    take_profit_pct: float = 0.0         # % above entry to take profit (0 = disabled)
    max_drawdown_pct: float = 0.0        # kill switch: stop trading if drawdown exceeds this (0 = disabled)
    max_holding_bars: int = 0            # max bars to hold a position (0 = unlimited)
    cooldown_bars: int = 0               # bars to wait after a trade before entering again

    def check_stop_loss(self, entry_price: float, current_low: float) -> bool:
        """True if stop-loss triggered."""
        if self.stop_loss_pct <= 0:
            return False
        stop_price = entry_price * (1 - self.stop_loss_pct / 100)
        return current_low <= stop_price

    def check_trailing_stop(self, peak_price: float, current_low: float) -> bool:
        """True if trailing stop triggered."""
        if self.trailing_stop_pct <= 0:
            return False
        trail_price = peak_price * (1 - self.trailing_stop_pct / 100)
        return current_low <= trail_price

    def check_take_profit(self, entry_price: float, current_high: float) -> bool:
        """True if take-profit triggered."""
        if self.take_profit_pct <= 0:
            return False
        tp_price = entry_price * (1 + self.take_profit_pct / 100)
        return current_high >= tp_price

    def check_max_drawdown(self, peak_capital: float, current_capital: float) -> bool:
        """True if max drawdown kill switch triggered."""
        if self.max_drawdown_pct <= 0 or peak_capital <= 0:
            return False
        dd = (peak_capital - current_capital) / peak_capital * 100
        return dd >= self.max_drawdown_pct

    def get_exit_price(self, entry_price: float, peak_price: float,
                       bar_high: float, bar_low: float, bar_close: float) -> float | None:
        """Return the estimated exit price if any risk rule triggers, else None.
        
        Priority: stop-loss > trailing stop > take-profit
        Uses conservative estimates (stop-loss at stop price, take-profit at TP price).
        """
        if self.stop_loss_pct > 0:
            stop_price = entry_price * (1 - self.stop_loss_pct / 100)
            if bar_low <= stop_price:
                return stop_price

        if self.trailing_stop_pct > 0:
            trail_price = peak_price * (1 - self.trailing_stop_pct / 100)
            if bar_low <= trail_price:
                return trail_price

        if self.take_profit_pct > 0:
            tp_price = entry_price * (1 + self.take_profit_pct / 100)
            if bar_high >= tp_price:
                return tp_price

        return None


# ── Open Position ─────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol: str
    side: str  # LONG (future: SHORT)
    entry_price: float
    qty: int
    entry_time: datetime
    entry_bar: int
    peak_price: float = 0.0  # highest price since entry (for trailing stop)

    def __post_init__(self):
        self.peak_price = self.entry_price

    def update_peak(self, high: float):
        self.peak_price = max(self.peak_price, high)

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.qty


# ── Trade Record ──────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    qty: int
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str  # signal | stop_loss | trailing_stop | take_profit | max_holding | end_of_data


# ── Backtest Report ───────────────────────────────────────────────────────────

@dataclass
class BacktestReport:
    """Comprehensive backtest results."""

    # Config
    symbol: str
    strategy_name: str
    initial_capital: float
    period: str  # "2024-01-01 to 2025-12-31"

    # Trade summary
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    expectancy: float  # avg PnL per trade

    # P&L
    total_gross_pnl: float
    total_costs: float
    total_net_pnl: float
    net_return_pct: float

    # Risk metrics
    max_drawdown_pct: float
    max_drawdown_amount: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    recovery_factor: float

    # Streaks
    max_win_streak: int
    max_loss_streak: int

    # Time
    avg_bars_held: float
    total_bars: int
    time_in_market_pct: float

    # Data
    equity_curve: pd.DataFrame   # bar-by-bar: timestamp, equity, drawdown
    trades: list[TradeRecord]

    def print_report(self):
        """Print formatted report to console."""
        w = 55
        print(f"\n{'='*w}")
        print(f"  BACKTEST REPORT")
        print(f"{'='*w}")
        print(f"  Strategy:        {self.strategy_name}")
        print(f"  Symbol:          {self.symbol}")
        print(f"  Period:          {self.period}")
        print(f"  Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"  Total Bars:      {self.total_bars}")

        print(f"\n{'─'*w}")
        print(f"  TRADE SUMMARY")
        print(f"{'─'*w}")
        print(f"  Total Trades:    {self.total_trades}")
        print(f"  Winners:         {self.winning_trades}  ({self.win_rate:.1%})")
        print(f"  Losers:          {self.losing_trades}")
        print(f"  Win Streak:      {self.max_win_streak}")
        print(f"  Loss Streak:     {self.max_loss_streak}")
        print(f"  Avg Bars Held:   {self.avg_bars_held:.1f}")
        print(f"  Time in Market:  {self.time_in_market_pct:.1%}")

        print(f"\n{'─'*w}")
        print(f"  PROFIT & LOSS")
        print(f"{'─'*w}")
        print(f"  Gross P&L:       ₹{self.total_gross_pnl:>12,.2f}")
        print(f"  Total Costs:     ₹{self.total_costs:>12,.2f}")
        print(f"  Net P&L:         ₹{self.total_net_pnl:>12,.2f}")
        print(f"  Net Return:       {self.net_return_pct:>11.2%}")
        print(f"  Avg Win:         ₹{self.avg_win:>12,.2f}")
        print(f"  Avg Loss:        ₹{self.avg_loss:>12,.2f}")
        print(f"  Largest Win:     ₹{self.largest_win:>12,.2f}")
        print(f"  Largest Loss:    ₹{self.largest_loss:>12,.2f}")
        print(f"  Profit Factor:    {self.profit_factor:>11.2f}")
        print(f"  Expectancy:      ₹{self.expectancy:>12,.2f} / trade")

        print(f"\n{'─'*w}")
        print(f"  RISK METRICS")
        print(f"{'─'*w}")
        print(f"  Max Drawdown:     {self.max_drawdown_pct:>11.2%}  (₹{self.max_drawdown_amount:,.0f})")
        print(f"  CAGR:             {self.cagr:>11.2%}")
        print(f"  Sharpe Ratio:     {self.sharpe_ratio:>11.2f}")
        print(f"  Sortino Ratio:    {self.sortino_ratio:>11.2f}")
        print(f"  Calmar Ratio:     {self.calmar_ratio:>11.2f}")
        print(f"  Recovery Factor:  {self.recovery_factor:>11.2f}")
        print(f"{'='*w}\n")

    def print_trades(self, max_rows: int = 50):
        """Print trade log."""
        if not self.trades:
            print("  No trades executed.")
            return

        print(f"\n{'─'*90}")
        print(f"  TRADE LOG (showing {'all' if len(self.trades) <= max_rows else f'last {max_rows}'} "
              f"of {len(self.trades)} trades)")
        print(f"{'─'*90}")
        print(f"  {'#':>3}  {'Entry Date':>12}  {'Exit Date':>12}  {'Entry':>10}  {'Exit':>10}"
              f"  {'Qty':>5}  {'Net P&L':>12}  {'Ret%':>7}  {'Exit Reason'}")
        print(f"  {'─'*3}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}"
              f"  {'─'*5}  {'─'*12}  {'─'*7}  {'─'*12}")

        trades_to_show = self.trades[-max_rows:]
        for i, t in enumerate(trades_to_show):
            idx = len(self.trades) - len(trades_to_show) + i + 1
            entry_d = t.entry_time.strftime("%Y-%m-%d") if isinstance(t.entry_time, datetime) else str(t.entry_time)[:10]
            exit_d = t.exit_time.strftime("%Y-%m-%d") if isinstance(t.exit_time, datetime) else str(t.exit_time)[:10]
            marker = "+" if t.net_pnl >= 0 else "-"
            print(f"  {idx:>3}  {entry_d:>12}  {exit_d:>12}  {t.entry_price:>10,.2f}"
                  f"  {t.exit_price:>10,.2f}  {t.qty:>5}  {t.net_pnl:>+12,.2f}"
                  f"  {t.return_pct:>+6.2f}%  {t.exit_reason}")
        print()


# ── Backtest Engine ───────────────────────────────────────────────────────────

class BacktestEngine:
    """Walk-forward backtest engine that feeds historical data through any Strategy object.

    Usage:
        engine = BacktestEngine(
            strategy=my_strategy,
            initial_capital=500_000,
            cost_model=CostModel(),
            position_sizer=PositionSizer(mode="fixed_pct", allocation_pct=90),
            risk_manager=RiskManager(stop_loss_pct=3, trailing_stop_pct=5),
        )
        report = engine.run(symbol="RELIANCE", df=historical_df)
        report.print_report()
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 100_000,
        cost_model: CostModel | None = None,
        position_sizer: PositionSizer | None = None,
        risk_manager: RiskManager | None = None,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.sizer = position_sizer or PositionSizer()
        self.risk = risk_manager or RiskManager()

    def run(self, symbol: str, df: pd.DataFrame, warmup_bars: int = 50) -> BacktestReport:
        """Run backtest on historical OHLCV DataFrame.

        df must have columns: timestamp, open, high, low, close, volume
        warmup_bars: number of initial bars to skip (for indicator warm-up)
        """
        df = df.reset_index(drop=True).copy()
        n = len(df)

        if n < warmup_bars + 5:
            raise ValueError(f"Not enough data: {n} bars, need at least {warmup_bars + 5}")

        capital = self.initial_capital
        peak_capital = capital
        position: Position | None = None
        trades: list[TradeRecord] = []
        cooldown_until = 0  # bar index until which we can't enter
        bars_in_market = 0
        killed = False  # max drawdown kill switch

        # Equity curve tracking (one row per bar)
        eq_timestamps = []
        eq_values = []
        eq_drawdowns = []

        for bar in range(n):
            ts = df["timestamp"].iloc[bar]
            bar_open = df["open"].iloc[bar]
            bar_high = df["high"].iloc[bar]
            bar_low = df["low"].iloc[bar]
            bar_close = df["close"].iloc[bar]

            # ── Update peak price for trailing stop ──
            if position:
                position.update_peak(bar_high)
                bars_in_market += 1

            # ── Risk management checks (before strategy) ──
            if position and not killed:
                exit_price = self.risk.get_exit_price(
                    position.entry_price, position.peak_price,
                    bar_high, bar_low, bar_close,
                )
                exit_reason = None
                if exit_price is not None:
                    # Determine which risk rule triggered
                    sl_price = position.entry_price * (1 - self.risk.stop_loss_pct / 100) if self.risk.stop_loss_pct > 0 else 0
                    trail_price = position.peak_price * (1 - self.risk.trailing_stop_pct / 100) if self.risk.trailing_stop_pct > 0 else 0
                    tp_price = position.entry_price * (1 + self.risk.take_profit_pct / 100) if self.risk.take_profit_pct > 0 else float('inf')

                    if self.risk.stop_loss_pct > 0 and bar_low <= sl_price:
                        exit_reason = "stop_loss"
                    elif self.risk.trailing_stop_pct > 0 and bar_low <= trail_price:
                        exit_reason = "trailing_stop"
                    elif self.risk.take_profit_pct > 0 and bar_high >= tp_price:
                        exit_reason = "take_profit"

                    if exit_reason:
                        trades.append(self._close_position(position, exit_price, ts, bar, exit_reason))
                        capital += trades[-1].net_pnl
                        position = None
                        cooldown_until = bar + self.risk.cooldown_bars

                # Max holding bars
                if position and self.risk.max_holding_bars > 0:
                    if (bar - position.entry_bar) >= self.risk.max_holding_bars:
                        trades.append(self._close_position(position, bar_close, ts, bar, "max_holding"))
                        capital += trades[-1].net_pnl
                        position = None
                        cooldown_until = bar + self.risk.cooldown_bars

            # ── Max drawdown kill switch ──
            if not killed and self.risk.check_max_drawdown(peak_capital, capital):
                if position:
                    trades.append(self._close_position(position, bar_close, ts, bar, "max_dd_kill"))
                    capital += trades[-1].net_pnl
                    position = None
                killed = True

            # ── Strategy signal (only after warmup, if not killed) ──
            if bar >= warmup_bars and not killed:
                window = df.iloc[:bar + 1]  # expanding window up to current bar
                signal = self.strategy.on_candle(symbol, window)

                if signal and signal.action == "BUY" and position is None and bar >= cooldown_until:
                    # Enter long position at next bar's open (more realistic)
                    # But since we're at bar close, use close as proxy
                    entry_price = bar_close
                    qty = self.sizer.calculate_qty(capital, entry_price, self.risk.stop_loss_pct)
                    if qty > 0:
                        entry_cost = self.cost_model.calculate("BUY", entry_price, qty)
                        if (entry_price * qty + entry_cost) <= capital:
                            position = Position(
                                symbol=symbol, side="LONG",
                                entry_price=entry_price, qty=qty,
                                entry_time=ts, entry_bar=bar,
                            )
                            capital -= entry_cost  # deduct entry costs upfront

                elif signal and signal.action == "SELL" and position is not None:
                    trades.append(self._close_position(position, bar_close, ts, bar, "signal"))
                    capital += trades[-1].net_pnl
                    position = None
                    cooldown_until = bar + self.risk.cooldown_bars

            # ── Record equity ──
            equity = capital + (position.unrealized_pnl(bar_close) if position else 0)
            peak_capital = max(peak_capital, equity)
            dd = (peak_capital - equity) / peak_capital if peak_capital > 0 else 0

            eq_timestamps.append(ts)
            eq_values.append(equity)
            eq_drawdowns.append(dd)

        # ── Close any remaining position at end of data ──
        if position:
            final_price = df["close"].iloc[-1]
            final_ts = df["timestamp"].iloc[-1]
            trades.append(self._close_position(position, final_price, final_ts, n - 1, "end_of_data"))
            capital += trades[-1].net_pnl

        # ── Build equity curve DataFrame ──
        equity_df = pd.DataFrame({
            "timestamp": eq_timestamps,
            "equity": eq_values,
            "drawdown": eq_drawdowns,
        })

        # ── Compute metrics ──
        return self._compute_report(symbol, df, trades, equity_df)

    def _close_position(
        self, pos: Position, exit_price: float,
        exit_time: datetime, exit_bar: int, reason: str,
    ) -> TradeRecord:
        """Close a position and create a TradeRecord."""
        gross_pnl = (exit_price - pos.entry_price) * pos.qty
        exit_cost = self.cost_model.calculate("SELL", exit_price, pos.qty)
        net_pnl = gross_pnl - exit_cost  # entry cost already deducted from capital
        return_pct = (exit_price - pos.entry_price) / pos.entry_price * 100

        return TradeRecord(
            symbol=pos.symbol, side=pos.side,
            entry_time=pos.entry_time, exit_time=exit_time,
            entry_price=pos.entry_price, exit_price=exit_price,
            qty=pos.qty,
            gross_pnl=gross_pnl, costs=exit_cost, net_pnl=net_pnl,
            return_pct=return_pct,
            bars_held=exit_bar - pos.entry_bar,
            exit_reason=reason,
        )

    def _compute_report(
        self, symbol: str, df: pd.DataFrame,
        trades: list[TradeRecord], equity_df: pd.DataFrame,
    ) -> BacktestReport:
        """Compute all metrics from trade list and equity curve."""
        n = len(df)
        period_start = df["timestamp"].iloc[0]
        period_end = df["timestamp"].iloc[-1]
        period_str = f"{str(period_start)[:10]} to {str(period_end)[:10]}"

        # Basic trade stats
        total = len(trades)
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl <= 0]

        win_rate = len(winners) / total if total > 0 else 0
        avg_win = np.mean([t.net_pnl for t in winners]) if winners else 0
        avg_loss = np.mean([t.net_pnl for t in losers]) if losers else 0
        largest_win = max((t.net_pnl for t in trades), default=0)
        largest_loss = min((t.net_pnl for t in trades), default=0)

        gross_wins = sum(t.net_pnl for t in winners)
        gross_losses = abs(sum(t.net_pnl for t in losers))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf') if gross_wins > 0 else 0

        total_gross = sum(t.gross_pnl for t in trades)
        total_costs = sum(t.costs for t in trades)
        # Add entry costs (not tracked in TradeRecord.costs which is exit-only)
        # Entry costs were deducted from capital directly — approximate from cost model
        for t in trades:
            total_costs += self.cost_model.calculate("BUY", t.entry_price, t.qty)
        total_net = sum(t.net_pnl for t in trades) - sum(
            self.cost_model.calculate("BUY", t.entry_price, t.qty) for t in trades
        )
        # More accurate: use equity curve
        final_equity = equity_df["equity"].iloc[-1] if not equity_df.empty else self.initial_capital
        total_net = final_equity - self.initial_capital
        net_return = total_net / self.initial_capital

        expectancy = total_net / total if total > 0 else 0

        # Drawdown metrics
        max_dd_pct = equity_df["drawdown"].max() if not equity_df.empty else 0
        eq_series = equity_df["equity"]
        peak_series = eq_series.cummax()
        dd_amount_series = peak_series - eq_series
        max_dd_amount = dd_amount_series.max() if not dd_amount_series.empty else 0

        # CAGR
        try:
            dt_start = pd.to_datetime(period_start)
            dt_end = pd.to_datetime(period_end)
            years = (dt_end - dt_start).days / 365.25
            if years > 0 and final_equity > 0:
                cagr = (final_equity / self.initial_capital) ** (1 / years) - 1
            else:
                cagr = 0
        except Exception:
            cagr = 0

        # Daily returns from equity curve (for Sharpe/Sortino)
        if len(eq_series) > 1:
            daily_returns = eq_series.pct_change().dropna()
            mean_ret = daily_returns.mean()
            std_ret = daily_returns.std()
            downside_ret = daily_returns[daily_returns < 0].std()

            sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
            sortino = (mean_ret / downside_ret) * np.sqrt(252) if downside_ret > 0 else 0
        else:
            sharpe = sortino = 0

        calmar = cagr / max_dd_pct if max_dd_pct > 0 else 0
        recovery = total_net / max_dd_amount if max_dd_amount > 0 else 0

        # Streaks
        max_win_streak = max_loss_streak = 0
        cur_win = cur_loss = 0
        for t in trades:
            if t.net_pnl > 0:
                cur_win += 1
                cur_loss = 0
                max_win_streak = max(max_win_streak, cur_win)
            else:
                cur_loss += 1
                cur_win = 0
                max_loss_streak = max(max_loss_streak, cur_loss)

        # Time in market
        bars_in_market = sum(t.bars_held for t in trades)
        time_in_market = bars_in_market / n if n > 0 else 0

        avg_bars = np.mean([t.bars_held for t in trades]) if trades else 0

        return BacktestReport(
            symbol=symbol,
            strategy_name=self.strategy.name,
            initial_capital=self.initial_capital,
            period=period_str,
            total_trades=total,
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            total_gross_pnl=total_gross,
            total_costs=total_costs,
            total_net_pnl=total_net,
            net_return_pct=net_return,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_amount=max_dd_amount,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            recovery_factor=recovery,
            max_win_streak=max_win_streak,
            max_loss_streak=max_loss_streak,
            avg_bars_held=avg_bars,
            total_bars=n,
            time_in_market_pct=time_in_market,
            equity_curve=equity_df,
            trades=trades,
        )


# ── Multi-Symbol Backtest ─────────────────────────────────────────────────────

def run_multi_symbol_backtest(
    strategy: Strategy,
    symbols: list[tuple[str, str, str]],  # [(symbol, exchange, interval), ...]
    initial_capital: float = 500_000,
    cost_model: CostModel | None = None,
    position_sizer: PositionSizer | None = None,
    risk_manager: RiskManager | None = None,
    warmup_bars: int = 50,
) -> list[BacktestReport]:
    """Run the same strategy across multiple symbols and return individual reports.

    Capital is split equally across symbols.
    """
    from src import storage

    n_symbols = len(symbols)
    capital_per_symbol = initial_capital / n_symbols if n_symbols > 0 else initial_capital

    reports = []
    for sym, exchange, interval in symbols:
        df = storage.load_candles(sym, exchange, interval)
        if df.empty:
            print(f"  Skipping {sym} — no data found")
            continue

        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=capital_per_symbol,
            cost_model=cost_model,
            position_sizer=position_sizer,
            risk_manager=risk_manager,
        )
        try:
            report = engine.run(sym, df, warmup_bars=warmup_bars)
            reports.append(report)
        except Exception as e:
            print(f"  Backtest failed for {sym}: {e}")

    return reports


def print_multi_symbol_summary(reports: list[BacktestReport]):
    """Print an aggregated summary table across multiple symbols."""
    if not reports:
        print("No reports to summarize.")
        return

    total_cap = sum(r.initial_capital for r in reports)
    total_net = sum(r.total_net_pnl for r in reports)
    total_trades = sum(r.total_trades for r in reports)
    total_winners = sum(r.winning_trades for r in reports)

    w = 70
    print(f"\n{'='*w}")
    print(f"  MULTI-SYMBOL BACKTEST SUMMARY")
    print(f"  Strategy: {reports[0].strategy_name}")
    print(f"{'='*w}")
    print(f"  {'Symbol':<15} {'Trades':>7} {'Win%':>7} {'Net P&L':>14} {'Return':>9} {'MaxDD':>9} {'Sharpe':>8}")
    print(f"  {'─'*15} {'─'*7} {'─'*7} {'─'*14} {'─'*9} {'─'*9} {'─'*8}")

    for r in reports:
        print(f"  {r.symbol:<15} {r.total_trades:>7} {r.win_rate:>6.1%}"
              f" {r.total_net_pnl:>+14,.0f} {r.net_return_pct:>+8.2%}"
              f" {r.max_drawdown_pct:>8.2%} {r.sharpe_ratio:>8.2f}")

    print(f"  {'─'*15} {'─'*7} {'─'*7} {'─'*14} {'─'*9} {'─'*9} {'─'*8}")
    overall_wr = total_winners / total_trades if total_trades > 0 else 0
    overall_ret = total_net / total_cap if total_cap > 0 else 0
    print(f"  {'TOTAL':<15} {total_trades:>7} {overall_wr:>6.1%}"
          f" {total_net:>+14,.0f} {overall_ret:>+8.2%}")
    print(f"{'='*w}\n")
