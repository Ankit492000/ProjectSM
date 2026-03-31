"""ProjectSM — CLI entry point for historical analysis and live trading modes."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime

import pandas as pd


def cmd_auth():
    """Run Upstox OAuth2 flow to obtain and cache access token."""
    from src.auth import run_auth_flow
    run_auth_flow()


def cmd_fetch(args):
    """Fetch historical data for a symbol and store it."""
    from src.data_fetcher import fetch_and_store, fetch_instruments

    if args.refresh_instruments:
        fetch_instruments(args.exchange)

    df = fetch_and_store(
        symbol=args.symbol,
        exchange=args.exchange,
        interval=args.interval,
        unit=args.unit,
        from_date=args.from_date,
        to_date=args.to_date,
    )
    print(f"Fetched {len(df)} candles for {args.symbol}")


def cmd_analyze(args):
    """Run analysis + backtest on historical data."""
    from src import storage
    from src.analyzer import add_indicators, backtest, print_backtest_report

    df = storage.load_candles(args.symbol, args.exchange, args.interval)
    if df.empty:
        print(f"No data found for {args.symbol}/{args.exchange}/{args.interval}. Run 'fetch' first.")
        return

    df = add_indicators(df)
    print(f"Loaded {len(df)} candles with indicators for {args.symbol}")

    # Run default EMA crossover backtest as demo
    import pandas_ta as ta
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)

    buy_signals = (df["ema_9"].shift(1) <= df["ema_21"].shift(1)) & (df["ema_9"] > df["ema_21"])
    sell_signals = (df["ema_9"].shift(1) >= df["ema_21"].shift(1)) & (df["ema_9"] < df["ema_21"])

    result = backtest(df, buy_signals, sell_signals)
    print_backtest_report(result, f"EMA 9/21 Crossover — {args.symbol}")


def cmd_fundamentals(args):
    """Fetch and display fundamental analysis for a stock."""
    from src.fundamentals import get_fundamentals, FundamentalScore, print_fundamental_report

    tickers = [t.strip() for t in args.tickers.split(",")]
    for ticker in tickers:
        # Ensure .NS / .BO suffix
        if "." not in ticker:
            ticker = ticker + (".NS" if args.exchange == "NSE" else ".BO")
        data = get_fundamentals(ticker, refresh=args.refresh)
        score = FundamentalScore.compute(data, sector_median_pe=args.sector_pe)
        print_fundamental_report(ticker, data, score)


def cmd_backtest(args):
    """Run full backtest: load strategy, simulate on historical data with costs + risk management."""
    import importlib
    from src import storage
    from src.backtester import (
        BacktestEngine, CostModel, PositionSizer, RiskManager,
        run_multi_symbol_backtest, print_multi_symbol_summary,
    )

    # ── Load strategy ──
    if args.strategy_module and args.strategy_class:
        mod = importlib.import_module(args.strategy_module)
        cls = getattr(mod, args.strategy_class)
        strategy = cls(name=args.strategy_class, params=_parse_params(args.params))
    else:
        # Default: load from strategies.yaml by name
        from src.strategy import StrategyRunner
        runner = StrategyRunner()
        matches = [s for s in runner.strategies if s.name == args.strategy]
        if not matches:
            print(f"Strategy '{args.strategy}' not found in strategies.yaml. Available: "
                  f"{[s.name for s in runner.strategies]}")
            return
        strategy = matches[0]

    # ── Config objects ──
    cost_model = CostModel(
        slippage_pct=args.slippage,
        is_intraday=args.intraday,
    )
    sizer = PositionSizer(
        mode=args.sizing_mode,
        allocation_pct=args.allocation_pct,
        fixed_amount=args.fixed_amount,
    )
    risk_mgr = RiskManager(
        stop_loss_pct=args.stop_loss,
        trailing_stop_pct=args.trailing_stop,
        take_profit_pct=args.take_profit,
        max_drawdown_pct=args.max_drawdown,
        max_holding_bars=args.max_holding,
        cooldown_bars=args.cooldown,
    )

    # ── Single or multi-symbol ──
    symbols_list = [s.strip() for s in args.symbols.split(",")]

    if len(symbols_list) == 1:
        sym = symbols_list[0]
        df = storage.load_candles(sym, args.exchange, args.interval)
        if df.empty:
            print(f"No data for {sym}/{args.exchange}/{args.interval}. Run 'fetch' first.")
            return

        if args.from_date:
            df = df[df["timestamp"] >= pd.to_datetime(args.from_date)]
        if args.to_date:
            df = df[df["timestamp"] <= pd.to_datetime(args.to_date)]

        print(f"Running backtest: {strategy.name} on {sym} ({len(df)} bars, "
              f"₹{args.capital:,.0f} capital)")

        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=args.capital,
            cost_model=cost_model,
            position_sizer=sizer,
            risk_manager=risk_mgr,
        )
        report = engine.run(sym, df, warmup_bars=args.warmup)
        report.print_report()
        if args.show_trades:
            report.print_trades()
    else:
        sym_tuples = [(s, args.exchange, args.interval) for s in symbols_list]
        print(f"Running multi-symbol backtest: {strategy.name} across {len(sym_tuples)} symbols "
              f"(₹{args.capital:,.0f} total capital)")

        reports = run_multi_symbol_backtest(
            strategy=strategy,
            symbols=sym_tuples,
            initial_capital=args.capital,
            cost_model=cost_model,
            position_sizer=sizer,
            risk_manager=risk_mgr,
            warmup_bars=args.warmup,
        )
        print_multi_symbol_summary(reports)

        if args.show_trades:
            for r in reports:
                print(f"\n--- {r.symbol} ---")
                r.print_trades()


def _parse_params(params_str: str | None) -> dict:
    """Parse 'key=val,key2=val2' string into a dict with auto-typed values."""
    if not params_str:
        return {}
    result = {}
    for item in params_str.split(","):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Auto-type
        if v.lower() in ("true", "false"):
            result[k] = v.lower() == "true"
        else:
            try:
                result[k] = int(v)
            except ValueError:
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
    return result


def cmd_live(args):
    """Start live mode: WebSocket feed → strategies → Telegram alerts."""
    from src.feed import MarketFeed
    from src.strategy import StrategyRunner
    from src.alerter import Alerter

    instruments = [i.strip() for i in args.instruments.split(",")]
    print(f"Starting live mode for: {instruments}")

    runner = StrategyRunner()
    alerter = Alerter()

    async def on_tick(symbol: str, tick: dict):
        signals = runner.process_tick(symbol, tick)
        for sig in signals:
            print(f"  SIGNAL: {sig.action} {sig.symbol} ({sig.confidence:.0%}) — {sig.reason}")
            await alerter.send_alert(sig)

    feed = MarketFeed(
        instruments=instruments,
        mode=args.mode,
        on_tick=on_tick,
        store_ticks=True,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run():
        strategy_names = [s.name for s in runner.strategies]
        await alerter.send_startup_message(instruments, strategy_names)
        await feed.connect()

    # Graceful shutdown
    def shutdown(sig, frame):
        print("\nShutting down...")
        loop.create_task(feed.disconnect())
        loop.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


def main():
    parser = argparse.ArgumentParser(
        prog="projectsm",
        description="Indian Stock Market Analysis & Real-Time Alert System",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # auth
    sub.add_parser("auth", help="Run Upstox OAuth2 login flow")

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch historical candle data")
    p_fetch.add_argument("symbol", help="Trading symbol (e.g. RELIANCE, NIFTY 50)")
    p_fetch.add_argument("-e", "--exchange", default="NSE", help="Exchange: NSE, BSE (default: NSE)")
    p_fetch.add_argument("-i", "--interval", default="day", help="Interval: 1,5,15,30,day,week,month")
    p_fetch.add_argument("-u", "--unit", default="days", help="Unit: minutes,hours,days,weeks,months")
    p_fetch.add_argument("--from-date", help="Start date (YYYY-MM-DD)")
    p_fetch.add_argument("--to-date", help="End date (YYYY-MM-DD)")
    p_fetch.add_argument("--refresh-instruments", action="store_true",
                         help="Re-download instrument master before fetching")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Run analysis + backtest on stored data")
    p_analyze.add_argument("symbol", help="Trading symbol")
    p_analyze.add_argument("-e", "--exchange", default="NSE")
    p_analyze.add_argument("-i", "--interval", default="day")

    # fundamentals
    p_fund = sub.add_parser("fundamentals", help="Fundamental analysis for Indian stocks")
    p_fund.add_argument("tickers", help="Ticker(s) comma-separated (e.g. RELIANCE,TCS,INFY)")
    p_fund.add_argument("-e", "--exchange", default="NSE", help="NSE or BSE (auto-appends .NS/.BO)")
    p_fund.add_argument("--sector-pe", type=float, default=25.0, help="Sector median P/E for relative valuation")
    p_fund.add_argument("--refresh", action="store_true", help="Force re-fetch (ignore cache)")

    # backtest
    p_bt = sub.add_parser("backtest", help="Full backtest with costs, risk management, and detailed metrics")
    p_bt.add_argument("symbols", help="Symbol(s) — comma-separated for multi-symbol (e.g. RELIANCE,TCS,INFY)")
    p_bt.add_argument("-s", "--strategy", default="ema_crossover",
                      help="Strategy name from strategies.yaml (default: ema_crossover)")
    p_bt.add_argument("--strategy-module", help="Custom strategy module path (e.g. strategies.rules.moving_avg_crossover)")
    p_bt.add_argument("--strategy-class", help="Custom strategy class name (e.g. EMACrossover)")
    p_bt.add_argument("--params", help="Strategy params as key=val,key=val (e.g. fast_period=9,slow_period=21)")
    p_bt.add_argument("-e", "--exchange", default="NSE")
    p_bt.add_argument("-i", "--interval", default="day")
    p_bt.add_argument("-c", "--capital", type=float, default=100_000, help="Starting capital in ₹ (default: 100000)")
    p_bt.add_argument("--from-date", help="Start date filter (YYYY-MM-DD)")
    p_bt.add_argument("--to-date", help="End date filter (YYYY-MM-DD)")
    p_bt.add_argument("--warmup", type=int, default=50, help="Warmup bars to skip (default: 50)")
    # Position sizing
    p_bt.add_argument("--sizing-mode", default="fixed_pct", choices=["fixed_pct", "fixed_amount", "risk_pct"],
                      help="Position sizing mode (default: fixed_pct)")
    p_bt.add_argument("--allocation-pct", type=float, default=90.0, help="Capital allocation %% per trade (default: 90)")
    p_bt.add_argument("--fixed-amount", type=float, default=50_000, help="Fixed ₹ per trade for fixed_amount mode")
    # Risk management
    p_bt.add_argument("--stop-loss", type=float, default=0, help="Stop-loss %% below entry (0=disabled)")
    p_bt.add_argument("--trailing-stop", type=float, default=0, help="Trailing stop %% from peak (0=disabled)")
    p_bt.add_argument("--take-profit", type=float, default=0, help="Take-profit %% above entry (0=disabled)")
    p_bt.add_argument("--max-drawdown", type=float, default=0, help="Kill switch: stop if drawdown exceeds %% (0=disabled)")
    p_bt.add_argument("--max-holding", type=int, default=0, help="Max bars to hold a position (0=unlimited)")
    p_bt.add_argument("--cooldown", type=int, default=0, help="Bars to wait after exit before re-entry")
    # Costs
    p_bt.add_argument("--slippage", type=float, default=0.05, help="Slippage %% (default: 0.05)")
    p_bt.add_argument("--intraday", action="store_true", help="Use intraday cost structure (lower STT)")
    # Output
    p_bt.add_argument("--show-trades", action="store_true", help="Print full trade log")

    # live
    p_live = sub.add_parser("live", help="Start live WebSocket feed with strategy alerts")
    p_live.add_argument("instruments", help="Comma-separated instrument_keys")
    p_live.add_argument("-m", "--mode", default="ltpc", choices=["ltpc", "full", "option_greeks"],
                        help="Feed mode (default: ltpc)")

    args = parser.parse_args()

    if args.command == "auth":
        cmd_auth()
    elif args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "fundamentals":
        cmd_fundamentals(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "live":
        cmd_live(args)


if __name__ == "__main__":
    main()
