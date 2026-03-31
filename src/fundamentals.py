"""Fundamental analysis data fetcher for Indian stocks.

Sources:
  - yfinance: P/E, P/B, ROE, debt/equity, market cap, dividends, earnings
  - Stored alongside price data in Parquet for reuse

Fundamental data is fetched once and cached. It changes quarterly (result season),
so a refresh every ~90 days is sufficient.

Usage in strategies:
  from src.fundamentals import get_fundamentals, FundamentalScore
  data = get_fundamentals("RELIANCE.NS")
  score = FundamentalScore(data)
  print(score.overall, score.value_score, score.quality_score)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

_cfg_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f).get("storage", {})

FUNDAMENTALS_DIR = Path(_cfg.get("base_dir", "data")) / "fundamentals"


# ── Fetch Fundamentals ────────────────────────────────────────────────────────

def get_fundamentals(ticker: str, refresh: bool = False) -> dict:
    """Fetch fundamental data for a stock. Caches to Parquet.

    ticker: yfinance-style ticker (e.g. RELIANCE.NS, TCS.NS, INFY.NS)
    For NSE stocks, append .NS. For BSE, append .BO.
    """
    cache_path = FUNDAMENTALS_DIR / f"{ticker.replace('.', '_')}.json"

    if cache_path.exists() and not refresh:
        import json
        data = json.loads(cache_path.read_text())
        # Refresh if older than 90 days
        cached_date = data.get("_fetched_at", "")
        if cached_date:
            try:
                age = (datetime.now() - datetime.fromisoformat(cached_date)).days
                if age < 90:
                    return data
            except ValueError:
                pass

    return _fetch_and_cache(ticker, cache_path)


def _fetch_and_cache(ticker: str, cache_path: Path) -> dict:
    """Fetch from yfinance and cache locally."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required for fundamentals. Run: pip install yfinance")

    print(f"Fetching fundamentals for {ticker}...")
    stock = yf.Ticker(ticker)
    info = stock.info or {}

    data = {
        "_ticker": ticker,
        "_fetched_at": datetime.now().isoformat(),

        # Valuation
        "market_cap": info.get("marketCap"),
        "pe_trailing": info.get("trailingPE"),
        "pe_forward": info.get("forwardPE"),
        "pb_ratio": info.get("priceToBook"),
        "ps_ratio": info.get("priceToSalesTrailing12Months"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "peg_ratio": info.get("pegRatio"),

        # Profitability
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "profit_margin": info.get("profitMargins"),
        "operating_margin": info.get("operatingMargins"),
        "gross_margin": info.get("grossMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),

        # Balance sheet
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
        "total_debt": info.get("totalDebt"),
        "total_cash": info.get("totalCash"),
        "free_cashflow": info.get("freeCashflow"),
        "operating_cashflow": info.get("operatingCashflow"),

        # Dividends
        "dividend_yield": info.get("dividendYield"),
        "dividend_rate": info.get("dividendRate"),
        "payout_ratio": info.get("payoutRatio"),
        "ex_dividend_date": info.get("exDividendDate"),

        # Shares & ownership
        "shares_outstanding": info.get("sharesOutstanding"),
        "float_shares": info.get("floatShares"),
        "held_pct_insiders": info.get("heldPercentInsiders"),
        "held_pct_institutions": info.get("heldPercentInstitutions"),

        # Analyst
        "target_mean_price": info.get("targetMeanPrice"),
        "target_high_price": info.get("targetHighPrice"),
        "target_low_price": info.get("targetLowPrice"),
        "recommendation": info.get("recommendationKey"),
        "num_analyst_opinions": info.get("numberOfAnalystOpinions"),

        # Sector/Industry
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "current_price": info.get("currentPrice") or info.get("previousClose"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "beta": info.get("beta"),
    }

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    cache_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"  Cached fundamentals → {cache_path}")

    return data


def fetch_financials_df(ticker: str) -> dict[str, pd.DataFrame]:
    """Fetch quarterly/annual income statement, balance sheet, cashflow as DataFrames."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required. Run: pip install yfinance")

    stock = yf.Ticker(ticker)
    return {
        "income_stmt": stock.quarterly_income_stmt,
        "balance_sheet": stock.quarterly_balance_sheet,
        "cashflow": stock.quarterly_cashflow,
        "income_stmt_annual": stock.income_stmt,
        "balance_sheet_annual": stock.balance_sheet,
        "cashflow_annual": stock.cashflow,
    }


# ── Fundamental Scoring ───────────────────────────────────────────────────────

@dataclass
class FundamentalScore:
    """Score a stock on Value, Quality, Growth, and overall fundamental strength.

    Each sub-score: 0.0 (worst) to 1.0 (best).
    Overall: weighted average.
    """
    value_score: float
    quality_score: float
    growth_score: float
    financial_health_score: float
    overall: float
    details: dict

    @staticmethod
    def compute(data: dict, sector_median_pe: float = 25.0) -> FundamentalScore:
        """Compute fundamental scores from fetched data.

        sector_median_pe: Use sector/Nifty 50 median P/E for relative valuation.
        Default 25 is roughly the Nifty 50 long-term average.
        """
        details = {}

        # ── Value Score (0-1) ──
        # Lower P/E, P/B, EV/EBITDA = more value
        value_parts = []

        pe = data.get("pe_trailing")
        if pe and pe > 0:
            pe_score = max(0, 1 - (pe / (sector_median_pe * 2)))  # 0 at 2x median, 1 at 0
            value_parts.append(pe_score)
            details["pe_score"] = pe_score

        pb = data.get("pb_ratio")
        if pb and pb > 0:
            pb_score = max(0, min(1, 1 - (pb - 1) / 5))  # 1 at P/B=1, 0 at P/B=6
            value_parts.append(pb_score)
            details["pb_score"] = pb_score

        ev_ebitda = data.get("ev_ebitda")
        if ev_ebitda and ev_ebitda > 0:
            ev_score = max(0, min(1, 1 - (ev_ebitda - 8) / 25))  # 1 at 8x, 0 at 33x
            value_parts.append(ev_score)
            details["ev_ebitda_score"] = ev_score

        div_yield = data.get("dividend_yield")
        if div_yield and div_yield > 0:
            div_score = min(div_yield / 0.04, 1.0)  # 1.0 at 4% yield
            value_parts.append(div_score * 0.5)  # weight dividends less
            details["dividend_score"] = div_score

        value_score = sum(value_parts) / max(len(value_parts), 1)

        # ── Quality Score (0-1) ──
        # Higher ROE, margins, stable earnings
        quality_parts = []

        roe = data.get("roe")
        if roe is not None:
            roe_score = min(max(roe / 0.25, 0), 1.0)  # 1 at 25%+ ROE
            quality_parts.append(roe_score)
            details["roe_score"] = roe_score

        op_margin = data.get("operating_margin")
        if op_margin is not None:
            margin_score = min(max(op_margin / 0.25, 0), 1.0)  # 1 at 25%+
            quality_parts.append(margin_score)
            details["margin_score"] = margin_score

        roa = data.get("roa")
        if roa is not None:
            roa_score = min(max(roa / 0.10, 0), 1.0)  # 1 at 10%+
            quality_parts.append(roa_score)
            details["roa_score"] = roa_score

        quality_score = sum(quality_parts) / max(len(quality_parts), 1)

        # ── Growth Score (0-1) ──
        growth_parts = []

        rev_growth = data.get("revenue_growth")
        if rev_growth is not None:
            rg_score = min(max(rev_growth / 0.20, 0), 1.0)  # 1 at 20%+
            growth_parts.append(rg_score)
            details["revenue_growth_score"] = rg_score

        earn_growth = data.get("earnings_growth")
        if earn_growth is not None:
            eg_score = min(max(earn_growth / 0.25, 0), 1.0)  # 1 at 25%+
            growth_parts.append(eg_score)
            details["earnings_growth_score"] = eg_score

        peg = data.get("peg_ratio")
        if peg and 0 < peg < 5:
            peg_score = max(0, 1 - (peg - 0.5) / 2.5)  # 1 at PEG 0.5, 0 at PEG 3
            growth_parts.append(peg_score)
            details["peg_score"] = peg_score

        growth_score = sum(growth_parts) / max(len(growth_parts), 1)

        # ── Financial Health Score (0-1) ──
        health_parts = []

        de = data.get("debt_to_equity")
        if de is not None:
            de_val = de / 100 if de > 10 else de  # normalize if in %
            de_score = max(0, 1 - de_val / 2)  # 1 at D/E=0, 0 at D/E=2
            health_parts.append(de_score)
            details["debt_equity_score"] = de_score

        cr = data.get("current_ratio")
        if cr is not None:
            cr_score = min(cr / 2.0, 1.0)  # 1 at CR=2+
            health_parts.append(cr_score)
            details["current_ratio_score"] = cr_score

        fcf = data.get("free_cashflow")
        if fcf is not None:
            fcf_score = 1.0 if fcf > 0 else 0.0  # positive FCF = good
            health_parts.append(fcf_score)
            details["fcf_positive"] = fcf_score

        health_score = sum(health_parts) / max(len(health_parts), 1)

        # ── Overall (weighted) ──
        overall = (
            value_score * 0.25
            + quality_score * 0.30
            + growth_score * 0.25
            + health_score * 0.20
        )

        return FundamentalScore(
            value_score=value_score,
            quality_score=quality_score,
            growth_score=growth_score,
            financial_health_score=health_score,
            overall=overall,
            details=details,
        )


def print_fundamental_report(ticker: str, data: dict, score: FundamentalScore):
    """Print a formatted fundamental analysis report."""
    w = 55
    print(f"\n{'='*w}")
    print(f"  FUNDAMENTAL ANALYSIS: {ticker}")
    print(f"  Sector: {data.get('sector', 'N/A')} | Industry: {data.get('industry', 'N/A')}")
    print(f"{'='*w}")

    print(f"\n  Valuation")
    print(f"  {'─'*40}")
    _pf("  P/E (Trailing)", data.get("pe_trailing"), ".1f")
    _pf("  P/E (Forward)", data.get("pe_forward"), ".1f")
    _pf("  P/B", data.get("pb_ratio"), ".2f")
    _pf("  EV/EBITDA", data.get("ev_ebitda"), ".1f")
    _pf("  PEG Ratio", data.get("peg_ratio"), ".2f")
    _pf("  Dividend Yield", data.get("dividend_yield"), ".2%")

    print(f"\n  Profitability")
    print(f"  {'─'*40}")
    _pf("  ROE", data.get("roe"), ".1%")
    _pf("  ROA", data.get("roa"), ".1%")
    _pf("  Operating Margin", data.get("operating_margin"), ".1%")
    _pf("  Profit Margin", data.get("profit_margin"), ".1%")

    print(f"\n  Growth")
    print(f"  {'─'*40}")
    _pf("  Revenue Growth", data.get("revenue_growth"), ".1%")
    _pf("  Earnings Growth", data.get("earnings_growth"), ".1%")

    print(f"\n  Financial Health")
    print(f"  {'─'*40}")
    _pf("  Debt/Equity", data.get("debt_to_equity"), ".1f")
    _pf("  Current Ratio", data.get("current_ratio"), ".2f")
    _pfm("  Free Cashflow", data.get("free_cashflow"))
    _pfm("  Market Cap", data.get("market_cap"))

    print(f"\n{'─'*w}")
    print(f"  SCORES (0.0 = worst, 1.0 = best)")
    print(f"  {'─'*40}")
    print(f"  Value:            {score.value_score:.2f}")
    print(f"  Quality:          {score.quality_score:.2f}")
    print(f"  Growth:           {score.growth_score:.2f}")
    print(f"  Financial Health: {score.financial_health_score:.2f}")
    print(f"  ────────────────────────")
    print(f"  OVERALL:          {score.overall:.2f}")

    if score.overall >= 0.7:
        verdict = "STRONG — fundamentally attractive"
    elif score.overall >= 0.5:
        verdict = "MODERATE — mixed signals"
    elif score.overall >= 0.3:
        verdict = "WEAK — fundamental concerns"
    else:
        verdict = "AVOID — poor fundamentals"
    print(f"  Verdict:          {verdict}")
    print(f"{'='*w}\n")


def _pf(label: str, val, fmt: str):
    """Print formatted metric."""
    if val is not None:
        print(f"  {label:<25} {val:{fmt}}")
    else:
        print(f"  {label:<25} N/A")


def _pfm(label: str, val):
    """Print formatted money value in Cr."""
    if val is not None:
        cr = val / 1e7
        print(f"  {label:<25} ₹{cr:,.0f} Cr")
    else:
        print(f"  {label:<25} N/A")
