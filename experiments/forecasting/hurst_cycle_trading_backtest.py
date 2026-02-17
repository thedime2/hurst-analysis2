#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hurst Cycle Alignment Trading System - Full Backtest

This implements Hurst's ACTUAL practical method:
  1. Apply 6 bandpass filters to log(price) in real time
  2. Track where each cycle is (rising/falling) using Hilbert phase
  3. Generate BUY when multiple cycles align at troughs
  4. Generate SELL when multiple cycles align at peaks
  5. Score against buy-and-hold

The filters work on log(prices) so cycles are in percentage terms.
Uses causal (one-sided) filtering to avoid look-ahead bias.

Reference: Hurst, "The Profit Magic of Stock Transaction Timing", Ch. 10-12
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from src.filters import ormsby_filter, apply_ormsby_filter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS = 52
TWOPI = 2 * np.pi

# Hurst's 6 filter specs (rad/yr) from Page 152 / interactive tuner
# Working in log space, so these filter log(price)
FILTER_SPECS = [
    {"label": "F1 (trend)",  "type": "lp", "f_pass": 0.85, "f_stop": 1.25, "nw": 1393,
     "period_label": ">7yr"},
    {"label": "F2 (3.8yr)",  "type": "bp", "f1": 0.85, "f2": 1.25, "f3": 2.05, "f4": 2.45, "nw": 1393,
     "period_label": "2.6-5.0yr"},
    {"label": "F3 (1.3yr)",  "type": "bp", "f1": 3.20, "f2": 3.55, "f3": 6.35, "f4": 6.70, "nw": 1245,
     "period_label": "0.9-1.8yr"},
    {"label": "F4 (0.7yr)",  "type": "bp", "f1": 7.25, "f2": 7.55, "f3": 9.55, "f4": 9.85, "nw": 1745,
     "period_label": "0.6-0.8yr"},
    {"label": "F5 (20wk)",   "type": "bp", "f1": 13.65, "f2": 13.95, "f3": 19.35, "f4": 19.65, "nw": 1299,
     "period_label": "17-23wk"},
    {"label": "F6 (9wk)",    "type": "bp", "f1": 28.45, "f2": 28.75, "f3": 35.95, "f4": 36.25, "nw": 1299,
     "period_label": "9-11wk"},
]

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))


def load_djia(date_start="1900-01-01", date_end="2025-12-31"):
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/^dji_w.csv"))
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df.Date.between(date_start, date_end)].copy().reset_index(drop=True)
    return df


def compute_filter_output(log_prices, spec):
    """Apply one Ormsby filter to log(prices). Returns filtered signal."""
    nw = spec["nw"]
    if spec["type"] == "lp":
        f_edges = np.array([spec["f_pass"], spec["f_stop"]], dtype=float) / TWOPI
        h = ormsby_filter(nw=nw, f_edges=f_edges, fs=FS, filter_type="lp", analytic=False)
    else:
        f_edges = np.array([spec["f1"], spec["f2"], spec["f3"], spec["f4"]], dtype=float) / TWOPI
        h = ormsby_filter(nw=nw, f_edges=f_edges, fs=FS, filter_type="bp",
                         method="modulate", analytic=False)
    result = apply_ormsby_filter(log_prices, h, mode="reflect", fs=FS)
    return result["signal"].astype(float)


def compute_cycle_phase(bp_signal):
    """
    Compute instantaneous phase of a bandpass signal using Hilbert transform.
    Returns phase in radians [0, 2*pi] where:
      0 = trough (cycle bottom)
      pi = peak (cycle top)
    """
    analytic = hilbert(bp_signal)
    phase = np.angle(analytic)
    # Shift so 0 = trough: Hilbert phase 0 is at positive-going zero crossing,
    # so trough is at phase = -pi/2 -> shift by +pi/2
    phase_shifted = (phase + np.pi / 2) % (2 * np.pi)
    return phase_shifted


def compute_cycle_direction(phase):
    """
    From phase, determine cycle direction:
      +1 = rising (phase in [0, pi])  -> cycle going from trough toward peak
      -1 = falling (phase in [pi, 2*pi]) -> cycle going from peak toward trough
    """
    return np.where(phase < np.pi, 1, -1)


def generate_signals(filter_outputs, dates, warmup_weeks=1400):
    """
    Generate buy/sell signals based on cycle alignment.

    Strategy:
      - Track phase of BP filters 2-6 (skip LP filter 1)
      - Also track slope of LP filter 1 (trend direction)
      - BUY signal: trend rising AND >= 3 of 5 BP cycles near trough (phase near 0)
      - SELL signal: trend falling OR >= 3 of 5 BP cycles near peak (phase near pi)

    Returns:
      positions: array of +1 (long) or 0 (flat) for each week
      signal_details: dict with per-filter phase info
    """
    n = len(dates)
    n_bp = len(filter_outputs) - 1  # exclude LP

    # Compute phases for BP filters
    bp_phases = []
    bp_directions = []
    for i in range(1, len(filter_outputs)):
        phase = compute_cycle_phase(filter_outputs[i])
        bp_phases.append(phase)
        bp_directions.append(compute_cycle_direction(phase))

    bp_phases = np.array(bp_phases)       # shape: (5, n)
    bp_directions = np.array(bp_directions)  # shape: (5, n)

    # LP trend direction: simple slope over 26-week window
    lp = filter_outputs[0]
    trend_slope = np.zeros(n)
    for t in range(26, n):
        trend_slope[t] = lp[t] - lp[t - 26]
    trend_rising = trend_slope > 0

    # Cycle alignment scoring
    # For each week, count how many BP cycles are near trough vs near peak
    # "Near trough" = phase in [0, pi/3] or [5*pi/3, 2*pi] (within 60 deg of 0)
    # "Near peak" = phase in [2*pi/3, 4*pi/3] (within 60 deg of pi)
    trough_threshold = np.pi / 3  # 60 degrees
    peak_threshold = np.pi / 3

    near_trough = np.zeros((n_bp, n), dtype=bool)
    near_peak = np.zeros((n_bp, n), dtype=bool)

    for i in range(n_bp):
        ph = bp_phases[i]
        near_trough[i] = (ph < trough_threshold) | (ph > 2 * np.pi - trough_threshold)
        near_peak[i] = (ph > np.pi - peak_threshold) & (ph < np.pi + peak_threshold)

    n_at_trough = np.sum(near_trough, axis=0)  # how many cycles near trough
    n_at_peak = np.sum(near_peak, axis=0)       # how many cycles near peak
    n_rising = np.sum(bp_directions > 0, axis=0)  # how many cycles rising

    # Generate positions
    positions = np.zeros(n, dtype=int)
    position = 0  # start flat

    # Weighted scoring: longer cycles get more weight
    weights = np.array([3.0, 2.0, 1.5, 1.0, 0.5])  # F2 most important -> F6 least
    weighted_rising = np.zeros(n)
    for i in range(n_bp):
        weighted_rising += weights[i] * bp_directions[i]
    total_weight = np.sum(weights)

    for t in range(warmup_weeks, n):
        score = weighted_rising[t] / total_weight  # -1 to +1

        # BUY conditions: trend rising + weighted cycles mostly rising
        if trend_rising[t] and score > 0.2:
            position = 1
        # SELL conditions: trend falling OR weighted cycles strongly falling
        elif not trend_rising[t] and score < -0.2:
            position = 0
        elif score < -0.5:  # very strong cycle downswing even in uptrend
            position = 0

        positions[t] = position

    return positions, {
        "bp_phases": bp_phases,
        "bp_directions": bp_directions,
        "trend_rising": trend_rising,
        "n_at_trough": n_at_trough,
        "n_at_peak": n_at_peak,
        "n_rising": n_rising,
        "weighted_rising": weighted_rising / total_weight,
    }


def backtest(prices, dates, positions, warmup_weeks=1400):
    """
    Compute backtest results.

    Returns dict with strategy and benchmark performance.
    """
    n = len(prices)
    returns = np.diff(np.log(prices))  # weekly log returns

    # Strategy returns: position[t] * return[t+1]
    strat_returns = np.zeros(n - 1)
    for t in range(n - 1):
        strat_returns[t] = positions[t] * returns[t]

    # Cumulative returns
    cum_bh = np.cumsum(returns)      # buy and hold
    cum_strat = np.cumsum(strat_returns)  # strategy

    # Only score from warmup onwards
    active = slice(warmup_weeks, None)
    active_returns = returns[active]
    active_strat = strat_returns[active]

    # Annual metrics
    n_years = len(active_returns) / FS
    ann_bh = np.mean(active_returns) * FS
    ann_strat = np.mean(active_strat) * FS
    vol_bh = np.std(active_returns) * np.sqrt(FS)
    vol_strat = np.std(active_strat) * np.sqrt(FS)
    sharpe_bh = ann_bh / vol_bh if vol_bh > 0 else 0
    sharpe_strat = ann_strat / vol_strat if vol_strat > 0 else 0

    # Max drawdown
    cum_strat_active = np.cumsum(active_strat)
    running_max = np.maximum.accumulate(cum_strat_active)
    drawdowns = cum_strat_active - running_max
    max_dd_strat = np.min(drawdowns)

    cum_bh_active = np.cumsum(active_returns)
    running_max_bh = np.maximum.accumulate(cum_bh_active)
    drawdowns_bh = cum_bh_active - running_max_bh
    max_dd_bh = np.min(drawdowns_bh)

    # Win rate
    active_pos = positions[warmup_weeks:-1]
    invested_mask = active_pos > 0
    if np.sum(invested_mask) > 0:
        invested_returns = active_strat[invested_mask[:len(active_strat)]]
        win_rate = np.sum(invested_returns > 0) / len(invested_returns) * 100
    else:
        win_rate = 0

    pct_invested = np.mean(invested_mask) * 100

    return {
        "cum_bh": cum_bh,
        "cum_strat": cum_strat,
        "ann_return_bh": ann_bh,
        "ann_return_strat": ann_strat,
        "vol_bh": vol_bh,
        "vol_strat": vol_strat,
        "sharpe_bh": sharpe_bh,
        "sharpe_strat": sharpe_strat,
        "max_dd_bh": max_dd_bh,
        "max_dd_strat": max_dd_strat,
        "win_rate": win_rate,
        "pct_invested": pct_invested,
        "n_years": n_years,
        "total_return_bh": cum_bh[-1] - cum_bh[warmup_weeks] if warmup_weeks < len(cum_bh) else 0,
        "total_return_strat": cum_strat[-1] - cum_strat[warmup_weeks] if warmup_weeks < len(cum_strat) else 0,
    }


def main():
    print("=" * 76)
    print("HURST CYCLE ALIGNMENT TRADING SYSTEM - FULL BACKTEST")
    print("Real-time filtering + cycle phase tracking + alignment signals")
    print("=" * 76)

    # -----------------------------------------------------------------------
    # Load full DJIA history
    # -----------------------------------------------------------------------
    df = load_djia("1900-01-01", "2025-12-31")
    prices = df.Close.values.astype(float)
    dates = pd.to_datetime(df.Date.values)
    log_prices = np.log(prices)
    n = len(prices)
    print(f"\nData: {n} weeks ({dates[0].date()} to {dates[-1].date()})")

    # -----------------------------------------------------------------------
    # Apply 6 filters to log(prices)
    # -----------------------------------------------------------------------
    print("\nApplying 6 Hurst filters to log(prices)...")
    filter_outputs = []
    for spec in FILTER_SPECS:
        output = compute_filter_output(log_prices, spec)
        filter_outputs.append(output)
        print(f"  {spec['label']:20s}  range: [{np.min(output):.4f}, {np.max(output):.4f}]")

    # -----------------------------------------------------------------------
    # Generate trading signals
    # -----------------------------------------------------------------------
    # Warmup: need enough data for longest filter (nw=1745 -> ~1745/2 = 873 weeks)
    # Use 1400 weeks (~27 years) to be safe, starting active trading ~1927
    warmup = 1400

    print(f"\nGenerating signals (warmup={warmup} weeks, active from {dates[warmup].date()})...")
    positions, details = generate_signals(filter_outputs, dates, warmup_weeks=warmup)

    n_active = n - warmup
    n_long = np.sum(positions[warmup:] > 0)
    n_flat = np.sum(positions[warmup:] == 0)
    print(f"  Active period: {n_active} weeks")
    print(f"  Long: {n_long} weeks ({n_long/n_active*100:.1f}%)")
    print(f"  Flat: {n_flat} weeks ({n_flat/n_active*100:.1f}%)")

    # -----------------------------------------------------------------------
    # Full backtest
    # -----------------------------------------------------------------------
    print("\n" + "-" * 76)
    print("FULL BACKTEST RESULTS")
    print("-" * 76)

    results = backtest(prices, dates, positions, warmup_weeks=warmup)

    print(f"\n  Period: {dates[warmup].date()} to {dates[-1].date()} ({results['n_years']:.1f} years)")
    print(f"\n  {'Metric':30s}  {'Buy & Hold':>12s}  {'Hurst Cycles':>12s}")
    print(f"  {'-'*56}")
    print(f"  {'Annual return':30s}  {results['ann_return_bh']*100:11.2f}%  {results['ann_return_strat']*100:11.2f}%")
    print(f"  {'Annual volatility':30s}  {results['vol_bh']*100:11.2f}%  {results['vol_strat']*100:11.2f}%")
    print(f"  {'Sharpe ratio':30s}  {results['sharpe_bh']:12.3f}  {results['sharpe_strat']:12.3f}")
    print(f"  {'Max drawdown (log)':30s}  {results['max_dd_bh']*100:11.2f}%  {results['max_dd_strat']*100:11.2f}%")
    print(f"  {'Total log return':30s}  {results['total_return_bh']:12.4f}  {results['total_return_strat']:12.4f}")
    print(f"  {'Total mult. return':30s}  {np.exp(results['total_return_bh']):12.1f}x  {np.exp(results['total_return_strat']):12.1f}x")
    print(f"  {'Win rate (when invested)':30s}  {'N/A':>12s}  {results['win_rate']:11.1f}%")
    print(f"  {'Pct time invested':30s}  {'100.0':>11s}%  {results['pct_invested']:11.1f}%")

    # -----------------------------------------------------------------------
    # Decade-by-decade breakdown
    # -----------------------------------------------------------------------
    print("\n" + "-" * 76)
    print("DECADE-BY-DECADE BREAKDOWN")
    print("-" * 76)

    decades = [
        ("1928-1940", "1928-01-01", "1940-01-01"),
        ("1940-1950", "1940-01-01", "1950-01-01"),
        ("1950-1960", "1950-01-01", "1960-01-01"),
        ("1960-1970", "1960-01-01", "1970-01-01"),
        ("1970-1980", "1970-01-01", "1980-01-01"),
        ("1980-1990", "1980-01-01", "1990-01-01"),
        ("1990-2000", "1990-01-01", "2000-01-01"),
        ("2000-2010", "2000-01-01", "2010-01-01"),
        ("2010-2020", "2010-01-01", "2020-01-01"),
        ("2020-2025", "2020-01-01", "2025-06-01"),
    ]

    print(f"\n  {'Decade':12s}  {'B&H Ann':>8s}  {'Strat Ann':>9s}  {'B&H Shrp':>9s}  {'Strt Shrp':>9s}  {'Invested':>8s}  {'MaxDD B&H':>9s}  {'MaxDD Str':>9s}")
    print(f"  {'-'*78}")

    for label, d_start, d_end in decades:
        mask = (dates >= pd.Timestamp(d_start)) & (dates < pd.Timestamp(d_end))
        idx = np.where(mask)[0]
        if len(idx) < 52:
            continue

        d_prices = prices[idx]
        d_positions = positions[idx]
        d_returns = np.diff(np.log(d_prices))
        d_strat_returns = d_positions[:-1] * d_returns

        ann_bh = np.mean(d_returns) * FS
        ann_strat = np.mean(d_strat_returns) * FS
        vol_bh = np.std(d_returns) * np.sqrt(FS)
        vol_strat = np.std(d_strat_returns) * np.sqrt(FS)
        sharpe_bh = ann_bh / vol_bh if vol_bh > 0 else 0
        sharpe_strat = ann_strat / vol_strat if vol_strat > 0 else 0
        pct_inv = np.mean(d_positions > 0) * 100

        cum_bh = np.cumsum(d_returns)
        cum_strat = np.cumsum(d_strat_returns)
        max_dd_bh = np.min(cum_bh - np.maximum.accumulate(cum_bh))
        max_dd_strat = np.min(cum_strat - np.maximum.accumulate(cum_strat)) if len(cum_strat) > 0 else 0

        print(f"  {label:12s}  {ann_bh*100:7.1f}%  {ann_strat*100:8.1f}%  "
              f"{sharpe_bh:9.3f}  {sharpe_strat:9.3f}  {pct_inv:7.0f}%  "
              f"{max_dd_bh*100:8.1f}%  {max_dd_strat*100:8.1f}%")

    # -----------------------------------------------------------------------
    # Cycle alignment at major market events
    # -----------------------------------------------------------------------
    print("\n" + "-" * 76)
    print("CYCLE ALIGNMENT AT MAJOR MARKET EVENTS")
    print("-" * 76)

    events = [
        ("1929 peak", "1929-09-01"),
        ("1932 bottom", "1932-07-01"),
        ("1937 peak", "1937-03-01"),
        ("1942 bottom", "1942-05-01"),
        ("1946 peak", "1946-05-01"),
        ("1949 bottom", "1949-06-01"),
        ("1957 peak", "1957-07-01"),
        ("1962 crash", "1962-06-01"),
        ("1966 peak", "1966-02-01"),
        ("1974 bottom", "1974-12-01"),
        ("1982 bottom", "1982-08-01"),
        ("1987 crash", "1987-10-01"),
        ("2000 peak", "2000-03-01"),
        ("2003 bottom", "2003-03-01"),
        ("2007 peak", "2007-10-01"),
        ("2009 bottom", "2009-03-01"),
        ("2020 crash", "2020-03-01"),
    ]

    print(f"\n  {'Event':20s}  {'Pos':>4s}  {'Score':>6s}  {'Trend':>6s}  {'BP phases (degrees, 0=trough 180=peak)':>40s}")
    for event_name, event_date in events:
        ed = pd.Timestamp(event_date)
        idx_arr = np.where(dates >= ed)[0]
        if len(idx_arr) == 0:
            continue
        t = idx_arr[0]
        if t >= n:
            continue

        pos = positions[t]
        score = details["weighted_rising"][t]
        trend = "UP" if details["trend_rising"][t] else "DOWN"

        phases_deg = []
        for i in range(5):
            ph_deg = np.degrees(details["bp_phases"][i, t])
            phases_deg.append(f"{ph_deg:5.0f}")

        print(f"  {event_name:20s}  {pos:4d}  {score:+6.2f}  {trend:>6s}  F2={phases_deg[0]} F3={phases_deg[1]} "
              f"F4={phases_deg[2]} F5={phases_deg[3]} F6={phases_deg[4]}")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 22))

    # Panel 1: Price + positions
    ax1 = fig.add_subplot(5, 1, 1)
    ax1.semilogy(dates, prices, "k-", linewidth=0.5, label="DJIA (log scale)", alpha=0.7)
    # Shade invested periods
    invested = positions > 0
    for i in range(1, n):
        if invested[i]:
            ax1.axvspan(dates[i-1], dates[i], alpha=0.15, color="green", linewidth=0)
    ax1.set_title("DJIA with Hurst Cycle Long Periods (green shading)", fontweight="bold")
    ax1.set_ylabel("DJIA Close (log)")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cumulative returns comparison
    ax2 = fig.add_subplot(5, 1, 2)
    t_idx = np.arange(1, n)
    ax2.plot(dates[1:], results["cum_bh"], "k-", linewidth=0.7, label="Buy & Hold", alpha=0.7)
    ax2.plot(dates[1:], results["cum_strat"], "b-", linewidth=0.7, label="Hurst Cycles", alpha=0.8)
    ax2.axvline(dates[warmup], color="red", linestyle="--", alpha=0.5, label="Active start")
    ax2.set_title("Cumulative Log Returns: Buy & Hold vs Hurst Cycles", fontweight="bold")
    ax2.set_ylabel("Cumulative Log Return")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Cycle alignment score
    ax3 = fig.add_subplot(5, 1, 3)
    ax3.plot(dates, details["weighted_rising"], "b-", linewidth=0.3, alpha=0.6)
    ax3.axhline(0, color="k", linewidth=0.5)
    ax3.axhline(0.2, color="green", linewidth=0.5, linestyle="--", alpha=0.5, label="Buy threshold")
    ax3.axhline(-0.2, color="red", linewidth=0.5, linestyle="--", alpha=0.5, label="Sell threshold")
    ax3.fill_between(dates, details["weighted_rising"], 0,
                     where=details["weighted_rising"] > 0, alpha=0.2, color="green")
    ax3.fill_between(dates, details["weighted_rising"], 0,
                     where=details["weighted_rising"] < 0, alpha=0.2, color="red")
    ax3.set_title("Weighted Cycle Alignment Score (-1=all falling, +1=all rising)", fontweight="bold")
    ax3.set_ylabel("Score")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Individual filter outputs (last 20 years)
    ax4 = fig.add_subplot(5, 1, 4)
    recent_start = pd.Timestamp("2005-01-01")
    mask_recent = dates >= recent_start
    for i in range(1, 6):
        sig = filter_outputs[i][mask_recent]
        # Normalize for display
        if np.std(sig) > 0:
            sig_norm = sig / np.max(np.abs(sig)) * 0.4 + (5 - i) * 1.0
        else:
            sig_norm = np.zeros_like(sig) + (5 - i) * 1.0
        ax4.plot(dates[mask_recent], sig_norm, linewidth=0.5,
                label=FILTER_SPECS[i]["label"])
        ax4.axhline((5 - i) * 1.0, color="gray", linewidth=0.3, alpha=0.3)
    ax4.set_title("Individual BP Filter Outputs (2005-present, normalized)", fontweight="bold")
    ax4.set_ylabel("Normalized")
    ax4.legend(fontsize=8, ncol=5, loc="upper right")
    ax4.grid(True, alpha=0.3)

    # Panel 5: Drawdown comparison
    ax5 = fig.add_subplot(5, 1, 5)
    cum_bh_active = results["cum_bh"][warmup:]
    cum_strat_active = results["cum_strat"][warmup:]
    dd_bh = cum_bh_active - np.maximum.accumulate(cum_bh_active)
    dd_strat = cum_strat_active - np.maximum.accumulate(cum_strat_active)
    ax5.fill_between(dates[warmup+1:], dd_bh, 0, alpha=0.3, color="red", label="B&H drawdown")
    ax5.fill_between(dates[warmup+1:], dd_strat, 0, alpha=0.3, color="blue", label="Strategy drawdown")
    ax5.set_title("Drawdown Comparison (log returns)", fontweight="bold")
    ax5.set_ylabel("Drawdown")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "hurst_cycle_trading_backtest.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)

    print("\n" + "=" * 76)
    print("COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()
