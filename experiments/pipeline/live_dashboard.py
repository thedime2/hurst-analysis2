# -*- coding: utf-8 -*-
"""
Live Dashboard: Weekly State Vector & Regime Monitoring

Phase D of trading_methodology.md:
  - Weekly update of all state variables
  - Alert system for major regime transitions
  - Position sizing recommendations based on current state

Outputs:
  1. Current state summary (text)
  2. 6-panel dashboard figure with recent history
  3. Alert log for regime transitions

Reference: prd/trading_methodology.md, Phase D
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backtest_trading_methodology import (
    FS, TWOPI, FILTER_SPECS, NOMINAL_FREQS, ASYMMETRY_RATIOS,
    load_weekly, apply_filter, compute_state_vector,
    compute_expected_moves, strategy_integrated,
)


def detect_alerts(state, n, lookback=13):
    """
    Scan for regime transitions in recent history.
    Returns list of alert dicts.
    """
    alerts = []

    # F2 zone transitions
    for t in range(max(1, n - lookback), n):
        if state['f2_zone'][t] != state['f2_zone'][t - 1]:
            alerts.append({
                'week': t,
                'type': 'F2_ZONE',
                'severity': 'HIGH',
                'message': f"F2 zone: {state['f2_zone'][t-1]} -> {state['f2_zone'][t]}"
            })

    # Sync score threshold crossings
    sw = state['sync_weighted']
    for t in range(max(1, n - lookback), n):
        if sw[t - 1] > -0.3 and sw[t] < -0.3:
            alerts.append({
                'week': t, 'type': 'SYNC_BULL', 'severity': 'MEDIUM',
                'message': f"Sync score crossed bullish threshold ({sw[t]:.2f})"
            })
        elif sw[t - 1] < 0.3 and sw[t] > 0.3:
            alerts.append({
                'week': t, 'type': 'SYNC_BEAR', 'severity': 'MEDIUM',
                'message': f"Sync score crossed bearish threshold ({sw[t]:.2f})"
            })
        # Extreme readings
        if sw[t] < -0.6:
            if t == n - 1:
                alerts.append({
                    'week': t, 'type': 'SYNC_EXTREME_BULL', 'severity': 'HIGH',
                    'message': f"STRONG trough alignment ({sw[t]:.2f}) - major bottom signal"
                })
        elif sw[t] > 0.6:
            if t == n - 1:
                alerts.append({
                    'week': t, 'type': 'SYNC_EXTREME_BEAR', 'severity': 'HIGH',
                    'message': f"STRONG peak alignment ({sw[t]:.2f}) - major top signal"
                })

    # Amplitude regime changes
    for i in range(1, 6):
        fname = ['', 'F2', 'F3', 'F4', 'F5', 'F6'][i]
        for t in range(max(1, n - lookback), n):
            prev = state['amp_regime'][i][t - 1]
            curr = state['amp_regime'][i][t]
            if prev != curr:
                sev = 'HIGH' if fname == 'F2' else 'LOW'
                alerts.append({
                    'week': t, 'type': f'{fname}_REGIME',
                    'severity': sev,
                    'message': f"{fname} amplitude regime: {prev} -> {curr}"
                })

    return alerts


def print_state_report(state, dates, close, exp_moves, alerts, n):
    """Print formatted text report of current state."""
    t = n - 1
    date_str = pd.Timestamp(dates[t]).strftime('%Y-%m-%d')
    price = close[t]

    print("=" * 70)
    print(f"  HURST CYCLE DASHBOARD — {date_str}")
    print(f"  Price: {price:.2f}")
    print("=" * 70)

    # Current state vector
    print(f"\n--- State Vector ---")
    print(f"  Sync Score (weighted): {state['sync_weighted'][t]:+.3f}")
    print(f"  Sync Score (equal):    {state['sync_score'][t]:+.3f}")
    print(f"  F2 Zone:               {state['f2_zone'][t]}")
    print(f"  F4 Env Change (13wk):  {state['f4_env_change'][t]:+.1%}")

    # Per-band status
    print(f"\n--- Band Status ---")
    print(f"  {'Band':<6} {'Phase':>8} {'Envelope':>10} {'Regime':>8} {'R_i':>6} {'Score':>7}")
    print(f"  {'-'*50}")
    filter_names = ['', 'F2', 'F3', 'F4', 'F5', 'F6']
    for i in range(1, 6):
        ph = state['phases'][i][t]
        env = state['envelopes'][i][t]
        regime = state['amp_regime'][i][t]
        ri = state['freq_ratio'][i][t]
        score = -np.cos(ph)
        # Phase descriptor
        if ph < np.pi / 3 or ph > 5 * np.pi / 3:
            ph_desc = "TROUGH"
        elif np.pi / 3 <= ph < 2 * np.pi / 3:
            ph_desc = "rising"
        elif 2 * np.pi / 3 <= ph < 4 * np.pi / 3:
            ph_desc = "PEAK"
        else:
            ph_desc = "falling"
        print(f"  {filter_names[i]:<6} {ph_desc:>8} {env:>10.4f} {regime:>8} {ri:>6.2f} {score:>+7.2f}")

    # Market regime classification (from PRD Step 2)
    sw = state['sync_weighted'][t]
    f2z = state['f2_zone'][t]
    de4 = state['f4_env_change'][t]
    de4_rising = de4 > 0.3

    if f2z == 'trough' and sw < -0.4:
        regime = "MAJOR BOTTOM"
        action = "Maximum long"
    elif f2z in ('trough', 'rising') and sw < -0.2:
        regime = "BULL TREND"
        action = "Stay long, buy dips"
    elif f2z == 'peak' and sw > 0.4:
        regime = "MAJOR TOP"
        action = "Exit or short"
    elif f2z in ('peak', 'falling') and sw > 0.2:
        regime = "BEAR TREND"
        action = "Flat or short"
    else:
        regime = "NEUTRAL"
        action = "Reduced position"

    print(f"\n--- Market Regime ---")
    print(f"  Classification: {regime}")
    print(f"  Recommended:    {action}")
    if de4_rising:
        print(f"  WARNING: F4 envelope rising ({de4:+.1%}) — volatility increasing")

    # Position sizing (from PRD Step 3)
    int_pos = strategy_integrated(state)
    current_pos = int_pos[t]
    n_active = sum(1 for i in range(1, 6) if state['amp_regime'][i][t] != 'WEAK')

    print(f"\n--- Position Sizing ---")
    print(f"  Integrated signal:  {current_pos:+.2f}")
    print(f"  Active bands:       {n_active}/5")

    # Expected moves
    if exp_moves is not None:
        em = exp_moves['expected_move'][t]
        ed = exp_moves['expected_duration'][t]
        sl = exp_moves['stop_level'][t]
        db = exp_moves['dominant_band'][t]
        db_name = filter_names[db]
        print(f"\n--- Expected Move (Step 4) ---")
        print(f"  Dominant band:      {db_name}")
        print(f"  Expected move:      {em:.4f} log ({(np.exp(em)-1)*100:.1f}%)")
        print(f"  Expected duration:  {ed:.0f} weeks")
        print(f"  Stop level (p90):   {sl:.4f} log ({(np.exp(sl)-1)*100:.1f}%)")

    # Alerts
    if alerts:
        recent = [a for a in alerts if a['week'] >= n - 4]
        if recent:
            print(f"\n--- Recent Alerts (last 4 weeks) ---")
            for a in recent:
                wk_date = pd.Timestamp(dates[a['week']]).strftime('%Y-%m-%d')
                marker = "!!!" if a['severity'] == 'HIGH' else "  >"
                print(f"  {marker} [{wk_date}] {a['message']}")
    else:
        print(f"\n  No alerts in recent history.")

    print()


def plot_dashboard(state, dates, log_prices, close, exp_moves, label,
                   save_path, show_weeks=260):
    """Plot 6-panel live dashboard showing recent history."""
    n = len(log_prices)
    sl = slice(max(0, n - show_weeks), n)
    d = dates[sl]

    fig, axes = plt.subplots(6, 1, figsize=(16, 24))
    fig.suptitle(f'Live Dashboard: {label} (last {show_weeks} weeks)',
                 fontsize=14, fontweight='bold')

    # 1. Price with F2 zones
    ax = axes[0]
    lp = log_prices[sl]
    ax.plot(d, np.exp(lp), 'k-', linewidth=1.2)
    zone_colors = {'trough': 'green', 'rising': 'lightgreen',
                   'peak': 'red', 'falling': 'lightsalmon'}
    f2z = state['f2_zone'][sl]
    for zone, color in zone_colors.items():
        mask = np.array([z == zone for z in f2z])
        if np.any(mask):
            ax.fill_between(d, np.exp(lp).min(), np.exp(lp).max(),
                           where=mask, color=color, alpha=0.15, label=f'F2 {zone}')
    ax.set_ylabel('Price')
    ax.set_title('Price with F2 Phase Zones')
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # 2. Synchronicity score
    ax = axes[1]
    sw = state['sync_weighted'][sl]
    ax.fill_between(d, sw, 0, where=sw < 0, color='green', alpha=0.4, label='Bullish')
    ax.fill_between(d, sw, 0, where=sw > 0, color='red', alpha=0.4, label='Bearish')
    ax.plot(d, sw, 'k-', linewidth=0.8)
    ax.axhline(-0.3, color='green', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(0.3, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axhline(-0.6, color='green', linewidth=1.2, linestyle='-', alpha=0.5)
    ax.axhline(0.6, color='red', linewidth=1.2, linestyle='-', alpha=0.5)
    ax.set_ylabel('Sync Score')
    ax.set_title('Amplitude-Weighted Synchronicity')
    ax.set_ylim(-1.2, 1.2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. BP filter envelopes (F2-F6)
    ax = axes[2]
    filter_names = {1: 'F2 (3.8yr)', 2: 'F3 (1.3yr)', 3: 'F4 (40wk)',
                    4: 'F5 (20wk)', 5: 'F6 (10wk)'}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in range(1, 6):
        env = state['envelopes'][i][sl]
        ax.plot(d, env, linewidth=1.0, color=colors[i - 1],
                label=filter_names[i], alpha=0.8)
    ax.set_ylabel('Envelope')
    ax.set_title('Band Envelopes (F2-F6)')
    ax.legend(fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)

    # 4. Integrated position
    ax = axes[3]
    int_pos = strategy_integrated(state)
    p = int_pos[sl]
    ax.fill_between(d, p, 0, where=p > 0, color='green', alpha=0.5)
    ax.fill_between(d, p, 0, where=p < 0, color='red', alpha=0.5)
    ax.plot(d, p, 'k-', linewidth=0.5)
    ax.set_ylabel('Position')
    ax.set_title('Integrated Strategy Position')
    ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.3)

    # 5. Expected move sizing
    ax = axes[4]
    if exp_moves is not None:
        em = exp_moves['expected_move'][sl]
        sl_stop = exp_moves['stop_level'][sl]
        ax.plot(d, em, 'b-', linewidth=1.0, label='Expected Move')
        ax.plot(d, sl_stop, 'r--', linewidth=0.8, alpha=0.7, label='Stop (p90)')
        ax.legend(fontsize=9)
    ax.set_ylabel('Log Amplitude')
    ax.set_title('Expected Move & Stop Level')
    ax.grid(True, alpha=0.3)

    # 6. Band calibration
    ax = axes[5]
    cal_names = {1: 'F2', 2: 'F3', 3: 'F4', 4: 'F5', 5: 'F6'}
    for i in range(1, 6):
        fr = state['freq_ratio'][i][sl]
        ax.plot(d, fr, linewidth=0.8, color=colors[i - 1],
                label=cal_names[i], alpha=0.7)
    ax.axhline(1.0, color='k', linewidth=1)
    ax.axhline(0.8, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.axhline(1.2, color='k', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.set_ylabel('R_i')
    ax.set_title('Band Calibration (Nominal / Instantaneous)')
    ax.set_ylim(0.3, 2.0)
    ax.legend(fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Dashboard saved: {save_path}")
    plt.close()


# ---------- main ----------
if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)

    # Run on both DJIA and SPX with most recent data
    datasets = [
        {'symbol': 'djia', 'start': '1980-01-01', 'end': '2026-12-31',
         'label': 'DJIA'},
        {'symbol': 'spx', 'start': '1980-01-01', 'end': '2026-12-31',
         'label': 'SPX'},
    ]

    for ds in datasets:
        print(f"\n{'='*70}")
        print(f"LIVE DASHBOARD: {ds['label']}")
        print(f"{'='*70}")

        close, dates = load_weekly(ds['symbol'], ds['start'], ds['end'])
        n = len(close)
        print(f"Data: {n} samples through {pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}")

        log_prices = np.log(close)

        print("Computing state vector...")
        state = compute_state_vector(log_prices)

        print("Computing expected moves...")
        exp_moves = compute_expected_moves(state, warmup=520)

        print("Scanning for alerts...")
        alerts = detect_alerts(state, n, lookback=13)

        # Print report
        print_state_report(state, dates, close, exp_moves, alerts, n)

        # Plot dashboard
        fig_path = os.path.join(base_dir, f'fig_dashboard_{ds["symbol"]}.png')
        plot_dashboard(state, dates, log_prices, close, exp_moves,
                       ds['label'], fig_path, show_weeks=260)

    print("\n" + "=" * 70)
    print("DASHBOARD COMPLETE")
    print("=" * 70)
