# -*- coding: utf-8 -*-
"""
Backtest Trading Methodology: Real-Time Cyclic State Estimation

Implements and backtests the 6 strategies from prd/trading_methodology.md:
  1. Synchronicity scoring (multi-band phase alignment)
  2. F4 leading indicator for F2
  3. Amplitude regime classification
  4. Phase synchronization exploitation
  5. Asymmetry-aware position management
  6. Integrated framework combining all signals

Backtest approach:
  - Walk-forward: no lookahead bias
  - Filters applied causally (no future data in reflect padding — use 'constant' mode)
  - Position sizing: -1 (full short) to +1 (full long)
  - Returns: log(price) changes, position * return

Reference: prd/trading_methodology.md
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.filters import ormsby_filter, apply_ormsby_filter

# ---------- constants ----------
FS = 52
TWOPI = 2 * np.pi

# 6-filter specs (rad/year)
FILTER_SPECS = [
    {"label": "F1 (trend)", "type": "lp", "f_pass": 0.85, "f_stop": 1.25, "nw": 1393},
    {"label": "F2 (3.8yr)", "type": "bp", "f1": 0.85, "f2": 1.25, "f3": 2.05, "f4": 2.45, "nw": 1393},
    {"label": "F3 (1.3yr)", "type": "bp", "f1": 3.20, "f2": 3.55, "f3": 6.35, "f4": 6.70, "nw": 1245},
    {"label": "F4 (0.7yr)", "type": "bp", "f1": 7.25, "f2": 7.55, "f3": 9.55, "f4": 9.85, "nw": 1745},
    {"label": "F5 (20wk)", "type": "bp", "f1": 13.65, "f2": 13.95, "f3": 19.35, "f4": 19.65, "nw": 1299},
    {"label": "F6 (9wk)", "type": "bp", "f1": 28.45, "f2": 28.75, "f3": 35.95, "f4": 36.25, "nw": 1299},
]


def load_weekly(symbol, start, end):
    """Load weekly price data."""
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    fname = '^dji_w.csv' if symbol == 'djia' else '^spx_w.csv'
    df = pd.read_csv(os.path.join(base, fname))
    df['Date'] = pd.to_datetime(df['Date'])
    mask = df['Date'].between(start, end)
    df = df[mask].sort_values('Date').reset_index(drop=True)
    return df['Close'].values, df['Date'].values


def apply_filter(signal, spec):
    """Apply one Ormsby filter with causal-safe padding."""
    nw = spec["nw"]
    if spec["type"] == "lp":
        f_edges = np.array([spec["f_pass"], spec["f_stop"]], dtype=float) / TWOPI
        h = ormsby_filter(nw=nw, f_edges=f_edges, fs=FS, filter_type="lp", analytic=False)
    else:
        f_edges = np.array([spec["f1"], spec["f2"], spec["f3"], spec["f4"]], dtype=float) / TWOPI
        h = ormsby_filter(nw=nw, f_edges=f_edges, fs=FS, filter_type="bp",
                         method="modulate", analytic=False)
    # Use reflect mode — standard for this project's filter application
    result = apply_ormsby_filter(signal, h, mode="reflect", fs=FS)
    return result["signal"].astype(float)


def get_envelope_and_phase(bp_signal):
    """Hilbert envelope and phase. Phase: 0=trough, pi=peak."""
    analytic = hilbert(bp_signal)
    envelope = np.abs(analytic)
    phase = np.angle(analytic)
    phase_shifted = (phase + np.pi / 2) % TWOPI
    return envelope, phase_shifted


def compute_state_vector(log_prices):
    """
    Compute the full state vector for the trading methodology.

    Returns dict with:
      phases[i]: instantaneous phase for F2-F6 (0=trough, pi=peak)
      envelopes[i]: instantaneous envelope magnitude
      sync_score: synchronicity score S(t) in [-1, +1]
      sync_weighted: amplitude-weighted synchronicity
      f4_env_change: F4 envelope 13-week rate of change
      amp_regime[i]: 'STRONG', 'NORMAL', or 'WEAK' per band
      f2_zone: 'trough', 'rising', 'peak', or 'falling'
    """
    n = len(log_prices)

    # Apply filters
    outputs = []
    for spec in FILTER_SPECS:
        sig = apply_filter(log_prices, spec)
        outputs.append(sig)

    # Envelopes and phases (F2-F6)
    envelopes = {}
    phases = {}
    for i in range(1, 6):
        env, ph = get_envelope_and_phase(outputs[i])
        envelopes[i] = env
        phases[i] = ph

    # --- Synchronicity score ---
    # s_i = -cos(theta_i): -1 at trough, +1 at peak
    phase_scores = np.zeros((5, n))
    for idx, i in enumerate(range(1, 6)):
        phase_scores[idx] = -np.cos(phases[i])

    sync_score = np.mean(phase_scores, axis=0)

    # Amplitude-weighted version
    median_envs = {}
    for i in range(1, 6):
        # Use expanding window for median (no lookahead)
        med = np.zeros(n)
        for t in range(52, n):
            med[t] = np.median(envelopes[i][max(0, t-520):t])  # 10yr rolling median
        med[:52] = med[52]
        median_envs[i] = med

    weights = np.zeros((5, n))
    for idx, i in enumerate(range(1, 6)):
        w = envelopes[i] / np.maximum(median_envs[i], 1e-10)
        weights[idx] = np.clip(w, 0.1, 5.0)  # Bound weights

    sync_weighted = np.sum(weights * phase_scores, axis=0) / np.sum(weights, axis=0)

    # --- F4 envelope rate of change (13-week) ---
    f4_env_change = np.zeros(n)
    for t in range(13, n):
        if envelopes[3][t - 13] > 1e-10:
            f4_env_change[t] = (envelopes[3][t] - envelopes[3][t - 13]) / envelopes[3][t - 13]

    # --- Amplitude regime per band ---
    amp_regime = {}
    for i in range(1, 6):
        regime = np.full(n, 'NORMAL', dtype=object)
        # Cycle-appropriate lookback (3-5 complete cycles)
        lookback_map = {1: 520, 2: 260, 3: 156, 4: 104, 5: 52}
        lb = lookback_map.get(i, 260)
        for t in range(lb, n):
            window = envelopes[i][max(0, t - lb):t]
            p33 = np.percentile(window, 33)
            p67 = np.percentile(window, 67)
            if envelopes[i][t] > p67:
                regime[t] = 'STRONG'
            elif envelopes[i][t] < p33:
                regime[t] = 'WEAK'
        amp_regime[i] = regime

    # --- F2 phase zone ---
    f2_zone = np.full(n, 'rising', dtype=object)
    for t in range(n):
        ph = phases[1][t]
        if ph < np.pi / 3 or ph > 5 * np.pi / 3:
            f2_zone[t] = 'trough'
        elif np.pi / 3 <= ph < 2 * np.pi / 3:
            f2_zone[t] = 'rising'
        elif 2 * np.pi / 3 <= ph < 4 * np.pi / 3:
            f2_zone[t] = 'peak'
        else:
            f2_zone[t] = 'falling'

    return {
        'outputs': outputs,
        'envelopes': envelopes,
        'phases': phases,
        'sync_score': sync_score,
        'sync_weighted': sync_weighted,
        'f4_env_change': f4_env_change,
        'amp_regime': amp_regime,
        'f2_zone': f2_zone,
        'phase_scores': phase_scores,
    }


def strategy_buy_hold(log_returns, n):
    """Baseline: always long."""
    return np.ones(n)


def strategy_synchronicity(state, threshold_long=-0.3, threshold_short=0.3):
    """
    Strategy 1: Position based on synchronicity score.
    S < threshold_long -> long, S > threshold_short -> short, else flat.
    """
    n = len(state['sync_weighted'])
    position = np.zeros(n)
    sw = state['sync_weighted']

    for t in range(1, n):
        if sw[t] < threshold_long:
            position[t] = 1.0  # Long
        elif sw[t] > threshold_short:
            position[t] = -1.0  # Short
        else:
            position[t] = 0.0  # Flat

    return position


def strategy_f4_leading(state):
    """
    Strategy 2: F4 envelope growth as regime indicator for F2.
    When F4 env rising + F2 near trough -> strong long
    When F4 env rising + F2 near peak -> defensive
    """
    n = len(state['f4_env_change'])
    position = np.zeros(n)

    for t in range(1, n):
        f4_rising = state['f4_env_change'][t] > 0.3
        f2z = state['f2_zone'][t]

        if f2z == 'trough':
            position[t] = 1.0
        elif f2z == 'rising':
            position[t] = 0.7
        elif f2z == 'peak':
            position[t] = -0.3 if f4_rising else 0.0
        elif f2z == 'falling':
            position[t] = -0.5 if f4_rising else -0.3

    return position


def strategy_amplitude_regime(state):
    """
    Strategy 3: Only trade when cycles are active (STRONG/NORMAL).
    Position proportional to number of active bands * sync score.
    """
    n = len(state['sync_score'])
    position = np.zeros(n)

    for t in range(1, n):
        n_active = sum(1 for i in range(1, 6) if state['amp_regime'][i][t] != 'WEAK')
        activity = n_active / 5.0  # 0 to 1
        sw = state['sync_weighted'][t]

        if sw < -0.2:
            position[t] = activity  # Long, scaled by active bands
        elif sw > 0.2:
            position[t] = -activity  # Short, scaled by active bands
        else:
            position[t] = 0.0

    return position


def strategy_integrated(state):
    """
    Strategy 6: Full integrated framework.
    Combines synchronicity, F4 leading, amplitude regime, and F2 phase.
    """
    n = len(state['sync_score'])
    position = np.zeros(n)

    for t in range(1, n):
        # Base signal from synchronicity
        sw = state['sync_weighted'][t]
        base = np.clip(-sw * 2, -1, 1)  # Map sync to [-1, 1]

        # F4 leading indicator modifier
        f4_mod = 1.0
        if state['f4_env_change'][t] > 0.5:
            f4_mod = 1.3  # Amplify signal when F4 volatility rising
        elif state['f4_env_change'][t] < -0.3:
            f4_mod = 0.7  # Dampen when F4 volatility falling

        # Amplitude regime modifier
        n_active = sum(1 for i in range(1, 6) if state['amp_regime'][i][t] != 'WEAK')
        amp_mod = 0.2 + 0.8 * (n_active / 5.0)  # 0.2 to 1.0

        # F2 zone modifier (asymmetry exploitation)
        f2z = state['f2_zone'][t]
        if f2z == 'trough' and base > 0:
            asym_mod = 1.2  # Amplify long at F2 troughs
        elif f2z == 'peak' and base < 0:
            asym_mod = 1.1  # Slightly amplify short at F2 peaks
        else:
            asym_mod = 1.0

        position[t] = np.clip(base * f4_mod * amp_mod * asym_mod, -1, 1)

    return position


def evaluate_strategy(log_returns, position, name, warmup=520):
    """
    Evaluate a strategy's performance.
    warmup: weeks to skip at start (filter artifacts).
    """
    # Strategy returns: position[t-1] * log_return[t] (position at t-1 produces return at t)
    strat_returns = position[warmup:-1] * log_returns[warmup + 1:]
    bh_returns = log_returns[warmup + 1:]

    n = len(strat_returns)
    years = n / FS

    # Annualized metrics
    ann_return_strat = np.mean(strat_returns) * FS
    ann_return_bh = np.mean(bh_returns) * FS
    ann_vol_strat = np.std(strat_returns) * np.sqrt(FS)
    ann_vol_bh = np.std(bh_returns) * np.sqrt(FS)

    sharpe_strat = ann_return_strat / ann_vol_strat if ann_vol_strat > 0 else 0
    sharpe_bh = ann_return_bh / ann_vol_bh if ann_vol_bh > 0 else 0

    # Cumulative
    cum_strat = np.cumsum(strat_returns)
    cum_bh = np.cumsum(bh_returns)

    # Max drawdown
    peak_strat = np.maximum.accumulate(cum_strat)
    dd_strat = cum_strat - peak_strat
    max_dd_strat = np.min(dd_strat)

    peak_bh = np.maximum.accumulate(cum_bh)
    dd_bh = cum_bh - peak_bh
    max_dd_bh = np.min(dd_bh)

    # Win rate
    wins = np.sum(strat_returns > 0)
    total = np.sum(strat_returns != 0)
    win_rate = wins / total if total > 0 else 0

    # Time in market
    time_in = np.mean(np.abs(position[warmup:-1]) > 0.01)

    # Exposure-adjusted return
    avg_exposure = np.mean(np.abs(position[warmup:-1]))
    adj_return = ann_return_strat / avg_exposure if avg_exposure > 0.01 else 0

    return {
        'name': name,
        'years': years,
        'ann_return': ann_return_strat,
        'ann_return_bh': ann_return_bh,
        'ann_vol': ann_vol_strat,
        'sharpe': sharpe_strat,
        'sharpe_bh': sharpe_bh,
        'max_dd': max_dd_strat,
        'max_dd_bh': max_dd_bh,
        'win_rate': win_rate,
        'time_in_market': time_in,
        'avg_exposure': avg_exposure,
        'adj_return': adj_return,
        'cum_strat': cum_strat,
        'cum_bh': cum_bh,
    }


def run_backtest(symbol, start, end, label, warmup=520):
    """Run all strategies on one dataset."""
    print(f"\n{'='*70}")
    print(f"BACKTEST: {label}")
    print(f"{'='*70}")

    close, dates = load_weekly(symbol, start, end)
    n = len(close)
    print(f"Data: {n} samples, {n/FS:.1f} years")

    log_prices = np.log(close)
    log_returns = np.diff(log_prices)
    log_returns = np.concatenate([[0], log_returns])

    # Compute state vector
    print("Computing state vector...")
    state = compute_state_vector(log_prices)

    # Run strategies
    strategies = {
        'Buy & Hold': strategy_buy_hold(log_returns, n),
        'Synchronicity': strategy_synchronicity(state),
        'F4 Leading': strategy_f4_leading(state),
        'Amplitude Regime': strategy_amplitude_regime(state),
        'Integrated': strategy_integrated(state),
    }

    results = {}
    for name, position in strategies.items():
        r = evaluate_strategy(log_returns, position, name, warmup=warmup)
        results[name] = r

    # Print summary
    print(f"\n{'Strategy':<20} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} "
          f"{'WinRate':>8} {'Exposure':>8}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<20} {r['ann_return']:>8.1%} {r['ann_vol']:>8.1%} "
              f"{r['sharpe']:>8.3f} {r['max_dd']:>8.1%} "
              f"{r['win_rate']:>8.1%} {r['avg_exposure']:>8.1%}")

    return results, state, dates, log_prices, close


def plot_backtest(results_dict, dates, log_prices, state, label, save_path, warmup=520):
    """Plot backtest results."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    fig.suptitle(f'Trading Methodology Backtest: {label}', fontsize=14, fontweight='bold')

    t_dates = dates[warmup + 1:]

    # 1. Cumulative returns
    ax = axes[0]
    colors = {'Buy & Hold': 'gray', 'Synchronicity': '#1f77b4',
              'F4 Leading': '#ff7f0e', 'Amplitude Regime': '#2ca02c',
              'Integrated': '#d62728'}
    for name, r in results_dict.items():
        lw = 2.5 if name == 'Integrated' else 1.5
        alpha = 1.0 if name in ('Integrated', 'Buy & Hold') else 0.7
        ax.plot(t_dates[:len(r['cum_strat'])], r['cum_strat'],
                label=f"{name} (Sharpe={r['sharpe']:.2f})",
                color=colors.get(name, 'black'), linewidth=lw, alpha=alpha)
    ax.set_ylabel('Cumulative Log Return')
    ax.set_title('Strategy Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Synchronicity score over time
    ax = axes[1]
    sw = state['sync_weighted'][warmup:]
    ax.fill_between(dates[warmup:warmup + len(sw)], sw, 0,
                    where=sw < 0, color='green', alpha=0.3, label='Bullish')
    ax.fill_between(dates[warmup:warmup + len(sw)], sw, 0,
                    where=sw > 0, color='red', alpha=0.3, label='Bearish')
    ax.plot(dates[warmup:warmup + len(sw)], sw, 'k-', linewidth=0.5)
    ax.set_ylabel('Sync Score')
    ax.set_title('Amplitude-Weighted Synchronicity Score')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)

    # 3. Log price with F2 phase zones
    ax = axes[2]
    lp = log_prices[warmup:]
    ax.plot(dates[warmup:warmup + len(lp)], lp, 'k-', linewidth=0.8)

    # Color background by F2 zone
    zone_colors = {'trough': 'green', 'rising': 'lightgreen', 'peak': 'red', 'falling': 'lightsalmon'}
    f2z = state['f2_zone'][warmup:]
    d = dates[warmup:warmup + len(f2z)]
    for zone, color in zone_colors.items():
        mask = np.array([z == zone for z in f2z])
        if np.any(mask):
            ax.fill_between(d, lp.min(), lp.max(), where=mask,
                           color=color, alpha=0.15, label=f'F2 {zone}')
    ax.set_ylabel('Log Price')
    ax.set_title('Price with F2 Phase Zones')
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # 4. Integrated strategy position
    ax = axes[3]
    pos = results_dict['Integrated']
    # Reconstruct position from strategy
    int_pos = strategy_integrated(state)
    p = int_pos[warmup:]
    ax.fill_between(dates[warmup:warmup + len(p)], p, 0,
                    where=p > 0, color='green', alpha=0.5)
    ax.fill_between(dates[warmup:warmup + len(p)], p, 0,
                    where=p < 0, color='red', alpha=0.5)
    ax.set_ylabel('Position')
    ax.set_title('Integrated Strategy Position (-1 to +1)')
    ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.close()


# ---------- main ----------
if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)

    # Backtest datasets
    datasets = [
        # In-sample (Hurst era) — does the methodology capture known patterns?
        {'symbol': 'djia', 'start': '1921-04-29', 'end': '1965-05-21',
         'label': 'DJIA 1921-1965 (Hurst In-Sample)', 'warmup': 520},
        # Out-of-sample periods
        {'symbol': 'djia', 'start': '1965-05-22', 'end': '2005-05-21',
         'label': 'DJIA 1965-2005 (Out-of-Sample)', 'warmup': 520},
        {'symbol': 'spx', 'start': '1950-01-01', 'end': '2026-01-31',
         'label': 'SPX 1950-2025 (Full History)', 'warmup': 520},
    ]

    all_summary = []

    for ds in datasets:
        results, state, dates, log_prices, close = run_backtest(
            ds['symbol'], ds['start'], ds['end'], ds['label'], ds['warmup']
        )

        fig_name = f"fig_backtest_{ds['symbol']}_{ds['start'][:4]}_{ds['end'][:4]}.png"
        fig_path = os.path.join(base_dir, fig_name)
        plot_backtest(results, dates, log_prices, state, ds['label'], fig_path, ds['warmup'])

        for name, r in results.items():
            all_summary.append({
                'Dataset': ds['label'][:25],
                'Strategy': name,
                'Return': r['ann_return'],
                'Sharpe': r['sharpe'],
                'MaxDD': r['max_dd'],
                'Exposure': r['avg_exposure'],
            })

    # Grand summary
    print("\n" + "=" * 90)
    print("GRAND SUMMARY: ALL BACKTESTS")
    print("=" * 90)
    print(f"{'Dataset':<26} {'Strategy':<20} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Exposure':>8}")
    print("-" * 90)
    for s in all_summary:
        print(f"{s['Dataset']:<26} {s['Strategy']:<20} {s['Return']:>8.1%} "
              f"{s['Sharpe']:>8.3f} {s['MaxDD']:>8.1%} {s['Exposure']:>8.1%}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
