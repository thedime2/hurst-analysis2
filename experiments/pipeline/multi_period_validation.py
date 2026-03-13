# -*- coding: utf-8 -*-
"""
Multi-Period Pipeline Validation + Market Extremes Analysis

Part 1: Run the full pipeline on new time periods:
  - DJIA 1965-2025 (post-Hurst era)
  - SPX 1985-2025 (modern SPX)
  Compare w0, line counts, envelope R² to Hurst-era baseline.

Part 2: Market extremes analysis — harmonic behavior at crises:
  - 1929 crash, 1987 crash, 2000 dot-com, 2008 GFC, 2020 COVID
  Does the synchronicity score provide early warning or only confirmation?

Reference: prd/nominal_model_pipeline.md, Remaining Work items 2-3
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.pipeline.derive_nominal_model import derive_nominal_model, load_data
from src.pipeline.comb_bank import (
    design_narrowband_cmw_bank, run_cmw_comb_bank, extract_lines_from_narrowband
)
from src.pipeline.validation import validate_model
from src.filters import ormsby_filter, apply_ormsby_filter

OUT_DIR = os.path.dirname(__file__)

FS = 52
TWOPI = 2 * np.pi

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
    """Apply one Ormsby filter."""
    nw = spec["nw"]
    is_bp = spec["type"] == "bp"

    if spec["type"] == "lp":
        f_edges = np.array([spec["f_pass"], spec["f_stop"]], dtype=float) / TWOPI
        h = ormsby_filter(nw=nw, f_edges=f_edges, fs=FS, filter_type="lp", analytic=False)
    else:
        f_edges = np.array([spec["f1"], spec["f2"], spec["f3"], spec["f4"]], dtype=float) / TWOPI
        h = ormsby_filter(nw=nw, f_edges=f_edges, fs=FS, filter_type="bp",
                         method="modulate", analytic=is_bp)

    result = apply_ormsby_filter(signal, h, mode="reflect", fs=FS)

    if is_bp and result['phase'] is not None:
        result['phase_shifted'] = (result['phasew'] + np.pi / 2) % TWOPI
    else:
        result['phase_shifted'] = None
    return result


def compute_sync_score(log_prices):
    """Compute synchronicity score from 6-filter setup."""
    filter_results = []
    for spec in FILTER_SPECS:
        fr = apply_filter(log_prices, spec)
        filter_results.append(fr)

    n = len(log_prices)
    phase_scores = np.zeros((5, n))
    envelopes = {}

    for idx in range(5):
        i = idx + 1  # F2-F6 (index 1-5)
        fr = filter_results[i]
        phase = fr['phase_shifted']
        if phase is not None:
            phase_scores[idx] = -np.cos(phase)
        envelopes[i] = fr['envelope']

    sync_score = np.mean(phase_scores, axis=0)
    return sync_score, envelopes, filter_results


# =========================================================================
# PART 1: Multi-Period Pipeline Validation
# =========================================================================

def run_multi_period():
    """Run pipeline on multiple time periods."""
    print("=" * 70)
    print("PART 1: Multi-Period Pipeline Validation")
    print("=" * 70)

    datasets = [
        {'symbol': 'djia', 'freq': 'weekly', 'start': '1921-04-29', 'end': '1965-05-21',
         'label': 'DJIA 1921-1965 (Hurst baseline)'},
        {'symbol': 'djia', 'freq': 'weekly', 'start': '1965-05-22', 'end': '2025-01-31',
         'label': 'DJIA 1965-2025 (post-Hurst)'},
        {'symbol': 'spx', 'freq': 'weekly', 'start': '1985-01-01', 'end': '2025-01-31',
         'label': 'SPX 1985-2025 (modern)'},
    ]

    results_list = []

    for ds in datasets:
        print(f"\n--- {ds['label']} ---")
        try:
            result = derive_nominal_model(
                symbol=ds['symbol'], freq=ds['freq'],
                start=ds['start'], end=ds['end'],
                verbose=False
            )

            # Validation
            val = validate_model(
                result.nominal_lines, result.peak_freqs,
                result.close, result.fs, result.groups
            )

            # Narrowband CMW
            nb_params = design_narrowband_cmw_bank(
                w0=result.w0, max_N=34, fs=result.fs,
                fwhm_factor=0.5, omega_min=0.5
            )
            log_prices = np.log(result.close)
            nb_result = run_cmw_comb_bank(log_prices, result.fs, nb_params, analytic=True)
            confirmed = extract_lines_from_narrowband(nb_result, result.w0)

            r = {
                'label': ds['label'],
                'w0': result.w0,
                'n_fourier_lines': len(result.nominal_lines),
                'n_cmw_lines': len(confirmed),
                'spectral_match': val['spectral']['match_fraction'],
                'reconstruction_r2': val['reconstruction']['r_squared'],
                'envelope_r2': val['envelope'].get('r_squared', 0),
                'overall_score': val['score'],
                'result': result,
            }
            results_list.append(r)

            print(f"  w0 = {r['w0']:.4f} rad/yr (T = {2*np.pi/r['w0']:.1f} yr)")
            print(f"  Fourier lines: {r['n_fourier_lines']}, CMW confirmed: {r['n_cmw_lines']}")
            print(f"  Spectral match: {r['spectral_match']:.1%}")
            print(f"  Reconstruction R2: {r['reconstruction_r2']:.4f}")
            print(f"  Envelope R2: {r['envelope_r2']:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results_list.append({
                'label': ds['label'],
                'w0': np.nan, 'n_fourier_lines': 0, 'n_cmw_lines': 0,
                'spectral_match': 0, 'reconstruction_r2': 0, 'envelope_r2': 0,
                'overall_score': 0, 'result': None,
            })

    # Summary table
    print("\n" + "=" * 90)
    print("MULTI-PERIOD SUMMARY")
    print("=" * 90)
    hurst_w0 = 0.3676
    print(f"{'Dataset':<35} {'w0':>8} {'dw0%':>6} {'Fourier':>8} {'CMW':>5} "
          f"{'Spec%':>6} {'R2rec':>7} {'R2env':>7}")
    print("-" * 90)
    for r in results_list:
        delta = (r['w0'] - hurst_w0) / hurst_w0 * 100 if not np.isnan(r['w0']) else 0
        print(f"{r['label']:<35} {r['w0']:>8.4f} {delta:>+5.1f}% {r['n_fourier_lines']:>8} "
              f"{r['n_cmw_lines']:>5} {r['spectral_match']:>6.1%} "
              f"{r['reconstruction_r2']:>7.4f} {r['envelope_r2']:>7.4f}")

    return results_list


# =========================================================================
# PART 2: Market Extremes Analysis
# =========================================================================

EXTREME_EVENTS = [
    {'name': '1929 Crash',  'bottom_date': '1932-07-08', 'symbol': 'djia',
     'window_start': '1921-04-29', 'window_end': '1940-01-01'},
    {'name': '1987 Crash',  'bottom_date': '1987-12-04', 'symbol': 'djia',
     'window_start': '1978-01-01', 'window_end': '1995-01-01'},
    {'name': '2000 Dot-com', 'bottom_date': '2002-10-09', 'symbol': 'spx',
     'window_start': '1990-01-01', 'window_end': '2008-01-01'},
    {'name': '2008 GFC',    'bottom_date': '2009-03-09', 'symbol': 'spx',
     'window_start': '1998-01-01', 'window_end': '2015-01-01'},
    {'name': '2020 COVID',  'bottom_date': '2020-03-23', 'symbol': 'spx',
     'window_start': '2000-01-01', 'window_end': '2025-01-31'},
]


def analyze_extreme(event):
    """Analyze one market extreme event."""
    print(f"\n--- {event['name']} (bottom: {event['bottom_date']}) ---")

    try:
        close, dates = load_weekly(event['symbol'], event['window_start'], event['window_end'])
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return None

    n = len(close)
    if n < 1000:
        print(f"  WARNING: Only {n} samples, need ~1000+ for reliable filter output")

    log_prices = np.log(close)

    # Compute sync score
    sync, envelopes, filter_results = compute_sync_score(log_prices)

    # Find bottom date in data
    bottom_dt = pd.to_datetime(event['bottom_date'])
    dates_pd = pd.to_datetime(dates)
    diffs = np.abs((dates_pd - bottom_dt).total_seconds())
    bottom_idx = np.argmin(diffs)

    # Analysis window: ±2 years around bottom
    warmup = 520  # 10 years warmup for filters
    start_idx = max(warmup, bottom_idx - 104)  # 2 years before
    end_idx = min(n - 1, bottom_idx + 104)     # 2 years after

    # Metrics at bottom
    sync_at_bottom = sync[bottom_idx]

    # When did sync first drop below -0.5? (early warning)
    early_warning_weeks = 0
    for t in range(bottom_idx, max(warmup, bottom_idx - 260), -1):
        if sync[t] > -0.5:
            early_warning_weeks = bottom_idx - t
            break

    # Min sync in window (should be ~-1.0 at major bottoms)
    min_sync = np.min(sync[start_idx:end_idx])
    min_sync_idx = start_idx + np.argmin(sync[start_idx:end_idx])
    min_sync_date = str(dates_pd[min_sync_idx].date()) if min_sync_idx < len(dates_pd) else 'N/A'

    # Envelope spike at bottom
    env_spikes = {}
    for band in range(1, 6):
        if envelopes[band] is not None:
            env_at_bottom = envelopes[band][bottom_idx]
            env_median = np.median(envelopes[band][warmup:bottom_idx])
            env_spikes[band] = env_at_bottom / env_median if env_median > 0 else 0

    print(f"  Sync at bottom: {sync_at_bottom:.3f}")
    print(f"  Min sync: {min_sync:.3f} (at {min_sync_date})")
    print(f"  Weeks from -0.5 to bottom: {early_warning_weeks}")
    print(f"  Envelope spikes: " + ", ".join(f"F{i+1}={v:.1f}x" for i, v in env_spikes.items()))

    return {
        'name': event['name'],
        'bottom_date': event['bottom_date'],
        'sync_at_bottom': sync_at_bottom,
        'min_sync': min_sync,
        'min_sync_date': min_sync_date,
        'early_warning_weeks': early_warning_weeks,
        'env_spikes': env_spikes,
        'sync': sync,
        'envelopes': envelopes,
        'dates': dates,
        'log_prices': log_prices,
        'bottom_idx': bottom_idx,
        'start_idx': start_idx,
        'end_idx': end_idx,
    }


def run_market_extremes():
    """Analyze all market extreme events."""
    print("\n" + "=" * 70)
    print("PART 2: Market Extremes Analysis")
    print("=" * 70)

    results = []
    for event in EXTREME_EVENTS:
        r = analyze_extreme(event)
        if r is not None:
            results.append(r)

    # Summary table
    if results:
        print("\n" + "=" * 90)
        print("MARKET EXTREMES SUMMARY")
        print("=" * 90)
        print(f"{'Event':<15} {'Bottom':>12} {'Sync@Bot':>9} {'MinSync':>8} "
              f"{'Warning(wk)':>11} {'F2 spike':>9} {'F3 spike':>9}")
        print("-" * 80)
        for r in results:
            f2_spike = r['env_spikes'].get(1, 0)
            f3_spike = r['env_spikes'].get(2, 0)
            print(f"{r['name']:<15} {r['bottom_date']:>12} {r['sync_at_bottom']:>9.3f} "
                  f"{r['min_sync']:>8.3f} {r['early_warning_weeks']:>11} "
                  f"{f2_spike:>9.1f}x {f3_spike:>9.1f}x")

        print("\nKey question: Early warning or confirmation?")
        avg_warning = np.mean([r['early_warning_weeks'] for r in results])
        avg_min_sync = np.mean([r['min_sync'] for r in results])
        print(f"  Average early warning: {avg_warning:.0f} weeks before bottom")
        print(f"  Average min sync: {avg_min_sync:.3f}")
        hit_minus1 = sum(1 for r in results if r['min_sync'] < -0.8)
        print(f"  Events with sync < -0.8: {hit_minus1}/{len(results)}")

    return results


def plot_market_extremes(results):
    """Plot market extremes dashboard."""
    n_events = len(results)
    if n_events == 0:
        return

    fig, axes = plt.subplots(n_events, 3, figsize=(18, 4 * n_events))
    if n_events == 1:
        axes = axes.reshape(1, -1)

    for row, r in enumerate(results):
        dates = pd.to_datetime(r['dates'])
        si, ei = r['start_idx'], r['end_idx']
        bi = r['bottom_idx']

        # Column 1: Price
        ax = axes[row, 0]
        ax.plot(dates[si:ei], np.exp(r['log_prices'][si:ei]), 'k-', linewidth=0.8)
        ax.axvline(dates[bi], color='red', linewidth=1.5, linestyle='--', alpha=0.7)
        ax.set_ylabel('Price')
        ax.set_title(f"{r['name']} — Price", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Column 2: Sync score
        ax = axes[row, 1]
        sync = r['sync'][si:ei]
        ax.fill_between(dates[si:ei], sync, 0,
                        where=np.array(sync) < 0, color='green', alpha=0.3)
        ax.fill_between(dates[si:ei], sync, 0,
                        where=np.array(sync) > 0, color='red', alpha=0.3)
        ax.plot(dates[si:ei], sync, 'k-', linewidth=0.5)
        ax.axvline(dates[bi], color='red', linewidth=1.5, linestyle='--', alpha=0.7)
        ax.axhline(-0.8, color='blue', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.set_ylabel('Sync Score')
        ax.set_title(f"Sync Score (at bottom: {r['sync_at_bottom']:.2f})", fontsize=10)
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.3)

        # Column 3: Envelopes
        ax = axes[row, 2]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        labels = ['F2', 'F3', 'F4', 'F5', 'F6']
        for band in range(1, 6):
            if r['envelopes'][band] is not None:
                env = r['envelopes'][band][si:ei]
                # Normalize to [0,1] for comparison
                env_norm = env / np.max(env) if np.max(env) > 0 else env
                ax.plot(dates[si:ei], env_norm, color=colors[band-1],
                        linewidth=0.6, alpha=0.7, label=labels[band-1])
        ax.axvline(dates[bi], color='red', linewidth=1.5, linestyle='--', alpha=0.7)
        ax.set_ylabel('Normalized Envelope')
        ax.set_title(f"Band Envelopes (warning: {r['early_warning_weeks']}wk)", fontsize=10)
        ax.legend(fontsize=7, ncol=5)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Market Extremes: Harmonic Model Behavior at Crises',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_market_extremes.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")


def plot_multi_period(results_list):
    """Plot multi-period comparison."""
    valid = [r for r in results_list if r['result'] is not None]
    if not valid:
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, r in enumerate(valid):
        ax = axes[i]
        result = r['result']
        mask = (result.omega_yr > 0.3) & (result.omega_yr < 14)
        ax.semilogy(result.omega_yr[mask], result.amp[mask], 'b-', linewidth=0.3, alpha=0.5)
        ax.semilogy(result.peak_freqs, result.peak_amps, 'r.', markersize=3)

        # Mark nominal lines
        for line in result.nominal_lines:
            ax.axvline(line['frequency'], color='green', alpha=0.2, linewidth=0.5)

        ax.set_xlabel('w (rad/yr)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f"{r['label']}\nw0={r['w0']:.4f}, {r['n_fourier_lines']} lines",
                     fontsize=9)
        ax.set_xlim(0, 14)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Multi-Period Pipeline Validation', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_multi_period_validation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")


def main():
    # Part 1
    pipeline_results = run_multi_period()
    plot_multi_period(pipeline_results)

    # Part 2
    extreme_results = run_market_extremes()
    plot_market_extremes(extreme_results)

    print("\n" + "=" * 70)
    print("ALL DONE -- Multi-Period Validation + Market Extremes")
    print("=" * 70)


if __name__ == '__main__':
    main()
