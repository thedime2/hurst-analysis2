# -*- coding: utf-8 -*-
"""
Phase 6E: Daily Modern DJIA & SPX — Extended Harmonic Analysis

Uses daily data (post-1953, uniform ~252 trading days/yr) for both DJIA
and SPX. Tests whether the harmonic series extends to high N at daily
resolution and whether DJIA and SPX share the same fundamental.

Reference: prd/modern_validation_plan.md
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt

from utils_validation import (
    load_data, run_spectrum, measure_line_spacing,
    compute_harmonic_fit, hurst_similarity_score, HURST_SPACING
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


DATASETS = [
    {'symbol': 'djia', 'freq': 'daily', 'start': '1953-01-01', 'end': '2025-12-31',
     'label': 'DJIA Daily 1953-2025', 'color': 'steelblue'},
    {'symbol': 'djia', 'freq': 'daily', 'start': '1985-01-01', 'end': '2025-12-31',
     'label': 'DJIA Daily 1985-2025', 'color': 'royalblue'},
    {'symbol': 'spx', 'freq': 'daily', 'start': '1953-01-01', 'end': '2025-12-31',
     'label': 'SPX Daily 1953-2025', 'color': 'darkgreen'},
    {'symbol': 'spx', 'freq': 'daily', 'start': '1985-01-01', 'end': '2025-12-31',
     'label': 'SPX Daily 1985-2025', 'color': 'limegreen'},
]


def analyze_dataset(ds, omega_max=30.0, max_N=80):
    """Run full analysis on one dataset."""
    data = load_data(symbol=ds['symbol'], freq=ds['freq'],
                     start=ds['start'], end=ds['end'])

    print(f"\n  {ds['label']}: {data['n_samples']} samples, "
          f"fs={data['fs']:.1f}/yr, {data['years']:.1f} yr")

    spec = run_spectrum(data['close'], data['fs'], omega_max=omega_max,
                        peak_min_distance=2, peak_prominence_frac=0.005)

    spacing = measure_line_spacing(spec['peak_freqs'], omega_max=omega_max)
    harmonic = compute_harmonic_fit(spec['peak_freqs'],
                                    spacing=HURST_SPACING, max_N=max_N)
    score = hurst_similarity_score(spec, spacing, harmonic)

    h_sp = spacing.get('harmonic_spacing', np.nan)
    max_detected_N = max(harmonic['N_values']) if harmonic['N_values'] else 0

    print(f"    Peaks: {spec['n_peaks']}, spacing={h_sp:.4f}, "
          f"cov={harmonic['coverage']:.1%} of N=1..{max_N}")
    print(f"    Highest N={max_detected_N}, R2={harmonic['r_squared']:.4f}, "
          f"score={score['total_score']:.3f}")

    return {
        'ds': ds, 'data': data, 'spec': spec,
        'spacing': spacing, 'harmonic': harmonic, 'score': score,
        'max_N_detected': max_detected_N,
    }


def plot_results(results):
    """3-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 6E: Daily DJIA & SPX Extended Harmonic Analysis',
                 fontsize=14, fontweight='bold')

    # Panel 1: DJIA spectra overlay
    ax = axes[0, 0]
    for r in results:
        if r['ds']['symbol'] != 'djia':
            continue
        mask = r['spec']['omega'] <= 30
        ax.plot(r['spec']['omega'][mask], r['spec']['amp'][mask],
                color=r['ds']['color'], alpha=0.6, linewidth=0.5,
                label=r['ds']['label'])
    for N in range(1, 81):
        ax.axvline(N * HURST_SPACING, color='red', alpha=0.05, linewidth=0.3)
    ax.set_yscale('log')
    ax.set_xlim(0, 30)
    ax.set_xlabel('Frequency (rad/yr)')
    ax.set_ylabel('Amplitude')
    ax.set_title('DJIA Daily Spectra')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 2: SPX spectra overlay
    ax = axes[0, 1]
    for r in results:
        if r['ds']['symbol'] != 'spx':
            continue
        mask = r['spec']['omega'] <= 30
        ax.plot(r['spec']['omega'][mask], r['spec']['amp'][mask],
                color=r['ds']['color'], alpha=0.6, linewidth=0.5,
                label=r['ds']['label'])
    for N in range(1, 81):
        ax.axvline(N * HURST_SPACING, color='red', alpha=0.05, linewidth=0.3)
    ax.set_yscale('log')
    ax.set_xlim(0, 30)
    ax.set_xlabel('Frequency (rad/yr)')
    ax.set_ylabel('Amplitude')
    ax.set_title('SPX Daily Spectra')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 3: Harmonic fit all datasets
    ax = axes[1, 0]
    N_ref = np.arange(0, 81)
    ax.plot(N_ref, N_ref * HURST_SPACING, 'k--', linewidth=1, alpha=0.5,
            label='0.3676*N')
    for r in results:
        N_vals = r['harmonic']['N_values']
        omega_vals = r['harmonic']['omega_values']
        ax.plot(N_vals, omega_vals, 'o', markersize=4, alpha=0.6,
                color=r['ds']['color'], label=r['ds']['label'])
    ax.set_xlabel('Harmonic number N')
    ax.set_ylabel('Frequency (rad/yr)')
    ax.set_title('Harmonic Grid Fit (all datasets)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary bar chart
    ax = axes[1, 1]
    labels = [r['ds']['label'] for r in results]
    spacings = [r['spacing'].get('harmonic_spacing', np.nan) for r in results]
    scores = [r['score']['total_score'] for r in results]
    colors = [r['ds']['color'] for r in results]

    x = np.arange(len(results))
    width = 0.35
    bars1 = ax.bar(x - width/2, spacings, width, color=colors, alpha=0.7,
                   label='Harmonic spacing')
    ax.axhline(HURST_SPACING, color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('Spacing (rad/yr)', color='black')
    ax.set_ylim(0.35, 0.39)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(' Daily ', '\n') for l in labels],
                       fontsize=7, rotation=0)
    ax.set_title('Harmonic Spacing & Similarity Score')

    ax2 = ax.twinx()
    ax2.bar(x + width/2, scores, width, color=colors, alpha=0.3,
            edgecolor=colors, linewidth=1.5, label='Score')
    ax2.set_ylabel('Similarity Score', color='gray')
    ax2.set_ylim(0.5, 1.0)

    ax.legend(fontsize=8, loc='upper left')
    ax2.legend(fontsize=8, loc='upper right')

    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("PHASE 6E: DAILY MODERN DJIA & SPX EXTENDED ANALYSIS")
    print("=" * 70)

    results = []
    for ds in DATASETS:
        r = analyze_dataset(ds, omega_max=30.0, max_N=80)
        results.append(r)

    fig = plot_results(results)
    path = os.path.join(SCRIPT_DIR, 'phase6E_daily_modern.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path}")

    # Summary table
    print(f"\n{'='*90}")
    print("PHASE 6E RESULTS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Dataset':<28} {'N':>6} {'Peaks':>6} {'HarmSpc':>8} {'MaxN':>5} "
          f"{'Cov%':>5} {'R2':>7} {'Score':>6}")
    print(f"{'-'*90}")
    for r in results:
        h_sp = r['spacing'].get('harmonic_spacing', np.nan)
        print(f"{r['ds']['label']:<28} "
              f"{r['data']['n_samples']:>6} "
              f"{r['spec']['n_peaks']:>6} "
              f"{h_sp:>8.4f} "
              f"{r['max_N_detected']:>5} "
              f"{r['harmonic']['coverage']:>4.0%} "
              f"{r['harmonic']['r_squared']:>7.4f} "
              f"{r['score']['total_score']:>6.3f}")
    print(f"{'-'*90}")
    print(f"Reference: Hurst spacing = {HURST_SPACING} rad/yr")
    print("\nDone.")
