# -*- coding: utf-8 -*-
"""
Phase 6A: Daily Hurst-Era Reproduction

Uses daily DJIA data (1921-1965) with empirical fs (trading days/year).
Tests whether daily resolution reveals harmonics beyond N=34 and whether
the same 0.3676 rad/yr fundamental is confirmed at higher resolution.

Option A approach: Use trading days as-is, compute fs empirically.
Pre-1953: ~284 trading days/yr (Saturday sessions)
Post-1953: ~252 trading days/yr

Reference: prd/modern_validation_plan.md, prd/code_analysis_findings.md
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


def analyze_daily_vs_weekly(symbol='djia', start='1921-04-29', end='1965-05-21',
                            omega_max_weekly=12.5, omega_max_daily=22.0,
                            max_N_daily=60):
    """
    Compare daily and weekly spectra for the same time window.
    Daily data extends to higher frequencies, potentially revealing N>34.
    """
    # Load weekly
    print("Loading weekly data...")
    w_data = load_data(symbol=symbol, freq='weekly', start=start, end=end)
    print(f"  Weekly: {w_data['n_samples']} samples, fs={w_data['fs']:.1f}/yr, "
          f"{w_data['years']:.1f} yr")

    # Load daily
    print("Loading daily data...")
    d_data = load_data(symbol=symbol, freq='daily', start=start, end=end)
    print(f"  Daily:  {d_data['n_samples']} samples, fs={d_data['fs']:.1f}/yr, "
          f"{d_data['years']:.1f} yr")

    # Weekly spectrum (baseline)
    print("\nComputing weekly spectrum...")
    w_spec = run_spectrum(w_data['close'], w_data['fs'],
                          omega_max=omega_max_weekly,
                          peak_min_distance=2, peak_prominence_frac=0.005)
    w_spacing = measure_line_spacing(w_spec['peak_freqs'], omega_max=omega_max_weekly)
    w_harmonic = compute_harmonic_fit(w_spec['peak_freqs'],
                                      spacing=HURST_SPACING, max_N=34)
    w_score = hurst_similarity_score(w_spec, w_spacing, w_harmonic)

    h_sp = w_spacing.get('harmonic_spacing', np.nan)
    print(f"  Weekly: {w_spec['n_peaks']} peaks, spacing={h_sp:.4f}, "
          f"cov={w_harmonic['coverage']:.1%}, score={w_score['total_score']:.3f}")

    # Daily spectrum — analyze up to omega_max_daily
    print("\nComputing daily spectrum...")
    d_spec = run_spectrum(d_data['close'], d_data['fs'],
                          omega_max=omega_max_daily,
                          peak_min_distance=2, peak_prominence_frac=0.005)
    d_spacing = measure_line_spacing(d_spec['peak_freqs'], omega_max=omega_max_daily)
    d_harmonic = compute_harmonic_fit(d_spec['peak_freqs'],
                                      spacing=HURST_SPACING, max_N=max_N_daily)
    d_score = hurst_similarity_score(d_spec, d_spacing, d_harmonic)

    h_sp_d = d_spacing.get('harmonic_spacing', np.nan)
    print(f"  Daily:  {d_spec['n_peaks']} peaks, spacing={h_sp_d:.4f}, "
          f"cov={d_harmonic['coverage']:.1%} of N=1..{max_N_daily}, "
          f"score={d_score['total_score']:.3f}")

    # Also check: daily spectrum restricted to weekly range for apples-to-apples
    d_spec_low = run_spectrum(d_data['close'], d_data['fs'],
                              omega_max=omega_max_weekly,
                              peak_min_distance=2, peak_prominence_frac=0.005)
    d_spacing_low = measure_line_spacing(d_spec_low['peak_freqs'],
                                         omega_max=omega_max_weekly)
    d_harmonic_low = compute_harmonic_fit(d_spec_low['peak_freqs'],
                                          spacing=HURST_SPACING, max_N=34)
    d_score_low = hurst_similarity_score(d_spec_low, d_spacing_low, d_harmonic_low)

    h_sp_dl = d_spacing_low.get('harmonic_spacing', np.nan)
    print(f"  Daily (0-12.5 only): {d_spec_low['n_peaks']} peaks, "
          f"spacing={h_sp_dl:.4f}, cov={d_harmonic_low['coverage']:.1%}, "
          f"score={d_score_low['total_score']:.3f}")

    return {
        'weekly': {'data': w_data, 'spec': w_spec, 'spacing': w_spacing,
                   'harmonic': w_harmonic, 'score': w_score},
        'daily': {'data': d_data, 'spec': d_spec, 'spacing': d_spacing,
                  'harmonic': d_harmonic, 'score': d_score},
        'daily_low': {'data': d_data, 'spec': d_spec_low,
                      'spacing': d_spacing_low,
                      'harmonic': d_harmonic_low, 'score': d_score_low},
    }


def plot_daily_vs_weekly(results):
    """4-panel comparison: daily vs weekly for Hurst era."""
    w = results['weekly']
    d = results['daily']
    dl = results['daily_low']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phase 6A: Daily vs Weekly DJIA Spectrum (1921-1965)\n'
                 f'Weekly: fs={w["data"]["fs"]:.1f}/yr, '
                 f'Daily: fs={d["data"]["fs"]:.1f}/yr',
                 fontsize=14, fontweight='bold')

    # Panel 1: Overlay spectra (low-freq range)
    ax = axes[0, 0]
    w_mask = w['spec']['omega'] <= 13
    d_mask = d['spec']['omega'] <= 13
    ax.plot(w['spec']['omega'][w_mask], w['spec']['amp'][w_mask],
            'b-', alpha=0.6, linewidth=0.8, label='Weekly')
    ax.plot(d['spec']['omega'][d_mask], d['spec']['amp'][d_mask],
            'r-', alpha=0.4, linewidth=0.5, label='Daily')
    # Harmonic grid
    for N in range(1, 35):
        ax.axvline(N * HURST_SPACING, color='green', alpha=0.1, linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlim(0, 13)
    ax.set_xlabel('Frequency (rad/yr)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Spectral Overlay (0-13 rad/yr)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 2: Daily spectrum extended range
    ax = axes[0, 1]
    d_mask2 = d['spec']['omega'] <= 22
    ax.plot(d['spec']['omega'][d_mask2], d['spec']['amp'][d_mask2],
            'r-', alpha=0.7, linewidth=0.5)
    # Peaks
    ax.plot(d['spec']['peak_freqs'], d['spec']['peak_amps'],
            'kv', markersize=4, alpha=0.7)
    # Harmonic grid extended
    for N in range(1, 61):
        ax.axvline(N * HURST_SPACING, color='green', alpha=0.08, linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlim(0, 22)
    ax.set_xlabel('Frequency (rad/yr)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Daily Spectrum Extended (0-22 rad/yr, {d["spec"]["n_peaks"]} peaks)')
    ax.grid(True, alpha=0.2)

    # Panel 3: Harmonic fit comparison
    ax = axes[1, 0]
    # Weekly
    w_N = w['harmonic']['N_values']
    w_omega = w['harmonic']['omega_values']
    ax.plot(w_N, w_omega, 'bs', markersize=6, label='Weekly', alpha=0.7)
    # Daily (low range)
    dl_N = dl['harmonic']['N_values']
    dl_omega = dl['harmonic']['omega_values']
    ax.plot(dl_N, dl_omega, 'r^', markersize=5, label='Daily (0-12.5)', alpha=0.7)
    # Reference line
    N_ref = np.arange(0, 35)
    ax.plot(N_ref, N_ref * HURST_SPACING, 'g--', linewidth=1,
            label=f'0.3676*N', alpha=0.7)
    ax.set_xlabel('Harmonic number N')
    ax.set_ylabel('Frequency (rad/yr)')
    ax.set_title('Harmonic Fit: Weekly vs Daily (N=1..34)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Daily extended harmonic fit
    ax = axes[1, 1]
    d_N = d['harmonic']['N_values']
    d_omega = d['harmonic']['omega_values']
    ax.plot(d_N, d_omega, 'r^', markersize=5, alpha=0.7, label='Daily peaks')
    N_ref_ext = np.arange(0, 61)
    ax.plot(N_ref_ext, N_ref_ext * HURST_SPACING, 'g--', linewidth=1,
            label=f'0.3676*N', alpha=0.7)
    ax.set_xlabel('Harmonic number N')
    ax.set_ylabel('Frequency (rad/yr)')

    h_sp = d['spacing'].get('harmonic_spacing', np.nan)
    max_N = max(d_N) if d_N else 0
    ax.set_title(f'Daily Extended Harmonic Fit (N up to {max_N}, '
                 f'spacing={h_sp:.4f})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("PHASE 6A: DAILY HURST-ERA REPRODUCTION")
    print("  Daily DJIA 1921-1965, Option A (trading days, empirical fs)")
    print("=" * 70)

    results = analyze_daily_vs_weekly()

    fig = plot_daily_vs_weekly(results)
    path = os.path.join(SCRIPT_DIR, 'phase6A_daily_vs_weekly.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path}")

    # Summary
    w = results['weekly']
    d = results['daily']
    dl = results['daily_low']

    print(f"\n{'='*70}")
    print("PHASE 6A RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Weekly':>12} {'Daily(low)':>12} {'Daily(ext)':>12}")
    print(f"{'-'*70}")
    print(f"{'Samples':<30} {w['data']['n_samples']:>12} "
          f"{d['data']['n_samples']:>12} {d['data']['n_samples']:>12}")
    print(f"{'fs (samples/yr)':<30} {w['data']['fs']:>12.1f} "
          f"{d['data']['fs']:>12.1f} {d['data']['fs']:>12.1f}")
    print(f"{'Peaks detected':<30} {w['spec']['n_peaks']:>12} "
          f"{dl['spec']['n_peaks']:>12} {d['spec']['n_peaks']:>12}")

    w_hsp = w['spacing'].get('harmonic_spacing', np.nan)
    dl_hsp = dl['spacing'].get('harmonic_spacing', np.nan)
    d_hsp = d['spacing'].get('harmonic_spacing', np.nan)
    print(f"{'Harmonic spacing':<30} {w_hsp:>12.4f} {dl_hsp:>12.4f} {d_hsp:>12.4f}")
    print(f"{'Coverage (of max_N)':<30} {w['harmonic']['coverage']:>11.1%} "
          f"{dl['harmonic']['coverage']:>11.1%} {d['harmonic']['coverage']:>11.1%}")
    print(f"{'Harmonic R2':<30} {w['harmonic']['r_squared']:>12.4f} "
          f"{dl['harmonic']['r_squared']:>12.4f} {d['harmonic']['r_squared']:>12.4f}")
    print(f"{'Envelope R2':<30} {w['spec']['envelope_upper_r2']:>12.3f} "
          f"{dl['spec']['envelope_upper_r2']:>12.3f} {d['spec']['envelope_upper_r2']:>12.3f}")
    print(f"{'Similarity score':<30} {w['score']['total_score']:>12.3f} "
          f"{dl['score']['total_score']:>12.3f} {d['score']['total_score']:>12.3f}")

    max_N_d = max(d['harmonic']['N_values']) if d['harmonic']['N_values'] else 0
    print(f"\nHighest harmonic detected (daily): N={max_N_d} "
          f"({max_N_d * HURST_SPACING:.2f} rad/yr, "
          f"period={2*np.pi/(max_N_d*HURST_SPACING)*12:.1f} months)" if max_N_d > 0 else "")

    print("\nDone.")
