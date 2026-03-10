# -*- coding: utf-8 -*-
"""
Phase 6C: Sliding-Window Spectral Evolution

Tracks how the harmonic structure evolves over the full 130-year DJIA record
using 20-year sliding windows with 5-year steps.

Output: 4-panel figure showing spectral "DNA" of the DJIA over 130 years.

Reference: prd/modern_validation_plan.md
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils_validation import (
    load_data, run_spectrum, measure_line_spacing,
    compute_harmonic_fit, hurst_similarity_score, HURST_SPACING
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def sliding_window_analysis(symbol='djia', freq='weekly',
                            window_years=20, step_years=5,
                            omega_max=12.5):
    """
    Run spectral analysis on sliding windows across the full data record.

    Returns list of dicts, one per window, with all metrics.
    """
    # Load full data range
    data = load_data(symbol=symbol, freq=freq)
    dates = data['dates']
    close = data['close']

    start_date = dates.iloc[0]
    end_date = dates.iloc[-1]
    total_years = (end_date - start_date).days / 365.25

    print(f"Full record: {start_date.strftime('%Y-%m-%d')} to "
          f"{end_date.strftime('%Y-%m-%d')} ({total_years:.1f} yr, "
          f"{len(close)} samples)")

    # Generate window start years
    first_year = start_date.year
    last_possible_start = end_date.year - window_years
    start_years = list(range(first_year, last_possible_start + 1, step_years))

    results = []
    for sy in start_years:
        wy_start = f"{sy}-01-01"
        wy_end = f"{sy + window_years}-12-31"

        try:
            wd = load_data(symbol=symbol, freq=freq, start=wy_start, end=wy_end)
        except Exception as e:
            print(f"  Window {sy}-{sy+window_years}: SKIP ({e})")
            continue

        if wd['n_samples'] < 500:
            print(f"  Window {sy}-{sy+window_years}: SKIP (only {wd['n_samples']} samples)")
            continue

        # Spectrum
        spec = run_spectrum(wd['close'], wd['fs'], omega_max=omega_max,
                            peak_min_distance=2, peak_prominence_frac=0.005)

        # Metrics
        spacing = measure_line_spacing(spec['peak_freqs'], omega_max=omega_max)
        harmonic = compute_harmonic_fit(spec['peak_freqs'], spacing=HURST_SPACING, max_N=34)
        score = hurst_similarity_score(spec, spacing, harmonic)

        mid_year = sy + window_years / 2

        results.append({
            'start_year': sy,
            'end_year': sy + window_years,
            'mid_year': mid_year,
            'n_samples': wd['n_samples'],
            'n_peaks': spec['n_peaks'],
            'harmonic_spacing': spacing.get('harmonic_spacing', np.nan),
            'mean_spacing': spacing['mean_spacing'],
            'coverage': harmonic['coverage'],
            'harmonic_r2': harmonic['r_squared'],
            'rms_error': harmonic['rms_error'],
            'envelope_r2': spec['envelope_upper_r2'],
            'envelope_k': spec['envelope_upper_k'],
            'similarity_score': score['total_score'],
            'spacing_score': score['spacing_score'],
            'envelope_score': score['envelope_score'],
            'coverage_score': score['coverage_score'],
            'fit_score': score['fit_score'],
            'omega_sub': spec['omega_sub'],
            'amp_sub': spec['amp_sub'],
        })

        h_sp = spacing.get('harmonic_spacing', np.nan)
        print(f"  {sy}-{sy+window_years}: {spec['n_peaks']} peaks, "
              f"spacing={h_sp:.4f}, cov={harmonic['coverage']:.1%}, "
              f"score={score['total_score']:.3f}")

    return results


def plot_results(results, title_prefix="DJIA Weekly"):
    """Generate 4-panel evolution figure."""
    mid_years = [r['mid_year'] for r in results]
    labels = [f"{r['start_year']}-{r['end_year']}" for r in results]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'Phase 6C: {title_prefix} — Sliding-Window Spectral Evolution\n'
                 f'({results[0]["start_year"]}-{results[-1]["end_year"]}, '
                 f'{len(results)} windows of 20yr with 5yr step)',
                 fontsize=14, fontweight='bold')

    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # =========================================================================
    # Panel 1: Waterfall spectrogram (top, full width)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :])

    # Build 2D array: each row is a spectrum, interpolated to common omega grid
    omega_common = np.linspace(0.5, 12.5, 300)
    spec_matrix = np.zeros((len(results), len(omega_common)))

    for i, r in enumerate(results):
        spec_matrix[i, :] = np.interp(omega_common, r['omega_sub'], r['amp_sub'])

    # Normalize each row to [0,1] for visibility
    for i in range(len(results)):
        row_max = spec_matrix[i, :].max()
        if row_max > 0:
            spec_matrix[i, :] /= row_max

    im = ax1.pcolormesh(omega_common, mid_years, spec_matrix,
                        cmap='hot', shading='auto')
    # Add harmonic grid
    for N in range(1, 35):
        ax1.axvline(N * HURST_SPACING, color='cyan', alpha=0.15, linewidth=0.5)

    ax1.set_xlabel('Frequency (rad/yr)')
    ax1.set_ylabel('Window center year')
    ax1.set_title('Normalized Spectral Waterfall (rows normalized to peak)')
    plt.colorbar(im, ax=ax1, label='Normalized amplitude', shrink=0.6)

    # =========================================================================
    # Panel 2: Harmonic spacing vs time
    # =========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    h_spacings = [r['harmonic_spacing'] for r in results]
    ax2.plot(mid_years, h_spacings, 'bo-', markersize=5, linewidth=1.5)
    ax2.axhline(HURST_SPACING, color='red', linestyle='--', linewidth=1.5,
                label=f'Hurst: {HURST_SPACING}')
    ax2.fill_between(mid_years,
                     [HURST_SPACING - 0.01] * len(mid_years),
                     [HURST_SPACING + 0.01] * len(mid_years),
                     color='red', alpha=0.1, label='+/- 0.01')
    ax2.set_xlabel('Window center year')
    ax2.set_ylabel('Harmonic spacing (rad/yr)')
    ax2.set_title('Fitted Harmonic Spacing vs Time')
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.34, 0.40)
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 3: Similarity score vs time
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    scores = [r['similarity_score'] for r in results]
    ax3.plot(mid_years, scores, 'go-', markersize=5, linewidth=1.5, label='Total')
    ax3.plot(mid_years, [r['spacing_score'] for r in results],
             'r.--', alpha=0.6, markersize=4, label='Spacing')
    ax3.plot(mid_years, [r['coverage_score'] for r in results],
             'b.--', alpha=0.6, markersize=4, label='Coverage')
    ax3.plot(mid_years, [r['envelope_score'] for r in results],
             'm.--', alpha=0.6, markersize=4, label='Envelope')
    ax3.set_xlabel('Window center year')
    ax3.set_ylabel('Score (0-1)')
    ax3.set_title('Hurst Similarity Score vs Time')
    ax3.legend(fontsize=8, loc='lower left')
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 4: Coverage and peak count vs time
    # =========================================================================
    ax4 = fig.add_subplot(gs[2, 0])
    coverages = [r['coverage'] * 100 for r in results]
    n_peaks = [r['n_peaks'] for r in results]
    ax4.bar(mid_years, coverages, width=3.5, alpha=0.6, color='steelblue',
            label='Harmonic coverage %')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(mid_years, n_peaks, 'ro-', markersize=4, linewidth=1,
                  label='Peak count')
    ax4.set_xlabel('Window center year')
    ax4.set_ylabel('Coverage of N=1..34 (%)', color='steelblue')
    ax4_twin.set_ylabel('Peak count', color='red')
    ax4.set_title('Harmonic Coverage & Peak Count vs Time')
    ax4.legend(fontsize=8, loc='upper left')
    ax4_twin.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 5: Envelope k and R2 vs time
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, 1])
    env_r2 = [r['envelope_r2'] for r in results]
    ax5.plot(mid_years, env_r2, 'ms-', markersize=5, linewidth=1.5,
             label='Envelope R2')
    ax5.set_xlabel('Window center year')
    ax5.set_ylabel('Envelope 1/w fit R2')
    ax5.set_title('Envelope Power-Law Quality vs Time')
    ax5.set_ylim(0.8, 1.0)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    return fig


# =========================================================================
# MAIN
# =========================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("PHASE 6C: SLIDING-WINDOW SPECTRAL EVOLUTION")
    print("  20-year windows, 5-year step, full DJIA record")
    print("=" * 70)

    results = sliding_window_analysis('djia', 'weekly',
                                      window_years=20, step_years=5,
                                      omega_max=12.5)

    fig = plot_results(results, title_prefix="DJIA Weekly")
    path = os.path.join(SCRIPT_DIR, 'phase6C_sliding_window.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path}")

    # Summary stats
    h_spacings = [r['harmonic_spacing'] for r in results
                  if not np.isnan(r['harmonic_spacing'])]
    scores = [r['similarity_score'] for r in results]

    print(f"\n{'='*70}")
    print(f"PHASE 6C SUMMARY ({len(results)} windows)")
    print(f"{'='*70}")
    print(f"  Harmonic spacing: mean={np.mean(h_spacings):.4f}, "
          f"std={np.std(h_spacings):.4f}, "
          f"range=[{np.min(h_spacings):.4f}, {np.max(h_spacings):.4f}]")
    print(f"  Hurst reference:  0.3676 rad/yr")
    print(f"  Max deviation:    {np.max(np.abs(np.array(h_spacings) - HURST_SPACING)):.4f} rad/yr "
          f"({np.max(np.abs(np.array(h_spacings) - HURST_SPACING))/HURST_SPACING*100:.1f}%)")
    print(f"  Similarity score: mean={np.mean(scores):.3f}, "
          f"range=[{np.min(scores):.3f}, {np.max(scores):.3f}]")
    print(f"\nDone.")
