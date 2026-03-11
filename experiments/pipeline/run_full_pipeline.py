# -*- coding: utf-8 -*-
"""
Full Nominal Model Pipeline Demo

Runs the complete 10-stage pipeline on DJIA 1921-1965 (Hurst baseline),
then explores narrowband CMW for individual harmonic resolution.

Outputs:
  - Pipeline summary with w0 estimation and nominal model
  - Validation report (spectral consistency, reconstruction R², envelope)
  - 6-filter design specifications
  - Narrowband CMW analysis: can we resolve individual harmonics?
  - Comparison figures

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing (1970)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.pipeline.derive_nominal_model import derive_nominal_model
from src.pipeline.comb_bank import (
    design_narrowband_cmw_bank, design_extended_cmw_bank,
    run_cmw_comb_bank, extract_lines_from_narrowband
)
from src.pipeline.validation import validate_model
from src.pipeline.filter_design import design_analysis_filters

# Output directory
OUT_DIR = os.path.dirname(__file__)


def run_baseline_pipeline():
    """Stage A: Run core pipeline on DJIA 1921-1965."""
    print("=" * 70)
    print("PHASE A: Core Pipeline — DJIA 1921-1965 (Hurst Baseline)")
    print("=" * 70)

    result = derive_nominal_model(
        symbol='djia', freq='weekly',
        start='1921-04-29', end='1965-05-21',
        prominence_frac=0.01, min_distance=2,
        verbose=True
    )

    return result


def run_validation(result):
    """Stage B: Validate the derived model."""
    print("\n" + "=" * 70)
    print("PHASE B: Model Validation")
    print("=" * 70)

    val = validate_model(
        result.nominal_lines, result.peak_freqs,
        result.close, result.fs, result.groups
    )

    print(f"\n--- Validation Results ---")
    print(f"8A Spectral consistency: {val['spectral']['match_fraction']:.1%} matched "
          f"({'PASS' if val['spectral']['pass'] else 'FAIL'})")
    print(f"8B Reconstruction R²:    {val['reconstruction']['r_squared']:.3f} "
          f"({'PASS' if val['reconstruction']['pass'] else 'FAIL'})")
    print(f"8C Cycle counting:       {val['cycle_count']['lines_checked']} lines checked "
          f"({'PASS' if val['cycle_count']['pass'] else 'FAIL'})")
    print(f"8D Envelope 1/w fit:     R²={val['envelope'].get('r_squared', 0):.3f} "
          f"({'PASS' if val['envelope']['pass'] else 'FAIL'})")
    print(f"\nOverall: {val['n_pass']}/4 tests passed, score={val['score']:.2f}")

    return val


def run_filter_design(result):
    """Stage C: Design 6-filter bank."""
    print("\n" + "=" * 70)
    print("PHASE C: 6-Filter Design")
    print("=" * 70)

    filters = design_analysis_filters(
        group_boundaries=result.group_boundaries,
        w0=result.w0, fs=result.fs
    )

    print(filters['summary'])
    return filters


def run_narrowband_cmw(result):
    """
    Stage D: Narrowband CMW Analysis

    This is the KEY exploration: can CMW with FWHM ~ w0/2 resolve
    individual harmonics? If yes, we can build a much richer model.
    """
    print("\n" + "=" * 70)
    print("PHASE D: Narrowband CMW Analysis")
    print("=" * 70)

    # Design narrowband bank targeting each harmonic
    print(f"\nDesigning narrowband CMW bank (w0={result.w0:.4f}, fwhm_factor=0.5)...")
    nb_params = design_narrowband_cmw_bank(
        w0=result.w0, max_N=34, fs=result.fs,
        fwhm_factor=0.5, omega_min=0.5
    )
    print(f"  {len(nb_params)} narrowband CMW filters designed")
    print(f"  N range: {nb_params[0]['N']}-{nb_params[-1]['N']}")
    print(f"  Freq range: {nb_params[0]['f0']:.2f}-{nb_params[-1]['f0']:.2f} rad/yr")
    print(f"  FWHM: {nb_params[0]['fwhm']:.3f} rad/yr")

    # Apply to signal
    print("\nApplying narrowband CMW bank to log(prices)...")
    log_prices = np.log(result.close)
    nb_result = run_cmw_comb_bank(
        log_prices, result.fs, nb_params, analytic=True
    )

    # Extract confirmed lines
    confirmed = extract_lines_from_narrowband(nb_result, result.w0)
    print(f"\n  Confirmed harmonics: {len(confirmed)} / {len(nb_params)}")

    # Show confirmed lines
    print(f"\n  {'N':>3} {'Freq':>7} {'Period':>8} {'Amp':>8} {'CV%':>6} {'Conf':>6}")
    print(f"  {'---':>3} {'-------':>7} {'--------':>8} {'--------':>8} {'------':>6} {'------':>6}")
    for line in confirmed:
        print(f"  {line['N']:>3d} {line['frequency']:>7.2f} "
              f"{line['period_wk']:>8.1f}wk {line['amplitude']:>8.4f} "
              f"{line['freq_cv']*100:>5.1f}% {line['confidence']:>6s}")

    # Compare with Fourier-derived model
    fourier_Ns = set(l['N'] for l in result.nominal_lines)
    cmw_Ns = set(l['N'] for l in confirmed)
    both = fourier_Ns & cmw_Ns
    fourier_only = fourier_Ns - cmw_Ns
    cmw_only = cmw_Ns - fourier_Ns

    print(f"\n  --- Comparison: Fourier vs Narrowband CMW ---")
    print(f"  Both methods:   {sorted(both)}")
    print(f"  Fourier only:   {sorted(fourier_only)}")
    print(f"  CMW only:       {sorted(cmw_only)}")

    # Also test with narrower FWHM (0.3)
    print(f"\n  --- Testing ultra-narrow FWHM (factor=0.3) ---")
    nb_narrow = design_narrowband_cmw_bank(
        w0=result.w0, max_N=34, fs=result.fs,
        fwhm_factor=0.3, omega_min=0.5
    )
    nb_narrow_result = run_cmw_comb_bank(
        log_prices, result.fs, nb_narrow, analytic=True
    )
    confirmed_narrow = extract_lines_from_narrowband(nb_narrow_result, result.w0)
    print(f"  Confirmed with fwhm_factor=0.3: {len(confirmed_narrow)}")

    # And wider (0.8)
    print(f"\n  --- Testing wider FWHM (factor=0.8) ---")
    nb_wide = design_narrowband_cmw_bank(
        w0=result.w0, max_N=34, fs=result.fs,
        fwhm_factor=0.8, omega_min=0.5
    )
    nb_wide_result = run_cmw_comb_bank(
        log_prices, result.fs, nb_wide, analytic=True
    )
    confirmed_wide = extract_lines_from_narrowband(nb_wide_result, result.w0)
    print(f"  Confirmed with fwhm_factor=0.8: {len(confirmed_wide)}")

    return {
        'nb_params': nb_params,
        'nb_result': nb_result,
        'confirmed': confirmed,
        'confirmed_narrow': confirmed_narrow,
        'confirmed_wide': confirmed_wide,
    }


def plot_results(result, validation, nb_data):
    """Generate summary figures."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Panel 1: Spectrum with nominal lines ---
    ax = axes[0, 0]
    omega = result.omega_yr
    amp = result.amp
    mask = (omega > 0.3) & (omega < 14)
    ax.semilogy(omega[mask], amp[mask], 'b-', alpha=0.5, linewidth=0.5)
    ax.semilogy(result.peak_freqs, result.peak_amps, 'r.', markersize=3,
                label='Detected peaks')
    # Mark nominal lines
    for line in result.nominal_lines:
        ax.axvline(line['frequency'], color='green', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('ω (rad/yr)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Lanczos Spectrum + {len(result.nominal_lines)} Nominal Lines')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 14)

    # --- Panel 2: w0 estimation diagnostics ---
    ax = axes[0, 1]
    if result.nominal_df is not None and len(result.nominal_df) > 0:
        N_vals = result.nominal_df['N'].values
        freqs = result.nominal_df['frequency'].values
        ax.plot(N_vals, freqs, 'bo', markersize=5, label='Nominal lines')
        # Expected line
        N_line = np.arange(0, max(N_vals) + 1)
        ax.plot(N_line, N_line * result.w0, 'r--', linewidth=1,
                label=f'ω = {result.w0:.4f} × N')
        ax.set_xlabel('Harmonic N')
        ax.set_ylabel('ω (rad/yr)')
        ax.set_title(f'Harmonic Fit: w0={result.w0:.4f} rad/yr')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No nominal lines', transform=ax.transAxes,
                ha='center', va='center')

    # --- Panel 3: Narrowband CMW envelopes ---
    ax = axes[1, 0]
    nb_result = nb_data['nb_result']
    confirmed_Ns = set(l['N'] for l in nb_data['confirmed'])
    for i, output in enumerate(nb_result['filter_outputs']):
        spec = nb_result['filter_specs'][i]
        if 'N' not in spec:
            continue
        N = spec['N']
        if output['envelope'] is not None:
            env = output['envelope']
            color = 'green' if N in confirmed_Ns else 'gray'
            alpha = 0.7 if N in confirmed_Ns else 0.2
            ax.plot(np.mean(env), spec['f0'], 'o', color=color, alpha=alpha,
                    markersize=4)
    ax.set_xlabel('Mean Envelope Amplitude')
    ax.set_ylabel('ω (rad/yr)')
    ax.set_title(f'Narrowband CMW: {len(nb_data["confirmed"])} confirmed harmonics')

    # --- Panel 4: FWHM comparison ---
    ax = axes[1, 1]
    factors = [0.3, 0.5, 0.8]
    counts = [
        len(nb_data['confirmed_narrow']),
        len(nb_data['confirmed']),
        len(nb_data['confirmed_wide']),
    ]
    bars = ax.bar(range(len(factors)), counts, tick_label=[str(f) for f in factors],
                  color=['#2196F3', '#4CAF50', '#FF9800'])
    ax.set_xlabel('FWHM Factor (× w0)')
    ax.set_ylabel('Confirmed Harmonics')
    ax.set_title('Narrowband CMW: FWHM Factor Comparison')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha='center', fontsize=10)

    plt.suptitle(f'Nominal Model Pipeline — {result.label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_pipeline_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {fig_path}")


def plot_narrowband_detail(result, nb_data):
    """Detailed narrowband CMW figure showing envelopes and frequencies."""
    nb_result = nb_data['nb_result']
    confirmed_Ns = set(l['N'] for l in nb_data['confirmed'])

    # Select a subset of filters to show (every other one in the 7-12 rad/yr range)
    display_filters = []
    for i, spec in enumerate(nb_result['filter_specs']):
        if 'N' not in spec:
            continue
        if 7.0 <= spec['f0'] <= 12.5:
            display_filters.append(i)

    if not display_filters:
        print("No filters in 7-12 rad/yr range to display")
        return

    n_show = min(len(display_filters), 15)
    fig, axes = plt.subplots(n_show, 2, figsize=(16, n_show * 1.5), sharex='col')

    t_yr = np.arange(len(result.close)) / result.fs

    for row, fi in enumerate(display_filters[:n_show]):
        output = nb_result['filter_outputs'][fi]
        spec = nb_result['filter_specs'][fi]
        N = spec['N']
        confirmed = N in confirmed_Ns

        # Envelope
        ax = axes[row, 0] if n_show > 1 else axes[0]
        if output['envelope'] is not None:
            color = 'green' if confirmed else 'gray'
            ax.plot(t_yr, output['envelope'], color=color, linewidth=0.5)
            ax.set_ylabel(f'N={N}', fontsize=7, rotation=0, labelpad=25)
        ax.tick_params(labelsize=6)
        if row == 0:
            ax.set_title('Envelope', fontsize=9)

        # Instantaneous frequency
        ax = axes[row, 1] if n_show > 1 else axes[1]
        if output['frequency'] is not None:
            freq_rad = output['frequency'] * 2 * np.pi  # cycles/yr -> rad/yr
            ax.plot(t_yr, freq_rad, 'b.', markersize=0.3, alpha=0.3)
            ax.axhline(N * result.w0, color='red', linewidth=0.5, alpha=0.7)
            ax.set_ylim(spec['f0'] - 1, spec['f0'] + 1)
        ax.tick_params(labelsize=6)
        if row == 0:
            ax.set_title('Inst. Frequency (rad/yr)', fontsize=9)

    axes[-1, 0].set_xlabel('Time (years)', fontsize=8)
    axes[-1, 1].set_xlabel('Time (years)', fontsize=8)

    plt.suptitle(f'Narrowband CMW Detail (7-12 rad/yr) — {result.label}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_narrowband_cmw_detail.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Phase A: Core pipeline
    result = run_baseline_pipeline()

    # Phase B: Validation
    validation = run_validation(result)

    # Phase C: Filter design
    filters = run_filter_design(result)

    # Phase D: Narrowband CMW exploration
    nb_data = run_narrowband_cmw(result)

    # Figures
    print("\n" + "=" * 70)
    print("Generating figures...")
    print("=" * 70)
    plot_results(result, validation, nb_data)
    plot_narrowband_detail(result, nb_data)

    # Save nominal model to CSV
    if result.nominal_df is not None:
        csv_path = os.path.join(OUT_DIR, 'nominal_model_derived.csv')
        result.nominal_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"\nNominal model saved: {csv_path}")

    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)
