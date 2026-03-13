# -*- coding: utf-8 -*-
"""
Reconstruction R² Fix — Feed 79 CMW-Confirmed Lines into Reconstruction

The current pipeline uses only ~17 Fourier-derived lines, giving R²=0.12.
Narrowband CMW has confirmed 79 harmonics (N=2-80) on daily DJIA data.
This script tests whether using more lines (and adding a trend term)
significantly improves reconstruction quality.

Key insight: validate_reconstruction() works on mean-centered log(price)
with only cos/sin terms. It cannot capture secular growth (~75% of
log-price variance). Adding constant + linear time terms should help.

Test matrix:
  A) 17 Fourier lines (baseline)
  B) ~33 weekly CMW lines
  C) 79 daily CMW lines
  D) Full N=1..80 (all possible harmonics)
  E) Each of A-D with linear trend term added

Reference: prd/nominal_model_pipeline.md, Stage 8B
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
from src.pipeline.validation import validate_reconstruction

OUT_DIR = os.path.dirname(__file__)


def reconstruct_with_trend(nominal_lines, close_prices, fs, use_log=True):
    """
    Reconstruction with constant + linear trend + harmonics.

    The standard validate_reconstruction() centers y and uses only cos/sin.
    This version adds [1, t] columns to capture secular growth, which
    accounts for ~75% of log-price variance.
    """
    if not nominal_lines or len(close_prices) < 10:
        return {'r_squared': 0.0, 'pass': False}

    y = np.log(close_prices) if use_log else close_prices.copy()
    n = len(y)
    t = np.arange(n) / fs  # time in years

    freqs = [line['frequency'] for line in nominal_lines]
    n_lines = len(freqs)

    # Design matrix: [1, t, cos(w1*t), sin(w1*t), cos(w2*t), sin(w2*t), ...]
    A = np.zeros((n, 2 + 2 * n_lines))
    A[:, 0] = 1.0       # constant
    A[:, 1] = t          # linear trend

    for j, w in enumerate(freqs):
        A[:, 2 + 2*j] = np.cos(w * t)
        A[:, 2 + 2*j + 1] = np.sin(w * t)

    # Least-squares fit
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return {'r_squared': 0.0, 'pass': False}

    reconstruction = A @ coeffs
    residual = y - reconstruction

    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Extract per-line amplitudes
    line_amps = []
    for j in range(n_lines):
        a_cos = coeffs[2 + 2*j]
        a_sin = coeffs[2 + 2*j + 1]
        amp = np.sqrt(a_cos**2 + a_sin**2)
        line_amps.append(float(amp))

    return {
        'r_squared': float(r_squared),
        'reconstruction': reconstruction,
        'residual': residual,
        'trend_const': float(coeffs[0]),
        'trend_slope': float(coeffs[1]),
        'line_amplitudes': line_amps,
        'pass': r_squared > 0.70,
        'n_lines': n_lines,
    }


def make_full_lines(w0, max_N=80):
    """Create nominal lines for every harmonic N=1..max_N."""
    lines = []
    for N in range(1, max_N + 1):
        freq = N * w0
        lines.append({
            'N': N,
            'frequency': float(freq),
            'amplitude': 1.0 / freq,  # placeholder 1/w
        })
    return lines


def main():
    print("=" * 70)
    print("RECONSTRUCTION R² FIX: Testing with 79 CMW-confirmed lines")
    print("=" * 70)

    # --- Step 1: Get w0 and Fourier lines from weekly pipeline ---
    print("\nStep 1: Running weekly pipeline for baseline...")
    weekly_result = derive_nominal_model(
        symbol='djia', freq='weekly',
        start='1921-04-29', end='1965-05-21',
        verbose=False
    )
    w0 = weekly_result.w0
    fourier_lines = weekly_result.nominal_lines
    print(f"  w0 = {w0:.4f} rad/yr")
    print(f"  Fourier lines: {len(fourier_lines)}")

    # --- Step 2: Weekly narrowband CMW ---
    print("\nStep 2: Weekly narrowband CMW (N=2..34)...")
    weekly_data = load_data('djia', 'weekly', '1921-04-29', '1965-05-21')
    nb_weekly_params = design_narrowband_cmw_bank(
        w0=w0, max_N=34, fs=weekly_data['fs'],
        fwhm_factor=0.5, omega_min=0.5
    )
    log_weekly = np.log(weekly_data['close'])
    nb_weekly_result = run_cmw_comb_bank(
        log_weekly, weekly_data['fs'], nb_weekly_params, analytic=True
    )
    weekly_cmw_lines = extract_lines_from_narrowband(nb_weekly_result, w0)
    print(f"  Weekly CMW confirmed: {len(weekly_cmw_lines)}")

    # --- Step 3: Daily narrowband CMW ---
    print("\nStep 3: Daily narrowband CMW (N=2..80)...")
    daily_data = load_data('djia', 'daily', '1921-04-29', '1965-05-21')
    nb_daily_params = design_narrowband_cmw_bank(
        w0=w0, max_N=80, fs=daily_data['fs'],
        fwhm_factor=0.5, omega_min=0.5
    )
    log_daily = np.log(daily_data['close'])
    nb_daily_result = run_cmw_comb_bank(
        log_daily, daily_data['fs'], nb_daily_params, analytic=True
    )
    daily_cmw_lines = extract_lines_from_narrowband(nb_daily_result, w0)
    print(f"  Daily CMW confirmed: {len(daily_cmw_lines)}")

    # --- Step 4: Build line sets ---
    full_lines = make_full_lines(w0, max_N=80)

    line_sets = {
        'A: Fourier (baseline)': fourier_lines,
        'B: Weekly CMW': weekly_cmw_lines,
        'C: Daily CMW (79)': daily_cmw_lines,
        'D: Full N=1..80': full_lines,
    }

    # --- Step 5: Test reconstruction on weekly data ---
    print("\n" + "=" * 70)
    print("RECONSTRUCTION RESULTS — Weekly DJIA 1921-1965")
    print("=" * 70)

    close = weekly_data['close']
    fs = weekly_data['fs']

    print(f"\n{'Line Set':<25} {'N':>4} {'R² (no trend)':>14} {'R² (+ trend)':>14} {'Pass':>6}")
    print("-" * 70)

    results = {}
    for name, lines in line_sets.items():
        # Without trend
        r_no = validate_reconstruction(lines, close, fs)
        # With trend
        r_tr = reconstruct_with_trend(lines, close, fs)

        results[name] = {'no_trend': r_no, 'with_trend': r_tr}

        pass_str = 'YES' if r_tr['pass'] else 'no'
        print(f"{name:<25} {len(lines):>4} {r_no['r_squared']:>14.4f} "
              f"{r_tr['r_squared']:>14.4f} {pass_str:>6}")

    # --- Step 6: Also test on daily data ---
    print(f"\n{'Line Set':<25} {'N':>4} {'R² daily (no trend)':>20} {'R² daily (+ trend)':>20}")
    print("-" * 75)

    daily_close = daily_data['close']
    daily_fs = daily_data['fs']
    for name, lines in line_sets.items():
        r_no = validate_reconstruction(lines, daily_close, daily_fs)
        r_tr = reconstruct_with_trend(lines, daily_close, daily_fs)
        print(f"{name:<25} {len(lines):>4} {r_no['r_squared']:>20.4f} "
              f"{r_tr['r_squared']:>20.4f}")

    # --- Step 7: Plot best reconstruction ---
    print("\nGenerating figure...")
    best_name = 'C: Daily CMW (79)'
    best_result = reconstruct_with_trend(
        line_sets[best_name], close, fs
    )

    baseline_result = reconstruct_with_trend(
        line_sets['A: Fourier (baseline)'], close, fs
    )

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # Panel 1: Original vs reconstructions
    ax = axes[0]
    y = np.log(close)
    t_yr = np.arange(len(y)) / fs
    dates_yr = t_yr + 1921.33

    ax.plot(dates_yr, y, 'k-', linewidth=0.8, label='Original log(DJIA)', alpha=0.8)
    ax.plot(dates_yr, baseline_result['reconstruction'], 'r-', linewidth=0.8,
            label=f'Fourier ({len(fourier_lines)} lines) R²={baseline_result["r_squared"]:.3f}',
            alpha=0.7)
    ax.plot(dates_yr, best_result['reconstruction'], 'b-', linewidth=0.8,
            label=f'Daily CMW ({len(daily_cmw_lines)} lines) R²={best_result["r_squared"]:.3f}',
            alpha=0.7)
    ax.set_ylabel('log(Price)')
    ax.set_title('Reconstruction Comparison: Fourier vs CMW-79 Lines + Trend')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Residuals
    ax = axes[1]
    ax.plot(dates_yr, baseline_result['residual'], 'r-', linewidth=0.5,
            label=f'Fourier residual (std={np.std(baseline_result["residual"]):.4f})', alpha=0.6)
    ax.plot(dates_yr, best_result['residual'], 'b-', linewidth=0.5,
            label=f'CMW-79 residual (std={np.std(best_result["residual"]):.4f})', alpha=0.6)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_ylabel('Residual (log)')
    ax.set_title('Reconstruction Residuals')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: R² bar chart
    ax = axes[2]
    names = list(results.keys())
    r2_no = [results[n]['no_trend']['r_squared'] for n in names]
    r2_tr = [results[n]['with_trend']['r_squared'] for n in names]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, r2_no, width, label='No trend', color='lightcoral')
    bars2 = ax.bar(x + width/2, r2_tr, width, label='With trend', color='steelblue')

    ax.axhline(0.70, color='green', linestyle='--', linewidth=1, label='Target R²=0.70')
    ax.set_xticks(x)
    ax.set_xticklabels([n.split(':')[0] + ':' + n.split(':')[1][:15] for n in names],
                        fontsize=8)
    ax.set_ylabel('R²')
    ax.set_title('Reconstruction R² Comparison')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                ha='center', fontsize=7)

    plt.suptitle('Reconstruction R² Fix — 79 CMW Lines + Trend Term',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_reconstruction_79lines.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline (Fourier, no trend):  R² = {results['A: Fourier (baseline)']['no_trend']['r_squared']:.4f}")
    print(f"Baseline (Fourier, + trend):   R² = {results['A: Fourier (baseline)']['with_trend']['r_squared']:.4f}")
    print(f"Best (CMW-79, + trend):        R² = {results['C: Daily CMW (79)']['with_trend']['r_squared']:.4f}")
    print(f"Full (N=1..80, + trend):       R² = {results['D: Full N=1..80']['with_trend']['r_squared']:.4f}")

    improvement = (results['C: Daily CMW (79)']['with_trend']['r_squared'] -
                   results['A: Fourier (baseline)']['no_trend']['r_squared'])
    print(f"\nImprovement: +{improvement:.4f} R²")
    target_met = results['C: Daily CMW (79)']['with_trend']['r_squared'] > 0.70
    print(f"Target R² > 0.70: {'MET' if target_met else 'NOT MET'}")

    print("\nKey insight: Adding linear trend captures secular growth (~75% of")
    print("log-price variance). More harmonics capture cyclic components.")
    print("Together, the reconstruction should significantly exceed the")
    print("original R²=0.12 from 17 Fourier lines without trend.")


if __name__ == '__main__':
    main()
