# -*- coding: utf-8 -*-
"""
Test Harmonics Beyond N=80 — Does Hurst's Structure Extend Further?

Daily DJIA (fs≈252) has Nyquist at ~396 rad/yr. With w0=0.3676,
this allows testing to N≈1078 theoretically. Existing analysis
confirmed 79/79 harmonics (N=2-80). This script tests N=81-200.

Questions:
  1. How many harmonics confirm beyond N=80?
  2. What's the maximum confirmed N and its period?
  3. Does the 1/ω envelope law still hold?
  4. Do high-N harmonics become less frequency-stable?

Reference: prd/hurst_unified_theory_v2.md, Open Question 1
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from src.pipeline.derive_nominal_model import derive_nominal_model, load_data, compute_spectrum
from src.pipeline.comb_bank import (
    design_narrowband_cmw_bank, run_cmw_comb_bank, extract_lines_from_narrowband
)

OUT_DIR = os.path.dirname(__file__)


def main():
    print("=" * 70)
    print("HARMONICS BEYOND N=80 — Testing Extended Range")
    print("=" * 70)

    # --- Step 1: Get w0 from weekly pipeline ---
    print("\nStep 1: Getting w0 from weekly pipeline...")
    weekly_result = derive_nominal_model(
        symbol='djia', freq='weekly',
        start='1921-04-29', end='1965-05-21',
        verbose=False
    )
    w0 = weekly_result.w0
    print(f"  w0 = {w0:.4f} rad/yr (T = {2*np.pi/w0:.1f} yr)")

    # --- Step 2: Load daily data ---
    print("\nStep 2: Loading daily DJIA 1921-1965...")
    daily = load_data('djia', 'daily', '1921-04-29', '1965-05-21')
    fs = daily['fs']
    nyquist = np.pi * fs
    max_N_possible = int(nyquist * 0.9 / w0)
    print(f"  {daily['n_samples']} samples, fs={fs:.1f}")
    print(f"  Nyquist: {nyquist:.1f} rad/yr")
    print(f"  Max possible N: {max_N_possible}")

    # --- Step 3: Compute Lanczos spectrum on daily data ---
    print("\nStep 3: Computing daily Lanczos spectrum...")
    omega_yr, amp = compute_spectrum(daily['close'], fs)

    # --- Step 4: Run narrowband CMW for N=2..80 (baseline) ---
    print("\nStep 4: Narrowband CMW N=2..80 (baseline)...")
    log_prices = np.log(daily['close'])

    nb_base_params = design_narrowband_cmw_bank(
        w0=w0, max_N=80, fs=fs, fwhm_factor=0.5, omega_min=0.5
    )
    nb_base_result = run_cmw_comb_bank(log_prices, fs, nb_base_params, analytic=True)
    base_confirmed = extract_lines_from_narrowband(nb_base_result, w0)
    print(f"  Baseline confirmed: {len(base_confirmed)}/79")

    # --- Step 5: Run narrowband CMW for N=81..200 (extended) ---
    print("\nStep 5: Narrowband CMW N=81..200 (extended)...")
    nb_ext_params = design_narrowband_cmw_bank(
        w0=w0, max_N=200, fs=fs, fwhm_factor=0.5, omega_min=80 * w0 * 0.9
    )
    print(f"  {len(nb_ext_params)} extended filters designed")
    if len(nb_ext_params) > 0:
        print(f"  Freq range: {nb_ext_params[0]['f0']:.2f} - {nb_ext_params[-1]['f0']:.2f} rad/yr")

    nb_ext_result = run_cmw_comb_bank(log_prices, fs, nb_ext_params, analytic=True)
    ext_confirmed = extract_lines_from_narrowband(nb_ext_result, w0)
    print(f"  Extended confirmed: {len(ext_confirmed)}/{len(nb_ext_params)}")

    # --- Step 6: Also test N=200..400 ---
    print("\nStep 6: Narrowband CMW N=200..400 (ultra-high)...")
    nb_ultra_params = design_narrowband_cmw_bank(
        w0=w0, max_N=400, fs=fs, fwhm_factor=0.5, omega_min=200 * w0 * 0.9
    )
    if len(nb_ultra_params) > 0:
        print(f"  {len(nb_ultra_params)} ultra-high filters designed")
        nb_ultra_result = run_cmw_comb_bank(log_prices, fs, nb_ultra_params, analytic=True)
        ultra_confirmed = extract_lines_from_narrowband(nb_ultra_result, w0)
        print(f"  Ultra-high confirmed: {len(ultra_confirmed)}/{len(nb_ultra_params)}")
    else:
        ultra_confirmed = []
        print("  No filters in this range (beyond Nyquist)")

    # --- Combine all confirmed lines ---
    all_confirmed = base_confirmed + ext_confirmed + ultra_confirmed

    # --- Step 7: Analysis ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if all_confirmed:
        max_N = max(l['N'] for l in all_confirmed)
        min_period_wk = 2 * np.pi / (max_N * w0) * 52
        min_period_days = min_period_wk * 7

        print(f"\nTotal confirmed harmonics: {len(all_confirmed)}")
        print(f"  N=2..80:    {len(base_confirmed)}")
        print(f"  N=81..200:  {len(ext_confirmed)}")
        print(f"  N=201..400: {len(ultra_confirmed)}")
        print(f"\nMax confirmed N: {max_N}")
        print(f"  Period: {min_period_wk:.1f} weeks ({min_period_days:.0f} days)")

        # 1/ω envelope test
        Ns = np.array([l['N'] for l in all_confirmed])
        amps = np.array([l['amplitude'] for l in all_confirmed])
        freqs_conf = Ns * w0

        # Log-log fit: log(amp) = log(k) - alpha*log(freq)
        valid = (amps > 0) & (freqs_conf > 0)
        if np.sum(valid) > 5:
            log_f = np.log(freqs_conf[valid])
            log_a = np.log(amps[valid])
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_f, log_a)
            r2 = r_value ** 2
            print(f"\n1/w Envelope fit (all confirmed):")
            print(f"  slope = {slope:.3f} (expected: -1.0)")
            print(f"  R2 = {r2:.4f}")
            print(f"  Consistent with 1/w: {'YES' if abs(slope + 1) < 0.3 and r2 > 0.8 else 'NO'}")

            # Separate fit for extended range only
            ext_all = ext_confirmed + ultra_confirmed
            if len(ext_all) >= 5:
                Ns_ext = np.array([l['N'] for l in ext_all])
                amps_ext = np.array([l['amplitude'] for l in ext_all])
                freqs_ext = Ns_ext * w0
                valid_ext = (amps_ext > 0) & (freqs_ext > 0)
                if np.sum(valid_ext) > 3:
                    slope_ext, _, r_ext, _, _ = stats.linregress(
                        np.log(freqs_ext[valid_ext]), np.log(amps_ext[valid_ext]))
                    print(f"\n1/w fit (N>80 only):")
                    print(f"  slope = {slope_ext:.3f}, R2 = {r_ext**2:.4f}")

        # Frequency stability vs N
        cvs = np.array([l['freq_cv'] for l in all_confirmed])
        print(f"\nFrequency stability:")
        base_cvs = [l['freq_cv'] for l in base_confirmed]
        ext_cvs = [l['freq_cv'] for l in ext_confirmed] if ext_confirmed else []
        print(f"  N=2..80 mean CV: {np.mean(base_cvs):.3f}")
        if ext_cvs:
            print(f"  N=81..200 mean CV: {np.mean(ext_cvs):.3f}")

        # Detailed table for extended harmonics
        if ext_confirmed:
            print(f"\n{'N':>4} {'Freq':>8} {'Period':>10} {'Amp':>10} {'CV%':>6} {'Conf':>6}")
            print("-" * 50)
            for l in sorted(ext_confirmed, key=lambda x: x['N']):
                period_d = 2 * np.pi / l['frequency'] * 365.25
                print(f"{l['N']:>4} {l['frequency']:>8.2f} {period_d:>8.1f}d "
                      f"{l['amplitude']:>10.6f} {l['freq_cv']*100:>5.1f}% {l['confidence']:>6}")
    else:
        print("\nNo confirmed harmonics found!")

    # --- Step 8: Plot ---
    print("\nGenerating figure...")
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # Panel 1: Spectrum with confirmed harmonics
    ax = axes[0]
    mask = (omega_yr > 25) & (omega_yr < 80)
    ax.semilogy(omega_yr[mask], amp[mask], 'b-', linewidth=0.3, alpha=0.5)

    # Mark confirmed harmonics
    for l in base_confirmed:
        ax.axvline(l['frequency'], color='green', alpha=0.15, linewidth=0.5)
    for l in ext_confirmed:
        ax.axvline(l['frequency'], color='red', alpha=0.3, linewidth=0.8)
    for l in ultra_confirmed:
        ax.axvline(l['frequency'], color='purple', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('w (rad/yr)')
    ax.set_ylabel('Amplitude (log)')
    ax.set_title(f'Daily DJIA Spectrum (25-80 rad/yr) with Confirmed Harmonics\n'
                 f'Green: N≤80 ({len(base_confirmed)}), '
                 f'Red: N=81-200 ({len(ext_confirmed)}), '
                 f'Purple: N>200 ({len(ultra_confirmed)})')
    ax.grid(True, alpha=0.3)

    # Panel 2: Amplitude vs N (log-log)
    ax = axes[1]
    if all_confirmed:
        Ns = np.array([l['N'] for l in all_confirmed])
        amps = np.array([l['amplitude'] for l in all_confirmed])

        # Color by range
        for l in base_confirmed:
            ax.scatter(l['N'], l['amplitude'], c='green', s=15, alpha=0.7, zorder=3)
        for l in ext_confirmed:
            ax.scatter(l['N'], l['amplitude'], c='red', s=20, alpha=0.8, zorder=3)
        for l in ultra_confirmed:
            ax.scatter(l['N'], l['amplitude'], c='purple', s=20, alpha=0.8, zorder=3)

        # 1/N fit line
        N_fit = np.linspace(2, max(Ns) + 10, 200)
        valid = (amps > 0) & (Ns > 0)
        if np.sum(valid) > 3:
            # Fit k/N^alpha
            log_N = np.log(Ns[valid])
            log_a = np.log(amps[valid])
            slope, intercept, _, _, _ = stats.linregress(log_N, log_a)
            k_fit = np.exp(intercept)
            ax.plot(N_fit, k_fit * N_fit ** slope, 'k--', linewidth=1,
                    label=f'Fit: A ~ N^{slope:.2f} (R2={r2:.3f})', alpha=0.7)

        ax.axvline(80, color='orange', linestyle=':', linewidth=1, label='N=80 boundary')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Harmonic Number N')
        ax.set_ylabel('Amplitude')
        ax.set_title('Amplitude vs Harmonic Number — Does 1/N Persist?')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Panel 3: Frequency stability (CV%) vs N
    ax = axes[2]
    if all_confirmed:
        for l in base_confirmed:
            ax.scatter(l['N'], l['freq_cv'] * 100, c='green', s=15, alpha=0.7)
        for l in ext_confirmed:
            ax.scatter(l['N'], l['freq_cv'] * 100, c='red', s=20, alpha=0.8)
        for l in ultra_confirmed:
            ax.scatter(l['N'], l['freq_cv'] * 100, c='purple', s=20, alpha=0.8)

        ax.axvline(80, color='orange', linestyle=':', linewidth=1, label='N=80 boundary')
        ax.axhline(30, color='gray', linestyle='--', linewidth=0.5, label='30% CV threshold')
        ax.set_xlabel('Harmonic Number N')
        ax.set_ylabel('Frequency CV (%)')
        ax.set_title('Frequency Stability vs N — Do High Harmonics Become Unstable?')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Harmonics Beyond N=80 — Extended Range Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_harmonics_beyond_N80.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
