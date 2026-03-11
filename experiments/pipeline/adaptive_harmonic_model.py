# -*- coding: utf-8 -*-
"""
Adaptive Harmonic Model -- Time-Varying Amplitude Projection

The static sinusoidal model fails because amplitudes aren't constant.
This script tests adaptive approaches:

  1) CMW envelope extrapolation: use recent envelope trend to project amplitude
  2) Windowed LS: fit sinusoids on a rolling window, project 1 window forward
  3) Hybrid: Ormsby grouped bands for trend, CMW for oscillatory calibration

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing (1970)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

from src.pipeline.derive_nominal_model import derive_nominal_model
from src.pipeline.comb_bank import design_narrowband_cmw_bank, run_cmw_comb_bank
from src.filters.funcOrmsby import ormsby_filter, apply_ormsby_filter

OUT_DIR = os.path.dirname(__file__)
W0 = 0.3676  # rad/yr


def load_data():
    """Load DJIA and run pipeline."""
    result = derive_nominal_model(
        symbol='djia', freq='weekly',
        start='1921-04-29', end='1965-05-21',
        prominence_frac=0.01, min_distance=2, verbose=False
    )
    return result.close, result.dates, result.fs


def remove_trend(log_prices, t_yr, fs, w0, n_fit):
    """Remove linear + N=1,2 trend, fitted only on training data."""
    # Linear trend from training only
    linear_coeffs = np.polyfit(t_yr[:n_fit], log_prices[:n_fit], 1)
    linear_trend = np.polyval(linear_coeffs, t_yr)
    lin_detrended = log_prices - linear_trend

    # N=1,2 from training only
    t_fit = t_yr[:n_fit]
    X_slow = np.column_stack([
        np.cos(w0 * t_fit), np.sin(w0 * t_fit),
        np.cos(2*w0 * t_fit), np.sin(2*w0 * t_fit),
    ])
    c_slow, _, _, _ = lstsq(X_slow, lin_detrended[:n_fit], rcond=None)

    # Project full series
    X_full = np.column_stack([
        np.cos(w0 * t_yr), np.sin(w0 * t_yr),
        np.cos(2*w0 * t_yr), np.sin(2*w0 * t_yr),
    ])
    slow_component = X_full @ c_slow
    residual = lin_detrended - slow_component

    return residual, linear_coeffs, c_slow


# =============================================================================
# Model 1: Static LS (baseline -- expected to fail)
# =============================================================================

def model_static_ls(residual, t_yr, n_fit, w0, harmonics):
    """Fit constant-amplitude sinusoids on training, project to holdback."""
    n_hold = len(residual) - n_fit
    t_fit = t_yr[:n_fit]
    t_hold = t_yr[n_fit:]
    y_fit = residual[:n_fit]
    y_hold = residual[n_fit:]

    n_harm = len(harmonics)
    X_fit = np.zeros((n_fit, 2 * n_harm))
    for j, N in enumerate(harmonics):
        X_fit[:, 2*j] = np.cos(N * w0 * t_fit)
        X_fit[:, 2*j+1] = np.sin(N * w0 * t_fit)

    c, _, _, _ = lstsq(X_fit, y_fit, rcond=None)
    y_fit_hat = X_fit @ c

    X_hold = np.zeros((n_hold, 2 * n_harm))
    for j, N in enumerate(harmonics):
        X_hold[:, 2*j] = np.cos(N * w0 * t_hold)
        X_hold[:, 2*j+1] = np.sin(N * w0 * t_hold)
    y_hold_hat = X_hold @ c

    return y_fit_hat, y_hold_hat


# =============================================================================
# Model 2: Windowed LS (rolling window, 1-step projection)
# =============================================================================

def model_windowed_ls(residual, t_yr, n_fit, w0, harmonics, window_yr=10.0, step_wk=13):
    """
    Rolling-window LS: fit sinusoids on a sliding window, project forward.

    For each step:
    1. Fit A,B for each harmonic on [t - window, t]
    2. Project forward by step_wk weeks
    3. Stitch projections together

    This naturally adapts amplitudes as the window slides.
    """
    fs = 1.0 / (t_yr[1] - t_yr[0])  # samples per year
    window_samples = int(window_yr * fs)
    step_samples = step_wk
    n_total = len(residual)
    n_harm = len(harmonics)

    projection = np.full(n_total, np.nan)

    # Start projecting from after first full window
    start = window_samples
    t_idx = start

    while t_idx < n_total:
        # Fit window: [t_idx - window_samples, t_idx)
        i0 = max(0, t_idx - window_samples)
        i1 = t_idx
        t_win = t_yr[i0:i1]
        y_win = residual[i0:i1]

        # Build design matrix
        X_win = np.zeros((len(t_win), 2 * n_harm))
        for j, N in enumerate(harmonics):
            X_win[:, 2*j] = np.cos(N * w0 * t_win)
            X_win[:, 2*j+1] = np.sin(N * w0 * t_win)

        c, _, _, _ = lstsq(X_win, y_win, rcond=None)

        # Project forward by step_samples
        proj_end = min(t_idx + step_samples, n_total)
        t_proj = t_yr[t_idx:proj_end]
        X_proj = np.zeros((len(t_proj), 2 * n_harm))
        for j, N in enumerate(harmonics):
            X_proj[:, 2*j] = np.cos(N * w0 * t_proj)
            X_proj[:, 2*j+1] = np.sin(N * w0 * t_proj)

        proj = X_proj @ c
        projection[t_idx:proj_end] = proj

        t_idx += step_samples

    return projection


# =============================================================================
# Model 3: CMW Envelope Extrapolation
# =============================================================================

def model_cmw_adaptive(residual, t_yr, n_fit, fs, w0, harmonics,
                        extrap_window_yr=5.0):
    """
    Use CMW to extract instantaneous amplitude and phase for each harmonic.
    Extrapolate the envelope using recent trend (linear fit on last N years).
    Project forward using: envelope(t) * cos(w*t + phase(t_end)).
    """
    n_total = len(residual)
    n_hold = n_total - n_fit

    # Design CMW bank
    cmw_params = []
    for N in harmonics:
        f0 = N * w0
        fwhm = max(w0 * 0.5, f0 * 0.1)
        cmw_params.append({
            'f0': f0, 'fwhm': fwhm, 'N': N,
            'period_yr': 2 * np.pi / f0,
            'label': f'N={N}',
        })

    # Apply CMW to FULL residual (we'll only use training portion for fitting)
    bank_result = run_cmw_comb_bank(residual, fs, cmw_params, analytic=True)

    # For each harmonic, extrapolate envelope and project phase
    projection = np.zeros(n_total)
    extrap_samples = int(extrap_window_yr * fs)

    for i, output in enumerate(bank_result['filter_outputs']):
        N = harmonics[i]
        if output['envelope'] is None or output['phase'] is None:
            continue

        env = output['envelope']
        phase = output['phase']
        sig = np.real(output['signal'])

        # Use only training data for envelope extrapolation
        env_train = env[:n_fit]
        phase_train = phase[:n_fit]

        # Fit linear trend to last extrap_window years of envelope
        i0 = max(0, n_fit - extrap_samples)
        t_extrap = t_yr[i0:n_fit]
        env_extrap = env_train[i0:n_fit]

        # Linear fit on log(envelope) for multiplicative extrapolation
        valid = env_extrap > 0
        if valid.sum() > 10:
            log_env = np.log(env_extrap[valid])
            t_valid = t_extrap[valid]
            p = np.polyfit(t_valid, log_env, 1)
            # Extrapolated envelope
            env_projected = np.exp(np.polyval(p, t_yr[n_fit:]))
            # Clip to reasonable range (0.1x to 10x of recent mean)
            recent_mean = np.mean(env_extrap[-52:]) if len(env_extrap) >= 52 else np.mean(env_extrap)
            env_projected = np.clip(env_projected, recent_mean * 0.1, recent_mean * 10)
        else:
            env_projected = np.full(n_hold, np.mean(env_train))

        # Phase at end of training
        end_phase = phase_train[-1]
        omega = N * w0

        # Project: A(t) * cos(w*t + phase_offset)
        # Phase evolves as w*t, offset calibrated at boundary
        phase_offset = end_phase - omega * t_yr[n_fit - 1]

        for t_idx in range(n_fit, n_total):
            j = t_idx - n_fit
            if j < len(env_projected):
                amp = env_projected[j]
            else:
                amp = env_projected[-1]
            projection[t_idx] += amp * np.cos(omega * t_yr[t_idx] + phase_offset)

        # Training portion: use actual CMW output
        projection[:n_fit] += sig[:n_fit]

    return projection


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(y_actual, y_predicted, label=""):
    """Compute R2 and correlation."""
    ss_res = np.sum((y_actual - y_predicted)**2)
    ss_tot = np.sum((y_actual - np.mean(y_actual))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    # Handle NaN in predictions
    valid = ~np.isnan(y_predicted)
    if valid.sum() > 2:
        corr = np.corrcoef(y_actual[valid], y_predicted[valid])[0, 1]
    else:
        corr = 0
    rmse = np.sqrt(np.mean((y_actual[valid] - y_predicted[valid])**2))
    if label:
        print(f"  {label}: R2={r2:.4f}, corr={corr:.4f}, RMSE={rmse:.4f}")
    return r2, corr, rmse


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ADAPTIVE HARMONIC MODEL")
    print("=" * 70)

    close, dates, fs = load_data()
    log_prices = np.log(close)
    t_yr = np.arange(len(close)) / fs
    n_total = len(close)

    # Split: 80% train, 20% holdback
    holdback_frac = 0.20
    n_fit = int(n_total * (1 - holdback_frac))
    n_hold = n_total - n_fit
    print(f"Data: {n_total} samples, {n_total/fs:.1f} yr")
    print(f"Train: {n_fit} ({n_fit/fs:.1f} yr), Hold: {n_hold} ({n_hold/fs:.1f} yr)")

    # Remove trend (linear + N=1,2)
    residual, linear_coeffs, c_slow = remove_trend(log_prices, t_yr, fs, W0, n_fit)
    y_fit = residual[:n_fit]
    y_hold = residual[n_fit:]
    print(f"Residual std: train={np.std(y_fit):.4f}, hold={np.std(y_hold):.4f}")

    harmonics = list(range(3, 25))  # N=3 to N=24 (avoid over-fitting high-N)
    print(f"Harmonics: N={harmonics[0]} to N={harmonics[-1]} ({len(harmonics)} total)")

    # === Model 1: Static LS ===
    print("\n--- Model 1: Static LS (baseline) ---")
    y_fit_static, y_hold_static = model_static_ls(residual, t_yr, n_fit, W0, harmonics)
    print("  Training:")
    evaluate(y_fit, y_fit_static, "  Fit")
    print("  Holdback:")
    r2_static, corr_static, rmse_static = evaluate(y_hold, y_hold_static, "  Hold")

    # === Model 2: Windowed LS (10yr window, 13wk step) ===
    print("\n--- Model 2: Windowed LS (10yr window, 13wk step) ---")
    proj_win10 = model_windowed_ls(residual, t_yr, n_fit, W0, harmonics,
                                    window_yr=10.0, step_wk=13)
    valid_hold = ~np.isnan(proj_win10[n_fit:])
    print("  Holdback:")
    r2_win10, corr_win10, _ = evaluate(y_hold[valid_hold], proj_win10[n_fit:][valid_hold], "  Hold")

    # Also test 5yr and 15yr windows
    print("\n--- Model 2b: Windowed LS (5yr window, 13wk step) ---")
    proj_win5 = model_windowed_ls(residual, t_yr, n_fit, W0, harmonics,
                                   window_yr=5.0, step_wk=13)
    valid5 = ~np.isnan(proj_win5[n_fit:])
    r2_win5, corr_win5, _ = evaluate(y_hold[valid5], proj_win5[n_fit:][valid5], "  Hold")

    print("\n--- Model 2c: Windowed LS (15yr window, 13wk step) ---")
    proj_win15 = model_windowed_ls(residual, t_yr, n_fit, W0, harmonics,
                                    window_yr=15.0, step_wk=13)
    valid15 = ~np.isnan(proj_win15[n_fit:])
    r2_win15, corr_win15, _ = evaluate(y_hold[valid15], proj_win15[n_fit:][valid15], "  Hold")

    # === Model 3: CMW envelope extrapolation ===
    print("\n--- Model 3: CMW Envelope Extrapolation (5yr extrap window) ---")
    proj_cmw = model_cmw_adaptive(residual, t_yr, n_fit, fs, W0, harmonics,
                                   extrap_window_yr=5.0)
    r2_cmw, corr_cmw, _ = evaluate(y_hold, proj_cmw[n_fit:], "  Hold")

    # === Model 4: Fewer harmonics (N=3-10 only) ===
    print("\n--- Model 4: Windowed LS, fewer harmonics (N=3-10) ---")
    harmonics_few = list(range(3, 11))
    proj_few = model_windowed_ls(residual, t_yr, n_fit, W0, harmonics_few,
                                  window_yr=10.0, step_wk=13)
    valid_few = ~np.isnan(proj_few[n_fit:])
    r2_few, corr_few, _ = evaluate(y_hold[valid_few], proj_few[n_fit:][valid_few], "  Hold")

    # === Model 5: Windowed LS with shorter projection (4wk step) ===
    print("\n--- Model 5: Windowed LS (10yr, 4wk step -- shorter projection) ---")
    proj_short = model_windowed_ls(residual, t_yr, n_fit, W0, harmonics,
                                    window_yr=10.0, step_wk=4)
    valid_short = ~np.isnan(proj_short[n_fit:])
    r2_short, corr_short, _ = evaluate(y_hold[valid_short], proj_short[n_fit:][valid_short], "  Hold")

    # =========================================================================
    # Summary and Plots
    # =========================================================================

    print("\n" + "=" * 70)
    print("SUMMARY: Holdback Performance")
    print("=" * 70)
    results = [
        ("Static LS (33 harm)", r2_static, corr_static),
        ("Windowed LS 5yr/13wk", r2_win5, corr_win5),
        ("Windowed LS 10yr/13wk", r2_win10, corr_win10),
        ("Windowed LS 15yr/13wk", r2_win15, corr_win15),
        ("Windowed LS 10yr/4wk", r2_short, corr_short),
        ("CMW envelope extrap", r2_cmw, corr_cmw),
        ("Windowed LS few (N=3-10)", r2_few, corr_few),
    ]
    print(f"  {'Model':<30} {'R2':>8} {'Corr':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    for name, r2, corr in results:
        print(f"  {name:<30} {r2:>8.4f} {corr:>8.4f}")

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    t_hold = t_yr[n_fit:]

    # Panel 1: All models on holdback
    ax = axes[0, 0]
    ax.plot(t_hold, y_hold, 'b-', lw=1, label='Actual')
    ax.plot(t_hold, y_hold_static, 'r-', lw=0.7, alpha=0.5,
            label=f'Static (R2={r2_static:.3f})')
    valid_w = ~np.isnan(proj_win10[n_fit:])
    ax.plot(t_hold[valid_w], proj_win10[n_fit:][valid_w], 'g-', lw=0.7,
            label=f'Windowed 10yr (R2={r2_win10:.3f})')
    ax.plot(t_hold, proj_cmw[n_fit:], 'm-', lw=0.7, alpha=0.5,
            label=f'CMW extrap (R2={r2_cmw:.3f})')
    ax.set_title('Holdback: Model Comparison')
    ax.set_xlabel('Time (years)')
    ax.legend(fontsize=7)

    # Panel 2: Windowed LS window size comparison
    ax = axes[0, 1]
    ax.plot(t_hold, y_hold, 'b-', lw=1, label='Actual')
    for proj, lbl, clr in [
        (proj_win5, f'5yr (R2={r2_win5:.3f})', 'orange'),
        (proj_win10, f'10yr (R2={r2_win10:.3f})', 'green'),
        (proj_win15, f'15yr (R2={r2_win15:.3f})', 'purple'),
    ]:
        v = ~np.isnan(proj[n_fit:])
        ax.plot(t_hold[v], proj[n_fit:][v], '-', color=clr, lw=0.7, label=lbl)
    ax.set_title('Window Size Comparison')
    ax.set_xlabel('Time (years)')
    ax.legend(fontsize=7)

    # Panel 3: Full series with windowed LS
    ax = axes[1, 0]
    ax.plot(t_yr, residual, 'b-', lw=0.3, alpha=0.5, label='Actual')
    valid_all = ~np.isnan(proj_win10)
    ax.plot(t_yr[valid_all], proj_win10[valid_all], 'g-', lw=0.5, label='Windowed 10yr')
    ax.axvline(t_yr[n_fit], color='black', ls='--', lw=1, label='Train/Hold split')
    ax.set_title('Full Series: Windowed LS Model')
    ax.set_xlabel('Time (years)')
    ax.legend(fontsize=7)

    # Panel 4: Bar chart comparison
    ax = axes[1, 1]
    names = [r[0] for r in results]
    r2s = [max(r[1], -1) for r in results]  # Clip for display
    corrs = [r[2] for r in results]
    x = np.arange(len(names))
    w = 0.35
    ax.barh(x - w/2, r2s, w, label='R2', color='steelblue')
    ax.barh(x + w/2, corrs, w, label='Correlation', color='coral')
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Score')
    ax.set_title('Model Comparison')
    ax.legend(fontsize=8)

    # Panel 5: Holdback error comparison
    ax = axes[2, 0]
    err_static = y_hold - y_hold_static
    err_win10 = y_hold - proj_win10[n_fit:]
    ax.plot(t_hold, err_static, 'r-', lw=0.5, alpha=0.5, label=f'Static (RMSE={rmse_static:.4f})')
    v = ~np.isnan(err_win10)
    rmse_w10 = np.sqrt(np.nanmean(err_win10[v]**2))
    ax.plot(t_hold[v], err_win10[v], 'g-', lw=0.5, label=f'Windowed (RMSE={rmse_w10:.4f})')
    ax.axhline(0, color='black', lw=0.3)
    ax.set_title('Holdback Prediction Error')
    ax.set_xlabel('Time (years)')
    ax.legend(fontsize=8)

    # Panel 6: Fewer harmonics
    ax = axes[2, 1]
    ax.plot(t_hold, y_hold, 'b-', lw=1, label='Actual')
    v_few = ~np.isnan(proj_few[n_fit:])
    ax.plot(t_hold[v_few], proj_few[n_fit:][v_few], 'g-', lw=0.7,
            label=f'N=3-10 (R2={r2_few:.3f})')
    v_short = ~np.isnan(proj_short[n_fit:])
    ax.plot(t_hold[v_short], proj_short[n_fit:][v_short], 'm-', lw=0.7,
            label=f'4wk step (R2={r2_short:.3f})')
    ax.set_title('Fewer Harmonics & Shorter Step')
    ax.set_xlabel('Time (years)')
    ax.legend(fontsize=7)

    plt.suptitle("Adaptive Harmonic Model: Holdback Validation (DJIA 1921-1965)",
                 fontsize=14, fontweight='bold')
    path = os.path.join(OUT_DIR, 'fig_adaptive_harmonic_model.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {path}")
    print("=" * 70)
