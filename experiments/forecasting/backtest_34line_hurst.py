#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hurst 34-Line Spectral Model: Full Backtest

Uses Hurst's ACTUAL nominal model from Figure AI-8:
  w_n = n * 0.3676 rad/yr, for n = 1..34
  Period range: 17.1 years (line 1) to 6.06 months (line 34)

Fits amplitude and phase for each line via least squares.
Tests in-sample reconstruction and out-of-sample prediction.

Reference: Hurst, Figure AI-8, "Low-Frequency Portion: Spectral Model"
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, hilbert

# ---------------------------------------------------------------------------
# Hurst's exact 34-line nominal model (Figure AI-8)
# ---------------------------------------------------------------------------
DELTA_W = 0.3676  # rad/yr -- the fundamental spacing
N_LINES = 34
HURST_FREQS_RAD_YR = np.array([n * DELTA_W for n in range(1, N_LINES + 1)])
FS = 52  # weeks per year
TWOPI = 2 * np.pi

# Convert to rad/week for time-domain model
HURST_FREQS_RAD_WK = HURST_FREQS_RAD_YR / FS

# Hurst's nominal cycle labels from AI-8
HURST_NOMINAL_CYCLES = {
    1: "18.0 Y", 2: "9.0 Y", 4: "4.3 Y", 7: "3.0 Y",
    10: "18.0 M", 16: "12.0 M", 23: "9.0 M", 34: "6.0 M",
}

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))


def load_djia(date_start="1900-01-01", date_end="2025-12-31"):
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/^dji_w.csv"))
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df.Date.between(date_start, date_end)].copy().reset_index(drop=True)
    return df


def fit_harmonic_model(prices, freqs_rad_wk):
    """
    Fit x(t) = a0 + sum_i [A_i cos(w_i t) + B_i sin(w_i t)] via least squares.
    Returns coeffs, reconstructed signal, amplitudes, phases.
    """
    n = len(prices)
    t = np.arange(n, dtype=float)
    n_freqs = len(freqs_rad_wk)

    X = np.ones((n, 1 + 2 * n_freqs))
    for i, w in enumerate(freqs_rad_wk):
        X[:, 1 + 2*i]     = np.cos(w * t)
        X[:, 1 + 2*i + 1] = np.sin(w * t)

    coeffs, _, _, _ = np.linalg.lstsq(X, prices, rcond=None)
    reconstructed = X @ coeffs

    # Extract amplitude and phase for each line
    amplitudes = np.zeros(n_freqs)
    phases = np.zeros(n_freqs)
    for i in range(n_freqs):
        a = coeffs[1 + 2*i]
        b = coeffs[1 + 2*i + 1]
        amplitudes[i] = np.sqrt(a**2 + b**2)
        phases[i] = np.arctan2(-b, a)  # phase such that A*cos(wt + phi)

    return coeffs, reconstructed, amplitudes, phases


def predict_forward(coeffs, freqs_rad_wk, n_fit, n_predict):
    """Extrapolate harmonic model forward from t=n_fit to t=n_fit+n_predict."""
    t = np.arange(n_fit, n_fit + n_predict, dtype=float)
    n_freqs = len(freqs_rad_wk)

    X = np.ones((n_predict, 1 + 2 * n_freqs))
    for i, w in enumerate(freqs_rad_wk):
        X[:, 1 + 2*i]     = np.cos(w * t)
        X[:, 1 + 2*i + 1] = np.sin(w * t)

    return X @ coeffs


def reconstruct_components(coeffs, freqs_rad_wk, n_points, groups=None):
    """
    Reconstruct individual frequency group contributions.

    groups: dict mapping group_name -> list of line indices (0-based)
    Returns dict of group_name -> signal array
    """
    t = np.arange(n_points, dtype=float)
    result = {"dc": np.full(n_points, coeffs[0])}

    if groups is None:
        for i in range(len(freqs_rad_wk)):
            a = coeffs[1 + 2*i]
            b = coeffs[1 + 2*i + 1]
            sig = a * np.cos(freqs_rad_wk[i] * t) + b * np.sin(freqs_rad_wk[i] * t)
            result[f"line_{i+1}"] = sig
    else:
        for name, indices in groups.items():
            sig = np.zeros(n_points)
            for i in indices:
                a = coeffs[1 + 2*i]
                b = coeffs[1 + 2*i + 1]
                sig += a * np.cos(freqs_rad_wk[i] * t) + b * np.sin(freqs_rad_wk[i] * t)
            result[name] = sig

    return result


def find_turning_points(signal, order=8):
    max_idx = argrelextrema(signal, np.greater, order=order)[0]
    min_idx = argrelextrema(signal, np.less, order=order)[0]
    return max_idx, min_idx


def score_turning_points(actual_tp, predicted_tp, tolerance_weeks=6):
    hits = 0
    matched_predicted = set()
    for a_tp in actual_tp:
        distances = np.abs(predicted_tp - a_tp)
        if len(distances) == 0:
            continue
        closest_idx = np.argmin(distances)
        if distances[closest_idx] <= tolerance_weeks:
            hits += 1
            matched_predicted.add(closest_idx)

    hit_rate = hits / len(actual_tp) if len(actual_tp) > 0 else 0
    false_alarms = len(predicted_tp) - len(matched_predicted)
    false_alarm_rate = false_alarms / len(predicted_tp) if len(predicted_tp) > 0 else 0
    return hit_rate, false_alarm_rate, {
        "hits": hits, "total_actual": len(actual_tp),
        "total_predicted": len(predicted_tp), "false_alarms": false_alarms,
    }


def main():
    print("=" * 72)
    print("HURST 34-LINE SPECTRAL MODEL (Figure AI-8)")
    print("w_n = n * 0.3676 rad/yr, n = 1..34")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    df_full = load_djia("1921-04-29", "1965-05-21")
    prices_full = df_full.Close.values.astype(float)
    dates_full = pd.to_datetime(df_full.Date.values)
    n_full = len(prices_full)

    # Split: fit on 1921-1955, test on 1955-1965
    split_date = pd.Timestamp("1955-01-01")
    split_idx = np.searchsorted(dates_full, split_date)

    prices_fit = prices_full[:split_idx]
    prices_test = prices_full[split_idx:]
    dates_fit = dates_full[:split_idx]
    dates_test = dates_full[split_idx:]
    n_fit = len(prices_fit)
    n_test = len(prices_test)

    print(f"\nFull dataset: {n_full} weeks ({dates_full[0].date()} to {dates_full[-1].date()})")
    print(f"Fit period:   {n_fit} weeks ({dates_fit[0].date()} to {dates_fit[-1].date()})")
    print(f"Test period:  {n_test} weeks ({dates_test[0].date()} to {dates_test[-1].date()})")

    # -----------------------------------------------------------------------
    # Part 1: In-sample fit (full 1921-1965) with all 34 Hurst lines
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("PART 1: IN-SAMPLE FIT -- 34 Hurst Lines (full 1921-1965)")
    print("-" * 72)

    coeffs_full, recon_full, amps_full, phases_full = fit_harmonic_model(
        prices_full, HURST_FREQS_RAD_WK
    )

    residual_full = prices_full - recon_full
    ss_total = np.sum((prices_full - np.mean(prices_full)) ** 2)
    ss_residual = np.sum(residual_full ** 2)
    r_squared = 1 - ss_residual / ss_total
    rmse = np.sqrt(np.mean(residual_full ** 2))

    print(f"\nIn-sample R-squared:  {r_squared:.4f} ({r_squared*100:.1f}%)")
    print(f"In-sample RMSE:       {rmse:.2f} points")
    print(f"Mean price:           {np.mean(prices_full):.2f}")
    print(f"RMSE / Mean:          {rmse/np.mean(prices_full)*100:.1f}%")

    # Line-by-line amplitude table (matching Hurst's AI-8 format)
    print(f"\n  {'N':>3s}  {'w_n (rad/yr)':>12s}  {'T_yr':>6s}  {'T_mo':>6s}  {'T_wk':>6s}  {'Amp':>8s}  {'Phase':>8s}  {'Nominal':>8s}")
    print("  " + "-" * 72)

    total_amp_sq = np.sum(amps_full ** 2)
    cumulative_pct = 0.0

    for i in range(N_LINES):
        freq = HURST_FREQS_RAD_YR[i]
        t_yr = TWOPI / freq
        t_mo = t_yr * 12
        t_wk = t_yr * FS
        pct = (amps_full[i] ** 2) / total_amp_sq * 100
        cumulative_pct += pct
        nom = HURST_NOMINAL_CYCLES.get(i + 1, "")
        print(f"  {i+1:3d}  {freq:12.4f}  {t_yr:6.2f}  {t_mo:6.1f}  {t_wk:6.1f}  {amps_full[i]:8.2f}  {np.degrees(phases_full[i]):8.1f}  {nom:>8s}")

    # Group by Hurst's nominal cycles
    print(f"\n  Amplitude by Hurst cycle group:")
    groups = {
        "18Y  (lines 1-2)":   list(range(0, 2)),
        "4.3Y (lines 3-7)":   list(range(2, 7)),
        "18M  (lines 8-12)":  list(range(7, 12)),
        "12M  (lines 13-19)": list(range(12, 19)),
        "9M   (lines 20-26)": list(range(19, 26)),
        "6M   (lines 27-34)": list(range(26, 34)),
    }
    for name, indices in groups.items():
        group_energy = np.sum(amps_full[indices] ** 2)
        group_pct = group_energy / total_amp_sq * 100
        group_rms = np.sqrt(group_energy / 2)  # RMS of sum of sinusoids
        print(f"    {name}: energy={group_pct:5.1f}%  RMS~{group_rms:.1f} pts")

    # -----------------------------------------------------------------------
    # Part 2: Out-of-sample prediction
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("PART 2: OUT-OF-SAMPLE (fit 1921-1955, predict 1955-1965)")
    print("-" * 72)

    coeffs_fit, recon_fit, amps_fit, phases_fit = fit_harmonic_model(
        prices_fit, HURST_FREQS_RAD_WK
    )

    r2_fit = 1 - np.sum((prices_fit - recon_fit)**2) / np.sum((prices_fit - np.mean(prices_fit))**2)
    print(f"\nTraining R-squared: {r2_fit:.4f} ({r2_fit*100:.1f}%)")

    # Predict test period
    predicted = predict_forward(coeffs_fit, HURST_FREQS_RAD_WK, n_fit, n_test)
    pred_error = prices_test - predicted
    rmse_oos = np.sqrt(np.mean(pred_error ** 2))
    mae_oos = np.mean(np.abs(pred_error))

    print(f"Out-of-sample RMSE:   {rmse_oos:.2f} points")
    print(f"Out-of-sample MAE:    {mae_oos:.2f} points")
    print(f"Test mean price:      {np.mean(prices_test):.2f}")
    print(f"RMSE / Mean:          {rmse_oos/np.mean(prices_test)*100:.1f}%")

    # Correlation between predicted and actual
    corr = np.corrcoef(prices_test, predicted)[0, 1]
    print(f"Correlation (actual vs predicted): {corr:.4f}")

    # -----------------------------------------------------------------------
    # Part 3: Direction and turning point analysis (on RAW prices, not detrended)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("PART 3: TURNING POINT + DIRECTION (raw price, not detrended)")
    print("-" * 72)

    for horizon_name, horizon_wk in [("1 week", 1), ("4 weeks", 4), ("13 weeks", 13), ("26 weeks", 26)]:
        actual_dir = np.sign(prices_test[horizon_wk:] - prices_test[:-horizon_wk])
        pred_dir = np.sign(predicted[horizon_wk:] - predicted[:-horizon_wk])
        n_compare = min(len(actual_dir), len(pred_dir))
        correct = np.sum(actual_dir[:n_compare] == pred_dir[:n_compare])
        accuracy = correct / n_compare * 100
        print(f"  {horizon_name:>10s} direction accuracy: {accuracy:.1f}% ({correct}/{n_compare})")

    print("\n  Turning point scoring (raw prices):")
    for order, label in [(4, "~2 month"), (8, "~4 month"), (13, "~6 month"), (20, "~10 month")]:
        actual_max, actual_min = find_turning_points(prices_test, order=order)
        pred_max, pred_min = find_turning_points(predicted, order=order)
        actual_tp = np.sort(np.concatenate([actual_max, actual_min]))
        pred_tp = np.sort(np.concatenate([pred_max, pred_min]))

        for tol in [4, 8, 13]:
            hr, far, d = score_turning_points(actual_tp, pred_tp, tolerance_weeks=tol)
            if tol == 8:
                print(f"    {label:>10s} scale (tol=+-{tol}wk): hit={hr*100:.0f}% "
                      f"({d['hits']}/{d['total_actual']}), "
                      f"false_alarm={far*100:.0f}% ({d['false_alarms']}/{d['total_predicted']})")

    # -----------------------------------------------------------------------
    # Part 4: Phase stability across windows
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("PART 4: PHASE + AMPLITUDE STABILITY ACROSS WINDOWS")
    print("-" * 72)

    window_years = 20
    window_wk = window_years * FS
    step_wk = 5 * FS

    amp_history = []
    phase_history = []
    window_labels = []

    start = 0
    while start + window_wk <= n_full:
        window_prices = prices_full[start:start + window_wk]
        c, _, a, p = fit_harmonic_model(window_prices, HURST_FREQS_RAD_WK)
        amp_history.append(a)
        phase_history.append(p)
        s_date = dates_full[start]
        e_date = dates_full[min(start + window_wk - 1, n_full - 1)]
        window_labels.append(f"{s_date.year}-{e_date.year}")
        start += step_wk

    amp_history = np.array(amp_history)
    phase_history = np.array(phase_history)

    print(f"\n  {len(window_labels)} windows: {', '.join(window_labels)}")
    print(f"\n  {'N':>3s}  {'w_n':>8s}  {'Amp CV':>8s}  {'Phase Std':>10s}  {'Amp Stable':>10s}  {'Ph Stable':>10s}  {'Nom':>8s}")
    print("  " + "-" * 68)

    n_both_stable = 0
    for i in range(N_LINES):
        # Amplitude stability (coefficient of variation)
        amp_cv = np.std(amp_history[:, i]) / np.mean(amp_history[:, i]) if np.mean(amp_history[:, i]) > 0.01 else 99
        amp_stable = amp_cv < 0.5

        # Phase stability (circular std)
        R = np.abs(np.mean(np.exp(1j * phase_history[:, i])))
        circ_std = np.sqrt(-2 * np.log(max(R, 1e-10)))
        ph_stable = circ_std < 1.0

        both = amp_stable and ph_stable
        if both:
            n_both_stable += 1

        nom = HURST_NOMINAL_CYCLES.get(i + 1, "")
        print(f"  {i+1:3d}  {HURST_FREQS_RAD_YR[i]:8.4f}  {amp_cv:8.2f}  {circ_std:10.3f} rad  "
              f"{'YES' if amp_stable else 'no':>10s}  {'YES' if ph_stable else 'no':>10s}  {nom:>8s}")

    print(f"\n  Both stable: {n_both_stable}/{N_LINES} ({n_both_stable/N_LINES*100:.0f}%)")

    # -----------------------------------------------------------------------
    # Part 5: Compare to Hurst's claimed amplitudes (if available from AI-8)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("PART 5: AMPLITUDE ENVELOPE CHECK (do amplitudes follow k/w?)")
    print("-" * 72)

    # Check if amplitudes roughly follow 1/w or 1/n
    log_n = np.log(np.arange(1, N_LINES + 1))
    log_amp = np.log(np.maximum(amps_full, 0.01))
    slope, intercept = np.polyfit(log_n, log_amp, 1)
    print(f"\n  log(Amplitude) vs log(n) regression:")
    print(f"    slope = {slope:.3f} (expect ~-1.0 for k/w envelope)")
    print(f"    R-squared = {np.corrcoef(log_n, log_amp)[0,1]**2:.3f}")
    print(f"    Interpretation: amplitude ~ n^({slope:.2f}) = w^({slope:.2f})")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 18))

    # Panel 1: In-sample fit
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(dates_full, prices_full, "k-", linewidth=0.8, label="Actual DJIA", alpha=0.7)
    ax1.plot(dates_full, recon_full, "b-", linewidth=0.8, label=f"34-line Hurst model (R2={r_squared:.3f})", alpha=0.8)
    ax1.axvline(split_date, color="red", linestyle="--", alpha=0.7, label="Train/Test split")
    ax1.set_title("In-Sample: Hurst 34-Line Model vs Actual DJIA (1921-1965)", fontweight="bold")
    ax1.set_ylabel("DJIA Close")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Out-of-sample prediction
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(dates_test, prices_test, "k-", linewidth=0.8, label="Actual DJIA")
    ax2.plot(dates_test, predicted, "r-", linewidth=0.8, label=f"Predicted (corr={corr:.3f})", alpha=0.8)
    ax2.set_title("Out-of-Sample Price Prediction (1955-1965)", fontweight="bold")
    ax2.set_ylabel("DJIA Close")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Amplitude spectrum (bar chart comparing to k/w)
    ax3 = fig.add_subplot(4, 1, 3)
    line_numbers = np.arange(1, N_LINES + 1)
    ax3.bar(line_numbers, amps_full, color="steelblue", alpha=0.7, label="Fitted amplitude")
    # Overlay k/w fit
    k_fit = np.exp(intercept)
    kw_model = k_fit * line_numbers ** slope
    ax3.plot(line_numbers, kw_model, "r-", linewidth=2, label=f"k/n^{abs(slope):.2f} fit")
    ax3.set_xlabel("Line number n")
    ax3.set_ylabel("Amplitude (points)")
    ax3.set_title("Line Amplitudes vs Hurst k/w Envelope", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Amplitude stability across windows
    ax4 = fig.add_subplot(4, 1, 4)
    for w_idx, label in enumerate(window_labels):
        ax4.plot(line_numbers, amp_history[w_idx], "o-", markersize=3, linewidth=0.8, label=label, alpha=0.7)
    ax4.set_xlabel("Line number n")
    ax4.set_ylabel("Amplitude (points)")
    ax4.set_title("Amplitude Stability Across 20-Year Windows", fontweight="bold")
    ax4.legend(fontsize=8, ncol=3)
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "backtest_34line_hurst.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  34-line Hurst model (w_n = n * 0.3676, n=1..34)")
    print(f"  In-sample R2:         {r_squared*100:.1f}%")
    print(f"  Out-of-sample corr:   {corr:.3f}")
    print(f"  Amplitude law:        A ~ w^({slope:.2f})")
    print(f"  Stable lines (A+ph):  {n_both_stable}/{N_LINES}")
    print("=" * 72)


if __name__ == "__main__":
    main()
