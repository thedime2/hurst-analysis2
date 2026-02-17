#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
27-Line Parametric Model: In-Sample Fit + Out-of-Sample Backtest

Tests Hurst's framework for forecasting by:
1. Fitting 27 sinusoids (at nominal model frequencies) via least squares
2. In-sample reconstruction quality (1921-1965)
3. Out-of-sample: fit on 1921-1955, predict 1955-1965 turning points
4. Score: did predicted turning points match actual?

Reference: Hurst, "The Profit Magic of Stock Transaction Timing", Appendix A
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
from scipy.signal import argrelextrema

from src.spectral import lanczos_spectrum

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS = 52  # samples per year
TWOPI = 2 * np.pi

# 27 nominal line frequencies (rad/yr) from Phase 3
NOMINAL_FREQS_RAD_YR = np.array([
    2.2768, 3.3604, 3.9378, 4.2273, 4.4554, 4.7251, 5.1168, 5.5133,
    5.7726, 6.2028, 6.7514, 7.0568, 7.6708, 7.8985, 8.3364, 8.8493,
    9.0705, 9.2539, 9.4614, 9.6728, 10.1637, 10.3876, 10.8129, 11.1258,
    11.4806, 11.7310, 11.9471,
])

# Convert to rad/week for time-domain model
NOMINAL_FREQS_RAD_WK = NOMINAL_FREQS_RAD_YR / FS

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))


def load_djia(date_start="1900-01-01", date_end="2025-12-31"):
    """Load weekly DJIA data."""
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/^dji_w.csv"))
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df.Date.between(date_start, date_end)].copy().reset_index(drop=True)
    return df


def fit_harmonic_model(prices, freqs_rad_wk):
    """
    Fit x(t) = a0 + sum_i [A_i cos(w_i t) + B_i sin(w_i t)] via least squares.

    Parameters:
        prices: 1D array of price data
        freqs_rad_wk: frequencies in rad/week

    Returns:
        coeffs: (a0, A1, B1, A2, B2, ...) array
        reconstructed: fitted signal
        design_matrix: the X matrix used
    """
    n = len(prices)
    t = np.arange(n, dtype=float)
    n_freqs = len(freqs_rad_wk)

    # Design matrix: [1, cos(w1*t), sin(w1*t), cos(w2*t), sin(w2*t), ...]
    X = np.ones((n, 1 + 2 * n_freqs))
    for i, w in enumerate(freqs_rad_wk):
        X[:, 1 + 2*i]     = np.cos(w * t)
        X[:, 1 + 2*i + 1] = np.sin(w * t)

    # Solve via least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, prices, rcond=None)
    reconstructed = X @ coeffs

    return coeffs, reconstructed, X


def predict_forward(coeffs, freqs_rad_wk, n_fit, n_predict):
    """
    Extrapolate the harmonic model forward.

    Parameters:
        coeffs: fitted coefficients from fit_harmonic_model
        freqs_rad_wk: frequencies in rad/week
        n_fit: number of points used in fitting
        n_predict: number of points to predict

    Returns:
        predicted: array of length n_predict
    """
    t = np.arange(n_fit, n_fit + n_predict, dtype=float)
    n_freqs = len(freqs_rad_wk)

    X = np.ones((n_predict, 1 + 2 * n_freqs))
    for i, w in enumerate(freqs_rad_wk):
        X[:, 1 + 2*i]     = np.cos(w * t)
        X[:, 1 + 2*i + 1] = np.sin(w * t)

    return X @ coeffs


def detrend_prices(prices):
    """Remove linear trend, return detrended + trend coefficients."""
    n = len(prices)
    t = np.arange(n, dtype=float)
    # Fit linear trend
    p = np.polyfit(t, prices, 1)
    trend = np.polyval(p, t)
    return prices - trend, p


def find_turning_points(signal, order=8):
    """Find local maxima and minima in a signal."""
    max_idx = argrelextrema(signal, np.greater, order=order)[0]
    min_idx = argrelextrema(signal, np.less, order=order)[0]
    return max_idx, min_idx


def score_turning_points(actual_tp, predicted_tp, tolerance_weeks=6):
    """
    Score turning point predictions.

    For each actual turning point, check if there's a predicted one within tolerance.

    Returns:
        hit_rate: fraction of actual TPs matched by predicted TPs
        false_alarm_rate: fraction of predicted TPs not matching any actual TP
        details: dict with matched/unmatched lists
    """
    hits = 0
    matched_actual = []
    matched_predicted = set()

    for a_tp in actual_tp:
        distances = np.abs(predicted_tp - a_tp)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] <= tolerance_weeks:
            hits += 1
            matched_actual.append(a_tp)
            matched_predicted.add(closest_idx)

    hit_rate = hits / len(actual_tp) if len(actual_tp) > 0 else 0
    false_alarms = len(predicted_tp) - len(matched_predicted)
    false_alarm_rate = false_alarms / len(predicted_tp) if len(predicted_tp) > 0 else 0

    return hit_rate, false_alarm_rate, {
        "hits": hits,
        "total_actual": len(actual_tp),
        "total_predicted": len(predicted_tp),
        "false_alarms": false_alarms,
    }


def main():
    print("=" * 72)
    print("27-LINE PARAMETRIC MODEL: IN-SAMPLE FIT + OUT-OF-SAMPLE BACKTEST")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    df_full = load_djia("1921-04-29", "1965-05-21")
    prices_full = df_full.Close.values.astype(float)
    dates_full = pd.to_datetime(df_full.Date.values)
    n_full = len(prices_full)
    print(f"\nFull dataset: {n_full} weeks, {dates_full[0].date()} to {dates_full[-1].date()}")

    # Split: fit on 1921-1955, test on 1955-1965
    split_date = pd.Timestamp("1955-01-01")
    split_idx = np.searchsorted(dates_full, split_date)

    prices_fit = prices_full[:split_idx]
    prices_test = prices_full[split_idx:]
    dates_fit = dates_full[:split_idx]
    dates_test = dates_full[split_idx:]
    n_fit = len(prices_fit)
    n_test = len(prices_test)
    print(f"Fit period:  {n_fit} weeks, {dates_fit[0].date()} to {dates_fit[-1].date()}")
    print(f"Test period: {n_test} weeks, {dates_test[0].date()} to {dates_test[-1].date()}")

    # -----------------------------------------------------------------------
    # Part 1: In-sample fit (full 1921-1965)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("PART 1: IN-SAMPLE FIT (full 1921-1965)")
    print("-" * 72)

    coeffs_full, recon_full, X_full = fit_harmonic_model(prices_full, NOMINAL_FREQS_RAD_WK)

    residual_full = prices_full - recon_full
    ss_total = np.sum((prices_full - np.mean(prices_full)) ** 2)
    ss_residual = np.sum(residual_full ** 2)
    r_squared = 1 - ss_residual / ss_total
    rmse = np.sqrt(np.mean(residual_full ** 2))

    print(f"\nIn-sample R-squared:  {r_squared:.4f} ({r_squared*100:.1f}%)")
    print(f"In-sample RMSE:       {rmse:.2f} points")
    print(f"Mean price:           {np.mean(prices_full):.2f}")
    print(f"RMSE / Mean price:    {rmse/np.mean(prices_full)*100:.1f}%")

    # Amplitude of each line
    print(f"\nLine amplitudes (top 10 by amplitude):")
    amps = []
    for i in range(len(NOMINAL_FREQS_RAD_YR)):
        a = coeffs_full[1 + 2*i]
        b = coeffs_full[1 + 2*i + 1]
        amp = np.sqrt(a**2 + b**2)
        amps.append(amp)
    amps = np.array(amps)
    sorted_idx = np.argsort(amps)[::-1]

    print(f"  {'Line':>4s}  {'Freq (rad/yr)':>14s}  {'Period (wk)':>11s}  {'Amplitude':>10s}")
    for rank, idx in enumerate(sorted_idx[:10]):
        freq = NOMINAL_FREQS_RAD_YR[idx]
        period_wk = TWOPI / freq * FS
        print(f"  {idx+1:4d}  {freq:14.4f}  {period_wk:11.1f}  {amps[idx]:10.2f}")

    # -----------------------------------------------------------------------
    # Part 2: Out-of-sample backtest
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("PART 2: OUT-OF-SAMPLE BACKTEST (fit 1921-1955, predict 1955-1965)")
    print("-" * 72)

    # Fit on training period only
    coeffs_fit, recon_fit, X_fit = fit_harmonic_model(prices_fit, NOMINAL_FREQS_RAD_WK)

    # In-sample quality on training set
    ss_total_fit = np.sum((prices_fit - np.mean(prices_fit)) ** 2)
    ss_res_fit = np.sum((prices_fit - recon_fit) ** 2)
    r2_fit = 1 - ss_res_fit / ss_total_fit
    print(f"\nTraining R-squared: {r2_fit:.4f}")

    # Predict forward
    predicted = predict_forward(coeffs_fit, NOMINAL_FREQS_RAD_WK, n_fit, n_test)

    # Raw prediction error
    pred_error = prices_test - predicted
    rmse_oos = np.sqrt(np.mean(pred_error ** 2))
    print(f"Out-of-sample RMSE: {rmse_oos:.2f} points")
    print(f"Test period mean:   {np.mean(prices_test):.2f}")
    print(f"RMSE / Mean price:  {rmse_oos/np.mean(prices_test)*100:.1f}%")

    # -----------------------------------------------------------------------
    # Part 3: Turning point analysis (detrended)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("PART 3: TURNING POINT PREDICTION (detrended)")
    print("-" * 72)

    # Detrend: remove linear trend from full series, refit harmonics
    prices_detrended, trend_p = detrend_prices(prices_full)

    # Fit harmonic model to detrended prices (training period)
    detrended_fit = prices_detrended[:split_idx]
    detrended_test = prices_detrended[split_idx:]

    coeffs_dt, recon_dt, _ = fit_harmonic_model(detrended_fit, NOMINAL_FREQS_RAD_WK)
    predicted_dt = predict_forward(coeffs_dt, NOMINAL_FREQS_RAD_WK, n_fit, n_test)

    # Find turning points in actual detrended test data
    for order in [4, 8, 13, 20]:
        period_label = f"~{order*2} weeks"
        actual_max, actual_min = find_turning_points(detrended_test, order=order)
        pred_max, pred_min = find_turning_points(predicted_dt, order=order)
        actual_tp = np.sort(np.concatenate([actual_max, actual_min]))
        pred_tp = np.sort(np.concatenate([pred_max, pred_min]))

        for tol in [4, 6, 8, 13]:
            hr, far, details = score_turning_points(actual_tp, pred_tp, tolerance_weeks=tol)
            if tol == 6:
                print(f"\n  Cycle scale {period_label} (order={order}, tol=+-{tol}wk):")
                print(f"    Actual turning points:    {details['total_actual']}")
                print(f"    Predicted turning points: {details['total_predicted']}")
                print(f"    Hits:                     {details['hits']}")
                print(f"    Hit rate:                 {hr*100:.1f}%")
                print(f"    False alarm rate:         {far*100:.1f}%")

    # -----------------------------------------------------------------------
    # Part 4: Direction prediction (simpler test)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("PART 4: DIRECTION PREDICTION (weekly, monthly, quarterly)")
    print("-" * 72)

    for horizon_name, horizon_wk in [("1 week", 1), ("4 weeks", 4), ("13 weeks", 13), ("26 weeks", 26)]:
        # Actual direction
        actual_dir = np.sign(detrended_test[horizon_wk:] - detrended_test[:-horizon_wk])
        pred_dir = np.sign(predicted_dt[horizon_wk:] - predicted_dt[:-horizon_wk])

        n_compare = min(len(actual_dir), len(pred_dir))
        actual_dir = actual_dir[:n_compare]
        pred_dir = pred_dir[:n_compare]

        correct = np.sum(actual_dir == pred_dir)
        total = len(actual_dir)
        accuracy = correct / total * 100
        print(f"  {horizon_name:>10s} direction accuracy: {accuracy:.1f}% ({correct}/{total})")

    # -----------------------------------------------------------------------
    # Part 5: Rolling window stability test
    # -----------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("PART 5: PHASE STABILITY (are fitted phases stable across windows?)")
    print("-" * 72)

    # Fit on multiple overlapping windows, compare phase estimates
    window_years = 20
    window_wk = window_years * FS
    step_wk = 5 * FS  # 5-year steps

    phase_history = []
    window_labels = []

    start = 0
    while start + window_wk <= n_full:
        window_prices = prices_full[start:start + window_wk]
        # Detrend window
        dt_win, _ = detrend_prices(window_prices)
        c, _, _ = fit_harmonic_model(dt_win, NOMINAL_FREQS_RAD_WK)

        # Extract phases
        phases = []
        for i in range(len(NOMINAL_FREQS_RAD_YR)):
            a = c[1 + 2*i]
            b = c[1 + 2*i + 1]
            phase = np.arctan2(b, a)
            phases.append(phase)
        phase_history.append(np.array(phases))

        start_date = dates_full[start]
        end_date = dates_full[min(start + window_wk - 1, n_full - 1)]
        window_labels.append(f"{start_date.year}-{end_date.year}")
        start += step_wk

    phase_history = np.array(phase_history)  # shape: (n_windows, 27)

    # Phase stability: circular std for each frequency across windows
    print(f"\n  {len(window_labels)} windows of {window_years} years, {step_wk//FS}-year steps")
    print(f"\n  {'Line':>4s}  {'Freq':>8s}  {'Phase StdDev':>12s}  {'Stable?':>8s}")

    n_stable = 0
    for i in range(len(NOMINAL_FREQS_RAD_YR)):
        # Circular standard deviation
        phases_i = phase_history[:, i]
        R = np.abs(np.mean(np.exp(1j * phases_i)))
        circ_std = np.sqrt(-2 * np.log(max(R, 1e-10)))
        stable = circ_std < 1.0  # less than ~57 degrees
        if stable:
            n_stable += 1
        print(f"  {i+1:4d}  {NOMINAL_FREQS_RAD_YR[i]:8.3f}  {circ_std:12.3f} rad  {'YES' if stable else 'no':>8s}")

    print(f"\n  Stable lines: {n_stable}/{len(NOMINAL_FREQS_RAD_YR)} ({n_stable/len(NOMINAL_FREQS_RAD_YR)*100:.0f}%)")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Fig 1: In-sample fit
    ax = axes[0]
    ax.plot(dates_full, prices_full, "k-", linewidth=0.8, label="Actual DJIA", alpha=0.7)
    ax.plot(dates_full, recon_full, "b-", linewidth=0.8, label=f"27-line model (R2={r_squared:.3f})", alpha=0.8)
    ax.axvline(split_date, color="red", linestyle="--", alpha=0.7, label="Train/Test split")
    ax.set_title("In-Sample: 27-Line Harmonic Model vs Actual DJIA (1921-1965)", fontweight="bold")
    ax.set_ylabel("DJIA Close")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Fig 2: Out-of-sample (price level)
    ax = axes[1]
    ax.plot(dates_test, prices_test, "k-", linewidth=0.8, label="Actual DJIA")
    ax.plot(dates_test, predicted, "r-", linewidth=0.8, label="Predicted (27-line extrapolation)", alpha=0.8)
    ax.set_title("Out-of-Sample: Price Level Prediction (1955-1965)", fontweight="bold")
    ax.set_ylabel("DJIA Close")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Fig 3: Out-of-sample (detrended - turning points)
    ax = axes[2]
    ax.plot(dates_test, detrended_test, "k-", linewidth=0.7, label="Actual (detrended)", alpha=0.7)
    ax.plot(dates_test, predicted_dt, "r-", linewidth=0.7, label="Predicted (detrended)", alpha=0.8)

    # Mark turning points
    order_tp = 8
    act_max, act_min = find_turning_points(detrended_test, order=order_tp)
    pre_max, pre_min = find_turning_points(predicted_dt, order=order_tp)
    ax.plot(dates_test[act_max], detrended_test[act_max], "kv", markersize=6, label="Actual peaks")
    ax.plot(dates_test[act_min], detrended_test[act_min], "k^", markersize=6, label="Actual troughs")
    ax.plot(dates_test[pre_max], predicted_dt[pre_max], "rv", markersize=5, alpha=0.7, label="Predicted peaks")
    ax.plot(dates_test[pre_min], predicted_dt[pre_min], "r^", markersize=5, alpha=0.7, label="Predicted troughs")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Out-of-Sample: Detrended Turning Point Prediction", fontweight="bold")
    ax.set_ylabel("Detrended Price")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "backtest_27line_model.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
