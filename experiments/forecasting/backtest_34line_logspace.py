#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hurst 34-Line Model in LOG SPACE

Key insight: DJIA grows exponentially, so cycles are multiplicative.
In log(price), exponential growth = linear trend, and Hurst's sinusoids
model percentage oscillations around that trend.

Model: log(price) = a + b*t + sum_{n=1}^{34} [A_n cos(w_n t) + B_n sin(w_n t)]

Tests:
  A) Full 34 lines + linear trend on log(prices)
  B) 33 lines (drop line 1, the unstable 17yr cycle) + linear trend
  C) Rolling 10-year fit, 2-year forecast (multiple windows)
  D) Post-1965 validation (does it work in modern era?)

Reference: Hurst Figure AI-8, w_n = n * 0.3676 rad/yr
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

# ---------------------------------------------------------------------------
DELTA_W = 0.3676
N_LINES = 34
FS = 52
TWOPI = 2 * np.pi
HURST_FREQS_RAD_YR = np.array([n * DELTA_W for n in range(1, N_LINES + 1)])
HURST_FREQS_RAD_WK = HURST_FREQS_RAD_YR / FS

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))


def load_djia(date_start="1900-01-01", date_end="2025-12-31"):
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/^dji_w.csv"))
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df.Date.between(date_start, date_end)].copy().reset_index(drop=True)
    return df


def build_design_matrix(n_points, freqs_rad_wk, include_trend=True):
    """
    Build design matrix: [1, t, cos(w1*t), sin(w1*t), cos(w2*t), sin(w2*t), ...]
    """
    t = np.arange(n_points, dtype=float)
    n_freqs = len(freqs_rad_wk)
    n_cols = (2 if include_trend else 1) + 2 * n_freqs

    X = np.zeros((n_points, n_cols))
    col = 0
    X[:, col] = 1.0  # DC offset
    col += 1
    if include_trend:
        X[:, col] = t  # linear trend
        col += 1
    for i, w in enumerate(freqs_rad_wk):
        X[:, col] = np.cos(w * t)
        col += 1
        X[:, col] = np.sin(w * t)
        col += 1

    return X


def fit_log_harmonic_model(prices, freqs_rad_wk, include_trend=True):
    """
    Fit log(price) = a + b*t + sum [A_n cos + B_n sin] via least squares.
    Returns coeffs, reconstructed LOG prices, and per-line amplitudes.
    """
    log_prices = np.log(prices)
    X = build_design_matrix(len(prices), freqs_rad_wk, include_trend)

    coeffs, _, _, _ = np.linalg.lstsq(X, log_prices, rcond=None)
    recon_log = X @ coeffs

    n_freqs = len(freqs_rad_wk)
    start_col = 2 if include_trend else 1
    amplitudes = np.zeros(n_freqs)
    phases = np.zeros(n_freqs)
    for i in range(n_freqs):
        a = coeffs[start_col + 2*i]
        b = coeffs[start_col + 2*i + 1]
        amplitudes[i] = np.sqrt(a**2 + b**2)
        phases[i] = np.arctan2(-b, a)

    return coeffs, recon_log, amplitudes, phases


def predict_log_forward(coeffs, freqs_rad_wk, n_fit, n_predict, include_trend=True):
    """Extrapolate log-space model forward."""
    t = np.arange(n_fit, n_fit + n_predict, dtype=float)
    n_freqs = len(freqs_rad_wk)
    n_cols = (2 if include_trend else 1) + 2 * n_freqs

    X = np.zeros((n_predict, n_cols))
    col = 0
    X[:, col] = 1.0
    col += 1
    if include_trend:
        X[:, col] = t
        col += 1
    for i, w in enumerate(freqs_rad_wk):
        X[:, col] = np.cos(w * t)
        col += 1
        X[:, col] = np.sin(w * t)
        col += 1

    return X @ coeffs


def find_turning_points(signal, order=8):
    max_idx = argrelextrema(signal, np.greater, order=order)[0]
    min_idx = argrelextrema(signal, np.less, order=order)[0]
    return max_idx, min_idx


def score_turning_points(actual_tp, predicted_tp, tolerance_weeks=8):
    if len(actual_tp) == 0 or len(predicted_tp) == 0:
        return 0, 1, {"hits": 0, "total_actual": len(actual_tp),
                       "total_predicted": len(predicted_tp), "false_alarms": len(predicted_tp)}
    hits = 0
    matched_predicted = set()
    for a_tp in actual_tp:
        distances = np.abs(predicted_tp - a_tp)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] <= tolerance_weeks:
            hits += 1
            matched_predicted.add(closest_idx)
    hit_rate = hits / len(actual_tp)
    false_alarms = len(predicted_tp) - len(matched_predicted)
    false_alarm_rate = false_alarms / len(predicted_tp) if len(predicted_tp) > 0 else 0
    return hit_rate, false_alarm_rate, {
        "hits": hits, "total_actual": len(actual_tp),
        "total_predicted": len(predicted_tp), "false_alarms": false_alarms}


def evaluate_model(prices_test, predicted_log, dates_test, label):
    """Evaluate a model's predictions. Returns dict of metrics."""
    predicted_prices = np.exp(predicted_log)
    actual_log = np.log(prices_test)

    # Log-space metrics
    log_residual = actual_log - predicted_log
    ss_total_log = np.sum((actual_log - np.mean(actual_log))**2)
    ss_res_log = np.sum(log_residual**2)
    r2_log = 1 - ss_res_log / ss_total_log if ss_total_log > 0 else 0

    # Price-space metrics
    price_residual = prices_test - predicted_prices
    rmse_price = np.sqrt(np.mean(price_residual**2))
    corr = np.corrcoef(prices_test, predicted_prices)[0, 1]
    mape = np.mean(np.abs(price_residual) / prices_test) * 100

    # Direction accuracy
    results = {"label": label, "r2_log": r2_log, "rmse": rmse_price,
               "corr": corr, "mape": mape}

    for horizon_name, h in [("1wk", 1), ("4wk", 4), ("13wk", 13), ("26wk", 26)]:
        actual_dir = np.sign(prices_test[h:] - prices_test[:-h])
        pred_dir = np.sign(predicted_prices[h:] - predicted_prices[:-h])
        n = min(len(actual_dir), len(pred_dir))
        acc = np.sum(actual_dir[:n] == pred_dir[:n]) / n * 100
        results[f"dir_{horizon_name}"] = acc

    # Turning points
    for order, scale in [(4, "2mo"), (8, "4mo"), (13, "6mo")]:
        act_max, act_min = find_turning_points(prices_test, order=order)
        pre_max, pre_min = find_turning_points(predicted_prices, order=order)
        actual_tp = np.sort(np.concatenate([act_max, act_min]))
        pred_tp = np.sort(np.concatenate([pre_max, pre_min]))
        hr, far, d = score_turning_points(actual_tp, pred_tp, tolerance_weeks=8)
        results[f"tp_hit_{scale}"] = hr * 100
        results[f"tp_far_{scale}"] = far * 100
        results[f"tp_detail_{scale}"] = d

    return results, predicted_prices


def print_results(r):
    """Print evaluation results."""
    print(f"\n  {r['label']}:")
    print(f"    Log-space R2:    {r['r2_log']:.4f} ({r['r2_log']*100:.1f}%)")
    print(f"    Price RMSE:      {r['rmse']:.1f}")
    print(f"    Correlation:     {r['corr']:.4f}")
    print(f"    MAPE:            {r['mape']:.1f}%")
    print(f"    Direction:  1wk={r['dir_1wk']:.1f}%  4wk={r['dir_4wk']:.1f}%  13wk={r['dir_13wk']:.1f}%  26wk={r['dir_26wk']:.1f}%")
    print(f"    Turning pts (hit%/false%):  "
          f"2mo={r['tp_hit_2mo']:.0f}/{r['tp_far_2mo']:.0f}  "
          f"4mo={r['tp_hit_4mo']:.0f}/{r['tp_far_4mo']:.0f}  "
          f"6mo={r['tp_hit_6mo']:.0f}/{r['tp_far_6mo']:.0f}")


def main():
    print("=" * 76)
    print("HURST 34-LINE MODEL IN LOG SPACE")
    print("log(price) = a + b*t + sum_{n} [A_n cos(w_n t) + B_n sin(w_n t)]")
    print("=" * 76)

    # -----------------------------------------------------------------------
    # Load data -- use extended range for post-1965 test
    # -----------------------------------------------------------------------
    df_all = load_djia("1921-04-29", "2025-12-31")
    prices_all = df_all.Close.values.astype(float)
    dates_all = pd.to_datetime(df_all.Date.values)

    # Define periods
    hurst_end = pd.Timestamp("1965-05-21")
    split_1955 = pd.Timestamp("1955-01-01")
    modern_start = pd.Timestamp("1965-05-21")
    modern_end = pd.Timestamp("2025-01-01")

    # Indices
    idx_hurst_end = np.searchsorted(dates_all, hurst_end)
    idx_1955 = np.searchsorted(dates_all, split_1955)

    prices_hurst = prices_all[:idx_hurst_end]
    dates_hurst = dates_all[:idx_hurst_end]
    prices_fit = prices_all[:idx_1955]
    prices_test = prices_all[idx_1955:idx_hurst_end]
    dates_test = dates_all[idx_1955:idx_hurst_end]
    n_fit = len(prices_fit)
    n_test = len(prices_test)

    print(f"\nHurst period: {len(prices_hurst)} weeks ({dates_hurst[0].date()} to {dates_hurst[-1].date()})")
    print(f"Fit period:   {n_fit} weeks (to {dates_all[idx_1955-1].date()})")
    print(f"Test period:  {n_test} weeks ({dates_test[0].date()} to {dates_test[-1].date()})")

    # -----------------------------------------------------------------------
    # MODEL A: 34 lines + linear trend in log space
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("MODEL A: 34 lines + linear trend in log space")
    print("=" * 76)

    freqs_34 = HURST_FREQS_RAD_WK

    # In-sample (full Hurst period)
    c_full, recon_full, amps_full, ph_full = fit_log_harmonic_model(prices_hurst, freqs_34)
    recon_price_full = np.exp(recon_full)
    ss_t = np.sum((np.log(prices_hurst) - np.mean(np.log(prices_hurst)))**2)
    ss_r = np.sum((np.log(prices_hurst) - recon_full)**2)
    r2_insample = 1 - ss_r / ss_t
    print(f"\n  In-sample (1921-1965):")
    print(f"    Log-space R2: {r2_insample:.4f} ({r2_insample*100:.1f}%)")
    print(f"    Trend: log(price) = {c_full[0]:.4f} + {c_full[1]:.6f} * t")
    print(f"    Annual growth rate: {np.exp(c_full[1]*52)*100 - 100:.2f}%/yr")

    # Line amplitudes in log space (percentage interpretation)
    print(f"\n  Top 10 lines by log-amplitude (approximate % swing):")
    print(f"    {'N':>3s}  {'w (rad/yr)':>10s}  {'Period':>8s}  {'Log Amp':>8s}  {'~% swing':>9s}")
    sorted_idx = np.argsort(amps_full)[::-1]
    for rank in range(10):
        i = sorted_idx[rank]
        freq = HURST_FREQS_RAD_YR[i]
        period = TWOPI / freq
        pct = (np.exp(amps_full[i]) - 1) * 100  # approximate % swing
        print(f"    {i+1:3d}  {freq:10.4f}  {period:7.2f}yr  {amps_full[i]:8.4f}  {pct:8.1f}%")

    # Out-of-sample: fit on 1921-1955, predict 1955-1965
    c_fit, recon_fit, _, _ = fit_log_harmonic_model(prices_fit, freqs_34)
    pred_log_A = predict_log_forward(c_fit, freqs_34, n_fit, n_test)
    results_A, pred_prices_A = evaluate_model(prices_test, pred_log_A, dates_test, "Model A: 34 lines")
    print_results(results_A)

    # -----------------------------------------------------------------------
    # MODEL B: 33 lines (drop line 1 = 17yr) + linear trend
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("MODEL B: 33 lines (drop 17-year cycle) + linear trend")
    print("=" * 76)

    freqs_33 = HURST_FREQS_RAD_WK[1:]  # drop line 1

    c_fit_B, _, _, _ = fit_log_harmonic_model(prices_fit, freqs_33)
    pred_log_B = predict_log_forward(c_fit_B, freqs_33, n_fit, n_test)
    results_B, pred_prices_B = evaluate_model(prices_test, pred_log_B, dates_test, "Model B: 33 lines (no 17yr)")
    print_results(results_B)

    # -----------------------------------------------------------------------
    # MODEL C: Rolling window (10yr fit, 2yr predict)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("MODEL C: Rolling 10-year window, 2-year prediction")
    print("=" * 76)

    fit_window_wk = 10 * FS
    pred_horizon_wk = 2 * FS
    step_wk = 1 * FS  # step 1 year

    all_actual = []
    all_predicted = []
    all_dates = []
    rolling_results = []

    pos = 0
    while pos + fit_window_wk + pred_horizon_wk <= len(prices_hurst):
        w_prices = prices_hurst[pos:pos + fit_window_wk]
        t_prices = prices_hurst[pos + fit_window_wk:pos + fit_window_wk + pred_horizon_wk]
        t_dates = dates_hurst[pos + fit_window_wk:pos + fit_window_wk + pred_horizon_wk]

        if len(t_prices) < pred_horizon_wk:
            break

        c_w, _, _, _ = fit_log_harmonic_model(w_prices, freqs_34)
        p_log = predict_log_forward(c_w, freqs_34, fit_window_wk, pred_horizon_wk)
        p_prices = np.exp(p_log)

        all_actual.extend(t_prices)
        all_predicted.extend(p_prices)
        all_dates.extend(t_dates)

        # Per-window direction at 13 weeks
        if len(t_prices) > 13:
            act_d = np.sign(t_prices[13:] - t_prices[:-13])
            pre_d = np.sign(p_prices[13:] - p_prices[:-13])
            n_c = min(len(act_d), len(pre_d))
            dir_acc = np.sum(act_d[:n_c] == pre_d[:n_c]) / n_c * 100
            corr_w = np.corrcoef(t_prices, p_prices)[0, 1]
            mape_w = np.mean(np.abs(t_prices - p_prices) / t_prices) * 100
            rolling_results.append({
                "start": dates_hurst[pos].year,
                "pred_start": t_dates[0].year,
                "dir_13wk": dir_acc,
                "corr": corr_w,
                "mape": mape_w,
            })

        pos += step_wk

    print(f"\n  {len(rolling_results)} rolling windows")
    print(f"  {'Window':>12s}  {'Predict':>8s}  {'Dir 13wk':>9s}  {'Corr':>8s}  {'MAPE':>8s}")
    for r in rolling_results:
        print(f"  {r['start']:>12d}  {r['pred_start']:>8d}  {r['dir_13wk']:8.1f}%  {r['corr']:8.3f}  {r['mape']:7.1f}%")

    mean_dir = np.mean([r["dir_13wk"] for r in rolling_results])
    mean_corr = np.mean([r["corr"] for r in rolling_results])
    mean_mape = np.mean([r["mape"] for r in rolling_results])
    print(f"\n  AVERAGES:     dir_13wk={mean_dir:.1f}%  corr={mean_corr:.3f}  MAPE={mean_mape:.1f}%")

    # Aggregate turning point score across all rolling predictions
    all_actual_arr = np.array(all_actual)
    all_predicted_arr = np.array(all_predicted)
    print(f"\n  Aggregate turning points across all rolling windows ({len(all_actual_arr)} pts):")
    for order, scale in [(4, "2mo"), (8, "4mo"), (13, "6mo")]:
        am, an = find_turning_points(all_actual_arr, order=order)
        pm, pn = find_turning_points(all_predicted_arr, order=order)
        atp = np.sort(np.concatenate([am, an]))
        ptp = np.sort(np.concatenate([pm, pn]))
        hr, far, d = score_turning_points(atp, ptp, tolerance_weeks=8)
        print(f"    {scale}: hit={hr*100:.0f}% ({d['hits']}/{d['total_actual']}), "
              f"false_alarm={far*100:.0f}% ({d['false_alarms']}/{d['total_predicted']})")

    # -----------------------------------------------------------------------
    # MODEL D: Post-1965 validation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("MODEL D: POST-1965 VALIDATION")
    print("Does the Hurst structure persist in modern data?")
    print("=" * 76)

    # Fit on full Hurst period, test on post-1965 decades
    test_periods = [
        ("1965-1975", "1965-05-22", "1975-05-21"),
        ("1975-1985", "1975-05-22", "1985-05-21"),
        ("1985-1995", "1985-05-22", "1995-05-21"),
        ("1995-2005", "1995-05-22", "2005-05-21"),
        ("2005-2015", "2005-05-22", "2015-05-21"),
        ("2015-2025", "2015-05-22", "2025-05-21"),
    ]

    # For each decade: fit on prior 34 years, predict next decade
    for label, t_start, t_end in test_periods:
        # Find data range
        t_s = pd.Timestamp(t_start)
        t_e = pd.Timestamp(t_end)
        fit_s = t_s - pd.Timedelta(weeks=34*52)  # 34 years of training data

        mask_fit = (dates_all >= fit_s) & (dates_all < t_s)
        mask_test = (dates_all >= t_s) & (dates_all <= t_e)

        if np.sum(mask_fit) < 520 or np.sum(mask_test) < 52:
            print(f"\n  {label}: insufficient data, skipping")
            continue

        fit_p = prices_all[mask_fit]
        test_p = prices_all[mask_test]
        test_d = dates_all[mask_test]
        n_f = len(fit_p)
        n_t = len(test_p)

        c_d, _, _, _ = fit_log_harmonic_model(fit_p, freqs_34)
        pred_log_d = predict_log_forward(c_d, freqs_34, n_f, n_t)
        r_d, _ = evaluate_model(test_p, pred_log_d, test_d, f"Decade {label}")
        print_results(r_d)

    # -----------------------------------------------------------------------
    # Also: Lanczos spectrum comparison (Hurst period vs modern)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("SPECTRAL CHECK: Line spacing in modern data")
    print("=" * 76)

    from src.spectral import lanczos_spectrum

    for period_label, p_start, p_end in [
        ("Hurst 1921-1965", "1921-04-29", "1965-05-21"),
        ("Modern 1965-2009", "1965-05-22", "2009-05-21"),
        ("Recent 1980-2024", "1980-01-01", "2024-12-31"),
    ]:
        mask = (dates_all >= pd.Timestamp(p_start)) & (dates_all <= pd.Timestamp(p_end))
        p = prices_all[mask]
        if len(p) < 520:
            continue

        # Use log prices for spectrum
        log_p = np.log(p)
        w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(log_p, 1, 52)
        omega_yr = w * 52

        # Find peaks in 2-13 rad/yr range (where our lines live)
        mask_range = (omega_yr >= 2) & (omega_yr <= 13)
        w_range = omega_yr[mask_range]
        a_range = amp[mask_range]

        # Peak detection
        from scipy.signal import find_peaks
        peaks, props = find_peaks(a_range, distance=2, prominence=np.std(a_range)*0.3)
        peak_freqs = w_range[peaks]

        if len(peak_freqs) >= 2:
            spacings = np.diff(np.sort(peak_freqs))
            mean_spacing = np.median(spacings)
            print(f"\n  {period_label} ({len(p)} pts):")
            print(f"    Peaks found (2-13 rad/yr): {len(peak_freqs)}")
            print(f"    Median peak spacing: {mean_spacing:.4f} rad/yr (Hurst: 0.3676)")
            print(f"    Ratio to Hurst: {mean_spacing/0.3676:.2f}")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 20))

    # Panel 1: In-sample log space
    ax1 = fig.add_subplot(5, 1, 1)
    ax1.plot(dates_hurst, np.log(prices_hurst), "k-", linewidth=0.6, label="Actual log(DJIA)", alpha=0.7)
    ax1.plot(dates_hurst, recon_full, "b-", linewidth=0.8,
             label=f"34-line model (R2={r2_insample:.3f})", alpha=0.8)
    ax1.set_title("In-Sample: 34-Line Model in Log Space (1921-1965)", fontweight="bold")
    ax1.set_ylabel("log(DJIA)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: In-sample price space
    ax2 = fig.add_subplot(5, 1, 2)
    ax2.plot(dates_hurst, prices_hurst, "k-", linewidth=0.6, label="Actual DJIA", alpha=0.7)
    ax2.plot(dates_hurst, recon_price_full, "b-", linewidth=0.8, label="34-line model (exp)", alpha=0.8)
    ax2.axvline(split_1955, color="red", linestyle="--", alpha=0.6, label="1955 split")
    ax2.set_title("In-Sample: Model in Price Space", fontweight="bold")
    ax2.set_ylabel("DJIA Close")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Out-of-sample comparison A vs B
    ax3 = fig.add_subplot(5, 1, 3)
    ax3.plot(dates_test, prices_test, "k-", linewidth=0.8, label="Actual DJIA")
    ax3.plot(dates_test, pred_prices_A, "r-", linewidth=0.8,
             label=f"A: 34 lines (corr={results_A['corr']:.3f})", alpha=0.8)
    ax3.plot(dates_test, pred_prices_B, "b--", linewidth=0.8,
             label=f"B: 33 lines (corr={results_B['corr']:.3f})", alpha=0.8)
    ax3.set_title("Out-of-Sample: 1955-1965 (fit on 1921-1955)", fontweight="bold")
    ax3.set_ylabel("DJIA Close")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Rolling window results (clip to sane range to avoid overflow spikes)
    ax4 = fig.add_subplot(5, 1, 4)
    if len(all_actual_arr) > 0 and len(all_predicted_arr) > 0:
        all_dates_arr = np.array(all_dates)
        # Clip predicted to avoid overflow spikes obscuring the plot
        p99 = np.nanpercentile(all_actual_arr, 99.5) * 3
        clipped_pred = np.clip(all_predicted_arr, 0, p99)
        ax4.plot(all_dates_arr, all_actual_arr, "k-", linewidth=0.5, label="Actual", alpha=0.6)
        ax4.plot(all_dates_arr, clipped_pred, "r-", linewidth=0.5, label="Rolling 2yr predictions (clipped)", alpha=0.6)
        ax4.set_ylim(0, p99)
    ax4.set_title("Rolling Window: 10yr Fit, 2yr Predict", fontweight="bold")
    ax4.set_ylabel("DJIA Close")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Line amplitudes in log space
    ax5 = fig.add_subplot(5, 1, 5)
    line_n = np.arange(1, N_LINES + 1)
    ax5.bar(line_n, amps_full * 100, color="steelblue", alpha=0.7, label="Log amplitude x100")
    ax5.set_xlabel("Line number n (w_n = n * 0.3676 rad/yr)")
    ax5.set_ylabel("Log Amplitude x100")
    ax5.set_title("Line Amplitudes in Log Space (~percentage contribution)", fontweight="bold")

    # Mark Hurst nominal cycles
    for n, label in [(1, "18Y"), (2, "9Y"), (4, "4.3Y"), (10, "18M"), (16, "12M"), (23, "9M"), (34, "6M")]:
        if n <= N_LINES:
            ax5.annotate(label, (n, amps_full[n-1]*100), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color="red")
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "backtest_34line_logspace.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)

    print("\n" + "=" * 76)
    print("COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()
