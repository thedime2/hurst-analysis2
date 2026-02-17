# -*- coding: utf-8 -*-
"""
Page 152: 6-Filter CMW Projection v2
======================================

Enhancements over v1:
  1. LOG SPACE: filter log(close), project in log, exp() back
     (cycles are multiplicative - R2 jumps from 16% to 78% per Phase 7)
  2. BEAT ENVELOPE PROJECTION: fit sinusoid to the amplitude envelope
     over the last few beat periods, project A(t) instead of static A
  3. LINEAR FREQUENCY TREND: fit dphi = dphi_0 + slope*t to capture
     systematic frequency drift (e.g. BP-2 running at 6.4yr vs nominal 3.8yr)

Same 6 filters (1 LP + 5 BP) from Hurst's page 152, applied as CMW
over the full weekly DJIA series. Display window 1935-1954, 100-week
projection beyond.

Reference: Hurst, "The Profit Magic of Stock Transaction Timing" (1970), p.152
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.loaders import getStooq
from src.time_frequency.cmw import apply_cmw


# ============================================================================
# Configuration
# ============================================================================

FS = 52
TWOPI = 2 * np.pi

DISPLAY_START = '1935-01-01'
DISPLAY_END = '1954-02-01'
PROJECTION_WEEKS = 100

FILTER_SPECS = [
    {'type': 'lp', 'f_pass': 0.85, 'f_stop': 1.25,
     'label': 'LP-1: Trend (>5 yr)', 'color': '#1f77b4'},
    {'type': 'bp', 'f1': 0.85, 'f2': 1.25, 'f3': 2.05, 'f4': 2.45,
     'label': 'BP-2: ~3.8 yr', 'color': '#ff7f0e'},
    {'type': 'bp', 'f1': 3.20, 'f2': 3.55, 'f3': 6.35, 'f4': 6.70,
     'label': 'BP-3: ~1.3 yr', 'color': '#2ca02c'},
    {'type': 'bp', 'f1': 7.25, 'f2': 7.55, 'f3': 9.55, 'f4': 9.85,
     'label': 'BP-4: ~0.7 yr', 'color': '#d62728'},
    {'type': 'bp', 'f1': 13.65, 'f2': 13.95, 'f3': 19.35, 'f4': 19.65,
     'label': 'BP-5: ~0.4 yr', 'color': '#9467bd'},
    {'type': 'bp', 'f1': 28.45, 'f2': 28.75, 'f3': 35.95, 'f4': 36.25,
     'label': 'BP-6: ~0.2 yr', 'color': '#8c564b'},
]


def spec_to_cmw(spec):
    if spec['type'] == 'lp':
        return 0.0, spec['f_pass'] + spec['f_stop']
    f0 = (spec['f2'] + spec['f3']) / 2.0
    lower = (spec['f1'] + spec['f2']) / 2.0
    upper = (spec['f3'] + spec['f4']) / 2.0
    return f0, upper - lower


# ============================================================================
# Beat envelope fitting
# ============================================================================

def fit_beat_envelope(envelope, period_samples, fs=FS):
    """
    Fit a sinusoidal model to the amplitude envelope:
      A(t) = A0 + A1 * cos(w_beat * t + phi_beat)

    The beat frequency is estimated from the envelope's dominant oscillation.
    Returns (A0, A1, w_beat, phi_beat) or None if fit fails.
    """
    # Use 5-10 nominal periods of envelope history
    n_samples = min(len(envelope), int(10 * period_samples))
    env = envelope[-n_samples:]
    t = np.arange(n_samples)

    A0 = np.mean(env)
    if A0 < 1e-10:
        return None

    # Estimate beat frequency from envelope spectrum
    env_centered = env - A0
    nfft = max(256, int(2 ** np.ceil(np.log2(len(env_centered)))))
    spectrum = np.abs(np.fft.rfft(env_centered, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs) * TWOPI  # rad/yr

    # Skip DC (idx 0) and very low freqs, find peak
    min_idx = max(1, int(0.1 / (freqs[1] - freqs[0])) if len(freqs) > 1 else 1)
    max_idx = len(spectrum) // 2  # don't look past half Nyquist
    if min_idx >= max_idx:
        return None

    peak_idx = min_idx + np.argmax(spectrum[min_idx:max_idx])
    w_beat_est = freqs[peak_idx]

    if w_beat_est < 0.05:  # too slow to fit
        return None

    # Refine with curve_fit
    A1_est = np.std(env) * np.sqrt(2)

    def model(t_arr, a0, a1, wb, phib):
        return a0 + a1 * np.cos(wb / fs * t_arr + phib)

    try:
        popt, _ = curve_fit(model, t, env,
                            p0=[A0, A1_est, w_beat_est, 0.0],
                            bounds=([0, 0, w_beat_est * 0.5, -np.pi],
                                    [A0 * 3, A0 * 3, w_beat_est * 2.0, np.pi]),
                            maxfev=2000)
        return popt  # (A0, A1, w_beat, phi_beat)
    except (RuntimeError, ValueError):
        return None


def project_beat_envelope(beat_params, n_forward, last_env_idx, fs=FS):
    """Project amplitude envelope forward using beat model."""
    A0, A1, w_beat, phi_beat = beat_params
    t_fwd = np.arange(last_env_idx + 1, last_env_idx + 1 + n_forward)
    A_proj = A0 + A1 * np.cos(w_beat / fs * t_fwd + phi_beat)
    # Clip to non-negative
    return np.maximum(A_proj, 0.0)


# ============================================================================
# Linear frequency trend
# ============================================================================

def fit_freq_trend(phase, period_samples):
    """
    Fit dphi(t) = dphi_0 + slope * t to the phase derivative.
    Returns (dphi_0, slope) at the END of the window, plus
    the extrapolation function.
    """
    n_samples = min(len(phase), int(5 * period_samples))
    ph = phase[-n_samples:]

    dphi = np.diff(ph)
    t = np.arange(len(dphi))

    # Robust: use polyfit degree 1
    coeffs = np.polyfit(t, dphi, 1)
    slope = coeffs[0]
    dphi_end = np.polyval(coeffs, len(dphi))

    return dphi_end, slope


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Page 152: 6-Filter CMW Projection v2")
    print("  [1] Log space  [2] Beat envelope  [3] Freq trend")
    print("=" * 70)

    # --- Load data ---
    df = getStooq('^dji', 'w')
    df = df.sort_values('Date').reset_index(drop=True)
    dates = pd.to_datetime(df['Date']).values
    close = df['Close'].values.astype(np.float64)
    log_close = np.log(close)

    print(f"\nData: {len(close)} weekly samples")
    print(f"  {pd.Timestamp(dates[0]).strftime('%Y-%m-%d')} to "
          f"{pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}")

    # --- Display window ---
    disp_s = np.searchsorted(dates, np.datetime64(DISPLAY_START))
    disp_e = np.searchsorted(dates, np.datetime64(DISPLAY_END))
    proj_e = min(disp_e + PROJECTION_WEEKS, len(close))
    n_proj = proj_e - disp_e

    print(f"\nDisplay: {DISPLAY_START} to {DISPLAY_END} ({disp_e - disp_s} weeks)")
    print(f"Projection: {n_proj} weeks to "
          f"{pd.Timestamp(dates[proj_e-1]).strftime('%Y-%m-%d')}")

    # --- Filter LOG(CLOSE) with 6 CMW filters ---
    print(f"\n--- Filtering log(close) with 6 CMW filters ---")
    filter_outputs = []

    for spec in FILTER_SPECS:
        f0, fwhm = spec_to_cmw(spec)
        analytic = (spec['type'] != 'lp')
        result = apply_cmw(log_close, f0, fwhm, FS, analytic=analytic)
        result['spec'] = spec
        result['f0'] = f0
        result['fwhm'] = fwhm
        filter_outputs.append(result)

        if analytic:
            print(f"  {spec['label']:20s}  f0={f0:.2f}  FWHM={fwhm:.2f}  T={TWOPI/f0:.2f}yr")
        else:
            print(f"  {spec['label']:20s}  f0={f0:.2f}  FWHM={fwhm:.2f}  (lowpass)")

    # --- Composite reconstruction in log space ---
    log_composite = np.zeros_like(log_close)
    for out in filter_outputs:
        sig = out['signal']
        log_composite += sig.real if np.iscomplexobj(sig) else sig

    # R2 in display window
    ss_res = np.sum((log_close[disp_s:disp_e] - log_composite[disp_s:disp_e])**2)
    ss_tot = np.sum((log_close[disp_s:disp_e] - np.mean(log_close[disp_s:disp_e]))**2)
    r2_recon = 1 - ss_res / ss_tot
    print(f"\nReconstruction R2 (log, display window): {r2_recon:.6f}")

    # --- Compute v1 and v2 projections ---
    print(f"\n--- Computing projections ---")

    proj_v1 = []   # v1: static A, median dphi
    proj_v2 = []   # v2: beat envelope + freq trend
    beat_info = []  # for display

    for i, out in enumerate(filter_outputs):
        spec = out['spec']
        f0 = out['f0']

        if spec['type'] == 'lp':
            # LP: linear extrapolation in log space (captures growth rate)
            sig = out['signal']
            lookback = min(104, disp_e - disp_s)  # 2 years
            segment = np.array(sig[disp_e - lookback:disp_e], dtype=np.float64)
            t_seg = np.arange(lookback)
            coeffs = np.polyfit(t_seg, segment, 1)
            t_fwd = np.arange(lookback, lookback + n_proj)
            lp_proj = np.polyval(coeffs, t_fwd)

            growth_yr = coeffs[0] * FS
            print(f"  {spec['label']:20s}  log growth: {growth_yr:.4f}/yr "
                  f"({np.exp(growth_yr)-1:.1%} annualized)")

            proj_v1.append(lp_proj)
            proj_v2.append(lp_proj)  # same for both
            beat_info.append(None)
            continue

        period_samples = TWOPI / f0 * FS
        env = out['envelope']
        phase = out['phase']

        # === v1: static amplitude + median dphi ===
        A_v1 = env[disp_e - 1]
        n_look = int(3 * period_samples)
        ph_window = phase[max(disp_s, disp_e - n_look):disp_e]
        dphi_v1 = np.median(np.diff(ph_window))
        phi_start = phase[disp_e - 1]

        t_fwd = np.arange(1, n_proj + 1)
        p_v1 = A_v1 * np.cos(phi_start + dphi_v1 * t_fwd)
        proj_v1.append(p_v1)

        # === v2: beat envelope + freq trend ===
        # Beat envelope
        env_history = env[:disp_e]
        beat_params = fit_beat_envelope(env_history, period_samples)

        if beat_params is not None:
            A0, A1, w_beat, phi_beat = beat_params
            A_proj = project_beat_envelope(beat_params, n_proj, disp_e - 1)
            T_beat = TWOPI / w_beat if w_beat > 0.01 else float('inf')
            beat_str = f"A0={A0:.3f} A1={A1:.3f} T_beat={T_beat:.1f}yr"
        else:
            # Fallback: use last-cycle mean amplitude
            last_cycle = int(period_samples)
            A_proj = np.full(n_proj, np.mean(env[disp_e - last_cycle:disp_e]))
            beat_str = "no beat fit (fallback mean)"

        beat_info.append(beat_params)

        # Frequency trend
        dphi_end, freq_slope = fit_freq_trend(phase[:disp_e], period_samples)

        # Project phase with linear chirp: phi(t) = phi_start + dphi_end*t + 0.5*slope*t^2
        phi_proj = phi_start + dphi_end * t_fwd + 0.5 * freq_slope * t_fwd**2

        p_v2 = A_proj * np.cos(phi_proj)
        proj_v2.append(p_v2)

        w_eff_v1 = dphi_v1 * FS
        w_eff_v2 = dphi_end * FS
        drift_rate = freq_slope * FS**2  # rad/yr^2
        print(f"  {spec['label']:20s}  A_v1={A_v1:.4f}  w_v1={w_eff_v1:.3f}  "
              f"w_v2={w_eff_v2:.3f}  drift={drift_rate:.4f} rad/yr^2  {beat_str}")

    # --- Sum composites ---
    composite_v1 = np.sum(proj_v1, axis=0)
    composite_v2 = np.sum(proj_v2, axis=0)

    # Convert back from log space
    actual_proj_log = log_close[disp_e:proj_e]
    actual_proj_price = close[disp_e:proj_e]

    price_v1 = np.exp(composite_v1)
    price_v2 = np.exp(composite_v2)

    # Also reconstruct in-sample composite as price
    composite_price = np.exp(log_composite)

    # --- Metrics ---
    def r2(a, p):
        ss_r = np.sum((a - p)**2)
        ss_t = np.sum((a - np.mean(a))**2)
        return 1 - ss_r / ss_t if ss_t > 0 else 0

    corr_v1 = np.corrcoef(actual_proj_price, price_v1)[0, 1]
    corr_v2 = np.corrcoef(actual_proj_price, price_v2)[0, 1]
    r2_v1_log = r2(actual_proj_log, composite_v1)
    r2_v2_log = r2(actual_proj_log, composite_v2)
    r2_v1_price = r2(actual_proj_price, price_v1)
    r2_v2_price = r2(actual_proj_price, price_v2)

    dir_v1 = np.mean(np.sign(np.diff(actual_proj_price)) == np.sign(np.diff(price_v1)))
    dir_v2 = np.mean(np.sign(np.diff(actual_proj_price)) == np.sign(np.diff(price_v2)))

    print(f"\n{'='*70}")
    print(f"PROJECTION COMPARISON ({n_proj} weeks)")
    print(f"{'='*70}")
    print(f"  {'Metric':<25}  {'v1 (static)':>12}  {'v2 (beat+drift)':>15}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*15}")
    print(f"  {'Correlation':25s}  {corr_v1:>12.4f}  {corr_v2:>15.4f}")
    print(f"  {'R2 (log space)':25s}  {r2_v1_log:>12.4f}  {r2_v2_log:>15.4f}")
    print(f"  {'R2 (price space)':25s}  {r2_v1_price:>12.4f}  {r2_v2_price:>15.4f}")
    print(f"  {'Direction match':25s}  {dir_v1*100:>11.1f}%  {dir_v2*100:>14.1f}%")
    print(f"  {'End price (actual)':25s}  {actual_proj_price[-1]:>12.1f}  {actual_proj_price[-1]:>15.1f}")
    print(f"  {'End price (projected)':25s}  {price_v1[-1]:>12.1f}  {price_v2[-1]:>15.1f}")
    err_v1 = (price_v1[-1] / actual_proj_price[-1] - 1) * 100
    err_v2 = (price_v2[-1] / actual_proj_price[-1] - 1) * 100
    print(f"  {'End price error':25s}  {err_v1:>+11.1f}%  {err_v2:>+14.1f}%")

    # ========================================================================
    # PLOTTING
    # ========================================================================

    dates_disp = dates[disp_s:disp_e]
    dates_proj = dates[disp_e:proj_e]

    n_filters = len(FILTER_SPECS)
    fig = plt.figure(figsize=(18, 30))
    gs = fig.add_gridspec(n_filters + 2, 2, width_ratios=[3, 1],
                          hspace=0.3, wspace=0.15)

    # === Row 0: Price composite + projections ===
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(dates_disp, close[disp_s:disp_e], 'k-', lw=1.2, label='DJIA Close')
    ax0.plot(dates_disp, composite_price[disp_s:disp_e], 'b-', lw=1, alpha=0.6,
             label=f'Composite (R2={r2_recon:.4f})')
    ax0.plot(dates_proj, actual_proj_price, 'k-', lw=1.2, alpha=0.3,
             label='Actual (holdout)')
    ax0.plot(dates_proj, price_v1, 'c-', lw=1.5, alpha=0.7,
             label=f'v1 static (corr={corr_v1:.3f})')
    ax0.plot(dates_proj, price_v2, 'm-', lw=1.5, alpha=0.8,
             label=f'v2 beat+drift (corr={corr_v2:.3f})')
    ax0.axvline(dates[disp_e], color='grey', ls='--', lw=1, alpha=0.5)
    ax0.set_ylabel('Price', fontsize=9)
    ax0.set_title('v1 vs v2: Price Projection (exp of log-space composite)',
                  fontsize=11, fontweight='bold')
    ax0.legend(loc='upper left', fontsize=8)
    ax0.grid(True, alpha=0.2)
    ax0.xaxis.set_major_locator(mdates.YearLocator(2))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # === Row 1: Log-space composite ===
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(dates_disp, log_close[disp_s:disp_e], 'k-', lw=1.2, label='log(DJIA)')
    ax1.plot(dates_disp, log_composite[disp_s:disp_e], 'b-', lw=1, alpha=0.6,
             label='Composite')
    ax1.plot(dates_proj, actual_proj_log, 'k-', lw=1.2, alpha=0.3, label='Actual')
    ax1.plot(dates_proj, composite_v1, 'c-', lw=1.2, alpha=0.7,
             label=f'v1 R2={r2_v1_log:.3f}')
    ax1.plot(dates_proj, composite_v2, 'm-', lw=1.2, alpha=0.8,
             label=f'v2 R2={r2_v2_log:.3f}')
    ax1.axvline(dates[disp_e], color='grey', ls='--', lw=1, alpha=0.5)
    ax1.set_ylabel('log(price)', fontsize=9)
    ax1.set_title('Log-Space Projection', fontsize=10)
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # === Rows 2-7: Individual filters ===
    for i, (out, pv1, pv2, bi) in enumerate(zip(filter_outputs, proj_v1, proj_v2, beat_info)):
        spec = out['spec']
        color = spec['color']

        # --- Left: signal + envelope + projections ---
        ax_sig = fig.add_subplot(gs[i + 2, 0])

        sig_real = out['signal'].real if np.iscomplexobj(out['signal']) else out['signal']
        ax_sig.plot(dates_disp, sig_real[disp_s:disp_e], color=color, lw=0.5, alpha=0.6)

        if out['envelope'] is not None:
            env_disp = out['envelope'][disp_s:disp_e]
            ax_sig.plot(dates_disp, env_disp, color=color, lw=1.2, alpha=0.7)
            ax_sig.plot(dates_disp, -env_disp, color=color, lw=1.2, alpha=0.7)

            # v2 projected envelope
            if bi is not None:
                A_proj = project_beat_envelope(bi, n_proj, disp_e - 1)
                ax_sig.plot(dates_proj, A_proj, 'm--', lw=1, alpha=0.6)
                ax_sig.plot(dates_proj, -A_proj, 'm--', lw=1, alpha=0.6)

        # v1 projection
        ax_sig.plot(dates_proj, pv1, 'c-', lw=1, alpha=0.6, label='v1')
        # v2 projection
        ax_sig.plot(dates_proj, pv2, 'm-', lw=1.2, alpha=0.8, label='v2')

        ax_sig.axvline(dates[disp_e], color='grey', ls='--', lw=0.8, alpha=0.5)
        ax_sig.axhline(0, color='grey', lw=0.3)
        ax_sig.set_ylabel(spec['label'], fontsize=8, rotation=0, labelpad=80, ha='left')
        ax_sig.grid(True, alpha=0.15)
        ax_sig.tick_params(axis='y', labelsize=7)
        ax_sig.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_sig.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        if i == 0:
            ax_sig.legend(fontsize=7, loc='upper right')

        # --- Right: phase/freq or envelope detail ---
        ax_pf = fig.add_subplot(gs[i + 2, 1])

        if out['phase'] is not None:
            # Wrapped phase
            phasew = np.angle(out['signal'][disp_s:disp_e])
            ax_pf.plot(dates_disp, phasew, color=color, lw=0.3, alpha=0.5)
            ax_pf.set_ylim(-np.pi - 0.3, np.pi + 0.3)
            ax_pf.axhline(0, color='grey', lw=0.3)

            # Inst freq on twin
            ax_freq = ax_pf.twinx()
            freq_rad = out['frequency'][disp_s:disp_e] * TWOPI
            ax_freq.plot(dates_disp, freq_rad, color='darkred', lw=0.4, alpha=0.5)
            f0 = out['f0']
            ax_freq.axhline(f0, color='darkred', lw=0.5, ls='--', alpha=0.5)
            ax_freq.set_ylabel('w (rad/yr)', fontsize=7, color='darkred')
            ax_freq.tick_params(axis='y', labelsize=6, colors='darkred')
            if f0 > 0:
                ax_freq.set_ylim(max(0, f0 * 0.5), f0 * 1.5)

            # Show beat period if fitted
            if bi is not None:
                T_beat = TWOPI / bi[2] if bi[2] > 0.01 else float('inf')
                ax_pf.set_title(f'T_beat={T_beat:.1f}yr', fontsize=7)
        else:
            sig_lp = out['signal'][disp_s:disp_e]
            ax_pf.plot(dates_disp, sig_lp, color=color, lw=1)
            ax_pf.set_ylabel('LP', fontsize=7)

        ax_pf.grid(True, alpha=0.15)
        ax_pf.tick_params(axis='both', labelsize=6)
        ax_pf.xaxis.set_major_locator(mdates.YearLocator(4))
        ax_pf.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.suptitle("Page 152: 6-Filter CMW Projection v2\n"
                 "Log space | Beat envelope (magenta dashed) | Freq drift\n"
                 "v1=cyan (static A, median dphi) | v2=magenta (beat A, chirp phi)",
                 fontsize=12, fontweight='bold', y=1.0)

    outpath = os.path.join(os.path.dirname(__file__), 'cmw_6filter_projection_v2.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {outpath}")

    try:
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    except Exception:
        plt.close()


if __name__ == '__main__':
    main()
