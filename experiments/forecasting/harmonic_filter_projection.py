# -*- coding: utf-8 -*-
"""
Harmonic Filter Bank Projection
================================

Decomposes DJIA (log prices) into a complete bank of ~444 CMW filters
at Hurst's harmonic spacing w_n = n * 0.3676 rad/yr (n=0..N), with
constant FWHM = 0.3676 rad/yr (adjacent filters overlap at half-max).

Then projects each filter forward using two methods:
  A) Last-cycle amplitude + average phase increment
  B) Hilbert trend extrapolation (linear phase fit + median amplitude)

Out-of-sample test: fit on data up to 2020, project 5 years, compare to actual.

Reference: Hurst, "The Profit Magic of Stock Transaction Timing" (1970)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.loaders import getStooq
from src.time_frequency.cmw import cmw_freq_domain


# ============================================================================
# Constants
# ============================================================================

FS = 52                      # samples per year (weekly)
W_FUND = 0.3676              # fundamental spacing (rad/yr)
NYQUIST = np.pi * FS         # ~163.4 rad/yr
N_HARMONICS = int(NYQUIST / W_FUND)  # ~444
FWHM = W_FUND                # constant bandwidth = fundamental spacing

CUTOFF_DATE = '2020-01-01'
DISPLAY_START = '1935-01-01'
DISPLAY_END = '1954-02-01'
PROJECTION_WEEKS = 260       # ~5 years


# ============================================================================
# Data loading
# ============================================================================

def load_data():
    """Load weekly DJIA, return dates and log(close)."""
    df = getStooq('^dji', 'w')
    df = df.sort_values('Date').reset_index(drop=True)
    dates = pd.to_datetime(df['Date']).values
    log_close = np.log(df['Close'].values)
    return dates, log_close, df


# ============================================================================
# Filter bank via single FFT - NO equalization, uses scale factor instead
# ============================================================================

def apply_harmonic_bank(signal, fs=FS, w_fund=W_FUND, fwhm=FWHM):
    """
    Apply complete harmonic filter bank via FFT.

    The Gaussian tiling has a nearly constant gain of ~2.1x. We compute this
    scale factor and return it so the caller can normalize.

    Returns (results_list, scale_factor)
    """
    L = len(signal)
    nfft = int(2 ** np.ceil(np.log2(L)))
    n_harmonics = int((np.pi * fs) / w_fund)

    # Single FFT of input
    signal_fft = np.fft.fft(signal, n=nfft)

    # Compute total transfer function for scale factor
    total_H = np.zeros(nfft, dtype=np.float64)
    filter_Hs = []

    for n in range(n_harmonics + 1):
        f0 = n * w_fund
        analytic = (n > 0)
        cmw = cmw_freq_domain(f0, fwhm, fs, nfft, analytic=analytic)
        H = cmw['H']
        filter_Hs.append((n, f0, H, analytic))
        total_H += H  # sum of all filter transfer functions

    # Scale factor: the Gaussian tiling gain at signal frequencies
    # Weighted by signal energy at each frequency
    signal_power = np.abs(signal_fft) ** 2
    weighted_gain = np.sum(total_H * signal_power) / np.sum(signal_power)
    scale_factor = weighted_gain

    # Apply each filter
    results = []
    for n, f0, H, analytic in filter_Hs:
        filtered = np.fft.ifft(H * signal_fft)[:L]

        entry = {'f0': f0, 'n': n, 'fwhm': fwhm}

        if analytic:
            entry['signal'] = filtered
            entry['envelope'] = np.abs(filtered)
            entry['phase'] = np.unwrap(np.angle(filtered))
        else:
            entry['signal'] = filtered.real
            entry['envelope'] = None
            entry['phase'] = None

        results.append(entry)

    return results, scale_factor


# ============================================================================
# Reconstruction
# ============================================================================

def reconstruct(bank_results, scale_factor=1.0):
    """Sum real parts of all filter outputs, divided by scale factor."""
    composite = np.zeros(len(bank_results[0]['signal']), dtype=np.float64)
    for r in bank_results:
        if np.iscomplexobj(r['signal']):
            composite += r['signal'].real
        else:
            composite += r['signal']
    return composite / scale_factor


# ============================================================================
# Projection Method A: Last-cycle extrapolation
# ============================================================================

def _damping_envelope(n_forward, period_samples, halflife_cycles=2.0):
    """Exponential decay: amplitude halves every halflife_cycles periods."""
    halflife = halflife_cycles * period_samples
    t = np.arange(1, n_forward + 1)
    return np.exp(-np.log(2) * t / halflife)


def project_method_a(bank_results, n_forward, significant_ns, scale_factor,
                     fs=FS, damping=True):
    """
    For each significant filter: use last 3 cycles to get amplitude and phase rate.
    Project forward as A * cos(phi_now + dphi_avg * t) * damping.
    """
    projections = []

    for r in bank_results:
        f0 = r['f0']
        n = r['n']

        if n == 0:
            # Lowpass: extend last value (constant)
            last_val = r['signal'][-1] if not np.iscomplexobj(r['signal']) else r['signal'][-1].real
            proj = np.full(n_forward, float(last_val))
            projections.append({'n': n, 'f0': f0, 'projection': proj})
            continue

        if n not in significant_ns:
            projections.append({'n': n, 'f0': f0, 'projection': np.zeros(n_forward)})
            continue

        period_samples = 2 * np.pi / f0 * fs

        lookback = int(3 * period_samples)
        L = len(r['signal'])
        start = max(0, L - lookback)

        env = r['envelope'][start:]
        phase = r['phase'][start:]

        if len(env) < 10:
            projections.append({'n': n, 'f0': f0, 'projection': np.zeros(n_forward)})
            continue

        last_cycle_len = min(int(period_samples), len(env))
        A = np.mean(env[-last_cycle_len:])

        dphi = np.diff(phase)
        dphi_avg = np.median(dphi)
        phi_now = phase[-1]

        t_forward = np.arange(1, n_forward + 1)
        proj = A * np.cos(phi_now + dphi_avg * t_forward)

        if damping:
            proj *= _damping_envelope(n_forward, period_samples)

        projections.append({'n': n, 'f0': f0, 'projection': proj,
                           'amplitude': A, 'dphi': dphi_avg})

    return projections


# ============================================================================
# Projection Method B: Hilbert trend extrapolation
# ============================================================================

def project_method_b(bank_results, n_forward, significant_ns, scale_factor,
                     fs=FS, damping=True):
    """
    For each significant filter: fit linear model to phase (captures actual
    frequency), use median amplitude from last 2 cycles, with damping.
    """
    projections = []

    for r in bank_results:
        f0 = r['f0']
        n = r['n']

        if n == 0:
            # Lowpass: linear extrapolation from last 5 years
            sig = r['signal'] if not np.iscomplexobj(r['signal']) else r['signal'].real
            sig = np.asarray(sig, dtype=np.float64)
            lookback = min(260, len(sig))
            segment = sig[-lookback:]
            t_seg = np.arange(lookback)
            coeffs = np.polyfit(t_seg, segment, 1)
            t_forward = np.arange(lookback, lookback + n_forward)
            proj = np.polyval(coeffs, t_forward)
            projections.append({'n': n, 'f0': f0, 'projection': proj,
                               'method': 'linear_extrap',
                               'slope_per_yr': coeffs[0] * fs})
            continue

        if n not in significant_ns:
            projections.append({'n': n, 'f0': f0, 'projection': np.zeros(n_forward)})
            continue

        period_samples = 2 * np.pi / f0 * fs

        lookback = int(5 * period_samples)
        L = len(r['signal'])
        start = max(0, L - lookback)

        env = r['envelope'][start:]
        phase = r['phase'][start:]

        if len(env) < 10:
            projections.append({'n': n, 'f0': f0, 'projection': np.zeros(n_forward)})
            continue

        # Fit linear model to phase
        t_seg = np.arange(len(phase))
        coeffs = np.polyfit(t_seg, phase, 1)
        w_eff = coeffs[0]

        # Median amplitude from last 2 cycles
        last_2 = min(int(2 * period_samples), len(env))
        A = np.median(env[-last_2:])

        phi_now = np.polyval(coeffs, len(phase))

        t_forward = np.arange(1, n_forward + 1)
        proj = A * np.cos(phi_now + w_eff * t_forward)

        if damping:
            proj *= _damping_envelope(n_forward, period_samples, halflife_cycles=3.0)

        projections.append({'n': n, 'f0': f0, 'projection': proj,
                           'amplitude': A, 'w_eff': w_eff * fs,
                           'method': 'hilbert_trend'})

    return projections


# ============================================================================
# Composite projection
# ============================================================================

def sum_projections(projections, scale_factor=1.0, anchor_value=None):
    """
    Sum all individual filter projections, apply scale correction.

    If anchor_value is provided, shift the composite so that its first value
    matches the anchor. This corrects for phase decorrelation at the
    in-sample/out-of-sample boundary.
    """
    composite = np.zeros_like(projections[0]['projection'])
    for p in projections:
        composite += p['projection']
    composite = composite / scale_factor

    if anchor_value is not None:
        offset = anchor_value - composite[0]
        composite += offset

    return composite


# ============================================================================
# Metrics
# ============================================================================

def r_squared(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("HARMONIC FILTER BANK PROJECTION")
    print("w_n = n * 0.3676 rad/yr, FWHM = 0.3676 rad/yr, n=0..%d" % N_HARMONICS)
    print("=" * 70)

    # --- Load data ---
    dates, log_close, df = load_data()
    N = len(log_close)
    print(f"\nData: {N} weekly samples")
    print(f"  Range: {pd.Timestamp(dates[0]).strftime('%Y-%m-%d')} to "
          f"{pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}")

    # --- Find cutoff index ---
    cutoff_dt = np.datetime64(CUTOFF_DATE)
    cutoff_idx = np.searchsorted(dates, cutoff_dt)
    print(f"\nCutoff: {CUTOFF_DATE} (index {cutoff_idx})")
    print(f"  Training: {cutoff_idx} samples")
    print(f"  Holdout:  {N - cutoff_idx} samples")

    n_forward = min(PROJECTION_WEEKS, N - cutoff_idx)
    print(f"  Projection horizon: {n_forward} weeks ({n_forward/52:.1f} years)")

    # --- Apply filter bank to TRAINING data ---
    print(f"\nApplying {N_HARMONICS + 1} harmonic filters...")
    train_signal = log_close[:cutoff_idx]
    bank, scale_factor = apply_harmonic_bank(train_signal)
    print(f"  Done. {len(bank)} filters, scale factor = {scale_factor:.4f}")

    # --- Reconstruction check ---
    composite = reconstruct(bank, scale_factor)
    r2_full = r_squared(train_signal, composite)
    rmse_full = rmse(train_signal, composite)
    print(f"\nReconstruction (full training period):")
    print(f"  R2 = {r2_full:.6f}")
    print(f"  RMSE = {rmse_full:.6f}")

    # Page 152 display window
    disp_start_dt = np.datetime64(DISPLAY_START)
    disp_end_dt = np.datetime64(DISPLAY_END)
    disp_start_idx = np.searchsorted(dates, disp_start_dt)
    disp_end_idx = min(np.searchsorted(dates, disp_end_dt), cutoff_idx)

    r2_disp = None
    if disp_end_idx > disp_start_idx:
        r2_disp = r_squared(train_signal[disp_start_idx:disp_end_idx],
                            composite[disp_start_idx:disp_end_idx])
        print(f"\nReconstruction (page 152 window):")
        print(f"  R2 = {r2_disp:.6f}")

    # --- Identify significant harmonics ---
    print(f"\n--- Harmonic energy analysis ---")
    amplitudes = {}
    for r in bank:
        if r['n'] == 0:
            continue
        amplitudes[r['n']] = np.mean(r['envelope'])

    sorted_amps = sorted(amplitudes.items(), key=lambda x: x[1], reverse=True)

    # 99% energy threshold
    total_bp_energy = sum(a**2 for a in amplitudes.values())
    cumulative = 0
    significant_ns = set()
    for n_h, amp in sorted_amps:
        cumulative += amp**2
        significant_ns.add(n_h)
        if cumulative / total_bp_energy > 0.99:
            break

    print(f"  Total harmonics: {len(amplitudes)}")
    print(f"  Significant (99% energy): {len(significant_ns)}")

    print(f"\n  Top 20 harmonics by amplitude:")
    print(f"  {'n':>4}  {'w (rad/yr)':>10}  {'T (yr)':>8}  {'Amplitude':>10}")
    for n_h, amp in sorted_amps[:20]:
        f0 = n_h * W_FUND
        T = 2 * np.pi / f0
        print(f"  {n_h:>4}  {f0:>10.3f}  {T:>8.2f}  {amp:>10.6f}")

    # Amplitude envelope fit
    freqs_all = np.array([n * W_FUND for n in amplitudes.keys()])
    amps_all = np.array(list(amplitudes.values()))
    mask = (amps_all > 0) & (freqs_all > 0.3)
    log_fit = None
    slope = None
    if np.sum(mask) > 10:
        log_fit = np.polyfit(np.log(freqs_all[mask]), np.log(amps_all[mask]), 1)
        slope = log_fit[0]
        print(f"\n  Amplitude envelope: A ~ w^({slope:.2f})")

    # --- LP filter diagnostics ---
    lp_signal = bank[0]['signal']
    print(f"\n--- LP filter diagnostics ---")
    print(f"  LP last value: {lp_signal[-1]:.4f}")
    print(f"  LP last value / scale: {lp_signal[-1] / scale_factor:.4f}")
    print(f"  Actual log(price) at cutoff: {train_signal[-1]:.4f}")

    # --- Projections ---
    print(f"\n--- Projecting {n_forward} weeks forward ---")
    print(f"  Using {len(significant_ns)} significant harmonics + LP trend")

    # Anchor projection to last reconstruction value for continuity
    anchor = composite[-1]
    print(f"  Anchor value: {anchor:.4f} (reconstruction at cutoff)")

    # Also try with just top 5 filters (per Phase 7: only 5/34 stable)
    top5_ns = set(n for n, _ in sorted_amps[:5])

    # Method A: damped, 27 harmonics
    proj_a = project_method_a(bank, n_forward, significant_ns, scale_factor, damping=True)
    composite_a = sum_projections(proj_a, scale_factor, anchor_value=anchor)

    # Method B: damped, 27 harmonics
    proj_b = project_method_b(bank, n_forward, significant_ns, scale_factor, damping=True)
    composite_b = sum_projections(proj_b, scale_factor, anchor_value=anchor)

    # Method A undamped, top 5 only
    proj_a5 = project_method_a(bank, n_forward, top5_ns, scale_factor, damping=False)
    composite_a5 = sum_projections(proj_a5, scale_factor, anchor_value=anchor)

    # Method B undamped, top 5 only
    proj_b5 = project_method_b(bank, n_forward, top5_ns, scale_factor, damping=False)
    composite_b5 = sum_projections(proj_b5, scale_factor, anchor_value=anchor)

    actual_holdout = log_close[cutoff_idx:cutoff_idx + n_forward]

    print(f"\n  Actual holdout: {actual_holdout[0]:.4f} -> {actual_holdout[-1]:.4f}")

    # Naive baselines
    naive_flat = np.full(n_forward, train_signal[-1])
    lookback = min(260, cutoff_idx)
    t_train = np.arange(lookback)
    coeffs_naive = np.polyfit(t_train, train_signal[-lookback:], 1)
    t_fwd = np.arange(lookback, lookback + n_forward)
    naive_linear = np.polyval(coeffs_naive, t_fwd)

    # Historical growth rate for longer-term trend
    years_of_data = cutoff_idx / FS
    growth_rate = (train_signal[-1] - train_signal[0]) / years_of_data  # log growth/yr
    t_growth = np.arange(1, n_forward + 1) / FS
    naive_growth = train_signal[-1] + growth_rate * t_growth

    # Compute all R2 and RMSE
    results_table = {
        'Method A (27 damped)': composite_a,
        'Method B (27 damped)': composite_b,
        'Method A (top 5 undamped)': composite_a5,
        'Method B (top 5 undamped)': composite_b5,
        'Baseline (flat)': naive_flat,
        'Baseline (5yr linear)': naive_linear,
        'Baseline (hist growth)': naive_growth,
    }

    print(f"\n  {'Method':<30}  {'R2':>8}  {'RMSE':>8}  {'MAE':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name, pred in results_table.items():
        r2_val = r_squared(actual_holdout, pred)
        rmse_val = rmse(actual_holdout, pred)
        mae_val = mae(actual_holdout, pred)
        print(f"  {name:<30}  {r2_val:>8.4f}  {rmse_val:>8.4f}  {mae_val:>8.4f}")

    # Energy ranking for projections
    energies = [(p['n'], p['f0'], np.std(p['projection']))
                for p in proj_b if p['n'] > 0 and np.std(p['projection']) > 0]
    energies.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop 10 projected filter energies:")
    print(f"  {'n':>4}  {'w (rad/yr)':>10}  {'T (yr)':>8}  {'Proj Std':>10}")
    for n_h, f0, e in energies[:10]:
        T = 2 * np.pi / f0
        print(f"  {n_h:>4}  {f0:>10.3f}  {T:>8.2f}  {e:>10.6f}")

    # ========================================================================
    # PLOTTING
    # ========================================================================

    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(5, 2, hspace=0.35)

    # --- 1: Reconstruction (page 152 window) ---
    ax1 = fig.add_subplot(gs[0, 0])
    if disp_end_idx > disp_start_idx and r2_disp is not None:
        t_disp = np.arange(disp_start_idx, disp_end_idx)
        ax1.plot(t_disp, train_signal[disp_start_idx:disp_end_idx],
                 'k-', lw=1, label='log(DJIA)')
        ax1.plot(t_disp, composite[disp_start_idx:disp_end_idx],
                 'r--', lw=1, alpha=0.8, label=f'Composite (R2={r2_disp:.4f})')
        ax1.set_title('Reconstruction: Page 152 Window')
        ax1.legend(fontsize=8)
        ax1.set_ylabel('log(price)')

    # --- 2: Amplitude spectrum ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.loglog(freqs_all, amps_all, 'b.', markersize=3, alpha=0.5)
    sig_freqs = np.array([n * W_FUND for n in significant_ns])
    sig_amps = np.array([amplitudes[n] for n in significant_ns])
    ax2.loglog(sig_freqs, sig_amps, 'ro', markersize=4, alpha=0.7,
               label=f'{len(significant_ns)} significant')
    if log_fit is not None:
        w_fit = np.logspace(np.log10(0.3), np.log10(100), 100)
        a_fit = np.exp(log_fit[1]) * w_fit ** log_fit[0]
        ax2.loglog(w_fit, a_fit, 'g-', lw=1.5, label=f'A ~ w^({slope:.2f})')
    ax2.legend(fontsize=8)
    ax2.set_xlabel('w (rad/yr)')
    ax2.set_ylabel('Mean amplitude')
    ax2.set_title('Harmonic Amplitude Spectrum')
    ax2.grid(True, alpha=0.3)

    # --- 3: Full training reconstruction ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(train_signal, 'k-', lw=0.5, label='log(DJIA)')
    ax3.plot(composite, 'r-', lw=0.5, alpha=0.7, label=f'Composite (R2={r2_full:.4f})')
    ax3.set_title('Full Training Period Reconstruction')
    ax3.legend(fontsize=8)
    ax3.set_ylabel('log(price)')

    # --- 4: Forward projection ---
    ax4 = fig.add_subplot(gs[2, :])
    context_weeks = 260
    context_start = max(0, cutoff_idx - context_weeks)
    t_context = np.arange(context_start, cutoff_idx)
    t_proj = np.arange(cutoff_idx, cutoff_idx + n_forward)

    ax4.plot(t_context, log_close[context_start:cutoff_idx],
             'k-', lw=1.5, label='Actual (training)')
    ax4.plot(t_proj, actual_holdout,
             'k-', lw=1.5, alpha=0.4, label='Actual (holdout)')
    ax4.plot(t_proj, composite_a,
             'b-', lw=1.2, alpha=0.7, label=f'A 27-damped R2={r_squared(actual_holdout, composite_a):.3f}')
    ax4.plot(t_proj, composite_b,
             'r-', lw=1.2, alpha=0.7, label=f'B 27-damped R2={r_squared(actual_holdout, composite_b):.3f}')
    ax4.plot(t_proj, composite_a5,
             'b--', lw=1, alpha=0.5, label=f'A top5 R2={r_squared(actual_holdout, composite_a5):.3f}')
    ax4.plot(t_proj, composite_b5,
             'r--', lw=1, alpha=0.5, label=f'B top5 R2={r_squared(actual_holdout, composite_b5):.3f}')
    ax4.plot(t_proj, naive_growth,
             'g--', lw=1, alpha=0.6, label=f'Hist growth R2={r_squared(actual_holdout, naive_growth):.3f}')
    ax4.axvline(cutoff_idx, color='grey', ls='--', alpha=0.5)

    year_ticks = np.arange(2015, 2026)
    year_indices = [np.searchsorted(dates, np.datetime64(f'{y}-01-01')) for y in year_ticks]
    ax4.set_xticks(year_indices)
    ax4.set_xticklabels([str(y) for y in year_ticks], fontsize=8)
    ax4.set_title(f'5-Year Forward Projection ({len(significant_ns)} harmonics + trend)')
    ax4.legend(fontsize=7, loc='upper left')
    ax4.set_ylabel('log(price)')

    # --- 5: Top 10 filter projections (Method B) ---
    ax5 = fig.add_subplot(gs[3, :])
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (n_h, f0, _) in enumerate(energies[:min(10, len(energies))]):
        p = next(p for p in proj_b if p['n'] == n_h)
        T = 2 * np.pi / f0
        ax5.plot(t_proj, p['projection'] / scale_factor,
                 color=colors[i], lw=0.8, alpha=0.7,
                 label=f'n={n_h} T={T:.1f}yr')
    ax5.set_title('Top 10 Filter Projections (Method B, scaled)')
    ax5.legend(fontsize=7, ncol=2, loc='upper left')
    ax5.set_ylabel('Amplitude')
    ax5.set_xticks(year_indices)
    ax5.set_xticklabels([str(y) for y in year_ticks], fontsize=8)

    # --- 6: Reconstruction error ---
    ax6 = fig.add_subplot(gs[4, :])
    error = train_signal - composite
    ax6.plot(error, 'b-', lw=0.3, alpha=0.7)
    ax6.set_title(f'Reconstruction Error (mean={np.mean(error):.4f}, std={np.std(error):.4f})')
    ax6.set_ylabel('Error')
    ax6.axhline(0, color='k', lw=0.5)

    plt.tight_layout()
    outpath = 'experiments/forecasting/harmonic_filter_projection.png'
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
