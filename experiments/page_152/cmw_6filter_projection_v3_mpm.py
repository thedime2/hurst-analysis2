# -*- coding: utf-8 -*-
"""
Page 152: 6-Filter CMW Projection v3 -- Matrix Pencil Method
==============================================================

Uses MPM (Hua-Sarkar 1990) to decompose each filter's analytic output
into a sum of complex exponentials:

    z(t) = sum_k  c_k * z_k^t    (discrete-time)

where z_k = exp((alpha_k + j*w_k) / fs).

Three projection strategies compared:
  v1: Static amplitude + median phase rate (baseline from cmw_6filter_projection.py)
  v3a: Pure MPM -- frequency-gated poles projected to unit circle
  v3b: Hybrid -- MPM envelope model + v1 phase carrier (best of both)

Key design choices for v3:
  - Frequency gating: discard poles outside filter passband (prevents DC leakage)
  - Low model order: 2-3 modes per BP (avoids overfitting narrow-band signals)
  - Unit circle projection: |z_k| -> 1 (prevents blowup/decay)
  - End-weighted LS: 3x weight at boundary for continuity
  - Hybrid (v3b): use MPM to model amplitude modulation only

Reference: Hua & Sarkar, "Matrix Pencil Method for Estimating Parameters
of Exponentially Damped/Undamped Sinusoids in Noise", IEEE Trans ASSP, 1990.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.linalg import svd

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

MPM_FIT_CYCLES = 5      # last N cycles of each filter for MPM fit
MPM_MAX_ORDER = 4        # max exponential modes per filter (was 6, now 4)
SV_THRESHOLD = 0.05      # more aggressive pruning (was 0.02)

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


def get_passband(spec):
    """Return (low, high) passband edges in rad/yr, with 50% margin."""
    if spec['type'] == 'lp':
        return (0.0, spec['f_stop'] * 1.5)
    low = spec['f1'] * 0.5
    high = spec['f4'] * 1.5
    return (low, high)


# ============================================================================
# Matrix Pencil Method -- frequency-gated
# ============================================================================

def matrix_pencil(y, L=None, max_order=4, sv_thresh=0.05):
    """
    MPM: decompose y[n] into sum_k c_k * z_k^n.

    Returns poles, amps, order, sv_ratio.
    """
    N = len(y)
    if L is None:
        L = N // 3
    if max_order is None:
        max_order = min(L // 2, 20)

    rows = N - L
    H = np.zeros((rows, L + 1), dtype=complex)
    for col in range(L + 1):
        H[:, col] = y[col:col + rows]

    Y0 = H[:, :-1]
    Y1 = H[:, 1:]

    U, s, Vh = svd(Y0, full_matrices=False)
    sv_ratio = s / s[0]

    # Model order from SV gap
    M = 1
    for i in range(1, min(len(s), max_order)):
        if sv_ratio[i] > sv_thresh:
            M = i + 1
        else:
            break

    U_M = U[:, :M]
    s_M = s[:M]
    Vh_M = Vh[:M, :]

    S_inv = np.diag(1.0 / s_M)
    A = S_inv @ (U_M.conj().T @ Y1 @ Vh_M.conj().T)
    poles = np.linalg.eigvals(A)

    # Solve for amplitudes (end-weighted LS)
    n_idx = np.arange(N)
    Z = np.zeros((N, M), dtype=complex)
    for k in range(M):
        Z[:, k] = poles[k] ** n_idx

    weights = np.linspace(1.0, 3.0, N)
    W = np.diag(np.sqrt(weights))
    amps, _, _, _ = np.linalg.lstsq(W @ Z, W @ y, rcond=None)

    return poles, amps, M, sv_ratio


def frequency_gate_poles(poles, amps, passband_low, passband_high, fs=FS):
    """
    Keep only poles whose frequency falls within [passband_low, passband_high].
    Also discard near-DC poles (|w| < 0.1 rad/yr) for BP filters.
    """
    freqs = np.abs(np.angle(poles) * fs)  # rad/yr

    keep = []
    for i, (p, a, f) in enumerate(zip(poles, amps, freqs)):
        if passband_low <= f <= passband_high and f > 0.1:
            keep.append(i)

    if len(keep) == 0:
        # Fallback: keep the pole with largest amplitude
        keep = [np.argmax(np.abs(amps))]

    return poles[keep], amps[keep], freqs[keep]


def stabilize_poles(poles):
    """Project poles to unit circle: |z| -> 1."""
    return poles / np.abs(poles)


def evaluate_model(poles, amps, n_samples, t_start=0):
    """Evaluate sum_k c_k * z_k^t."""
    t = np.arange(t_start, t_start + n_samples)
    result = np.zeros(n_samples, dtype=complex)
    for k in range(len(poles)):
        result += amps[k] * poles[k] ** t
    return result


# ============================================================================
# MPM on envelope (for hybrid v3b)
# ============================================================================

def mpm_envelope(envelope, period_samples, max_order=3, sv_thresh=0.08):
    """
    Apply MPM to the REAL amplitude envelope to model beat modulation.
    Returns (poles, amps, order) for the envelope model.

    The envelope is real-valued and non-negative, so we model it as
    sum of real sinusoids (cosines).
    """
    n_fit = min(len(envelope), int(8 * period_samples))
    n_fit = max(n_fit, 60)
    env = envelope[-n_fit:].copy()

    # Remove mean (DC) before MPM
    env_mean = np.mean(env)
    env_centered = env - env_mean

    L = max(n_fit // 3, max_order + 2)
    try:
        poles, amps, order, sv = matrix_pencil(
            env_centered.astype(complex), L=L,
            max_order=max_order, sv_thresh=sv_thresh
        )
    except Exception:
        return None

    # Only keep low-frequency poles (envelope modulation is slow)
    # Beat frequencies should be < 2 rad/yr typically
    freqs = np.abs(np.angle(poles) * FS)
    keep = freqs < 5.0  # rad/yr -- very generous for envelope
    if not np.any(keep):
        return None

    poles_k = poles[keep]
    amps_k = amps[keep]

    # Stabilize
    poles_k = stabilize_poles(poles_k)

    return {
        'poles': poles_k,
        'amps': amps_k,
        'mean': env_mean,
        'n_fit': n_fit,
        'order': np.sum(keep),
    }


def project_envelope_mpm(env_model, n_forward, t_start):
    """Project envelope forward using MPM model."""
    osc = evaluate_model(env_model['poles'], env_model['amps'], n_forward, t_start=t_start)
    projected = env_model['mean'] + osc.real
    return np.maximum(projected, 0.0)  # envelope is non-negative


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Page 152: 6-Filter CMW Projection v3 -- Matrix Pencil Method")
    print("  v3a: Frequency-gated MPM (unit-circle poles)")
    print("  v3b: Hybrid (MPM envelope + v1 phase)")
    print("=" * 70)

    # --- Load data ---
    df = getStooq('^dji', 'w')
    df = df.sort_values('Date').reset_index(drop=True)
    dates = pd.to_datetime(df['Date']).values
    close = df['Close'].values.astype(np.float64)

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

    # --- Filter with CMW ---
    print(f"\n--- Applying 6 CMW filters ---")
    filter_outputs = []

    for spec in FILTER_SPECS:
        f0, fwhm = spec_to_cmw(spec)
        analytic = (spec['type'] != 'lp')
        result = apply_cmw(close, f0, fwhm, FS, analytic=analytic)
        result['spec'] = spec
        result['f0'] = f0
        result['fwhm'] = fwhm
        filter_outputs.append(result)

        if analytic:
            print(f"  {spec['label']:20s}  f0={f0:.2f}  FWHM={fwhm:.2f}  T={TWOPI/f0:.2f}yr")
        else:
            print(f"  {spec['label']:20s}  f0={f0:.2f}  FWHM={fwhm:.2f}  (lowpass)")

    # --- Composite reconstruction ---
    composite = np.zeros_like(close)
    for out in filter_outputs:
        sig = out['signal']
        composite += sig.real if np.iscomplexobj(sig) else sig

    disp_prices = close[disp_s:disp_e]
    disp_composite = composite[disp_s:disp_e]
    rms_orig = np.sqrt(np.mean(disp_prices**2))
    rms_err = np.sqrt(np.mean((disp_prices - disp_composite)**2))
    pct_energy = (1 - rms_err / rms_orig) * 100
    print(f"\nComposite energy captured: {pct_energy:.1f}%")

    # ========================================================================
    # PROJECTIONS
    # ========================================================================

    proj_v1 = []    # v1: static A + median dphi
    proj_v3a = []   # v3a: pure MPM (freq-gated, unit circle)
    proj_v3b = []   # v3b: hybrid (MPM envelope + v1 phase)
    mpm_diag = []   # diagnostics for plotting

    for i, out in enumerate(filter_outputs):
        spec = out['spec']
        f0 = out['f0']

        if spec['type'] == 'lp':
            # LP: linear extrapolation (same for all methods)
            sig = out['signal']
            lookback = min(52, disp_e - disp_s)
            segment = np.array(sig[disp_e - lookback:disp_e], dtype=np.float64)
            t_seg = np.arange(lookback)
            coeffs = np.polyfit(t_seg, segment, 1)
            t_fwd = np.arange(lookback, lookback + n_proj)
            lp_proj = np.polyval(coeffs, t_fwd)

            proj_v1.append(lp_proj)
            proj_v3a.append(lp_proj)
            proj_v3b.append(lp_proj)
            mpm_diag.append(None)
            print(f"\n  {spec['label']:20s}  LP linear extrap, slope={coeffs[0]*FS:.2f}/yr")
            continue

        # ---- BP filter ----
        period_samples = TWOPI / f0 * FS
        env = out['envelope']
        phase = out['phase']
        sig_complex = out['signal']

        # --- v1: static amplitude + median phase rate ---
        A_v1 = env[disp_e - 1]
        n_lookback = int(3 * period_samples)
        phase_window = phase[max(disp_s, disp_e - n_lookback):disp_e]
        dphi_v1 = np.median(np.diff(phase_window))
        phi_start = phase[disp_e - 1]
        w_eff_v1 = dphi_v1 * FS

        t_fwd = np.arange(1, n_proj + 1)
        pv1 = A_v1 * np.cos(phi_start + dphi_v1 * t_fwd)
        proj_v1.append(pv1)

        # --- v3a: Pure MPM (freq-gated) ---
        n_fit = int(MPM_FIT_CYCLES * period_samples)
        n_fit = min(n_fit, disp_e)
        n_fit = max(n_fit, 50)

        sig_fit = sig_complex[disp_e - n_fit:disp_e]
        L = max(n_fit // 3, MPM_MAX_ORDER + 2)

        try:
            poles, amps, order_raw, sv_ratio = matrix_pencil(
                sig_fit, L=L, max_order=MPM_MAX_ORDER, sv_thresh=SV_THRESHOLD
            )

            # Frequency gate
            pb_low, pb_high = get_passband(spec)
            poles_g, amps_g, freqs_g = frequency_gate_poles(poles, amps, pb_low, pb_high)

            # Stabilize (unit circle)
            poles_s = stabilize_poles(poles_g)

            # Re-solve amplitudes with gated+stabilized poles (end-weighted)
            n_idx = np.arange(n_fit)
            Z = np.zeros((n_fit, len(poles_s)), dtype=complex)
            for k in range(len(poles_s)):
                Z[:, k] = poles_s[k] ** n_idx
            weights = np.linspace(1.0, 4.0, n_fit)
            W = np.diag(np.sqrt(weights))
            amps_s, _, _, _ = np.linalg.lstsq(W @ Z, W @ sig_fit, rcond=None)

            # Evaluate fit and projection
            fit_v3a = evaluate_model(poles_s, amps_s, n_fit, t_start=0)
            proj_raw = evaluate_model(poles_s, amps_s, n_proj, t_start=n_fit)

            # Anchor: ensure continuity
            gap = sig_fit[-1] - fit_v3a[-1]
            proj_anchored = proj_raw + gap

            pv3a = proj_anchored.real
            proj_v3a.append(pv3a)

            # Boundary fit quality
            n_bnd = max(n_fit // 5, 10)
            bnd_err = np.sqrt(np.mean(np.abs(sig_fit[-n_bnd:] - fit_v3a[-n_bnd:])**2))
            bnd_rms = np.sqrt(np.mean(np.abs(sig_fit[-n_bnd:])**2))
            bnd_pct = (1 - bnd_err / bnd_rms) * 100 if bnd_rms > 0 else 0

            diag = {
                'poles_raw': poles, 'amps_raw': amps, 'order_raw': order_raw,
                'poles_gated': poles_g, 'poles_stable': poles_s, 'amps_stable': amps_s,
                'freqs_gated': freqs_g,
                'sv_ratio': sv_ratio, 'n_fit': n_fit,
                'fit_signal': fit_v3a,
                'boundary_fit_pct': bnd_pct,
                'passband': (pb_low, pb_high),
            }
        except Exception as e:
            print(f"    MPM failed for {spec['label']}: {e}")
            proj_v3a.append(pv1.copy())  # fallback to v1
            diag = None

        # --- v3b: Hybrid (MPM envelope + v1 phase) ---
        env_model = mpm_envelope(env[:disp_e], period_samples)

        if env_model is not None:
            A_proj = project_envelope_mpm(env_model, n_proj, t_start=env_model['n_fit'])
            pv3b = A_proj * np.cos(phi_start + dphi_v1 * t_fwd)
            env_info = f"MPM env order={env_model['order']}"
        else:
            # Fallback: use v1 static amplitude
            pv3b = pv1.copy()
            env_info = "fallback=v1"

        proj_v3b.append(pv3b)

        if diag is not None:
            diag['env_model'] = env_model
        mpm_diag.append(diag)

        # Print summary
        n_gated = len(diag['poles_gated']) if diag else 0
        print(f"\n  {spec['label']}")
        print(f"    v1:  A={A_v1:.1f}  w_eff={w_eff_v1:.3f} rad/yr")
        if diag:
            print(f"    v3a: MPM order {order_raw} -> {n_gated} gated poles  "
                  f"boundary_fit={diag['boundary_fit_pct']:.1f}%")
            print(f"         freqs: {diag['freqs_gated']} rad/yr")
        print(f"    v3b: {env_info} + v1 phase")

    # --- Composite projections ---
    comp_v1 = np.sum(proj_v1, axis=0)
    comp_v3a = np.sum(proj_v3a, axis=0)
    comp_v3b = np.sum(proj_v3b, axis=0)

    actual_proj = close[disp_e:proj_e]

    # --- Metrics ---
    def r2(a, p):
        ss_r = np.sum((a - p)**2)
        ss_t = np.sum((a - np.mean(a))**2)
        return 1 - ss_r / ss_t if ss_t > 0 else 0

    metrics = {}
    for name, comp in [('v1', comp_v1), ('v3a', comp_v3a), ('v3b', comp_v3b)]:
        corr = np.corrcoef(actual_proj, comp)[0, 1]
        r2_val = r2(actual_proj, comp)
        rmse = np.sqrt(np.mean((actual_proj - comp)**2))
        direction = np.mean(np.sign(np.diff(actual_proj)) == np.sign(np.diff(comp)))
        err = (comp[-1] / actual_proj[-1] - 1) * 100
        metrics[name] = {'corr': corr, 'r2': r2_val, 'rmse': rmse,
                         'dir': direction, 'end_err': err, 'end_price': comp[-1]}

    print(f"\n{'='*70}")
    print(f"PROJECTION COMPARISON ({n_proj} weeks)")
    print(f"{'='*70}")
    print(f"  {'Metric':<22}  {'v1 (static)':>12}  {'v3a (MPM)':>12}  {'v3b (hybrid)':>12}")
    print(f"  {'-'*22}  {'-'*12}  {'-'*12}  {'-'*12}")
    for key, label in [('corr', 'Correlation'), ('r2', 'R2'),
                        ('rmse', 'RMSE'), ('dir', 'Direction match'),
                        ('end_price', 'End price (proj)'), ('end_err', 'End price error')]:
        vals = [metrics[n][key] for n in ['v1', 'v3a', 'v3b']]
        if key == 'dir':
            print(f"  {label:22s}  {vals[0]*100:>11.1f}%  {vals[1]*100:>11.1f}%  {vals[2]*100:>11.1f}%")
        elif key == 'end_err':
            print(f"  {label:22s}  {vals[0]:>+11.1f}%  {vals[1]:>+11.1f}%  {vals[2]:>+11.1f}%")
        elif key in ('rmse', 'end_price'):
            print(f"  {label:22s}  {vals[0]:>12.1f}  {vals[1]:>12.1f}  {vals[2]:>12.1f}")
        else:
            print(f"  {label:22s}  {vals[0]:>12.4f}  {vals[1]:>12.4f}  {vals[2]:>12.4f}")
    print(f"  {'End price (actual)':22s}  {actual_proj[-1]:>12.1f}")

    # --- Per-filter comparison ---
    print(f"\n--- Per-filter projection correlation with actual ---")
    for i, (out, pv1, pv3a, pv3b) in enumerate(
            zip(filter_outputs, proj_v1, proj_v3a, proj_v3b)):
        spec = out['spec']
        sig_real = out['signal'].real if np.iscomplexobj(out['signal']) else out['signal']
        if proj_e <= len(sig_real):
            af = sig_real[disp_e:proj_e]
            s1 = np.std(pv1)
            s3a = np.std(pv3a)
            s3b = np.std(pv3b)
            c1 = np.corrcoef(af, pv1)[0, 1] if s1 > 0 else 0
            c3a = np.corrcoef(af, pv3a)[0, 1] if s3a > 0 else 0
            c3b = np.corrcoef(af, pv3b)[0, 1] if s3b > 0 else 0
            print(f"  {spec['label']:20s}  v1={c1:+.3f}  v3a={c3a:+.3f}  v3b={c3b:+.3f}")

    # ========================================================================
    # PLOTTING
    # ========================================================================

    dates_disp = dates[disp_s:disp_e]
    dates_proj = dates[disp_e:proj_e]

    n_filters = len(FILTER_SPECS)
    fig = plt.figure(figsize=(20, 34))
    gs = fig.add_gridspec(n_filters + 1, 3, width_ratios=[3, 1, 1],
                          hspace=0.35, wspace=0.25)

    # === Row 0: Composite ===
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(dates_disp, close[disp_s:disp_e], 'k-', lw=1.2, label='DJIA Close')
    ax0.plot(dates_disp, composite[disp_s:disp_e], 'b-', lw=0.8, alpha=0.4,
             label=f'Composite ({pct_energy:.1f}%)')
    ax0.plot(dates_proj, actual_proj, 'k-', lw=1.2, alpha=0.3, label='Actual (holdout)')
    ax0.plot(dates_proj, comp_v1, 'c-', lw=1.5, alpha=0.7,
             label=f'v1 static (R2={metrics["v1"]["r2"]:.3f})')
    ax0.plot(dates_proj, comp_v3a, 'r-', lw=1.5, alpha=0.7,
             label=f'v3a MPM (R2={metrics["v3a"]["r2"]:.3f})')
    ax0.plot(dates_proj, comp_v3b, 'm-', lw=2, alpha=0.8,
             label=f'v3b hybrid (R2={metrics["v3b"]["r2"]:.3f})')
    ax0.axvline(dates[disp_e], color='grey', ls='--', lw=1, alpha=0.5)
    ax0.set_ylabel('Price', fontsize=9)
    ax0.set_title('v1 (static) vs v3a (MPM) vs v3b (hybrid) -- 100-Week Projection',
                  fontsize=12, fontweight='bold')
    ax0.legend(loc='upper left', fontsize=8)
    ax0.grid(True, alpha=0.2)
    ax0.xaxis.set_major_locator(mdates.YearLocator(2))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # === Rows 1-6: Individual filters ===
    for i, (out, pv1, pv3a, pv3b, diag) in enumerate(
            zip(filter_outputs, proj_v1, proj_v3a, proj_v3b, mpm_diag)):
        spec = out['spec']
        color = spec['color']

        # --- Left panel: signal + projections ---
        ax_sig = fig.add_subplot(gs[i + 1, 0])

        sig_real = out['signal'].real if np.iscomplexobj(out['signal']) else out['signal']
        ax_sig.plot(dates_disp, sig_real[disp_s:disp_e], color=color, lw=0.5, alpha=0.6)

        if out['envelope'] is not None:
            env_disp = out['envelope'][disp_s:disp_e]
            ax_sig.plot(dates_disp, env_disp, color=color, lw=1, alpha=0.5)
            ax_sig.plot(dates_disp, -env_disp, color=color, lw=1, alpha=0.5)

        # MPM fit window
        if diag is not None:
            n_fit = diag['n_fit']
            fit_s = disp_e - n_fit
            if fit_s >= disp_s:
                fit_dates = dates[fit_s:disp_e]
                ax_sig.plot(fit_dates, diag['fit_signal'].real, 'k-', lw=1, alpha=0.6,
                           label=f'MPM fit ({diag["boundary_fit_pct"]:.0f}%)')

        # Actual filter output in projection (faint)
        if proj_e <= len(sig_real):
            ax_sig.plot(dates_proj, sig_real[disp_e:proj_e], color=color,
                       lw=0.5, alpha=0.25, ls='-')

        # Projections
        ax_sig.plot(dates_proj, pv1, 'c-', lw=1, alpha=0.5, label='v1')
        ax_sig.plot(dates_proj, pv3a, 'r-', lw=1, alpha=0.6, label='v3a')
        ax_sig.plot(dates_proj, pv3b, 'm-', lw=1.5, alpha=0.8, label='v3b')

        ax_sig.axvline(dates[disp_e], color='grey', ls='--', lw=0.8, alpha=0.5)
        ax_sig.axhline(0, color='grey', lw=0.3)
        ax_sig.set_ylabel(spec['label'], fontsize=8, rotation=0, labelpad=80, ha='left')
        ax_sig.grid(True, alpha=0.15)
        ax_sig.tick_params(axis='y', labelsize=7)
        ax_sig.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_sig.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        if i == 0:
            ax_sig.legend(fontsize=7, loc='upper right')

        # --- Middle panel: pole-zero plot ---
        ax_pz = fig.add_subplot(gs[i + 1, 1])

        if diag is not None:
            theta = np.linspace(0, 2 * np.pi, 200)
            ax_pz.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.5, alpha=0.3)

            # Raw poles (before gating)
            pr = diag['poles_raw']
            ax_pz.scatter(pr.real, pr.imag, c='grey', s=20, marker='x',
                         linewidths=0.8, alpha=0.4, label='Raw')

            # Stabilized poles (after gating)
            ps = diag['poles_stable']
            ax_pz.scatter(ps.real, ps.imag, c='magenta', s=60, marker='o',
                         facecolors='none', linewidths=2, label='Gated+stable')

            ax_pz.set_xlim(-1.3, 1.3)
            ax_pz.set_ylim(-1.3, 1.3)
            ax_pz.set_aspect('equal')
            ax_pz.axhline(0, color='grey', lw=0.3)
            ax_pz.axvline(0, color='grey', lw=0.3)

            # Annotate passband as arc
            pb_low, pb_high = diag['passband']
            for w_edge in [pb_low, pb_high]:
                ang = w_edge / FS  # rad per sample
                ax_pz.plot([0, np.cos(ang)], [0, np.sin(ang)],
                          'g-', lw=0.5, alpha=0.4)

            n_g = len(diag['poles_gated'])
            ax_pz.set_title(f'Poles: {diag["order_raw"]} raw -> {n_g} gated', fontsize=7)
            if i == 0:
                ax_pz.legend(fontsize=6, loc='lower left')
        else:
            ax_pz.text(0.5, 0.5, 'LP\n(linear)', transform=ax_pz.transAxes,
                      ha='center', va='center', fontsize=8)
        ax_pz.tick_params(axis='both', labelsize=6)

        # --- Right panel: singular values + freq annotation ---
        ax_sv = fig.add_subplot(gs[i + 1, 2])

        if diag is not None:
            sv = diag['sv_ratio']
            n_sv = min(len(sv), 12)
            colors_bar = [color if sv[j] > SV_THRESHOLD else 'lightgrey'
                          for j in range(n_sv)]
            ax_sv.bar(range(n_sv), sv[:n_sv], color=colors_bar, alpha=0.7)
            ax_sv.axhline(SV_THRESHOLD, color='red', ls='--', lw=1, alpha=0.7)
            ax_sv.set_yscale('log')
            ax_sv.set_ylim(1e-4, 2)
            ax_sv.set_title('SV + gated freqs', fontsize=7)

            # Annotate gated frequencies
            fg = diag['freqs_gated']
            as_s = diag['amps_stable']
            lines = []
            for k in range(len(fg)):
                lines.append(f'w={fg[k]:.2f} |c|={np.abs(as_s[k]):.1f}')
            ax_sv.text(0.95, 0.95, '\n'.join(lines), transform=ax_sv.transAxes,
                      fontsize=6, va='top', ha='right', family='monospace',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            ax_sv.text(0.5, 0.5, 'LP', transform=ax_sv.transAxes,
                      ha='center', va='center', fontsize=8)
        ax_sv.tick_params(axis='both', labelsize=6)

    fig.suptitle("Page 152: 6-Filter CMW + Matrix Pencil Method Projection\n"
                 "Left: signal + fit + projection (cyan=v1, red=v3a MPM, magenta=v3b hybrid)\n"
                 "Middle: z-plane (grey=raw, magenta=gated+stabilized) | Right: singular values",
                 fontsize=11, fontweight='bold', y=1.0)

    outpath = os.path.join(os.path.dirname(__file__), 'cmw_6filter_projection_v3_mpm.png')
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
