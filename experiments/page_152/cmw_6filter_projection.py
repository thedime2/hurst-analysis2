# -*- coding: utf-8 -*-
"""
Page 152: 6-Filter CMW Decomposition + 100-Week Projection
============================================================

Uses the same 6 filters (1 LP + 5 BP) from Hurst's page 152 decomposition,
applied as Complex Morlet Wavelets over the FULL weekly DJIA series.

Displays the page 152 window (1935-1954) with filter outputs, envelopes,
phase, and frequency. Projects each filter 100 weeks beyond the display
window using:
  - Amplitude: current envelope value at projection start
  - Phase: last phase value + average phase increment per sample

Reference: Hurst, "The Profit Magic of Stock Transaction Timing" (1970), p.152
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.loaders import getStooq
from src.time_frequency.cmw import cmw_freq_domain, apply_cmw


# ============================================================================
# Configuration
# ============================================================================

FS = 52
TWOPI = 2 * np.pi

DISPLAY_START = '1935-01-01'
DISPLAY_END = '1954-02-01'
PROJECTION_WEEKS = 100

# Page 152 filter specs (rad/year) - matched to CMW via ormsby_spec_to_cmw_params
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

# Convert to CMW parameters (same logic as ormsby_spec_to_cmw_params)
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def spec_to_cmw(spec):
    """Convert filter spec to CMW f0/fwhm."""
    if spec['type'] == 'lp':
        f0 = 0.0
        fwhm = spec['f_pass'] + spec['f_stop']
    else:
        f0 = (spec['f2'] + spec['f3']) / 2.0
        lower = (spec['f1'] + spec['f2']) / 2.0
        upper = (spec['f3'] + spec['f4']) / 2.0
        fwhm = upper - lower
    return f0, fwhm


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Page 152: 6-Filter CMW Decomposition + 100-Week Projection")
    print("=" * 70)

    # --- Load full weekly DJIA ---
    df = getStooq('^dji', 'w')
    df = df.sort_values('Date').reset_index(drop=True)
    dates = pd.to_datetime(df['Date']).values
    close = df['Close'].values.astype(np.float64)
    print(f"\nData: {len(close)} weekly samples")
    print(f"  {pd.Timestamp(dates[0]).strftime('%Y-%m-%d')} to "
          f"{pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}")

    # --- Display window indices ---
    disp_s = np.searchsorted(dates, np.datetime64(DISPLAY_START))
    disp_e = np.searchsorted(dates, np.datetime64(DISPLAY_END))
    proj_e = disp_e + PROJECTION_WEEKS

    # Ensure we don't exceed data
    proj_e = min(proj_e, len(close))
    actual_proj_weeks = proj_e - disp_e

    print(f"\nDisplay window: {DISPLAY_START} to {DISPLAY_END}")
    print(f"  Indices: {disp_s} to {disp_e} ({disp_e - disp_s} samples)")
    print(f"  Projection: {actual_proj_weeks} weeks beyond display end")
    proj_start_date = pd.Timestamp(dates[disp_e]).strftime('%Y-%m-%d')
    proj_end_date = pd.Timestamp(dates[min(proj_e - 1, len(dates) - 1)]).strftime('%Y-%m-%d')
    print(f"  Projection dates: {proj_start_date} to {proj_end_date}")

    # --- Apply 6 CMW filters to FULL series ---
    print(f"\nApplying 6 CMW filters to full series (reflection mode via FFT)...")
    filter_outputs = []

    for i, spec in enumerate(FILTER_SPECS):
        f0, fwhm = spec_to_cmw(spec)
        analytic = (spec['type'] != 'lp')

        result = apply_cmw(close, f0, fwhm, FS, analytic=analytic)
        result['spec'] = spec
        result['f0'] = f0
        result['fwhm'] = fwhm

        if analytic:
            T_yr = TWOPI / f0
            print(f"  {spec['label']:20s}  f0={f0:.2f} FWHM={fwhm:.2f} rad/yr  T={T_yr:.2f}yr")
        else:
            print(f"  {spec['label']:20s}  f0={f0:.2f} FWHM={fwhm:.2f} rad/yr  (lowpass)")

        filter_outputs.append(result)

    # --- Composite reconstruction ---
    composite = np.zeros_like(close)
    for out in filter_outputs:
        sig = out['signal']
        composite += sig.real if np.iscomplexobj(sig) else sig

    # Energy in display window
    disp_prices = close[disp_s:disp_e]
    disp_composite = composite[disp_s:disp_e]
    rms_orig = np.sqrt(np.mean(disp_prices**2))
    rms_err = np.sqrt(np.mean((disp_prices - disp_composite)**2))
    pct_energy = (1 - rms_err / rms_orig) * 100
    print(f"\nComposite energy captured: {pct_energy:.1f}%")

    # --- Compute projections for each filter ---
    print(f"\nComputing 100-week projections from display window end...")

    projections = []
    for i, out in enumerate(filter_outputs):
        spec = out['spec']
        f0 = out['f0']

        if spec['type'] == 'lp':
            # LP: linear extrapolation from last 52 weeks of display window
            sig = out['signal']
            lookback = min(52, disp_e - disp_s)
            segment = sig[disp_e - lookback:disp_e]
            t_seg = np.arange(lookback)
            coeffs = np.polyfit(t_seg, segment.real, 1)
            t_fwd = np.arange(lookback, lookback + actual_proj_weeks)
            proj = np.polyval(coeffs, t_fwd)
            projections.append({
                'projection': proj,
                'spec': spec,
                'method': 'linear_extrap',
                'slope_wk': coeffs[0],
            })
            print(f"  {spec['label']:20s}  LP slope: {coeffs[0]*FS:.4f}/yr")
            continue

        # BP filter: use envelope and phase at projection start
        env = out['envelope']
        phase = out['phase']

        # Amplitude: current envelope at display end
        A = env[disp_e - 1]

        # Phase rate: average dphi/dt over last N cycles
        period_samples = TWOPI / f0 * FS
        n_lookback = int(3 * period_samples)
        phase_window = phase[max(disp_s, disp_e - n_lookback):disp_e]

        dphi = np.diff(phase_window)
        dphi_avg = np.median(dphi)

        # Phase at projection start
        phi_start = phase[disp_e - 1]

        # Effective frequency (from phase rate)
        w_eff = dphi_avg * FS  # rad/yr

        # Project
        t_fwd = np.arange(1, actual_proj_weeks + 1)
        proj = A * np.cos(phi_start + dphi_avg * t_fwd)

        T_yr = TWOPI / f0
        T_eff = TWOPI / w_eff if w_eff > 0 else float('inf')
        print(f"  {spec['label']:20s}  A={A:.1f}  w_eff={w_eff:.3f} rad/yr "
              f"(T_nom={T_yr:.2f}yr, T_eff={T_eff:.2f}yr)")

        projections.append({
            'projection': proj,
            'amplitude': A,
            'dphi': dphi_avg,
            'phi_start': phi_start,
            'w_eff': w_eff,
            'spec': spec,
            'method': 'phase_extrap',
        })

    # Composite projection
    composite_proj = np.zeros(actual_proj_weeks)
    for p in projections:
        composite_proj += p['projection']

    # Actual prices in projection window
    actual_proj = close[disp_e:proj_e]

    # ========================================================================
    # PLOTTING
    # ========================================================================

    print(f"\nGenerating figure...")

    # Time axes
    dates_disp = dates[disp_s:disp_e]
    dates_proj = dates[disp_e:proj_e]
    dates_full = dates[disp_s:proj_e]

    n_filters = len(FILTER_SPECS)
    # Layout: composite + 6 filters x 2 (signal + phase/freq)
    fig = plt.figure(figsize=(18, 28))
    gs = fig.add_gridspec(n_filters + 1, 2, width_ratios=[3, 1],
                          hspace=0.3, wspace=0.15)

    # === Row 0: Composite price + projection ===
    ax_comp = fig.add_subplot(gs[0, :])
    ax_comp.plot(dates_disp, close[disp_s:disp_e], 'k-', lw=1.2,
                 label='DJIA Close')
    ax_comp.plot(dates_disp, composite[disp_s:disp_e], 'b-', lw=1, alpha=0.7,
                 label=f'Composite ({pct_energy:.1f}%)')
    # Projection
    ax_comp.plot(dates_proj, actual_proj, 'k-', lw=1.2, alpha=0.3,
                 label='Actual (projection window)')
    ax_comp.plot(dates_proj, composite_proj, 'm-', lw=1.5, alpha=0.8,
                 label='Projected composite')
    ax_comp.axvline(dates[disp_e], color='grey', ls='--', lw=1, alpha=0.5)
    ax_comp.set_ylabel('Price', fontsize=9)
    ax_comp.set_title('DJIA Close + 6-Filter Composite + 100-Week Projection',
                      fontsize=11, fontweight='bold')
    ax_comp.legend(loc='upper left', fontsize=8)
    ax_comp.grid(True, alpha=0.2)
    ax_comp.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_comp.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # === Rows 1-6: Individual filters ===
    for i, (out, proj) in enumerate(zip(filter_outputs, projections)):
        spec = out['spec']
        color = spec['color']

        # --- Left panel: signal + envelope + projection ---
        ax_sig = fig.add_subplot(gs[i + 1, 0])

        # In-sample signal
        sig = out['signal']
        sig_real = sig.real if np.iscomplexobj(sig) else sig
        ax_sig.plot(dates_disp, sig_real[disp_s:disp_e],
                    color=color, lw=0.5, alpha=0.6)

        # Envelope (BP only)
        if out['envelope'] is not None:
            env_disp = out['envelope'][disp_s:disp_e]
            ax_sig.plot(dates_disp, env_disp, color=color, lw=1.2, alpha=0.7)
            ax_sig.plot(dates_disp, -env_disp, color=color, lw=1.2, alpha=0.7)

        # Projection
        ax_sig.plot(dates_proj, proj['projection'],
                    color='magenta', lw=1.2, alpha=0.8)

        # Projection envelope (BP only)
        if spec['type'] != 'lp':
            ax_sig.plot(dates_proj, np.abs(proj['projection']),
                        color='magenta', lw=0.8, alpha=0.4, ls='--')
            ax_sig.plot(dates_proj, -np.abs(proj['projection']),
                        color='magenta', lw=0.8, alpha=0.4, ls='--')

        ax_sig.axvline(dates[disp_e], color='grey', ls='--', lw=0.8, alpha=0.5)
        ax_sig.axhline(0, color='grey', lw=0.3)
        ax_sig.set_ylabel(spec['label'], fontsize=8, rotation=0,
                          labelpad=80, ha='left')
        ax_sig.grid(True, alpha=0.15)
        ax_sig.tick_params(axis='y', labelsize=7)
        ax_sig.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_sig.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # --- Right panel: phase + frequency (BP only) ---
        ax_pf = fig.add_subplot(gs[i + 1, 1])

        if out['phase'] is not None:
            # Wrapped phase
            phasew = np.angle(out['signal'][disp_s:disp_e])
            ax_pf.plot(dates_disp, phasew, color=color, lw=0.3, alpha=0.5)
            ax_pf.set_ylabel('Phase (rad)', fontsize=7)
            ax_pf.set_ylim(-np.pi - 0.3, np.pi + 0.3)
            ax_pf.axhline(0, color='grey', lw=0.3)
            ax_pf.axhline(np.pi, color='grey', lw=0.2, ls=':')
            ax_pf.axhline(-np.pi, color='grey', lw=0.2, ls=':')

            # Instantaneous frequency on twin axis
            ax_freq = ax_pf.twinx()
            freq_disp = out['frequency'][disp_s:disp_e]
            # Convert from cycles/yr to rad/yr
            freq_rad = freq_disp * TWOPI
            ax_freq.plot(dates_disp, freq_rad, color='darkred', lw=0.4, alpha=0.5)
            f0 = out['f0']
            ax_freq.axhline(f0, color='darkred', lw=0.5, ls='--', alpha=0.5)

            # Show effective frequency from projection
            if 'w_eff' in proj:
                ax_freq.axhline(proj['w_eff'], color='magenta', lw=1, ls='-',
                                alpha=0.7, label=f"w_eff={proj['w_eff']:.2f}")
                ax_freq.legend(fontsize=6, loc='upper right')

            ax_freq.set_ylabel('w (rad/yr)', fontsize=7, color='darkred')
            ax_freq.tick_params(axis='y', labelsize=6, colors='darkred')
            # Set freq axis to reasonable range around center
            if f0 > 0:
                ax_freq.set_ylim(max(0, f0 * 0.5), f0 * 1.5)
        else:
            # LP: show the trend slope
            sig_lp = out['signal'][disp_s:disp_e]
            ax_pf.plot(dates_disp, sig_lp, color=color, lw=1)
            ax_pf.set_ylabel('LP value', fontsize=7)

        ax_pf.grid(True, alpha=0.15)
        ax_pf.tick_params(axis='both', labelsize=6)
        ax_pf.xaxis.set_major_locator(mdates.YearLocator(4))
        ax_pf.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.suptitle("Page 152: 6-Filter CMW Decomposition + 100-Week Projection\n"
                 "In-sample (colored) | Projection (magenta) | Phase & Frequency (right)",
                 fontsize=13, fontweight='bold', y=1.0)

    outpath = os.path.join(os.path.dirname(__file__),
                           'cmw_6filter_projection.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {outpath}")

    # --- Summary stats ---
    print(f"\n--- Projection Summary ---")
    print(f"  Composite at projection start: {composite[disp_e-1]:.1f}")
    print(f"  Composite projection[0]:       {composite_proj[0]:.1f}")
    print(f"  Actual price at proj start:    {close[disp_e]:.1f}")
    print(f"  Actual price at proj end:      {close[proj_e-1]:.1f}")
    print(f"  Projected price at proj end:   {composite_proj[-1]:.1f}")

    # Correlation between projected and actual direction
    actual_diff = np.diff(actual_proj)
    proj_diff = np.diff(composite_proj)
    direction_match = np.mean(np.sign(actual_diff) == np.sign(proj_diff))
    print(f"\n  Direction match (weekly): {direction_match*100:.1f}%")

    # Correlation
    corr = np.corrcoef(actual_proj, composite_proj)[0, 1]
    print(f"  Correlation:              {corr:.4f}")

    try:
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    except Exception:
        plt.close()


if __name__ == '__main__':
    main()
