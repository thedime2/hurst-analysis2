# -*- coding: utf-8 -*-
"""
Figure AI-3: Comb Filter Time-Domain Outputs
Appendix A, Figure AI-3 Reproduction

Shows FC-1 through FC-10 filter outputs stacked with vertical offsets,
matching Hurst's "Comb Output Example" layout.

Changes from v1:
  - ALL CSV data used for filtering (no date restriction) to avoid edge effects
  - Tighter vertical spacing between tracks
  - Wrapped phase overlaid on each track (thin gray)
  - 2-panel side-by-side: Ormsby | CMW

Two figures:
  fig_AI3_weekly.png  - Weekly data (fs=52)
  fig_AI3_daily.png   - Daily data  (fs~275)

Display window: 1934-12-07 to 1940-01-26

Reference: J.M. Hurst, Appendix A, Figure AI-3, p.193
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from src.time_frequency.cmw import ormsby_spec_to_cmw_params, apply_cmw

from utils_ai import (
    load_weekly_data, load_daily_data,
    design_comb_bank, make_ormsby_kernels, apply_comb_bank,
    get_window, daily_nw, FS_WEEKLY, NW_WEEKLY,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

N_DISPLAY  = 10    # FC-1 through FC-10
NORM_SIGMA = 3.0   # number of std-devs that maps to ±1 normalized unit
SPACING    = 4.0   # vertical spacing between zero-lines (normalized units)
TRACK_AMP  = 1.5   # target half-amplitude of each track in normalized units


def apply_cmw_bank_limited(signal, specs, fs, n_filters):
    """Apply CMW to first n_filters of specs."""
    results = []
    for spec in specs[:n_filters]:
        params = ormsby_spec_to_cmw_params(spec)
        out = apply_cmw(signal, f0=params['f0'], fwhm=params['fwhm'], fs=fs)
        out['spec'] = spec
        results.append(out)
    return results


def stacked_plot(ax, outputs, s_idx, e_idx, fs,
                 label_prefix='', waveform_color='steelblue',
                 env_color='crimson', phase_color='gray'):
    """
    Single-axis stacked layout: FC-1 at top, FC-N at bottom.
    Overlays wrapped phase (thin, secondary axis scaling).
    """
    n_samp = e_idx - s_idx
    samp_per_week = fs / FS_WEEKLY
    weeks = np.arange(n_samp) / samp_per_week

    # Compute global RMS from all tracks for consistent normalization
    rms_vals = [np.sqrt(np.mean(outputs[i]['signal'][s_idx:e_idx].real**2))
                for i in range(N_DISPLAY)]
    rms_vals = [r for r in rms_vals if r > 0]
    global_rms = np.median(rms_vals) if rms_vals else 1.0
    scale = TRACK_AMP / (NORM_SIGMA * global_rms)

    ytick_pos, ytick_lab = [], []

    for i in range(N_DISPLAY):
        offset = (N_DISPLAY - 1 - i) * SPACING   # FC-1 at top

        out = outputs[i]
        sig = out['signal'][s_idx:e_idx].real * scale
        env = out['envelope'][s_idx:e_idx] * scale if out['envelope'] is not None else None

        # Zero reference line
        ax.axhline(offset, color='silver', linewidth=0.4, zorder=1)

        # Waveform
        ax.plot(weeks, sig + offset, '-', color=waveform_color,
                linewidth=0.6, alpha=0.9, zorder=3)

        # Envelope (both ±)
        if env is not None:
            env_sm = uniform_filter1d(env, size=max(1, int(samp_per_week * 2)))
            ax.plot(weeks,  env_sm + offset, '--', color=env_color,
                    linewidth=0.75, alpha=0.7, zorder=2)
            ax.plot(weeks, -env_sm + offset, '--', color=env_color,
                    linewidth=0.75, alpha=0.7, zorder=2)

        # Wrapped phase overlay: scale to ±1 normalized unit (thin, on top)
        phasew = out.get('phasew')
        if phasew is not None:
            ph = phasew[s_idx:e_idx]
            ph_scaled = ph / np.pi * TRACK_AMP * 0.6   # ±π → ±0.6 track units
            ax.plot(weeks, ph_scaled + offset, '-', color=phase_color,
                    linewidth=0.5, alpha=0.45, zorder=4)

        # Y label
        fc = out['spec']['f_center']
        T_wk = 2 * np.pi / fc * FS_WEEKLY
        ytick_pos.append(offset)
        ytick_lab.append(f'FC-{i+1}\n{fc:.1f}r/y\n{T_wk:.0f}wk')

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab, fontsize=7.5, family='monospace')
    ax.set_xlim(0, weeks[-1] if len(weeks) > 0 else 1)
    ax.set_ylim(-SPACING * 0.55, (N_DISPLAY - 0.45) * SPACING)
    ax.set_xlabel('Weeks', fontsize=10)
    ax.grid(True, axis='x', alpha=0.18)
    ax.set_title(label_prefix, fontsize=11, fontweight='bold')


# ============================================================================
# GENERATE FIGURES
# ============================================================================

for data_label, load_fn in [
    ('Weekly', load_weekly_data),
    ('Daily',  load_daily_data),
]:
    print("=" * 70)
    print(f"AI-3: {data_label} data")
    print("=" * 70)

    if data_label == 'Weekly':
        close, dates_dt = load_fn()   # ALL data
        fs = FS_WEEKLY
        nw = NW_WEEKLY
    else:
        close, dates_dt, fs = load_fn()   # ALL data
        nw = daily_nw(fs)

    print(f"  Loaded {len(close)} points (ALL CSV data), fs={fs:.1f}, nw={nw}")

    # Display window
    s_idx, e_idx = get_window(dates_dt)
    n_disp = e_idx - s_idx
    print(f"  Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END} "
          f"({n_disp} samples = {n_disp / (fs/FS_WEEKLY):.0f} weeks)")

    # --- Ormsby filters ---
    print("  Applying Ormsby comb filters to ALL data...")
    specs = design_comb_bank(fs=fs, nw=nw)
    filters = make_ormsby_kernels(specs, fs=fs)
    orm_outputs = apply_comb_bank(close, filters, fs=fs)
    print("  Done (Ormsby).")

    # --- CMW filters (only FC-1..10 for speed) ---
    print("  Applying CMW filters (FC-1..10)...")
    cmw_outputs = apply_cmw_bank_limited(close, specs, fs=fs, n_filters=N_DISPLAY)
    # CMW apply_cmw does not return phasew; compute it
    for out in cmw_outputs:
        if out['phase'] is not None:
            out['phasew'] = np.angle(out['signal'])
    print("  Done (CMW).")
    print()

    # --- Figure ---
    fig, (ax_orm, ax_cmw) = plt.subplots(1, 2, figsize=(20, 12), sharey=True)

    stacked_plot(ax_orm, orm_outputs[:N_DISPLAY], s_idx, e_idx, fs,
                 label_prefix=f'Ormsby FIR  ({data_label})',
                 waveform_color='steelblue', env_color='firebrick',
                 phase_color='dimgray')

    stacked_plot(ax_cmw, cmw_outputs, s_idx, e_idx, fs,
                 label_prefix=f'CMW Gaussian  ({data_label})',
                 waveform_color='darkorange', env_color='forestgreen',
                 phase_color='dimgray')

    # Shared legend (use ax_orm)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='steelblue',  lw=0.8, label='Waveform (real)'),
        Line2D([0], [0], color='firebrick',  lw=0.8, ls='--', label='Envelope'),
        Line2D([0], [0], color='dimgray',    lw=0.6, alpha=0.6, label='Wrapped phase'),
    ]
    ax_orm.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)

    # Date annotation
    for ax in (ax_orm, ax_cmw):
        ax.text(0.01, 0.99, DATE_DISPLAY_START, transform=ax.transAxes,
                fontsize=7, va='top', ha='left', color='gray')
        ax.text(0.99, 0.99, DATE_DISPLAY_END, transform=ax.transAxes,
                fontsize=7, va='top', ha='right', color='gray')

    fig.suptitle(
        f'COMB OUTPUT EXAMPLE  -  Figure AI-3\n'
        f'{data_label} DJIA  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  '
        f'|  FC-1..FC-10  |  Blue=waveform  Red=envelope  Gray=wrapped phase',
        fontsize=11, fontweight='bold', y=0.995
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    out_path = os.path.join(SCRIPT_DIR, f'fig_AI3_{data_label.lower()}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print()

print("Done.")
