# -*- coding: utf-8 -*-
"""
Figure AI-4 v3: Spacing Diagnostic — Per-Cycle Frequency Divergence

This script investigates whether different decimation spacings produce
genuinely different measured frequency VALUES (not just density), and
which spacing best matches Hurst's AI-4 reference.

Two mechanisms for spacing-induced frequency change:
  1. Timing quantization error (partially corrected by parabolic interpolation)
  2. Missed peaks — different cycles selected, changing which P-P interval is measured

Both cause the zig-zag pattern to differ across spacings. The reference overlay
reveals which spacing produces zig-zags closest to Hurst's original.

Figures produced:
  fig_AI4_v3_spacing_diag_waveforms.png  — waveforms with per-spacing peak marks
  fig_AI4_v3_spacing_diag_freqtrace.png  — per-filter frequency trace, all spacings
  fig_AI4_v3_spacing_diag_overlay.png    — all spacings overlaid on reference image

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figure AI-4, p.194
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks

from src.filters import design_hurst_comb_bank, create_filter_kernels, apply_filter_bank

from utils_ai import (
    load_weekly_data,
    get_window, FS_WEEKLY, NW_WEEKLY,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_IMAGE  = os.path.join(SCRIPT_DIR, '../../references/appendix_a/figure_AI4_v2.png')

YMIN, YMAX  = 7.5, 12.5
CLIP_FRAC   = 0.30
X_MAX_LABEL = 275
SPACINGS    = [1, 4, 7, 10]
SP_COLORS   = ['steelblue', 'forestgreen', 'firebrick', 'darkorange']
SP_LABELS   = ['spacing=1 (dense)', 'spacing=4', 'spacing=7', 'spacing=10']

# Filters to examine in detail (indices 0-22, FC-1..FC-23)
DIAG_FILTERS = [0, 3, 6, 9]   # FC-1, FC-4, FC-7, FC-10


# ============================================================================
# MEASUREMENT FUNCTIONS (same as main spacing script)
# ============================================================================

def parabolic_peak(y, idx):
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    y0, y1, y2 = float(y[idx-1]), float(y[idx]), float(y[idx+1])
    d = y0 - 2.0*y1 + y2
    if abs(d) < 1e-14:
        return float(idx)
    return idx + np.clip(0.5*(y0-y2)/d, -1.0, 1.0)


def extract_peaks_spaced(analytic_full, f_center, fs, s_idx, e_idx, spacing,
                          return_troughs=False):
    """
    Extract peak (and optionally trough) positions in FULL-window weeks,
    from a NaN-gapped spaced output. Returns fractional week positions
    after parabolic sub-sample correction.
    """
    seg      = analytic_full[s_idx:e_idx].real
    valid_idx = np.where(~np.isnan(seg))[0]
    if len(valid_idx) < 4:
        return np.array([]), np.array([])

    compact    = seg[valid_idx]
    fs_compact = fs / spacing
    T_compact  = 2 * np.pi / f_center * fs_compact
    min_d      = max(2, int(T_compact * 0.45))

    def compact_to_weeks(cf_arr):
        if len(cf_arr) == 0:
            return np.array([])
        i0   = np.clip(np.floor(cf_arr).astype(int), 0, len(valid_idx)-1)
        i1   = np.clip(i0 + 1, 0, len(valid_idx)-1)
        frac = cf_arr - i0
        return valid_idx[i0] + frac * (valid_idx[i1] - valid_idx[i0])

    peak_ci, _ = find_peaks(compact, distance=min_d)
    peak_cf    = np.array([parabolic_peak(compact, i) for i in peak_ci])
    peak_wk    = compact_to_weeks(peak_cf)

    if not return_troughs:
        return peak_wk, compact[peak_ci] if len(peak_ci) else np.array([])

    trough_ci, _ = find_peaks(-compact, distance=min_d)
    trough_cf    = np.array([parabolic_peak(-compact, i) for i in trough_ci])
    trough_wk    = compact_to_weeks(trough_cf)
    trough_amp   = compact[trough_ci] if len(trough_ci) else np.array([])

    return peak_wk, compact[peak_ci], trough_wk, trough_amp


def measure_pp_full(peak_wk, f_center, fs):
    """PP frequency measurements from pre-computed peak positions."""
    if len(peak_wk) < 2:
        return np.array([]), np.array([])
    dt_yr = np.diff(peak_wk) / fs
    freqs = 2 * np.pi / np.where(dt_yr > 0, dt_yr, np.nan)
    times = peak_wk[1:]
    ok    = (np.isfinite(freqs) &
             (freqs >= f_center * (1-CLIP_FRAC)) &
             (freqs <= f_center * (1+CLIP_FRAC)) &
             (freqs >= YMIN) & (freqs <= YMAX))
    return times[ok], freqs[ok]


# ============================================================================
# LOAD DATA AND DESIGN FILTER BANK
# ============================================================================

print("Loading weekly DJIA data...")
close, dates_dt = load_weekly_data()
s_idx, e_idx = get_window(dates_dt)
n_weeks = e_idx - s_idx
print(f"Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} weeks)")

specs_base  = design_hurst_comb_bank(
    n_filters=23, w1_start=7.2, w_step=0.2,
    passband_width=0.2, skirt_width=0.3,
    nw=NW_WEEKLY, fs=FS_WEEKLY
)
filters_base = create_filter_kernels(specs_base, fs=FS_WEEKLY,
                                      filter_type='modulate', analytic=True)


# ============================================================================
# APPLY ALL SPACINGS (keep full filter_outputs for waveform access)
# ============================================================================

results_by_sp = {}   # spacing -> filter_outputs list
peaks_by_sp   = {}   # spacing -> {fidx: (peak_wk, peak_amp, trough_wk, trough_amp)}
meas_by_sp    = {}   # spacing -> {fidx: (t_weeks, f_radyr)} [PP]

for sp in SPACINGS:
    print(f"\nspacing={sp}  (nw_dec={NW_WEEKLY//sp})")
    result  = apply_filter_bank(close, filters_base, fs=FS_WEEKLY,
                                 mode='reflect', spacing=sp, startidx=0,
                                 interp='none')
    outputs = result['filter_outputs']
    results_by_sp[sp] = outputs

    pk_dict, meas_dict = {}, {}
    for fidx in DIAG_FILTERS:
        fc = specs_base[fidx]['f_center']
        z  = outputs[fidx]['signal']
        pw, pa, tw, ta = extract_peaks_spaced(
            z, fc, FS_WEEKLY, s_idx, e_idx, sp, return_troughs=True)
        pk_dict[fidx]   = (pw, pa, tw, ta)
        t_pp, f_pp      = measure_pp_full(pw, fc, FS_WEEKLY)
        meas_dict[fidx] = (t_pp, f_pp)
        print(f"  FC-{fidx+1}: {len(pw)} peaks, {len(t_pp)} PP measurements")

    peaks_by_sp[sp] = pk_dict
    meas_by_sp[sp]  = meas_dict


# ============================================================================
# FIGURE 1: WAVEFORMS WITH PER-SPACING PEAK MARKS
# For each diagnostic filter: plot spacing=1 dense waveform behind;
# overlay colored dot markers showing where each spacing detects peaks.
# ============================================================================

print("\nGenerating Figure 1: waveform peak positions per spacing...")

weeks_arr = np.arange(n_weeks, dtype=float)

fig1, axes1 = plt.subplots(len(DIAG_FILTERS), 1, figsize=(18, 14),
                            gridspec_kw={'hspace': 0.55})

for row, fidx in enumerate(DIAG_FILTERS):
    ax  = axes1[row]
    fc  = specs_base[fidx]['f_center']
    T_wk = 2 * np.pi / fc * FS_WEEKLY

    # Dense spacing=1 waveform as background
    sig1 = results_by_sp[1][fidx]['signal'][s_idx:e_idx].real
    ax.plot(weeks_arr, sig1, color='#AAAAAA', linewidth=0.5, zorder=1, label='_nolegend_')
    ax.axhline(0, color='#CCCCCC', linewidth=0.4, zorder=0)

    # Amplitude envelope reference lines
    env_rms = np.sqrt(np.nanmean(sig1**2))
    for sign in [1, -1]:
        ax.axhline(sign * 3 * env_rms, color='#DDDDDD', linewidth=0.35, linestyle='--')

    # Peak markers for each spacing (except spacing=1 which is too dense to show well)
    for sp, clr, lbl in zip(SPACINGS, SP_COLORS, SP_LABELS):
        pw, pa, tw, ta = peaks_by_sp[sp][fidx]
        mkr_size = {1: 2.5, 4: 5, 7: 7, 10: 9}[sp]
        mkr_style= {1: '.', 4: 'o', 7: 's', 10: '^'}[sp]
        if len(pw) > 0:
            # Get signal amplitude at (interpolated) peak week positions
            pw_int = np.clip(np.round(pw).astype(int), 0, n_weeks-1)
            amp_at_peak = sig1[pw_int]
            ax.plot(pw, amp_at_peak, mkr_style, color=clr, markersize=mkr_size,
                    alpha=0.85, zorder=3+list(SPACINGS).index(sp),
                    label=f'{lbl} ({len(pw)} peaks)')
        if len(tw) > 0 and sp > 1:
            tw_int = np.clip(np.round(tw).astype(int), 0, n_weeks-1)
            amp_at_trough = sig1[tw_int]
            ax.plot(tw, amp_at_trough, mkr_style, color=clr, markersize=mkr_size,
                    alpha=0.40, zorder=3+list(SPACINGS).index(sp),
                    label='_nolegend_', markerfacecolor='white', markeredgecolor=clr)

    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.set_xlim(0, n_weeks)
    ax.set_xlabel('Weeks' if row == len(DIAG_FILTERS)-1 else '', fontsize=8)
    ax.set_ylabel('Amplitude', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis='x', alpha=0.15)
    ax.set_title(
        f'FC-{fidx+1}  fc={fc:.1f} r/y  T={T_wk:.0f}wk  —  Peaks detected per spacing  '
        f'(squares=spacing=7, triangles=spacing=10)',
        fontsize=8.5, fontweight='bold'
    )

fig1.suptitle(
    f'Waveform Peak Positions vs Decimation Spacing\n'
    f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}  |  Background: dense spacing=1 waveform  '
    f'|  Filled=peaks, Open=troughs (spacing>1)',
    fontsize=10, fontweight='bold'
)
out1 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_spacing_diag_waveforms.png')
fig1.savefig(out1, dpi=140, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved: {out1}")


# ============================================================================
# FIGURE 2: PER-FILTER FREQUENCY TRACE DIVERGENCE
# For each diagnostic filter: frequency vs weeks for all spacings overlaid.
# This directly shows whether different spacings give different freq VALUES.
# ============================================================================

print("Generating Figure 2: frequency trace divergence per filter...")

fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12),
                            gridspec_kw={'hspace': 0.45, 'wspace': 0.2})

for panel, fidx in enumerate(DIAG_FILTERS):
    ax  = axes2.flat[panel]
    fc  = specs_base[fidx]['f_center']
    T_wk = 2 * np.pi / fc * FS_WEEKLY

    # Compute timing errors vs spacing=1 baseline
    t_base, f_base = meas_by_sp[1][fidx]
    timing_rms = {}

    for sp, clr, lbl in zip(SPACINGS, SP_COLORS, SP_LABELS):
        t, f = meas_by_sp[sp][fidx]
        if len(t) == 0:
            continue

        # Connect measurements with lines (Hurst style)
        ax.plot(t, f, '-o', color=clr, linewidth=1.0 if sp > 1 else 0.6,
                markersize=4.5 if sp > 1 else 2.5, alpha=0.85,
                zorder=3+SPACINGS.index(sp), label=lbl)

        # Timing error vs spacing=1: find nearest spacing=1 measurement per spacing>1 point
        if sp > 1 and len(t_base) > 0:
            freq_errors = []
            for tj, fj in zip(t, f):
                dist = np.abs(t_base - tj)
                nearest = np.argmin(dist)
                if dist[nearest] < T_wk * 0.6:  # within 60% of one period
                    freq_errors.append(fj - f_base[nearest])
            if freq_errors:
                rms_e = np.sqrt(np.mean(np.array(freq_errors)**2))
                timing_rms[sp] = rms_e

    # Add filter center reference line
    ax.axhline(fc, color='black', linewidth=0.6, linestyle=':', alpha=0.5,
               label=f'fc = {fc:.1f} r/y')

    # ±CLIP_FRAC band
    ax.axhspan(fc*(1-CLIP_FRAC), fc*(1+CLIP_FRAC), alpha=0.05, color='gray')

    ax.set_xlim(0, n_weeks)
    ax.set_ylim(max(YMIN, fc*0.6), min(YMAX, fc*1.4))
    ax.set_xlabel('Weeks', fontsize=8)
    ax.set_ylabel('rad/yr', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.2)

    rms_str = '  '.join([f'sp={sp}: ±{v:.2f}r/y' for sp, v in timing_rms.items()])
    ax.set_title(
        f'FC-{fidx+1}  fc={fc:.1f}r/y  T={T_wk:.0f}wk  |  PP frequency trace\n'
        f'Freq RMS error vs sp=1: {rms_str if rms_str else "n/a"}',
        fontsize=8, fontweight='bold'
    )

fig2.suptitle(
    f'Per-Cycle Frequency Divergence with Decimation Spacing\n'
    f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}  |  PP measurement (1/cycle)  '
    f'|  "Freq RMS error" = RMS of (freq_sp_N - freq_sp_1) at matched time points',
    fontsize=10, fontweight='bold'
)
out2 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_spacing_diag_freqtrace.png')
fig2.savefig(out2, dpi=140, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: {out2}")


# ============================================================================
# FIGURE 3: ALL SPACINGS OVERLAID ON REFERENCE IMAGE
# Full 23-filter FVT plot for each spacing, overlaid on Hurst's AI-4 scan.
# Shows which spacing best aligns with Hurst's original dot positions.
# ============================================================================

print("Generating Figure 3: all spacings vs reference overlay...")

# Recompute PP for ALL 23 filters for each spacing
all_meas = {}   # spacing -> [(t_wk, f_radyr, fc) per filter]
for sp in SPACINGS:
    meas_all = []
    for fidx in range(23):
        fc = specs_base[fidx]['f_center']
        z  = results_by_sp[sp][fidx]['signal']
        pw, _ = extract_peaks_spaced(z, fc, FS_WEEKLY, s_idx, e_idx, sp,
                                      return_troughs=False)
        t_pp, f_pp = measure_pp_full(pw, fc, FS_WEEKLY)
        meas_all.append((t_pp, f_pp, fc))
    all_meas[sp] = meas_all

_cmap  = plt.colormaps['coolwarm']
COLORS = [_cmap(i / 22) for i in range(23)]

def plot_fvt(ax, meas_list, connect=True, dot_size=1.8, lw=0.75, alpha=0.85):
    for i, (t, f, fc) in enumerate(meas_list):
        if len(t) == 0:
            continue
        c = COLORS[i]
        if connect and len(t) > 1:
            ax.plot(t, f, '-', color=c, linewidth=lw, alpha=alpha, zorder=3)
        ax.plot(t, f, 'o', color=c, markersize=dot_size, alpha=alpha, zorder=4)
    for ref in range(8, 13):
        ax.axhline(ref, color='#BBBBBB', linewidth=0.5, zorder=1)
    for wk in range(0, n_weeks + 50, 25):
        ax.axvline(wk, color='#DDDDDD', linewidth=0.3, zorder=1)
    ax.set_xlim(0, X_MAX_LABEL)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(range(0, X_MAX_LABEL + 1, 25))
    ax.set_xticklabels([str(x) if x % 50 == 0 else '' for x in range(0, X_MAX_LABEL+1, 25)],
                        fontsize=7)
    ax.set_yticks([8, 9, 10, 11, 12])
    ax.tick_params(labelsize=7)
    ax.set_xlabel('Weeks', fontsize=8)
    ax.set_ylabel('rad/yr', fontsize=8)

if os.path.exists(REF_IMAGE):
    ref_img = mpimg.imread(REF_IMAGE)

    fig3, axes3 = plt.subplots(2, 2, figsize=(22, 16),
                                gridspec_kw={'hspace': 0.40, 'wspace': 0.08})

    for ax, sp, lbl in zip(axes3.flat, SPACINGS, SP_LABELS):
        ax.imshow(ref_img, extent=[0, X_MAX_LABEL, YMIN, YMAX],
                  aspect='auto', alpha=0.45, zorder=1)
        plot_fvt(ax, all_meas[sp], connect=True, dot_size=2.5, lw=0.9, alpha=0.90)
        n_pts = sum(len(t) for t, f, _ in all_meas[sp])
        ax.set_title(
            f'{lbl}  |  {n_pts} pts total ({n_pts/23:.1f} avg/filter)  |  PP measurement\n'
            f'Reference at 45% opacity — do dots align with Hurst?',
            fontsize=9, fontweight='bold'
        )

    fig3.suptitle(
        f'AI-4 Spacing Comparison vs Hurst Reference\n'
        f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}  |  '
        f'PP (full-cycle) measurement  |  All 23 filters, amplitude-unclipped',
        fontsize=11, fontweight='bold'
    )
    out3 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_spacing_diag_overlay.png')
    fig3.savefig(out3, dpi=140, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved: {out3}")
else:
    print(f"  Reference image not found: {REF_IMAGE} — skipping Figure 3")
    out3 = None


# ============================================================================
# QUANTITATIVE TIMING ERROR SUMMARY
# ============================================================================

print()
print("=" * 70)
print("Timing error (spacing N vs spacing=1 baseline)  [PP measurements]")
print("=" * 70)
print(f"{'Filter':>8}  {'fc(r/y)':>8}  {'T(wk)':>6}  ", end='')
for sp in SPACINGS[1:]:
    print(f"  sp={sp} RMS(r/y)", end='')
print()

for fidx in DIAG_FILTERS:
    fc   = specs_base[fidx]['f_center']
    T_wk = 2 * np.pi / fc * FS_WEEKLY
    t1, f1 = meas_by_sp[1][fidx]
    print(f"  FC-{fidx+1:2d}  {fc:>8.1f}  {T_wk:>6.1f}  ", end='')
    for sp in SPACINGS[1:]:
        t_sp, f_sp = meas_by_sp[sp][fidx]
        errors = []
        for tj, fj in zip(t_sp, f_sp):
            if len(t1) == 0:
                break
            dist = np.abs(t1 - tj)
            nearest = np.argmin(dist)
            if dist[nearest] < T_wk * 0.6:
                errors.append(fj - f1[nearest])
        rms = np.sqrt(np.mean(np.array(errors)**2)) if errors else float('nan')
        pct = 100 * rms / fc if np.isfinite(rms) else float('nan')
        print(f"  {rms:>8.3f} ({pct:.1f}%)", end='')
    print()

print()
print("Note: RMS frequency error = sqrt(mean((f_sp_N - f_sp_1)^2)) at matched time points")
print("  — quantifies how much the zig-zag VALUES shift with spacing")
print()
print("Output files:")
for out in [out1, out2, out3]:
    if out:
        print(f"  {os.path.basename(out)}")
print("\nDone.")
