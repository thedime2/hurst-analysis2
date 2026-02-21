# -*- coding: utf-8 -*-
"""
Figure AI-4 v3 Final: Frequency Versus Time — Best Reproduction

Hurst-style single-panel reproduction of Figure AI-4 using:
  - Raw PT interleaved measurements (no smoothing of any kind)
  - Point placed at the time of the SECOND event (peak or trough)
  - Consecutive points joined by straight line segments
  - All 23 filters from the standard adjacent comb bank (step=0.2 r/y)
  - Amplitude gating to suppress gap filters (those whose window RMS
    falls below 30 % of the median — these are Hurst's "meaningless" filters)
  - Filter labels at BOTH left and right sides
  - Gridlines every 25 weeks (vertical) and at integer rad/yr (horizontal)
  - Colour ramp from blue (low fc) to red (high fc)

Also produces a 4-panel comparison figure showing:
  Panel 1: Reference image (Hurst's original AI-4)
  Panel 2: PT raw (all 23 filters, no gating)
  Panel 3: PT gated (gap filters removed)
  Panel 4: PT gated with harmonic-aligned spacing (step=0.3676)

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
    design_comb_bank, make_ormsby_kernels, apply_comb_bank,
    get_window, FS_WEEKLY, NW_WEEKLY,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_IMAGE  = os.path.join(SCRIPT_DIR, '../../references/appendix_a/figure_AI4_v2.png')

YMIN, YMAX   = 7.5, 12.5
CLIP_FRAC    = 0.30
AMP_GATE_THR = 0.30    # gate filters whose window RMS < 30 % of median
X_MAX_LABEL  = 275     # last x-axis label to show (matches Hurst)


# ============================================================================
# MEASUREMENT UTILITIES (raw PT, point at second event, no smoothing)
# ============================================================================

def parabolic_peak(y, idx):
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    y0, y1, y2 = float(y[idx-1]), float(y[idx]), float(y[idx+1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-14:
        return float(idx)
    return idx + np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0)


def find_peaks_sub(signal, f_center, fs, min_dist_frac=0.55):
    T_samp = 2 * np.pi / f_center * fs
    min_d  = max(3, int(T_samp * min_dist_frac))
    idx, _ = find_peaks(signal, distance=min_d)
    return np.array([parabolic_peak(signal, i) for i in idx])


def find_troughs_sub(signal, f_center, fs):
    return find_peaks_sub(-signal, f_center, fs)


def measure_pt(signal_real, f_center, fs):
    """
    Interleaved P→T / T→P half-period.
    Point placed at TIME OF SECOND EVENT.
    Skips consecutive same-type events.
    """
    peaks   = find_peaks_sub(signal_real, f_center, fs)
    troughs = find_troughs_sub(signal_real, f_center, fs)
    events  = ([(t, 'P') for t in peaks] + [(t, 'T') for t in troughs])
    events.sort(key=lambda x: x[0])
    if len(events) < 2:
        return np.array([]), np.array([])
    times, freqs = [], []
    for k in range(len(events) - 1):
        t1, e1 = events[k]
        t2, e2 = events[k+1]
        if e1 == e2:
            continue
        dt_yr = (t2 - t1) / fs
        if dt_yr <= 0:
            continue
        times.append(t2)
        freqs.append(np.pi / dt_yr)
    return np.array(times), np.array(freqs)


def clip_to_window(times, freqs, s_idx, e_idx, f_center):
    if len(times) == 0:
        return np.array([]), np.array([])
    mask  = (times >= s_idx) & (times < e_idx)
    t     = (times[mask] - s_idx).astype(float)   # weeks (weekly data → 1 sample/week)
    f     = freqs[mask]
    valid = ((f >= f_center * (1 - CLIP_FRAC)) & (f <= f_center * (1 + CLIP_FRAC)) &
             (f >= YMIN) & (f <= YMAX))
    return t[valid], f[valid]


def amplitude_gate(outputs, s_idx, e_idx, threshold=AMP_GATE_THR):
    """
    Return a boolean mask: True = filter has sufficient energy in display window.
    Filters whose RMS < threshold × median_RMS are gated out.
    """
    rms_vals = np.array([
        np.sqrt(np.mean(np.abs(out['signal'][s_idx:e_idx]) ** 2))
        for out in outputs
    ])
    median_rms = np.median(rms_vals)
    keep = rms_vals >= threshold * median_rms
    return keep, rms_vals, median_rms


# ============================================================================
# LOAD DATA AND APPLY STANDARD ADJACENT COMB BANK
# ============================================================================

print("Loading weekly DJIA data...")
close, dates_dt = load_weekly_data()
s_idx, e_idx = get_window(dates_dt)
n_weeks = e_idx - s_idx
print(f"Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} weeks)")

specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)

print("Applying Ormsby comb bank (step=0.2 r/y, adjacent)...")
outputs = apply_comb_bank(close, filters, fs=FS_WEEKLY)
print("Done.")


# ============================================================================
# AMPLITUDE GATING
# ============================================================================

keep_mask, rms_vals, median_rms = amplitude_gate(outputs, s_idx, e_idx)

print("\nAmplitude gating results (threshold = 30% of median RMS):")
print(f"  Median RMS: {median_rms:.4f}")
for i, (keep, rms, spec) in enumerate(zip(keep_mask, rms_vals, specs)):
    fc  = spec['f_center']
    pct = rms / median_rms * 100
    tag = '' if keep else '  ** GATED **'
    print(f"  FC-{i+1:2d}  fc={fc:.1f}  rms={rms:.4f}  ({pct:.0f}%){tag}")


# ============================================================================
# COMPUTE PT MEASUREMENTS (no smoothing)
# ============================================================================

print("\nComputing PT measurements (no smoothing, point at 2nd event)...")

meas_all   = []    # all 23 filters
meas_gated = []    # gated (gap filters removed)

for i, out in enumerate(outputs):
    fc  = specs[i]['f_center']
    sig = out['signal'].real
    t, f = measure_pt(sig, fc, FS_WEEKLY)
    t, f = clip_to_window(t, f, s_idx, e_idx, fc)
    meas_all.append((t, f, fc, i + 1))        # store filter number
    if keep_mask[i]:
        meas_gated.append((t, f, fc, i + 1))

n_all   = sum(len(t) for t, f, _, _ in meas_all)
n_gated = sum(len(t) for t, f, _, _ in meas_gated)
print(f"  All filters:   {n_all} points  ({n_all/23:.1f} avg/filter)")
print(f"  After gating:  {n_gated} points over {len(meas_gated)} filters  "
      f"({n_gated/max(len(meas_gated),1):.1f} avg/filter)")


# ============================================================================
# ALSO COMPUTE FOR HARMONIC-ALIGNED SPACING (0.3676 r/y step)
# ============================================================================

print("\nApplying harmonic-aligned comb bank (step=0.3676 r/y)...")

harm_step = 0.3676
fc_start_harm = 7.2 + 0.3 + 0.2 / 2.0    # W1_START + SKIRT + PASSBAND/2
n_harm = max(1, int((12.0 - fc_start_harm) / harm_step) + 1)

specs_harm  = design_hurst_comb_bank(
    n_filters=n_harm,
    w1_start=7.2,
    w_step=harm_step,
    passband_width=0.2,
    skirt_width=0.3,
    nw=NW_WEEKLY,
    fs=FS_WEEKLY,
)
filters_harm = create_filter_kernels(specs_harm, fs=FS_WEEKLY,
                                     filter_type='modulate', analytic=True)
result_harm  = apply_filter_bank(close, filters_harm, fs=FS_WEEKLY)
outputs_harm = result_harm['filter_outputs']

keep_harm, _, _ = amplitude_gate(outputs_harm, s_idx, e_idx)

meas_harm = []
for i, out in enumerate(outputs_harm):
    if not keep_harm[i]:
        continue
    fc  = specs_harm[i]['f_center']
    sig = out['signal'].real
    t, f = measure_pt(sig, fc, FS_WEEKLY)
    t, f = clip_to_window(t, f, s_idx, e_idx, fc)
    meas_harm.append((t, f, fc, i + 1))

n_harm_pts = sum(len(t) for t, f, _, _ in meas_harm)
print(f"  Harmonic-aligned: {n_harm} filters, {n_harm_pts} points "
      f"({n_harm_pts/max(len(meas_harm),1):.1f} avg/filter after gating)")


# ============================================================================
# COLOUR RAMP
# ============================================================================

_cmap = plt.colormaps['coolwarm']
N_FILT = len(specs)
COLORS_23 = [_cmap(i / (N_FILT - 1)) for i in range(N_FILT)]


# ============================================================================
# PLOT HELPER: SINGLE FVT PANEL (no smoothing, straight line segments)
# ============================================================================

def plot_fvt_panel(ax, meas_list, title, ymin=YMIN, ymax=YMAX,
                   label_right=True, alpha=0.85, lw=0.75, dot_size=2.0,
                   color_by_idx=True, single_color=None):
    """
    Plot FVT for a list of (t, f, fc, filter_num) tuples.
    color_by_idx=True: use filter index in the 23-filter ramp.
    Otherwise use single_color.
    """
    for item in meas_list:
        t, f, fc, fnum = item
        if len(t) == 0:
            continue
        # Color: use position in 23-filter ramp by fc value, or single color
        if color_by_idx:
            idx23 = round((fc - 7.6) / 0.2)
            idx23 = max(0, min(idx23, N_FILT - 1))
            c = COLORS_23[idx23]
        else:
            c = single_color or 'steelblue'

        # Straight line segments joining measurement points (no smoothing)
        ax.plot(t, f, '-', color=c, linewidth=lw, alpha=alpha, zorder=3)
        ax.plot(t, f, 'o', color=c, markersize=dot_size, zorder=4)

        # Labels at start and end
        if len(t) > 0:
            ax.text(t[0]  - 2, f[0],  str(fnum), fontsize=5.5, color=c,
                    ha='right', va='center', zorder=5)
        if label_right and len(t) > 0:
            ax.text(t[-1] + 2, f[-1], str(fnum), fontsize=5.5, color=c,
                    ha='left',  va='center', zorder=5)

    # Gridlines
    for ref in range(8, 13):
        ax.axhline(ref, color='#AAAAAA', linewidth=0.6, zorder=1)
    for wk in range(0, n_weeks + 50, 25):
        ax.axvline(wk, color='#CCCCCC', linewidth=0.35, zorder=1)

    ax.set_xlim(0, max(n_weeks, X_MAX_LABEL))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('WEEKS', fontsize=10)
    ax.set_ylabel('RADIANS/YEAR', fontsize=10)
    ax.set_xticks(range(0, X_MAX_LABEL + 1, 25))
    ax.set_xticklabels([str(x) if x <= X_MAX_LABEL else '' for x in range(0, X_MAX_LABEL + 1, 25)],
                       fontsize=8)
    ax.set_yticks([8, 9, 10, 11, 12])
    ax.set_title(title, fontsize=9, fontweight='bold')


# ============================================================================
# FIGURE 1: HURST-STYLE SINGLE PANEL (gated, adjacent step)
# ============================================================================

print("\nGenerating Figure 1: Hurst-style single panel (gated, step=0.2 r/y)...")

fig1, ax1 = plt.subplots(figsize=(13, 9))
plot_fvt_panel(ax1, meas_gated,
               'FREQUENCY VERSUS TIME  —  Figure AI-4\n'
               f'Ormsby FIR  |  PT raw  |  No smoothing  |  '
               f'Amplitude-gated ({len(meas_gated)}/23 filters shown)',
               label_right=True)

# Right-side y-axis (matches Hurst's layout)
ax1r = ax1.twinx()
ax1r.set_ylim(YMIN, YMAX)
ax1r.set_yticks([8, 9, 10, 11, 12])
ax1r.tick_params(labelsize=9)

# Annotate date range
ax1.text(0.01, 0.98, DATE_DISPLAY_START, transform=ax1.transAxes,
         fontsize=7, va='top', color='gray')
ax1.text(0.99, 0.98, DATE_DISPLAY_END, transform=ax1.transAxes,
         fontsize=7, va='top', ha='right', color='gray')

fig1.tight_layout()
out1 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_final.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved: {out1}")


# ============================================================================
# FIGURE 2: 4-PANEL COMPARISON
#   [0] Reference image  [1] PT all 23 (no gate)
#   [2] PT gated (adj)   [3] PT gated (harmonic)
# ============================================================================

print("Generating Figure 2: 4-panel comparison...")

fig2, axes2 = plt.subplots(2, 2, figsize=(22, 16),
                            gridspec_kw={'wspace': 0.15, 'hspace': 0.35})

# Panel 0: reference image
ax00 = axes2[0, 0]
if os.path.exists(REF_IMAGE):
    ref_img = mpimg.imread(REF_IMAGE)
    ax00.imshow(ref_img, aspect='auto')
    ax00.axis('off')
    ax00.set_title("Hurst's Original AI-4 (reference scan)", fontsize=9, fontweight='bold')
else:
    ax00.text(0.5, 0.5, 'Reference image not found\n' + REF_IMAGE,
              transform=ax00.transAxes, ha='center', va='center', fontsize=9)
    ax00.axis('off')
    ax00.set_title("Reference (not found)", fontsize=9)

# Panel 1: PT all filters, no gating
ax01 = axes2[0, 1]
plot_fvt_panel(ax01, meas_all,
               f'PT raw — all 23 filters (no amplitude gate)\n'
               f'{n_all} points ({n_all/23:.1f} avg/filter)',
               label_right=False)

# Panel 2: PT gated (adjacent step)
ax10 = axes2[1, 0]
n_g  = len(meas_gated)
plot_fvt_panel(ax10, meas_gated,
               f'PT raw — amplitude-gated ({n_g}/23 filters, thr={AMP_GATE_THR:.0%})\n'
               f'{n_gated} points ({n_gated/max(n_g,1):.1f} avg/filter)',
               label_right=True)

# Panel 3: PT gated harmonic-aligned
ax11 = axes2[1, 1]
n_hg = len(meas_harm)
plot_fvt_panel(ax11, meas_harm,
               f'PT raw — harmonic-aligned (step={harm_step} r/y, gated)\n'
               f'{n_harm_pts} points ({n_harm_pts/max(n_hg,1):.1f} avg/filter)',
               label_right=True, color_by_idx=False, single_color=None)

fig2.suptitle(
    f'AI-4 Comparison  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}\n'
    'PT measurements (point at 2nd event)  |  No smoothing  |  Straight line segments',
    fontsize=11, fontweight='bold'
)
out2 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_4panel.png')
fig2.savefig(out2, dpi=130, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: {out2}")


# ============================================================================
# FIGURE 3: REFERENCE UNDERLAY + GATED PT OVERLAY
# ============================================================================

print("Generating Figure 3: Reference underlay + gated PT overlay...")

fig3, ax3 = plt.subplots(figsize=(14, 9))

if os.path.exists(REF_IMAGE):
    ref_img = mpimg.imread(REF_IMAGE)
    ax3.imshow(ref_img, extent=[0, X_MAX_LABEL, YMIN, YMAX],
               aspect='auto', alpha=0.50, zorder=1)

plot_fvt_panel(ax3, meas_gated,
               f'AI-4  |  Reference Underlay + PT Gated Overlay\n'
               f'{len(meas_gated)}/23 filters  |  No smoothing  |  '
               f'Reference at 50% opacity',
               label_right=True, alpha=0.90, lw=0.8, dot_size=2.5)

out3 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_final_overlay.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"  Saved: {out3}")


# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 60)
print("AI-4 v3 Final Summary")
print("=" * 60)
print(f"Display window : {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} wks)")
print(f"Method         : PT interleaved, point at 2nd event, no smoothing")
print(f"All 23 filters : {n_all} pts  ({n_all/23:.1f} avg/filter)")
print(f"After gating   : {n_gated} pts over {len(meas_gated)} filters")
print(f"  Gated filters: {[i+1 for i, k in enumerate(keep_mask) if not k]}")
print(f"Harmonic-align : {n_harm_pts} pts over {n_hg} filters")
print()
print("Output files:")
for f in [out1, out2, out3]:
    print(f"  {os.path.basename(f)}")
print()
print("Done.")
