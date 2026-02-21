# -*- coding: utf-8 -*-
"""
Figure AI-4 v3: Reference Image Overlay — Alignment Diagnostic

Loads the hi-res reference scan (figure_AI4_v2.png) and underlays it on a
matplotlib axis mapped to the correct data coordinates, then overlays our
computed FVT measurements on top.

This lets us see EXACTLY where Hurst placed each frequency measurement point
relative to our values, which measurement method (PT, PP, ZC) matches best,
and whether Hurst places the point at the second event or the midpoint.

Figures produced:
  fig_AI4_v3_overlay_pt.png   - PT half-period measurements overlaid
  fig_AI4_v3_overlay_pp.png   - PP full-period measurements overlaid
  fig_AI4_v3_overlay_zc.png   - Zero-crossing half-period overlaid
  fig_AI4_v3_overlay_all.png  - All three methods on one panel (different colors)

Reference coordinate mapping:
  The AI-4 reference shows:
    x-axis: 0 to 275 weeks (last labeled tick); actual display ~267 weeks
    y-axis: 8 to 12 rad/yr with gridlines at integers
  We map the image to data extent: x=[0, 275], y=[7.5, 12.5]
  (the image margins place the plot area slightly inset — approximated here)

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

from utils_ai import (
    load_weekly_data,
    design_comb_bank, make_ormsby_kernels, apply_comb_bank,
    get_window, FS_WEEKLY, NW_WEEKLY,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_IMAGE  = os.path.join(SCRIPT_DIR, '../../references/appendix_a/figure_AI4_v2.png')

YMIN, YMAX   = 7.5, 12.5
CLIP_FRAC    = 0.30     # discard measurements > ±30 % from filter centre
X_MAX_REF    = 275      # last labeled x-tick in reference image (weeks)


# ============================================================================
# MEASUREMENT UTILITIES (no smoothing — raw events only)
# ============================================================================

def parabolic_peak(y, idx):
    """Sub-sample peak position via 3-point parabolic interpolation."""
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    y0, y1, y2 = float(y[idx-1]), float(y[idx]), float(y[idx+1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-14:
        return float(idx)
    return idx + np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0)


def find_peaks_sub(signal, f_center, fs, min_dist_frac=0.55):
    """Find peaks with parabolic sub-sample refinement."""
    T_samp = 2 * np.pi / f_center * fs
    min_d  = max(3, int(T_samp * min_dist_frac))
    idx, _ = find_peaks(signal, distance=min_d)
    return np.array([parabolic_peak(signal, i) for i in idx])


def find_troughs_sub(signal, f_center, fs):
    return find_peaks_sub(-signal, f_center, fs)


def find_zerocross(signal):
    """Return fractional-sample zero-crossing positions (sign changes)."""
    s = np.asarray(signal, dtype=float)
    nz = np.where(np.sign(s) != 0)[0]
    crossings = []
    for i in range(len(nz) - 1):
        a, b = nz[i], nz[i+1]
        if np.sign(s[a]) != np.sign(s[b]):
            frac = abs(s[a]) / (abs(s[a]) + abs(s[b]))
            crossings.append(a + frac)
    return np.array(crossings)


def measure_pt(signal_real, f_center, fs):
    """
    Interleaved P→T and T→P half-period measurements.
    Point is placed at the TIME OF THE SECOND EVENT.
    Returns (times_samples, freqs_radyr).
    """
    peaks  = find_peaks_sub(signal_real, f_center, fs)
    troughs = find_troughs_sub(signal_real, f_center, fs)
    events = ([(t, 'P') for t in peaks] + [(t, 'T') for t in troughs])
    events.sort(key=lambda x: x[0])
    if len(events) < 2:
        return np.array([]), np.array([])
    times, freqs = [], []
    for k in range(len(events) - 1):
        t1, e1 = events[k]
        t2, e2 = events[k+1]
        # Skip same-type consecutive events (two peaks in a row = missed trough)
        if e1 == e2:
            continue
        dt_yr = (t2 - t1) / fs
        if dt_yr <= 0:
            continue
        # Half-period → angular frequency
        freq = np.pi / dt_yr
        # Point placed at the SECOND event (t2), in sample units
        times.append(t2)
        freqs.append(freq)
    return np.array(times), np.array(freqs)


def measure_pp(signal_real, f_center, fs):
    """
    Peak-to-peak full-period measurements.
    Point placed at the SECOND PEAK.
    """
    peaks = find_peaks_sub(signal_real, f_center, fs)
    if len(peaks) < 2:
        return np.array([]), np.array([])
    dt_yr = np.diff(peaks) / fs
    freqs = 2 * np.pi / dt_yr
    times = peaks[1:]
    return times, freqs


def measure_zc(signal_real, fs):
    """
    Zero-crossing half-period measurements.
    Point placed at the MIDPOINT between consecutive crossings.
    """
    crossings = find_zerocross(signal_real)
    if len(crossings) < 2:
        return np.array([]), np.array([])
    dt_yr = np.diff(crossings) / fs
    freqs = np.pi / dt_yr
    times = (crossings[:-1] + crossings[1:]) / 2.0
    return times, freqs


def clip_to_window(times, freqs, s_idx, e_idx, f_center, samp_per_wk=1.0):
    """Restrict to display window and ±CLIP_FRAC of filter centre. Return weeks."""
    if len(times) == 0:
        return np.array([]), np.array([])
    mask = (times >= s_idx) & (times < e_idx)
    t = (times[mask] - s_idx) / samp_per_wk
    f = freqs[mask]
    valid = ((f >= f_center * (1 - CLIP_FRAC)) & (f <= f_center * (1 + CLIP_FRAC)) &
             (f >= YMIN) & (f <= YMAX))
    return t[valid], f[valid]


# ============================================================================
# LOAD DATA AND APPLY FILTERS
# ============================================================================

print("Loading weekly DJIA data...")
close, dates_dt = load_weekly_data()
s_idx, e_idx = get_window(dates_dt)
n_weeks = e_idx - s_idx
print(f"Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} weeks)")

specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)

print("Applying Ormsby comb bank...")
outputs = apply_comb_bank(close, filters, fs=FS_WEEKLY)
print("Done.")


# ============================================================================
# COMPUTE MEASUREMENTS (all three methods, all 23 filters)
# ============================================================================

print("Computing PT, PP, ZC measurements (no smoothing)...")

all_pt = []   # list of (t_weeks, f_radyr) per filter
all_pp = []
all_zc = []

for i, out in enumerate(outputs):
    fc  = specs[i]['f_center']
    sig = out['signal'].real

    t_pt, f_pt = measure_pt(sig, fc, FS_WEEKLY)
    t_pp, f_pp = measure_pp(sig, fc, FS_WEEKLY)
    t_zc, f_zc = measure_zc(sig, FS_WEEKLY)

    t_pt, f_pt = clip_to_window(t_pt, f_pt, s_idx, e_idx, fc)
    t_pp, f_pp = clip_to_window(t_pp, f_pp, s_idx, e_idx, fc)
    t_zc, f_zc = clip_to_window(t_zc, f_zc, s_idx, e_idx, fc)

    all_pt.append((t_pt, f_pt))
    all_pp.append((t_pp, f_pp))
    all_zc.append((t_zc, f_zc))

# Summary
for label, mlist in [('PT', all_pt), ('PP', all_pp), ('ZC', all_zc)]:
    total = sum(len(t) for t, f in mlist)
    print(f"  {label}: {total} total points across 23 filters  "
          f"(avg {total/23:.1f}/filter)")


# ============================================================================
# COLOUR RAMP: dark-blue (low freq) → dark-red (high freq)
# ============================================================================

_cmap = plt.colormaps['coolwarm']
N_FILT = len(specs)
COLORS = [_cmap(i / (N_FILT - 1)) for i in range(N_FILT)]


# ============================================================================
# HELPER: PLOT FVT ON AN AXIS (no smoothing, straight line segments)
# ============================================================================

def plot_fvt(ax, mlist, title, alpha=0.85, connect=True, dot_size=2.5):
    """
    Plot frequency-vs-time for all 23 filters.
    connect=True: join consecutive points with straight line segments.
    connect=False: scatter dots only.
    """
    for i, (t, f) in enumerate(mlist):
        if len(t) == 0:
            continue
        color = COLORS[i]
        label_num = str(i + 1)
        if connect:
            ax.plot(t, f, '-', color=color, linewidth=0.7, alpha=alpha, zorder=3)
            ax.plot(t, f, 'o', color=color, markersize=dot_size, zorder=4)
        else:
            ax.plot(t, f, 'o', color=color, markersize=dot_size, alpha=alpha, zorder=3)
        # Labels at start and end
        if len(t) > 0:
            ax.text(t[0]  - 1.5, f[0],  label_num, fontsize=5, color=color,
                    ha='right', va='center', zorder=5)
            ax.text(t[-1] + 1.5, f[-1], label_num, fontsize=5, color=color,
                    ha='left',  va='center', zorder=5)

    # Gridlines: horizontal at integers, vertical every 25 wks
    for ref in range(8, 13):
        ax.axhline(ref, color='#AAAAAA', linewidth=0.5, zorder=1)
    for wk in range(0, n_weeks + 1, 25):
        ax.axvline(wk, color='#CCCCCC', linewidth=0.35, zorder=1)

    ax.set_xlim(0, max(n_weeks, X_MAX_REF))
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel('WEEKS', fontsize=10)
    ax.set_ylabel('RADIANS/YEAR', fontsize=10)
    ax.set_xticks(range(0, X_MAX_REF + 1, 25))
    ax.set_yticks([8, 9, 10, 11, 12])
    ax.set_title(title, fontsize=10, fontweight='bold')


# ============================================================================
# FIGURE 1: REFERENCE UNDERLAY + PT MEASUREMENTS
# ============================================================================

def overlay_figure(mlist, method_label, out_name, connect=True):
    """Create a figure with reference image underlay and our measurements on top."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 10),
                              gridspec_kw={'wspace': 0.05})

    # Left: our measurements alone
    ax_l = axes[0]
    plot_fvt(ax_l, mlist, f'Our {method_label} Measurements\n(no smoothing, raw)',
             connect=connect)

    # Right: reference image underlay + our measurements
    ax_r = axes[1]
    if os.path.exists(REF_IMAGE):
        ref_img = mpimg.imread(REF_IMAGE)
        # Map reference image to approximate data coordinates.
        # The reference plot area sits inside the image with some margin.
        # Approximate: full image width ≈ x=[0, X_MAX_REF], height ≈ y=[YMIN, YMAX]
        # (adjustment may be needed based on visual inspection)
        ax_r.imshow(ref_img, extent=[0, X_MAX_REF, YMIN, YMAX],
                    aspect='auto', alpha=0.55, zorder=1)
    plot_fvt(ax_r, mlist, f'Reference Underlay + {method_label} Overlay',
             alpha=0.9, connect=connect)

    fig.suptitle(
        f'FREQUENCY VERSUS TIME  —  Figure AI-4\n'
        f'Method: {method_label}  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  '
        f'|  Ormsby FIR  |  No smoothing',
        fontsize=11, fontweight='bold'
    )
    out = os.path.join(SCRIPT_DIR, out_name)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


print("\nGenerating overlay figures...")
overlay_figure(all_pt, 'PT half-period (point at 2nd event)',
               'fig_AI4_v3_overlay_pt.png', connect=True)
overlay_figure(all_pp, 'PP full-period (point at 2nd peak)',
               'fig_AI4_v3_overlay_pp.png', connect=True)
overlay_figure(all_zc, 'ZC half-period (point at midpoint)',
               'fig_AI4_v3_overlay_zc.png', connect=True)


# ============================================================================
# FIGURE 2: ALL THREE METHODS ON ONE PANEL (with reference underlay)
# ============================================================================

print("Generating all-methods comparison on reference underlay...")

fig_all, ax_all = plt.subplots(figsize=(14, 10))

if os.path.exists(REF_IMAGE):
    ref_img = mpimg.imread(REF_IMAGE)
    ax_all.imshow(ref_img, extent=[0, X_MAX_REF, YMIN, YMAX],
                  aspect='auto', alpha=0.50, zorder=1)

# PT: solid dots, blue shades
for i, (t, f) in enumerate(all_pt):
    if len(t) == 0: continue
    ax_all.plot(t, f, '-', color='steelblue', linewidth=0.5, alpha=0.6, zorder=3)
    ax_all.plot(t, f, 'o', color='steelblue', markersize=2.0, alpha=0.7, zorder=4)

# PP: triangles, red shades
for i, (t, f) in enumerate(all_pp):
    if len(t) == 0: continue
    ax_all.plot(t, f, '^', color='firebrick', markersize=2.0, alpha=0.6, zorder=3)

# ZC: crosses, green
for i, (t, f) in enumerate(all_zc):
    if len(t) == 0: continue
    ax_all.plot(t, f, '+', color='forestgreen', markersize=3.0, alpha=0.5, zorder=3,
                markeredgewidth=0.7)

# Gridlines
for ref in range(8, 13):
    ax_all.axhline(ref, color='#888888', linewidth=0.5, zorder=2)
for wk in range(0, X_MAX_REF + 1, 25):
    ax_all.axvline(wk, color='#BBBBBB', linewidth=0.35, zorder=2)

from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color='steelblue', marker='o', markersize=5,
           linestyle='-', label='PT half-period (point at 2nd event)'),
    Line2D([0], [0], color='firebrick', marker='^', markersize=5,
           linestyle='None', label='PP full-period (point at 2nd peak)'),
    Line2D([0], [0], color='forestgreen', marker='+', markersize=5,
           linestyle='None', markeredgewidth=1.0, label='ZC half-period (midpoint)'),
]
ax_all.legend(handles=legend_handles, loc='lower right', fontsize=8, framealpha=0.85)

ax_all.set_xlim(0, max(n_weeks, X_MAX_REF))
ax_all.set_ylim(YMIN, YMAX)
ax_all.set_xlabel('WEEKS', fontsize=11)
ax_all.set_ylabel('RADIANS/YEAR', fontsize=11)
ax_all.set_xticks(range(0, X_MAX_REF + 1, 25))
ax_all.set_yticks([8, 9, 10, 11, 12])
ax_all.set_title(
    'AI-4  |  All Three Methods on Reference Underlay\n'
    'Blue=PT  Red=PP  Green=ZC  |  Reference image at 50% opacity',
    fontsize=10, fontweight='bold'
)

out_all = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_overlay_all.png')
fig_all.savefig(out_all, dpi=150, bbox_inches='tight')
plt.close(fig_all)
print(f"  Saved: {out_all}")

print("\nDone.")
print("\nAlignment inspection notes:")
print("  - If PT dots align with Hurst's lines: PT is the correct method")
print("  - If PP dots align better: full-period is correct")
print("  - If dots are consistently shifted left/right: adjust point placement")
print("    (try t1 midpoint instead of t2, or vice versa)")
print("  - If frequency values are systematically high/low: check clip fraction")
