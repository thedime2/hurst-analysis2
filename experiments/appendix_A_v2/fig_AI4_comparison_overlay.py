# -*- coding: utf-8 -*-
"""
Figure AI-4 Comparison: Our reproduction vs Hurst's reference.

Two panels side-by-side:
  Left:  Our fig_AI4_hurst_style.png (zero-crossing, weekly)
  Right: references/appendix_a/figure_AI4_v2.png

Also generates fig_AI4_smoothed.png:
  Our reproduction using peak-to-peak (full period, one point per cycle)
  with a 2-point moving average - closer to Hurst's smoothed presentation.
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
    measure_zerocross_halfperiod,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))

REF_IMAGE   = os.path.join(BASE_DIR, 'references/appendix_a/figure_AI4_v2.png')
OUR_IMAGE   = os.path.join(SCRIPT_DIR, 'fig_AI4_hurst_style.png')

N_DISPLAY  = 23
YMIN, YMAX = 7.4, 12.6
CLIP_FRAC  = 0.28

_cmap  = plt.colormaps['coolwarm']
COLORS = [_cmap(i / (N_DISPLAY - 1)) for i in range(N_DISPLAY)]


# ============================================================================
# SIDE-BY-SIDE COMPARISON
# ============================================================================

fig_cmp, (ax_ours, ax_ref) = plt.subplots(1, 2, figsize=(20, 9))

# Left: our figure
if os.path.exists(OUR_IMAGE):
    img_ours = mpimg.imread(OUR_IMAGE)
    ax_ours.imshow(img_ours)
    ax_ours.axis('off')
    ax_ours.set_title('Our Reproduction\n(Zero-Crossing Half-Period, Weekly Ormsby)',
                      fontsize=11, fontweight='bold')
else:
    ax_ours.text(0.5, 0.5, 'fig_AI4_hurst_style.png\nnot found\n(run fig_AI4_freq_vs_time.py first)',
                 ha='center', va='center', transform=ax_ours.transAxes, fontsize=10)
    ax_ours.axis('off')

# Right: reference
if os.path.exists(REF_IMAGE):
    img_ref = mpimg.imread(REF_IMAGE)
    ax_ref.imshow(img_ref)
    ax_ref.axis('off')
    ax_ref.set_title("Hurst's Original  (Figure AI-4 v2)",
                     fontsize=11, fontweight='bold')
else:
    ax_ref.text(0.5, 0.5, 'Reference image not found', ha='center', va='center',
                transform=ax_ref.transAxes, fontsize=10)
    ax_ref.axis('off')

fig_cmp.suptitle(
    'FIGURE AI-4 COMPARISON: Our Reproduction vs Hurst Original\n'
    'Key differences: (1) Our lines noisier (zero-crossing = 2 pts/cycle vs Hurst ~1 pt/cycle)\n'
    '(2) Hurst\'s figure shows smoothed/hand-drawn lines; (3) Structural groupings match',
    fontsize=10, y=1.01
)
fig_cmp.tight_layout()
out_cmp = os.path.join(SCRIPT_DIR, 'fig_AI4_vs_reference.png')
fig_cmp.savefig(out_cmp, dpi=130, bbox_inches='tight')
plt.close(fig_cmp)
print(f"Saved: {out_cmp}")


# ============================================================================
# SMOOTHED REPRODUCTION: Peak-to-peak + moving-average smoothing
# ============================================================================

def measure_peak_period(signal_real, fs, f_center):
    """Full-period peak-to-peak frequency (one point per cycle)."""
    T_samples = 2 * np.pi / f_center * fs
    min_dist  = max(3, int(T_samples * 0.6))
    peaks, _  = find_peaks(signal_real, distance=min_dist)
    if len(peaks) < 2:
        return np.array([]), np.array([])
    dt = np.diff(peaks) / fs
    freqs = 2 * np.pi / dt
    times = peaks[1:].astype(float)
    return times, freqs


def measure_trough_period(signal_real, fs, f_center):
    """Full-period trough-to-trough frequency (one point per cycle)."""
    T_samples = 2 * np.pi / f_center * fs
    min_dist  = max(3, int(T_samples * 0.6))
    troughs, _ = find_peaks(-signal_real, distance=min_dist)
    if len(troughs) < 2:
        return np.array([]), np.array([])
    dt = np.diff(troughs) / fs
    freqs = 2 * np.pi / dt
    times = troughs[1:].astype(float)
    return times, freqs


def moving_average(arr, n=2):
    """Simple n-point centred moving average."""
    if len(arr) <= n:
        return arr
    result = np.convolve(arr, np.ones(n) / n, mode='valid')
    # Pad to same length (keep original at boundaries)
    pad = (len(arr) - len(result)) // 2
    return np.concatenate([arr[:pad], result, arr[len(arr) - (len(arr) - len(result) - pad):]])


def filter_window(times, freqs, s_idx, e_idx, f_center, samp_per_week=1.0):
    if len(times) == 0:
        return np.array([]), np.array([])
    mask = (times >= s_idx) & (times < e_idx)
    t = (times[mask] - s_idx) / samp_per_week
    f = freqs[mask]
    valid = ((f >= f_center * (1 - CLIP_FRAC)) & (f <= f_center * (1 + CLIP_FRAC)) &
             (f >= YMIN) & (f <= YMAX))
    return t[valid], f[valid]


print("\nLoading weekly data for smoothed figure...")
close_w, dates_w = load_weekly_data()
specs_w   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters_w = make_ormsby_kernels(specs_w, fs=FS_WEEKLY)
outputs_w = apply_comb_bank(close_w, filters_w, fs=FS_WEEKLY)

s_idx, e_idx = get_window(dates_w)
samp_pw      = 1.0   # weekly: 1 sample/week
n_weeks      = (e_idx - s_idx)

fig_sm, axes_sm = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw={'wspace': 0.12})

methods_info = [
    ('Zero-Crossing Half-Period\n(2 pts/cycle, unsmoothed)', 'zc'),
    ('Zero-Crossing  +  2-pt Moving Average\n(closer to Hurst\'s smoothed presentation)', 'sm'),
]

for ax, (title, mode) in zip(axes_sm, methods_info):
    for i, out in enumerate(outputs_w):
        spec = specs_w[i]
        fc   = spec['f_center']
        sig_real = out['signal'].real

        t, f = measure_zerocross_halfperiod(sig_real, FS_WEEKLY)
        t, f = filter_window(t, f, s_idx, e_idx, fc)

        if len(t) == 0:
            continue

        if mode == 'sm' and len(f) >= 3:
            f = moving_average(f, n=3)

        color = COLORS[i]
        ax.plot(t, f, '-o', color=color, markersize=2.0,
                linewidth=0.8, alpha=0.85, zorder=3)

        # Filter labels at both ends
        ax.text(t[0] - 2, f[0], str(i + 1), fontsize=5.5,
                color=color, ha='right', va='center')
        ax.text(t[-1] + 2, f[-1], str(i + 1), fontsize=5.5,
                color=color, ha='left', va='center')

    # Horizontal reference lines
    for ref in [8, 9, 10, 11, 12]:
        ax.axhline(ref, color='silver', linewidth=0.5, zorder=1)

    ax.set_xlim(0, n_weeks)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel('Weeks', fontsize=10)
    ax.set_ylabel('Radians/Year', fontsize=10)
    ax.set_xticks(np.arange(0, n_weeks + 1, 25))
    ax.set_yticks([8, 9, 10, 11, 12])
    ax.grid(True, axis='x', alpha=0.2)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=8)

    # Date annotations
    ax.text(0.01, 0.99, DATE_DISPLAY_START, transform=ax.transAxes,
            fontsize=7, va='top', color='gray')
    ax.text(0.99, 0.99, DATE_DISPLAY_END, transform=ax.transAxes,
            fontsize=7, va='top', ha='right', color='gray')

fig_sm.suptitle(
    'FIGURE AI-4: Effect of Smoothing on Frequency vs Time\n'
    'Ormsby FIR  |  Weekly DJIA  |  All 23 Filters  '
    '|  Hurst\'s original has ~5-10 pts/filter (smooth curves)',
    fontsize=11, fontweight='bold'
)
out_sm = os.path.join(SCRIPT_DIR, 'fig_AI4_smoothed.png')
fig_sm.savefig(out_sm, dpi=150, bbox_inches='tight')
plt.close(fig_sm)
print(f"Saved: {out_sm}")


# ============================================================================
# TRIPLE PANEL: Reference | Our ZC | Our smoothed
# ============================================================================

fig_tri, axes_tri = plt.subplots(1, 3, figsize=(28, 9),
                                  gridspec_kw={'wspace': 0.08})

# Panel 1: Reference image
ax_r = axes_tri[0]
if os.path.exists(REF_IMAGE):
    ax_r.imshow(mpimg.imread(REF_IMAGE))
    ax_r.axis('off')
    ax_r.set_title("Hurst's Original AI-4\n(figure_AI4_v2.png)",
                   fontsize=11, fontweight='bold')

# Panels 2 & 3: our versions (using saved smoothed figure data from above)
for ax, (title, mode) in zip(axes_tri[1:], methods_info):
    for i, out in enumerate(outputs_w):
        spec = specs_w[i]
        fc   = spec['f_center']
        sig_real = out['signal'].real

        t, f = measure_zerocross_halfperiod(sig_real, FS_WEEKLY)
        t, f = filter_window(t, f, s_idx, e_idx, fc)
        if len(t) == 0:
            continue
        if mode == 'sm' and len(f) >= 3:
            f = moving_average(f, n=3)

        color = COLORS[i]
        ax.plot(t, f, '-o', color=color, markersize=1.8,
                linewidth=0.75, alpha=0.85, zorder=3)
        ax.text(t[0] - 2, f[0], str(i + 1), fontsize=5,
                color=color, ha='right', va='center')
        ax.text(t[-1] + 2, f[-1], str(i + 1), fontsize=5,
                color=color, ha='left', va='center')

    for ref in [8, 9, 10, 11, 12]:
        ax.axhline(ref, color='silver', linewidth=0.5, zorder=1)
    ax.set_xlim(0, n_weeks)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel('Weeks', fontsize=10)
    ax.set_ylabel('Radians/Year', fontsize=10)
    ax.set_xticks(np.arange(0, n_weeks + 1, 25))
    ax.set_yticks([8, 9, 10, 11, 12])
    ax.grid(True, axis='x', alpha=0.2)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

fig_tri.suptitle(
    'FIGURE AI-4  |  Hurst Reference vs Reproductions  |  Weekly DJIA Ormsby Comb Filters',
    fontsize=12, fontweight='bold', y=1.01
)
out_tri = os.path.join(SCRIPT_DIR, 'fig_AI4_triple_comparison.png')
fig_tri.savefig(out_tri, dpi=130, bbox_inches='tight')
plt.close(fig_tri)
print(f"Saved: {out_tri}")
print("\nDone.")
