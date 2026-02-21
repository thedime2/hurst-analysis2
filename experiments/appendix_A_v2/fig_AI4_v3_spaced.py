# -*- coding: utf-8 -*-
"""
Figure AI-4 v3: Spaced Filter Bank Comparison

Tests different filter spacing strategies to see which best reproduces
Hurst's AI-4 Frequency-vs-Time figure. The hypothesis is that wider-spaced
filters (especially harmonic-aligned at 0.3676 rad/yr intervals) produce
cleaner FVT measurements by eliminating gap filters that fall in
anti-resonance zones.

Filter spacings tested:
  0.20 rad/yr — current (adjacent, heavily overlapping skirts)
  0.37 rad/yr — Hurst harmonic spacing (0.3676 rad/yr, rounded)
  0.40 rad/yr — moderate gap between filters
  0.60 rad/yr — wide gap (every 3rd harmonic roughly)

For each spacing:
  - Design new comb bank with same passband/skirt widths but wider step
  - Apply Ormsby FIR filters (analytic=True, method='modulate')
  - Measure FVT using PT interleaved (point at second event, no smoothing)
  - Plot as: (a) dots only, (b) connected line segments

Figures produced:
  fig_AI4_v3_spaced_grid.png      — 4×2 grid: 4 spacings × (dots | lines)
  fig_AI4_v3_spaced_harmonic.png  — Single panel: harmonic-aligned spacing, lines
  fig_AI4_v3_spaced_compare.png   — 4 spacings on one axis (lines, different colors)

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figure AI-4, p.194
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from src.filters import design_hurst_comb_bank, create_filter_kernels, apply_filter_bank

from utils_ai import (
    load_weekly_data,
    get_window, FS_WEEKLY, NW_WEEKLY,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed filter bank parameters (only step varies)
W1_START       = 7.2     # lower skirt edge of first filter (rad/yr)
PASSBAND_WIDTH = 0.2     # flat passband (rad/yr)
SKIRT_WIDTH    = 0.3     # transition band each side (rad/yr)
N_FILTERS_MAX  = 23      # max number of filters (fewer if step is wider)

YMIN, YMAX   = 7.5, 12.5     # y-axis limits (rad/yr)
CLIP_FRAC    = 0.30           # accept measurements within ±30 % of filter centre

# Spacings to test
SPACINGS = [0.20, 0.3676, 0.40, 0.60]
SPACING_LABELS = ['0.20 r/y (current, adjacent)', '0.3676 r/y (harmonic spacing)',
                  '0.40 r/y (moderate gap)', '0.60 r/y (wide gap)']


# ============================================================================
# MEASUREMENT UTILITIES (identical to v3_overlay: raw PT, no smoothing)
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
    PT interleaved: point placed at time of second event, frequency = π/Δt_half.
    Skips same-type consecutive events.
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
    t     = (times[mask] - s_idx).astype(float)   # convert to weeks (weekly data)
    f     = freqs[mask]
    valid = ((f >= f_center * (1 - CLIP_FRAC)) & (f <= f_center * (1 + CLIP_FRAC)) &
             (f >= YMIN) & (f <= YMAX))
    return t[valid], f[valid]


# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading weekly DJIA data...")
close, dates_dt = load_weekly_data()
s_idx, e_idx = get_window(dates_dt)
n_weeks = e_idx - s_idx
print(f"Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} weeks)")


# ============================================================================
# DESIGN, FILTER, AND MEASURE FOR EACH SPACING
# ============================================================================

all_results = {}   # spacing -> list of (t_wk, f_radyr) per filter

for step in SPACINGS:
    print(f"\nSpacing = {step:.4f} rad/yr")

    # How many filters fit between W1_START centre and 12.0 rad/yr?
    # Centre of filter k (0-indexed): fc_k = W1_START + W_STEP_half + k * step
    # where W_STEP_half = SKIRT_WIDTH + PASSBAND_WIDTH/2 (offset from w1 to centre)
    fc_start = W1_START + SKIRT_WIDTH + PASSBAND_WIDTH / 2.0
    n_filters = max(1, int((12.0 - fc_start) / step) + 1)
    n_filters = min(n_filters, N_FILTERS_MAX)

    print(f"  fc_start = {fc_start:.2f} r/y, n_filters = {n_filters}")

    specs = design_hurst_comb_bank(
        n_filters=n_filters,
        w1_start=W1_START,
        w_step=step,
        passband_width=PASSBAND_WIDTH,
        skirt_width=SKIRT_WIDTH,
        nw=NW_WEEKLY,
        fs=FS_WEEKLY,
    )

    centres = [round(s['f_center'], 2) for s in specs]
    print(f"  Filter centres: {centres}")

    # Analytic Ormsby kernels (modulate method, analytic=True)
    filters = create_filter_kernels(specs, fs=FS_WEEKLY, filter_type='modulate',
                                    analytic=True)

    result  = apply_filter_bank(close, filters, fs=FS_WEEKLY)
    outputs = result['filter_outputs']

    meas_list = []
    for i, out in enumerate(outputs):
        fc  = specs[i]['f_center']
        sig = out['signal'].real
        t, f = measure_pt(sig, fc, FS_WEEKLY)
        t, f = clip_to_window(t, f, s_idx, e_idx, fc)
        meas_list.append((t, f, fc))

    n_pts = sum(len(t) for t, f, _ in meas_list)
    print(f"  Total measurements: {n_pts}  (avg {n_pts/len(meas_list):.1f}/filter)")

    all_results[step] = meas_list


# ============================================================================
# COLOUR HELPERS
# ============================================================================

def color_ramp(n):
    cmap = plt.colormaps['coolwarm']
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# ============================================================================
# PLOT HELPER: FVT PANEL
# ============================================================================

def plot_panel(ax, meas_list, title, connect=True, dot_size=2.0, alpha=0.80):
    """Plot FVT for one filter bank. meas_list: list of (t, f, fc)."""
    colors = color_ramp(len(meas_list))
    for i, (t, f, fc) in enumerate(meas_list):
        if len(t) == 0:
            continue
        c = colors[i]
        if connect:
            ax.plot(t, f, '-', color=c, linewidth=0.7, alpha=alpha, zorder=3)
        ax.plot(t, f, 'o', color=c, markersize=dot_size, alpha=alpha, zorder=4)
        if len(t) > 0:
            ax.text(t[0]  - 2, f[0],  f'{i+1}', fontsize=5, color=c,
                    ha='right', va='center')
            ax.text(t[-1] + 2, f[-1], f'{i+1}', fontsize=5, color=c,
                    ha='left',  va='center')

    for ref in range(8, 13):
        ax.axhline(ref, color='#BBBBBB', linewidth=0.5, zorder=1)
    for wk in range(0, n_weeks + 1, 25):
        ax.axvline(wk, color='#DDDDDD', linewidth=0.35, zorder=1)

    ax.set_xlim(0, n_weeks)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel('Weeks', fontsize=8)
    ax.set_ylabel('rad/yr', fontsize=8)
    ax.set_xticks(range(0, n_weeks + 1, 50))
    ax.set_yticks([8, 9, 10, 11, 12])
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8, fontweight='bold')


# ============================================================================
# FIGURE 1: 4×2 GRID (4 spacings × dots | lines)
# ============================================================================

print("\nGenerating Figure 1: 4×2 grid (spacings × dots/lines)...")

fig1, axes1 = plt.subplots(4, 2, figsize=(22, 24),
                            gridspec_kw={'wspace': 0.15, 'hspace': 0.38})

for row, step in enumerate(SPACINGS):
    mlist = all_results[step]
    n_f   = len(mlist)
    lbl   = SPACING_LABELS[row]

    # Left column: dots only
    plot_panel(axes1[row, 0], mlist,
               f'Step={step:.4f} r/y  |  {n_f} filters  |  DOTS only',
               connect=False, dot_size=2.5)

    # Right column: connected lines
    plot_panel(axes1[row, 1], mlist,
               f'Step={step:.4f} r/y  |  {n_f} filters  |  LINES (PT, no smoothing)',
               connect=True, dot_size=1.5)

fig1.suptitle(
    f'AI-4 Spaced Filter Comparison  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}\n'
    'Ormsby FIR (analytic, modulate)  |  PT measurements  |  Point at 2nd event\n'
    'Passband=0.2 r/y, Skirt=0.3 r/y  (only step varies)',
    fontsize=11, fontweight='bold'
)
out1 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_spaced_grid.png')
fig1.savefig(out1, dpi=120, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved: {out1}")


# ============================================================================
# FIGURE 2: HARMONIC-ALIGNED SPACING SINGLE PANEL
# ============================================================================

print("Generating Figure 2: Harmonic-aligned spacing (0.3676 r/y)...")

harm_step = 0.3676
mlist_harm = all_results[harm_step]

fig2, ax2 = plt.subplots(figsize=(13, 9))
plot_panel(ax2, mlist_harm,
           f'AI-4  |  Harmonic-Aligned Filters (step={harm_step} r/y)\n'
           f'PT measurements  |  Point at 2nd event  |  No smoothing',
           connect=True, dot_size=2.0)

# Right-side labels (y-axis twin)
ax2r = ax2.twinx()
ax2r.set_ylim(YMIN, YMAX)
ax2r.set_yticks([8, 9, 10, 11, 12])
ax2r.tick_params(labelsize=9)
ax2r.set_ylabel('RADIANS/YEAR', fontsize=10)

ax2.set_xlabel('WEEKS', fontsize=11)
ax2.set_ylabel('RADIANS/YEAR', fontsize=11)
ax2.set_xticks(range(0, n_weeks + 1, 25))
ax2.set_xticklabels([str(x) if x <= 250 else '' for x in range(0, n_weeks + 1, 25)])

fig2.tight_layout()
out2 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_spaced_harmonic.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: {out2}")


# ============================================================================
# FIGURE 3: ALL 4 SPACINGS ON ONE AXIS (lines, different opacity/color)
# ============================================================================

print("Generating Figure 3: All 4 spacings overlaid...")

fig3, ax3 = plt.subplots(figsize=(14, 9))

step_colors  = ['steelblue', 'firebrick', 'forestgreen', 'darkorange']
step_alphas  = [0.35, 0.65, 0.65, 0.65]
step_lws     = [0.5, 0.8, 0.8, 0.8]

for j, step in enumerate(SPACINGS):
    mlist = all_results[step]
    for t, f, fc in mlist:
        if len(t) == 0:
            continue
        ax3.plot(t, f, '-', color=step_colors[j], linewidth=step_lws[j],
                 alpha=step_alphas[j], zorder=j+2)
        ax3.plot(t, f, 'o', color=step_colors[j], markersize=1.5,
                 alpha=step_alphas[j] + 0.1, zorder=j+2)

for ref in range(8, 13):
    ax3.axhline(ref, color='#AAAAAA', linewidth=0.5, zorder=1)
for wk in range(0, n_weeks + 1, 25):
    ax3.axvline(wk, color='#CCCCCC', linewidth=0.35, zorder=1)

from matplotlib.lines import Line2D
leg_handles = [
    Line2D([0], [0], color=c, linewidth=1.5, label=f'Step={s:.4f} — {l}')
    for s, l, c in zip(SPACINGS, SPACING_LABELS, step_colors)
]
ax3.legend(handles=leg_handles, loc='lower right', fontsize=8, framealpha=0.90)

ax3.set_xlim(0, n_weeks)
ax3.set_ylim(YMIN, YMAX)
ax3.set_xlabel('WEEKS', fontsize=11)
ax3.set_ylabel('RADIANS/YEAR', fontsize=11)
ax3.set_xticks(range(0, n_weeks + 1, 25))
ax3.set_xticklabels([str(x) if x <= 250 else '' for x in range(0, n_weeks + 1, 25)])
ax3.set_yticks([8, 9, 10, 11, 12])
ax3.set_title(
    'AI-4  |  All Spacings Overlaid  |  PT measurements  |  No smoothing\n'
    f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}  '
    '|  Passband=0.2, Skirt=0.3 r/y (fixed)',
    fontsize=10, fontweight='bold'
)

out3 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_spaced_compare.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"  Saved: {out3}")


# ============================================================================
# SUMMARY TABLE
# ============================================================================

print()
print("=" * 60)
print("Spaced Filter Comparison Summary")
print("=" * 60)
print(f"{'Step':>10}  {'Filters':>8}  {'Total pts':>10}  {'Avg/filter':>10}")
for step in SPACINGS:
    mlist = all_results[step]
    n_f   = len(mlist)
    n_pts = sum(len(t) for t, f, _ in mlist)
    print(f"{step:>10.4f}  {n_f:>8d}  {n_pts:>10d}  {n_pts/n_f:>10.1f}")
print()
print("Done.")
