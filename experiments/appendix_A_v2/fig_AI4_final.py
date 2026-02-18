# -*- coding: utf-8 -*-
"""
Figure AI-4 Final: Best Reproduction of Hurst's Frequency Versus Time Plot

Findings from brute-force scheme comparison:
  - PP (peak-to-peak, at 2nd peak) gives ~7 pts/filter (matches Hurst's ~6-10)
  - Parabolic sub-sample interpolation at peaks for sub-weekly precision
  - 3-point centred moving average smoothing matches Hurst's smooth hand-drawn lines
  - TT (trough-to-trough) gives nearly identical result, included for reference

Outputs:
  fig_AI4_final.png            - Hurst-style single panel (PP scheme, smoothed)
  fig_AI4_final_comparison.png - Our best vs Hurst reference side-by-side
  fig_AI4_final_triple.png     - Reference | PP raw | PP smoothed
"""

import sys, os
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
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
REF_IMAGE  = os.path.join(BASE_DIR, 'references/appendix_a/figure_AI4_v2.png')

N_FILTERS = 23
YMIN, YMAX  = 7.4, 12.6
CLIP_FRAC   = 0.30      # ±30% of centre frequency (slightly loose to match Hurst)
SMOOTH_N    = 3         # points for moving average (3 = smooth, 1 = raw)

_cmap  = plt.colormaps['coolwarm']
COLORS = [_cmap(i / (N_FILTERS - 1)) for i in range(N_FILTERS)]


# ============================================================================
# SUB-SAMPLE PEAK/TROUGH DETECTION
# ============================================================================

def parabolic_peak(y, idx):
    """3-point parabolic sub-sample peak position."""
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    y0, y1, y2 = float(y[idx-1]), float(y[idx]), float(y[idx+1])
    denom = y0 - 2.0*y1 + y2
    if abs(denom) < 1e-14:
        return float(idx)
    delta = 0.5 * (y0 - y2) / denom
    return idx + np.clip(delta, -1.0, 1.0)


def find_peaks_sub(signal, f_center, fs, min_dist_frac=0.55):
    """Peaks with distance guard + parabolic interpolation."""
    T_samp = 2 * np.pi / f_center * fs
    min_d  = max(3, int(T_samp * min_dist_frac))
    idx, _ = find_peaks(signal, distance=min_d)
    return np.array([parabolic_peak(signal, i) for i in idx])


def find_troughs_sub(signal, f_center, fs, min_dist_frac=0.55):
    """Troughs with sub-sample interpolation."""
    return find_peaks_sub(-signal, f_center, fs, min_dist_frac)


# ============================================================================
# MEASUREMENT SCHEMES
# ============================================================================

def scheme_PP(peaks, fs):
    """Peak-to-peak full period, placed at 2nd peak."""
    if len(peaks) < 2:
        return np.array([]), np.array([])
    dt    = np.diff(peaks) / fs
    freqs = 2 * np.pi / dt
    times = peaks[1:]
    return times, freqs


def scheme_TT(troughs, fs):
    """Trough-to-trough full period, placed at 2nd trough."""
    return scheme_PP(troughs, fs)


def scheme_PP_TT(peaks, troughs, fs):
    """PP and TT merged chronologically (gives zig-zag at ~2 pts/cycle)."""
    tp, fp = scheme_PP(peaks, fs)
    tt, ft = scheme_TT(troughs, fs)
    if len(tp) == 0 and len(tt) == 0:
        return np.array([]), np.array([])
    all_t = np.concatenate([tp, tt])
    all_f = np.concatenate([fp, ft])
    order = np.argsort(all_t)
    return all_t[order], all_f[order]


def scheme_PT_TP(peaks, troughs, fs):
    """
    PT+TP interleaved: all peaks and troughs in chronological order,
    consecutive half-period measured between adjacent events.
    Frequency = pi / half_period.  Placed at 2nd event of each pair.
    Gives ~2 pts/cycle with natural zig-zag from AM.
    """
    events = ([(t, 'P') for t in peaks] +
              [(t, 'T') for t in troughs])
    events.sort(key=lambda x: x[0])
    if len(events) < 2:
        return np.array([]), np.array([])
    times, freqs = [], []
    for k in range(len(events) - 1):
        t1, _ = events[k]
        t2, _ = events[k + 1]
        dt = (t2 - t1) / fs
        if dt > 0:
            freqs.append(np.pi / dt)
            times.append(t2)
    return np.array(times), np.array(freqs)


# ============================================================================
# CLIP AND SMOOTH
# ============================================================================

def clip_to_window(times, freqs, s_idx, e_idx, f_center):
    """Convert to weeks-from-display-start, apply frequency clipping."""
    if len(times) == 0:
        return np.array([]), np.array([])
    mask = (times >= s_idx) & (times < e_idx)
    t = times[mask] - s_idx
    f = freqs[mask]
    valid = ((f >= f_center * (1 - CLIP_FRAC)) & (f <= f_center * (1 + CLIP_FRAC)) &
             (f >= YMIN) & (f <= YMAX))
    return t[valid], f[valid]


def smooth_ma(t, f, n):
    """Centred n-point moving average; returns (t_trimmed, f_smoothed)."""
    if n <= 1 or len(f) < n:
        return t, f
    kernel  = np.ones(n) / n
    f_sm    = np.convolve(f, kernel, mode='valid')
    pad     = (len(f) - len(f_sm)) // 2
    t_sm    = t[pad: pad + len(f_sm)]
    return t_sm, f_sm


# ============================================================================
# LOAD DATA AND APPLY FILTERS
# ============================================================================

print("Loading weekly DJIA data (all CSV rows)...")
close, dates = load_weekly_data()
specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)
outputs = apply_comb_bank(close, filters, fs=FS_WEEKLY)

s_idx, e_idx = get_window(dates)
n_weeks = e_idx - s_idx
print(f"Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} weeks)")

# ============================================================================
# COMPUTE MEASUREMENTS FOR ALL FILTERS
# ============================================================================

measurements = []  # each entry: {times_raw, freqs_raw, times_sm, freqs_sm, center, idx}
meas_pttp  = []  # PT_TP interleaved (2 pts/cycle, zig-zag)

for i, out in enumerate(outputs):
    fc       = specs[i]['f_center']
    sig_real = out['signal'].real

    pk = find_peaks_sub(sig_real, fc, FS_WEEKLY)
    tr = find_troughs_sub(sig_real, fc, FS_WEEKLY)

    # Primary: PP scheme
    t_raw, f_raw = scheme_PP(pk, FS_WEEKLY)
    t_raw, f_raw = clip_to_window(t_raw, f_raw, s_idx, e_idx, fc)
    t_sm, f_sm = smooth_ma(t_raw, f_raw, SMOOTH_N)
    measurements.append({
        'times_raw': t_raw, 'freqs_raw': f_raw,
        'times_sm':  t_sm,  'freqs_sm':  f_sm,
        'center':    fc,
        'idx':       i,
    })

    # Alternative: PT_TP interleaved (captures zig-zag from AM/beating)
    t_pt, f_pt = scheme_PT_TP(pk, tr, FS_WEEKLY)
    t_pt, f_pt = clip_to_window(t_pt, f_pt, s_idx, e_idx, fc)
    t_pt_sm, f_pt_sm = smooth_ma(t_pt, f_pt, SMOOTH_N)
    meas_pttp.append({
        'times_raw': t_pt,    'freqs_raw': f_pt,
        'times_sm':  t_pt_sm, 'freqs_sm':  f_pt_sm,
        'center':    fc,
        'idx':       i,
    })

total_pts   = sum(len(m['times_raw']) for m in measurements)
total_sm    = sum(len(m['times_sm'])  for m in measurements)
total_pttp  = sum(len(m['times_raw']) for m in meas_pttp)
total_pt_sm = sum(len(m['times_sm'])  for m in meas_pttp)

# Also build 5-pt smoothed PT_TP (best Hurst match: ~12 pts/filter + variation)
SMOOTH_N5 = 5
meas_pttp_5 = []
for m in meas_pttp:
    t5, f5 = smooth_ma(m['times_raw'], m['freqs_raw'], SMOOTH_N5)
    meas_pttp_5.append({
        'times_raw': m['times_raw'], 'freqs_raw': m['freqs_raw'],
        'times_sm':  t5,             'freqs_sm':  f5,
        'center':    m['center'],    'idx':        m['idx'],
    })
total_pt5 = sum(len(m['times_sm']) for m in meas_pttp_5)

print(f"\nPP raw:              {total_pts} pts total ({total_pts // N_FILTERS} avg/filter)")
print(f"PP {SMOOTH_N}-pt MA:       {total_sm}  pts total ({total_sm  // N_FILTERS} avg/filter)")
print(f"PT_TP raw:           {total_pttp} pts total ({total_pttp // N_FILTERS} avg/filter)")
print(f"PT_TP {SMOOTH_N}-pt MA:    {total_pt_sm} pts total ({total_pt_sm // N_FILTERS} avg/filter)")
print(f"PT_TP {SMOOTH_N5}-pt MA:    {total_pt5} pts total ({total_pt5 // N_FILTERS} avg/filter)  <-- BEST")
print()


# ============================================================================
# HELPER: DRAW ONE FVT PANEL
# ============================================================================

def draw_fvt(ax, measurements, use_smooth=True, show_labels=True,
             markersize=2.0, linewidth=0.8, alpha=0.90):
    for m in measurements:
        t = m['times_sm'] if use_smooth else m['times_raw']
        f = m['freqs_sm'] if use_smooth else m['freqs_raw']
        if len(t) == 0:
            continue
        color = COLORS[m['idx']]
        ax.plot(t, f, '-o', color=color,
                markersize=markersize, linewidth=linewidth, alpha=alpha, zorder=3)
        if show_labels:
            ax.text(t[0] - 2, f[0],  str(m['idx'] + 1), fontsize=5.5,
                    color=color, ha='right', va='center')
            ax.text(t[-1] + 2, f[-1], str(m['idx'] + 1), fontsize=5.5,
                    color=color, ha='left',  va='center')

    for ref in [8, 9, 10, 11, 12]:
        ax.axhline(ref, color='silver', linewidth=0.5, zorder=1)

    ax.set_xlim(0, n_weeks)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(np.arange(0, n_weeks + 1, 25))
    ax.set_yticks([8, 9, 10, 11, 12])
    ax.set_xlabel('Weeks', fontsize=10)
    ax.set_ylabel('Radians/Year', fontsize=10)
    ax.grid(True, axis='x', alpha=0.2)
    ax.text(0.01, 0.99, DATE_DISPLAY_START, transform=ax.transAxes,
            fontsize=7, va='top', color='gray')
    ax.text(0.99, 0.99, DATE_DISPLAY_END, transform=ax.transAxes,
            fontsize=7, va='top', ha='right', color='gray')


# ============================================================================
# FIGURE 1: HURST-STYLE SINGLE PANEL  (PT_TP + 5pt MA - best Hurst match)
# ============================================================================
# Uses PT+TP half-period interleaved with 5-pt moving average.
# This captures the frequency drift and filter crossings visible in Hurst's
# original figure, while keeping density (~12 pts/filter) close to Hurst's.

fig1, ax1 = plt.subplots(figsize=(11, 7))
draw_fvt(ax1, meas_pttp_5, use_smooth=True, show_labels=True)
ax1.set_title(
    f'FIGURE A I-4  |  Frequency Versus Time\n'
    f'Weekly DJIA  |  23 Ormsby Comb Filters  '
    f'|  PT+TP half-period + {SMOOTH_N5}-pt MA ({total_pt5 // N_FILTERS} avg/filter)',
    fontsize=11, fontweight='bold', pad=10
)
fig1.tight_layout()
out1 = os.path.join(SCRIPT_DIR, 'fig_AI4_final.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"Saved: {out1}")


# ============================================================================
# FIGURE 2: SIDE-BY-SIDE COMPARISON  (our smoothed vs Hurst reference)
# ============================================================================

fig2, (ax_our, ax_ref) = plt.subplots(1, 2, figsize=(20, 9))

# Left: our best reproduction (PT_TP + 5pt MA)
draw_fvt(ax_our, meas_pttp_5, use_smooth=True)
ax_our.set_title(
    f'Our Reproduction\n(PT+TP half-period  +  {SMOOTH_N5}-pt MA)',
    fontsize=11, fontweight='bold'
)

# Right: Hurst reference
if os.path.exists(REF_IMAGE):
    ax_ref.imshow(mpimg.imread(REF_IMAGE))
ax_ref.axis('off')
ax_ref.set_title("Hurst's Original (Figure AI-4 v2)", fontsize=11, fontweight='bold')

fig2.suptitle(
    'FIGURE AI-4 COMPARISON  |  Our PP-Smoothed Reproduction vs Hurst Original\n'
    'Weekly DJIA  |  Ormsby FIR  |  Parabolic sub-sample interpolation at peaks',
    fontsize=10, y=1.01
)
fig2.tight_layout()
out2 = os.path.join(SCRIPT_DIR, 'fig_AI4_final_comparison.png')
fig2.savefig(out2, dpi=130, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: {out2}")


# ============================================================================
# FIGURE 3: TRIPLE PANEL  (Reference | PP raw | PP smoothed)
# ============================================================================

fig3, axes3 = plt.subplots(1, 3, figsize=(28, 9),
                            gridspec_kw={'wspace': 0.10})

# Panel 1: Hurst reference
if os.path.exists(REF_IMAGE):
    axes3[0].imshow(mpimg.imread(REF_IMAGE))
axes3[0].axis('off')
axes3[0].set_title("Hurst's Original AI-4", fontsize=10, fontweight='bold')

# Panel 2: PP raw (no smoothing)
draw_fvt(axes3[1], measurements, use_smooth=False, markersize=1.8)
axes3[1].set_title(
    f'PP raw (unsmoothed)\n~{total_pts // N_FILTERS} pts/filter',
    fontsize=10, fontweight='bold'
)

# Panel 3: PP smoothed
draw_fvt(axes3[2], measurements, use_smooth=True, markersize=1.8)
axes3[2].set_title(
    f'PP + {SMOOTH_N}-pt moving average\n~{total_sm // N_FILTERS} pts/filter',
    fontsize=10, fontweight='bold'
)

fig3.suptitle(
    'FIGURE AI-4  |  Hurst Reference vs PP Scheme (Raw and Smoothed)\n'
    'Weekly DJIA  |  Ormsby FIR  |  Parabolic sub-sample interpolation',
    fontsize=11, fontweight='bold', y=1.01
)
out3 = os.path.join(SCRIPT_DIR, 'fig_AI4_final_triple.png')
fig3.savefig(out3, dpi=130, bbox_inches='tight')
plt.close(fig3)
print(f"Saved: {out3}")


# ============================================================================
# FIGURE 4: FOUR-PANEL COMPARISON
# Reference | PP raw | PP smoothed | PT_TP smoothed
# ============================================================================

fig4, axes4 = plt.subplots(1, 4, figsize=(36, 9),
                            gridspec_kw={'wspace': 0.12})

# Panel 1: Hurst reference
if os.path.exists(REF_IMAGE):
    axes4[0].imshow(mpimg.imread(REF_IMAGE))
axes4[0].axis('off')
axes4[0].set_title("Hurst's Original AI-4", fontsize=10, fontweight='bold')

# Panel 2: PP raw
draw_fvt(axes4[1], measurements, use_smooth=False, markersize=1.8, linewidth=0.7)
axes4[1].set_title(
    f'A: PP raw (~{total_pts // N_FILTERS} pts/filter)\n1 pt per cycle, no smoothing',
    fontsize=9.5, fontweight='bold'
)

# Panel 3: PP smoothed
draw_fvt(axes4[2], measurements, use_smooth=True, markersize=1.8, linewidth=0.8)
axes4[2].set_title(
    f'A: PP + {SMOOTH_N}-pt MA (~{total_sm // N_FILTERS} pts/filter)\nSmoothed full-period',
    fontsize=9.5, fontweight='bold'
)

# Panel 4: PT_TP with 5-pt smoothing (Hurst density + zig-zag)
draw_fvt(axes4[3], meas_pttp_5, use_smooth=True, markersize=1.5, linewidth=0.65, alpha=0.80)
axes4[3].set_title(
    f'E: PT+TP + {SMOOTH_N5}-pt MA (~{total_pt5 // N_FILTERS} pts/filter)\nHalf-period, zig-zag, Hurst density',
    fontsize=9.5, fontweight='bold'
)

fig4.suptitle(
    'FIGURE AI-4  |  Four-Panel Scheme Comparison vs Hurst Reference\n'
    'Weekly DJIA  |  Ormsby FIR  |  Parabolic sub-sample interpolation at peaks/troughs',
    fontsize=11, fontweight='bold', y=1.01
)
out4 = os.path.join(SCRIPT_DIR, 'fig_AI4_final_4panel.png')
fig4.savefig(out4, dpi=130, bbox_inches='tight')
plt.close(fig4)
print(f"Saved: {out4}")


print("\nDone.")
print()
print("Key findings:")
print(f"  PP raw:         {total_pts} pts ({total_pts // N_FILTERS} avg/filter) - within Hurst's range")
print(f"  PP {SMOOTH_N}-pt MA:    {total_sm} pts ({total_sm // N_FILTERS} avg/filter) - smoothest")
print(f"  PT_TP raw:      {total_pttp} pts ({total_pttp // N_FILTERS} avg/filter) - max detail")
print(f"  PT_TP {SMOOTH_N}-pt MA: {total_pt_sm} pts ({total_pt_sm // N_FILTERS} avg/filter) - zig-zag preserved")
print(f"  Hurst's original: ~7-10 pts/filter (estimated from figure density)")
print(f"  Best structural match: PP raw (7 avg/filter)")
print(f"  Best zig-zag match:   PT_TP smoothed (shows AM/beating frequency variation)")
