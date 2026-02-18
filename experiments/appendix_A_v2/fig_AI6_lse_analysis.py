# -*- coding: utf-8 -*-
"""
Figure AI-6: LSE Frequency vs Time Analysis
Appendix A, Figure AI-6 Reproduction

Applies sliding-window 1-mode MPM (Matrix Pencil Method) Prony analysis
to each comb filter output. Each window yields a single frequency estimate
plotted as a short horizontal line segment, reproducing Hurst's
"Smoothing Filtered Outputs" figure.

Dense clustering of line segments around spectral lines emerges from
the stability of frequency estimates when a filter is centered on a
spectral line.

Reference: J.M. Hurst, Appendix A, Figure AI-6, p.197
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt

from utils_ai import (
    load_weekly_data,
    design_comb_bank, make_ormsby_kernels, apply_comb_bank,
    FS_WEEKLY, NW_WEEKLY,
    prony_sliding_lse,
    DATE_ANALYSIS_START, DATE_ANALYSIS_END,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# CONFIGURATION
# ============================================================================

# For AI-6, use a wider time window than AI-3: full analysis period shown as
# weeks from start (week 0 = start of analysis window).
# Hurst's figure shows weeks 20-300 on x-axis.
DISPLAY_WEEK_START = 0
DISPLAY_WEEK_END   = 310

# LSE window size: 1.5 cycles, 50% overlap
N_PERIODS   = 1.5
STEP_FRAC   = 0.5

YMIN, YMAX = 0.0, 12.5
LINE_LW    = 0.6   # line segment linewidth
LINE_ALPHA = 0.65


# ============================================================================
# MAIN
# ============================================================================

print("=" * 70)
print("Figure AI-6: LSE Frequency vs Time Analysis")
print("=" * 70)
print()

print("Loading weekly data...")
close, dates_dt = load_weekly_data()
N = len(close)
print(f"  {N} weekly samples ({DATE_ANALYSIS_START} to {DATE_ANALYSIS_END})")
print()

print("Designing 23 comb filters and applying to full data record...")
specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)
outputs = apply_comb_bank(close, filters, fs=FS_WEEKLY)
print()

# ============================================================================
# RUN SLIDING-WINDOW MPM ON EACH FILTER
# ============================================================================

print("Running sliding-window MPM Prony on all 23 filters...")
print(f"  n_periods={N_PERIODS}, step_frac={STEP_FRAC}")
print()

all_segments = []   # one entry per filter: list of (t_start, t_end, freq)

for i, out in enumerate(outputs):
    spec   = specs[i]
    fc     = spec['f_center']
    sig_cx = out['signal']   # complex analytic

    t_ctr, f_est, t_starts, t_ends = prony_sliding_lse(
        sig_cx, fs=FS_WEEKLY, f_center=fc,
        n_periods=N_PERIODS, step_frac=STEP_FRAC
    )

    # Convert to weeks (the time axis starts at week 0 = first data point)
    all_segments.append({
        'fc': fc,
        'label': spec['label'],
        't_starts': t_starts,   # in samples (= weeks for weekly data)
        't_ends':   t_ends,
        'f_est':    f_est,
    })

    n_seg = len(t_ctr)
    if i < 4 or i >= len(specs) - 2:
        print(f"  FC-{i+1:>2d} (fc={fc:.1f}): {n_seg:>4d} segments  "
              f"f_range=[{f_est.min() if n_seg>0 else 0:.2f}, "
              f"{f_est.max() if n_seg>0 else 0:.2f}] rad/yr")
    elif i == 4:
        print(f"  ...")

print()

# ============================================================================
# PLOT
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 10))

for seg_data in all_segments:
    fc      = seg_data['fc']
    t_s     = seg_data['t_starts']
    t_e     = seg_data['t_ends']
    f_e     = seg_data['f_est']

    if len(f_e) == 0:
        continue

    # Filter to display window
    mask_disp = (t_e >= DISPLAY_WEEK_START) & (t_s <= DISPLAY_WEEK_END)
    t_s_disp = t_s[mask_disp]
    t_e_disp = t_e[mask_disp]
    f_disp   = f_e[mask_disp]

    # Draw each segment as a horizontal line
    for ts, te, f in zip(t_s_disp, t_e_disp, f_disp):
        if YMIN <= f <= YMAX:
            ax.plot([ts, te], [f, f], '-', color='black',
                    linewidth=LINE_LW, alpha=LINE_ALPHA, solid_capstyle='butt')

# Reference frequency grid lines (matching Hurst's y-axis)
for ref_f in np.arange(0, 13, 1):
    ax.axhline(ref_f, color='silver', linewidth=0.3, linestyle='-', zorder=0)

# Vertical grid every 20 weeks
for t_v in np.arange(DISPLAY_WEEK_START, DISPLAY_WEEK_END + 1, 20):
    ax.axvline(t_v, color='silver', linewidth=0.3, linestyle='-', zorder=0)

ax.set_xlim(DISPLAY_WEEK_START, DISPLAY_WEEK_END)
ax.set_ylim(YMIN, YMAX)
ax.set_xlabel('Weeks', fontsize=11)
ax.set_ylabel('Radians / Year', fontsize=11)

# Y-axis ticks matching Hurst's figure
ax.set_yticks(np.arange(0, 13, 1))

# X-axis ticks at 20-week intervals
ax.set_xticks(np.arange(0, DISPLAY_WEEK_END + 1, 20))
ax.tick_params(axis='x', labelsize=8)

ax.set_title('LSE, FREQUENCY VS TIME ANALYSIS\n'
             'FIGURE AI-6  --  Smoothing Filtered Outputs\n'
             '1-Mode MPM Prony  |  Weekly DJIA  |  1921-1965',
             fontsize=11, fontweight='bold', pad=10)

fig.tight_layout()
out_path = os.path.join(SCRIPT_DIR, 'fig_AI6_lse_analysis.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_path}")

# ============================================================================
# ALSO SAVE THE SEGMENT DATA FOR AI-7 USE
# ============================================================================

lse_data_path = os.path.join(SCRIPT_DIR, 'ai6_lse_segments.npz')
np.savez(lse_data_path,
         f_estimates=np.concatenate([s['f_est'] for s in all_segments]),
         filter_centers=np.concatenate([[s['fc']] * len(s['f_est'])
                                        for s in all_segments]),
         allow_pickle=False)
print(f"Saved LSE data: {lse_data_path}")
print()
print("Done.")
