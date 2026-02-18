# -*- coding: utf-8 -*-
"""
Figure AI-5: Modulation Sidebands
Appendix A, Figure AI-5 Reproduction

Shows 6 frequency bands as shaded regions demonstrating amplitude modulation
of instantaneous frequency around the spectral line centers.

The shaded "sideband" envelope at each center frequency shows how the
measured instantaneous frequency wanders above and below the nominal
center over the display window.

Center frequencies (from Hurst AI-5):
  11.8, 11.0, 10.2, 9.4, 8.6, 7.8 rad/yr
  (T = 27.7, 29.7, 32.0, 34.7, 38.0, 41.9 weeks)

Method:
  1. For each of the 6 nominal filters, apply Ormsby and collect all
     zero-crossing half-period frequency measurements over the full
     analysis window (not just display window).
  2. Interpolate to a common time grid.
  3. Show the time-varying instantaneous frequency as a filled region
     (above and below the center line) with diagonal hatching.

Reference: J.M. Hurst, Appendix A, Figure AI-5
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

from utils_ai import (
    load_weekly_data,
    design_comb_bank, make_ormsby_kernels, apply_comb_bank,
    get_window, FS_WEEKLY, NW_WEEKLY,
    measure_zerocross_halfperiod,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Hurst's AI-5 center frequencies and corresponding T_weeks
# (matched to nearest filter in the 23-filter comb bank)
AI5_CENTERS_RADYR = [11.8, 11.0, 10.2, 9.4, 8.6, 7.8]   # top to bottom
AI5_PERIODS_WK    = [2 * np.pi / f * FS_WEEKLY for f in AI5_CENTERS_RADYR]

# Smoothing window for envelope (in weeks)
SMOOTH_WK = 20


def find_nearest_filter(specs, target_f):
    """Return index of filter with center closest to target_f."""
    centers = np.array([s['f_center'] for s in specs])
    return int(np.argmin(np.abs(centers - target_f)))


# ============================================================================
# MAIN
# ============================================================================

print("=" * 70)
print("Figure AI-5: Modulation Sidebands")
print("=" * 70)

print("Loading weekly data and applying comb filters...")
close, dates_dt = load_weekly_data()
specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)
outputs = apply_comb_bank(close, filters, fs=FS_WEEKLY)

s_idx, e_idx = get_window(dates_dt)
n_disp = e_idx - s_idx
weeks_full = np.arange(len(close))   # full time axis in weeks

print(f"  Display: {n_disp} weeks ({DATE_DISPLAY_START} to {DATE_DISPLAY_END})")
print()

# ============================================================================
# COLLECT FREQUENCY MEASUREMENTS FOR EACH OF THE 6 BANDS
# ============================================================================

# Dense time grid for interpolation (over display window)
t_grid = np.linspace(s_idx, e_idx - 1, n_disp * 4)   # 4x oversampled
t_grid_weeks = (t_grid - s_idx)                        # 0-based weeks

band_data = []
for target_f in AI5_CENTERS_RADYR:
    fi = find_nearest_filter(specs, target_f)
    spec = specs[fi]
    fc = spec['f_center']

    out = outputs[fi]
    sig_real = out['signal'].real

    # Zero-crossing measurements over FULL signal (for smoother envelope)
    times_raw, freqs_raw = measure_zerocross_halfperiod(sig_real, FS_WEEKLY)

    # Filter to display window + some margin (±50 weeks)
    margin = 50
    mask = (times_raw >= s_idx - margin) & (times_raw <= e_idx + margin)
    times_w = times_raw[mask]
    freqs_w = freqs_raw[mask]

    # Sanity filter: keep within ±40% of center
    valid = (freqs_w > fc * 0.6) & (freqs_w < fc * 1.4)
    times_w = times_w[valid]
    freqs_w = freqs_w[valid]

    if len(times_w) < 3:
        print(f"  WARNING: Too few measurements for fc={fc:.1f}")
        band_data.append(None)
        continue

    # Interpolate frequency deviation (f - fc) onto the dense grid
    try:
        interp_fn = interp1d(times_w, freqs_w - fc,
                             kind='linear', bounds_error=False,
                             fill_value=(freqs_w[0] - fc, freqs_w[-1] - fc))
        delta_f_grid = interp_fn(t_grid)
    except Exception:
        delta_f_grid = np.zeros(len(t_grid))

    # Smooth the deviation (moving average)
    smooth_samp = max(1, SMOOTH_WK * 4)   # 4x oversampled
    delta_smooth = uniform_filter1d(delta_f_grid, size=smooth_samp)

    # Also compute the raw (less smoothed) version for the fine detail
    raw_smooth = uniform_filter1d(delta_f_grid, size=max(1, int(SMOOTH_WK / 2) * 4))

    T_wk = 2 * np.pi / fc * FS_WEEKLY
    print(f"  fc={fc:.1f} rad/yr  T={T_wk:.1f}wk  "
          f"  filter=FC-{fi+1}  n_meas={len(times_w)}"
          f"  delta_range=[{freqs_w.min()-fc:+.2f}, {freqs_w.max()-fc:+.2f}]")

    band_data.append({
        'fc': fc,
        'T_wk': T_wk,
        'filter_idx': fi,
        't_grid_weeks': t_grid_weeks,
        'delta_smooth': delta_smooth,
        'raw_smooth': raw_smooth,
        'times_raw_w': times_w - s_idx,     # relative to display start
        'freqs_raw': freqs_w,
    })

print()

# ============================================================================
# PLOT
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 18))

hatch_styles = ['///', '\\\\\\', '///', '\\\\\\', '///', '///']
band_colors  = ['#2060A0', '#2060A0', '#2060A0', '#2060A0', '#2060A0', '#2060A0']
# Slightly different shades for alternating bands
band_colors = ['#1a5fa0', '#1f5f9e', '#2461a2', '#2060a0', '#1c5e9e', '#1858a0']

for band_idx, (bd, hatch_style, bcolor) in enumerate(
        zip(band_data, hatch_styles, band_colors)):
    if bd is None:
        continue

    fc   = bd['fc']
    T_wk = bd['T_wk']
    tw   = bd['t_grid_weeks']
    dr   = bd['raw_smooth']      # raw deviation (fine detail)
    ds   = bd['delta_smooth']    # smoothed deviation (envelope)

    # Vertical offset: each band centered at 0 in data coords,
    # but we stack them. We'll use fc directly as the Y position.

    # Center horizontal line (thin solid black)
    ax.axhline(fc, color='black', linewidth=0.8, linestyle='-', zorder=3)

    # Shaded sideband: fill between fc+dr_top and fc+dr_bot
    # Use the raw_smooth for the zig-zag detail (like Hurst's figure)
    upper = fc + dr
    lower = fc - dr   # symmetric shading about center

    # In Hurst's figure the shading is NOT symmetric -- it shows the
    # actual f(t) deviating above and below the center line.
    # So the shaded region spans from min(fc, fc+dr) to max(fc, fc+dr)
    # at each time, filled on both sides of the center.
    pos_mask = dr >= 0
    neg_mask = dr < 0

    # Fill above center (positive deviations)
    ax.fill_between(tw, fc, fc + dr,
                    where=dr >= 0,
                    color=bcolor, alpha=0.4,
                    hatch=hatch_style, linewidth=0.0,
                    zorder=2, label=None)

    # Fill below center (negative deviations)
    ax.fill_between(tw, fc + dr, fc,
                    where=dr < 0,
                    color=bcolor, alpha=0.4,
                    hatch=hatch_style, linewidth=0.0,
                    zorder=2, label=None)

    # Raw measurement scatter (faint dots)
    ax.scatter(bd['times_raw_w'], bd['freqs_raw'],
               s=6, color='black', alpha=0.4, zorder=4, marker='o')

    # Label on left: omega value
    ax.text(-4, fc, f'{fc:.1f}', fontsize=9, fontweight='bold',
            va='center', ha='right', color='black')

    # Label on right: T in weeks
    ax.text(n_disp + 4, fc, f'{T_wk:.1f}', fontsize=9, fontweight='bold',
            va='center', ha='left', color='black')

# Axis formatting
ax.set_xlim(-10, n_disp + 10)
ax.set_ylim(7.0, 12.5)
ax.set_xlabel('Weeks', fontsize=11)

# Left Y-axis: omega (rad/yr)
ax.set_ylabel('omega - RAD./YR.', fontsize=11)

# Right Y-axis: T (weeks)
ax_right = ax.twinx()
ax_right.set_ylim(7.0, 12.5)
T_ticks  = [2 * np.pi / f * FS_WEEKLY for f in AI5_CENTERS_RADYR]
f_ticks  = AI5_CENTERS_RADYR
ax_right.set_yticks(f_ticks)
ax_right.set_yticklabels([f'{t:.1f}' for t in T_ticks], fontsize=9)
ax_right.set_ylabel('T - WKS.', fontsize=11)

# Grid
ax.grid(True, alpha=0.15, axis='x')

# Week-axis ticks
ax.set_xticks(np.arange(0, n_disp + 1, 25))

# Title
ax.set_title('MODULATION SIDEBANDS\nFIGURE AI-5',
              fontsize=14, fontweight='bold', pad=12)

# Legend
patch_shade = mpatches.Patch(facecolor='#1a5fa0', alpha=0.4,
                              hatch='///', label='Frequency deviation')
line_center = plt.Line2D([0], [0], color='black', linewidth=0.8,
                          label='Nominal center')
dot_meas    = plt.Line2D([0], [0], marker='o', color='black', alpha=0.4,
                          linestyle='None', markersize=4, label='Half-cycle measurement')
ax.legend(handles=[line_center, patch_shade, dot_meas],
          loc='upper right', fontsize=8, framealpha=0.9)

fig.tight_layout()
out_path = os.path.join(SCRIPT_DIR, 'fig_AI5_sidebands.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_path}")
print("Done.")
