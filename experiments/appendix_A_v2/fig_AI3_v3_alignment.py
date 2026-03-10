# -*- coding: utf-8 -*-
"""
Figure AI-3: Reference Image Alignment + Decimation Spacing Brute-Force

PURPOSE
-------
Align our FC-1..FC-10 Ormsby comb filter outputs with the reference scan
(figure_AI3_v2.png) and brute-force the decimation spacing + startidx
to find the combination that best reproduces Hurst's visual dot pattern.

ALIGNMENT METHOD — PIXEL CALIBRATION
-------------------------------------
The reference image is displayed at its NATURAL aspect ratio (no stretch).
Four pixel positions of known data points are identified in the image:
  - CAL_WEEK0_PX / CAL_WEEK250_PX: pixel x of the "0" and "250" x-axis ticks
  - CAL_FC1_ZERO_PX / CAL_FC10_ZERO_PX: pixel y of FC-1 and FC-10 zero lines

From these 4 values the coordinate mapping is computed automatically:
  pixel_x = CAL_WEEK0_PX + week * px_per_week
  pixel_y = a * data_y + b
The imshow extent is derived so that our data coordinate system aligns
perfectly with the image's data area. The image is never distorted.

Figures produced:
  fig_AI3_align_check.png        -- alignment verification overlay
  fig_AI3_align_spacing_grid.png -- brute-force spacing grid (dots only)
  fig_AI3_align_best.png         -- reference + best-match spaced dots

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figure AI-3, p.193
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
REF_IMAGE  = os.path.join(SCRIPT_DIR,
                           '../../references/appendix_a/figure_AI3_v2.png')

# ============================================================================
# DISPLAY / TRACK LAYOUT PARAMETERS  (match fig_AI3_v3.py)
# ============================================================================

N_DISPLAY  = 10       # FC-1 .. FC-10
SPACING    = 4.5      # vertical distance between track zero-lines
TARGET_AMP = 2.5      # FC-1 target half-amplitude (sets the reference scale)
AMP_BOOST  = 1.0     # global amplitude multiplier (>1 = louder, try 1.1-1.3)

# ============================================================================
# ALIGNMENT — PIXEL CALIBRATION
# ============================================================================
# Identify these 4 pixel positions in the reference image (use an image editor).
# The script computes the full coordinate mapping from them automatically.
#
# X calibration: pixel x of the "0" and "250" ticks on the x-axis
# Y calibration: pixel y of the FC-1 and FC-10 zero lines

CAL_WEEK0_PX     = 293+48  #493     # pixel x of the "0" x-axis tick  (measured from gridlines)
CAL_WEEK250_PX   = 1813    # pixel x of the "250" x-axis tick (measured from gridlines)
CAL_FC1_ZERO_PX  = 558     # pixel y of FC-1 zero line (detected, darkness=0.004)
CAL_FC10_ZERO_PX = 3150    # pixel y of FC-10 zero line (558 + 9*289, from uniform spacing)

# Per-track vertical offsets in DATA-Y UNITS (positive = shift track UP in plot).
# FC-1..5 are locked at 0.0 (already aligned).
# FC-6..10 are tunable to account for non-uniform spacing in the scan.
# Measured from image scan: actual_pixel vs expected_pixel at uniform 288px spacing.

# 7 + higher = vertical up

CAL_FC_Y_OFFSETS = [
    0.0,    # FC-1  (locked)
    0.0,    # FC-2  (locked)
    0.0,    # FC-3  (locked)
    0.0,    # FC-4  (locked)
    0.0,    # FC-5  (locked)
   +1.10,   # FC-6  (actual ~2011px vs expected 1998, +13px lower)
   +2.11,   # FC-7  (actual ~2293px vs expected 2286, +7px lower)
   +4.40,   # FC-8  (actual ~2556px vs expected 2574, -18px higher)
   +5.50,   # FC-9  (actual ~2849px vs expected 2862, -13px higher)
   +5.55,   # FC-10 (actual ~3163px vs expected 3150, +13px lower)
]

# ============================================================================
# BRUTE-FORCE SPACING PARAMETERS (Step 3)
# ============================================================================

SPACING_RANGE  = [2, 3, 4, 5, 6, 7, 8, 10, 12]
STARTIDX_RANGE = [0, 1, 2]
BEST_SPACING   = 2
BEST_STARTIDX  = 0
DOT_SIZE       = 3.0


def track_y(i):
    """Data-y position of track i's zero line, including per-track offset."""
    return (N_DISPLAY - 1 - i) * SPACING + CAL_FC_Y_OFFSETS[i]


# ============================================================================
# LOAD DATA AND APPLY FILTERS (spacing=1 for baseline)
# ============================================================================

print("Loading weekly DJIA data...")
close, dates_dt = load_weekly_data()
s_idx, e_idx = get_window(dates_dt)
n_weeks = e_idx - s_idx
weeks   = np.arange(n_weeks, dtype=float)

print(f"Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} weeks)")

specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters_base = make_ormsby_kernels(specs, fs=FS_WEEKLY)

print("Applying Ormsby comb bank (spacing=1)...")
outputs_dense = apply_comb_bank(close, filters_base, fs=FS_WEEKLY)
print("Done.")


# ============================================================================
# COMPUTE PER-TRACK SCALE FACTORS (same logic as fig_AI3_v3.py)
# ============================================================================

# Compute FC-1's scale factor, then apply it to ALL tracks.
# This preserves the natural amplitude ratios between filters —
# tracks with more energy appear larger, matching Hurst's image.
rms_vals = np.array([
    np.sqrt(np.mean(outputs_dense[i]['signal'][s_idx:e_idx].real ** 2))
    for i in range(N_DISPLAY)
])
rms_vals = np.where(rms_vals > 0, rms_vals, 1.0)

# FC-1 sets the reference: scale so its peaks reach ~TARGET_AMP
fc1_scale = TARGET_AMP / (3.0 * rms_vals[0]) * AMP_BOOST
scale_factors = [fc1_scale] * N_DISPLAY   # same scale for every track


# ============================================================================
# LOAD REFERENCE IMAGE + COMPUTE COORDINATE MAPPING
# ============================================================================

def load_ref_image():
    if not os.path.exists(REF_IMAGE):
        return None
    return mpimg.imread(REF_IMAGE)


ref_img = load_ref_image()

# Data coordinate system:
#   x = week number (0 .. n_weeks-1)
#   y = FC-10 zero line at 0, FC-1 zero line at (N_DISPLAY-1)*SPACING = 40.5
FC1_DATA_Y  = (N_DISPLAY - 1) * SPACING   # 40.5
FC10_DATA_Y = 0.0

if ref_img is None:
    print(f"WARNING: Reference image not found: {REF_IMAGE}")
    print("Figures 1 and 3 will be skipped.")
    IMG_EXTENT   = None
    IMG_PANEL_W  = 10.5
    IMG_PANEL_H  = 13.0
    PLOT_XLIM    = (-5, n_weeks + 5)
    PLOT_YLIM    = (-SPACING, FC1_DATA_Y + SPACING)
else:
    img_h, img_w = ref_img.shape[:2]
    pixel_ar     = img_w / img_h

    # --- X mapping: pixel_x = CAL_WEEK0_PX + week * px_per_week ---
    px_per_week = (CAL_WEEK250_PX - CAL_WEEK0_PX) / 250.0

    # --- Y mapping: pixel_y = a * data_y + b ---
    #   FC-10 (data_y=0)    -> pixel_y = CAL_FC10_ZERO_PX  => b = CAL_FC10_ZERO_PX
    #   FC-1  (data_y=40.5) -> pixel_y = CAL_FC1_ZERO_PX   => a = (FC1_px - FC10_px) / 40.5
    b_y = CAL_FC10_ZERO_PX
    a_y = (CAL_FC1_ZERO_PX - CAL_FC10_ZERO_PX) / FC1_DATA_Y   # negative

    # Image edges in data coordinates:
    x_left   = (0      - CAL_WEEK0_PX) / px_per_week
    x_right  = (img_w  - CAL_WEEK0_PX) / px_per_week
    y_top    = (0      - b_y) / a_y     # pixel 0 = top of image
    y_bottom = (img_h  - b_y) / a_y     # pixel img_h = bottom of image

    IMG_EXTENT  = [x_left, x_right, y_bottom, y_top]
    IMG_PANEL_W = 10.5
    IMG_PANEL_H = IMG_PANEL_W / pixel_ar   # match image AR => no distortion

    # Plot limits: show our full data range, clipped to roughly the image
    PLOT_XLIM = (x_left, min(x_right, n_weeks + 5))
    PLOT_YLIM = (y_bottom, y_top)

    print(f"Image: {img_w} x {img_h} px  |  AR = {pixel_ar:.3f}")
    print(f"Calibration: week0@{CAL_WEEK0_PX}px  week250@{CAL_WEEK250_PX}px  "
          f"-> {px_per_week:.2f} px/wk")
    print(f"Calibration: FC1@{CAL_FC1_ZERO_PX}px  FC10@{CAL_FC10_ZERO_PX}px  "
          f"-> {abs(a_y):.2f} px/data-y-unit")
    print(f"Image extent in data coords: "
          f"x=[{x_left:.1f}, {x_right:.1f}]  y=[{y_bottom:.1f}, {y_top:.1f}]")
    print(f"Figure panel: {IMG_PANEL_W:.1f} x {IMG_PANEL_H:.1f} in  (no distortion)")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

_cmap  = plt.colormaps['tab10']
track_colors = [_cmap(i / N_DISPLAY) for i in range(N_DISPLAY)]


def draw_image_bg(ax, alpha=0.50):
    """Draw the reference image as background, mapped to data coordinates."""
    if ref_img is not None and IMG_EXTENT is not None:
        ax.imshow(ref_img, extent=IMG_EXTENT, aspect='auto',
                  alpha=alpha, zorder=0)


def setup_axes(ax, show_fc_labels=True):
    """Set standard axis limits, ticks, and gridlines."""
    ax.set_xlim(PLOT_XLIM)
    ax.set_ylim(PLOT_YLIM)
    ax.set_xticks(list(range(0, n_weeks + 1, 25)))
    ax.set_xticklabels([str(x) if x <= 250 else '' for x in range(0, n_weeks + 1, 25)],
                        fontsize=7)
    if show_fc_labels:
        ax.set_yticks([track_y(i) for i in range(N_DISPLAY)])
        ax.set_yticklabels([f'FC-{i+1}' for i in range(N_DISPLAY)], fontsize=8)
    else:
        ax.set_yticks([])
    ax.set_xlabel('WKS.', fontsize=9)
    ax.grid(True, axis='x', alpha=0.15, color='black', linewidth=0.3)


def draw_dense_waveform(ax, alpha_line=0.85, linestyle=':', lw=0.85):
    """Draw our spacing=1 continuous waveforms for all tracks."""
    for i in range(N_DISPLAY):
        offset = track_y(i)
        sig    = outputs_dense[i]['signal'][s_idx:e_idx].real * scale_factors[i]
        ax.axhline(offset, color='#444444', linewidth=0.5, linestyle='--', zorder=1)
        ax.plot(weeks, sig + offset, linestyle=linestyle, color='black',
                linewidth=lw, alpha=alpha_line, zorder=3)
        ax.text(-4, offset, f'FC-{i+1}', fontsize=6.5, ha='right',
                va='center', color='navy', zorder=5)
    ax.axvline(0,           color='black', linewidth=0.6, linestyle='--', zorder=2)
    ax.axvline(n_weeks - 1, color='black', linewidth=0.6, linestyle='--', zorder=2)
    setup_axes(ax, show_fc_labels=True)


# ============================================================================
# FIGURE 1: ALIGNMENT CHECK
# ============================================================================

print("\nGenerating Figure 1: alignment check...")

fig1, axes1 = plt.subplots(1, 2, figsize=(IMG_PANEL_W * 2, IMG_PANEL_H),
                            gridspec_kw={'wspace': 0.06})

# Left: our waveform only
ax_l = axes1[0]
draw_dense_waveform(ax_l, alpha_line=0.90)
ax_l.set_title('Our Reproduction (spacing=1, dotted)\nFC-1..FC-10 individually normalised',
               fontsize=10, fontweight='bold')

# Right: reference image + our waveform overlaid
ax_r = axes1[1]
draw_image_bg(ax_r, alpha=0.60)
draw_dense_waveform(ax_r, alpha_line=0.70, linestyle='-')
ax_r.set_title(
    f'Reference Image + Our Waveform\n'
    f'CAL: wk0@{CAL_WEEK0_PX}px  wk250@{CAL_WEEK250_PX}px  '
    f'FC1@{CAL_FC1_ZERO_PX}px  FC10@{CAL_FC10_ZERO_PX}px',
    fontsize=8, fontweight='bold'
)

fig1.suptitle(
    f'AI-3 Alignment Check  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}\n'
    f'Tune CAL_* pixel positions at top of script until tracks align.',
    fontsize=11, fontweight='bold'
)
out1 = os.path.join(SCRIPT_DIR, 'fig_AI3_align_check.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved: {out1}")


# ============================================================================
# FIGURE 2: SPACING BRUTE-FORCE GRID
# ============================================================================

print("\nGenerating Figure 2: spacing brute-force grid...")

from src.filters import design_hurst_comb_bank, create_filter_kernels, apply_filter_bank

specs_bank   = design_hurst_comb_bank(
    n_filters=N_DISPLAY, w1_start=7.2, w_step=0.2,
    passband_width=0.2, skirt_width=0.3,
    nw=NW_WEEKLY, fs=FS_WEEKLY
)
filters_bank = create_filter_kernels(specs_bank, fs=FS_WEEKLY,
                                      filter_type='modulate', analytic=True)

print("  Applying filter bank for each (spacing, startidx) combination...")
results_cache = {}
for sp in SPACING_RANGE:
    for si in STARTIDX_RANGE:
        if si >= sp:
            continue
        key = (sp, si)
        result = apply_filter_bank(close, filters_bank, fs=FS_WEEKLY,
                                    mode='reflect', spacing=sp, startidx=si,
                                    interp='none')
        results_cache[key] = result['filter_outputs']
        print(f"    spacing={sp} startidx={si}: nw_dec={NW_WEEKLY//sp}")

n_sp = len(SPACING_RANGE)
n_si = len(STARTIDX_RANGE)

fig2, axes2 = plt.subplots(n_sp, n_si, figsize=(n_si * 9, n_sp * 3.8),
                            gridspec_kw={'hspace': 0.45, 'wspace': 0.06})

for row, sp in enumerate(SPACING_RANGE):
    for col, si in enumerate(STARTIDX_RANGE):
        ax  = axes2[row, col] if n_sp > 1 else axes2[col]
        key = (sp, si)

        draw_image_bg(ax, alpha=0.45)

        if key not in results_cache:
            ax.set_title(f'spacing={sp} start={si}\n(startidx>=spacing, skip)',
                          fontsize=7)
            continue

        outputs = results_cache[key]

        for i in range(N_DISPLAY):
            offset = track_y(i)
            sig    = outputs[i]['signal'][s_idx:e_idx].real * scale_factors[i]
            valid  = ~np.isnan(sig)
            x_pts  = weeks[valid]
            y_pts  = sig[valid] + offset

            ax.scatter(x_pts, y_pts, s=DOT_SIZE, color=track_colors[i],
                       alpha=0.90, zorder=3, linewidths=0)
            ax.axhline(offset, color='#444444', linewidth=0.45, linestyle='--', zorder=1)

        ax.axvline(0,           color='black', linewidth=0.5, linestyle='--', zorder=2)
        ax.axvline(n_weeks - 1, color='black', linewidth=0.5, linestyle='--', zorder=2)

        n_dots = sum(np.sum(~np.isnan(results_cache[key][i]['signal'][s_idx:e_idx].real))
                     for i in range(N_DISPLAY))
        setup_axes(ax, show_fc_labels=False)
        ax.set_xticks(range(0, n_weeks + 1, 50))
        ax.set_xticklabels([str(x) for x in range(0, n_weeks + 1, 50)], fontsize=6)
        ax.tick_params(labelsize=6)

        ax.set_title(
            f'spacing={sp}  startidx={si}  nw_dec={NW_WEEKLY//sp}\n'
            f'{n_dots} total dots  ({n_dots/N_DISPLAY:.0f}/filter avg)',
            fontsize=7.5, fontweight='bold'
        )
        if col == 0:
            ax.set_ylabel(f'sp={sp}', fontsize=8, rotation=0, labelpad=35,
                           va='center', ha='right')
        if row == n_sp - 1:
            ax.set_xlabel(f'startidx={si}', fontsize=8)

fig2.suptitle(
    f'AI-3 Spacing Brute-Force  |  Dots = spaced Ormsby outputs (no connecting lines)\n'
    f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}  |  Reference at 45% opacity  |  '
    f'FC-1..FC-10 coloured by track',
    fontsize=12, fontweight='bold'
)
out2 = os.path.join(SCRIPT_DIR, 'fig_AI3_align_spacing_grid.png')
fig2.savefig(out2, dpi=120, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: {out2}")


# ============================================================================
# FIGURE 3: BEST-MATCH OVERLAY
# ============================================================================

print(f"\nGenerating Figure 3: best-match overlay "
      f"(spacing={BEST_SPACING}, startidx={BEST_STARTIDX})...")

key_best = (BEST_SPACING, BEST_STARTIDX)
if key_best not in results_cache:
    result_best = apply_filter_bank(close, filters_bank, fs=FS_WEEKLY,
                                     mode='reflect', spacing=BEST_SPACING,
                                     startidx=BEST_STARTIDX, interp='none')
    outputs_best = result_best['filter_outputs']
else:
    outputs_best = results_cache[key_best]

fig3, axes3 = plt.subplots(1, 2, figsize=(IMG_PANEL_W * 2, IMG_PANEL_H),
                            gridspec_kw={'wspace': 0.06})

for panel, (ax, img_alpha, our_alpha, title_sfx) in enumerate(zip(
        axes3,
        [0.0, 0.55],
        [0.88, 0.92],
        ['Our Dots Only', 'Reference Image + Our Dots'])):

    if img_alpha > 0:
        draw_image_bg(ax, alpha=img_alpha)

    for i in range(N_DISPLAY):
        offset = track_y(i)
        sig    = outputs_best[i]['signal'][s_idx:e_idx].real * scale_factors[i]
        valid  = ~np.isnan(sig)
        x_pts  = weeks[valid]
        y_pts  = sig[valid] + offset

        ax.scatter(x_pts, y_pts, s=7, color=track_colors[i],
                   alpha=our_alpha, zorder=4, linewidths=0)
        ax.plot(weeks[valid], sig[valid] + offset, '-',
                color=track_colors[i], linewidth=0.4, alpha=0.35, zorder=3)
        ax.axhline(offset, color='#444444', linewidth=0.5, linestyle='--', zorder=1)
        ax.text(PLOT_XLIM[0] + 3, offset, f'FC-{i+1}', fontsize=6.5,
                ha='left', va='center', color=track_colors[i], zorder=5)

    ax.axvline(0,           color='black', linewidth=0.6, linestyle='--', zorder=2)
    ax.axvline(n_weeks - 1, color='black', linewidth=0.6, linestyle='--', zorder=2)
    setup_axes(ax, show_fc_labels=False)
    ax.set_title(
        f'spacing={BEST_SPACING}  startidx={BEST_STARTIDX}  '
        f'nw_dec={NW_WEEKLY//BEST_SPACING}  |  {title_sfx}',
        fontsize=10, fontweight='bold'
    )

fig3.suptitle(
    f'AI-3 Best-Match Spacing Overlay  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}\n'
    f'Dots = spaced Ormsby at spacing={BEST_SPACING}, startidx={BEST_STARTIDX}  |  '
    f'Thin line = connecting segments (for reference)',
    fontsize=11, fontweight='bold'
)
out3 = os.path.join(SCRIPT_DIR, 'fig_AI3_align_best.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"  Saved: {out3}")


# ============================================================================
# DOT DENSITY SUMMARY
# ============================================================================

print()
print("=" * 60)
print("Dot density per filter (spacing=1 baseline)")
print("=" * 60)
print(f"{'FC':>5}  {'fc(r/y)':>8}  {'T(wk)':>7}  {'peaks':>6}")
for i in range(N_DISPLAY):
    sig   = outputs_dense[i]['signal'][s_idx:e_idx].real
    fc    = specs[i]['f_center']
    T_wk  = 2 * np.pi / fc * FS_WEEKLY
    from scipy.signal import find_peaks as _fp
    min_d = max(3, int(T_wk * 0.45))
    pk, _ = _fp(sig, distance=min_d)
    print(f"  FC-{i+1:2d}  {fc:>8.1f}  {T_wk:>7.1f}  {len(pk):>6}")

print()
print("Output files:")
for out in [out1, out2, out3]:
    print(f"  {os.path.basename(out)}")
print("\nDone.")
