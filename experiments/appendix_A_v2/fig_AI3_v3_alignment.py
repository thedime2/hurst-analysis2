# -*- coding: utf-8 -*-
"""
Figure AI-3: Reference Image Alignment + Decimation Spacing Brute-Force

PURPOSE
-------
Align our FC-1..FC-10 Ormsby comb filter outputs with the reference scan
(figure_AI3_v2.png) and brute-force the decimation spacing + startidx
to find the combination that best reproduces Hurst's visual dot pattern.

HYPOTHESIS
----------
The "dotted" appearance of Hurst's AI-3 waveforms may represent actual
discrete computed samples at decimated positions (one dot every N weeks),
NOT a continuous dotted linestyle. If so, counting dot density in the
reference image reveals the decimation spacing Hurst used.

WORKFLOW
--------
  Step 1 — Alignment check (Figure 1):
    Overlay our spacing=1 continuous waveform on the reference image.
    Tune the IMG_* parameters below until FC tracks visually line up.

  Step 2 — Dot counting (manual):
    With alignment confirmed, count dots per full cycle for 2-3 FC tracks.
    Example: FC-1 has period ~43 wk; if you count ~6 dots/cycle → spacing≈7.

  Step 3 — Spacing brute-force (Figure 2):
    Plot dots (no connecting lines) for spacing = SPACING_RANGE, startidx 0..2.
    Identify which combination best matches the reference dot positions.

  Step 4 — Best-match overlay (Figure 3):
    Reference image + best spaced dots overlaid at full resolution.

TUNABLE ALIGNMENT PARAMETERS (tune Step 1 first)
-------------------------------------------------
  IMG_CROP_*   — crop fractions to remove borders/title/labels from scan
  IMG_X_START  — week number at left edge of image's data region
  IMG_X_END    — week number at right edge
  IMG_Y_BOTTOM — data y-value at bottom of image's plot area
  IMG_Y_TOP    — data y-value at top of image's plot area

  Start with the default values and visually inspect Figure 1 overlay.
  Adjust IMG_X_START/END to slide/stretch horizontally.
  Adjust IMG_Y_BOTTOM/TOP to slide/stretch vertically.

Figures produced:
  fig_AI3_align_check.png        — alignment verification overlay
  fig_AI3_align_spacing_grid.png — brute-force spacing grid (dots only)
  fig_AI3_align_best.png         — reference + best-match spaced dots

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
TARGET_AMP = 2.0      # normal track half-amplitude
BIG_AMP    = 4.0      # large track half-amplitude
BIG_THRESH = 1.7      # RMS ratio to trigger ±4 scale

# ============================================================================
# ALIGNMENT PARAMETERS — TUNE THESE (Step 1)
# ============================================================================

# --- Image crop: fraction of image W or H to remove from each edge ---
# Removes the outer border/title/axis-label margin of the scanned image.
# Increase to crop more; decrease to crop less.
IMG_CROP_L = 0.055   # fraction to remove from left   (y-axis label area)
IMG_CROP_R = 0.015   # fraction to remove from right  (right margin)
IMG_CROP_T = 0.065   # fraction to remove from top    (title area)
IMG_CROP_B = 0.060   # fraction to remove from bottom (x-axis label + WKS label)

# --- Horizontal data mapping: week numbers at left/right of cropped image ---
# IMG_X_START = week number aligned with left edge of cropped image
# IMG_X_END   = week number aligned with right edge of cropped image
IMG_X_START = -2.0    # typically near 0 (left axis tick)
IMG_X_END   = 270.0   # last visible data column in image (~267-275 weeks)

# --- Vertical data mapping: data-y at bottom/top of cropped image ---
# The image shows FC-1 at top (offset = (N-1)*SPACING) and FC-10 at bottom (offset=0).
# Set these to match what our coordinate system shows at the top/bottom of the frame.
IMG_Y_TOP    = (N_DISPLAY - 0.35) * SPACING   # data-y at top of image
IMG_Y_BOTTOM = -SPACING * 0.65                # data-y at bottom of image

# ============================================================================
# BRUTE-FORCE SPACING PARAMETERS (Step 3)
# ============================================================================

# Decimation spacings to test (weeks between computed samples)
SPACING_RANGE = [3, 4, 5, 6, 7, 8, 10, 12]

# Start indices to test (phase offset within first spacing-period)
# 0 = first data point, 1 = second data point, etc.
STARTIDX_RANGE = [0, 1, 2]

# Best spacing+startidx to highlight in Figure 3 (set after inspecting Figure 2)
BEST_SPACING  = 7
BEST_STARTIDX = 0

# Dot size for spaced outputs (in points)
DOT_SIZE = 3.0


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

rms_vals = np.array([
    np.sqrt(np.mean(outputs_dense[i]['signal'][s_idx:e_idx].real ** 2))
    for i in range(N_DISPLAY)
])
rms_vals    = np.where(rms_vals > 0, rms_vals, 1.0)
median_rms  = np.median(rms_vals)

scale_factors = []
amp_limits    = []
for i in range(N_DISPLAY):
    ratio = rms_vals[i] / median_rms
    amp   = BIG_AMP if ratio >= BIG_THRESH else TARGET_AMP
    scale_factors.append(amp / (3.0 * rms_vals[i]))
    amp_limits.append(amp)


# ============================================================================
# LOAD AND CROP REFERENCE IMAGE
# ============================================================================

def load_cropped_ref():
    if not os.path.exists(REF_IMAGE):
        return None
    img = mpimg.imread(REF_IMAGE)
    h, w = img.shape[:2]
    c_l = int(IMG_CROP_L * w)
    c_r = int((1 - IMG_CROP_R) * w)
    c_t = int(IMG_CROP_T * h)
    c_b = int((1 - IMG_CROP_B) * h)
    return img[c_t:c_b, c_l:c_r]


ref_cropped = load_cropped_ref()
if ref_cropped is None:
    print(f"WARNING: Reference image not found: {REF_IMAGE}")
    print("Figures 1 and 3 will be skipped.")


# ============================================================================
# HELPER: draw our waveform on ax (continuous dotted, spacing=1)
# ============================================================================

def draw_dense_waveform(ax, alpha_line=0.85, linestyle=':', lw=0.85):
    for i in range(N_DISPLAY):
        offset = (N_DISPLAY - 1 - i) * SPACING
        sig    = outputs_dense[i]['signal'][s_idx:e_idx].real * scale_factors[i]
        ax.axhline(offset, color='#888888', linewidth=0.4, zorder=1)
        ax.plot(weeks, sig + offset, linestyle=linestyle, color='black',
                linewidth=lw, alpha=alpha_line, zorder=3)
        # FC label
        ax.text(-4, offset, f'FC-{i+1}', fontsize=6.5, ha='right',
                va='center', color='navy', zorder=5)

    ax.axvline(0,           color='black', linewidth=0.6, linestyle='--', zorder=2)
    ax.axvline(n_weeks - 1, color='black', linewidth=0.6, linestyle='--', zorder=2)
    ax.set_xlim(IMG_X_START - 1, IMG_X_END + 1)
    ax.set_ylim(IMG_Y_BOTTOM, IMG_Y_TOP)
    ax.set_xticks(list(range(0, n_weeks + 1, 25)))
    ax.set_xticklabels([str(x) if x <= 250 else '' for x in range(0, n_weeks + 1, 25)],
                        fontsize=7)
    ax.set_yticks([(N_DISPLAY - 1 - i) * SPACING for i in range(N_DISPLAY)])
    ax.set_yticklabels([f'FC-{i+1}' for i in range(N_DISPLAY)], fontsize=8)
    ax.set_xlabel('WKS.', fontsize=9)
    ax.grid(True, axis='x', alpha=0.15, color='black', linewidth=0.3)


def draw_image_bg(ax, alpha=0.50):
    if ref_cropped is not None:
        ax.imshow(ref_cropped,
                  extent=[IMG_X_START, IMG_X_END, IMG_Y_BOTTOM, IMG_Y_TOP],
                  aspect='auto', alpha=alpha, zorder=0)


# ============================================================================
# FIGURE 1: ALIGNMENT CHECK
# Two panels: (left) our dense dotted waveform alone;
#             (right) reference image + our waveform overlaid.
# Inspect the right panel and tune IMG_* parameters above until they align.
# ============================================================================

print("\nGenerating Figure 1: alignment check...")

fig1, axes1 = plt.subplots(1, 2, figsize=(22, 13),
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
    f'Reference Image (alpha=0.60) + Our Waveform (alpha=0.70)\n'
    f'Crop L={IMG_CROP_L} R={IMG_CROP_R} T={IMG_CROP_T} B={IMG_CROP_B}  |  '
    f'X=[{IMG_X_START},{IMG_X_END}]  Y=[{IMG_Y_BOTTOM:.1f},{IMG_Y_TOP:.1f}]',
    fontsize=9, fontweight='bold'
)

fig1.suptitle(
    f'AI-3 Alignment Check  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}\n'
    f'Tune IMG_CROP_* and IMG_X/Y_* parameters at top of script until tracks align.',
    fontsize=11, fontweight='bold'
)
out1 = os.path.join(SCRIPT_DIR, 'fig_AI3_align_check.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved: {out1}")


# ============================================================================
# FIGURE 2: SPACING BRUTE-FORCE GRID
# Rows = spacings, columns = startidx values.
# Each panel shows the reference image at 45% opacity behind
# our spaced Ormsby dots (no connecting lines).
# ============================================================================

print("\nGenerating Figure 2: spacing brute-force grid...")

from src.filters import design_hurst_comb_bank, create_filter_kernels, apply_filter_bank

# Redesign filter bank using the same parameters
specs_bank   = design_hurst_comb_bank(
    n_filters=N_DISPLAY, w1_start=7.2, w_step=0.2,
    passband_width=0.2, skirt_width=0.3,
    nw=NW_WEEKLY, fs=FS_WEEKLY
)
filters_bank = create_filter_kernels(specs_bank, fs=FS_WEEKLY,
                                      filter_type='modulate', analytic=True)

# Cache applied results keyed by (spacing, startidx)
print("  Applying filter bank for each (spacing, startidx) combination...")
results_cache = {}
for sp in SPACING_RANGE:
    for si in STARTIDX_RANGE:
        if si >= sp:
            continue   # startidx must be < spacing
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

# Normalize filter outputs using same scale factors
# For each track, use the same amplitude normalisation as spacing=1

_cmap  = plt.colormaps['tab10']
track_colors = [_cmap(i / N_DISPLAY) for i in range(N_DISPLAY)]

for row, sp in enumerate(SPACING_RANGE):
    for col, si in enumerate(STARTIDX_RANGE):
        ax  = axes2[row, col] if n_sp > 1 else axes2[col]
        key = (sp, si)

        # Image background
        draw_image_bg(ax, alpha=0.45)

        if key not in results_cache:
            ax.set_title(f'spacing={sp} start={si}\n(startidx>=spacing, skip)',
                          fontsize=7)
            continue

        outputs = results_cache[key]

        for i in range(N_DISPLAY):
            offset = (N_DISPLAY - 1 - i) * SPACING
            sig    = outputs[i]['signal'][s_idx:e_idx].real

            # Normalise with same scale factor as spacing=1
            sig_scaled = sig * scale_factors[i]

            # Non-NaN positions only (the spaced dots)
            valid = ~np.isnan(sig_scaled)
            x_pts = weeks[valid]
            y_pts = (sig_scaled[valid]) + offset

            ax.scatter(x_pts, y_pts, s=DOT_SIZE, color=track_colors[i],
                       alpha=0.90, zorder=3, linewidths=0)

            # Zero-line
            ax.axhline(offset, color='#AAAAAA', linewidth=0.35, zorder=1)

        ax.axvline(0,           color='black', linewidth=0.5, linestyle='--', zorder=2)
        ax.axvline(n_weeks - 1, color='black', linewidth=0.5, linestyle='--', zorder=2)

        n_dots = sum(np.sum(~np.isnan(results_cache[key][i]['signal'][s_idx:e_idx].real))
                     for i in range(N_DISPLAY))
        ax.set_xlim(IMG_X_START, IMG_X_END)
        ax.set_ylim(IMG_Y_BOTTOM, IMG_Y_TOP)
        ax.set_xticks(range(0, int(IMG_X_END) + 1, 50))
        ax.set_xticklabels([str(x) for x in range(0, int(IMG_X_END) + 1, 50)], fontsize=6)
        ax.set_yticks([])
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
# Reference image + best (spacing, startidx) overlaid at full resolution.
# Change BEST_SPACING / BEST_STARTIDX at top of file after inspecting Fig 2.
# ============================================================================

print(f"\nGenerating Figure 3: best-match overlay "
      f"(spacing={BEST_SPACING}, startidx={BEST_STARTIDX})...")

key_best = (BEST_SPACING, BEST_STARTIDX)
if key_best not in results_cache:
    # Recompute if needed (e.g. startidx=0 was skipped)
    result_best = apply_filter_bank(close, filters_bank, fs=FS_WEEKLY,
                                     mode='reflect', spacing=BEST_SPACING,
                                     startidx=BEST_STARTIDX, interp='none')
    outputs_best = result_best['filter_outputs']
else:
    outputs_best = results_cache[key_best]

fig3, axes3 = plt.subplots(1, 2, figsize=(22, 13),
                            gridspec_kw={'wspace': 0.06})

for panel, (ax, img_alpha, our_alpha, title_sfx) in enumerate(zip(
        axes3,
        [0.0, 0.55],   # image alpha: left=no image, right=with image
        [0.88, 0.92],  # our dot alpha
        ['Our Dots Only', 'Reference Image + Our Dots'])):

    if img_alpha > 0:
        draw_image_bg(ax, alpha=img_alpha)

    for i in range(N_DISPLAY):
        offset = (N_DISPLAY - 1 - i) * SPACING
        sig    = outputs_best[i]['signal'][s_idx:e_idx].real * scale_factors[i]

        valid  = ~np.isnan(sig)
        x_pts  = weeks[valid]
        y_pts  = sig[valid] + offset

        # Dots with cross-hair lines (helps visual alignment)
        ax.scatter(x_pts, y_pts, s=7, color=track_colors[i],
                   alpha=our_alpha, zorder=4, linewidths=0)

        # Thin connecting line (optional, toggle off to see pure dots)
        ax.plot(weeks[valid], sig[valid] + offset, '-',
                color=track_colors[i], linewidth=0.4, alpha=0.35, zorder=3)

        # Zero-line
        ax.axhline(offset, color='#888888', linewidth=0.35, zorder=1)

        # Track label
        ax.text(IMG_X_START + 1, offset, f'FC-{i+1}', fontsize=6.5,
                ha='left', va='center', color=track_colors[i], zorder=5)

    ax.axvline(0,           color='black', linewidth=0.6, linestyle='--', zorder=2)
    ax.axvline(n_weeks - 1, color='black', linewidth=0.6, linestyle='--', zorder=2)
    ax.set_xlim(IMG_X_START, IMG_X_END)
    ax.set_ylim(IMG_Y_BOTTOM, IMG_Y_TOP)
    ax.set_xticks(range(0, int(IMG_X_END) + 1, 25))
    ax.set_xticklabels([str(x) if x % 50 == 0 else '' for x in range(0, int(IMG_X_END)+1, 25)],
                        fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel('WKS.', fontsize=9)
    ax.grid(True, axis='x', alpha=0.12, linewidth=0.3)
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
# DOT DENSITY SUMMARY (helps manual counting)
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
print("To estimate Hurst's spacing:")
print("  Count dots per full cycle for 2-3 FC tracks in the reference image.")
print("  spacing ~= T_wk / (dots_per_cycle)")
print()
print("Output files:")
for out in [out1, out2, out3]:
    print(f"  {os.path.basename(out)}")
print("\nDone.")
