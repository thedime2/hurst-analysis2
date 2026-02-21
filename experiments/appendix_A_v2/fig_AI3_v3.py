# -*- coding: utf-8 -*-
"""
Figure AI-3 v3: Comb Filter Outputs — Reference-Aligned Style

Reproduces Hurst's "Comb Output Example" (Figure AI-3, p.193) matching the
visual style of the hi-res reference scan (figure_AI3_v2.png):

- Single panel (Ormsby FIR only)
- FC-1 through FC-10 stacked with FC-1 at top
- Each track individually normalised to ±2 amplitude units
  (FC-10 gets ±4 if its RMS is ≳ 2× the median)
- Waveform as DOTTED line (:), zero-line as thin solid grey
- No envelope overlay, no phase overlay
- Two vertical dashed reference lines at the display window start/end
- Label "FC-N" at the left of each track

Display window: 1934-12-07 to 1940-01-26 (~267 weeks)
Note: the date labels "5-24-40" and "3-29-46" printed in Hurst's book are a
      known editorial/transcription error (see PRD). The window above is correct.

Also produces a reference-overlay figure: reference PNG underlay with our
waveform overlaid to confirm alignment of peaks and cycle count.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figure AI-3, p.193
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils_ai import (
    load_weekly_data,
    design_comb_bank, make_ormsby_kernels, apply_comb_bank,
    get_window, FS_WEEKLY, NW_WEEKLY,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_IMAGE  = os.path.join(SCRIPT_DIR, '../../references/appendix_a/figure_AI3_v2.png')

N_DISPLAY  = 10      # FC-1 through FC-10
SPACING    = 4.5     # vertical spacing between zero-lines (in normalised ±2 units)
TARGET_AMP = 2.0     # target half-amplitude for each track (±2)
BIG_AMP    = 4.0     # tracks with RMS ≳ 2× median get ±4
BIG_THRESH = 1.7     # ratio of track RMS to median that triggers ±4 scale


# ============================================================================
# LOAD DATA AND APPLY FILTERS
# ============================================================================

print("Loading weekly DJIA data (all dates)...")
close, dates_dt = load_weekly_data()
s_idx, e_idx = get_window(dates_dt)
n_weeks = e_idx - s_idx
weeks = np.arange(n_weeks, dtype=float)

print(f"Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} weeks)")

specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)

print("Applying Ormsby comb bank to all data...")
outputs = apply_comb_bank(close, filters, fs=FS_WEEKLY)
print("Done.")


# ============================================================================
# COMPUTE PER-TRACK SCALE FACTORS
# ============================================================================

# RMS of real part within display window for each track
rms_vals = np.array([
    np.sqrt(np.mean(outputs[i]['signal'][s_idx:e_idx].real ** 2))
    for i in range(N_DISPLAY)
])
rms_vals = np.where(rms_vals > 0, rms_vals, 1.0)
median_rms = np.median(rms_vals)

# Each track gets its own scale so that ±3σ ≈ ±2 (or ±4 if large)
scale_factors = []
amp_limits    = []
for i in range(N_DISPLAY):
    ratio = rms_vals[i] / median_rms
    if ratio >= BIG_THRESH:
        # Large track: normalise to ±4
        amp = BIG_AMP
    else:
        amp = TARGET_AMP
    # 3σ maps to ±amp: scale = amp / (3 * rms)
    scale = amp / (3.0 * rms_vals[i])
    scale_factors.append(scale)
    amp_limits.append(amp)

print("\nTrack amplitudes (±N):")
for i in range(N_DISPLAY):
    fc = specs[i]['f_center']
    T_wk = 2 * np.pi / fc * FS_WEEKLY
    print(f"  FC-{i+1:2d}  fc={fc:.1f} r/y  T={T_wk:.1f}wk  "
          f"rms={rms_vals[i]:.3f}  scale=±{amp_limits[i]:.0f}")


# ============================================================================
# FIGURE 1: HURST-STYLE SINGLE PANEL (matches reference layout)
# ============================================================================

print("\nGenerating Figure 1: Hurst-style AI-3...")

fig1, ax1 = plt.subplots(figsize=(13, 14))

ytick_pos, ytick_lab = [], []

for i in range(N_DISPLAY):
    offset = (N_DISPLAY - 1 - i) * SPACING   # FC-1 at top

    sig = outputs[i]['signal'][s_idx:e_idx].real * scale_factors[i]

    # Zero reference line (thin solid grey)
    ax1.axhline(offset, color='#888888', linewidth=0.5, zorder=1)

    # Waveform as dotted line (matching Hurst's figure style)
    ax1.plot(weeks, sig + offset, linestyle=':', color='black',
             linewidth=0.8, alpha=0.92, zorder=3)

    # Amplitude limit tick marks at left (±2 or ±4)
    amp = amp_limits[i]
    ax1.text(-6, offset + amp, f'+{amp:.0f}', fontsize=6.5, ha='right',
             va='center', color='#444444')
    ax1.text(-6, offset,       '0',           fontsize=6.5, ha='right',
             va='center', color='#888888')
    ax1.text(-6, offset - amp, f'-{amp:.0f}', fontsize=6.5, ha='right',
             va='center', color='#444444')

    # Label: FC-N
    fc  = specs[i]['f_center']
    T_wk = 2 * np.pi / fc * FS_WEEKLY
    ytick_pos.append(offset)
    ytick_lab.append(f'FC-{i+1}')

# Vertical dashed reference lines at week 0 (start) and week n_weeks-1 (end)
# These are the display-window boundaries (editorial note: dates in book are wrong)
ax1.axvline(0,           color='black', linewidth=0.6, linestyle='--', zorder=2)
ax1.axvline(n_weeks - 1, color='black', linewidth=0.6, linestyle='--', zorder=2)

# ±amp horizontal limit lines for each track (thin dashed)
for i in range(N_DISPLAY):
    offset = (N_DISPLAY - 1 - i) * SPACING
    amp = amp_limits[i]
    for sign in [+1, -1]:
        ax1.axhline(offset + sign * amp, color='#BBBBBB', linewidth=0.35,
                    linestyle='--', zorder=1)

ax1.set_yticks(ytick_pos)
ax1.set_yticklabels(ytick_lab, fontsize=9, family='monospace')
ax1.set_xlim(-8, n_weeks + 2)
ax1.set_ylim(-SPACING * 0.6, (N_DISPLAY - 0.4) * SPACING)
ax1.set_xlabel('WKS.', fontsize=10, labelpad=4)
ax1.set_xticks(list(range(0, n_weeks + 1, 25)))

# x-axis labels: 25, 50, 75, ..., 250 (stop at 250 to match reference)
xtick_labels = [str(x) if x <= 250 else '' for x in range(0, n_weeks + 1, 25)]
ax1.set_xticklabels(xtick_labels, fontsize=8)
ax1.grid(True, axis='x', alpha=0.18, color='black', linewidth=0.3)

ax1.set_title(
    'COMB OUTPUT  FIGURE AI-3\n'
    f'Ormsby FIR  |  DJIA Weekly  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  '
    f'|  FC-1 through FC-10',
    fontsize=11, fontweight='bold', pad=10
)
ax1.text(n_weeks * 0.5, (N_DISPLAY - 0.1) * SPACING,
         'EXAMPLE', fontsize=10, ha='center', style='italic', color='#555555')

fig1.tight_layout()
out1 = os.path.join(SCRIPT_DIR, 'fig_AI3_v3_hurst_style.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved: {out1}")


# ============================================================================
# FIGURE 2: REFERENCE OVERLAY (reference image underlay + our waveform)
# ============================================================================
# Only if reference file exists
if os.path.exists(REF_IMAGE):
    print("\nGenerating Figure 2: Reference image overlay...")

    ref_img = mpimg.imread(REF_IMAGE)

    # The reference image shows 10 panels, x: 0..~267 weeks, y: stacked panels.
    # We approximate the mapping:
    #   - The reference x-axis runs 0 to ~267 weeks
    #   - The reference y-axis is stacked vertically with FC-1 at top

    fig2, axes2 = plt.subplots(1, 2, figsize=(22, 14))

    # Left panel: our reproduction
    ax_l = axes2[0]
    for i in range(N_DISPLAY):
        offset = (N_DISPLAY - 1 - i) * SPACING
        sig = outputs[i]['signal'][s_idx:e_idx].real * scale_factors[i]
        ax_l.axhline(offset, color='#888888', linewidth=0.5, zorder=1)
        ax_l.plot(weeks, sig + offset, linestyle=':', color='black',
                  linewidth=0.8, zorder=3)
        ax_l.text(-6, offset, f'FC-{i+1}', fontsize=7, ha='right',
                  va='center', color='navy')
    ax_l.axvline(0,           color='black', linewidth=0.6, linestyle='--')
    ax_l.axvline(n_weeks - 1, color='black', linewidth=0.6, linestyle='--')
    ax_l.set_xlim(-8, n_weeks + 2)
    ax_l.set_ylim(-SPACING * 0.6, (N_DISPLAY - 0.4) * SPACING)
    ax_l.set_xticks(list(range(0, n_weeks + 1, 25)))
    ax_l.set_xticklabels([str(x) if x <= 250 else '' for x in range(0, n_weeks + 1, 25)])
    ax_l.set_xlabel('WKS.', fontsize=10)
    ax_l.set_title('Our Reproduction (Ormsby FIR)', fontsize=11, fontweight='bold')
    ax_l.grid(True, axis='x', alpha=0.18)

    # Right panel: reference image
    ax_r = axes2[1]
    ax_r.imshow(ref_img, aspect='auto')
    ax_r.axis('off')
    ax_r.set_title('Reference: Hurst Figure AI-3 (figure_AI3_v2.png)', fontsize=11, fontweight='bold')

    fig2.suptitle(
        f'AI-3 Comparison  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  '
        f'|  Note: reference dates "5-24-40" / "3-29-46" are editorial errors',
        fontsize=10, fontweight='bold'
    )
    fig2.tight_layout()
    out2 = os.path.join(SCRIPT_DIR, 'fig_AI3_v3_comparison.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {out2}")
else:
    print(f"  Reference image not found: {REF_IMAGE}")
    print("  Skipping overlay figure.")

print("\nDone.")
