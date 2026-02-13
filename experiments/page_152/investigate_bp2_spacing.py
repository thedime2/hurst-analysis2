# -*- coding: utf-8 -*-
"""
Investigate BP-2 Spaced Filter for Page 152

Compares a full-length Ormsby bandpass (nw=1393, fs=52) with a short filter
(nw=199) applied to decimated input (every 7th sample, fs=52/7).

The key insight: decimating the input by 7 then filtering with 199 taps at
the reduced sample rate covers the same total time span (199*7 = 1393 weeks)
as the full 1393-tap filter, giving equivalent frequency resolution.

Steps:
  1. Baseline: full nw=1393 real Ormsby BP applied to full DJIA dataset
  2. Spaced: decimate input by 7, apply nw=199 filter at fs/7, place back
  3. Overlay comparison (continuous line vs sparse dots)
  4. Offset sweep (start=0..6) to find best match with Hurst's reference
  5. Best match overlaid on scanned reference image

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing, p. 152
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg

from src.filters import (
    ormsby_filter,
    apply_ormsby_filter,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
ref_image_path = os.path.join(script_dir, '../../references/page_152/filter2_4point5years.png')

# Hurst's full analysis window (filter over entire dataset)
DATE_START = '1921-04-29'
DATE_END = '1965-05-21'

# Display window (matches page 45 convention)
DISPLAY_START = '1935-01-01'
DISPLAY_END = '1952-02-01'

#DF start a bit earlier to fix drift
DISPLAY_START = '1934-08-01'
DISPLAY_END = '1952-04-01'

# Data parameters
FS = 52
TWOPI = 2 * np.pi

# BP-2 filter specification (all frequencies in rad/yr)
BP2_SPEC = {
    'f1': 0.85, 'f2': 1.25, 'f3': 2.05, 'f4': 2.45,
    'f_center': (1.25 + 2.05) / 2,
    'bandwidth': 2.05 - 1.25,
    'label': 'BP-2: ~3.8 yr',
}

# Filter lengths
NW_FULL = 1393
NW_SHORT = 199

# Spacing (decimation factor)
STEP = 7

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("Investigate BP-2 Spaced Filter (Page 152)")
print("=" * 70)
print()

print("Loading weekly DJIA data...")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between(DATE_START, DATE_END)].copy()
close_prices = df_hurst.Close.values
dates = df_hurst.Date.values
n_points = len(close_prices)

print(f"  Loaded {n_points} samples from {DATE_START} to {DATE_END}")

# Display window indices
dates_dt = pd.to_datetime(dates)
mask = (dates_dt >= pd.to_datetime(DISPLAY_START)) & \
       (dates_dt <= pd.to_datetime(DISPLAY_END))
disp_idx = np.where(mask)[0]
s_idx, e_idx = disp_idx[0], disp_idx[-1] + 1
disp_dates = dates_dt[s_idx:e_idx]
print(f"  Display window: {DISPLAY_START} to {DISPLAY_END} ({e_idx - s_idx} samples)")

period_yr = TWOPI / BP2_SPEC['f_center']
print(f"\n  BP-2: fc={BP2_SPEC['f_center']:.2f} rad/yr, "
      f"T={period_yr:.2f} yr ({period_yr*52:.0f} wk)")
print(f"  Edges: [{BP2_SPEC['f1']:.2f}, {BP2_SPEC['f2']:.2f}, "
      f"{BP2_SPEC['f3']:.2f}, {BP2_SPEC['f4']:.2f}] rad/yr")
print()

# ============================================================================
# STEP 1: Baseline full filter (nw=1393, fs=52)
# ============================================================================

print("--- Step 1: Baseline full filter (nw=1393, fs=52) ---")

# f_edges in cycles/yr for ormsby_filter
f_edges_full = np.array([BP2_SPEC['f1'], BP2_SPEC['f2'],
                          BP2_SPEC['f3'], BP2_SPEC['f4']]) / TWOPI

h_full = ormsby_filter(nw=NW_FULL, f_edges=f_edges_full, fs=FS,
                       filter_type='bp', method='modulate', analytic=False)
result_full = apply_ormsby_filter(close_prices, h_full, mode='reflect', fs=FS)
output_full = result_full['signal']

print(f"  Kernel length: {len(h_full)}")
print(f"  Output range (display): [{output_full[s_idx:e_idx].min():.2f}, "
      f"{output_full[s_idx:e_idx].max():.2f}]")

# ============================================================================
# STEP 2: Decimate input, filter at reduced rate, place back
# ============================================================================

print(f"\n--- Step 2: Decimate-first approach (nw={NW_SHORT}, step={STEP}) ---")

fs_dec = FS / STEP
print(f"  Decimated sample rate: {fs_dec:.2f} samples/yr")
print(f"  Effective time span: {NW_SHORT} taps * {STEP} wk/tap = {NW_SHORT * STEP} weeks")

# Design filter using built-in spacing parameter (auto-adjusts fs)
h_short = ormsby_filter(nw=NW_SHORT, f_edges=f_edges_full, fs=FS, spacing=STEP,
                        filter_type='bp', method='modulate', analytic=False)
print(f"  Short kernel length: {len(h_short)}")

# Apply with built-in spacing/startidx for each start index
spaced_outputs = {}
for start in range(STEP):
    result = apply_ormsby_filter(close_prices, h_short, mode='reflect',
                                 fs=FS, spacing=STEP, startidx=start)
    spaced_outputs[start] = result['signal']

# Show stats for start=0 (default)
valid_disp = spaced_outputs[0][s_idx:e_idx]
valid_vals = valid_disp[~np.isnan(valid_disp)]
print(f"  Start=0 output range (display): [{valid_vals.min():.2f}, {valid_vals.max():.2f}]")
print(f"  Samples in display window: {len(valid_vals)} "
      f"(of {e_idx - s_idx} total positions)")

# ============================================================================
# STEP 2 PLOT: Comparison (full continuous vs spaced dots)
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(16, 5))

# Full filter as continuous black line
ax1.plot(disp_dates, output_full[s_idx:e_idx],
         'k-', linewidth=0.8, label=f'Full filter (nw={NW_FULL}, fs={FS})', alpha=0.7)

# Spaced output (start=0) as red dots
out_s0 = spaced_outputs[0][s_idx:e_idx]
ax1.plot(disp_dates, out_s0,
         'ro', markersize=3, label=f'Spaced (nw={NW_SHORT}, step={STEP}, start=0)',
         zorder=5)

ax1.axhline(0, color='gray', linewidth=0.3)
ax1.set_title(f"BP-2: Full (nw={NW_FULL}) vs Decimate-First "
              f"(nw={NW_SHORT}, step={STEP})")
ax1.set_ylabel('Amplitude')
ax1.set_xlabel('Date')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.2)
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.tight_layout()

out1 = os.path.join(script_dir, 'bp2_full_vs_spaced_comparison.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
print(f"  Saved: {out1}")

# ============================================================================
# STEP 3: Offset Sweep (start=0 through 6)
# ============================================================================

print(f"\n--- Step 3: Offset sweep (start=0 to {STEP - 1}) ---")

fig2, axes = plt.subplots(STEP, 1, figsize=(16, 14), sharex=True, sharey=True)

for start in range(STEP):
    ax = axes[start]

    # Background: full baseline
    ax.plot(disp_dates, output_full[s_idx:e_idx],
            '-', color='lightgray', linewidth=0.6)

    # Spaced dots
    out_s = spaced_outputs[start][s_idx:e_idx]
    ax.plot(disp_dates, out_s,
            'k.', markersize=3)

    ax.axhline(0, color='gray', linewidth=0.3)
    ax.set_ylabel(f'start={start}', fontsize=8, rotation=0, labelpad=35, ha='left')
    ax.grid(True, alpha=0.15)
    ax.tick_params(axis='y', labelsize=7)

axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].set_xlabel('Date')
plt.xticks(rotation=45)

fig2.suptitle(f"BP-2 Start Index Sweep: nw={NW_SHORT}, step={STEP}, start=0..{STEP - 1}\n"
              f"(gray = full nw={NW_FULL} baseline)",
              fontsize=11, fontweight='bold')
plt.tight_layout()

out2 = os.path.join(script_dir, 'bp2_offset_sweep_1to7.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")

# ============================================================================
# STEP 4: Best match overlay on reference image
# ============================================================================

print(f"\n--- Step 4: Overlay on reference image ---")

if os.path.exists(ref_image_path):
    ref_img = mpimg.imread(ref_image_path)
    img_h, img_w = ref_img.shape[:2]
    print(f"  Reference image: {img_w} x {img_h} pixels")

    # --- Alignment calibration ---
    # The dashed zero-line for BP-2 (extending from the "2" label) sits at
    # approximately 35% from the bottom of the image.  Everything above that
    # line includes the partially-visible lowpass ("1") trace.
    # Tunable: adjust ZERO_FRAC if the overlay zero-line doesn't match.
    ZERO_FRAC = 0.44  # fraction from image bottom where y=0 sits

    # Set y_min / y_max so that y=0 maps to ZERO_FRAC from bottom:
    #   y_min + ZERO_FRAC * (y_max - y_min) = 0
    #   => y_min = -ZERO_FRAC / (1 - ZERO_FRAC) * y_max
    y_max = 70.0   # large enough to cover filter-1 traces above zero
    y_min = -ZERO_FRAC / (1 - ZERO_FRAC) * y_max  # ~ -37.7

    disp_start_num = mdates.date2num(pd.to_datetime(DISPLAY_START))
    disp_end_num = mdates.date2num(pd.to_datetime(DISPLAY_END))

    print(f"  Overlay y-extent: [{y_min:.1f}, {y_max:.1f}]  "
          f"(zero line at {ZERO_FRAC*100:.0f}% from bottom)")

    # Generate overlay for each start index
    for start in range(STEP):
        fig3, ax3 = plt.subplots(figsize=(16, 5))

        ax3.imshow(ref_img, aspect='auto', alpha=0.5,
                   extent=[disp_start_num, disp_end_num, y_min, y_max],
                   zorder=0)

        # Overlay spaced output as red dots
        out_s = spaced_outputs[start][s_idx:e_idx]
        dates_num = mdates.date2num(disp_dates)
        ax3.plot(dates_num, out_s,
                 'r.', markersize=4,
                 label=f'Spaced output (start={start})', zorder=5)

        ax3.axhline(0, color='blue', linewidth=0.5, alpha=0.5, zorder=1)
        ax3.set_ylim(y_min, y_max)
        ax3.set_title(f"BP-2 Overlay on Reference -- start={start}", fontsize=10)
        ax3.set_ylabel('Amplitude')
        ax3.legend(fontsize=8)

        ax3.xaxis_date()
        ax3.xaxis.set_major_locator(mdates.YearLocator(2))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        out3 = os.path.join(script_dir, f'bp2_overlay_start{start}.png')
        fig3.savefig(out3, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out3}")
        plt.close(fig3)
else:
    print(f"  WARNING: Reference image not found at {ref_image_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("--- Summary ---")
print(f"  Full filter:  nw={NW_FULL}, fs={FS}, continuous output")
print(f"  Short filter: nw={NW_SHORT}, fs={fs_dec:.2f}, step={STEP}")
print(f"  Effective span: {NW_SHORT}*{STEP} = {NW_SHORT * STEP} weeks")
print(f"  Start indices tested: 0 to {STEP - 1}")
print(f"  Display: {DISPLAY_START} to {DISPLAY_END}")
print()
print("  Review the offset sweep plot to identify which start index best")
print("  matches the reference image, then check overlay plots.")
print()

plt.show()
print("Done.")
