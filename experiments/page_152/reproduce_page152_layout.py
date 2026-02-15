# -*- coding: utf-8 -*-
"""
Page 152: Unified Layout Reproduction with Vertical Offsets

Reproduces Hurst's six-filter decomposition using a single plot with vertical
offsets, matching the visual layout of the reference figure.

Layout:
  - Top: DJIA Close price (black solid line) + LP1 overlay (dashed)
  - Below: BP2, BP3, BP4, BP5, BP6 each vertically offset, ALL ON SAME SCALE

All filters share the same x-axis (time) and display window (1935-1954).
Bandpass filters are plotted on the SAME absolute amplitude scale so that
visual amplitudes faithfully reflect actual signal energy per band.

Can optionally overlay on the scanned reference image for alignment.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           p. 152, Figure IX-4
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-blocking, file-only rendering
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg

from src.filters import ormsby_filter, apply_ormsby_filter
from src.filters.decimation import decimate_signal

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
ref_image_path = os.path.join(script_dir, '../../references/page_152/filter_decomposition_v2.png')

# Hurst's analysis window (use full dataset for filter computation)
DATE_START = '1921-04-29'
DATE_END = '1965-05-21'

# Display window (visible portion of plot)
DISPLAY_START = '1934-12-25'
DISPLAY_END = '1952-01-28'

# Data parameters
FS = 52
TWOPI = 2 * np.pi

# Toggle reference image overlay
OVERLAY_REFERENCE = True

# ============================================================================
# FILTER SPECIFICATIONS (all frequencies in rad/year)
#
# spacing: decimation factor (1 = no decimation)
# startidx: 0-based starting index for decimation
# nw: filter length AT FULL RATE (nw / spacing used for decimated filter)
# ============================================================================

FILTER_SPECS = [
    {
        'type': 'lp',
        'f_pass': 0.85,
        'f_stop': 1.25,
        'nw': 1393,
        'spacing': 1,
        'startidx': 0,
        'index': 0,
        'label': '1',
    },
    {
        'type': 'bp',
        'f1': 0.85, 'f2': 1.25, 'f3': 2.05, 'f4': 2.45,
        'nw': 1393,
        'spacing': 7,
        'startidx': 2,
        'index': 1,
        'label': '2',
    },
    {
        'type': 'bp',
        'f1': 3.20, 'f2': 3.55, 'f3': 6.35, 'f4': 6.70,
        'nw': 1245,
        'spacing': 6,
        'startidx': 2,
        'index': 2,
        'label': '3',
    },
    {
        'type': 'bp',
        'f1': 7.25, 'f2': 7.55, 'f3': 9.55, 'f4': 9.85,
        'nw': 1745,
        'spacing': 2,
        'startidx': 0,
        'index': 3,
        'label': '4',
    },
    {
        'type': 'bp',
        'f1': 13.65, 'f2': 13.95, 'f3': 19.35, 'f4': 19.65,
        'nw': 1299,
        'spacing': 2,
        'startidx': 0,
        'index': 4,
        'label': '5',
    },
    {
        'type': 'bp',
        'f1': 28.45, 'f2': 28.75, 'f3': 35.95, 'f4': 36.25,
        'nw': 1299,
        'spacing': 1,
        'startidx': 0,
        'index': 5,
        'label': '6',
    },
]

# ============================================================================
# PRINT FILTER SUMMARY
# ============================================================================

print("=" * 70)
print("Page 152: Unified Layout Reproduction")
print("=" * 70)
print()

for spec in FILTER_SPECS:
    spacing = spec.get('spacing', 1)
    nw_eff = int(spec['nw'] / spacing) if spacing > 1 else spec['nw']
    if nw_eff % 2 == 0:
        nw_eff += 1
    sp_str = f"  spacing={spacing}" if spacing > 1 else ""

    if spec['type'] == 'lp':
        print(f"  Filter {spec['label']:2s}: LP  pass<{spec['f_pass']:.2f} rad/yr  "
              f"nw={nw_eff}{sp_str}")
    else:
        fc = (spec['f2'] + spec['f3']) / 2
        T = TWOPI / fc
        print(f"  Filter {spec['label']:2s}: BP  [{spec['f1']:.2f}-{spec['f4']:.2f}] rad/yr  "
              f"fc={fc:.2f}  T={T:.2f}yr  nw={nw_eff}{sp_str}")
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading weekly DJIA data...")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between(DATE_START, DATE_END)].copy()
close_prices = df_hurst.Close.values
dates = df_hurst.Date.values

n_points = len(close_prices)
print(f"  Loaded {n_points} samples from {DATE_START} to {DATE_END}")

# Get display window indices
dates_dt = pd.to_datetime(dates)
mask = (dates_dt >= pd.to_datetime(DISPLAY_START)) & \
       (dates_dt <= pd.to_datetime(DISPLAY_END))
disp_idx = np.where(mask)[0]
s_idx, e_idx = disp_idx[0], disp_idx[-1] + 1
disp_dates = dates_dt[s_idx:e_idx]
disp_prices = close_prices[s_idx:e_idx]

print(f"  Display window: {DISPLAY_START} to {DISPLAY_END} ({e_idx - s_idx} samples)")
print()

# ============================================================================
# CREATE AND APPLY FILTERS (with spacing support)
# ============================================================================

print("Creating and applying filters...")
outputs = []

for spec in FILTER_SPECS:
    spacing = spec.get('spacing', 1)
    startidx = spec.get('startidx', 0)

    # Effective filter length for decimated rate
    nw = spec['nw']
    if spacing > 1:
        nw = int(nw / spacing)
        if nw % 2 == 0:
            nw += 1
    fs_eff = FS / spacing if spacing > 1 else FS

    # Build kernel
    if spec['type'] == 'lp':
        f_edges = np.array([spec['f_pass'], spec['f_stop']]) / TWOPI
        h = ormsby_filter(nw=nw, f_edges=f_edges, fs=fs_eff,
                          filter_type='lp', analytic=False)
    else:
        f_edges = np.array([spec['f1'], spec['f2'],
                            spec['f3'], spec['f4']]) / TWOPI
        h = ormsby_filter(nw=nw, f_edges=f_edges, fs=fs_eff,
                          filter_type='bp', method='modulate', analytic=False)

    # Decimate input if spacing > 1
    if spacing > 1:
        decimated, indices = decimate_signal(close_prices, spacing, startidx + 1)
        result = apply_ormsby_filter(decimated, h, mode='reflect', fs=fs_eff)
        # Place back into full-length array with NaN gaps
        full_output = np.full(n_points, np.nan, dtype=float)
        for out_idx, orig_idx in enumerate(indices):
            if out_idx < len(result['signal']):
                full_output[orig_idx] = result['signal'][out_idx]
        result['signal'] = full_output
    else:
        result = apply_ormsby_filter(close_prices, h, mode='reflect', fs=FS)

    result['spec'] = spec
    outputs.append(result)

    label = f"Filter {spec['label']}"
    if spacing > 1:
        print(f"  {label}: nw={nw}, spacing={spacing}, startidx={startidx}")
    else:
        print(f"  {label}: nw={nw}")

print()

# ============================================================================
# COMPUTE SAME-SCALE AMPLITUDE FOR ALL BANDPASS FILTERS
# ============================================================================

# Find max absolute amplitude across ALL bandpass outputs in the display window
# This ensures all 5 bandpass filters use the same y-scale
bp_max_amp = 0
for out in outputs:
    if out['spec']['type'] == 'bp':
        sig = out['signal'][s_idx:e_idx]
        valid = sig[~np.isnan(sig)] if np.any(np.isnan(sig)) else sig
        if len(valid) > 0:
            bp_max_amp = max(bp_max_amp, np.max(np.abs(valid)))

print(f"Bandpass same-scale max amplitude: {bp_max_amp:.2f}")
print("  (All 5 bandpass filters will use this as +/- range)")
print()

# ============================================================================
# UNIFIED LAYOUT PLOT
# ============================================================================

print("Creating unified layout plot...")

fig, ax = plt.subplots(figsize=(16, 12))

# =========================================================================
# MANUAL TUNING PARAMETERS -- edit these to align with reference overlay
# =========================================================================
#
# Vertical axis goes from 0 (bottom) to 600 (top).
# Each value below is the vertical CENTER of that row.

PRICE_CENTER = 367          # << TWEAK: vertical center of price+LP1
PRICE_HEIGHT = 300          # << TWEAK: vertical extent for price data

# larger equals higher vertically
BP_CENTERS = {              # << TWEAK: vertical center of each BP row
    'bp2': 205,
    'bp3': 127,
    'bp4':  85,
    'bp5':  60,
    'bp6':  40,
}

BP_SCALE = 1.85              # << TWEAK: data-units per DJIA-point
                            #    Same value for ALL 5 BP filters = TRUE same scale
                            #    Larger = taller oscillations

# Horizontal framing pads (adds blank space without removing any data points)
# DF 50 / 17 align top most ones 
# but 58  / 13 caters for warp in last few

X_PAD_LEFT_WEEKS = 53      # << TWEAK: extra space before first sample
X_PAD_RIGHT_WEEKS = 13    # << TWEAK: extra space after last sample

# Reference image overlay position (shifts/stretches image within framed axes)
REF_X_SHIFT = 0             # << TWEAK: days to shift image right (+) or left (-)
REF_Y_SHIFT = 0             # << TWEAK: data-units to shift image up (+) or down (-)
REF_X_SCALE = 1.0           # << TWEAK: >1 stretches image wider
REF_Y_SCALE = 1.0           # << TWEAK: >1 stretches image taller
REF_ALPHA   = 0.6          # << TWEAK: overlay transparency (0=invisible, 1=opaque)


# =========================================================================
# END TUNING PARAMETERS
# =========================================================================

PLOT_TOP = 600
PLOT_BOTTOM = 0
bp_names = ['bp2', 'bp3', 'bp4', 'bp5', 'bp6']
xlim_left = disp_dates[0] - pd.Timedelta(weeks=X_PAD_LEFT_WEEKS)
xlim_right = disp_dates[-1] + pd.Timedelta(weeks=X_PAD_RIGHT_WEEKS)

# --- Plot DJIA price at top ---
p = disp_prices
p_min, p_max = np.min(p), np.max(p)
p_bottom = PRICE_CENTER - PRICE_HEIGHT / 2
p_scaled = (p - p_min) / (p_max - p_min) * PRICE_HEIGHT + p_bottom
ax.plot(disp_dates, p_scaled, 'k-', linewidth=1.0, zorder=3)
#
# --- Plot LP-1 overlay on price ---
lp_sig = outputs[0]['signal'][s_idx:e_idx]
lp_scaled = (lp_sig - p_min) / (p_max - p_min) * PRICE_HEIGHT + p_bottom
ax.plot(disp_dates, lp_scaled, 'b--', linewidth=1.5, alpha=0.8, zorder=2)

# Label "1"
ax.text(0.01, (PRICE_CENTER - PLOT_BOTTOM) / (PLOT_TOP - PLOT_BOTTOM),
        '1', transform=ax.transAxes, fontsize=10, ha='center', va='center',
        fontweight='bold', zorder=5)

# --- Plot 5 bandpass filters on TRUE SAME SCALE ---
# BP_SCALE is data-units per DJIA-point, identical for all 5 rows.
# This means a 10-point oscillation in BP-3 looks the same height as
# a 10-point oscillation in BP-6.
bp_outputs = outputs[1:]
bp_labels = ['2', '3', '4', '5', '6']

print(f"  BP same-scale factor: {BP_SCALE:.1f} data-units per DJIA-point")
print(f"  Max BP amplitude: {bp_max_amp:.2f} DJIA-points")
print(f"  Max visual extent: +/- {bp_max_amp * BP_SCALE:.1f} data-units")

for bp_out, bp_name, bp_label in zip(bp_outputs, bp_names, bp_labels):
    center_y = BP_CENTERS[bp_name]
    sig = bp_out['signal'][s_idx:e_idx]

    # TRUE same scale: identical data-units-per-DJIA-point for every row
    sig_scaled = sig * BP_SCALE + center_y

    # Handle NaN (from spacing) - plot as dots
    has_nans = np.any(np.isnan(sig))
    if has_nans:
        valid = ~np.isnan(sig_scaled)
        ax.plot(disp_dates[valid], sig_scaled[valid], 'k.', markersize=2, zorder=2)
    else:
        ax.plot(disp_dates, sig_scaled, 'k-', linewidth=0.7, zorder=2)

    # Horizontal baseline
    ax.axhline(center_y, color='gray', linewidth=0.3, linestyle='-', alpha=0.3)

    # Label
    y_frac = (center_y - PLOT_BOTTOM) / (PLOT_TOP - PLOT_BOTTOM)
    ax.text(0.01, y_frac, bp_label,
            transform=ax.transAxes, fontsize=10, ha='center', va='center',
            fontweight='bold', zorder=5)

# --- Overlay reference image (full, uncropped) ---
if OVERLAY_REFERENCE and os.path.exists(ref_image_path):
    print(f"  Overlaying reference image: {os.path.basename(ref_image_path)}")
    ref_img = mpimg.imread(ref_image_path)

    # Default extent = fill framed x-range; then apply shift/scale
    x_left = mdates.date2num(xlim_left)
    x_right = mdates.date2num(xlim_right)
    x_mid = (x_left + x_right) / 2
    y_mid = (PLOT_BOTTOM + PLOT_TOP) / 2
    x_half = (x_right - x_left) / 2 * REF_X_SCALE
    y_half = (PLOT_TOP - PLOT_BOTTOM) / 2 * REF_Y_SCALE

    ext = [x_mid - x_half + REF_X_SHIFT,
           x_mid + x_half + REF_X_SHIFT,
           y_mid - y_half + REF_Y_SHIFT,
           y_mid + y_half + REF_Y_SHIFT]

    ax.imshow(ref_img, aspect='auto', alpha=REF_ALPHA, extent=ext, zorder=0)

# --- Formatting ---
ax.set_xlabel('Date', fontsize=11, fontweight='bold')
ax.set_title('Page 152: Six-Filter Structural Decomposition (Figure IX-4)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(xlim_left, xlim_right)

ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45, ha='right', fontsize=9)

ax.grid(True, alpha=0.95, linestyle='-', linewidth=0.4, axis='x')
ax.set_axisbelow(True)

ax.set_ylim(PLOT_BOTTOM, PLOT_TOP)
ax.set_yticks([])

plt.tight_layout()

# Save
out_path = os.path.join(script_dir, 'page152_unified_layout.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {out_path}")

# Also save without overlay
if OVERLAY_REFERENCE:
    for child in ax.get_children():
        if isinstance(child, matplotlib.image.AxesImage):
            child.set_visible(False)
    out_clean = os.path.join(script_dir, 'page152_unified_layout_clean.png')
    fig.savefig(out_clean, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_clean}")

plt.close(fig)

print()
print("=" * 70)
print("Done.")
print("=" * 70)
print()
print(f"  TRUE same-scale: {BP_SCALE:.1f} data-units per DJIA-point for all 5 BP filters")
print(f"  Max BP amplitude: +/- {bp_max_amp:.2f} DJIA points")
if OVERLAY_REFERENCE:
    print(f"  Reference overlay: ON (alpha={REF_ALPHA})")
print()
