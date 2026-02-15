# -*- coding: utf-8 -*-
"""
Page 152: Unified Layout Reproduction with Vertical Offsets

Reproduces Hurst's six-filter decomposition using a single plot with vertical
offsets, matching the visual layout of the reference figure.

Layout:
  - Top: DJIA Close price (black solid line)
  - Overlay: LP1 lowpass filter (blue dashed line)
  - Below: BP2, BP3, BP4, BP5, BP6 each vertically offset with their own scale

All filters share the same x-axis (time) and display window (1935-1954).

Filter specifications are from visual estimates of Hurst's graphics.
This reproduction aims to match the visual layout first, then will be
refined iteratively to match the reference output.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           p. 152, Figure IX-4
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.filters import (
    ormsby_filter,
    apply_ormsby_filter,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')

# Hurst's analysis window (use full dataset for filter computation)
DATE_START = '1921-04-29'
DATE_END = '1965-05-21'

# Display window (visible portion of plot)
DISPLAY_START = '1935-01-01'
DISPLAY_END = '1954-02-01'

# Data parameters
FS = 52
TWOPI = 2 * np.pi

# ============================================================================
# FILTER SPECIFICATIONS (all frequencies in rad/year)
# ============================================================================

# User-estimated values from visual comparison with Hurst's page 152 graphics
FILTER_SPECS = [
    {
        'type': 'lp',
        'f_pass': 0.85,
        'f_stop': 1.25,
        'f_center': (0.85 + 1.25) / 2,
        'bandwidth': 1.25 - 0.85,
        'nw': 1393,
        'index': 0,
        'label': 'LP-1: Trend (>5 yr)',
        'color': 'blue',
        'linestyle': '--',  # Dashed for overlay
        'linewidth': 1.5,
    },
    {
        'type': 'bp',
        'f1': 0.85, 'f2': 1.25, 'f3': 2.05, 'f4': 2.45,
        'f_center': (1.25 + 2.05) / 2,
        'bandwidth': 2.05 - 1.25,
        'nw': 1393,
        'index': 1,
        'label': 'BP-2: ~3.8 yr',
        'color': 'black',
        'linestyle': '-',
        'linewidth': 0.8,
    },
    {
        'type': 'bp',
        'f1': 3.20, 'f2': 3.55, 'f3': 6.35, 'f4': 6.70,
        'f_center': (3.55 + 6.35) / 2,
        'bandwidth': 6.35 - 3.55,
        'nw': 1245,
        'index': 2,
        'label': 'BP-3: ~1.3 yr',
        'color': 'black',
        'linestyle': '-',
        'linewidth': 0.8,
    },
    {
        'type': 'bp',
        'f1': 7.25, 'f2': 7.55, 'f3': 9.55, 'f4': 9.85,
        'f_center': (7.55 + 9.55) / 2,
        'bandwidth': 9.55 - 7.55,
        'nw': 1745,
        'index': 3,
        'label': 'BP-4: ~0.7 yr',
        'color': 'black',
        'linestyle': '-',
        'linewidth': 0.8,
    },
    {
        'type': 'bp',
        'f1': 13.65, 'f2': 13.95, 'f3': 19.35, 'f4': 19.65,
        'f_center': (13.95 + 19.35) / 2,
        'bandwidth': 19.35 - 13.95,
        'nw': 1299,
        'index': 4,
        'label': 'BP-5: ~0.4 yr',
        'color': 'black',
        'linestyle': '-',
        'linewidth': 0.8,
    },
    {
        'type': 'bp',
        'f1': 28.45, 'f2': 28.75, 'f3': 35.95, 'f4': 36.25,
        'f_center': (28.75 + 35.95) / 2,
        'bandwidth': 35.95 - 28.75,
        'nw': 1299,
        'index': 5,
        'label': 'BP-6: ~0.2 yr',
        'color': 'black',
        'linestyle': '-',
        'linewidth': 0.8,
    },
]

# ============================================================================
# PRINT FILTER SUMMARY
# ============================================================================

print("=" * 70)
print("Page 152: Six-Filter Decomposition — Unified Layout Reproduction")
print("=" * 70)
print()

for spec in FILTER_SPECS:
    if spec['type'] == 'lp':
        period = TWOPI / spec['f_pass']
        print(f"  {spec['label']:20s}  LP: pass<{spec['f_pass']:.2f}, "
              f"stop>{spec['f_stop']:.2f} rad/yr  (T>{period:.1f} yr)  nw={spec['nw']}")
    else:
        period = TWOPI / spec['f_center']
        print(f"  {spec['label']:20s}  BP: [{spec['f1']:.2f}, {spec['f2']:.2f}, "
              f"{spec['f3']:.2f}, {spec['f4']:.2f}] rad/yr  "
              f"fc={spec['f_center']:.2f}  T={period:.2f} yr  nw={spec['nw']}")
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
# CREATE FILTER KERNELS
# ============================================================================

print("Creating filter kernels...")
kernels = []
for spec in FILTER_SPECS:
    if spec['type'] == 'lp':
        f_edges = np.array([spec['f_pass'], spec['f_stop']]) / TWOPI
        h = ormsby_filter(nw=spec['nw'], f_edges=f_edges, fs=FS,
                          filter_type='lp', analytic=False)
    else:
        f_edges = np.array([spec['f1'], spec['f2'],
                            spec['f3'], spec['f4']]) / TWOPI
        h = ormsby_filter(nw=spec['nw'], f_edges=f_edges, fs=FS,
                          filter_type='bp', method='modulate',
                          analytic=False)
    kernels.append({
        'kernel': h,
        'spec': spec,
        'nw': spec['nw']
    })
    print(f"  Created {spec['label']:20s}  kernel size: {len(h)}")

print()

# ============================================================================
# APPLY FILTERS
# ============================================================================

print("Applying filters to DJIA data...")
outputs = []
for filt in kernels:
    result = apply_ormsby_filter(close_prices, filt['kernel'],
                                 mode='reflect', fs=FS)
    result['spec'] = filt['spec']
    outputs.append(result)

print(f"  Applied {len(outputs)} filters")
print()

# ============================================================================
# NORMALIZATION FUNCTION
# ============================================================================

def normalize_signal(sig, method='zscore'):
    """
    Normalize signal for display.

    method='zscore': (x - mean) / std
    method='minmax': (x - min) / (max - min), then scale to [-1, 1]
    """
    if np.all(sig == 0):
        return sig

    if method == 'zscore':
        mean = np.mean(sig)
        std = np.std(sig)
        if std > 0:
            return (sig - mean) / std
        else:
            return sig - mean
    elif method == 'minmax':
        vmin = np.min(sig)
        vmax = np.max(sig)
        if vmax > vmin:
            normalized = 2 * (sig - vmin) / (vmax - vmin) - 1
        else:
            normalized = sig - vmin
        return normalized
    else:
        return sig

# ============================================================================
# UNIFIED LAYOUT PLOT
# ============================================================================

print("Creating unified layout plot...")

fig, ax = plt.subplots(figsize=(16, 10))

# Vertical offsets for each filter (in data units after normalization)
offsets = {
    'price': 0,      # Top (no offset)
    'lp1': 0,        # Overlaid on price (same level)
    'bp2': -100,     # First offset below
    'bp3': -200,
    'bp4': -300,
    'bp5': -400,
    'bp6': -500,
}

# Scaling factors for each signal (adjust amplitude for visibility)
scales = {
    'price': 1.0,     # Price in original units
    'lp1': 1.0,       # Same scale as price
    'bp2': 30.0,      # Bandpass: scale up for visibility
    'bp3': 30.0,
    'bp4': 30.0,
    'bp5': 30.0,
    'bp6': 30.0,
}

# Plot DJIA price (top)
price_normalized = normalize_signal(disp_prices, method='zscore')
price_scaled = price_normalized * scales['price'] + offsets['price']
ax.plot(disp_dates, price_scaled, 'k-', linewidth=1.2, zorder=3)

# Plot each filter output with vertical offset
filter_names = ['lp1', 'bp2', 'bp3', 'bp4', 'bp5', 'bp6']
filter_nums = [1, 2, 3, 4, 5, 6]  # Display numbers

for idx, (filt_output, filt_name, filt_num) in enumerate(zip(outputs, filter_names, filter_nums)):
    spec = filt_output['spec']
    sig = filt_output['signal']

    # Extract real part if complex
    if np.iscomplexobj(sig):
        sig_real = sig.real
    else:
        sig_real = sig

    # Extract display window
    sig_display = sig_real[s_idx:e_idx]

    # Normalize and scale
    sig_normalized = normalize_signal(sig_display, method='zscore')
    sig_scaled = sig_normalized * scales[filt_name] + offsets[filt_name]

    # Plot
    ax.plot(disp_dates, sig_scaled,
            color=spec['color'],
            linestyle=spec['linestyle'],
            linewidth=spec['linewidth'],
            alpha=0.9,
            zorder=2)

    # Add horizontal reference line at baseline
    ax.axhline(offsets[filt_name], color='gray', linewidth=0.4,
               linestyle='-', alpha=0.2)

    # Add filter number label on the left (using axes fraction coordinates)
    ax.text(-0.08, (offsets[filt_name] - offsets['bp6']) / (offsets['price'] - offsets['bp6']),
            str(filt_num),
            transform=ax.transAxes,
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                      edgecolor='black', linewidth=1),
            zorder=4)

# Formatting
ax.set_xlabel('Date', fontsize=11, fontweight='bold')
ax.set_ylabel('Amplitude (normalized)', fontsize=11, fontweight='bold')
ax.set_title('Page 152: Six-Filter Structural Decomposition\nUnified Layout',
             fontsize=13, fontweight='bold', pad=15)

# X-axis date formatting
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45, ha='right')

# Grid
ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.4, axis='x')
ax.set_axisbelow(True)

# Y-axis limits (leave some margin)
ax.set_ylim(offsets['bp6'] - 50, offsets['price'] + 50)

# Clean up y-axis (not needed since we have left labels)
ax.set_yticks([])

plt.tight_layout()

# Save
out_path = os.path.join(script_dir, 'page152_unified_layout.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {out_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("Plot Generation Complete")
print("=" * 70)
print()
print("Output: page152_unified_layout.png")
print()
print("Next steps:")
print("  1. Compare with reference images:")
print("     - references/page_152/filter_decomposition_v2.png")
print("     - references/page_152/filter_decomposition_filters2to5.png")
print()
print("  2. BP-2 refinement results (Phase 2A):")
print("     - Best match: spacing=7, startidx=2 (beat_score=0.594)")
print("     - See: bp2_refinement/ folder for comparison plots")
print()
print("  3. Next: Apply refinement to BP-3 through BP-6")
print()

