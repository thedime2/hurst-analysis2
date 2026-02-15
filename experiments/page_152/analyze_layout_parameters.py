# -*- coding: utf-8 -*-
"""
Analyze Page 152 Unified Layout Parameters

Compare current unified layout output with reference images to identify
which parameters need refinement:
  - Filter frequencies (f1, f2, f3, f4)
  - Spacing (decimation)
  - Starting index (phase alignment)
  - Scaling/amplitude

Focus: BP-2 first (most cycles, easiest to verify)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.filters import ormsby_filter, apply_ormsby_filter

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')

DATE_START = '1921-04-29'
DATE_END = '1965-05-21'
DISPLAY_START = '1935-01-01'
DISPLAY_END = '1954-02-01'

FS = 52
TWOPI = 2 * np.pi

# Current filter specs (from reproduce_page152_layout.py)
BP2_CURRENT = {
    'f1': 0.85, 'f2': 1.25, 'f3': 2.05, 'f4': 2.45,
    'f_center': (1.25 + 2.05) / 2,
    'bandwidth': 2.05 - 1.25,
    'nw': 1393,
    'label': 'BP-2: ~3.8 yr (Current)',
}

# Nominal target from Phase 3 / Cyclitec
BP2_NOMINAL = {
    'f_center': 1.40,  # 54-month cycle
    'period': TWOPI / 1.40,  # ~4.5 years
    'label': 'BP-2 Nominal (54-month)',
}

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("Analyze Page 152 Layout Parameters")
print("=" * 70)
print()

df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between(DATE_START, DATE_END)].copy()
close_prices = df_hurst.Close.values
dates = df_hurst.Date.values

dates_dt = pd.to_datetime(dates)
mask = (dates_dt >= pd.to_datetime(DISPLAY_START)) & \
       (dates_dt <= pd.to_datetime(DISPLAY_END))
disp_idx = np.where(mask)[0]
s_idx, e_idx = disp_idx[0], disp_idx[-1] + 1
disp_dates = dates_dt[s_idx:e_idx]

print(f"Loaded {len(close_prices)} samples ({DATE_START} to {DATE_END})")
print(f"Display window: {DISPLAY_START} to {DISPLAY_END} ({e_idx - s_idx} samples)")
print()

# ============================================================================
# ANALYSIS 1: Filter Frequency Comparison
# ============================================================================

print("=" * 70)
print("Analysis 1: Filter Frequency")
print("=" * 70)
print()

print(f"BP-2 Current (visual estimate):")
print(f"  Frequencies: f1={BP2_CURRENT['f1']:.2f}, f2={BP2_CURRENT['f2']:.2f}, "
      f"f3={BP2_CURRENT['f3']:.2f}, f4={BP2_CURRENT['f4']:.2f}")
print(f"  Center: {BP2_CURRENT['f_center']:.2f} rad/yr")
print(f"  Bandwidth: {BP2_CURRENT['bandwidth']:.2f} rad/yr")
print(f"  Period: {TWOPI/BP2_CURRENT['f_center']:.2f} years")
print()

print(f"BP-2 Nominal (Cyclitec 54-month target):")
print(f"  Center: {BP2_NOMINAL['f_center']:.2f} rad/yr")
print(f"  Period: {BP2_NOMINAL['period']:.2f} years")
print()

delta = BP2_CURRENT['f_center'] - BP2_NOMINAL['f_center']
pct = (delta / BP2_NOMINAL['f_center']) * 100
print(f"Difference: {delta:+.2f} rad/yr ({pct:+.1f}%)")
print(f"  >> Current is HIGHER in frequency (shifts toward faster cycles)")
print()

# ============================================================================
# ANALYSIS 2: Check for Spacing Clues in Reference
# ============================================================================

print("=" * 70)
print("Analysis 2: Spacing / Decimation")
print("=" * 70)
print()

print("Reference image observations:")
print("  - BP-2 (row 2): Appears to show ~1 sample per week (continuous)")
print("  - BP-3 (row 3): Faster oscillations, appears continuous")
print("  - BP-4 (row 4): Very fast, may show spacing")
print("  - BP-5 (row 5): High frequency, may show spacing")
print("  - BP-6 (row 6): Highest frequency, may show spacing")
print()

print("Previous investigation (investigate_bp2_spacing.py):")
print("  - BP-2 with spacing=7 showed good alignment")
print("  - nw_short = nw_full / spacing = 1393 / 7 = 199")
print()

print("Hypothesis for Part 2:")
print("  1. BP-2: spacing=1 (or small) to match continuous appearance")
print("  2. BP-3,4: spacing=3-5 (moderate decimation)")
print("  3. BP-5,6: spacing=7-9 (more decimation for sparse appearance)")
print()

# ============================================================================
# ANALYSIS 3: Amplitude / Scaling
# ============================================================================

print("=" * 70)
print("Analysis 3: Scaling & Normalization")
print("=" * 70)
print()

# Apply BP-2 filter to get statistics
f_edges = np.array([BP2_CURRENT['f1'], BP2_CURRENT['f2'],
                    BP2_CURRENT['f3'], BP2_CURRENT['f4']]) / TWOPI
h = ormsby_filter(nw=BP2_CURRENT['nw'], f_edges=f_edges, fs=FS,
                  filter_type='bp', method='modulate', analytic=False)
result = apply_ormsby_filter(close_prices, h, mode='reflect', fs=FS)
bp2_output = result['signal'][s_idx:e_idx]

print(f"BP-2 Output Statistics (display window):")
print(f"  Mean: {np.mean(bp2_output):.2f}")
print(f"  Std: {np.std(bp2_output):.2f}")
print(f"  Min: {np.min(bp2_output):.2f}")
print(f"  Max: {np.max(bp2_output):.2f}")
print(f"  Range: {np.max(bp2_output) - np.min(bp2_output):.2f}")
print()

# Z-score normalized
bp2_zscore = (bp2_output - np.mean(bp2_output)) / np.std(bp2_output)
print(f"BP-2 Z-score normalized:")
print(f"  Mean: {np.mean(bp2_zscore):.4f}")
print(f"  Std: {np.std(bp2_zscore):.4f}")
print(f"  Range: [{np.min(bp2_zscore):.2f}, {np.max(bp2_zscore):.2f}]")
print()

print("Current scaling in reproduce_page152_layout.py:")
print("  scales['bp2'] = 30.0  (multiply z-score by 30)")
print(f"  Resulting range: [{30*np.min(bp2_zscore):.1f}, {30*np.max(bp2_zscore):.1f}]")
print()

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================

print("=" * 70)
print("Summary & Recommendations for Part 2")
print("=" * 70)
print()

print("Current Status (from Part 1):")
print("  [OK] Layout matches reference visually")
print("  [OK] Numbered labels (1-6) on left side")
print("  [OK] Vertical offsets create readable channels")
print()

print("Part 2 Refinement Strategy:")
print()
print("Phase 2A: BP-2 Refinement (focus first)")
print("  1. Visual comparison of current vs reference")
print("  2. Sweep: spacing (1, 3, 5, 7) × startidx (0..spacing-1)")
print("  3. For each combo, generate plot and score visual alignment")
print("  4. Lock in best BP-2 parameters")
print()

print("Phase 2B: BP-3 to BP-6 Refinement")
print("  1. Apply same strategy to each filter")
print("  2. May require higher spacing for higher frequencies")
print("  3. Cross-check: sum of filters captures ~96% energy")
print()

print("Phase 2C: LP-1 and Final Integration")
print("  1. Verify LP1 tracks trend without excessive lag")
print("  2. Generate final unified layout with refined parameters")
print("  3. Save parameters to CSV")
print()

print("Phase 3: Analysis")
print("  1. Compare refined to nominal model frequencies")
print("  2. Cross-check with Lanczos spectrum")
print("  3. Update documentation with findings")
print()

print("=" * 70)
print()
