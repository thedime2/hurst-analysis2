# -*- coding: utf-8 -*-
"""
Phase 2A: BP-2 Parameter Refinement

Systematically search for optimal BP-2 parameters by sweeping:
  - Spacing (decimation factor): 1, 3, 5, 7
  - Starting index (phase alignment): 0 to spacing-1
  - Optionally: frequency shifts (small adjustments to f1,f2,f3,f4)

Strategy:
  1. For each spacing x startidx combination, generate unified layout plot
  2. Compare visually with reference image (BP-2 row)
  3. Compute quantitative alignment metrics
  4. Rank combinations by fit quality
  5. Select best BP-2 parameters

Reference: BP-2 should show clear oscillations with modulation envelope
  - ~3.8 year period (1.65 rad/yr center)
  - Clear beat patterns from sideband modulation
  - Visual alignment with reference row 2
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

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

# BP-2 current specification
BP2_SPEC = {
    'f1': 0.85, 'f2': 1.25, 'f3': 2.05, 'f4': 2.45,
    'f_center': (1.25 + 2.05) / 2,
    'bandwidth': 2.05 - 1.25,
    'nw': 1393,
    'label': 'BP-2: ~3.8 yr',
}

# Parameter search space
SPACING_VALUES = [1, 3, 5, 7]
FREQ_SHIFTS = [0.0]  # No frequency shift in first pass; can add +/-0.1, 0.2 later

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("Phase 2A: BP-2 Parameter Refinement")
print("=" * 80)
print()

print("Loading data...")
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

print(f"  Loaded {len(close_prices)} samples")
print(f"  Display window: {e_idx - s_idx} samples ({DISPLAY_START} to {DISPLAY_END})")
print()

# ============================================================================
# HELPER: Normalize signal
# ============================================================================

def normalize_signal(sig, method='zscore'):
    """Normalize signal for display."""
    if np.all(sig == 0):
        return sig
    if method == 'zscore':
        mean = np.mean(sig)
        std = np.std(sig)
        return (sig - mean) / std if std > 0 else (sig - mean)
    elif method == 'minmax':
        vmin, vmax = np.min(sig), np.max(sig)
        return 2 * (sig - vmin) / (vmax - vmin) - 1 if vmax > vmin else (sig - vmin)
    return sig

# ============================================================================
# HELPER: Compute quantitative metrics
# ============================================================================

def compute_cycle_count(signal, fs=52):
    """Count approximate cycles in signal (using zero crossings)."""
    if len(signal) < 2:
        return 0
    crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(crossings) // 2  # Each cycle has 2 zero crossings

def compute_amplitude_stats(signal):
    """Compute peak amplitude statistics."""
    peaks = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0] + 1
    if len(peaks) == 0:
        return 0, 0
    peak_heights = np.abs(signal[peaks])
    return np.mean(peak_heights), np.std(peak_heights)

def compute_beat_signature(signal, threshold=0.5):
    """
    Detect beat signature (modulated envelope).
    Returns a simple score: fraction of peaks with modulation.
    """
    peaks = np.where((signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:]))[0] + 1
    if len(peaks) < 4:
        return 0

    # Check if peak heights vary (indicating modulation/beats)
    peak_heights = np.abs(signal[peaks])
    cv = np.std(peak_heights) / (np.mean(peak_heights) + 1e-10)  # Coefficient of variation
    return min(cv, 1.0)  # Cap at 1.0

# ============================================================================
# MAIN REFINEMENT LOOP
# ============================================================================

results = []

print("=" * 80)
print("Parameter Search: Spacing x Starting Index")
print("=" * 80)
print()

for spacing in SPACING_VALUES:
    for startidx in range(spacing):
        for freq_shift in FREQ_SHIFTS:

            # Adjust frequencies
            f1 = BP2_SPEC['f1'] + freq_shift
            f2 = BP2_SPEC['f2'] + freq_shift
            f3 = BP2_SPEC['f3'] + freq_shift
            f4 = BP2_SPEC['f4'] + freq_shift

            # Adjust filter length for spacing
            nw = int(BP2_SPEC['nw'] / spacing)
            if nw % 2 == 0:
                nw += 1  # Keep odd

            # Create filter
            f_edges = np.array([f1, f2, f3, f4]) / TWOPI
            h = ormsby_filter(nw=nw, f_edges=f_edges, fs=FS/spacing,
                            filter_type='bp', method='modulate', analytic=False)

            # Apply filter with spacing
            from src.filters.decimation import decimate_signal, interpolate_sparse

            # Decimate input
            # Note: decimate_signal uses 1-based offset (offset = startidx + 1)
            if spacing > 1:
                decimated, indices = decimate_signal(close_prices, spacing, startidx + 1)
            else:
                decimated = close_prices.copy()
                indices = np.arange(len(close_prices))

            # Apply filter
            result = apply_ormsby_filter(decimated, h, mode='reflect', fs=FS/spacing)
            bp2_filtered = result['signal']

            # Place back with NaN spacing
            if spacing > 1:
                bp2_output = np.full_like(close_prices, np.nan, dtype=float)
                for out_idx, orig_idx in enumerate(indices):
                    if out_idx < len(bp2_filtered):
                        bp2_output[orig_idx] = bp2_filtered[out_idx]
            else:
                bp2_output = bp2_filtered

            # Extract display window
            bp2_display = bp2_output[s_idx:e_idx]

            # Compute metrics
            valid_mask = ~np.isnan(bp2_display)
            valid_signal = bp2_display[valid_mask]

            if len(valid_signal) > 10:
                cycles = compute_cycle_count(valid_signal, FS/spacing)
                mean_amp, std_amp = compute_amplitude_stats(valid_signal)
                beat_score = compute_beat_signature(valid_signal)
                data_points = np.sum(valid_mask)
            else:
                cycles = 0
                mean_amp = 0
                std_amp = 0
                beat_score = 0
                data_points = 0

            # Store result
            result_dict = {
                'spacing': spacing,
                'startidx': startidx,
                'freq_shift': freq_shift,
                'nw': nw,
                'f_center': (f2 + f3) / 2,
                'cycles': cycles,
                'mean_amp': mean_amp,
                'std_amp': std_amp,
                'beat_score': beat_score,
                'data_points': data_points,
                'signal': bp2_display,
            }
            results.append(result_dict)

            print(f"spacing={spacing}, startidx={startidx}, freq_shift={freq_shift:+.1f}: "
                  f"cycles={cycles}, amp={mean_amp:.1f}+/-{std_amp:.1f}, beat={beat_score:.2f}, "
                  f"points={data_points}")

print()
print("=" * 80)
print("Top 6 Results Ranked by Beat Signature")
print("=" * 80)
print()

# Sort by beat score (higher is better for modulation/beats)
results_sorted = sorted(results, key=lambda x: x['beat_score'], reverse=True)

# Display top 6
for rank, res in enumerate(results_sorted[:6], 1):
    print(f"{rank}. spacing={res['spacing']}, startidx={res['startidx']}, "
          f"freq_shift={res['freq_shift']:+.1f}")
    print(f"   f_center={res['f_center']:.2f} rad/yr, cycles={res['cycles']}, "
          f"beat_score={res['beat_score']:.3f}, mean_amp={res['mean_amp']:.1f}")
    print()

# ============================================================================
# GENERATE COMPARISON PLOTS FOR TOP 6
# ============================================================================

print("Generating comparison plots for top 6 candidates...")
print()

output_dir = Path(script_dir) / 'bp2_refinement'
output_dir.mkdir(exist_ok=True)

for rank, res in enumerate(results_sorted[:6], 1):

    spacing = res['spacing']
    startidx = res['startidx']
    freq_shift = res['freq_shift']

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Full window with all data
    ax = axes[0]
    ax.plot(disp_dates, res['signal'], 'b.-', linewidth=0.8, markersize=4)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_title(f"Rank {rank}: BP-2 Refinement - spacing={spacing}, startidx={startidx}, "
                 f"freq_shift={freq_shift:+.1f}, beat_score={res['beat_score']:.3f}",
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Amplitude', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Plot 2: Normalized (like layout plot)
    ax = axes[1]
    sig_norm = normalize_signal(res['signal'], method='zscore')
    sig_scaled = sig_norm * 30  # Same scaling as layout
    ax.plot(disp_dates, sig_scaled, 'k.-', linewidth=0.8, markersize=4)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_title('Normalized (z-score) x 30', fontsize=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Scaled Amplitude', fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save
    filename = f"bp2_rank{rank}_s{spacing}_idx{startidx}_shift{freq_shift:+.1f}.png"
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {filepath.name}")

print()
print("=" * 80)
print("Analysis Complete")
print("=" * 80)
print()

print("Summary:")
print(f"  Total combinations tested: {len(results)}")
print(f"  Output directory: {output_dir}")
print()

print("Recommendation:")
best = results_sorted[0]
print(f"  Best match: spacing={best['spacing']}, startidx={best['startidx']}, "
      f"freq_shift={best['freq_shift']:+.1f}")
print(f"  Center frequency: {best['f_center']:.2f} rad/yr")
print(f"  Beat score: {best['beat_score']:.3f}")
print()

print("Next step:")
print("  1. Visually compare rank 1-6 plots with reference")
print("  2. Select best match based on visual alignment")
print("  3. Update reproduce_page152_layout.py with best BP-2 parameters")
print("  4. Proceed to BP-3, BP-4, BP-5, BP-6 refinement")
print()
