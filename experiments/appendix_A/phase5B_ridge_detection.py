# -*- coding: utf-8 -*-
"""
Phase 5B: Ridge Detection and Tracking

Extracts continuous frequency ridges from the CMW scalogram computed
in Phase 5A. Ridges are continuous curves tracing where spectral energy
concentrates in the time-frequency plane. Produces:

  Figure 1: Scalogram with ridge curves overlaid
  Figure 2: Ridge frequency vs time (CMW version of Figure AI-6)
  Figure 3: Ridge statistics (duration + frequency distributions)
  Figure 4: Ridge vs Phase 2 comb filter comparison (7.6-12 rad/yr)

Reference: J.M. Hurst, Appendix A, Figure AI-6
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.time_frequency import (
    compute_scalogram,
    detect_ridges,
    match_ridges_to_nominal,
    compute_ridge_statistics,
)
from src.filters import (
    design_hurst_comb_bank,
    create_filter_kernels,
    apply_filter_bank,
)
from src.spectral import measure_freq_at_peaks, measure_freq_at_troughs

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
nominal_path = os.path.join(script_dir, '../../data/processed/nominal_model.csv')

DATE_START = '1921-04-29'
DATE_END = '1965-05-21'
DISPLAY_START = '1935-01-01'
DISPLAY_END = '1954-02-01'

FS = 52
TWOPI = 2 * np.pi

# Scalogram parameters (same as 5A)
FREQ_LO = 0.5
FREQ_HI = 80.0
N_SCALES = 200
Q_FACTOR = 5.0

# Ridge detection parameters
MIN_PROMINENCE = 0.08
MIN_DURATION_SAMPLES = 52  # 1 year minimum
MAX_FREQ_JUMP = None       # auto from frequency spacing

print("=" * 70)
print("Phase 5B: Ridge Detection and Tracking")
print("=" * 70)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading weekly DJIA data...")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between(DATE_START, DATE_END)].copy()
close_prices = df_hurst.Close.values
dates_dt = pd.to_datetime(df_hurst.Date.values)
date_nums = mdates.date2num(dates_dt)
n_samples = len(close_prices)

print(f"  {n_samples} samples")

# Display window
mask = (dates_dt >= pd.to_datetime(DISPLAY_START)) & \
       (dates_dt <= pd.to_datetime(DISPLAY_END))
disp_idx = np.where(mask)[0]
s_idx, e_idx = disp_idx[0], disp_idx[-1] + 1

# Load nominal model
nominal = pd.read_csv(nominal_path)
nominal_freqs = nominal['frequency'].values
print(f"  Nominal model: {len(nominal_freqs)} lines")
print()

# ============================================================================
# COMPUTE SCALOGRAM
# ============================================================================

print(f"Computing scalogram ({N_SCALES} scales)...")
scalo = compute_scalogram(
    close_prices,
    freq_range=(FREQ_LO, FREQ_HI),
    n_scales=N_SCALES,
    fs=FS,
    fwhm_mode='constant_q',
    q_factor=Q_FACTOR,
    freq_spacing='log',
)

matrix = scalo['matrix']
freqs = scalo['frequencies']
print(f"  Matrix shape: {matrix.shape}")

# ============================================================================
# DETECT RIDGES
# ============================================================================

print(f"Detecting ridges (prominence={MIN_PROMINENCE}, "
      f"min_duration={MIN_DURATION_SAMPLES} samples)...")

ridges = detect_ridges(
    matrix, freqs,
    min_prominence=MIN_PROMINENCE,
    max_freq_jump=MAX_FREQ_JUMP,
    min_duration_samples=MIN_DURATION_SAMPLES,
)

stats = compute_ridge_statistics(ridges)
print(f"  Found {stats['n_ridges']} ridges")
print(f"  Mean duration: {stats['mean_duration_years']:.1f} years")
print(f"  Median duration: {stats['median_duration_years']:.1f} years")
print(f"  Coverage: {stats['total_coverage']:.1%}")
print(f"  Frequency range: {stats['freq_range'][0]:.2f} to "
      f"{stats['freq_range'][1]:.2f} rad/yr")
print(f"  Mean drift rate: {stats['mean_drift_rate']:.4f} rad/yr/yr")
print()

# ============================================================================
# MATCH TO NOMINAL MODEL
# ============================================================================

matches, unmatched_r, unmatched_n = match_ridges_to_nominal(
    ridges, nominal_freqs, tolerance=0.5
)

print(f"Nominal model matching (tolerance=0.5 rad/yr):")
print(f"  Matched: {len(matches)} ridges to {len(set(m['nominal_line'] for m in matches))} nominal lines")
print(f"  Unmatched ridges: {len(unmatched_r)}")
print(f"  Unmatched nominal lines: {len(unmatched_n)}")
print()

# ============================================================================
# FIGURE 1: Scalogram + Ridges Overlaid
# ============================================================================

print("Generating Figure 1: Scalogram with ridges...")

fig1, ax1 = plt.subplots(figsize=(18, 10))

log_matrix = np.log10(matrix + 1e-6)
pcm = ax1.pcolormesh(
    date_nums, freqs, log_matrix,
    cmap='inferno', shading='auto', rasterized=True,
)
ax1.set_yscale('log')

# Overlay ridges
cmap_ridges = plt.cm.Set1
n_colors = min(len(ridges), 20)
for i, ridge in enumerate(ridges):
    color = cmap_ridges(i % 10 / 10.0)
    ridge_dates = date_nums[ridge['time_indices']]
    ax1.plot(ridge_dates, ridge['frequencies'],
             '-', color=color, linewidth=1.0, alpha=0.8)

# Nominal model lines
for nf in nominal_freqs:
    ax1.axhline(nf, color='cyan', linewidth=0.4, alpha=0.4, linestyle='--')

ax1.set_ylabel('Frequency (rad/yr)', fontsize=12)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_title(
    f'CMW Scalogram with {stats["n_ridges"]} Detected Ridges',
    fontsize=14, fontweight='bold'
)
plt.colorbar(pcm, ax=ax1, label='log10(Envelope)', pad=0.02)

ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_ylim(FREQ_LO, FREQ_HI)
yticks = [0.5, 1, 2, 4, 8, 16, 32, 64]
yticks = [y for y in yticks if FREQ_LO <= y <= FREQ_HI]
ax1.set_yticks(yticks)
ax1.set_yticklabels([f'{y:.1f}' if y < 1 else f'{y:.0f}' for y in yticks])
ax1.grid(True, alpha=0.15, which='both')
plt.xticks(rotation=45)

plt.tight_layout()
out1 = os.path.join(script_dir, 'phase5B_ridges_on_scalogram.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
print(f"  Saved: {out1}")

# ============================================================================
# FIGURE 2: Ridge Frequency vs Time (AI-6 style)
# ============================================================================

print("Generating Figure 2: Ridge frequency vs time...")

fig2, ax2 = plt.subplots(figsize=(18, 8))

# Plot each ridge as a colored trace
for i, ridge in enumerate(ridges):
    color = cmap_ridges(i % 10 / 10.0)
    t_years = ridge['time_indices'] / FS
    ax2.plot(t_years, ridge['frequencies'],
             '.', color=color, markersize=1.5, alpha=0.6)

# Overlay nominal model horizontal lines
for j, nf in enumerate(nominal_freqs):
    ax2.axhline(nf, color='gray', linewidth=0.5, alpha=0.5, linestyle='-')
    ax2.text(-0.5, nf, f'{nf:.1f}', fontsize=6, color='gray',
             va='center', ha='right')

ax2.set_xlabel('Time (years from start)', fontsize=12)
ax2.set_ylabel('Frequency (rad/yr)', fontsize=12)
ax2.set_title(
    f'Ridge Frequency vs Time -- CMW-Based (cf. Figure AI-6)\n'
    f'{stats["n_ridges"]} ridges, mean duration {stats["mean_duration_years"]:.1f} yr',
    fontsize=13, fontweight='bold'
)
ax2.set_ylim(0, max(nominal_freqs.max() + 2, stats['freq_range'][1] + 2))
ax2.set_xlim(-1, n_samples / FS + 1)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
out2 = os.path.join(script_dir, 'phase5B_ridge_freq_vs_time.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")

# ============================================================================
# FIGURE 3: Ridge Statistics
# ============================================================================

print("Generating Figure 3: Ridge statistics...")

fig3, ((ax3a, ax3b), (ax3c, ax3d)) = plt.subplots(2, 2, figsize=(14, 10))

# 3A: Duration histogram
durations_yr = [r['duration_years'] for r in ridges]
ax3a.hist(durations_yr, bins=20, color='steelblue', edgecolor='white')
ax3a.axvline(np.mean(durations_yr), color='red', linewidth=1.5,
             label=f'Mean={np.mean(durations_yr):.1f} yr')
ax3a.axvline(np.median(durations_yr), color='orange', linewidth=1.5,
             linestyle='--', label=f'Median={np.median(durations_yr):.1f} yr')
ax3a.set_xlabel('Duration (years)')
ax3a.set_ylabel('Count')
ax3a.set_title('Ridge Duration Distribution')
ax3a.legend()

# 3B: Mean frequency distribution
mean_freqs = [r['mean_freq'] for r in ridges]
ax3b.hist(mean_freqs, bins=30, color='coral', edgecolor='white')
for nf in nominal_freqs:
    ax3b.axvline(nf, color='gray', linewidth=0.5, alpha=0.5)
ax3b.set_xlabel('Mean Frequency (rad/yr)')
ax3b.set_ylabel('Count')
ax3b.set_title('Ridge Mean Frequency Distribution')

# 3C: Drift rate histogram
drift_rates = stats['drift_rates']
ax3c.hist(drift_rates, bins=25, color='seagreen', edgecolor='white')
ax3c.axvline(0, color='red', linewidth=1.5, linestyle='--', label='Zero drift')
ax3c.axvline(np.mean(drift_rates), color='orange', linewidth=1.5,
             label=f'Mean={np.mean(drift_rates):.4f}')
ax3c.set_xlabel('Drift Rate (rad/yr per year)')
ax3c.set_ylabel('Count')
ax3c.set_title('Ridge Drift Rate Distribution')
ax3c.legend(fontsize=9)

# 3D: Frequency stability (std_freq vs mean_freq)
std_freqs = [r['std_freq'] for r in ridges]
ax3d.scatter(mean_freqs, std_freqs, c=durations_yr, cmap='viridis',
             s=30, alpha=0.7)
ax3d.set_xlabel('Mean Frequency (rad/yr)')
ax3d.set_ylabel('Frequency Std Dev (rad/yr)')
ax3d.set_title('Frequency Stability (color = duration)')
plt.colorbar(ax3d.collections[0], ax=ax3d, label='Duration (yr)')

fig3.suptitle('Ridge Statistics Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
out3 = os.path.join(script_dir, 'phase5B_ridge_statistics.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
print(f"  Saved: {out3}")

# ============================================================================
# FIGURE 4: Ridge vs Comb Filter Comparison (7.6-12 rad/yr)
# ============================================================================

print("Generating Figure 4: Ridge vs comb filter comparison...")
print("  Computing comb filter bank (23 filters)...")

# Recompute comb bank to get frequency traces
comb_specs = design_hurst_comb_bank(
    n_filters=23, w1_start=7.2, w_step=0.2,
    passband_width=0.2, skirt_width=0.3, nw=1999, fs=FS
)
comb_filters = create_filter_kernels(comb_specs, fs=FS,
                                      filter_type='modulate', analytic=True)
comb_results = apply_filter_bank(close_prices, comb_filters, fs=FS, mode='reflect')

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(18, 12), sharex=True)

# Panel A: Comb filter frequency traces (Phase 2/3 method)
print("  Measuring comb filter frequencies...")
for i, output in enumerate(comb_results['filter_outputs']):
    sig_real = output['signal'].real
    phase = output['phase']

    pk = measure_freq_at_peaks(sig_real, phase, fs=FS)
    tr = measure_freq_at_troughs(sig_real, phase, fs=FS)

    # Plot peaks in blue, troughs in red
    if pk['times'].size > 0:
        ax4a.plot(pk['times'], pk['freqs_phase'], 'b.', markersize=1.5, alpha=0.4)
    if tr['times'].size > 0:
        ax4a.plot(tr['times'], tr['freqs_phase'], 'r.', markersize=1.5, alpha=0.4)

# Nominal lines in overlap range
for nf in nominal_freqs:
    if 7.0 <= nf <= 12.5:
        ax4a.axhline(nf, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)

ax4a.set_ylim(7.0, 12.5)
ax4a.set_ylabel('Frequency (rad/yr)')
ax4a.set_title('Comb Filter Frequency Traces (Phase 2/3 Method)',
               fontweight='bold')
ax4a.grid(True, alpha=0.2)

# Panel B: CMW ridges in the same frequency range
ridges_in_range = [r for r in ridges
                   if 7.0 <= r['mean_freq'] <= 12.5]

for i, ridge in enumerate(ridges_in_range):
    color = cmap_ridges(i % 10 / 10.0)
    t_years = ridge['time_indices'] / FS
    ax4b.plot(t_years, ridge['frequencies'],
             '.', color=color, markersize=2, alpha=0.7)

for nf in nominal_freqs:
    if 7.0 <= nf <= 12.5:
        ax4b.axhline(nf, color='gray', linewidth=0.5, linestyle='-', alpha=0.5)

ax4b.set_ylim(7.0, 12.5)
ax4b.set_xlabel('Time (years from start)')
ax4b.set_ylabel('Frequency (rad/yr)')
ax4b.set_title(f'CMW Ridge Traces ({len(ridges_in_range)} ridges in 7-12.5 rad/yr)',
               fontweight='bold')
ax4b.grid(True, alpha=0.2)
ax4b.set_xlim(-1, n_samples / FS + 1)

fig4.suptitle('Frequency Tracking: Comb Filters vs CMW Ridges (7.0-12.5 rad/yr)',
              fontsize=14, fontweight='bold')
plt.tight_layout()
out4 = os.path.join(script_dir, 'phase5B_ridge_vs_comb.png')
fig4.savefig(out4, dpi=150, bbox_inches='tight')
print(f"  Saved: {out4}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================

print()
print("-" * 60)
print("Ridge Detection Summary")
print("-" * 60)
print(f"  Total ridges: {stats['n_ridges']}")
print(f"  Mean duration: {stats['mean_duration_years']:.1f} years")
print(f"  Median duration: {stats['median_duration_years']:.1f} years")
print(f"  Coverage: {stats['total_coverage']:.1%}")
print(f"  Drift rate: mean={stats['mean_drift_rate']:.4f}, "
      f"std={stats['std_drift_rate']:.4f} rad/yr/yr")
print()

print("  Nominal model matching:")
print(f"    Matched ridges: {len(matches)}/{stats['n_ridges']}")
print(f"    Matched nominal lines: "
      f"{len(set(m['nominal_line'] for m in matches))}/{len(nominal_freqs)}")
print()

print("  Ridges in comb bank range (7.6-12 rad/yr):")
ridges_comb = [r for r in ridges if 7.6 <= r['mean_freq'] <= 12.0]
print(f"    Count: {len(ridges_comb)}")
if ridges_comb:
    comb_drifts = [r['drift_rate'] for r in ridges_comb]
    print(f"    Mean drift: {np.mean(comb_drifts):.4f} rad/yr/yr")
    for r in ridges_comb:
        print(f"      Ridge {r['ridge_id']}: f={r['mean_freq']:.2f} +/- "
              f"{r['std_freq']:.3f} rad/yr, "
              f"duration={r['duration_years']:.1f} yr, "
              f"drift={r['drift_rate']:.4f}")

plt.show()
print()
print("Done.")
