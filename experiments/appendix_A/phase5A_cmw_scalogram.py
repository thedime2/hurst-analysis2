# -*- coding: utf-8 -*-
"""
Phase 5A: CMW Scalogram -- Dense Time-Frequency Representation

Computes a CMW scalogram across 150 log-spaced center frequencies
(0.5-80 rad/yr) using constant-Q design (Q=5). Produces:

  Figure 1: Full scalogram heatmap with nominal model lines + filter bands
  Figure 2: Display window (1935-1954) zoomed view
  Figure 3: Marginal spectrum vs Lanczos spectrum (validation)

Reference: J.M. Hurst, Appendix A; Cohen (2019) NeuroImage 199:81-86
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from src.time_frequency import compute_scalogram
from src.spectral import lanczos_spectrum

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

# Scalogram parameters
FREQ_LO = 0.5     # rad/yr (period ~12.6 yr)
FREQ_HI = 80.0    # rad/yr (period ~4.1 wk) -- up to near Nyquist
N_SCALES = 200
Q_FACTOR = 5.0

# Page 152 filter bands for overlay [f_lo, f_hi, label]
FILTER_BANDS = [
    (0.0,   1.25,  'LP-1'),
    (0.85,  2.45,  'BP-2'),
    (3.20,  6.70,  'BP-3'),
    (7.25,  9.85,  'BP-4'),
    (13.65, 19.65, 'BP-5'),
    (28.45, 36.25, 'BP-6'),
]

print("=" * 70)
print("Phase 5A: CMW Scalogram")
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

print(f"  {len(close_prices)} samples")
print(f"  Date range: {dates_dt[0].strftime('%Y-%m-%d')} to "
      f"{dates_dt[-1].strftime('%Y-%m-%d')}")

# Display window indices
mask = (dates_dt >= pd.to_datetime(DISPLAY_START)) & \
       (dates_dt <= pd.to_datetime(DISPLAY_END))
disp_idx = np.where(mask)[0]
s_idx, e_idx = disp_idx[0], disp_idx[-1] + 1
print(f"  Display window: {e_idx - s_idx} samples")

# Load nominal model
nominal = pd.read_csv(nominal_path)
nominal_freqs = nominal['frequency'].values
print(f"  Nominal model: {len(nominal_freqs)} lines "
      f"({nominal_freqs.min():.2f} to {nominal_freqs.max():.2f} rad/yr)")
print()

# ============================================================================
# COMPUTE SCALOGRAM
# ============================================================================

print(f"Computing scalogram: {N_SCALES} scales, "
      f"{FREQ_LO:.1f}-{FREQ_HI:.1f} rad/yr, Q={Q_FACTOR}...")

scalo = compute_scalogram(
    close_prices,
    freq_range=(FREQ_LO, FREQ_HI),
    n_scales=N_SCALES,
    fs=FS,
    fwhm_mode='constant_q',
    q_factor=Q_FACTOR,
    freq_spacing='log',
    analytic=True,
)

matrix = scalo['matrix']
freqs = scalo['frequencies']
print(f"  Matrix shape: {matrix.shape}")
print(f"  Frequency range: {freqs[0]:.3f} to {freqs[-1]:.3f} rad/yr")
print(f"  FWHM range: {scalo['fwhm_per_scale'][0]:.3f} to "
      f"{scalo['fwhm_per_scale'][-1]:.3f} rad/yr")
print(f"  Period range: {scalo['periods_weeks'][-1]:.1f} to "
      f"{scalo['periods_weeks'][0]:.1f} weeks")
print()

# ============================================================================
# FIGURE 1: Full Scalogram Heatmap
# ============================================================================

print("Generating Figure 1: Full scalogram heatmap...")

fig1, ax1 = plt.subplots(figsize=(18, 10))

# Convert dates to matplotlib date numbers for pcolormesh
date_nums = mdates.date2num(dates_dt)

# Log of envelope for better dynamic range visualization
log_matrix = np.log10(matrix + 1e-6)

pcm = ax1.pcolormesh(
    date_nums, freqs, log_matrix,
    cmap='inferno', shading='auto',
    rasterized=True,
)

ax1.set_yscale('log')
ax1.set_ylabel('Frequency (rad/yr)', fontsize=12)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_title(
    f'CMW Scalogram -- DJIA {DATE_START[:4]}-{DATE_END[:4]}  '
    f'(Q={Q_FACTOR}, {N_SCALES} scales)',
    fontsize=14, fontweight='bold'
)

# Colorbar
cbar = plt.colorbar(pcm, ax=ax1, label='log10(Envelope)', pad=0.02)

# Overlay nominal model lines
for nf in nominal_freqs:
    ax1.axhline(nf, color='cyan', linewidth=0.6, alpha=0.5, linestyle='--')

# Overlay page 152 filter bands (semi-transparent rectangles on right edge)
for f_lo, f_hi, label in FILTER_BANDS:
    if f_lo < FREQ_LO:
        f_lo = FREQ_LO
    if f_hi > FREQ_HI:
        continue
    # Draw a narrow vertical strip on the right
    x_right = date_nums[-1]
    x_width = (date_nums[-1] - date_nums[0]) * 0.015
    rect = Rectangle(
        (x_right, f_lo), x_width, f_hi - f_lo,
        linewidth=0.8, edgecolor='lime', facecolor='lime', alpha=0.3,
        clip_on=True, zorder=5,
    )
    ax1.add_patch(rect)
    ax1.text(x_right + x_width * 1.2, (f_lo + f_hi) / 2, label,
             fontsize=7, color='lime', va='center', ha='left',
             fontweight='bold', zorder=6)

# Format x-axis dates
ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax1.set_xlim(date_nums[0], date_nums[-1] + (date_nums[-1] - date_nums[0]) * 0.05)
plt.xticks(rotation=45)

# Set y-axis limits with some ticks
ax1.set_ylim(FREQ_LO, FREQ_HI)
yticks = [0.5, 1, 2, 4, 8, 16, 32, 64]
yticks = [y for y in yticks if FREQ_LO <= y <= FREQ_HI]
ax1.set_yticks(yticks)
ax1.set_yticklabels([f'{y:.1f}' if y < 1 else f'{y:.0f}' for y in yticks])
ax1.grid(True, alpha=0.15, which='both')

plt.tight_layout()
out1 = os.path.join(script_dir, 'phase5A_scalogram_full.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight')
print(f"  Saved: {out1}")

# ============================================================================
# FIGURE 2: Display Window Detail
# ============================================================================

print("Generating Figure 2: Display window detail...")

fig2, ax2 = plt.subplots(figsize=(18, 10))

# Slice to display window
log_disp = log_matrix[:, s_idx:e_idx]
date_nums_disp = date_nums[s_idx:e_idx]

pcm2 = ax2.pcolormesh(
    date_nums_disp, freqs, log_disp,
    cmap='inferno', shading='auto',
    rasterized=True,
)

ax2.set_yscale('log')
ax2.set_ylabel('Frequency (rad/yr)', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_title(
    f'CMW Scalogram -- DJIA {DISPLAY_START[:4]}-{DISPLAY_END[:4]} '
    f'(Display Window)',
    fontsize=14, fontweight='bold'
)

cbar2 = plt.colorbar(pcm2, ax=ax2, label='log10(Envelope)', pad=0.02)

# Overlay nominal model lines
for i, nf in enumerate(nominal_freqs):
    ax2.axhline(nf, color='cyan', linewidth=0.6, alpha=0.5, linestyle='--')
    # Label every 3rd line
    if i % 3 == 0:
        ax2.text(date_nums_disp[0], nf * 1.02, f'{nf:.1f}',
                 fontsize=6, color='cyan', alpha=0.7)

# Filter bands
for f_lo, f_hi, label in FILTER_BANDS:
    if f_lo < FREQ_LO:
        f_lo = FREQ_LO
    if f_hi > FREQ_HI:
        continue
    ax2.axhspan(f_lo, f_hi, alpha=0.08, color='lime', zorder=0)
    ax2.text(date_nums_disp[-1], (f_lo * f_hi) ** 0.5, f' {label}',
             fontsize=8, color='lime', va='center', fontweight='bold')

ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.set_ylim(FREQ_LO, FREQ_HI)
yticks = [0.5, 1, 2, 4, 8, 16, 32, 64]
yticks = [y for y in yticks if FREQ_LO <= y <= FREQ_HI]
ax2.set_yticks(yticks)
ax2.set_yticklabels([f'{y:.1f}' if y < 1 else f'{y:.0f}' for y in yticks])
ax2.grid(True, alpha=0.15, which='both')
plt.xticks(rotation=45)

plt.tight_layout()
out2 = os.path.join(script_dir, 'phase5A_scalogram_display.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")

# ============================================================================
# FIGURE 3: Marginal Spectrum vs Lanczos
# ============================================================================

print("Generating Figure 3: Marginal spectrum vs Lanczos...")

# Compute Lanczos spectrum for comparison
w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(
    close_prices, 1, FS
)
omega_yr = w * FS  # Convert to rad/yr

# Marginal spectrum: time-average of each scalogram row
marginal = np.mean(matrix, axis=1)

fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(14, 10))

# Panel A: Both spectra overlaid (log-log)
ax3a.semilogy(omega_yr, amp, 'b-', linewidth=0.8, alpha=0.7,
              label='Lanczos spectrum')
ax3a.semilogy(freqs, marginal, 'r-', linewidth=1.5,
              label='CMW marginal (time-avg envelope)')
ax3a.set_xlim(0, 45)
ax3a.set_xlabel('Frequency (rad/yr)')
ax3a.set_ylabel('Amplitude (log scale)')
ax3a.set_title('Marginal CMW Spectrum vs Lanczos Spectrum', fontweight='bold')
ax3a.legend(loc='upper right')
ax3a.grid(True, alpha=0.3)

# Mark nominal lines
for nf in nominal_freqs:
    ax3a.axvline(nf, color='gray', linewidth=0.3, alpha=0.4)

# Panel B: Both on log-log axes for envelope comparison
ax3b.loglog(omega_yr[omega_yr > 0], amp[omega_yr > 0], 'b-',
            linewidth=0.8, alpha=0.7, label='Lanczos spectrum')
ax3b.loglog(freqs, marginal, 'r-', linewidth=1.5,
            label='CMW marginal')

# Fit power law to marginal: a(w) = k/w
mask_fit = (freqs >= 2.0) & (freqs <= 30.0)
log_f = np.log(freqs[mask_fit])
log_m = np.log(marginal[mask_fit])
coeffs = np.polyfit(log_f, log_m, 1)
slope = coeffs[0]
intercept = coeffs[1]
fit_line = np.exp(intercept) * freqs ** slope
ax3b.loglog(freqs, fit_line, 'g--', linewidth=1.5,
            label=f'Power law fit: slope={slope:.2f}')

ax3b.set_xlim(0.4, 90)
ax3b.set_xlabel('Frequency (rad/yr)')
ax3b.set_ylabel('Amplitude (log scale)')
ax3b.set_title('Log-Log Comparison (Envelope Shape)', fontweight='bold')
ax3b.legend(loc='upper right')
ax3b.grid(True, alpha=0.3, which='both')

plt.tight_layout()
out3 = os.path.join(script_dir, 'phase5A_marginal_spectrum.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
print(f"  Saved: {out3}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print()
print("-" * 50)
print("Scalogram Summary")
print("-" * 50)
print(f"  Scales: {N_SCALES}")
print(f"  Frequency range: {FREQ_LO:.1f} - {FREQ_HI:.1f} rad/yr")
print(f"  Period range: {scalo['periods_weeks'][-1]:.1f} - "
      f"{scalo['periods_weeks'][0]:.1f} weeks")
print(f"  FWHM mode: constant_q (Q={Q_FACTOR})")
print(f"  Power law slope: {slope:.3f} "
      f"(expected ~-1.0 for a(w)=k/w)")
print(f"  Matrix dynamic range: {matrix.max():.1f} / {matrix[matrix>0].min():.4f} "
      f"= {matrix.max() / matrix[matrix>0].min():.0f}x")

# Find dominant frequency at a few time points
print()
print("  Dominant frequency at key dates:")
key_dates = ['1929-10-25', '1937-03-10', '1942-04-28', '1954-01-01']
for kd in key_dates:
    kd_dt = pd.to_datetime(kd)
    idx = np.argmin(np.abs(date_nums - mdates.date2num(kd_dt)))
    if 0 <= idx < matrix.shape[1]:
        col = matrix[:, idx]
        peak_idx = np.argmax(col)
        print(f"    {kd}: peak at {freqs[peak_idx]:.2f} rad/yr "
              f"(T={scalo['periods_weeks'][peak_idx]:.0f} wk), "
              f"amp={col[peak_idx]:.1f}")

plt.show()
print()
print("Done.")
