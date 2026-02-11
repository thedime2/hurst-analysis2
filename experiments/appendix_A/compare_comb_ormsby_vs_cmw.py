# -*- coding: utf-8 -*-
"""
Appendix A: Ormsby Comb Bank vs CMW Comparison

Applies the 23-filter comb bank from Hurst's Appendix A using both Ormsby
filters (existing) and FWHM-matched Complex Morlet Wavelets. Produces:
  1. Selected individual filter comparisons (FC-1, FC-12, FC-23)
  2. Time-frequency heatmap comparison (Ormsby vs CMW envelopes)

Comb bank: 23 filters, w1_start=7.2, step=0.2, passband=0.2, skirt=0.3 rad/yr
CMW matched: FWHM=0.50 rad/yr for all filters

Reference: J.M. Hurst, Appendix A, Figures AI-2 through AI-4
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.fft import fft

from src.filters import (
    design_hurst_comb_bank,
    create_filter_kernels,
    apply_filter_bank,
)
from src.time_frequency import ormsby_spec_to_cmw_params, apply_cmw_bank, cmw_freq_domain

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
NW = 1999  # Comb filter length (consistent with phase2_figure_AI2.py)

print("=" * 70)
print("Appendix A: Ormsby Comb Bank vs CMW Comparison")
print("=" * 70)
print()

# ============================================================================
# DESIGN COMB BANK
# ============================================================================

print("Designing comb filter bank...")
specs = design_hurst_comb_bank(
    n_filters=23, w1_start=7.2, w_step=0.2,
    passband_width=0.2, skirt_width=0.3, nw=NW, fs=FS
)

# Convert to CMW parameters
cmw_params = [ormsby_spec_to_cmw_params(s) for s in specs]

print(f"\nCMW matched parameters (all filters):")
print(f"  FWHM = {cmw_params[0]['fwhm']:.2f} rad/yr")
print(f"  sigma_f = {cmw_params[0]['sigma_f']:.4f} rad/yr")
print(f"  f0 range: {cmw_params[0]['f0']:.2f} to {cmw_params[-1]['f0']:.2f} rad/yr")
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

mask = (dates_dt >= pd.to_datetime(DISPLAY_START)) & \
       (dates_dt <= pd.to_datetime(DISPLAY_END))
disp_idx = np.where(mask)[0]
s_idx, e_idx = disp_idx[0], disp_idx[-1] + 1
disp_dates = dates_dt[s_idx:e_idx]
print(f"  {len(close_prices)} samples, display: {e_idx - s_idx} samples")
print()

# ============================================================================
# APPLY ORMSBY COMB BANK
# ============================================================================

print("Applying Ormsby comb bank (23 filters, modulate, analytic)...")
filters_orm = create_filter_kernels(specs, fs=FS, filter_type='modulate',
                                     analytic=True)
results_orm = apply_filter_bank(close_prices, filters_orm, fs=FS, mode='reflect')

# ============================================================================
# APPLY CMW BANK
# ============================================================================

print("Applying matched CMW bank (23 filters)...")
results_cmw = apply_cmw_bank(close_prices, cmw_params, fs=FS, analytic=True)

# ============================================================================
# PLOT 1: Selected individual filter comparisons
# ============================================================================

print("Generating individual filter comparisons...")

# Show filters 1, 12, 23 (first, middle, last)
selected = [0, 11, 22]
fig, axes = plt.subplots(len(selected), 1, figsize=(16, 12), sharex=True)

for ax_i, filt_idx in enumerate(selected):
    ax = axes[ax_i]
    spec = specs[filt_idx]
    orm_out = results_orm['filter_outputs'][filt_idx]
    cmw_out = results_cmw['filter_outputs'][filt_idx]

    # Ormsby signal and envelope
    sig_orm = orm_out['signal'].real[s_idx:e_idx]
    env_orm = orm_out['envelope'][s_idx:e_idx]

    # CMW signal and envelope
    sig_cmw = cmw_out['signal'].real[s_idx:e_idx]
    env_cmw = cmw_out['envelope'][s_idx:e_idx]

    ax.plot(disp_dates, sig_orm, 'b-', linewidth=0.5, alpha=0.6,
            label='Ormsby signal')
    ax.plot(disp_dates, sig_cmw, 'r-', linewidth=0.5, alpha=0.6,
            label='CMW signal')
    ax.plot(disp_dates, env_orm, 'b-', linewidth=1.2, alpha=0.5,
            label='Ormsby env')
    ax.plot(disp_dates, -env_orm, 'b-', linewidth=1.2, alpha=0.5)
    ax.plot(disp_dates, env_cmw, 'r--', linewidth=1.2, alpha=0.7,
            label='CMW env')
    ax.plot(disp_dates, -env_cmw, 'r--', linewidth=1.2, alpha=0.7)
    ax.axhline(0, color='gray', linewidth=0.3)

    period = TWOPI / spec['f_center']
    ax.set_ylabel(f"FC-{filt_idx+1}", fontsize=10, rotation=0, labelpad=40)
    ax.set_title(f"FC-{filt_idx+1}: fc={spec['f_center']:.2f} rad/yr "
                 f"(T={period:.1f} yr)", fontsize=10)
    ax.grid(True, alpha=0.2)
    if ax_i == 0:
        ax.legend(loc='upper right', fontsize=9)

axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].set_xlabel('Date')
plt.xticks(rotation=45)

fig.suptitle("Comb Bank: Ormsby (blue) vs CMW (red) — Selected Filters",
             fontsize=13, fontweight='bold')
plt.tight_layout()

out1 = os.path.join(script_dir, 'compare_comb_ormsby_vs_cmw_selected.png')
fig.savefig(out1, dpi=150, bbox_inches='tight')
print(f"  Saved: {out1}")

# ============================================================================
# PLOT 2: Time-frequency heatmap comparison
# ============================================================================

print("Generating time-frequency heatmap comparison...")

n_filters = len(specs)
n_disp = e_idx - s_idx
center_freqs = np.array([s['f_center'] for s in specs])

# Build envelope matrices
tf_orm = np.zeros((n_filters, n_disp))
tf_cmw = np.zeros((n_filters, n_disp))

for i in range(n_filters):
    tf_orm[i, :] = results_orm['filter_outputs'][i]['envelope'][s_idx:e_idx]
    tf_cmw[i, :] = results_cmw['filter_outputs'][i]['envelope'][s_idx:e_idx]

fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14))

# Common colorscale
vmax = max(tf_orm.max(), tf_cmw.max())

for ax, tf_data, title in [(ax1, tf_orm, 'Ormsby Comb Bank'),
                             (ax2, tf_cmw, 'CMW (FWHM-Matched)'),
                             (ax3, tf_orm - tf_cmw, 'Difference (Ormsby - CMW)')]:
    if 'Difference' in title:
        vmin_d = -vmax * 0.3
        vmax_d = vmax * 0.3
        im = ax.imshow(tf_data, aspect='auto', origin='upper',
                        cmap='RdBu_r', extent=[0, n_disp - 1,
                        center_freqs[-1], center_freqs[0]],
                        vmin=vmin_d, vmax=vmax_d, interpolation='bilinear')
    else:
        im = ax.imshow(tf_data, aspect='auto', origin='upper',
                        cmap='hot', extent=[0, n_disp - 1,
                        center_freqs[-1], center_freqs[0]],
                        vmin=0, vmax=vmax, interpolation='bilinear')

    ax.set_ylabel('Center Freq (rad/yr)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Amplitude')

    # Date ticks
    n_ticks = 10
    tick_idx = np.linspace(0, n_disp - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(disp_dates[i])[:4] for i in tick_idx], rotation=45)

ax3.set_xlabel('Date')

fig2.suptitle("Time-Frequency: Ormsby vs CMW Comb Bank (23 Filters)",
              fontsize=13, fontweight='bold')
plt.tight_layout()

out2 = os.path.join(script_dir, 'compare_comb_ormsby_vs_cmw_heatmap.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")

# ============================================================================
# PLOT 3: Frequency response overlay for all 23 filters
# ============================================================================

print("Generating frequency response comparison...")

nfft = 8192
freqs_norm = np.arange(nfft) / nfft
pos_mask_fft = freqs_norm <= 0.5
freqs_rad_axis = freqs_norm[pos_mask_fft] * FS * TWOPI

fig3, ax = plt.subplots(figsize=(14, 6))

# Sum responses
sum_orm = np.zeros(np.sum(pos_mask_fft))
sum_cmw = np.zeros(np.sum(pos_mask_fft))

for i, (filt, cmw_p) in enumerate(zip(filters_orm, cmw_params)):
    # Ormsby
    H_orm = fft(filt['kernel'], n=nfft)
    H_orm_mag = np.abs(H_orm[pos_mask_fft]) * 0.5
    sum_orm += H_orm_mag ** 2

    # CMW
    cmw_r = cmw_freq_domain(cmw_p['f0'], cmw_p['fwhm'], FS, nfft, analytic=True)
    H_cmw_full = cmw_r['H']
    freqs_cmw_full = cmw_r['freqs_rad']
    cmw_pos = freqs_cmw_full >= 0
    cmw_idx = np.where(cmw_pos)[0]
    sort_ord = np.argsort(freqs_cmw_full[cmw_idx])
    H_cmw_sorted = H_cmw_full[cmw_idx][sort_ord] * 0.5

    color_orm = plt.cm.Blues(0.3 + 0.7 * i / 22)
    color_cmw = plt.cm.Reds(0.3 + 0.7 * i / 22)

    if i == 0:
        ax.plot(freqs_rad_axis, H_orm_mag, '-', color=color_orm,
                linewidth=0.8, alpha=0.6, label='Ormsby')
        freqs_sorted = freqs_cmw_full[cmw_idx][sort_ord]
        ax.plot(freqs_sorted, H_cmw_sorted, '--', color=color_cmw,
                linewidth=0.8, alpha=0.6, label='CMW')
    else:
        ax.plot(freqs_rad_axis, H_orm_mag, '-', color=color_orm,
                linewidth=0.8, alpha=0.6)
        freqs_sorted = freqs_cmw_full[cmw_idx][sort_ord]
        ax.plot(freqs_sorted, H_cmw_sorted, '--', color=color_cmw,
                linewidth=0.8, alpha=0.6)

    # Accumulate CMW sum at same freq points as Ormsby
    cmw_at_orm_freqs = cmw_freq_domain(cmw_p['f0'], cmw_p['fwhm'], FS, nfft,
                                        analytic=True)
    H_cmw_aligned = cmw_at_orm_freqs['H'][pos_mask_fft] * 0.5
    sum_cmw += H_cmw_aligned ** 2

# Plot sum responses
sum_orm = np.sqrt(sum_orm)
sum_cmw = np.sqrt(sum_cmw)
ax.plot(freqs_rad_axis, sum_orm, 'b-', linewidth=2.5, label='Ormsby sum')
ax.plot(freqs_rad_axis, sum_cmw, 'r-', linewidth=2.5, label='CMW sum')
ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)

ax.set_xlim(6.5, 13)
ax.set_ylim(0, 1.5)
ax.set_xlabel('Frequency (rad/year)')
ax.set_ylabel('Magnitude')
ax.set_title('Comb Bank Frequency Response: Ormsby (blue) vs CMW (red)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()

out3 = os.path.join(script_dir, 'compare_comb_ormsby_vs_cmw_freqresp.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
print(f"  Saved: {out3}")

plt.show()
print("\nDone.")
