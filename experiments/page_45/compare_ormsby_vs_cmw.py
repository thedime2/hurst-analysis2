# -*- coding: utf-8 -*-
"""
Page 45: Ormsby vs CMW Comparison

Applies the same bandpass filter specification using both Ormsby (modulated,
analytic) and a matched Complex Morlet Wavelet (FWHM-designed in frequency
domain). Produces side-by-side comparison of filtered signals, envelopes,
and frequency responses.

Filter specification (rad/year): w1=3.2, w2=3.55, w3=6.35, w4=6.70
CMW matched: f0=4.95, FWHM=3.15 rad/yr

Reference: J.M. Hurst, Profit Magic, Chapter II, Figures II-9 & II-10
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.filters import ormsby_filter, apply_ormsby_filter
from src.time_frequency import ormsby_spec_to_cmw_params, cmw_freq_domain, apply_cmw

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

# Ormsby bandpass specification (rad/year)
ORMSBY_SPEC = {
    'type': 'bp',
    'f1': 3.20, 'f2': 3.55, 'f3': 6.35, 'f4': 6.70,
    'f_center': (3.55 + 6.35) / 2,
    'bandwidth': 6.35 - 3.55,
    'nw': 359 * 5,
    'index': 0,
    'label': 'BP: 3.2-6.7 rad/yr'
}

# Derive matched CMW parameters
CMW_PARAMS = ormsby_spec_to_cmw_params(ORMSBY_SPEC)

print("=" * 70)
print("Page 45: Ormsby vs CMW Comparison")
print("=" * 70)
print(f"  Ormsby edges (rad/yr): [{ORMSBY_SPEC['f1']}, {ORMSBY_SPEC['f2']}, "
      f"{ORMSBY_SPEC['f3']}, {ORMSBY_SPEC['f4']}]")
print(f"  Ormsby center: {ORMSBY_SPEC['f_center']:.2f} rad/yr, "
      f"nw={ORMSBY_SPEC['nw']}")
print(f"  CMW f0: {CMW_PARAMS['f0']:.2f} rad/yr, "
      f"FWHM: {CMW_PARAMS['fwhm']:.2f} rad/yr, "
      f"sigma_f: {CMW_PARAMS['sigma_f']:.4f} rad/yr")
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
# APPLY ORMSBY FILTER
# ============================================================================

print("Applying Ormsby bandpass (modulate, analytic)...")
f_edges_cyc = np.array([ORMSBY_SPEC['f1'], ORMSBY_SPEC['f2'],
                         ORMSBY_SPEC['f3'], ORMSBY_SPEC['f4']]) / TWOPI
h_ormsby = ormsby_filter(nw=ORMSBY_SPEC['nw'], f_edges=f_edges_cyc, fs=FS,
                         filter_type='bp', method='modulate', analytic=True)
result_ormsby = apply_ormsby_filter(close_prices, h_ormsby, mode='reflect', fs=FS)
sig_orm = result_ormsby['signal'].real
env_orm = result_ormsby['envelope']

# ============================================================================
# APPLY CMW FILTER
# ============================================================================

print("Applying matched CMW (freq-domain Gaussian)...")
result_cmw = apply_cmw(close_prices, CMW_PARAMS['f0'], CMW_PARAMS['fwhm'],
                        fs=FS, analytic=True)
sig_cmw = result_cmw['signal'].real
env_cmw = result_cmw['envelope']

print(f"  Ormsby: sig [{sig_orm[s_idx:e_idx].min():.2f}, "
      f"{sig_orm[s_idx:e_idx].max():.2f}], "
      f"env [{env_orm[s_idx:e_idx].min():.2f}, "
      f"{env_orm[s_idx:e_idx].max():.2f}]")
print(f"  CMW:    sig [{sig_cmw[s_idx:e_idx].min():.2f}, "
      f"{sig_cmw[s_idx:e_idx].max():.2f}], "
      f"env [{env_cmw[s_idx:e_idx].min():.2f}, "
      f"{env_cmw[s_idx:e_idx].max():.2f}]")
print()

# ============================================================================
# PLOT 1: Time-domain comparison (3 rows)
# ============================================================================

print("Generating time-domain comparison...")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

# Row 1: Ormsby
ax1.plot(disp_dates, sig_orm[s_idx:e_idx], 'b-', linewidth=0.7,
         label='Ormsby signal')
ax1.plot(disp_dates, env_orm[s_idx:e_idx], 'b-', linewidth=1.5, alpha=0.5)
ax1.plot(disp_dates, -env_orm[s_idx:e_idx], 'b-', linewidth=1.5, alpha=0.5)
ax1.axhline(0, color='gray', linewidth=0.4)
ax1.set_ylabel('Amplitude')
ax1.set_title(f"Ormsby Bandpass (modulate) — fc={ORMSBY_SPEC['f_center']:.2f} rad/yr")
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# Row 2: CMW
ax2.plot(disp_dates, sig_cmw[s_idx:e_idx], 'r-', linewidth=0.7,
         label='CMW signal')
ax2.plot(disp_dates, env_cmw[s_idx:e_idx], 'r-', linewidth=1.5, alpha=0.5)
ax2.plot(disp_dates, -env_cmw[s_idx:e_idx], 'r-', linewidth=1.5, alpha=0.5)
ax2.axhline(0, color='gray', linewidth=0.4)
ax2.set_ylabel('Amplitude')
ax2.set_title(f"CMW (FWHM design) — f0={CMW_PARAMS['f0']:.2f} rad/yr, "
              f"FWHM={CMW_PARAMS['fwhm']:.2f} rad/yr")
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# Row 3: Overlay — both signals and envelopes
ax3.plot(disp_dates, sig_orm[s_idx:e_idx], 'b-', linewidth=0.6,
         label='Ormsby signal', alpha=0.7)
ax3.plot(disp_dates, sig_cmw[s_idx:e_idx], 'r-', linewidth=0.6,
         label='CMW signal', alpha=0.7)
ax3.plot(disp_dates, env_orm[s_idx:e_idx], 'b-', linewidth=1.5,
         label='Ormsby envelope', alpha=0.5)
ax3.plot(disp_dates, -env_orm[s_idx:e_idx], 'b-', linewidth=1.5, alpha=0.5)
ax3.plot(disp_dates, env_cmw[s_idx:e_idx], 'r--', linewidth=1.5,
         label='CMW envelope', alpha=0.7)
ax3.plot(disp_dates, -env_cmw[s_idx:e_idx], 'r--', linewidth=1.5, alpha=0.7)
ax3.axhline(0, color='gray', linewidth=0.4)
ax3.set_ylabel('Envelope')
ax3.set_title("Envelope Overlay — Ormsby vs CMW")
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)

ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.set_xlabel('Date')
plt.xticks(rotation=45)

fig.suptitle("Page 45: Ormsby vs CMW — FWHM-Matched Bandpass Comparison",
             fontsize=13, fontweight='bold')
plt.tight_layout()

out1 = os.path.join(script_dir, 'compare_ormsby_vs_cmw_time.png')
fig.savefig(out1, dpi=150, bbox_inches='tight')
print(f"  Saved: {out1}")

# ============================================================================
# PLOT 2: Frequency response comparison
# ============================================================================

print("Generating frequency response comparison...")

nfft = 8192
# Ormsby frequency response
from scipy.fft import fft
H_orm = fft(h_ormsby, n=nfft)
freqs_norm = np.arange(nfft) / nfft
pos_mask = freqs_norm <= 0.5
freqs_rad_orm = freqs_norm[pos_mask] * FS * TWOPI
H_orm_mag = np.abs(H_orm[pos_mask]) * 0.5  # analytic normalization

# CMW frequency response
cmw_result = cmw_freq_domain(CMW_PARAMS['f0'], CMW_PARAMS['fwhm'], FS, nfft,
                              analytic=True)
H_cmw = cmw_result['H']
freqs_rad_cmw = cmw_result['freqs_rad']
# Extract positive frequencies for plotting
cmw_pos_mask = freqs_rad_cmw >= 0
cmw_pos_idx = np.where(cmw_pos_mask)[0]
# Sort by frequency for clean plot
sort_order = np.argsort(freqs_rad_cmw[cmw_pos_idx])
freqs_cmw_plot = freqs_rad_cmw[cmw_pos_idx][sort_order]
H_cmw_plot = H_cmw[cmw_pos_idx][sort_order] * 0.5  # match analytic scale

fig2, ax = plt.subplots(figsize=(12, 6))

ax.plot(freqs_rad_orm, H_orm_mag, 'b-', linewidth=1.5, label='Ormsby |H(f)|')
ax.plot(freqs_cmw_plot, H_cmw_plot, 'r--', linewidth=1.5, label='CMW Gaussian')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='0.5 gain (FWHM)')

# Mark Ormsby corners
for w, lbl in [(ORMSBY_SPEC['f1'], 'w1'), (ORMSBY_SPEC['f2'], 'w2'),
               (ORMSBY_SPEC['f3'], 'w3'), (ORMSBY_SPEC['f4'], 'w4')]:
    ax.axvline(w, color='blue', linestyle=':', alpha=0.3)
    ax.text(w, 1.05, lbl, ha='center', fontsize=8, color='blue')

# Mark FWHM points
lower_fwhm = (ORMSBY_SPEC['f1'] + ORMSBY_SPEC['f2']) / 2
upper_fwhm = (ORMSBY_SPEC['f3'] + ORMSBY_SPEC['f4']) / 2
ax.plot([lower_fwhm, upper_fwhm], [0.5, 0.5], 'ro', markersize=8,
        label=f'FWHM targets ({lower_fwhm:.2f}, {upper_fwhm:.2f})')

ax.set_xlim(0, 12)
ax.set_ylim(0, 1.15)
ax.set_xlabel('Frequency (rad/year)')
ax.set_ylabel('Magnitude')
ax.set_title('Frequency Response: Ormsby Trapezoid vs CMW Gaussian\n'
             f'fc={CMW_PARAMS["f0"]:.2f} rad/yr, FWHM={CMW_PARAMS["fwhm"]:.2f} rad/yr')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()

out2 = os.path.join(script_dir, 'compare_ormsby_vs_cmw_freq.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")

plt.show()
print("\nDone.")
