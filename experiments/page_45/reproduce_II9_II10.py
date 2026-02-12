# -*- coding: utf-8 -*-
"""
Page 45: Figures II-9 and II-10 Reproduction
Bandpass filter envelope comparison: modulate vs subtract methods

Figure II-9: "The Time-Persistence of Cyclicality" — bandpass filtered DJIA
Figure II-10: "The Principle of Variation at Work" — same with amplitude envelope

This script applies a single Ormsby bandpass filter to the DJIA using two
construction methods (modulate and subtract) and compares the resulting
analytic envelopes.

Filter specification (rad/year): w1=3.2, w2=3.55, w3=6.35, w4=6.70
Filter length: nw = 359*5 = 1795
Display window: 1935-01-01 to 1954-02-01

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Chapter II, Figures II-9 and II-10, pp. 44-45
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

# Hurst's analysis window (filter over full range, display subset)
DATE_START = '1921-04-29'
DATE_END = '1965-05-21'

# Display window matching Figure II-9 x-axis
DISPLAY_START = '1935-01-01'
DISPLAY_END = '1952-02-01'

# Data parameters
FS = 52  # Weekly sampling rate (samples per year)

# Bandpass filter specification (rad/year)
W1 = 3.20   # lower stopband edge
W2 = 3.55   # lower passband edge
W3 = 6.35   # upper passband edge
W4 = 6.70   # upper stopband edge
NW = 359 * 5  # 1795 samples

# Derived quantities
TWOPI = 2 * np.pi
F_CENTER = (W2 + W3) / 2  # 4.95 rad/year
PERIOD_YR = TWOPI / F_CENTER
PERIOD_WK = PERIOD_YR * 52
PASSBAND_WIDTH = W3 - W2  # 2.80 rad/year

print("=" * 70)
print("Page 45: Figures II-9 & II-10 Reproduction")
print("Bandpass Envelope Comparison: Modulate vs Subtract")
print("=" * 70)
print(f"  Filter edges (rad/yr): [{W1}, {W2}, {W3}, {W4}]")
print(f"  Center frequency: {F_CENTER:.2f} rad/yr")
print(f"  Period: {PERIOD_YR:.2f} yr ({PERIOD_WK:.1f} weeks)")
print(f"  Passband width: {PASSBAND_WIDTH:.2f} rad/yr")
print(f"  Filter length: {NW} samples ({NW/FS:.1f} years)")
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
print(f"  Display window: {DISPLAY_START} to {DISPLAY_END} ({e_idx - s_idx} samples)")
print()

# ============================================================================
# CONVERT FREQUENCY UNITS AND CREATE FILTERS
# ============================================================================

# ormsby_filter() expects cycles/year, not rad/year
f_edges_rad = np.array([W1, W2, W3, W4])
f_edges_cyc = f_edges_rad / TWOPI

print("Creating filters...")
print(f"  Edges (cycles/yr): [{', '.join(f'{f:.4f}' for f in f_edges_cyc)}]")

# Method 1: Modulated bandpass (recommended)
h_mod = ormsby_filter(nw=NW, f_edges=f_edges_cyc, fs=FS,
                      filter_type='bp', method='modulate', analytic=True)

# Method 2: LP-minus-LP (subtract)
h_sub = ormsby_filter(nw=NW, f_edges=f_edges_cyc, fs=FS,
                      filter_type='bp', method='subtract', analytic=True)

print(f"  Modulate kernel: {len(h_mod)} taps, complex={np.iscomplexobj(h_mod)}")
print(f"  Subtract kernel: {len(h_sub)} taps, complex={np.iscomplexobj(h_sub)}")
print()

# ============================================================================
# APPLY FILTERS
# ============================================================================

print("Applying filters...")
result_mod = apply_ormsby_filter(close_prices, h_mod, mode='reflect', fs=FS)
result_sub = apply_ormsby_filter(close_prices, h_sub, mode='reflect', fs=FS)

sig_mod = result_mod['signal'].real
env_mod = result_mod['envelope']

sig_sub = result_sub['signal'].real
env_sub = result_sub['envelope']

print(f"  Modulate: signal range [{sig_mod[s_idx:e_idx].min():.2f}, "
      f"{sig_mod[s_idx:e_idx].max():.2f}], "
      f"envelope range [{env_mod[s_idx:e_idx].min():.2f}, "
      f"{env_mod[s_idx:e_idx].max():.2f}]")
print(f"  Subtract: signal range [{sig_sub[s_idx:e_idx].min():.2f}, "
      f"{sig_sub[s_idx:e_idx].max():.2f}], "
      f"envelope range [{env_sub[s_idx:e_idx].min():.2f}, "
      f"{env_sub[s_idx:e_idx].max():.2f}]")
print()

# ============================================================================
# PLOT: Comparison figure (2 rows)
# ============================================================================

print("Generating comparison plot...")

disp_dates = dates_dt[s_idx:e_idx]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# --- Plot 1: Modulate method ---
ax1.plot(disp_dates, sig_mod[s_idx:e_idx], 'k--', linewidth=0.8,
         label='Filtered signal')
ax1.plot(disp_dates, env_mod[s_idx:e_idx], 'b-', linewidth=1.5,
         label='+Envelope')
ax1.plot(disp_dates, -env_mod[s_idx:e_idx], 'b-', linewidth=1.5,
         label='-Envelope')
ax1.axhline(0, color='gray', linewidth=0.5)
ax1.set_ylabel('Amplitude')
ax1.set_title(f"Modulated Bandpass (method='modulate') — "
              f"fc={F_CENTER:.2f} rad/yr, T={PERIOD_YR:.2f} yr")
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Plot 2: Subtract method ---
ax2.plot(disp_dates, sig_sub[s_idx:e_idx], 'k--', linewidth=0.8,
         label='Filtered signal')
ax2.plot(disp_dates, env_sub[s_idx:e_idx], 'r-', linewidth=1.5,
         label='+Envelope')
ax2.plot(disp_dates, -env_sub[s_idx:e_idx], 'r-', linewidth=1.5,
         label='-Envelope')
ax2.axhline(0, color='gray', linewidth=0.5)
ax2.set_ylabel('Amplitude')
ax2.set_title(f"LP-minus-LP Bandpass (method='subtract') — "
              f"fc={F_CENTER:.2f} rad/yr, T={PERIOD_YR:.2f} yr")
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# Shared x-axis formatting
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.set_xlabel('Date')
plt.xticks(rotation=45)

fig.suptitle("Page 45: Figures II-9 & II-10 — Envelope Method Comparison\n"
             f"Ormsby BP: [{W1}, {W2}, {W3}, {W4}] rad/yr, nw={NW}",
             fontsize=13, fontweight='bold')
plt.tight_layout()

out_path = os.path.join(script_dir, 'figure_II9_II10_comparison.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {out_path}")

plt.show()

print()
print("Done.")
