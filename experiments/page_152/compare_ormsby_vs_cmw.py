# -*- coding: utf-8 -*-
"""
Page 152: Ormsby vs CMW Six-Filter Decomposition Comparison

Applies the same 6 filter specifications (1 LP + 5 BP) using both Ormsby
filters and FWHM-matched Complex Morlet Wavelets. Produces:
  1. Stacked time-domain overlay (Ormsby signals + CMW envelopes)
  2. Frequency response comparison (6 panels: trapezoid vs Gaussian)

Reference: J.M. Hurst, Profit Magic, p. 152
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.fft import fft

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

# ============================================================================
# FILTER SPECIFICATIONS (rad/year)
# ============================================================================

FILTER_SPECS = [
    {'type': 'lp', 'f_pass': 0.85, 'f_stop': 1.25,
     'f_center': (0.85 + 1.25) / 2, 'bandwidth': 1.25 - 0.85,
     'nw': 1393, 'index': 0, 'label': 'LP-1: Trend (>5 yr)'},
    {'type': 'bp', 'f1': 0.85, 'f2': 1.25, 'f3': 2.05, 'f4': 2.45,
     'f_center': (1.25 + 2.05) / 2, 'bandwidth': 2.05 - 1.25,
     'Q': (1.25 + 2.05) / 2 / (2.05 - 1.25),
     'Q_target': (1.25 + 2.05) / 2 / (2.05 - 1.25),
     'nw': 1393, 'index': 1, 'label': 'BP-2: ~3.8 yr'},
    {'type': 'bp', 'f1': 3.20, 'f2': 3.55, 'f3': 6.35, 'f4': 6.70,
     'f_center': (3.55 + 6.35) / 2, 'bandwidth': 6.35 - 3.55,
     'Q': (3.55 + 6.35) / 2 / (6.35 - 3.55),
     'Q_target': (3.55 + 6.35) / 2 / (6.35 - 3.55),
     'nw': 1245, 'index': 2, 'label': 'BP-3: ~1.3 yr'},
    {'type': 'bp', 'f1': 7.25, 'f2': 7.55, 'f3': 9.55, 'f4': 9.85,
     'f_center': (7.55 + 9.55) / 2, 'bandwidth': 9.55 - 7.55,
     'Q': (7.55 + 9.55) / 2 / (9.55 - 7.55),
     'Q_target': (7.55 + 9.55) / 2 / (9.55 - 7.55),
     'nw': 1745, 'index': 3, 'label': 'BP-4: ~0.7 yr'},
    {'type': 'bp', 'f1': 13.65, 'f2': 13.95, 'f3': 19.35, 'f4': 19.65,
     'f_center': (13.95 + 19.35) / 2, 'bandwidth': 19.35 - 13.95,
     'Q': (13.95 + 19.35) / 2 / (19.35 - 13.95),
     'Q_target': (13.95 + 19.35) / 2 / (19.35 - 13.95),
     'nw': 1299, 'index': 4, 'label': 'BP-5: ~0.4 yr'},
    {'type': 'bp', 'f1': 28.45, 'f2': 28.75, 'f3': 35.95, 'f4': 36.25,
     'f_center': (28.75 + 35.95) / 2, 'bandwidth': 35.95 - 28.75,
     'Q': (28.75 + 35.95) / 2 / (35.95 - 28.75),
     'Q_target': (28.75 + 35.95) / 2 / (35.95 - 28.75),
     'nw': 1299, 'index': 5, 'label': 'BP-6: ~0.2 yr'},
]

# Convert to CMW parameters
CMW_PARAMS = [ormsby_spec_to_cmw_params(s) for s in FILTER_SPECS]

print("=" * 70)
print("Page 152: Ormsby vs CMW Six-Filter Comparison")
print("=" * 70)
for i, (spec, cmw) in enumerate(zip(FILTER_SPECS, CMW_PARAMS)):
    print(f"  Filter {i+1}: {spec['label']:20s}  "
          f"CMW f0={cmw['f0']:.2f}, FWHM={cmw['fwhm']:.2f} rad/yr")
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
# APPLY ALL FILTERS
# ============================================================================

print("Applying Ormsby filters (modulate, analytic where applicable)...")
ormsby_outputs = []
for spec in FILTER_SPECS:
    if spec['type'] == 'lp':
        f_edges = np.array([spec['f_pass'], spec['f_stop']]) / TWOPI
        h = ormsby_filter(nw=spec['nw'], f_edges=f_edges, fs=FS,
                          filter_type='lp', analytic=False)
    else:
        f_edges = np.array([spec['f1'], spec['f2'],
                            spec['f3'], spec['f4']]) / TWOPI
        h = ormsby_filter(nw=spec['nw'], f_edges=f_edges, fs=FS,
                          filter_type='bp', method='modulate', analytic=True)
    result = apply_ormsby_filter(close_prices, h, mode='reflect', fs=FS)
    result['spec'] = spec
    result['kernel'] = h
    ormsby_outputs.append(result)

print("Applying matched CMW filters...")
cmw_outputs = []
for params in CMW_PARAMS:
    use_analytic = params['f0'] != 0
    result = apply_cmw(close_prices, params['f0'], params['fwhm'],
                        fs=FS, analytic=use_analytic)
    result['spec'] = params
    cmw_outputs.append(result)

# ============================================================================
# PLOT 1: Stacked time-domain comparison (6 rows)
# ============================================================================

print("Generating stacked time-domain comparison...")

# Compute reconstructions first (needed for the top subplot)
recon_orm = np.zeros_like(close_prices, dtype=float)
recon_cmw = np.zeros_like(close_prices, dtype=float)
for orm_out, cmw_out in zip(ormsby_outputs, cmw_outputs):
    sig_o = orm_out['signal']
    sig_c = cmw_out['signal']
    recon_orm += sig_o.real if np.iscomplexobj(sig_o) else sig_o
    recon_cmw += sig_c.real if np.iscomplexobj(sig_c) else sig_c

rms_orig = np.sqrt(np.mean(close_prices[s_idx:e_idx]**2))
rms_res_orm = np.sqrt(np.mean((close_prices[s_idx:e_idx] - recon_orm[s_idx:e_idx])**2))
rms_res_cmw = np.sqrt(np.mean((close_prices[s_idx:e_idx] - recon_cmw[s_idx:e_idx])**2))
pct_orm = (1 - rms_res_orm / rms_orig) * 100
pct_cmw = (1 - rms_res_cmw / rms_orig) * 100

n_filters = len(FILTER_SPECS)
# 7 rows: reconstruction + 6 filter outputs
fig, axes = plt.subplots(n_filters + 1, 1, figsize=(16, 18), sharex=True,
                          gridspec_kw={'height_ratios': [1.5] + [1]*n_filters})

# Row 0: DJIA close prices + summed reconstructions
ax_top = axes[0]
ax_top.plot(disp_dates, close_prices[s_idx:e_idx], 'k-', linewidth=1.0,
            label='DJIA Close', alpha=0.8)
ax_top.plot(disp_dates, recon_orm[s_idx:e_idx], 'b-', linewidth=1.2,
            label=f'Ormsby sum ({pct_orm:.1f}%)', alpha=0.7)
ax_top.plot(disp_dates, recon_cmw[s_idx:e_idx], 'r--', linewidth=1.2,
            label=f'CMW sum ({pct_cmw:.1f}%)', alpha=0.7)
ax_top.set_ylabel('Price', fontsize=9)
ax_top.set_title('DJIA Close + Summed Reconstructions', fontsize=10)
ax_top.legend(loc='upper left', fontsize=8)
ax_top.grid(True, alpha=0.2)
ax_top.tick_params(axis='y', labelsize=7)

# Rows 1-6: Individual filter outputs
for i, (orm_out, cmw_out) in enumerate(zip(ormsby_outputs, cmw_outputs)):
    ax = axes[i + 1]
    spec = FILTER_SPECS[i]
    cmw_p = CMW_PARAMS[i]

    # Ormsby filtered signal
    sig_orm = orm_out['signal']
    if np.iscomplexobj(sig_orm):
        sig_orm = sig_orm.real
    ax.plot(disp_dates, sig_orm[s_idx:e_idx], 'b-', linewidth=0.5,
            alpha=0.6, label='Ormsby' if i == 1 else None)

    # CMW filtered signal
    sig_cmw = cmw_out['signal']
    if np.iscomplexobj(sig_cmw):
        sig_cmw = sig_cmw.real
    ax.plot(disp_dates, sig_cmw[s_idx:e_idx], 'r-', linewidth=0.5,
            alpha=0.6, label='CMW' if i == 1 else None)

    # Ormsby envelope (if available)
    if orm_out['envelope'] is not None:
        env_orm = orm_out['envelope'][s_idx:e_idx]
        ax.plot(disp_dates, env_orm, 'b-', linewidth=1.2, alpha=0.5,
                label='Ormsby env' if i == 1 else None)
        ax.plot(disp_dates, -env_orm, 'b-', linewidth=1.2, alpha=0.5)

    # CMW envelope (if available)
    if cmw_out['envelope'] is not None:
        env_cmw = cmw_out['envelope'][s_idx:e_idx]
        ax.plot(disp_dates, env_cmw, 'r--', linewidth=1.2, alpha=0.7,
                label='CMW env' if i == 1 else None)
        ax.plot(disp_dates, -env_cmw, 'r--', linewidth=1.2, alpha=0.7)

    ax.axhline(0, color='gray', linewidth=0.3)
    ax.set_ylabel(spec['label'], fontsize=8, rotation=0,
                   labelpad=75, ha='left')
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis='y', labelsize=7)

    if i == 1:  # First BP filter (LP has no envelope labels)
        ax.legend(loc='upper right', fontsize=7)

    # Info annotation
    if spec['type'] == 'bp':
        period = TWOPI / spec['f_center']
        ax.text(0.99, 0.92,
                f"fc={spec['f_center']:.1f}, FWHM={cmw_p['fwhm']:.2f} rad/yr",
                transform=ax.transAxes, fontsize=7, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].set_xlabel('Date')
plt.xticks(rotation=45)

fig.suptitle("Page 152: Ormsby (blue) vs CMW (red) — Six-Filter Decomposition",
             fontsize=13, fontweight='bold')
plt.tight_layout()

out1 = os.path.join(script_dir, 'compare_ormsby_vs_cmw_time.png')
fig.savefig(out1, dpi=150, bbox_inches='tight')
print(f"  Saved: {out1}")

# ============================================================================
# PLOT 2: Frequency response comparison (6 panels)
# ============================================================================

print("Generating frequency response comparison...")

nfft = 8192
freqs_norm = np.arange(nfft) / nfft
pos_mask_fft = freqs_norm <= 0.5
freqs_rad_axis = freqs_norm[pos_mask_fft] * FS * TWOPI

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
axes2 = axes2.flatten()

for i, (ax, orm_out, cmw_p) in enumerate(zip(axes2, ormsby_outputs, CMW_PARAMS)):
    spec = FILTER_SPECS[i]

    # Ormsby frequency response
    h = orm_out['kernel']
    H_orm = fft(h, n=nfft)
    H_orm_mag = np.abs(H_orm[pos_mask_fft])
    if np.iscomplexobj(h):
        H_orm_mag *= 0.5

    # CMW frequency response
    cmw_result = cmw_freq_domain(cmw_p['f0'], cmw_p['fwhm'], FS, nfft,
                                  analytic=(cmw_p['f0'] != 0))
    H_cmw_full = cmw_result['H']
    freqs_cmw_full = cmw_result['freqs_rad']
    cmw_pos = freqs_cmw_full >= 0
    cmw_idx = np.where(cmw_pos)[0]
    sort_ord = np.argsort(freqs_cmw_full[cmw_idx])
    freqs_cmw_plot = freqs_cmw_full[cmw_idx][sort_ord]
    H_cmw_plot = H_cmw_full[cmw_idx][sort_ord]
    if cmw_p['f0'] != 0:
        H_cmw_plot *= 0.5

    ax.plot(freqs_rad_axis, H_orm_mag, 'b-', linewidth=1.2, label='Ormsby')
    ax.plot(freqs_cmw_plot, H_cmw_plot, 'r--', linewidth=1.2, label='CMW')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4)

    # Mark FWHM targets
    if spec['type'] == 'bp':
        lower = (spec['f1'] + spec['f2']) / 2
        upper = (spec['f3'] + spec['f4']) / 2
        ax.plot([lower, upper], [0.5, 0.5], 'ro', markersize=5, zorder=5)
        ax.set_xlim(max(0, spec['f1'] - 1), spec['f4'] + 1)
    else:
        upper = (spec['f_pass'] + spec['f_stop']) / 2
        ax.plot(upper, 0.5, 'ro', markersize=5, zorder=5)
        ax.set_xlim(0, spec['f_stop'] + 1)

    ax.set_ylim(0, 1.15)
    ax.set_title(spec['label'], fontsize=10)
    ax.set_xlabel('Freq (rad/yr)', fontsize=8)
    ax.set_ylabel('|H(f)|', fontsize=8)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)

fig2.suptitle("Frequency Response: Ormsby Trapezoid vs CMW Gaussian (6 Filters)",
              fontsize=13, fontweight='bold')
plt.tight_layout()

out2 = os.path.join(script_dir, 'compare_ormsby_vs_cmw_freq.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
print(f"  Saved: {out2}")

# ============================================================================
# Reconstruction comparison
# ============================================================================

print()
print("--- Reconstruction Comparison ---")
print(f"  Ormsby reconstruction: {pct_orm:.1f}% energy captured")
print(f"  CMW reconstruction:   {pct_cmw:.1f}% energy captured")

plt.show()
print("\nDone.")
