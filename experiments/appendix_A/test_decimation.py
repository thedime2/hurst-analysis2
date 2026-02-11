# -*- coding: utf-8 -*-
"""
Test & Demo: Decimated Filtering with Spacing, Offset, and Interpolation

Validates the decimation infrastructure for both Ormsby and CMW filter banks.
Tests spacing=1 regression, then compares spacing=5 with different interpolation
methods against the full-resolution baseline.

Plots:
  1. Per-filter overlay: baseline vs sparse vs 3point vs cubic
  2. Envelope comparison: baseline vs interpolated
  3. Console: RMS error table

Data range: 1921-04-29 to 1965-05-21
Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing, p.213
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.filters import (
    design_hurst_comb_bank,
    create_filter_kernels,
    apply_filter_bank,
)
from src.time_frequency.cmw import apply_cmw_bank, ormsby_spec_to_cmw_params


# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
output_dir = os.path.join(script_dir, '../../data/processed')
os.makedirs(output_dir, exist_ok=True)

DATE_START = '1921-04-29'
DATE_END = '1965-05-21'
FS = 52

# Comb filter bank parameters (same as phase2)
N_FILTERS = 23
W1_START = 7.2
W_STEP = 0.2
PASSBAND_WIDTH = 0.2
SKIRT_WIDTH = 0.3
NW = 1999

# Test parameters
SPACING = 5
OFFSETS = [1, 3]
INTERP_METHODS = ['none', '3point', 'cubic', 'linear']

# Representative filters for detailed plots (0-indexed)
REPR_FILTERS = [0, 11, 22]  # FC-1, FC-12, FC-23

# Display window for plots
PLOT_START = '1935-01-01'
PLOT_END = '1954-02-01'


# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("Decimation Test: Ormsby & CMW Filter Banks")
print("=" * 80)
print()

print("Loading DJIA weekly data...")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between(DATE_START, DATE_END)]
close_prices = df_hurst.Close.values
dates = df_hurst.Date.values
dates_dt = pd.to_datetime(dates)
print(f"  Data points: {len(close_prices)}")
print(f"  Date range: {DATE_START} to {DATE_END}")
print()


# ============================================================================
# DESIGN FILTER BANK
# ============================================================================

print("Designing comb filter bank...")
specs = design_hurst_comb_bank(
    n_filters=N_FILTERS,
    w1_start=W1_START,
    w_step=W_STEP,
    passband_width=PASSBAND_WIDTH,
    skirt_width=SKIRT_WIDTH,
    nw=NW,
    fs=FS
)
filters = create_filter_kernels(filter_specs=specs, fs=FS,
                                filter_type='modulate', analytic=True)
print(f"  Created {len(filters)} Ormsby filters, nw={NW}")
print()


# ============================================================================
# TEST 1: REGRESSION - spacing=1 should match original
# ============================================================================

print("-" * 60)
print("TEST 1: Regression check (spacing=1 == original)")
print("-" * 60)

results_baseline = apply_filter_bank(close_prices, filters, fs=FS, mode='reflect')
results_s1 = apply_filter_bank(close_prices, filters, fs=FS, mode='reflect',
                               spacing=1, offset=1, interp='none')

all_match = True
for i in range(len(filters)):
    sig_base = results_baseline['filter_outputs'][i]['signal']
    sig_s1 = results_s1['filter_outputs'][i]['signal']
    if not np.allclose(sig_base, sig_s1, equal_nan=True):
        print(f"  FAIL: Filter {i} output differs!")
        all_match = False

if all_match:
    print("  PASS: All filter outputs identical for spacing=1")
print()


# ============================================================================
# TEST 2: ORMSBY DECIMATION WITH DIFFERENT INTERP METHODS
# ============================================================================

print("-" * 60)
print(f"TEST 2: Ormsby decimation (spacing={SPACING})")
print("-" * 60)

# Compute decimated results for each interpolation method
ormsby_results = {}
for method in INTERP_METHODS:
    ormsby_results[method] = apply_filter_bank(
        close_prices, filters, fs=FS, mode='reflect',
        spacing=SPACING, offset=1, interp=method
    )

# Verify output lengths
for method in INTERP_METHODS:
    for i in range(len(filters)):
        out = ormsby_results[method]['filter_outputs'][i]
        for key in ['signal', 'envelope', 'phase', 'phasew', 'frequency']:
            if out[key] is not None:
                assert len(out[key]) == len(close_prices), \
                    f"Length mismatch: filter {i}, key '{key}', method '{method}'"
print(f"  PASS: All outputs have correct length ({len(close_prices)})")

# Envelope non-negativity check for interpolated methods
for method in ['3point', 'cubic', 'linear']:
    for i in range(len(filters)):
        env = ormsby_results[method]['filter_outputs'][i]['envelope']
        if env is not None:
            valid = env[~np.isnan(env)]
            if np.any(valid < -1e-10):
                print(f"  WARN: Negative envelope for filter {i}, method '{method}'")

# RMS error table
print()
print(f"  RMS Error vs Baseline (spacing={SPACING}, offset=1):")
print(f"  {'Filter':>8s}  {'f_center':>8s}  {'3point':>10s}  {'cubic':>10s}  {'linear':>10s}")
print(f"  {'':>8s}  {'(rad/yr)':>8s}  {'RMS':>10s}  {'RMS':>10s}  {'RMS':>10s}")
print(f"  {'-'*8:>8s}  {'-'*8:>8s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}")

rms_all = {m: [] for m in ['3point', 'cubic', 'linear']}
for i in REPR_FILTERS:
    sig_base = results_baseline['filter_outputs'][i]['signal']
    fc = specs[i]['f_center']
    row = f"  FC-{i+1:>3d}    {fc:>8.2f}"

    # For complex signals, compare envelopes
    env_base = results_baseline['filter_outputs'][i]['envelope']

    for method in ['3point', 'cubic', 'linear']:
        env_dec = ormsby_results[method]['filter_outputs'][i]['envelope']
        # Mask NaNs from baseline valid region
        valid = ~np.isnan(env_base) & ~np.isnan(env_dec)
        if valid.any():
            rms = np.sqrt(np.mean((env_base[valid] - env_dec[valid])**2))
            rel_rms = rms / (np.mean(env_base[valid]) + 1e-10) * 100
            row += f"  {rel_rms:>8.1f}%%"
            rms_all[method].append(rel_rms)
        else:
            row += f"  {'N/A':>9s}"
    print(row)

print()
for method in ['3point', 'cubic', 'linear']:
    if rms_all[method]:
        mean_rms = np.mean(rms_all[method])
        print(f"  Mean relative RMS ({method}): {mean_rms:.1f}%%")
print()


# ============================================================================
# TEST 3: OFFSET VARIATION
# ============================================================================

print("-" * 60)
print(f"TEST 3: Offset variation (spacing={SPACING})")
print("-" * 60)

for off in OFFSETS:
    res = apply_filter_bank(
        close_prices, filters, fs=FS, mode='reflect',
        spacing=SPACING, offset=off, interp='3point'
    )
    env = res['filter_outputs'][0]['envelope']
    valid = ~np.isnan(env)
    env_base = results_baseline['filter_outputs'][0]['envelope']
    valid_both = valid & ~np.isnan(env_base)
    if valid_both.any():
        rms = np.sqrt(np.mean((env_base[valid_both] - env[valid_both])**2))
        rel = rms / (np.mean(env_base[valid_both]) + 1e-10) * 100
        print(f"  offset={off}: FC-1 envelope relative RMS = {rel:.1f}%%")
print()


# ============================================================================
# TEST 4: CMW BANK DECIMATION
# ============================================================================

print("-" * 60)
print(f"TEST 4: CMW bank decimation (spacing={SPACING})")
print("-" * 60)

# Convert Ormsby specs to CMW params (just representative filters)
cmw_params = [ormsby_spec_to_cmw_params(specs[i]) for i in REPR_FILTERS]
print(f"  CMW params for {len(cmw_params)} representative filters")

# Baseline CMW
cmw_baseline = apply_cmw_bank(close_prices, cmw_params, fs=FS, analytic=True)

# Decimated CMW
cmw_dec = apply_cmw_bank(close_prices, cmw_params, fs=FS, analytic=True,
                         spacing=SPACING, offset=1, interp='3point')

# Regression: spacing=1
cmw_s1 = apply_cmw_bank(close_prices, cmw_params, fs=FS, analytic=True,
                        spacing=1, offset=1, interp='none')
cmw_match = True
for i in range(len(cmw_params)):
    env_base = cmw_baseline['filter_outputs'][i]['envelope']
    env_s1 = cmw_s1['filter_outputs'][i]['envelope']
    if env_base is not None and env_s1 is not None:
        if not np.allclose(env_base, env_s1, equal_nan=True):
            print(f"  FAIL: CMW filter {i} output differs for spacing=1!")
            cmw_match = False

if cmw_match:
    print("  PASS: CMW spacing=1 regression")

# RMS for decimated
print(f"\n  CMW RMS Error (spacing={SPACING}, interp='3point'):")
for j, i in enumerate(REPR_FILTERS):
    env_base = cmw_baseline['filter_outputs'][j]['envelope']
    env_dec = cmw_dec['filter_outputs'][j]['envelope']
    if env_base is not None and env_dec is not None:
        valid = ~np.isnan(env_base) & ~np.isnan(env_dec)
        if valid.any():
            rms = np.sqrt(np.mean((env_base[valid] - env_dec[valid])**2))
            rel = rms / (np.mean(env_base[valid]) + 1e-10) * 100
            print(f"  FC-{i+1}: relative RMS = {rel:.1f}%%")
print()


# ============================================================================
# PLOT 1: ORMSBY - SIGNAL OVERLAY (REPRESENTATIVE FILTERS)
# ============================================================================

# Get plot window indices
plot_mask = (dates_dt >= pd.to_datetime(PLOT_START)) & \
            (dates_dt <= pd.to_datetime(PLOT_END))
plot_idx = np.where(plot_mask)[0]
sl = slice(plot_idx[0], plot_idx[-1] + 1)
dates_plot = dates_dt[sl]

fig, axes = plt.subplots(len(REPR_FILTERS), 1, figsize=(16, 4*len(REPR_FILTERS)),
                         sharex=True)

for ax_idx, fi in enumerate(REPR_FILTERS):
    ax = axes[ax_idx]
    fc = specs[fi]['f_center']

    # Baseline envelope
    env_base = results_baseline['filter_outputs'][fi]['envelope'][sl]
    ax.plot(dates_plot, env_base, 'k-', linewidth=1.5, alpha=0.8, label='Baseline (full)')

    # Sparse (none) - dots
    env_none = ormsby_results['none']['filter_outputs'][fi]['envelope'][sl]
    valid_none = ~np.isnan(env_none)
    ax.plot(dates_plot[valid_none], env_none[valid_none], 'r.', markersize=4,
            alpha=0.6, label=f'Sparse (s={SPACING})')

    # 3point
    env_3pt = ormsby_results['3point']['filter_outputs'][fi]['envelope'][sl]
    ax.plot(dates_plot, env_3pt, 'b-', linewidth=1, alpha=0.7, label='3point')

    # Cubic
    env_cub = ormsby_results['cubic']['filter_outputs'][fi]['envelope'][sl]
    ax.plot(dates_plot, env_cub, 'g--', linewidth=1, alpha=0.7, label='Cubic')

    ax.set_ylabel('Envelope')
    ax.set_title(f'FC-{fi+1}: {fc:.1f} rad/yr')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Date')
fig.suptitle(f'Ormsby Decimation Test: Envelope Comparison (spacing={SPACING})',
             fontsize=13, y=1.01)
fig.tight_layout()

fig_path = os.path.join(script_dir, 'test_decimation_ormsby_envelopes.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fig_path}")
plt.close(fig)


# ============================================================================
# PLOT 2: CMW - SIGNAL OVERLAY (REPRESENTATIVE FILTERS)
# ============================================================================

# Also compute CMW with cubic for comparison
cmw_cubic = apply_cmw_bank(close_prices, cmw_params, fs=FS, analytic=True,
                           spacing=SPACING, offset=1, interp='cubic')

fig2, axes2 = plt.subplots(len(REPR_FILTERS), 1, figsize=(16, 4*len(REPR_FILTERS)),
                           sharex=True)

for ax_idx, fi in enumerate(REPR_FILTERS):
    ax = axes2[ax_idx]
    fc = cmw_params[ax_idx]['f0']

    env_base = cmw_baseline['filter_outputs'][ax_idx]['envelope'][sl]
    ax.plot(dates_plot, env_base, 'k-', linewidth=1.5, alpha=0.8, label='Baseline')

    env_3pt = cmw_dec['filter_outputs'][ax_idx]['envelope'][sl]
    ax.plot(dates_plot, env_3pt, 'b-', linewidth=1, alpha=0.7, label='3point')

    env_cub = cmw_cubic['filter_outputs'][ax_idx]['envelope'][sl]
    ax.plot(dates_plot, env_cub, 'g--', linewidth=1, alpha=0.7, label='Cubic')

    ax.set_ylabel('Envelope')
    ax.set_title(f'CMW FC-{fi+1}: {fc:.1f} rad/yr')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

axes2[-1].set_xlabel('Date')
fig2.suptitle(f'CMW Decimation Test: Envelope Comparison (spacing={SPACING})',
              fontsize=13, y=1.01)
fig2.tight_layout()

fig2_path = os.path.join(script_dir, 'test_decimation_cmw_envelopes.png')
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"Saved: {fig2_path}")
plt.close(fig2)


# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 60)
print("DECIMATION TEST COMPLETE")
print("=" * 60)
