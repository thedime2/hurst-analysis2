# -*- coding: utf-8 -*-
"""
Page 152: Six-Filter Structural Decomposition Reproduction

Reproduces Hurst's six-filter decomposition of the DJIA:
  Filter 1: Lowpass  — passes < 0.85 rad/yr (periods > ~7.4 yr)
  Filter 2: Bandpass — 0.85–2.45 rad/yr (center 1.65, period ~3.8 yr)
  Filter 3: Bandpass — 3.20–6.70 rad/yr (center 4.95, period ~1.3 yr)
  Filter 4: Bandpass — 7.25–9.85 rad/yr (center 8.55, period ~0.7 yr)
  Filter 5: Bandpass — 13.65–19.65 rad/yr (center 16.65, period ~0.4 yr)
  Filter 6: Bandpass — 28.45–36.25 rad/yr (center 32.35, period ~0.2 yr)

Three rendering modes:
  1. Real-valued filters (no envelopes)
  2. Complex modulated bandpass with analytic envelopes
  3. Complex subtract bandpass with analytic envelopes

Filter specifications are initial estimates from visual inspection of Hurst's
graphics. These are starting points for reproduction.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           p. 152 and surrounding discussion
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
    apply_ormsby_filter
)

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')

# Hurst's analysis window
DATE_START = '1921-04-29'
DATE_END = '1965-05-21'

# Display window
DISPLAY_START = '1935-01-01'
DISPLAY_END = '1954-02-01'

# Data parameters
FS = 52
TWOPI = 2 * np.pi

# Spaced-kernel controls
DEFAULT_SPACING = 5
DEFAULT_OFFSET = 1
SPECIAL_SPACING = {
    'BP-2: ~3.8 yr': 7
}
SPECIAL_OFFSET = {
    'BP-2: ~3.8 yr': 3
}

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
        'label': 'LP-1: Trend (>5 yr)'
    },
    {
        'type': 'bp',
        'f1': 0.85, 'f2': 1.25, 'f3': 2.05, 'f4': 2.45,
        'f_center': (1.25 + 2.05) / 2,
        'bandwidth': 2.05 - 1.25,
        'Q': (1.25 + 2.05) / 2 / (2.05 - 1.25),
        'Q_target': (1.25 + 2.05) / 2 / (2.05 - 1.25),
        'nw': 1393,
        'index': 1,
        'label': 'BP-2: ~3.8 yr'
    },
    {
        'type': 'bp',
        'f1': 3.20, 'f2': 3.55, 'f3': 6.35, 'f4': 6.70,
        'f_center': (3.55 + 6.35) / 2,
        'bandwidth': 6.35 - 3.55,
        'Q': (3.55 + 6.35) / 2 / (6.35 - 3.55),
        'Q_target': (3.55 + 6.35) / 2 / (6.35 - 3.55),
        'nw': 1245,
        'index': 2,
        'label': 'BP-3: ~1.3 yr'
    },
    {
        'type': 'bp',
        'f1': 7.25, 'f2': 7.55, 'f3': 9.55, 'f4': 9.85,
        'f_center': (7.55 + 9.55) / 2,
        'bandwidth': 9.55 - 7.55,
        'Q': (7.55 + 9.55) / 2 / (9.55 - 7.55),
        'Q_target': (7.55 + 9.55) / 2 / (9.55 - 7.55),
        'nw': 1745,
        'index': 3,
        'label': 'BP-4: ~0.7 yr'
    },
    {
        'type': 'bp',
        'f1': 13.65, 'f2': 13.95, 'f3': 19.35, 'f4': 19.65,
        'f_center': (13.95 + 19.35) / 2,
        'bandwidth': 19.35 - 13.95,
        'Q': (13.95 + 19.35) / 2 / (19.35 - 13.95),
        'Q_target': (13.95 + 19.35) / 2 / (19.35 - 13.95),
        'nw': 1299,
        'index': 4,
        'label': 'BP-5: ~0.4 yr'
    },
    {
        'type': 'bp',
        'f1': 28.45, 'f2': 28.75, 'f3': 35.95, 'f4': 36.25,
        'f_center': (28.75 + 35.95) / 2,
        'bandwidth': 35.95 - 28.75,
        'Q': (28.75 + 35.95) / 2 / (35.95 - 28.75),
        'Q_target': (28.75 + 35.95) / 2 / (35.95 - 28.75),
        'nw': 1299,
        'index': 5,
        'label': 'BP-6: ~0.2 yr'
    },
]

# ============================================================================
# PRINT FILTER SUMMARY
# ============================================================================

print("=" * 70)
print("Page 152: Six-Filter Structural Decomposition")
print("=" * 70)
print()

for spec in FILTER_SPECS:
    if spec['type'] == 'lp':
        period = TWOPI / spec['f_pass']
        print(f"  {spec['label']:20s}  LP: pass<{spec['f_pass']:.2f}, "
              f"stop>{spec['f_stop']:.2f} rad/yr  "
              f"(T>{period:.1f} yr)  nw={spec['nw']}")
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
print(f"  Display window: {DISPLAY_START} to {DISPLAY_END} ({e_idx - s_idx} samples)")
print()


# ============================================================================
# HELPER: Create kernels with per-filter analytic control
# ============================================================================

def create_mixed_kernels(specs, fs, method, bp_analytic):
    """
    Create filter kernels where LP is always real-valued and BP filters
    use the specified analytic mode.
    """
    kernels = []
    for spec in specs:
        if spec['type'] == 'lp':
            # Lowpass: always real-valued (analytic envelope not meaningful)
            f_edges = np.array([spec['f_pass'], spec['f_stop']]) / TWOPI
            h = ormsby_filter(nw=spec['nw'], f_edges=f_edges, fs=fs,
                              filter_type='lp', analytic=False)
        else:
            # Bandpass: use specified method and analytic mode
            f_edges = np.array([spec['f1'], spec['f2'],
                                spec['f3'], spec['f4']]) / TWOPI
            h = ormsby_filter(nw=spec['nw'], f_edges=f_edges, fs=fs,
                              filter_type='bp', method=method,
                              analytic=bp_analytic)
        kernels.append({
            'kernel': h,
            'spec': spec,
            'nw': spec['nw']
        })
    return kernels


def _make_odd(n):
    """Return nearest odd integer >= 3."""
    n_int = max(3, int(round(n)))
    return n_int if (n_int % 2 == 1) else (n_int + 1)


def _expand_spaced_kernel(h_short, nw_long, spacing, offset):
    """
    Insert zeros between taps to create a spaced kernel with configurable offset.

    offset is 1-based and interpreted as a shift in the long-kernel index domain.
    """
    if spacing < 1:
        raise ValueError(f"spacing must be >= 1, got {spacing}")
    if offset < 1:
        raise ValueError(f"offset must be >= 1, got {offset}")
    if nw_long % 2 == 0:
        nw_long += 1

    h_long = np.zeros(nw_long, dtype=h_short.dtype)
    center_long = nw_long // 2
    center_short = len(h_short) // 2
    shift = offset - 1

    for i_short, tap in enumerate(h_short):
        rel = i_short - center_short
        i_long = center_long + rel * spacing + shift
        if 0 <= i_long < nw_long:
            h_long[i_long] = tap

    return h_long


def create_spaced_mixed_kernels(specs, fs, method, bp_analytic):
    """
    Create reduced Ormsby kernels and expand them with sparse spacing.

    Defaults: spacing=5, offset=1.
    Special case: BP-2 uses spacing=7 and offset=3.
    """
    kernels = []
    for spec in specs:
        label = spec['label']
        spacing = SPECIAL_SPACING.get(label, DEFAULT_SPACING)
        offset = SPECIAL_OFFSET.get(label, DEFAULT_OFFSET)

        # Reduce tap count by spacing factor, then place taps every "spacing" slots.
        nw_short = _make_odd(spec['nw'] / spacing)
        nw_long = _make_odd(spec['nw'])

        if spec['type'] == 'lp':
            f_edges = np.array([spec['f_pass'], spec['f_stop']]) / TWOPI
            h_short = ormsby_filter(
                nw=nw_short,
                f_edges=f_edges,
                fs=fs,
                filter_type='lp',
                analytic=False
            )
        else:
            f_edges = np.array([spec['f1'], spec['f2'], spec['f3'], spec['f4']]) / TWOPI
            h_short = ormsby_filter(
                nw=nw_short,
                f_edges=f_edges,
                fs=fs,
                filter_type='bp',
                method=method,
                analytic=bp_analytic
            )

        h_spaced = _expand_spaced_kernel(
            h_short=h_short, nw_long=nw_long, spacing=spacing, offset=offset
        )
        kernels.append({
            'kernel': h_spaced,
            'spec': spec,
            'nw': len(h_spaced),
            'nw_short': nw_short,
            'spacing': spacing,
            'offset': offset
        })
    return kernels


def apply_and_collect(kernels, signal, fs):
    """Apply all filter kernels and collect results."""
    outputs = []
    for filt in kernels:
        result = apply_ormsby_filter(signal, filt['kernel'],
                                     mode='reflect', fs=fs)
        result['spec'] = filt['spec']
        outputs.append(result)
    return outputs


# ============================================================================
# HELPER: Stacked 6-panel plot
# ============================================================================

def plot_decomposition(outputs, disp_dates, s_idx, e_idx,
                       title, show_envelope, filename):
    """Plot 6-panel stacked decomposition."""
    n_filters = len(outputs)
    fig, axes = plt.subplots(n_filters, 1, figsize=(16, 14), sharex=True)

    for i, (ax, out) in enumerate(zip(axes, outputs)):
        spec = out['spec']
        sig = out['signal']

        # Extract real part for display
        if np.iscomplexobj(sig):
            sig_real = sig.real[s_idx:e_idx]
        else:
            sig_real = sig[s_idx:e_idx]

        # Plot filtered signal
        ax.plot(disp_dates, sig_real, 'k-', linewidth=0.6)

        # Plot envelope if available and requested
        if show_envelope and out['envelope'] is not None:
            env = out['envelope'][s_idx:e_idx]
            ax.plot(disp_dates, env, 'b-', linewidth=1.2, alpha=0.8)
            ax.plot(disp_dates, -env, 'b-', linewidth=1.2, alpha=0.8)

        ax.axhline(0, color='gray', linewidth=0.4)
        ax.set_ylabel(spec['label'], fontsize=8, rotation=0,
                       labelpad=75, ha='left')
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='y', labelsize=7)

        # Add period info
        if spec['type'] == 'bp':
            period = TWOPI / spec['f_center']
            ax.text(0.99, 0.92,
                    f"fc={spec['f_center']:.1f} rad/yr, T={period:.2f} yr",
                    transform=ax.transAxes, fontsize=7, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.8))
        else:
            ax.text(0.99, 0.92,
                    f"pass<{spec['f_pass']:.2f} rad/yr",
                    transform=ax.transAxes, fontsize=7, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.8))

    # X-axis formatting on bottom plot
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlabel('Date')
    plt.xticks(rotation=45)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(script_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    return fig


# ============================================================================
# PLOT 1: Real-valued filters (no envelopes)
# ============================================================================

print("--- Plot 1: Real-valued filters ---")
kernels_real = create_spaced_mixed_kernels(FILTER_SPECS, FS, method='modulate',
                                           bp_analytic=False)
outputs_real = apply_and_collect(kernels_real, close_prices, FS)

fig1 = plot_decomposition(
    outputs_real, disp_dates, s_idx, e_idx,
    title="Page 152: Six-Filter Decomposition — Real-Valued Spaced Kernels",
    show_envelope=False,
    filename='page152_real_spaced_s5_bp2s7o3.png'
)

# ============================================================================
# PLOT 2: Complex modulated with envelopes
# ============================================================================

print("--- Plot 2: Complex modulated bandpass with envelopes ---")
kernels_mod = create_spaced_mixed_kernels(FILTER_SPECS, FS, method='modulate',
                                          bp_analytic=True)
outputs_mod = apply_and_collect(kernels_mod, close_prices, FS)

fig2 = plot_decomposition(
    outputs_mod, disp_dates, s_idx, e_idx,
    title="Page 152: Six-Filter Decomposition — Spaced Complex Modulated + Envelopes",
    show_envelope=True,
    filename='page152_complex_modulate_spaced_s5_bp2s7o3.png'
)

# ============================================================================
# PLOT 3: Complex subtract with envelopes
# ============================================================================

print("--- Plot 3: Complex subtract bandpass with envelopes ---")
kernels_sub = create_spaced_mixed_kernels(FILTER_SPECS, FS, method='subtract',
                                          bp_analytic=True)
outputs_sub = apply_and_collect(kernels_sub, close_prices, FS)

fig3 = plot_decomposition(
    outputs_sub, disp_dates, s_idx, e_idx,
    title="Page 152: Six-Filter Decomposition — Spaced Complex Subtract + Envelopes",
    show_envelope=True,
    filename='page152_complex_subtract_spaced_s5_bp2s7o3.png'
)

# ============================================================================
# SUMMARY: Reconstruction check
# ============================================================================

print()
print("--- Reconstruction Check ---")
# Sum of all real-valued filter outputs vs original
reconstruction = np.zeros_like(close_prices, dtype=float)
for out in outputs_real:
    sig = out['signal']
    if np.iscomplexobj(sig):
        reconstruction += sig.real
    else:
        reconstruction += sig

residual = close_prices - reconstruction
rms_orig = np.sqrt(np.mean(close_prices[s_idx:e_idx]**2))
rms_resid = np.sqrt(np.mean(residual[s_idx:e_idx]**2))
pct_captured = (1 - rms_resid / rms_orig) * 100

print(f"  RMS original (display window): {rms_orig:.2f}")
print(f"  RMS residual (display window): {rms_resid:.2f}")
print(f"  Energy captured: {pct_captured:.1f}%")
print(f"  Note: Gaps between filter passbands mean <100% is expected")

plt.show()

print()
print("Done.")
