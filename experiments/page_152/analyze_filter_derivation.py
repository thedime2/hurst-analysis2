# -*- coding: utf-8 -*-
"""
Analyze the Derivation of Hurst's 6 Page-152 Band-Pass Filters

This script investigates HOW Hurst selected the 6 filter frequencies for the
page 152 decomposition and WHY those specific values were chosen.

Analysis:
  1. Map Hurst's Table II-1 nominal cycles to angular frequencies
  2. Map the 27-line nominal model from Phase 3 to angular frequencies
  3. Overlay both on the Lanczos spectrum with the 6 filter passbands
  4. Show which nominal cycles fall inside each filter
  5. Compute spectral energy inside vs outside filter passbands
  6. Test the geometric cascade hypothesis (~2:1 period ratios)
  7. Compare actual filter specs to "optimal" specs derived from the model

Key question: Did Hurst design the 6 filters BEFORE or AFTER deriving the
nominal model? Evidence suggests the page 152 decomposition (Chapter IX) comes
before the Appendix A spectral analysis in the book's presentation, but the
spectral work was done first chronologically.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Chapter IX (p. 152) and Appendix A
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

from src.spectral import lanczos_spectrum

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
nominal_csv_path = os.path.join(script_dir, '../../data/processed/nominal_model.csv')

DATE_START = '1921-04-29'
DATE_END = '1965-05-21'
FS = 52
TWOPI = 2 * np.pi

# ============================================================================
# HURST'S TABLE II-1 NOMINAL CYCLES (from Profit Magic, 1970)
# ============================================================================

# Original Profit Magic periods (Chapter II, Table II-1)
PROFIT_MAGIC_CYCLES = {
    '18 year':    18.1,     # years
    '9 year':     9.0,
    '54 month':   4.48,     # 53.77 months = 4.48 years
    '18 month':   1.49,     # 17.93 months = 1.49 years
    '40 week':    0.750,    # 38.97 weeks = 0.750 years
    '20 week':    0.375,    # 19.48 weeks = 0.375 years
    '80 day':     0.187,    # 68.2 days = 9.74 weeks = 0.187 years
    '40 day':     0.0935,   # 34.1 days = 4.87 weeks
    '20 day':     0.0466,   # 17 days = 2.43 weeks
    '10 day':     0.0233,   # 8.5 days
    '5 day':      0.0118,   # 4.3 days
}

# Convert to rad/year
NOMINAL_FREQS = {}
for name, period_yr in PROFIT_MAGIC_CYCLES.items():
    NOMINAL_FREQS[name] = TWOPI / period_yr

# ============================================================================
# PAGE 152 FILTER SPECIFICATIONS (user-estimated from visual inspection)
# ============================================================================

FILTER_SPECS = [
    {
        'label': 'LP-1: Trend (>5 yr)',
        'type': 'lp',
        'f_pass': 0.85, 'f_stop': 1.25,
        'band_lo': 0.0, 'band_hi': 1.25,
        'color': '#1f77b4',
    },
    {
        'label': 'BP-2: ~3.8 yr',
        'type': 'bp',
        'f1': 0.85, 'f2': 1.25, 'f3': 2.05, 'f4': 2.45,
        'band_lo': 1.25, 'band_hi': 2.05,
        'f_center': (1.25 + 2.05) / 2,
        'color': '#ff7f0e',
    },
    {
        'label': 'BP-3: ~1.3 yr',
        'type': 'bp',
        'f1': 3.20, 'f2': 3.55, 'f3': 6.35, 'f4': 6.70,
        'band_lo': 3.55, 'band_hi': 6.35,
        'f_center': (3.55 + 6.35) / 2,
        'color': '#2ca02c',
    },
    {
        'label': 'BP-4: ~0.7 yr',
        'type': 'bp',
        'f1': 7.25, 'f2': 7.55, 'f3': 9.55, 'f4': 9.85,
        'band_lo': 7.55, 'band_hi': 9.55,
        'f_center': (7.55 + 9.55) / 2,
        'color': '#d62728',
    },
    {
        'label': 'BP-5: ~0.4 yr',
        'type': 'bp',
        'f1': 13.65, 'f2': 13.95, 'f3': 19.35, 'f4': 19.65,
        'band_lo': 13.95, 'band_hi': 19.35,
        'f_center': (13.95 + 19.35) / 2,
        'color': '#9467bd',
    },
    {
        'label': 'BP-6: ~0.2 yr',
        'type': 'bp',
        'f1': 28.45, 'f2': 28.75, 'f3': 35.95, 'f4': 36.25,
        'band_lo': 28.75, 'band_hi': 35.95,
        'f_center': (28.75 + 35.95) / 2,
        'color': '#8c564b',
    },
]

# ============================================================================
# LOAD DATA AND COMPUTE SPECTRUM
# ============================================================================

print("=" * 80)
print("Analysis: How Hurst Derived the 6 Page-152 Band-Pass Filters")
print("=" * 80)
print()

print("Loading DJIA weekly data...")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between(DATE_START, DATE_END)]
close_prices = df_hurst.Close.values
print(f"  {len(close_prices)} samples, {DATE_START} to {DATE_END}")

print("Computing Lanczos spectrum...")
w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(
    close_prices, 1, FS
)
omega_yr = w * FS

# Load Phase 3 nominal model
print("Loading Phase 3 nominal model...")
nominal_df = pd.read_csv(nominal_csv_path)
phase3_freqs = nominal_df['frequency'].values
phase3_periods_wk = nominal_df['period_weeks'].values
print(f"  {len(phase3_freqs)} lines, {phase3_freqs[0]:.2f}-{phase3_freqs[-1]:.2f} rad/yr")
print()

# ============================================================================
# ANALYSIS 1: MAP NOMINAL CYCLES TO FILTER PASSBANDS
# ============================================================================

print("=" * 60)
print("ANALYSIS 1: Nominal Cycles Mapped to Filter Passbands")
print("=" * 60)
print()

print(f"{'Nominal Cycle':>14s}  {'Period':>8s}  {'omega':>8s}  {'Filter':>20s}")
print(f"{'-'*14}  {'-'*8}  {'-'*8}  {'-'*20}")

for name, freq in sorted(NOMINAL_FREQS.items(), key=lambda x: x[1]):
    period_yr = TWOPI / freq
    # Find which filter captures this cycle
    captured_by = "NONE (gap)"
    for spec in FILTER_SPECS:
        if spec['type'] == 'lp':
            if freq <= spec['f_stop']:
                captured_by = spec['label']
                break
        else:
            if spec['f1'] <= freq <= spec['f4']:
                captured_by = spec['label']
                break
    print(f"  {name:>12s}  {period_yr:>7.2f}y  {freq:>7.2f}  {captured_by}")

print()

# ============================================================================
# ANALYSIS 2: GEOMETRIC CASCADE (PERIOD RATIOS)
# ============================================================================

print("=" * 60)
print("ANALYSIS 2: Geometric Cascade Between Filters")
print("=" * 60)
print()

print("Testing hypothesis: adjacent filters have ~2:1 period ratios")
print()

filter_centers = []
for spec in FILTER_SPECS:
    if spec['type'] == 'lp':
        # Use f_pass as effective center for LP
        filter_centers.append(spec['f_pass'] / 2)  # rough center
    else:
        filter_centers.append(spec['f_center'])

print(f"{'Filter':>20s}  {'Center':>8s}  {'Period':>8s}  {'Ratio to Prev':>14s}")
print(f"{'-'*20}  {'-'*8}  {'-'*8}  {'-'*14}")

prev_period = None
for i, spec in enumerate(FILTER_SPECS):
    if spec['type'] == 'lp':
        center = 0.0
        period_yr = float('inf')
        period_str = ">5 yr"
    else:
        center = spec['f_center']
        period_yr = TWOPI / center
        period_str = f"{period_yr:.2f}y"

    if prev_period is not None and period_yr != float('inf'):
        ratio = prev_period / period_yr
        ratio_str = f"{ratio:.2f}:1"
    else:
        ratio_str = "---"

    print(f"  {spec['label']:>18s}  {center:>7.2f}  {period_str:>8s}  {ratio_str:>14s}")

    if period_yr != float('inf'):
        prev_period = period_yr

print()
print("Expected ratios from Hurst's Principle of Harmonicity: ~2:1 or ~3:1")
print()

# ============================================================================
# ANALYSIS 3: SPECTRAL ENERGY INSIDE VS OUTSIDE PASSBANDS
# ============================================================================

print("=" * 60)
print("ANALYSIS 3: Spectral Energy Distribution")
print("=" * 60)
print()

total_energy = np.sum(amp**2)
filter_energies = []

for spec in FILTER_SPECS:
    if spec['type'] == 'lp':
        mask = omega_yr <= spec['f_stop']
    else:
        mask = (omega_yr >= spec['f2']) & (omega_yr <= spec['f3'])

    energy = np.sum(amp[mask]**2)
    pct = energy / total_energy * 100
    filter_energies.append((spec['label'], energy, pct))
    print(f"  {spec['label']:>20s}: {pct:5.1f}% of total spectral energy")

captured_energy = sum(e[1] for e in filter_energies)
gap_energy = total_energy - captured_energy
gap_pct = gap_energy / total_energy * 100
print(f"  {'Gaps (between)':>20s}: {gap_pct:5.1f}%")
print(f"  {'Total captured':>20s}: {100 - gap_pct:5.1f}%")
print()

# ============================================================================
# ANALYSIS 4: PHASE 3 NOMINAL LINES IN FILTER PASSBANDS
# ============================================================================

print("=" * 60)
print("ANALYSIS 4: Phase 3 Nominal Lines in Filter Passbands")
print("=" * 60)
print()

for spec in FILTER_SPECS:
    if spec['type'] == 'lp':
        mask = phase3_freqs <= spec['f_stop']
    else:
        mask = (phase3_freqs >= spec['f1']) & (phase3_freqs <= spec['f4'])

    lines_in = phase3_freqs[mask]
    n_lines = len(lines_in)

    if n_lines > 0:
        periods = TWOPI / lines_in * FS
        line_str = ', '.join([f"{f:.1f}" for f in lines_in])
        print(f"  {spec['label']:>18s}: {n_lines} lines [{line_str}] rad/yr")
    else:
        print(f"  {spec['label']:>18s}: 0 lines")

lines_in_any = np.zeros(len(phase3_freqs), dtype=bool)
for spec in FILTER_SPECS:
    if spec['type'] == 'lp':
        lines_in_any |= (phase3_freqs <= spec['f_stop'])
    else:
        lines_in_any |= (phase3_freqs >= spec['f1']) & (phase3_freqs <= spec['f4'])

n_outside = np.sum(~lines_in_any)
print(f"\n  Lines outside all filters: {n_outside}")
if n_outside > 0:
    outside_freqs = phase3_freqs[~lines_in_any]
    outside_periods = TWOPI / outside_freqs * FS
    for f, p in zip(outside_freqs, outside_periods):
        print(f"    {f:.2f} rad/yr ({p:.1f} weeks)")
print()

# ============================================================================
# ANALYSIS 5: OPTIMAL FILTERS FROM NOMINAL MODEL
# ============================================================================

print("=" * 60)
print("ANALYSIS 5: Optimal Filter Specs Derived From Nominal Model")
print("=" * 60)
print()

# Group nominal cycles that are close enough to be captured by a single filter.
# Hurst's Principle of Harmonicity says cycles come in ~2:1 ratios, so we group
# cycles that are within a factor of ~1.5 of each other.
# Then design a filter around each group.

nominal_sorted = sorted(NOMINAL_FREQS.items(), key=lambda x: x[1])
print("Nominal cycles in order of frequency:")
print()

# Define logical groupings based on the nominal model hierarchy
FILTER_GROUPS = [
    {
        'name': 'Long Trend',
        'cycles': ['18 year', '9 year'],
        'type': 'lp',
    },
    {
        'name': '54-month',
        'cycles': ['54 month'],
        'type': 'bp',
    },
    {
        'name': '18-month',
        'cycles': ['18 month'],
        'type': 'bp',
    },
    {
        'name': '40-week',
        'cycles': ['40 week'],
        'type': 'bp',
    },
    {
        'name': '20-week',
        'cycles': ['20 week'],
        'type': 'bp',
    },
    {
        'name': '80-day',
        'cycles': ['80 day'],
        'type': 'bp',
    },
]

print(f"{'Group':>14s}  {'Cycles':>30s}  {'Center':>8s}  {'BW (+-20%)':>12s}  {'Actual':>16s}")
print(f"{'-'*14}  {'-'*30}  {'-'*8}  {'-'*12}  {'-'*16}")

for i, group in enumerate(FILTER_GROUPS):
    cycle_freqs = [NOMINAL_FREQS[c] for c in group['cycles']]
    cycle_names = ', '.join(group['cycles'])

    if group['type'] == 'lp':
        # LP: pass everything below the lowest cycle in the group
        f_max = max(cycle_freqs)
        optimal_fpass = f_max * 1.2  # 20% margin
        actual = f"LP < {FILTER_SPECS[0]['f_stop']:.2f}"
        print(f"  {group['name']:>12s}  {cycle_names:>28s}  {'LP':>8s}  "
              f"<{optimal_fpass:.2f}       {actual:>16s}")
    else:
        f_center = np.mean(cycle_freqs)
        # +/- 20% variation bandwidth
        f_lo = f_center * 0.8
        f_hi = f_center * 1.2
        bw = f_hi - f_lo

        # Find actual filter
        actual_spec = FILTER_SPECS[i] if i < len(FILTER_SPECS) else None
        if actual_spec and actual_spec['type'] == 'bp':
            actual_center = actual_spec['f_center']
            actual_str = f"fc={actual_center:.2f}"
        else:
            actual_str = "---"

        print(f"  {group['name']:>12s}  {cycle_names:>28s}  {f_center:>7.2f}  "
              f"[{f_lo:.1f}-{f_hi:.1f}]  {actual_str:>16s}")

print()
print("Note: 'Optimal' assumes each filter captures ONE dominant nominal cycle")
print("with +/-20% bandwidth for the Principle of Variation.")
print()

# ============================================================================
# FIGURE: Diagnostic Overlay
# ============================================================================

print("Generating diagnostic overlay figure...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), height_ratios=[3, 1])

# --- Top panel: Lanczos spectrum + filter passbands + nominal lines ---
ax1.semilogy(omega_yr, amp, 'k-', linewidth=0.5, alpha=0.7, label='Lanczos Spectrum')

# Draw filter passbands as shaded rectangles
y_lo, y_hi = ax1.get_ylim()
for spec in FILTER_SPECS:
    if spec['type'] == 'lp':
        rect = Rectangle((0, 0.01), spec['f_stop'], 100,
                         alpha=0.12, facecolor=spec['color'],
                         edgecolor=spec['color'], linewidth=1.5)
    else:
        rect = Rectangle((spec['f2'], 0.01), spec['f3'] - spec['f2'], 100,
                         alpha=0.12, facecolor=spec['color'],
                         edgecolor=spec['color'], linewidth=1.5)
    ax1.add_patch(rect)

    # Draw transition bands (skirts) as lighter shading
    if spec['type'] == 'bp':
        # Lower skirt
        rect_lo = Rectangle((spec['f1'], 0.01), spec['f2'] - spec['f1'], 100,
                            alpha=0.05, facecolor=spec['color'])
        ax1.add_patch(rect_lo)
        # Upper skirt
        rect_hi = Rectangle((spec['f3'], 0.01), spec['f4'] - spec['f3'], 100,
                            alpha=0.05, facecolor=spec['color'])
        ax1.add_patch(rect_hi)

# Draw Hurst's Table II-1 nominal cycle frequencies as vertical lines
for name, freq in NOMINAL_FREQS.items():
    if freq <= 40:  # Only plot cycles within our display range
        ax1.axvline(x=freq, color='red', linewidth=1.0, alpha=0.6, linestyle='--')
        ax1.text(freq, 0.015, name, rotation=90, fontsize=6, color='red',
                ha='right', va='bottom', alpha=0.8)

# Draw Phase 3 nominal lines as thin vertical lines
for freq in phase3_freqs:
    ax1.axvline(x=freq, color='blue', linewidth=0.5, alpha=0.3, linestyle=':')

ax1.set_xlim(0, 40)
ax1.set_ylim(0.01, 100)
ax1.set_ylabel('Amplitude (log scale)', fontsize=11)
ax1.set_title('Hurst Page 152 Filter Derivation Analysis\n'
             'Lanczos Spectrum + Filter Passbands + Nominal Cycles',
             fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.2)

# Legend
legend_items = [
    mpatches.Patch(color='k', alpha=0.7, label='Lanczos Spectrum'),
    plt.Line2D([0], [0], color='red', linestyle='--', label='Table II-1 Nominal Cycles'),
    plt.Line2D([0], [0], color='blue', linestyle=':', alpha=0.5, label='Phase 3 Nominal Lines (27)'),
]
for spec in FILTER_SPECS:
    legend_items.append(mpatches.Patch(facecolor=spec['color'], alpha=0.3,
                                        label=spec['label']))
ax1.legend(handles=legend_items, fontsize=7, loc='upper right', ncol=2)

# --- Bottom panel: Period ratio cascade ---
filter_labels = []
filter_periods = []
for spec in FILTER_SPECS:
    if spec['type'] != 'lp':
        filter_labels.append(spec['label'])
        filter_periods.append(TWOPI / spec['f_center'])

# Also show nominal cycle periods
nominal_items = [(n, f) for n, f in sorted(NOMINAL_FREQS.items(), key=lambda x: x[1]) if f < 40 and TWOPI/f < 20]
nominal_names_sorted = [n for n, f in nominal_items]
nominal_periods = [TWOPI / f for n, f in nominal_items]

ax2.barh(range(len(filter_labels)), filter_periods, color=[s['color'] for s in FILTER_SPECS[1:]],
         alpha=0.6, height=0.6, label='Filter center periods')
ax2.set_yticks(range(len(filter_labels)))
ax2.set_yticklabels(filter_labels, fontsize=9)
ax2.set_xlabel('Period (years)', fontsize=11)
ax2.set_title('Filter Center Periods (Geometric Cascade)', fontsize=11)
ax2.grid(True, alpha=0.2, axis='x')

# Add ratio annotations
for i in range(1, len(filter_periods)):
    ratio = filter_periods[i-1] / filter_periods[i]
    mid_y = i - 0.5
    ax2.annotate(f'{ratio:.1f}:1', xy=(max(filter_periods) * 0.8, mid_y),
                fontsize=9, fontweight='bold', color='darkred',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

# Mark nominal cycle periods on the bar chart
for period, name in zip(nominal_periods, nominal_names_sorted):
    if period < max(filter_periods) * 1.2:
        ax2.axvline(x=period, color='red', linewidth=0.8, alpha=0.4, linestyle='--')
        ax2.text(period, len(filter_labels) - 0.2, name, rotation=90, fontsize=6,
                color='red', ha='right', va='top', alpha=0.7)

ax2.set_xlim(0, max(filter_periods) * 1.3)
ax2.invert_yaxis()

plt.tight_layout()
out_path = os.path.join(script_dir, 'filter_derivation_analysis.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {out_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("SUMMARY: How Hurst Derived the 6 Page-152 Filters")
print("=" * 80)
print()
print("DERIVATION CHAIN:")
print("  1. Fourier-Lanczos spectrum reveals discrete peaks with a(w)=k/w envelope")
print("  2. Overlapping comb filters confirm discrete line spectrum")
print("  3. Line grouping + LSE smoothing yield 27 nominal lines")
print("  4. Lines cluster around specific frequencies with ~0.37 rad/yr spacing")
print("  5. The DOMINANT lines correspond to nominal cycles (Table II-1)")
print()
print("FILTER DESIGN LOGIC:")
print("  - Each filter targets ONE dominant nominal cycle from the hierarchy")
print("  - Bandwidth accommodates the Principle of Variation (+/-20-30%)")
print("  - Adjacent filters follow ~2:1 period ratios (Principle of Harmonicity)")
print("  - Gaps between filters contain little spectral energy")
print("  - 6 filters capture the 6 most important cycles for trading (weekly data)")
print()
print("THE 2:1 CASCADE:")
print("  LP-1 (trend)  -->  9 yr")
print("  BP-2 (3.8 yr) -->  54-month cycle  (9/4.5 = 2:1)")
print("  BP-3 (1.3 yr) -->  18-month cycle  (4.5/1.5 = 3:1)")
print("  BP-4 (0.7 yr) -->  40-week cycle   (1.5/0.75 = 2:1)")
print("  BP-5 (0.4 yr) -->  20-week cycle   (0.75/0.375 = 2:1)")
print("  BP-6 (0.2 yr) -->  80-day cycle    (0.375/0.187 = 2:1)")
print()
print("WHY THESE SPECIFIC VALUES:")
print("  Hurst used 'Fourier analysis and a digital discovery algorithm' (his words)")
print("  to identify cycles that 'perfectly agreed with each other.' The nominal")
print("  cycle periods are EMPIRICAL -- derived from 44 years of DJIA data -- and")
print("  the filter frequencies are centered on these empirical periods with")
print("  bandwidths wide enough to capture natural variation but narrow enough")
print("  to separate adjacent nominal cycles.")
print()
print("KEY INSIGHT: The filters don't cover the full spectrum because there ISN'T")
print("significant spectral energy everywhere. The line spectrum means energy is")
print("concentrated at discrete frequencies, and the filters are designed to")
print("capture those concentrations while rejecting the gaps between them.")
print()

plt.show()
print("Done.")
