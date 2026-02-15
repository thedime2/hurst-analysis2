# -*- coding: utf-8 -*-
"""
Phase 5D: Filter Derivation Verification

Extends the analysis in analyze_filter_derivation.py with quantitative
verification of how Hurst derived the 6 page-152 filter frequencies:

  1. Cyclitec-to-filter mapping with quantitative error analysis
  2. Energy-optimal filter specs (maximize captured Lanczos energy)
  3. Sensitivity curves (energy vs filter center frequency)
  4. Missing cycle analysis (which Cyclitec cycles are excluded and why)

Produces a multi-panel figure and a comparison table.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Chapter IX (p. 152) and Appendix A;
           Cyclitec Services Training Course (1973-75)
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
NYQUIST = np.pi * FS  # ~163.4 rad/yr

# ============================================================================
# CYCLITEC COURSE NOMINAL CYCLES (strict harmonicity, 1973-75)
# ============================================================================

CYCLITEC_CYCLES = {
    '18 yr':   18.0,
    '9 yr':    9.0,
    '54 mo':   54.0 / 12.0,    # 4.5 yr
    '18 mo':   18.0 / 12.0,    # 1.5 yr
    '40 wk':   40.0 / 52.0,    # 0.769 yr
    '20 wk':   20.0 / 52.0,    # 0.385 yr
    '80 day':  80.0 / 365.25,  # 0.219 yr
    '40 day':  40.0 / 365.25,  # 0.110 yr
    '20 day':  20.0 / 365.25,  # 0.055 yr
    '10 day':  10.0 / 365.25,  # 0.027 yr
    '5 day':   5.0 / 365.25,   # 0.014 yr
}

CYCLITEC_FREQS = {name: TWOPI / period for name, period in CYCLITEC_CYCLES.items()}

# Page 152 filter specs (user-estimated, same as reproduce_decomposition.py)
FILTER_SPECS = [
    {'label': 'LP-1', 'type': 'lp',
     'f_pass': 0.85, 'f_stop': 1.25, 'f_center': 0.0,
     'band_lo': 0.0, 'band_hi': 1.25, 'color': '#1f77b4'},
    {'label': 'BP-2', 'type': 'bp',
     'f1': 0.85, 'f2': 1.25, 'f3': 2.05, 'f4': 2.45,
     'f_center': 1.65, 'band_lo': 0.85, 'band_hi': 2.45, 'color': '#ff7f0e'},
    {'label': 'BP-3', 'type': 'bp',
     'f1': 3.20, 'f2': 3.55, 'f3': 6.35, 'f4': 6.70,
     'f_center': 4.95, 'band_lo': 3.20, 'band_hi': 6.70, 'color': '#2ca02c'},
    {'label': 'BP-4', 'type': 'bp',
     'f1': 7.25, 'f2': 7.55, 'f3': 9.55, 'f4': 9.85,
     'f_center': 8.55, 'band_lo': 7.25, 'band_hi': 9.85, 'color': '#d62728'},
    {'label': 'BP-5', 'type': 'bp',
     'f1': 13.65, 'f2': 13.95, 'f3': 19.35, 'f4': 19.65,
     'f_center': 16.65, 'band_lo': 13.65, 'band_hi': 19.65, 'color': '#9467bd'},
    {'label': 'BP-6', 'type': 'bp',
     'f1': 28.45, 'f2': 28.75, 'f3': 35.95, 'f4': 36.25,
     'f_center': 32.35, 'band_lo': 28.45, 'band_hi': 36.25, 'color': '#8c564b'},
]

# Mapping: which Cyclitec cycle(s) each filter targets
FILTER_TO_CYCLITEC = {
    'LP-1': ['18 yr', '9 yr'],
    'BP-2': ['54 mo'],
    'BP-3': ['18 mo'],
    'BP-4': ['40 wk'],
    'BP-5': ['20 wk'],
    'BP-6': ['80 day'],
}

print("=" * 70)
print("Phase 5D: Filter Derivation Verification")
print("=" * 70)
print()

# ============================================================================
# LOAD DATA AND COMPUTE SPECTRUM
# ============================================================================

print("Loading DJIA weekly data...")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between(DATE_START, DATE_END)]
close_prices = df_hurst.Close.values
print(f"  {len(close_prices)} samples")

print("Computing Lanczos spectrum...")
w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(
    close_prices, 1, FS
)
omega_yr = w * FS
total_energy = np.sum(amp ** 2)

print("Loading Phase 3 nominal model...")
nominal_df = pd.read_csv(nominal_csv_path)
nominal_freqs = nominal_df['frequency'].values
print(f"  {len(nominal_freqs)} lines")
print()

# ============================================================================
# ANALYSIS 1: Cyclitec Mapping with Error Analysis
# ============================================================================

print("=" * 60)
print("ANALYSIS 1: Cyclitec Cycle -> Filter Mapping")
print("=" * 60)
print()

print(f"{'Cyclitec':>10s}  {'Period':>8s}  {'omega':>8s}  "
      f"{'Filter':>6s}  {'Filt fc':>8s}  {'Delta':>8s}  {'Status':>16s}")
print("-" * 80)

# Track which Cyclitec cycles are accessible with weekly data
for name, period_yr in sorted(CYCLITEC_CYCLES.items(),
                               key=lambda x: x[1], reverse=True):
    freq = CYCLITEC_FREQS[name]

    # Find which filter captures this
    mapped_filter = None
    for flabel, cycles in FILTER_TO_CYCLITEC.items():
        if name in cycles:
            mapped_filter = flabel
            break

    # Get filter center
    filt_center = '--'
    delta = '--'
    status = ''

    if freq > NYQUIST:
        status = 'ABOVE NYQUIST'
    elif mapped_filter is not None:
        spec = next(s for s in FILTER_SPECS if s['label'] == mapped_filter)
        if spec['type'] == 'lp':
            filt_center = f"LP<{spec['f_stop']:.1f}"
            delta = '--'
            status = 'CAPTURED (LP)'
        else:
            fc = spec['f_center']
            filt_center = f"{fc:.2f}"
            delta = f"{fc - freq:+.2f}"
            status = 'CAPTURED'
    else:
        status = 'NOT IN 6 FILTERS'

    print(f"  {name:>8s}  {period_yr:>7.3f}y  {freq:>7.2f}  "
          f"{'--' if mapped_filter is None else mapped_filter:>6s}  "
          f"{filt_center:>8s}  {delta:>8s}  {status}")

print()

# ============================================================================
# ANALYSIS 2: Energy Sensitivity and Cyclitec vs Visual Comparison
# ============================================================================

print("=" * 60)
print("ANALYSIS 2: Energy Sensitivity Around Cyclitec Targets")
print("=" * 60)
print()

# Note: naive energy maximization always favors lower frequencies
# because a(w)=k/w. Hurst designed filters for CYCLE ISOLATION,
# not energy maximization. The right question is: how close are
# the visual estimates to the Cyclitec targets, and how sensitive
# is the energy to small shifts around the intended center?


def compute_bandpass_energy(omega_yr, amp, f_center, bandwidth):
    """Compute spectral energy in a bandpass centered at f_center."""
    f_lo = f_center - bandwidth / 2
    f_hi = f_center + bandwidth / 2
    mask = (omega_yr >= f_lo) & (omega_yr <= f_hi)
    return np.sum(amp[mask] ** 2)


print(f"{'Filter':>6s}  {'Cyclitec':>8s}  {'Visual':>8s}  {'Vis-Cyc':>8s}  "
      f"{'E(Cyc)':>8s}  {'E(Vis)':>8s}  {'E ratio':>8s}")
print("-" * 70)

optimal_results = []

for spec in FILTER_SPECS:
    if spec['type'] == 'lp':
        mask = omega_yr <= spec['f_stop']
        energy_pct = np.sum(amp[mask] ** 2) / total_energy * 100
        print(f"  {spec['label']:>4s}  {'--':>8s}  {'LP':>8s}  {'--':>8s}  "
              f"{energy_pct:>7.1f}%  {energy_pct:>7.1f}%  {'--':>8s}")
        optimal_results.append({
            'label': spec['label'], 'type': 'lp',
            'cyclitec_freq': None, 'visual_center': None,
            'energy_cyclitec': energy_pct, 'energy_visual': energy_pct,
        })
        continue

    # Cyclitec target
    cyclitec_names = FILTER_TO_CYCLITEC[spec['label']]
    cyclitec_freq = np.mean([CYCLITEC_FREQS[c] for c in cyclitec_names])

    # Bandwidth
    bw_total = spec['f4'] - spec['f1']  # total span

    # Visual estimate
    visual_center = spec['f_center']

    # Energy at Cyclitec center vs Visual center
    energy_cyc = compute_bandpass_energy(omega_yr, amp, cyclitec_freq, bw_total)
    energy_vis = compute_bandpass_energy(omega_yr, amp, visual_center, bw_total)
    energy_cyc_pct = energy_cyc / total_energy * 100
    energy_vis_pct = energy_vis / total_energy * 100
    e_ratio = energy_vis / energy_cyc if energy_cyc > 0 else 0

    delta_cv = visual_center - cyclitec_freq

    print(f"  {spec['label']:>4s}  {cyclitec_freq:>7.2f}  {visual_center:>7.2f}  "
          f"{delta_cv:>+7.2f}  {energy_cyc_pct:>7.2f}%  {energy_vis_pct:>7.2f}%  "
          f"{e_ratio:>7.2f}x")

    # Sweep for sensitivity curve (narrow range: +/- 1 bandwidth)
    sweep_lo = max(bw_total / 2 + 0.1, visual_center - bw_total)
    sweep_hi = min(NYQUIST - bw_total / 2, visual_center + bw_total)
    sweep_centers = np.linspace(sweep_lo, sweep_hi, 200)
    sweep_energies = np.array([
        compute_bandpass_energy(omega_yr, amp, fc, bw_total)
        for fc in sweep_centers
    ]) / total_energy * 100

    optimal_results.append({
        'label': spec['label'], 'type': 'bp',
        'cyclitec_freq': cyclitec_freq, 'visual_center': visual_center,
        'energy_cyclitec': energy_cyc_pct, 'energy_visual': energy_vis_pct,
        'energy_ratio': e_ratio,
        'sweep_centers': sweep_centers, 'sweep_energies': sweep_energies,
        'bw_total': bw_total,
    })

print()
print("  E ratio > 1.0 means visual estimate captures MORE energy than Cyclitec center")
print("  (expected: visual shifted to capture spectral peaks within the nominal band)")
print()

# ============================================================================
# ANALYSIS 3: Missing Cycle Analysis
# ============================================================================

print("=" * 60)
print("ANALYSIS 3: Missing Cycles (Not in 6 Filters)")
print("=" * 60)
print()

for name, period_yr in sorted(CYCLITEC_CYCLES.items(),
                               key=lambda x: x[1], reverse=True):
    freq = CYCLITEC_FREQS[name]

    # Check if captured
    is_captured = any(name in cycles for cycles in FILTER_TO_CYCLITEC.values())

    if not is_captured:
        reason = ''
        if freq > NYQUIST:
            reason = 'Above Nyquist for weekly data'
        elif freq > FILTER_SPECS[-1]['f4']:
            reason = 'Above highest filter band'
        elif period_yr < 7.0 / 365.25:
            reason = 'Below weekly sampling limit'
        else:
            # Check if it falls in a gap
            in_gap = True
            for spec in FILTER_SPECS:
                if spec['type'] == 'lp':
                    if freq <= spec['f_stop']:
                        in_gap = False
                        break
                else:
                    if spec['f1'] <= freq <= spec['f4']:
                        in_gap = False
                        break
            if in_gap:
                reason = 'In gap between filters'
            else:
                reason = 'Covered but not primary target'

        print(f"  {name:>8s}: omega={freq:.2f} rad/yr, T={period_yr:.3f}y -- {reason}")

print()
print("  Note: Cycles below ~7 weeks (omega > ~47 rad/yr) exceed weekly Nyquist")
print("  at fs=52 samples/yr (Nyquist = pi*52 = 163.4 rad/yr), but are too fast")
print("  to resolve reliably with weekly data.")
print()

# ============================================================================
# FIGURE: Multi-Panel Verification
# ============================================================================

print("Generating multi-panel figure...")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 16),
                                      height_ratios=[3, 2, 1])

# --- Panel A: Spectrum + All Reference Points ---

ax1.semilogy(omega_yr, amp, 'k-', linewidth=0.5, alpha=0.7, label='Lanczos Spectrum')

# Filter passbands
for spec in FILTER_SPECS:
    if spec['type'] == 'lp':
        rect = Rectangle((0, 0.01), spec['f_stop'], 200,
                         alpha=0.1, facecolor=spec['color'])
    else:
        rect = Rectangle((spec['f1'], 0.01), spec['f4'] - spec['f1'], 200,
                         alpha=0.1, facecolor=spec['color'])
    ax1.add_patch(rect)

# Cyclitec target frequencies (red dashed)
for name, freq in CYCLITEC_FREQS.items():
    if freq <= 40:
        ax1.axvline(freq, color='red', linewidth=1.2, alpha=0.6, linestyle='--')
        ax1.text(freq + 0.1, 50, name, rotation=90, fontsize=6,
                color='red', alpha=0.8, va='top')

# Cyclitec target centers (green triangles)
for r in optimal_results:
    if r['type'] == 'bp' and r['cyclitec_freq'] is not None:
        ax1.plot(r['cyclitec_freq'], 0.015, '^', color='green',
                markersize=10, zorder=5)

# Visual estimate centers (blue diamonds)
for spec in FILTER_SPECS:
    if spec['type'] == 'bp':
        ax1.plot(spec['f_center'], 0.02, 'D', color='blue',
                markersize=8, zorder=5, alpha=0.7)

# Phase 3 nominal lines
for freq in nominal_freqs:
    ax1.axvline(freq, color='gray', linewidth=0.3, alpha=0.3, linestyle=':')

ax1.set_xlim(0, 40)
ax1.set_ylim(0.01, 100)
ax1.set_ylabel('Amplitude (log scale)', fontsize=11)
ax1.set_title(
    'Panel A: Lanczos Spectrum + Cyclitec Targets (red) + '
    'Visual Estimates (blue) + Energy-Optimal (green)',
    fontsize=12, fontweight='bold'
)
ax1.grid(True, alpha=0.2)

legend_items = [
    plt.Line2D([0], [0], color='k', linewidth=0.5, label='Lanczos Spectrum'),
    plt.Line2D([0], [0], color='red', linestyle='--', label='Cyclitec Targets'),
    plt.Line2D([0], [0], marker='D', color='blue', linestyle='None',
               markersize=6, label='Visual Estimates'),
    plt.Line2D([0], [0], marker='^', color='green', linestyle='None',
               markersize=8, label='Cyclitec Targets (center)'),
    plt.Line2D([0], [0], color='gray', linestyle=':', alpha=0.5,
               label='Phase 3 Lines'),
]
for spec in FILTER_SPECS:
    legend_items.append(mpatches.Patch(facecolor=spec['color'], alpha=0.2,
                                        label=spec['label']))
ax1.legend(handles=legend_items, fontsize=7, loc='upper right', ncol=2)

# --- Panel B: Energy Sensitivity Curves ---

for r in optimal_results:
    if r['type'] != 'bp':
        continue
    spec = next(s for s in FILTER_SPECS if s['label'] == r['label'])
    ax2.plot(r['sweep_centers'], r['sweep_energies'],
             '-', color=spec['color'], linewidth=1.5,
             label=f"{r['label']} (bw={r['bw_total']:.1f})")
    # Mark visual center (diamond)
    ax2.plot(r['visual_center'], r['energy_visual'], 'D',
             color=spec['color'], markersize=8, zorder=5)
    # Mark Cyclitec target (triangle)
    ax2.plot(r['cyclitec_freq'], r['energy_cyclitec'], '^',
             color=spec['color'], markersize=8, zorder=5,
             markeredgecolor='black', markeredgewidth=0.5)

ax2.set_xlabel('Filter Center Frequency (rad/yr)', fontsize=11)
ax2.set_ylabel('Captured Energy (%)', fontsize=11)
ax2.set_title(
    'Panel B: Energy Sensitivity (diamond=visual, triangle=Cyclitec)',
    fontsize=12, fontweight='bold'
)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.2)
ax2.set_xlim(0, 40)

# --- Panel C: Period Ratio Cascade ---

bp_specs = [s for s in FILTER_SPECS if s['type'] == 'bp']
bp_labels = [s['label'] for s in bp_specs]
bp_periods = [TWOPI / s['f_center'] for s in bp_specs]
bp_colors = [s['color'] for s in bp_specs]

ax3.barh(range(len(bp_labels)), bp_periods, color=bp_colors,
         alpha=0.6, height=0.6)
ax3.set_yticks(range(len(bp_labels)))
ax3.set_yticklabels(bp_labels, fontsize=10)
ax3.set_xlabel('Period (years)', fontsize=11)
ax3.set_title('Panel C: Period Ratio Cascade (Principle of Harmonicity)',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.2, axis='x')

# Add ratio annotations
for i in range(1, len(bp_periods)):
    ratio = bp_periods[i-1] / bp_periods[i]
    mid_y = i - 0.5
    ax3.annotate(f'{ratio:.1f}:1', xy=(max(bp_periods) * 0.75, mid_y),
                fontsize=10, fontweight='bold', color='darkred',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='lightyellow', alpha=0.8))

# Mark Cyclitec target periods
for name, period_yr in CYCLITEC_CYCLES.items():
    if 0.1 < period_yr < max(bp_periods) * 1.3:
        ax3.axvline(period_yr, color='red', linewidth=0.8, alpha=0.4,
                    linestyle='--')

ax3.set_xlim(0, max(bp_periods) * 1.3)
ax3.invert_yaxis()

plt.tight_layout()
out_fig = os.path.join(script_dir, 'phase5D_derivation_verification.png')
fig.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"  Saved: {out_fig}")

# ============================================================================
# SAVE COMPARISON TABLE
# ============================================================================

out_txt = os.path.join(script_dir, 'phase5D_optimal_vs_estimated.txt')
with open(out_txt, 'w') as f:
    f.write("Phase 5D: Filter Derivation Verification -- Comparison Table\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"{'Filter':>6s}  {'Cyclitec':>10s}  {'Visual':>8s}  {'Vis-Cyc':>8s}  "
            f"{'E(Cyc)':>8s}  {'E(Vis)':>8s}  {'E ratio':>8s}\n")
    f.write("-" * 70 + "\n")

    for r in optimal_results:
        if r['type'] == 'lp':
            f.write(f"  {r['label']:>4s}  {'LP':>10s}  {'LP':>8s}  {'--':>8s}  "
                    f"{r['energy_visual']:>7.1f}%  {r['energy_visual']:>7.1f}%  {'--':>8s}\n")
        else:
            ct = r['cyclitec_freq']
            vc = r['visual_center']
            delta = vc - ct
            er = r.get('energy_ratio', 0)
            f.write(f"  {r['label']:>4s}  {ct:>10.2f}  {vc:>8.2f}  {delta:>+7.2f}  "
                    f"{r['energy_cyclitec']:>7.2f}%  {r['energy_visual']:>7.2f}%  "
                    f"{er:>7.2f}x\n")

    f.write(f"\n  E ratio > 1.0 means visual estimate captures MORE energy\n")
    f.write(f"  than Cyclitec center (filter shifted to catch spectral peaks).\n\n")

    f.write("Key:\n")
    f.write("  Cyclitec = target frequency from Cyclitec Course (rad/yr)\n")
    f.write("  Visual = our center frequency estimated from book graphics (rad/yr)\n")
    f.write("  Vis-Cyc = visual - Cyclitec offset (positive = shifted higher)\n")
    f.write("  E ratio = E(Visual) / E(Cyclitec)\n")

print(f"  Saved: {out_txt}")

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()

print("  Visual vs Cyclitec center offsets:")
for r in optimal_results:
    if r['type'] == 'bp':
        delta_cv = r['visual_center'] - r['cyclitec_freq']
        er = r.get('energy_ratio', 0)
        print(f"    {r['label']}: Cyclitec={r['cyclitec_freq']:.2f}, "
              f"Visual={r['visual_center']:.2f} "
              f"(shift: {delta_cv:+.2f} rad/yr, "
              f"energy ratio: {er:.2f}x)")

print()
print("  Key findings:")
print("  - All visual estimates are shifted HIGHER than Cyclitec targets")
print("  - This is expected: a(w)=k/w envelope means energy is higher at")
print("    lower frequencies, so centering ON the Cyclitec target already")
print("    captures more low-freq energy than high-freq energy")
print("  - Hurst designed filters for CYCLE ISOLATION (one dominant cycle")
print("    per filter) not for maximum energy capture")
print("  - The Principle of Harmonicity (~2:1) determines filter SPACING")
print("  - The Principle of Variation (~20-30%) determines filter BANDWIDTH")

plt.show()
print()
print("Done.")
