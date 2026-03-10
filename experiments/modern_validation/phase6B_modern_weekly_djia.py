# -*- coding: utf-8 -*-
"""
Phase 6B: Modern Weekly DJIA — Transfer Test

Tests whether Hurst's 0.3676 rad/yr harmonic structure persists across
three eras of DJIA weekly data:

  Era 1: 1921-1965 (Hurst's original analysis window)
  Era 2: 1965-2005 (post-Hurst, pre-algorithmic)
  Era 3: 1985-2025 (modern, includes algo/HFT era)

Also runs on SPX weekly for cross-market comparison.

Reference: prd/hurst_unified_theory.md, prd/modern_validation_plan.md
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils_validation import (
    load_data, run_spectrum, measure_line_spacing,
    compute_harmonic_fit, hurst_similarity_score, HURST_SPACING
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# ERA DEFINITIONS
# =============================================================================

ERAS = [
    {'label': 'Hurst Era (1921-1965)',   'start': '1921-04-29', 'end': '1965-05-21',
     'color': '#1f77b4', 'short': 'Hurst'},
    {'label': 'Post-Hurst (1965-2005)',  'start': '1965-05-22', 'end': '2005-05-21',
     'color': '#ff7f0e', 'short': 'Post-Hurst'},
    {'label': 'Modern (1985-2025)',      'start': '1985-01-01', 'end': '2026-01-31',
     'color': '#2ca02c', 'short': 'Modern'},
]

SYMBOLS = [
    {'symbol': 'djia', 'name': 'DJIA'},
    {'symbol': 'spx',  'name': 'S&P 500'},
]

# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_era(symbol, freq, start, end, label, omega_max=14.0):
    """Run complete Hurst pipeline on one era."""
    print(f"\n{'='*60}")
    print(f"  {label} — {symbol.upper()} {freq}")
    print(f"{'='*60}")

    data = load_data(symbol=symbol, freq=freq, start=start, end=end)
    print(f"  Samples: {data['n_samples']}, fs={data['fs']:.1f}/yr, "
          f"duration: {data['years']:.1f} yr")

    # Spectrum with fine peak detection (min_distance=2 for harmonic-level)
    # Low prominence (0.005) to catch individual harmonics, not just major lobes
    spec = run_spectrum(data['close'], data['fs'], omega_max=omega_max,
                        peak_min_distance=2, peak_prominence_frac=0.005)
    print(f"  Peaks detected: {spec['n_peaks']}")
    print(f"  Envelope upper: k={spec['envelope_upper_k']:.2f}, "
          f"R2={spec['envelope_upper_r2']:.3f}")

    # Line spacing
    spacing = measure_line_spacing(spec['peak_freqs'], omega_max=12.5)
    h_sp = spacing.get('harmonic_spacing', np.nan)
    h_sp_str = f"{h_sp:.4f}" if not np.isnan(h_sp) else "N/A"
    print(f"  Harmonic spacing: {h_sp_str} rad/yr "
          f"(Hurst: {HURST_SPACING:.4f}, delta={abs(h_sp-HURST_SPACING):.4f})")
    print(f"  Raw mean spacing: {spacing['mean_spacing']:.4f} rad/yr")
    print(f"  Lines found (< 12.5 r/y): {spacing['n_lines']}")

    # Harmonic fit
    harmonic = compute_harmonic_fit(spec['peak_freqs'], spacing=HURST_SPACING, max_N=34)
    print(f"  Harmonic coverage: {harmonic['coverage']:.1%} of N=1..34")
    print(f"  Harmonic fit R2: {harmonic['r_squared']:.3f}")
    if harmonic['rms_error'] is not np.nan:
        print(f"  RMS error: {harmonic['rms_error']:.4f} rad/yr")

    # Similarity score
    score = hurst_similarity_score(spec, spacing, harmonic)
    print(f"  --- HURST SIMILARITY SCORE: {score['total_score']:.3f} ---")
    print(f"      Spacing: {score['spacing_score']:.2f}  Envelope: {score['envelope_score']:.2f}  "
          f"Coverage: {score['coverage_score']:.2f}  Fit: {score['fit_score']:.2f}")

    return {
        'data': data,
        'spec': spec,
        'spacing': spacing,
        'harmonic': harmonic,
        'score': score,
        'label': label,
    }


def main():
    print("=" * 70)
    print("PHASE 6B: MODERN WEEKLY TRANSFER TEST")
    print("Does Hurst's w_n = 0.3676*N persist in modern data?")
    print("=" * 70)

    all_results = {}

    # Run DJIA across 3 eras
    for era in ERAS:
        key = f"djia_{era['short']}"
        all_results[key] = analyze_era(
            'djia', 'weekly', era['start'], era['end'],
            f"DJIA — {era['label']}"
        )
        all_results[key]['era'] = era

    # Run SPX across 2 eras (SPX starts ~1928)
    spx_eras = [
        {'label': 'SPX Hurst-overlap (1928-1965)', 'start': '1928-01-01', 'end': '1965-05-21',
         'color': '#9467bd', 'short': 'SPX-Hurst'},
        {'label': 'SPX Modern (1985-2025)',         'start': '1985-01-01', 'end': '2026-01-31',
         'color': '#d62728', 'short': 'SPX-Modern'},
    ]
    for era in spx_eras:
        key = f"spx_{era['short']}"
        all_results[key] = analyze_era(
            'spx', 'weekly', era['start'], era['end'],
            era['label']
        )
        all_results[key]['era'] = era

    # =========================================================================
    # FIGURE 1: Spectra comparison (3 DJIA eras)
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Phase 6B: Fourier-Lanczos Spectra — DJIA Across 3 Eras',
                 fontsize=13, fontweight='bold')

    for ax, era in zip(axes, ERAS):
        key = f"djia_{era['short']}"
        r = all_results[key]
        spec = r['spec']

        ax.plot(spec['omega_sub'], spec['amp_sub'], '-', color=era['color'],
                linewidth=0.6, alpha=0.8)
        ax.plot(spec['peak_freqs'], spec['peak_amps'], 'v', color='black',
                markersize=3, alpha=0.6)

        # 1/ω envelope
        omega_env = np.linspace(0.5, 12, 200)
        if not np.isnan(spec['envelope_upper_k']):
            ax.plot(omega_env, spec['envelope_upper_k'] / omega_env, '--',
                    color='gray', linewidth=0.8, alpha=0.7)

        # Harmonic grid
        for N in range(1, 35):
            ax.axvline(N * HURST_SPACING, color='red', alpha=0.08, linewidth=0.5)

        score = r['score']['total_score']
        ax.set_ylabel('Amplitude')
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.5)
        h_sp = r['spacing'].get('harmonic_spacing', np.nan)
        sp_str = f"{h_sp:.4f}" if not np.isnan(h_sp) else "N/A"
        ax.set_title(f"{era['label']}  |  {spec['n_peaks']} peaks  |  "
                     f"spacing={sp_str} r/y  |  "
                     f"Score={score:.3f}", fontsize=10)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Angular Frequency ω (radians/year)')
    axes[-1].set_xlim(0, 14)
    fig.tight_layout()
    path1 = os.path.join(SCRIPT_DIR, 'phase6B_spectra_3eras.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {path1}")

    # =========================================================================
    # FIGURE 2: Harmonic fit ω_n vs N (all datasets)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Phase 6B: Harmonic Fit ω_n = 0.3676·N — All Datasets',
                 fontsize=13, fontweight='bold')

    plot_keys = [f"djia_{e['short']}" for e in ERAS] + \
                [f"spx_{e['short']}" for e in spx_eras]

    for idx, key in enumerate(plot_keys):
        ax = axes.flat[idx]
        r = all_results[key]
        h = r['harmonic']

        # Reference line
        N_line = np.linspace(0, 34, 100)
        ax.plot(N_line, HURST_SPACING * N_line, '-', color='black', linewidth=0.8)

        # Data points
        if h['N_values']:
            ax.scatter(h['N_values'], h['omega_values'], marker='x', s=40,
                       color=r['era']['color'], linewidths=1.5, zorder=5)

        ax.set_xlim(0, 35)
        ax.set_ylim(0, 13)
        ax.set_xlabel('N')
        ax.set_ylabel('ω (rad/yr)')
        ax.set_title(f"{r['label']}\nR2={h['r_squared']:.3f}, "
                     f"Coverage={h['coverage']:.0%}", fontsize=9)
        ax.grid(True, alpha=0.2)

    # Hide unused subplot
    if len(plot_keys) < 6:
        axes.flat[5].set_visible(False)

    fig.tight_layout()
    path2 = os.path.join(SCRIPT_DIR, 'phase6B_harmonic_fit_all.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path2}")

    # =========================================================================
    # FIGURE 3: Similarity score dashboard
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 6B: Hurst Similarity Scores', fontsize=13, fontweight='bold')

    # Bar chart of total scores
    ax = axes[0]
    labels = [all_results[k]['label'][:25] for k in plot_keys]
    scores = [all_results[k]['score']['total_score'] for k in plot_keys]
    colors = [all_results[k]['era']['color'] for k in plot_keys]
    bars = ax.barh(range(len(labels)), scores, color=colors, alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Hurst Similarity Score (0-1)')
    ax.set_xlim(0, 1)
    ax.axvline(0.8, color='green', linestyle='--', alpha=0.5, label='Strong match')
    ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Partial match')
    ax.legend(fontsize=8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9, fontweight='bold')
    ax.set_title('Total Score')

    # Component breakdown
    ax = axes[1]
    components = ['spacing_score', 'envelope_score', 'coverage_score', 'fit_score']
    comp_labels = ['Spacing (40%)', 'Envelope (20%)', 'Coverage (20%)', 'Fit (20%)']
    x = np.arange(len(plot_keys))
    width = 0.2
    for i, (comp, comp_label) in enumerate(zip(components, comp_labels)):
        vals = [all_results[k]['score'][comp] for k in plot_keys]
        ax.bar(x + i * width, vals, width, label=comp_label, alpha=0.8)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([all_results[k]['era']['short'] for k in plot_keys],
                        fontsize=7, rotation=30)
    ax.set_ylabel('Component Score')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title('Score Components')

    fig.tight_layout()
    path3 = os.path.join(SCRIPT_DIR, 'phase6B_similarity_scores.png')
    fig.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path3}")

    # =========================================================================
    # FIGURE 4: Line spacing comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 6B: Line Spacing Analysis', fontsize=13, fontweight='bold')

    # Spacing distributions
    ax = axes[0]
    for key in [f"djia_{e['short']}" for e in ERAS]:
        r = all_results[key]
        sp = r['spacing']['spacings']
        if len(sp) > 0:
            h_sp = r['spacing'].get('harmonic_spacing', np.nan)
            lbl = f"{r['era']['short']} (fit={h_sp:.4f})" if not np.isnan(h_sp) else r['era']['short']
            ax.hist(sp, bins=20, alpha=0.4, color=r['era']['color'],
                    label=lbl, density=True)
    ax.axvline(HURST_SPACING, color='red', linestyle='--', linewidth=1.5,
               label=f'Hurst 0.3676')
    ax.set_xlabel('Line spacing (rad/yr)')
    ax.set_ylabel('Density')
    ax.set_title('DJIA Spacing Distributions')
    ax.legend(fontsize=8)

    # Harmonic spacing vs era
    ax = axes[1]
    all_keys = plot_keys
    spacings_harm = [all_results[k]['spacing'].get('harmonic_spacing', np.nan) for k in all_keys]
    spacings_std = [all_results[k]['spacing'].get('harmonic_spacing_std', 0) for k in all_keys]
    era_labels = [all_results[k]['era']['short'] for k in all_keys]
    era_colors = [all_results[k]['era']['color'] for k in all_keys]

    ax.barh(range(len(all_keys)), spacings_harm, xerr=spacings_std,
            color=era_colors, alpha=0.7, capsize=3)
    ax.axvline(HURST_SPACING, color='red', linestyle='--', linewidth=1.5)
    ax.set_yticks(range(len(all_keys)))
    ax.set_yticklabels(era_labels, fontsize=8)
    ax.set_xlabel('Harmonic spacing (rad/yr)')
    ax.set_title('Fitted Harmonic Spacing (omega = spacing * N)')
    ax.set_xlim(0.30, 0.45)

    fig.tight_layout()
    path4 = os.path.join(SCRIPT_DIR, 'phase6B_line_spacing.png')
    fig.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path4}")

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 90)
    print("PHASE 6B RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Dataset':<30} {'Peaks':>5} {'HarmSpc':>8} {'Delta':>7} {'Cov%':>5} "
          f"{'H-R2':>6} {'Env-R2':>7} {'Score':>6}")
    print("-" * 90)
    for key in plot_keys:
        r = all_results[key]
        h_sp = r['spacing'].get('harmonic_spacing', np.nan)
        delta = abs(h_sp - HURST_SPACING) if not np.isnan(h_sp) else np.nan
        print(f"{r['label']:<30} {r['spec']['n_peaks']:>5} "
              f"{h_sp:>8.4f} {delta:>7.4f} "
              f"{r['harmonic']['coverage']:>4.0%} "
              f"{r['harmonic']['r_squared']:>6.3f} "
              f"{r['spec']['envelope_upper_r2']:>7.3f} "
              f"{r['score']['total_score']:>6.3f}")
    print("-" * 90)
    print(f"Reference: Hurst spacing = {HURST_SPACING} rad/yr, period = 17.1 yr")
    print()

    # Verdict
    djia_scores = [all_results[f"djia_{e['short']}"]['score']['total_score'] for e in ERAS]
    min_score = min(djia_scores)
    if min_score > 0.7:
        verdict = "STRONG TRANSFER — Hurst structure persists across all eras"
    elif min_score > 0.4:
        verdict = "PARTIAL TRANSFER — Structure present but degraded in some eras"
    else:
        verdict = "WEAK TRANSFER — Structure may be era-specific"
    print(f"VERDICT: {verdict}")
    print(f"  Hurst era score: {djia_scores[0]:.3f}")
    print(f"  Post-Hurst score: {djia_scores[1]:.3f}")
    print(f"  Modern score: {djia_scores[2]:.3f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
