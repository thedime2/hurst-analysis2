# -*- coding: utf-8 -*-
"""
'Meaningless Frequencies' Hypothesis Test — PRD Hypothesis 3

In Hurst's Appendix A, some comb filter outputs are classified as
"meaningless" based on low amplitude. Three hypotheses:

  H1. Anti-resonance: meaningless filters sit between harmonics (N*w0)
  H2. Destructive interference: two ~equal lines beat and cancel
  H3. Filter mismatch: Ormsby passband (0.2 r/y) vs harmonic spacing (0.37 r/y)

This script tests all three systematically using the 23-filter comb bank
on DJIA 1921-1965.

Reference: prd/hurst_spectral_analysis_prd.md, Hypothesis 3
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from src.pipeline.derive_nominal_model import load_data
from src.filters import design_hurst_comb_bank, create_filter_kernels, apply_filter_bank

OUT_DIR = os.path.dirname(__file__)

# Hurst's fundamental spacing
W0 = 0.3676  # rad/yr
FS = 52


def design_comb_bank():
    """Design the standard 23-filter comb bank matching Hurst's Appendix A."""
    specs = design_hurst_comb_bank(
        n_filters=23, w1_start=7.2, w_step=0.2,
        passband_width=0.2, skirt_width=0.3
    )
    return specs


def compute_filter_metrics(specs, log_prices):
    """Apply comb bank and compute per-filter metrics."""
    kernels = create_filter_kernels(specs, fs=FS)
    bank_result = apply_filter_bank(log_prices, kernels, fs=FS)
    outputs = bank_result['filter_outputs']

    metrics = []
    for i, (spec, output) in enumerate(zip(specs, outputs)):
        # Center frequency
        fc = (spec['f2'] + spec['f3']) / 2.0
        f1, f4 = spec['f1'], spec['f4']

        # Mean envelope amplitude (trim edges)
        env = output['envelope']
        if env is not None:
            n = len(env)
            trim = int(n * 0.1)
            mean_env = float(np.mean(env[trim:n-trim])) if trim > 0 else float(np.mean(env))
        else:
            mean_env = 0.0

        # H1: Distance to nearest harmonic
        harmonics = np.arange(1, 100) * W0
        dists = np.abs(harmonics - fc)
        nearest_N = np.argmin(dists) + 1
        dist_to_harmonic = float(dists[np.argmin(dists)])
        # Normalize by w0/2 (half-spacing)
        dist_normalized = dist_to_harmonic / (W0 / 2)

        # H2: Count harmonics within passband [f1, f4]
        in_band = harmonics[(harmonics >= f1) & (harmonics <= f4)]
        n_harmonics_in_band = len(in_band)

        # Beat ratio: if 2+ harmonics, ratio of closest pair amplitudes
        # (approximate: use 1/N amplitude model)
        beat_ratio = 0.0
        if n_harmonics_in_band >= 2:
            Ns_in_band = np.round(in_band / W0).astype(int)
            amps_model = 1.0 / Ns_in_band  # 1/N ~ 1/w model
            if len(amps_model) >= 2:
                sorted_amps = np.sort(amps_model)[::-1]
                beat_ratio = sorted_amps[1] / sorted_amps[0]  # ratio of 2nd/1st

        # H3: Alignment score — how well does filter center on a harmonic?
        # 0 = centered on harmonic, 1 = centered between harmonics
        fractional_N = fc / W0
        alignment = abs(fractional_N - round(fractional_N))  # 0-0.5
        alignment_score = alignment * 2  # normalize to 0-1

        metrics.append({
            'filter_idx': i,
            'fc': fc,
            'f1': f1,
            'f4': f4,
            'mean_env': mean_env,
            'nearest_N': nearest_N,
            'dist_to_harmonic': dist_to_harmonic,
            'dist_normalized': dist_normalized,
            'n_harmonics': n_harmonics_in_band,
            'beat_ratio': beat_ratio,
            'alignment_score': alignment_score,
        })

    return metrics


def classify_meaningful(metrics, threshold_frac=0.30):
    """Classify filters as meaningful/meaningless using amplitude threshold."""
    max_env = max(m['mean_env'] for m in metrics)
    threshold = threshold_frac * max_env

    for m in metrics:
        m['meaningful'] = m['mean_env'] >= threshold
        m['env_relative'] = m['mean_env'] / max_env

    n_meaningful = sum(1 for m in metrics if m['meaningful'])
    n_meaningless = sum(1 for m in metrics if not m['meaningful'])
    return metrics, n_meaningful, n_meaningless, threshold


def run_statistical_tests(metrics):
    """Run statistical tests on the three hypotheses."""
    meaningful = [m for m in metrics if m['meaningful']]
    meaningless = [m for m in metrics if not m['meaningful']]

    results = {}

    # H1: Distance to nearest harmonic
    dist_m = [m['dist_normalized'] for m in meaningful]
    dist_ml = [m['dist_normalized'] for m in meaningless]
    if len(dist_ml) >= 2 and len(dist_m) >= 2:
        t_stat, p_val = stats.ttest_ind(dist_ml, dist_m, alternative='greater')
        results['H1_anti_resonance'] = {
            'meaningful_mean': np.mean(dist_m),
            'meaningless_mean': np.mean(dist_ml),
            't_stat': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'conclusion': 'Meaningless filters are FARTHER from harmonics' if p_val < 0.05
                         else 'No significant difference in harmonic distance'
        }
    else:
        results['H1_anti_resonance'] = {'conclusion': 'Not enough samples', 'significant': False}

    # H2: Beat ratio (higher = more destructive interference)
    beat_m = [m['beat_ratio'] for m in meaningful if m['n_harmonics'] >= 2]
    beat_ml = [m['beat_ratio'] for m in meaningless if m['n_harmonics'] >= 2]
    if len(beat_ml) >= 2 and len(beat_m) >= 2:
        t_stat, p_val = stats.ttest_ind(beat_ml, beat_m, alternative='greater')
        results['H2_beating'] = {
            'meaningful_mean': np.mean(beat_m),
            'meaningless_mean': np.mean(beat_ml),
            't_stat': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'conclusion': 'Meaningless filters have MORE equal-amplitude pairs (stronger beating)'
                         if p_val < 0.05 else 'No significant difference in beat ratio'
        }
    else:
        results['H2_beating'] = {'conclusion': 'Not enough multi-harmonic filters', 'significant': False}

    # H3: Alignment score (higher = worse alignment)
    align_m = [m['alignment_score'] for m in meaningful]
    align_ml = [m['alignment_score'] for m in meaningless]
    if len(align_ml) >= 2 and len(align_m) >= 2:
        t_stat, p_val = stats.ttest_ind(align_ml, align_m, alternative='greater')
        results['H3_mismatch'] = {
            'meaningful_mean': np.mean(align_m),
            'meaningless_mean': np.mean(align_ml),
            't_stat': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'conclusion': 'Meaningless filters are LESS aligned with harmonics'
                         if p_val < 0.05 else 'No significant difference in alignment'
        }
    else:
        results['H3_mismatch'] = {'conclusion': 'Not enough samples', 'significant': False}

    # Correlation analysis: amplitude vs each metric
    envs = [m['env_relative'] for m in metrics]
    dists = [m['dist_normalized'] for m in metrics]
    aligns = [m['alignment_score'] for m in metrics]

    r_dist, p_dist = stats.pearsonr(envs, dists)
    r_align, p_align = stats.pearsonr(envs, aligns)

    results['correlations'] = {
        'env_vs_dist': {'r': r_dist, 'p': p_dist},
        'env_vs_alignment': {'r': r_align, 'p': p_align},
    }

    return results


def plot_results(metrics, test_results, threshold):
    """Generate 3-panel figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    meaningful = [m for m in metrics if m['meaningful']]
    meaningless = [m for m in metrics if not m['meaningful']]

    # Panel 1: Comb filter amplitudes with classification
    ax = axes[0]
    fcs = [m['fc'] for m in metrics]
    envs = [m['env_relative'] for m in metrics]
    colors = ['green' if m['meaningful'] else 'red' for m in metrics]

    ax.bar(range(len(fcs)), envs, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(0.30, color='orange', linestyle='--', linewidth=1.5,
               label='30% threshold')
    ax.set_xticks(range(0, len(fcs), 2))
    ax.set_xticklabels([f'{fc:.1f}' for fc in fcs[::2]], fontsize=7, rotation=45)
    ax.set_xlabel('Center Frequency (rad/yr)')
    ax.set_ylabel('Relative Envelope Amplitude')
    ax.set_title('H1: Filter Classification\n(green=meaningful, red=meaningless)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Distance to nearest harmonic vs amplitude
    ax = axes[1]
    for m in meaningful:
        ax.scatter(m['dist_normalized'], m['env_relative'], c='green', s=50,
                   edgecolors='black', linewidth=0.5, zorder=3)
    for m in meaningless:
        ax.scatter(m['dist_normalized'], m['env_relative'], c='red', s=50,
                   edgecolors='black', linewidth=0.5, zorder=3)

    # Fit line
    dists = [m['dist_normalized'] for m in metrics]
    envs_all = [m['env_relative'] for m in metrics]
    if len(dists) > 2:
        z = np.polyfit(dists, envs_all, 1)
        x_fit = np.linspace(0, max(dists), 100)
        ax.plot(x_fit, np.polyval(z, x_fit), 'k--', linewidth=1, alpha=0.5)

    r_val = test_results['correlations']['env_vs_dist']['r']
    p_val = test_results['correlations']['env_vs_dist']['p']
    ax.set_xlabel('Distance to Nearest Harmonic (× w0/2)')
    ax.set_ylabel('Relative Amplitude')
    ax.set_title(f'H1: Anti-Resonance\nr={r_val:.3f}, p={p_val:.3f}')
    ax.grid(True, alpha=0.3)

    # Panel 3: Alignment score vs amplitude
    ax = axes[2]
    for m in meaningful:
        ax.scatter(m['alignment_score'], m['env_relative'], c='green', s=50,
                   edgecolors='black', linewidth=0.5, zorder=3)
    for m in meaningless:
        ax.scatter(m['alignment_score'], m['env_relative'], c='red', s=50,
                   edgecolors='black', linewidth=0.5, zorder=3)

    aligns = [m['alignment_score'] for m in metrics]
    if len(aligns) > 2:
        z = np.polyfit(aligns, envs_all, 1)
        x_fit = np.linspace(0, max(aligns), 100)
        ax.plot(x_fit, np.polyval(z, x_fit), 'k--', linewidth=1, alpha=0.5)

    r_val = test_results['correlations']['env_vs_alignment']['r']
    p_val = test_results['correlations']['env_vs_alignment']['p']
    ax.set_xlabel('Alignment Score (0=on harmonic, 1=between)')
    ax.set_ylabel('Relative Amplitude')
    ax.set_title(f'H3: Filter-Harmonic Mismatch\nr={r_val:.3f}, p={p_val:.3f}')
    ax.grid(True, alpha=0.3)

    plt.suptitle("'Meaningless Frequencies' Hypothesis Testing — 23-Filter Comb Bank",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_meaningless_frequencies.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")


def main():
    print("=" * 70)
    print("MEANINGLESS FREQUENCIES HYPOTHESIS TEST")
    print("=" * 70)

    # Load data
    print("\nLoading DJIA 1921-1965...")
    data = load_data('djia', 'weekly', '1921-04-29', '1965-05-21')
    log_prices = np.log(data['close'])
    print(f"  {data['n_samples']} samples, fs={data['fs']:.1f}")

    # Design comb bank
    print("\nDesigning 23-filter comb bank...")
    specs = design_comb_bank()
    print(f"  {len(specs)} filters, fc range: "
          f"{(specs[0]['f2']+specs[0]['f3'])/2:.2f} - "
          f"{(specs[-1]['f2']+specs[-1]['f3'])/2:.2f} rad/yr")

    # Compute metrics
    print("\nApplying filters and computing metrics...")
    metrics = compute_filter_metrics(specs, log_prices)

    # Classify
    metrics, n_m, n_ml, threshold = classify_meaningful(metrics)
    print(f"\n  Meaningful: {n_m}, Meaningless: {n_ml}")
    print(f"  Threshold: 30% of max envelope")

    # Print detailed table
    print(f"\n{'Idx':>3} {'fc':>6} {'Env%':>6} {'Class':>10} {'Dist':>6} {'N_harm':>6} "
          f"{'Beat':>6} {'Align':>6}")
    print("-" * 60)
    for m in metrics:
        cls = 'MEANING' if m['meaningful'] else 'LESS'
        print(f"{m['filter_idx']:>3} {m['fc']:>6.2f} {m['env_relative']:>6.1%} "
              f"{cls:>10} {m['dist_normalized']:>6.2f} {m['n_harmonics']:>6} "
              f"{m['beat_ratio']:>6.2f} {m['alignment_score']:>6.2f}")

    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    test_results = run_statistical_tests(metrics)

    for hyp_name, result in test_results.items():
        if hyp_name == 'correlations':
            continue
        print(f"\n--- {hyp_name} ---")
        for k, v in result.items():
            print(f"  {k}: {v}")

    print("\n--- Correlations ---")
    for name, vals in test_results['correlations'].items():
        print(f"  {name}: r={vals['r']:.3f}, p={vals['p']:.3f}")

    # Plot
    plot_results(metrics, test_results, threshold)

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    n_significant = sum(1 for k, v in test_results.items()
                        if k != 'correlations' and v.get('significant', False))
    print(f"Hypotheses supported: {n_significant}/3")
    for hyp_name, result in test_results.items():
        if hyp_name == 'correlations':
            continue
        sig = 'SUPPORTED' if result.get('significant', False) else 'NOT SUPPORTED'
        print(f"  {hyp_name}: {sig} — {result['conclusion']}")


if __name__ == '__main__':
    main()
