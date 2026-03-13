# -*- coding: utf-8 -*-
"""
Stage 10: CMW Envelope Analysis — Modulation and Inter-Filter Coupling

Extracts amplitude modulation characteristics from narrowband CMW envelopes:
  1. Modulation spectrum per harmonic (FFT of envelope)
  2. Dominant modulation periods — do they match beat frequencies?
  3. Inter-filter coupling (cross-correlation of envelope pairs)
  4. Modulation index vs N — is AM depth constant across harmonics?

This was designed in nominal_model_pipeline.md but never implemented.

Reference: prd/nominal_model_pipeline.md, Stage 10
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

from src.pipeline.derive_nominal_model import derive_nominal_model, load_data
from src.pipeline.comb_bank import (
    design_narrowband_cmw_bank, run_cmw_comb_bank, extract_lines_from_narrowband
)

OUT_DIR = os.path.dirname(__file__)


def compute_modulation_spectrum(envelope, fs):
    """
    Compute the amplitude modulation spectrum of an envelope signal.

    Returns modulation frequencies (in cycles/year) and their amplitudes.
    """
    # Remove mean to get modulation signal
    env_centered = envelope - np.mean(envelope)

    # Zero-pad for frequency resolution
    nfft = max(2048, 2 ** int(np.ceil(np.log2(len(env_centered) * 4))))
    spectrum = np.abs(np.fft.rfft(env_centered, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)  # cycles/year

    # Convert to periods
    periods = np.zeros_like(freqs)
    periods[1:] = 1.0 / freqs[1:]

    return freqs, periods, spectrum


def find_dominant_modulation_periods(freqs, spectrum, min_period_yr=1.0, max_period_yr=25.0):
    """Find the top modulation periods within a range."""
    valid = (freqs > 0) & (1.0 / freqs >= min_period_yr) & (1.0 / freqs <= max_period_yr)
    if not np.any(valid):
        return [], []

    valid_freqs = freqs[valid]
    valid_spec = spectrum[valid]
    valid_periods = 1.0 / valid_freqs

    # Find peaks in modulation spectrum
    peaks, props = signal.find_peaks(valid_spec, prominence=0.01 * valid_spec.max())

    if len(peaks) == 0:
        # Just take the maximum
        idx = np.argmax(valid_spec)
        return [float(valid_periods[idx])], [float(valid_spec[idx])]

    # Sort by amplitude, take top 5
    sorted_idx = np.argsort(valid_spec[peaks])[::-1][:5]
    top_peaks = peaks[sorted_idx]

    top_periods = [float(valid_periods[p]) for p in top_peaks]
    top_amps = [float(valid_spec[p]) for p in top_peaks]

    return top_periods, top_amps


def compute_coupling_matrix(confirmed, nb_result, fs, max_lag_yr=5.0):
    """
    Compute cross-correlation of envelopes for all confirmed harmonic pairs.

    Returns: correlation matrix, lag matrix (in years)
    """
    # Get envelope data for confirmed harmonics
    confirmed_Ns = sorted(set(l['N'] for l in confirmed))
    n_conf = len(confirmed_Ns)

    # Map N to filter index
    N_to_idx = {}
    for i, spec in enumerate(nb_result['filter_specs']):
        if 'N' in spec and spec['N'] in confirmed_Ns:
            N_to_idx[spec['N']] = i

    # Extract envelopes, trimming edges
    envelopes = {}
    for N in confirmed_Ns:
        if N not in N_to_idx:
            continue
        idx = N_to_idx[N]
        env = nb_result['filter_outputs'][idx]['envelope']
        if env is not None:
            n = len(env)
            trim = int(n * 0.1)
            envelopes[N] = env[trim:n-trim] if trim > 0 else env

    # Compute correlation and lag matrices
    N_list = sorted(envelopes.keys())
    n = len(N_list)
    corr_matrix = np.zeros((n, n))
    lag_matrix = np.zeros((n, n))

    max_lag_samples = int(max_lag_yr * fs)

    for i, Ni in enumerate(N_list):
        for j, Nj in enumerate(N_list):
            if i == j:
                corr_matrix[i, j] = 1.0
                lag_matrix[i, j] = 0.0
                continue

            ei = envelopes[Ni]
            ej = envelopes[Nj]

            # Ensure same length
            min_len = min(len(ei), len(ej))
            ei = ei[:min_len]
            ej = ej[:min_len]

            # Normalize
            ei_norm = (ei - np.mean(ei)) / (np.std(ei) + 1e-10)
            ej_norm = (ej - np.mean(ej)) / (np.std(ej) + 1e-10)

            # Cross-correlation
            corr = np.correlate(ei_norm, ej_norm, mode='full') / min_len
            lags = np.arange(-min_len + 1, min_len)

            # Restrict to max lag
            lag_mask = np.abs(lags) <= max_lag_samples
            corr_restricted = corr[lag_mask]
            lags_restricted = lags[lag_mask]

            # Find peak correlation
            peak_idx = np.argmax(np.abs(corr_restricted))
            corr_matrix[i, j] = float(corr_restricted[peak_idx])
            lag_matrix[i, j] = float(lags_restricted[peak_idx]) / fs  # convert to years

    return N_list, corr_matrix, lag_matrix


def compute_modulation_index(confirmed, nb_result):
    """
    Compute modulation index for each harmonic.

    Modulation index = (env_max - env_min) / (env_max + env_min)
    Higher values = deeper AM modulation.
    """
    results = []
    for l in confirmed:
        N = l['N']
        # Find filter index
        for i, spec in enumerate(nb_result['filter_specs']):
            if 'N' in spec and spec['N'] == N:
                env = nb_result['filter_outputs'][i]['envelope']
                if env is not None:
                    n = len(env)
                    trim = int(n * 0.1)
                    env_trimmed = env[trim:n-trim] if trim > 0 else env

                    env_max = np.max(env_trimmed)
                    env_min = np.min(env_trimmed)
                    if env_max + env_min > 0:
                        mod_index = (env_max - env_min) / (env_max + env_min)
                    else:
                        mod_index = 0.0

                    # Also compute CV of envelope (alternative measure)
                    cv = np.std(env_trimmed) / np.mean(env_trimmed) if np.mean(env_trimmed) > 0 else 0

                    results.append({
                        'N': N,
                        'frequency': l['frequency'],
                        'mod_index': float(mod_index),
                        'env_cv': float(cv),
                        'env_max': float(env_max),
                        'env_min': float(env_min),
                    })
                break

    return results


def main():
    print("=" * 70)
    print("STAGE 10: CMW Envelope Analysis")
    print("=" * 70)

    # --- Setup ---
    print("\nRunning weekly pipeline for w0...")
    weekly_result = derive_nominal_model(
        symbol='djia', freq='weekly',
        start='1921-04-29', end='1965-05-21',
        verbose=False
    )
    w0 = weekly_result.w0
    expected_beat_period = 2 * np.pi / w0  # ~17.1 years

    print(f"  w0 = {w0:.4f} rad/yr")
    print(f"  Expected beat period (N vs N+1): {expected_beat_period:.1f} years")

    print("\nLoading daily DJIA 1921-1965...")
    daily = load_data('djia', 'daily', '1921-04-29', '1965-05-21')
    fs = daily['fs']
    log_prices = np.log(daily['close'])

    print(f"\nRunning narrowband CMW (N=2..80)...")
    nb_params = design_narrowband_cmw_bank(w0=w0, max_N=80, fs=fs, fwhm_factor=0.5, omega_min=0.5)
    nb_result = run_cmw_comb_bank(log_prices, fs, nb_params, analytic=True)
    confirmed = extract_lines_from_narrowband(nb_result, w0)
    print(f"  Confirmed: {len(confirmed)}/{len(nb_params)}")

    # =====================================================================
    # Part 1: Modulation spectrum per harmonic
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 1: Modulation Spectrum Analysis")
    print("=" * 70)

    modulation_results = []
    beat_period_hits = 0
    total_checked = 0

    for l in confirmed:
        N = l['N']
        for i, spec in enumerate(nb_result['filter_specs']):
            if 'N' in spec and spec['N'] == N:
                env = nb_result['filter_outputs'][i]['envelope']
                if env is not None:
                    freqs, periods, spectrum = compute_modulation_spectrum(env, fs)
                    top_periods, top_amps = find_dominant_modulation_periods(freqs, spectrum)

                    # Check if any dominant period is near the expected beat
                    near_beat = any(abs(p - expected_beat_period) / expected_beat_period < 0.20
                                   for p in top_periods)
                    total_checked += 1
                    if near_beat:
                        beat_period_hits += 1

                    modulation_results.append({
                        'N': N,
                        'top_periods': top_periods[:3],
                        'near_beat': near_beat,
                    })
                break

    print(f"\nHarmonics with 17.1yr modulation: {beat_period_hits}/{total_checked} "
          f"({beat_period_hits/total_checked*100:.0f}%)")

    # Aggregate dominant periods across all harmonics
    all_periods = []
    for mr in modulation_results:
        all_periods.extend(mr['top_periods'])

    if all_periods:
        print(f"\nTop modulation periods (aggregated):")
        period_bins = np.linspace(1, 25, 50)
        hist, edges = np.histogram(all_periods, bins=period_bins)
        top_bin_idx = np.argsort(hist)[::-1][:5]
        for idx in top_bin_idx:
            if hist[idx] > 0:
                center = (edges[idx] + edges[idx + 1]) / 2
                print(f"  {center:.1f} yr: {hist[idx]} occurrences")

    # =====================================================================
    # Part 2: Inter-filter coupling
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 2: Inter-Filter Coupling")
    print("=" * 70)

    # Use a subset for computational tractability (every 3rd harmonic)
    subset = [l for l in confirmed if l['N'] % 3 == 0 or l['N'] <= 10]
    print(f"\nComputing coupling matrix for {len(subset)} harmonics...")

    N_list, corr_matrix, lag_matrix = compute_coupling_matrix(subset, nb_result, fs)

    # Significant coupling pairs
    n = len(N_list)
    sig_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_matrix[i, j]) > 0.3:
                sig_pairs.append({
                    'N_i': N_list[i],
                    'N_j': N_list[j],
                    'corr': corr_matrix[i, j],
                    'lag_yr': lag_matrix[i, j],
                })

    print(f"\nSignificant coupling pairs (|r| > 0.3): {len(sig_pairs)}")
    if sig_pairs:
        # Sort by correlation strength
        sig_pairs.sort(key=lambda x: abs(x['corr']), reverse=True)
        print(f"{'N_i':>4} {'N_j':>4} {'r':>8} {'Lag(yr)':>8} {'Group':>12}")
        print("-" * 40)
        for sp in sig_pairs[:20]:
            # Same group?
            group_diff = abs(sp['N_j'] - sp['N_i'])
            group = 'adjacent' if group_diff <= 3 else ('near' if group_diff <= 10 else 'distant')
            print(f"{sp['N_i']:>4} {sp['N_j']:>4} {sp['corr']:>8.3f} "
                  f"{sp['lag_yr']:>8.2f} {group:>12}")

    # =====================================================================
    # Part 3: Modulation index vs N
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 3: Modulation Index vs N")
    print("=" * 70)

    mod_results = compute_modulation_index(confirmed, nb_result)

    if mod_results:
        Ns = [m['N'] for m in mod_results]
        mod_idxs = [m['mod_index'] for m in mod_results]
        env_cvs = [m['env_cv'] for m in mod_results]

        print(f"\nModulation index statistics:")
        print(f"  Mean: {np.mean(mod_idxs):.3f}")
        print(f"  Std:  {np.std(mod_idxs):.3f}")
        print(f"  CV:   {np.std(mod_idxs)/np.mean(mod_idxs):.3f}")

        print(f"\nEnvelope CV statistics:")
        print(f"  Mean: {np.mean(env_cvs):.3f}")
        print(f"  Std:  {np.std(env_cvs):.3f}")

        # Is modulation index constant with N? (would support equal-modulation hypothesis)
        from scipy import stats as scipy_stats
        slope, intercept, r_value, p_value, _ = scipy_stats.linregress(Ns, mod_idxs)
        print(f"\nModulation index vs N regression:")
        print(f"  slope = {slope:.5f}, R2 = {r_value**2:.4f}, p = {p_value:.4f}")
        if p_value > 0.05:
            print(f"  -> Modulation depth is CONSTANT across N (supports equal-AM hypothesis)")
        else:
            print(f"  -> Modulation depth VARIES with N (trend: {'increasing' if slope > 0 else 'decreasing'})")

    # =====================================================================
    # Figures
    # =====================================================================
    print("\nGenerating figures...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Top modulation periods histogram
    ax = axes[0, 0]
    if all_periods:
        ax.hist(all_periods, bins=np.linspace(1, 25, 50), color='steelblue',
                edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.axvline(expected_beat_period, color='red', linewidth=2, linestyle='--',
                   label=f'Expected beat: {expected_beat_period:.1f} yr')
        ax.axvline(expected_beat_period / 2, color='orange', linewidth=1.5, linestyle='--',
                   label=f'Half-beat: {expected_beat_period/2:.1f} yr')
    ax.set_xlabel('Modulation Period (years)')
    ax.set_ylabel('Count')
    ax.set_title('Dominant Modulation Periods Across All Harmonics')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Coupling matrix heatmap
    ax = axes[0, 1]
    if len(N_list) > 1:
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(N_list)))
        ax.set_yticks(range(len(N_list)))
        tick_labels = [str(N) for N in N_list]
        ax.set_xticklabels(tick_labels, fontsize=5, rotation=90)
        ax.set_yticklabels(tick_labels, fontsize=5)
        ax.set_xlabel('Harmonic N')
        ax.set_ylabel('Harmonic N')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
    ax.set_title('Envelope Cross-Correlation Matrix')

    # Panel 3: Lead/lag structure
    ax = axes[1, 0]
    if sig_pairs:
        for sp in sig_pairs:
            color = 'green' if sp['corr'] > 0 else 'red'
            ax.scatter(sp['N_i'], sp['lag_yr'], c=color, s=abs(sp['corr']) * 100,
                       alpha=0.6, edgecolors='black', linewidth=0.3)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_xlabel('Lower Harmonic N (in pair)')
        ax.set_ylabel('Lag (years, + = leads)')
        ax.set_title(f'Lead/Lag Structure ({len(sig_pairs)} significant pairs)')
    ax.grid(True, alpha=0.3)

    # Panel 4: Modulation index vs N
    ax = axes[1, 1]
    if mod_results:
        ax.scatter(Ns, mod_idxs, c='steelblue', s=30, alpha=0.7, edgecolors='black', linewidth=0.3)
        # Fit line
        if len(Ns) > 3:
            z = np.polyfit(Ns, mod_idxs, 1)
            x_fit = np.linspace(min(Ns), max(Ns), 100)
            ax.plot(x_fit, np.polyval(z, x_fit), 'r--', linewidth=1,
                    label=f'slope={z[0]:.5f}')
        ax.axhline(np.mean(mod_idxs), color='green', linestyle=':', linewidth=1,
                   label=f'mean={np.mean(mod_idxs):.3f}')
        ax.set_xlabel('Harmonic N')
        ax.set_ylabel('Modulation Index')
        ax.set_title('AM Modulation Depth vs Harmonic Number')
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Stage 10: CMW Envelope Analysis — Modulation & Coupling',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_stage10_envelope_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {fig_path}")

    print("\n" + "=" * 70)
    print("DONE -- Stage 10 CMW Envelope Analysis Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
