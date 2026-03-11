# -*- coding: utf-8 -*-
"""
Narrowband CMW Analysis on Daily DJIA — Extended AI-2/AI-3 Reproduction

This script applies narrowband CMW filters (one per harmonic) to daily
DJIA data, producing:

  Figure 1: AI-2 style — CMW frequency responses (designed Gaussian vs actual FFT)
  Figure 2: AI-3 style — stacked filter outputs with envelopes
  Figure 3: 3D mesh — frequency × time × amplitude (time-frequency-amplitude surface)

The daily data extends the frequency range to N~80 (vs N~34 for weekly),
giving a much richer picture of the harmonic structure.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing (1970),
           Appendix A, Figures AI-2 and AI-3
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from src.pipeline.derive_nominal_model import derive_nominal_model, load_data
from src.pipeline.comb_bank import (
    design_narrowband_cmw_bank, run_cmw_comb_bank, extract_lines_from_narrowband
)
from src.time_frequency.cmw import cmw_freq_domain

OUT_DIR = os.path.dirname(__file__)


# =============================================================================
# Stage 1: Load daily data and run pipeline for w0
# =============================================================================

def setup_data():
    """Load daily DJIA and derive w0 from weekly pipeline."""
    print("=" * 70)
    print("Loading data and estimating w0...")
    print("=" * 70)

    # Use weekly pipeline for w0 estimation (more robust)
    weekly_result = derive_nominal_model(
        symbol='djia', freq='weekly',
        start='1921-04-29', end='1965-05-21',
        verbose=False
    )
    w0 = weekly_result.w0
    print(f"  w0 from weekly pipeline: {w0:.4f} rad/yr")

    # Load daily data
    daily = load_data('djia', 'daily', '1921-04-29', '1965-05-21')
    print(f"  Daily data: {daily['n_samples']} samples, "
          f"fs={daily['fs']:.1f}, {daily['years']:.1f} years")

    return daily, w0, weekly_result


# =============================================================================
# Stage 2: Design and apply narrowband CMW bank
# =============================================================================

def run_narrowband_analysis(daily, w0, max_N=80, fwhm_factor=0.5):
    """Design and apply narrowband CMW to daily data."""
    print(f"\n{'=' * 70}")
    print(f"Designing narrowband CMW bank (N=2..{max_N}, fwhm_factor={fwhm_factor})...")
    print("=" * 70)

    fs = daily['fs']
    nb_params = design_narrowband_cmw_bank(
        w0=w0, max_N=max_N, fs=fs,
        fwhm_factor=fwhm_factor, omega_min=0.5
    )
    print(f"  {len(nb_params)} filters designed")
    print(f"  Freq range: {nb_params[0]['f0']:.2f} - {nb_params[-1]['f0']:.2f} rad/yr")

    # Apply to log(prices)
    log_prices = np.log(daily['close'])
    print(f"  Applying CMW bank to {len(log_prices)} samples...")
    nb_result = run_cmw_comb_bank(log_prices, fs, nb_params, analytic=True)

    # Extract confirmed lines
    confirmed = extract_lines_from_narrowband(nb_result, w0)
    print(f"  Confirmed harmonics: {len(confirmed)} / {len(nb_params)}")

    return nb_params, nb_result, confirmed


# =============================================================================
# Figure 1: AI-2 style — Frequency Response Plot
# =============================================================================

def plot_frequency_responses(nb_params, daily_fs, w0, confirmed):
    """
    Plot designed CMW frequency responses — extended AI-2 style.

    Shows Gaussian spectral shape of each narrowband CMW filter,
    colored by confirmation status. Unlike Hurst's AI-2 which showed
    23 Ormsby trapezoids in 7-12 rad/yr, this shows ~70 Gaussian
    curves spanning 0.5-30+ rad/yr.
    """
    print("\nGenerating Figure 1: Frequency responses...")

    confirmed_Ns = set(l['N'] for l in confirmed)

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), height_ratios=[2, 1])

    # --- Top panel: Full range frequency responses ---
    ax = axes[0]
    nfft = 65536
    freq_axis = np.fft.fftfreq(nfft, d=1.0 / daily_fs) * (2.0 * np.pi)
    pos_mask = (freq_axis >= 0) & (freq_axis <= 35)
    freq_pos = freq_axis[pos_mask]

    sum_response = np.zeros(np.sum(pos_mask))

    for i, params in enumerate(nb_params):
        f0 = params['f0']
        fwhm = params['fwhm']
        N = params['N']

        if f0 > 35:
            continue

        # Build designed Gaussian response
        result = cmw_freq_domain(f0, fwhm, daily_fs, nfft, analytic=True)
        H = result['H']
        H_pos = np.abs(H[pos_mask])
        # Normalize (analytic has ×2 factor)
        H_pos = H_pos / 2.0

        sum_response += H_pos

        # Color by confirmation
        if N in confirmed_Ns:
            color = cm.viridis(N / 80)
            alpha = 0.6
            lw = 0.8
        else:
            color = 'lightgray'
            alpha = 0.3
            lw = 0.4

        ax.fill_between(freq_pos, H_pos, alpha=alpha * 0.3, color=color)
        ax.plot(freq_pos, H_pos, color=color, linewidth=lw, alpha=alpha)

    # Sum response
    ax.plot(freq_pos, sum_response, 'k-', linewidth=1.5, alpha=0.7, label='Sum response')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Mark w0 grid
    for n in range(1, 100):
        freq = n * w0
        if freq > 35:
            break
        ax.axvline(freq, color='red', alpha=0.08, linewidth=0.3)

    ax.set_xlim(0, 35)
    ax.set_ylim(0, max(1.5, sum_response.max() * 1.1))
    ax.set_ylabel('Amplitude Response')
    ax.set_title(f'Narrowband CMW Filter Bank — {len(nb_params)} filters, '
                 f'FWHM={nb_params[0]["fwhm"]:.3f} rad/yr, '
                 f'w0={w0:.4f} rad/yr', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # Add period scale on top
    ax2 = ax.twiny()
    period_ticks = [20, 10, 5, 3, 2, 1, 0.5]  # years
    period_freqs = [2 * np.pi / p for p in period_ticks]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(period_freqs)
    ax2.set_xticklabels([f'{p}yr' if p >= 1 else f'{p*52:.0f}wk' for p in period_ticks],
                         fontsize=8)
    ax2.set_xlabel('Period', fontsize=9)

    # --- Bottom panel: Zoom into comb region (7-13 rad/yr) like original AI-2 ---
    ax = axes[1]
    zoom_mask = (freq_pos >= 6) & (freq_pos <= 14)
    freq_zoom = freq_pos[zoom_mask]

    for i, params in enumerate(nb_params):
        f0 = params['f0']
        fwhm = params['fwhm']
        N = params['N']

        if f0 < 5.5 or f0 > 14.5:
            continue

        result = cmw_freq_domain(f0, fwhm, daily_fs, nfft, analytic=True)
        H = result['H']
        H_zoom = np.abs(H[pos_mask][zoom_mask]) / 2.0

        if N in confirmed_Ns:
            color = cm.viridis(N / 80)
            alpha = 0.7
        else:
            color = 'lightgray'
            alpha = 0.3

        ax.fill_between(freq_zoom, H_zoom, alpha=alpha * 0.3, color=color)
        ax.plot(freq_zoom, H_zoom, color=color, linewidth=0.8, alpha=alpha)

        # Label every other filter
        if N % 2 == 0 and N in confirmed_Ns:
            ax.text(f0, 1.02, f'N={N}', ha='center', fontsize=5.5,
                    rotation=90, va='bottom')

    # Mark w0 grid in zoom
    for n in range(15, 40):
        freq = n * w0
        if 6 <= freq <= 14:
            ax.axvline(freq, color='red', alpha=0.15, linewidth=0.5)

    ax.set_xlim(6, 14)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('ω (rad/yr)', fontsize=10)
    ax.set_ylabel('Amplitude Response')
    ax.set_title('Zoom: Comb Region (6-14 rad/yr) — cf. Hurst Figure AI-2',
                 fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_daily_cmw_AI2_responses.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# =============================================================================
# Figure 2: AI-3 style — Stacked filter outputs with envelopes
# =============================================================================

def plot_stacked_outputs(nb_result, daily, w0, confirmed,
                         freq_range=(6.0, 13.0), time_window=None):
    """
    Plot stacked CMW filter outputs with envelopes — AI-3 style.

    Each filter's output is plotted at a vertical offset, with ±envelope
    shown in red. Confirmed harmonics are green/blue, others gray.
    """
    print("\nGenerating Figure 2: Stacked outputs (AI-3 style)...")

    confirmed_Ns = set(l['N'] for l in confirmed)
    fs = daily['fs']
    n_samples = len(daily['close'])

    # Time axis in years
    t_yr = np.arange(n_samples) / fs

    # Select time window (default: 1935-1945, ~10 years like Hurst's AI-3)
    if time_window is None:
        # 1935-1945 relative to start (1921.33)
        t_start = 1935 - 1921.33
        t_end = 1945 - 1921.33
    else:
        t_start, t_end = time_window
    t_mask = (t_yr >= t_start) & (t_yr <= t_end)
    t_display = t_yr[t_mask]

    # Select filters in frequency range
    display_filters = []
    for i, spec in enumerate(nb_result['filter_specs']):
        if 'N' not in spec:
            continue
        if freq_range[0] <= spec['f0'] <= freq_range[1]:
            display_filters.append(i)

    if not display_filters:
        print("  No filters in range")
        return

    n_filters = len(display_filters)
    fig, ax = plt.subplots(1, 1, figsize=(18, max(8, n_filters * 0.7)))

    # Normalize each filter to fill its lane (like Hurst's AI-3)
    spacing = 1.0  # unit spacing per filter

    y_labels = []
    y_positions = []

    for row, fi in enumerate(display_filters):
        output = nb_result['filter_outputs'][fi]
        spec = nb_result['filter_specs'][fi]
        N = spec['N']
        confirmed_flag = N in confirmed_Ns
        offset = (n_filters - 1 - row) * spacing

        y_positions.append(offset)
        period_wk = 2 * np.pi / spec['f0'] * 52
        y_labels.append(f"N={N}  {spec['f0']:.1f}r/y  ({period_wk:.0f}wk)")

        # Zero line
        ax.axhline(offset, color='gray', linewidth=0.3, alpha=0.5)

        if output['signal'] is not None:
            sig = np.real(output['signal'][t_mask])
            # Normalize to fill ±0.4 of spacing
            sig_max = np.max(np.abs(sig))
            if sig_max > 0:
                sig_norm = sig / sig_max * 0.4
            else:
                sig_norm = sig
            color = cm.viridis(N / 40) if confirmed_flag else 'gray'
            alpha = 0.8 if confirmed_flag else 0.3

            ax.plot(t_display + 1921.33, sig_norm + offset, color=color,
                    linewidth=0.4, alpha=alpha)

        if output['envelope'] is not None:
            env = output['envelope'][t_mask]
            if sig_max > 0:
                env_norm = env / sig_max * 0.4
            else:
                env_norm = env
            env_color = 'darkred' if confirmed_flag else 'lightgray'
            env_alpha = 0.7 if confirmed_flag else 0.2
            ax.plot(t_display + 1921.33, env_norm + offset, color=env_color,
                    linewidth=0.7, alpha=env_alpha)
            ax.plot(t_display + 1921.33, -env_norm + offset, color=env_color,
                    linewidth=0.7, alpha=env_alpha)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_title(f'Narrowband CMW Filter Outputs — Daily DJIA '
                 f'({freq_range[0]:.0f}-{freq_range[1]:.0f} rad/yr, '
                 f'{n_filters} filters)\n'
                 f'cf. Hurst Figure AI-3 (each filter normalized to fill its lane)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(t_display[0] + 1921.33, t_display[-1] + 1921.33)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_daily_cmw_AI3_stacked.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Also do a wider range version
    print("  Generating wide-range version (1-30 rad/yr)...")
    plot_stacked_outputs_wide(nb_result, daily, w0, confirmed)

    return path


def plot_stacked_outputs_wide(nb_result, daily, w0, confirmed):
    """Wide-range stacked outputs covering 1-30 rad/yr."""
    confirmed_Ns = set(l['N'] for l in confirmed)
    fs = daily['fs']
    n_samples = len(daily['close'])
    t_yr = np.arange(n_samples) / fs

    # 10-year window
    t_start = 1935 - 1921.33
    t_end = 1945 - 1921.33
    t_mask = (t_yr >= t_start) & (t_yr <= t_end)
    t_display = t_yr[t_mask]

    # Every 3rd filter from 1-30 rad/yr for readability
    display_filters = []
    for i, spec in enumerate(nb_result['filter_specs']):
        if 'N' not in spec:
            continue
        if 1.0 <= spec['f0'] <= 30.0 and spec['N'] % 3 == 0:
            display_filters.append(i)

    if not display_filters:
        return

    n_filters = len(display_filters)
    fig, ax = plt.subplots(1, 1, figsize=(18, max(10, n_filters * 0.55)))

    spacing = 1.0  # unit spacing, each filter normalized

    y_labels = []
    y_positions = []

    for row, fi in enumerate(display_filters):
        output = nb_result['filter_outputs'][fi]
        spec = nb_result['filter_specs'][fi]
        N = spec['N']
        confirmed_flag = N in confirmed_Ns
        offset = (n_filters - 1 - row) * spacing

        y_positions.append(offset)
        period_wk = 2 * np.pi / spec['f0'] * 52
        label = f"N={N}  {spec['f0']:.1f}r/y"
        if period_wk >= 52:
            label += f"  ({period_wk/52:.1f}yr)"
        else:
            label += f"  ({period_wk:.0f}wk)"
        y_labels.append(label)

        ax.axhline(offset, color='gray', linewidth=0.2, alpha=0.3)

        if output['envelope'] is not None:
            env = output['envelope'][t_mask]
            env_max = np.max(env)
            if env_max > 0:
                env_norm = env / env_max * 0.4
            else:
                env_norm = env
            color = cm.viridis(N / 80) if confirmed_flag else 'lightgray'
            alpha = 0.7 if confirmed_flag else 0.2
            ax.fill_between(t_display + 1921.33, -env_norm + offset,
                           env_norm + offset,
                           alpha=alpha * 0.3, color=color)
            ax.plot(t_display + 1921.33, env_norm + offset, color=color,
                    linewidth=0.6, alpha=alpha)
            ax.plot(t_display + 1921.33, -env_norm + offset, color=color,
                    linewidth=0.6, alpha=alpha)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=5.5)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_title(f'Narrowband CMW Envelopes — Daily DJIA, Full Range '
                 f'(every 3rd harmonic, N=3..{display_filters[-1]})\n'
                 f'w0={w0:.4f} rad/yr',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(t_display[0] + 1921.33, t_display[-1] + 1921.33)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_daily_cmw_AI3_wide.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# Figure 3: 3D Mesh — Frequency × Time × Amplitude
# =============================================================================

def plot_3d_spectrum(nb_result, daily, w0, confirmed):
    """
    3D mesh/surface plot: frequency × time × envelope amplitude.

    This is a time-frequency-amplitude surface where:
    - X axis: time (years)
    - Y axis: frequency (rad/yr) = harmonic number × w0
    - Z axis: envelope amplitude (instantaneous energy at that frequency)

    Essentially a CMW scalogram rendered as a 3D surface.
    """
    print("\nGenerating Figure 3: 3D spectrum mesh...")

    fs = daily['fs']
    n_samples = len(daily['close'])
    t_yr = np.arange(n_samples) / fs + 1921.33

    # Time window for manageable plot
    t_start_yr = 1930
    t_end_yr = 1960
    t_mask = (t_yr >= t_start_yr) & (t_yr <= t_end_yr)
    t_display = t_yr[t_mask]

    # Downsample time for mesh rendering (every 20th point ≈ monthly)
    step = 20
    t_down = t_display[::step]
    n_time = len(t_down)

    # Collect envelope data for all filters
    specs = nb_result['filter_specs']
    freqs = []
    envelopes = []

    for i, spec in enumerate(specs):
        if 'N' not in spec:
            continue
        f0 = spec['f0']
        if f0 > 30:  # Cap at 30 rad/yr for visibility
            continue

        output = nb_result['filter_outputs'][i]
        if output['envelope'] is not None:
            env = output['envelope'][t_mask][::step]
            freqs.append(f0)
            envelopes.append(env)

    n_freq = len(freqs)
    freqs = np.array(freqs)

    # Build 2D grid
    T, F = np.meshgrid(t_down, freqs)
    Z = np.array(envelopes)

    # Log scale for better visibility of weak high-freq content
    Z_log = np.log10(Z + 1e-6)
    Z_log = np.clip(Z_log, Z_log.max() - 3, Z_log.max())  # 3 decades dynamic range

    # --- 3D Surface Plot ---
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(T, F, Z_log, cmap='inferno',
                           linewidth=0, antialiased=True,
                           rcount=n_freq, ccount=min(200, n_time),
                           alpha=0.9)

    ax.set_xlabel('Year', fontsize=10, labelpad=10)
    ax.set_ylabel('ω (rad/yr)', fontsize=10, labelpad=10)
    ax.set_zlabel('log₁₀(Amplitude)', fontsize=10, labelpad=10)
    ax.set_title(f'3D Time-Frequency Spectrum — Daily DJIA 1930-1960\n'
                 f'{n_freq} narrowband CMW filters, w0={w0:.4f} rad/yr',
                 fontsize=13, fontweight='bold')

    # Add period labels on right side
    ax.view_init(elev=25, azim=-60)
    fig.colorbar(surf, shrink=0.5, aspect=20, label='log₁₀(Amplitude)')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_daily_cmw_3d_surface.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Top-down heatmap (2D projection) ---
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    im = ax.pcolormesh(T, F, Z_log, cmap='inferno', shading='gouraud')
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('ω (rad/yr)', fontsize=11)
    ax.set_title(f'Time-Frequency Heatmap — Daily DJIA\n'
                 f'{n_freq} narrowband CMW filters, w0={w0:.4f} rad/yr',
                 fontsize=13, fontweight='bold')

    # Add period scale on right
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    period_ticks_rad = [0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 15.0, 20.0, 30.0]
    period_labels = []
    for w in period_ticks_rad:
        T_yr = 2 * np.pi / w
        if T_yr >= 1:
            period_labels.append(f'{T_yr:.1f}yr')
        else:
            period_labels.append(f'{T_yr*52:.0f}wk')
    ax2.set_yticks(period_ticks_rad)
    ax2.set_yticklabels(period_labels, fontsize=8)
    ax2.set_ylabel('Period', fontsize=10)

    # Mark confirmed harmonics
    confirmed_Ns = set(l['N'] for l in confirmed)
    for spec in specs:
        if 'N' in spec and spec['N'] in confirmed_Ns and spec['f0'] <= 30:
            ax.axhline(spec['f0'], color='white', alpha=0.1, linewidth=0.3)

    fig.colorbar(im, ax=ax, label='log₁₀(Amplitude)', shrink=0.8)
    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, 'fig_daily_cmw_heatmap.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")

    # --- Wireframe version for clearer structure ---
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot wireframe with stride for clarity
    rstride = max(1, n_freq // 40)
    cstride = max(1, n_time // 100)
    ax.plot_wireframe(T, F, Z_log, rstride=rstride, cstride=cstride,
                      color='steelblue', linewidth=0.3, alpha=0.7)

    # Highlight confirmed harmonics with colored lines
    for i, spec in enumerate(specs):
        if 'N' not in spec or spec['f0'] > 30:
            continue
        N = spec['N']
        if N in confirmed_Ns and N % 3 == 0:
            idx = None
            for j, f in enumerate(freqs):
                if abs(f - spec['f0']) < 0.01:
                    idx = j
                    break
            if idx is not None:
                ax.plot(t_down, np.full_like(t_down, freqs[idx]),
                        Z_log[idx, :], color=cm.viridis(N / 80),
                        linewidth=1.0, alpha=0.8)

    ax.set_xlabel('Year', fontsize=10, labelpad=10)
    ax.set_ylabel('ω (rad/yr)', fontsize=10, labelpad=10)
    ax.set_zlabel('log₁₀(Amplitude)', fontsize=10, labelpad=10)
    ax.set_title(f'3D Wireframe Spectrum — Daily DJIA\n'
                 f'Highlighted: every 3rd confirmed harmonic',
                 fontsize=13, fontweight='bold')
    ax.view_init(elev=20, azim=-55)

    plt.tight_layout()
    path3 = os.path.join(OUT_DIR, 'fig_daily_cmw_3d_wireframe.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path3}")

    return path, path2, path3


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Setup
    daily, w0, weekly_result = setup_data()

    # Run narrowband CMW on daily data
    nb_params, nb_result, confirmed = run_narrowband_analysis(daily, w0, max_N=80)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Confirmed harmonics summary:")
    print(f"{'=' * 70}")
    high = [l for l in confirmed if l['confidence'] == 'high']
    med = [l for l in confirmed if l['confidence'] == 'medium']
    low = [l for l in confirmed if l['confidence'] == 'low']
    print(f"  High confidence: {len(high)}")
    print(f"  Medium: {len(med)}")
    print(f"  Low: {len(low)}")
    print(f"  Total: {len(confirmed)} / {len(nb_params)}")
    max_confirmed_N = max(l['N'] for l in confirmed) if confirmed else 0
    min_period = 2 * np.pi / (max_confirmed_N * w0) * 52 if max_confirmed_N > 0 else 0
    print(f"  Max N: {max_confirmed_N} (period={min_period:.1f} weeks)")

    # Figure 1: Frequency responses
    plot_frequency_responses(nb_params, daily['fs'], w0, confirmed)

    # Figure 2: Stacked outputs (AI-3 style)
    plot_stacked_outputs(nb_result, daily, w0, confirmed)

    # Figure 3: 3D spectrum
    plot_3d_spectrum(nb_result, daily, w0, confirmed)

    print(f"\n{'=' * 70}")
    print("All figures generated!")
    print("=" * 70)
