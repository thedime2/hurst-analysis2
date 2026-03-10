#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure: Six Hurst Filters Overlaid on Weekly + Daily Lanczos Spectra

Shows the Fourier-Lanczos amplitude spectrum for both weekly and daily DJIA
data, with the 6 page-152 filter passbands overlaid as shaded regions.
This visualizes how the filter bank partitions the frequency axis and which
spectral peaks each filter captures.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing, p.152
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.spectral.lanczos import lanczos_spectrum
from src.spectral.peak_detection import find_spectral_peaks, find_spectral_troughs
from src.spectral.envelopes import fit_upper_envelope, fit_lower_envelope, envelope_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
TWOPI = 2 * np.pi

# Hurst's analysis window
DATE_START = '1921-04-29'
DATE_END = '1965-05-21'

# Page 152 filter specifications (rad/yr)
FILTER_SPECS = [
    {'label': 'LP-1: Trend', 'type': 'lp', 'f_pass': 0.85, 'f_stop': 1.25,
     'color': '#2196F3', 'alpha': 0.15},
    {'label': 'BP-2: ~3.8yr', 'type': 'bp', 'f1': 0.85, 'f2': 1.25, 'f3': 2.05, 'f4': 2.45,
     'color': '#4CAF50', 'alpha': 0.15},
    {'label': 'BP-3: ~1.3yr', 'type': 'bp', 'f1': 3.20, 'f2': 3.55, 'f3': 6.35, 'f4': 6.70,
     'color': '#FF9800', 'alpha': 0.15},
    {'label': 'BP-4: ~0.7yr', 'type': 'bp', 'f1': 7.25, 'f2': 7.55, 'f3': 9.55, 'f4': 9.85,
     'color': '#F44336', 'alpha': 0.15},
    {'label': 'BP-5: ~20wk', 'type': 'bp', 'f1': 13.65, 'f2': 13.95, 'f3': 19.35, 'f4': 19.65,
     'color': '#9C27B0', 'alpha': 0.15},
    {'label': 'BP-6: ~9wk', 'type': 'bp', 'f1': 28.45, 'f2': 28.75, 'f3': 35.95, 'f4': 36.25,
     'color': '#795548', 'alpha': 0.15},
]


def load_weekly_data():
    csv_path = os.path.join(BASE_DIR, 'data/raw/^dji_w.csv')
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df_h = df[df.Date.between(DATE_START, DATE_END)].copy()
    return df_h.Close.values, 52


def load_daily_data():
    csv_path = os.path.join(BASE_DIR, 'data/raw/^dji_d.csv')
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df_h = df[df.Date.between(DATE_START, DATE_END)].copy()
    close = df_h.Close.values
    dates = pd.to_datetime(df_h.Date.values)
    total_years = (dates[-1] - dates[0]).days / 365.25
    fs_daily = len(close) / total_years
    return close, fs_daily


def compute_lanczos(close, fs):
    w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(close, 1, fs)
    omega_yr = w * fs
    return omega_yr, amp


def draw_filter_bands(ax, specs, ymin, ymax):
    """Draw filter passbands and skirt regions on spectrum plot."""
    for spec in specs:
        c = spec['color']
        a = spec['alpha']
        if spec['type'] == 'lp':
            # LP: flat passband from 0 to f_pass, skirt to f_stop
            ax.axvspan(0, spec['f_pass'], color=c, alpha=a, zorder=0)
            ax.axvspan(spec['f_pass'], spec['f_stop'], color=c, alpha=a * 0.5,
                       zorder=0, hatch='///')
        else:
            # BP: skirt f1-f2, passband f2-f3, skirt f3-f4
            ax.axvspan(spec['f1'], spec['f2'], color=c, alpha=a * 0.5,
                       zorder=0, hatch='///')
            ax.axvspan(spec['f2'], spec['f3'], color=c, alpha=a, zorder=0)
            ax.axvspan(spec['f3'], spec['f4'], color=c, alpha=a * 0.5,
                       zorder=0, hatch='///')


def plot_spectrum_with_filters(ax, omega_yr, amp, fs, title, max_freq=40):
    """Plot Lanczos spectrum with filter bands overlaid."""
    # Mask to display range
    mask = (omega_yr > 0.1) & (omega_yr <= max_freq)
    w_plot = omega_yr[mask]
    a_plot = amp[mask]

    # Spectrum
    ax.semilogy(w_plot, a_plot, 'k-', linewidth=0.5, alpha=0.8, zorder=2)

    # Peak detection and envelope (use low prominence for steep 1/w envelope)
    amp_range = np.max(amp) - np.min(amp)
    prom = 0.01 * amp_range
    pk_idx, pk_freq, pk_amp = find_spectral_peaks(
        amp, omega_yr, min_distance=3, prominence=prom, freq_range=(0.3, max_freq))
    tr_idx, tr_freq, tr_amp = find_spectral_troughs(
        amp, omega_yr, min_distance=3, prominence=prom, freq_range=(0.3, max_freq))

    # Envelopes
    try:
        upper_fit = fit_upper_envelope(pk_freq, pk_amp)
        lower_fit = fit_lower_envelope(tr_freq, tr_amp)
        w_env = np.linspace(0.3, max_freq, 500)
        ax.semilogy(w_env, envelope_model(w_env, upper_fit['k']),
                     'r--', linewidth=1.2, alpha=0.7, zorder=3,
                     label=f"a = {upper_fit['k']:.3f}/w (R2={upper_fit['r_squared']:.2f})")
        ax.semilogy(w_env, envelope_model(w_env, lower_fit['k']),
                     'b--', linewidth=1.2, alpha=0.7, zorder=3,
                     label=f"a = {lower_fit['k']:.3f}/w (R2={lower_fit['r_squared']:.2f})")
    except Exception as e:
        print(f"  Envelope fit warning: {e}")

    # Draw filter bands
    draw_filter_bands(ax, FILTER_SPECS, a_plot.min(), a_plot.max())

    # Mark peaks
    ax.semilogy(pk_freq, pk_amp, 'rv', markersize=3, alpha=0.6, zorder=4)

    ax.set_xlim(0, max_freq)
    ax.set_xlabel('Angular Frequency w (rad/yr)', fontsize=10)
    ax.set_ylabel('Amplitude (log)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc='upper right')

    # Nyquist annotation
    nyquist = np.pi * fs
    if nyquist < max_freq * 1.5:
        ax.axvline(nyquist, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.text(nyquist, a_plot.max() * 0.5, f'Nyquist\n{nyquist:.1f}',
                fontsize=7, ha='left', color='gray')


def main():
    print("=" * 70)
    print("Lanczos Spectra with 6 Hurst Filter Passbands Overlaid")
    print("=" * 70)

    # Load data
    print("\nLoading weekly data...")
    close_w, fs_w = load_weekly_data()
    print(f"  {len(close_w)} weekly samples, fs={fs_w}")

    print("Loading daily data...")
    close_d, fs_d = load_daily_data()
    print(f"  {len(close_d)} daily samples, fs={fs_d:.1f}")

    # Compute spectra
    print("\nComputing weekly Lanczos spectrum...")
    omega_w, amp_w = compute_lanczos(close_w, fs_w)
    print(f"  {len(omega_w)} frequency bins, max={omega_w[-1]:.1f} rad/yr")

    print("Computing daily Lanczos spectrum...")
    omega_d, amp_d = compute_lanczos(close_d, fs_d)
    print(f"  {len(omega_d)} frequency bins, max={omega_d[-1]:.1f} rad/yr")

    # =========================================================================
    # Figure 1: Full view (0-40 rad/yr) - Weekly and Daily side by side
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    plot_spectrum_with_filters(ax1, omega_w, amp_w, fs_w,
                               'Weekly Lanczos Spectrum + 6 Filter Passbands (DJIA 1921-1965)',
                               max_freq=40)
    plot_spectrum_with_filters(ax2, omega_d, amp_d, fs_d,
                               'Daily Lanczos Spectrum + 6 Filter Passbands (DJIA 1921-1965)',
                               max_freq=40)

    # Legend for filter bands
    legend_patches = [mpatches.Patch(color=s['color'], alpha=0.3, label=s['label'])
                      for s in FILTER_SPECS]
    fig.legend(handles=legend_patches, loc='lower center', ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out1 = os.path.join(SCRIPT_DIR, 'fig_lanczos_6filters_full.png')
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out1}")

    # =========================================================================
    # Figure 2: Zoomed view (0-14 rad/yr) showing first 4 filters in detail
    # =========================================================================
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(16, 12))

    plot_spectrum_with_filters(ax3, omega_w, amp_w, fs_w,
                               'Weekly Lanczos Spectrum - Zoomed (0-14 rad/yr)',
                               max_freq=14)
    plot_spectrum_with_filters(ax4, omega_d, amp_d, fs_d,
                               'Daily Lanczos Spectrum - Zoomed (0-14 rad/yr)',
                               max_freq=14)

    fig2.legend(handles=legend_patches[:4], loc='lower center', ncol=4, fontsize=9,
                bbox_to_anchor=(0.5, -0.02))
    fig2.tight_layout(rect=[0, 0.03, 1, 1])
    out2 = os.path.join(SCRIPT_DIR, 'fig_lanczos_6filters_zoomed.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")

    # =========================================================================
    # Figure 3: Linear amplitude (not log) to show filter-spectrum relationship
    # =========================================================================
    fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(16, 10))

    # Weekly linear
    mask_w = (omega_w > 0.1) & (omega_w <= 14)
    ax5.plot(omega_w[mask_w], amp_w[mask_w], 'k-', linewidth=0.5, alpha=0.8)
    draw_filter_bands(ax5, FILTER_SPECS[:4], 0, amp_w[mask_w].max())
    ax5.set_xlim(0, 14)
    ax5.set_ylabel('Amplitude (linear)', fontsize=10)
    ax5.set_title('Weekly Lanczos - Linear Amplitude (0-14 rad/yr)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.2)

    # Daily linear
    mask_d = (omega_d > 0.1) & (omega_d <= 14)
    ax6.plot(omega_d[mask_d], amp_d[mask_d], 'k-', linewidth=0.5, alpha=0.8)
    draw_filter_bands(ax6, FILTER_SPECS[:4], 0, amp_d[mask_d].max())
    ax6.set_xlim(0, 14)
    ax6.set_xlabel('Angular Frequency w (rad/yr)', fontsize=10)
    ax6.set_ylabel('Amplitude (linear)', fontsize=10)
    ax6.set_title('Daily Lanczos - Linear Amplitude (0-14 rad/yr)', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.2)

    fig3.tight_layout()
    out3 = os.path.join(SCRIPT_DIR, 'fig_lanczos_6filters_linear.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"Saved: {out3}")

    # =========================================================================
    # Print energy in each filter band
    # =========================================================================
    print("\n" + "-" * 70)
    print("SPECTRAL ENERGY IN EACH FILTER BAND")
    print("-" * 70)

    for data_label, omega, amp in [("Weekly", omega_w, amp_w), ("Daily", omega_d, amp_d)]:
        total_energy = np.sum(amp[omega > 0.1] ** 2)
        print(f"\n  {data_label} data:")
        for spec in FILTER_SPECS:
            if spec['type'] == 'lp':
                mask = (omega > 0) & (omega <= spec['f_stop'])
            else:
                mask = (omega >= spec['f1']) & (omega <= spec['f4'])
            band_energy = np.sum(amp[mask] ** 2)
            pct = band_energy / total_energy * 100
            print(f"    {spec['label']:20s}  energy={pct:6.2f}%")

        # Gap energy
        gap_energy = total_energy
        for spec in FILTER_SPECS:
            if spec['type'] == 'lp':
                mask = (omega > 0) & (omega <= spec['f_stop'])
            else:
                mask = (omega >= spec['f1']) & (omega <= spec['f4'])
            gap_energy -= np.sum(amp[mask] ** 2)
        print(f"    {'Gaps (uncaptured)':20s}  energy={gap_energy/total_energy*100:6.2f}%")

    plt.close('all')
    print("\nDone.")


if __name__ == '__main__':
    main()
