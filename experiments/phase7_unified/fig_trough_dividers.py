#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure: Lanczos Trough Dividing Points on Harmonic Index Plot (AI-7 Style)

Hurst used the low-frequency troughs (minima) in the wide lobes of the
Lanczos spectrum as natural dividing points between groups of harmonics.
These dividers define the boundaries of the "nominal cycles" in the
Detailed Nominal Model.

This script:
1. Computes the Lanczos spectrum for weekly DJIA data
2. Detects the wide-lobe troughs (deep minima between spectral groups)
3. Maps trough frequencies to harmonic numbers on the omega_n = 0.3676*N line
4. Plots the AI-7 style harmonic index plot with trough dividers as
   horizontal lines, showing how the Nominal Model groups emerge

Reference: J.M. Hurst, Appendix A, Figures AI-7 and AI-8
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from src.spectral.lanczos import lanczos_spectrum
from src.spectral.peak_detection import find_spectral_peaks, find_spectral_troughs
from src.spectral.envelopes import fit_upper_envelope, envelope_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))

# Hurst's analysis window
DATE_START = '1921-04-29'
DATE_END = '1965-05-21'
OMEGA_SPACING = 0.3676  # rad/yr per harmonic N
TWOPI = 2 * np.pi

# Nominal model CSV
NOMINAL_MODEL_PATH = os.path.join(BASE_DIR, 'data/processed/nominal_model.csv')

# Nominal cycle group definitions (from Hurst AI-8)
NOMINAL_GROUPS = [
    {'name': '18.0 Y', 'N_range': (1, 1), 'color': '#2196F3'},
    {'name': '9.0 Y', 'N_range': (2, 2), 'color': '#4CAF50'},
    {'name': '4.3 Y', 'N_range': (3, 4), 'color': '#FF9800'},
    {'name': '3.0 Y', 'N_range': (5, 7), 'color': '#F44336'},
    {'name': '18.0 M', 'N_range': (8, 12), 'color': '#9C27B0'},
    {'name': '12.0 M', 'N_range': (13, 19), 'color': '#795548'},
    {'name': '9.0 M', 'N_range': (20, 26), 'color': '#607D8B'},
    {'name': '6.0 M', 'N_range': (27, 34), 'color': '#E91E63'},
]


def load_weekly_data():
    csv_path = os.path.join(BASE_DIR, 'data/raw/^dji_w.csv')
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df_h = df[df.Date.between(DATE_START, DATE_END)].copy()
    return df_h.Close.values


def find_wide_lobe_troughs(omega_yr, amp, min_freq=0.3, max_freq=13.0):
    """
    Find the deep troughs between wide spectral lobes.

    These are the natural group boundaries in the Lanczos spectrum.
    The spectrum has very high dynamic range (DC ~ 271, peaks at 13 rad/yr ~ 5),
    so we use a low prominence threshold (1% of range) and min_distance=3 to
    catch the wide-lobe troughs that sit between the major spectral groups.
    """
    # Use 1% prominence to capture the troughs in the steep 1/w envelope
    amp_range = np.max(amp) - np.min(amp)
    prom = 0.01 * amp_range

    tr_idx, tr_freq, tr_amp = find_spectral_troughs(
        amp, omega_yr, min_distance=3, prominence=prom,
        freq_range=(min_freq, max_freq))

    return tr_idx, tr_freq, tr_amp


def map_freq_to_N(freq, spacing=OMEGA_SPACING):
    """Map frequency to continuous harmonic number."""
    return freq / spacing


def main():
    print("=" * 70)
    print("Lanczos Trough Dividers on Harmonic Index Plot")
    print("=" * 70)

    # Load data
    print("\nLoading weekly DJIA data...")
    close = load_weekly_data()
    print(f"  {len(close)} samples")

    # Compute Lanczos spectrum
    print("Computing Lanczos spectrum...")
    w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(close, 1, 52)
    omega_yr = w * 52
    print(f"  {len(omega_yr)} frequency bins")

    # Load nominal model
    print("Loading nominal model...")
    nm = pd.read_csv(NOMINAL_MODEL_PATH)
    print(f"  {len(nm)} spectral lines")

    # Find peaks and deep troughs
    print("\nDetecting spectral peaks...")
    amp_range = np.max(amp) - np.min(amp)
    prom = 0.01 * amp_range
    pk_idx, pk_freq, pk_amp = find_spectral_peaks(
        amp, omega_yr, min_distance=3, prominence=prom, freq_range=(0.3, 13.0))
    print(f"  {len(pk_freq)} peaks detected")

    print("Detecting wide-lobe trough dividers...")
    tr_idx, tr_freq, tr_amp = find_wide_lobe_troughs(omega_yr, amp)
    print(f"  {len(tr_freq)} deep troughs found:")
    for f, a in zip(tr_freq, tr_amp):
        N_cont = map_freq_to_N(f)
        T_yr = TWOPI / f
        print(f"    w={f:.3f} rad/yr  N={N_cont:.2f}  T={T_yr:.2f} yr  amp={a:.4f}")

    # Map nominal model to harmonic numbers
    fourier_N = []
    fourier_omega = []
    for _, row in nm.iterrows():
        N = round(row['frequency'] / OMEGA_SPACING)
        if 1 <= N <= 34:
            fourier_N.append(N)
            fourier_omega.append(row['frequency'])

    # Map troughs to continuous harmonic numbers
    trough_N_cont = [map_freq_to_N(f) for f in tr_freq]

    # =========================================================================
    # Figure 1: Lanczos spectrum with trough markers
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(16, 6))

    mask = (omega_yr > 0.1) & (omega_yr <= 13)
    ax1.semilogy(omega_yr[mask], amp[mask], 'k-', linewidth=0.6, alpha=0.8)
    ax1.semilogy(pk_freq, pk_amp, 'rv', markersize=4, alpha=0.6, label='Peaks')
    ax1.semilogy(tr_freq, tr_amp, 'b^', markersize=6, zorder=5,
                  label='Group dividing troughs')

    # Shade nominal groups
    for grp in NOMINAL_GROUPS:
        n_lo, n_hi = grp['N_range']
        w_lo = (n_lo - 0.5) * OMEGA_SPACING
        w_hi = (n_hi + 0.5) * OMEGA_SPACING
        if w_hi <= 13:
            ax1.axvspan(w_lo, w_hi, color=grp['color'], alpha=0.08, zorder=0)
            ax1.text((w_lo + w_hi) / 2, amp[mask].max() * 1.5,
                     grp['name'], fontsize=7, ha='center', va='bottom',
                     color=grp['color'], fontweight='bold')

    # Vertical lines at troughs
    for f in tr_freq:
        ax1.axvline(f, color='blue', linestyle=':', linewidth=0.8, alpha=0.4)

    ax1.set_xlim(0, 13)
    ax1.set_xlabel('Angular Frequency w (rad/yr)', fontsize=10)
    ax1.set_ylabel('Amplitude (log)', fontsize=10)
    ax1.set_title('Lanczos Spectrum with Group-Dividing Troughs Marked',
                   fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    fig1.tight_layout()
    out1 = os.path.join(SCRIPT_DIR, 'fig_trough_dividers_spectrum.png')
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out1}")

    # =========================================================================
    # Figure 2: AI-7 Style Harmonic Index Plot with Trough Dividers
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(11, 11))

    N_max = 34
    N_line = np.linspace(0, N_max, 200)
    omega_line = OMEGA_SPACING * N_line

    # Reference line: omega_N = 0.3676 * N
    ax2.plot(N_line, omega_line, '-', color='black', linewidth=0.8, zorder=1)
    ax2.text(16, 0.3676 * 16 + 0.2, f'$\\omega_n = {OMEGA_SPACING}\\, N$',
             fontsize=11,
             rotation=np.degrees(np.arctan(OMEGA_SPACING * (11 / N_max))),
             rotation_mode='anchor')

    # Fourier analysis points
    ax2.scatter(fourier_N, fourier_omega, marker='x', s=60, linewidths=1.5,
                color='black', zorder=4, label='Fourier Analysis')

    # Tick marks on diagonal
    for N in range(1, N_max + 1):
        omega_exact = N * OMEGA_SPACING
        if omega_exact <= 12.5:
            ax2.plot([N - 0.3, N + 0.3],
                     [omega_exact - 0.111, omega_exact + 0.111],
                     '-', color='black', linewidth=0.6, alpha=0.4)

    # TROUGH DIVIDERS as horizontal lines
    for f, N_cont in zip(tr_freq, trough_N_cont):
        if f <= 12.5:
            ax2.axhline(f, color='blue', linestyle='--', linewidth=1.0, alpha=0.5)
            # Also show as vertical line at corresponding N
            ax2.axvline(N_cont, color='blue', linestyle=':', linewidth=0.5, alpha=0.3)
            # Label
            ax2.text(N_max + 0.3, f, f'w={f:.2f}\nN={N_cont:.1f}',
                     fontsize=7, va='center', color='blue')

    # Shade nominal groups
    for grp in NOMINAL_GROUPS:
        n_lo, n_hi = grp['N_range']
        w_lo = (n_lo - 0.5) * OMEGA_SPACING
        w_hi = (n_hi + 0.5) * OMEGA_SPACING
        if w_hi <= 12.5:
            ax2.axhspan(w_lo, w_hi, color=grp['color'], alpha=0.06, zorder=0)
            ax2.text(-1.5, (w_lo + w_hi) / 2, grp['name'],
                     fontsize=8, ha='center', va='center',
                     color=grp['color'], fontweight='bold',
                     rotation=0)

    ax2.set_xlim(-2.5, N_max + 2)
    ax2.set_ylim(0, 12.5)
    ax2.set_xlabel('N  -->', fontsize=12)
    ax2.set_ylabel('$\\omega_n$  --  RADIANS/YEAR', fontsize=12)
    ax2.set_xticks(np.arange(0, N_max + 1, 2))
    ax2.set_yticks(np.arange(0, 13, 1))
    ax2.grid(True, alpha=0.25)

    ax2.set_title(
        'HARMONIC INDEX with LANCZOS TROUGH DIVIDERS\n'
        'DOW-JONES INDUSTRIAL AVERAGE\n'
        f'$\\omega_n = {OMEGA_SPACING}\\, N$ + Group Boundaries from Spectral Troughs',
        fontsize=11, fontweight='bold', pad=10
    )

    # Legend
    handle_fourier = mlines.Line2D([], [], marker='x', color='black',
                                    linestyle='None', markersize=7,
                                    label='Fourier Analysis')
    handle_divider = mlines.Line2D([], [], color='blue', linestyle='--',
                                    linewidth=1.0, label='Spectral Trough Divider')
    ax2.legend(handles=[handle_fourier, handle_divider],
               loc='lower right', fontsize=9, framealpha=0.9)

    ax2.text(0.97, 0.03, 'FIGURE AI-7 (Extended)', transform=ax2.transAxes,
             fontsize=10, ha='right', va='bottom', color='gray')

    fig2.tight_layout()
    out2 = os.path.join(SCRIPT_DIR, 'fig_trough_dividers_AI7.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")

    # =========================================================================
    # Figure 3: Combined view - Spectrum above, Harmonic plot below
    # =========================================================================
    fig3, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 16),
                                           gridspec_kw={'height_ratios': [1, 1.5]})

    # Top: Spectrum with troughs
    mask = (omega_yr > 0.1) & (omega_yr <= 13)
    ax_top.semilogy(omega_yr[mask], amp[mask], 'k-', linewidth=0.6, alpha=0.8)
    ax_top.semilogy(tr_freq, tr_amp, 'b^', markersize=8, zorder=5,
                     label='Group Dividers')
    for f in tr_freq:
        if f <= 13:
            ax_top.axvline(f, color='blue', linestyle=':', linewidth=0.8, alpha=0.4)
    for grp in NOMINAL_GROUPS:
        n_lo, n_hi = grp['N_range']
        w_lo = (n_lo - 0.5) * OMEGA_SPACING
        w_hi = (n_hi + 0.5) * OMEGA_SPACING
        if w_hi <= 13:
            ax_top.axvspan(w_lo, w_hi, color=grp['color'], alpha=0.08, zorder=0)
    ax_top.set_xlim(0, 13)
    ax_top.set_xlabel('w (rad/yr)', fontsize=10)
    ax_top.set_ylabel('Amplitude (log)', fontsize=10)
    ax_top.set_title('Lanczos Spectrum: Troughs Define Nominal Cycle Group Boundaries',
                      fontsize=11, fontweight='bold')
    ax_top.legend(fontsize=9)
    ax_top.grid(True, alpha=0.2)

    # Bottom: Harmonic index with dividers (rotated: w on Y, N on X)
    ax_bot.plot(N_line, omega_line, '-', color='black', linewidth=0.8, zorder=1)
    ax_bot.scatter(fourier_N, fourier_omega, marker='x', s=60, linewidths=1.5,
                   color='black', zorder=4)
    for f in tr_freq:
        if f <= 12.5:
            ax_bot.axhline(f, color='blue', linestyle='--', linewidth=1.0, alpha=0.5)
    for grp in NOMINAL_GROUPS:
        n_lo, n_hi = grp['N_range']
        w_lo = (n_lo - 0.5) * OMEGA_SPACING
        w_hi = (n_hi + 0.5) * OMEGA_SPACING
        if w_hi <= 12.5:
            ax_bot.axhspan(w_lo, w_hi, color=grp['color'], alpha=0.06, zorder=0)
            ax_bot.text(N_max + 0.5, (w_lo + w_hi) / 2, grp['name'],
                        fontsize=8, va='center', color=grp['color'], fontweight='bold')
    for N in range(1, N_max + 1):
        omega_exact = N * OMEGA_SPACING
        if omega_exact <= 12.5:
            ax_bot.plot([N - 0.3, N + 0.3],
                        [omega_exact - 0.111, omega_exact + 0.111],
                        '-', color='black', linewidth=0.6, alpha=0.4)
    ax_bot.set_xlim(0, N_max + 2)
    ax_bot.set_ylim(0, 12.5)
    ax_bot.set_xlabel('Harmonic Number N', fontsize=10)
    ax_bot.set_ylabel('w (rad/yr)', fontsize=10)
    ax_bot.set_title(f'Harmonic Index: w_n = {OMEGA_SPACING} * N with Trough Boundaries',
                      fontsize=11, fontweight='bold')
    ax_bot.grid(True, alpha=0.25)

    fig3.tight_layout()
    out3 = os.path.join(SCRIPT_DIR, 'fig_trough_dividers_combined.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"Saved: {out3}")

    # =========================================================================
    # Print divider analysis
    # =========================================================================
    print("\n" + "-" * 70)
    print("TROUGH DIVIDER ANALYSIS")
    print("-" * 70)
    print("\nTrough frequencies and their harmonic positions:")
    print(f"{'Trough w':>10s}  {'N (cont)':>8s}  {'Period':>10s}  {'Between Groups':>25s}")
    print("-" * 60)
    for f, N_c in zip(tr_freq, trough_N_cont):
        T = TWOPI / f
        # Identify which groups this divider separates
        below = ""
        above = ""
        for grp in NOMINAL_GROUPS:
            n_lo, n_hi = grp['N_range']
            if n_hi < N_c:
                below = grp['name']
            elif n_lo > N_c and not above:
                above = grp['name']
        if below and above:
            between = f"{below} | {above}"
        else:
            between = "edge"
        print(f"{f:10.3f}  {N_c:8.2f}  {T:8.2f} yr  {between:>25s}")

    plt.close('all')
    print("\nDone.")


if __name__ == '__main__':
    main()
