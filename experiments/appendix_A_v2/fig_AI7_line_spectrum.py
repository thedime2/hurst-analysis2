# -*- coding: utf-8 -*-
"""
Figure AI-7: Low Frequency Line Spectrum - Dow Jones Industrial Average
Appendix A, Figure AI-7 Reproduction

Shows the line spectrum with w_n = 0.3676*N demonstrating that the
spectral lines are harmonically related. Plots both:
  - Fourier analysis points (from nominal model, Phase 3)
  - Digital filter analysis points (from AI-6 MPM Prony results)

Reference: J.M. Hurst, Appendix A, Figure AI-7, p.198
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))

NOMINAL_MODEL_PATH = os.path.join(BASE_DIR, 'data/processed/nominal_model.csv')
LSE_DATA_PATH      = os.path.join(SCRIPT_DIR, 'ai6_lse_segments.npz')

# Hurst's nominal spacing
OMEGA_SPACING = 0.3676   # rad/yr per harmonic number N

# ============================================================================
# HELPERS
# ============================================================================

def map_to_harmonic(f_radyr, spacing=OMEGA_SPACING):
    """
    Map a frequency to the nearest harmonic number N such that
    omega_N = N * spacing is closest to f_radyr.
    Returns (N, omega_N, error).
    """
    N = round(f_radyr / spacing)
    if N < 1:
        N = 1
    omega_N = N * spacing
    error   = f_radyr - omega_N
    return int(N), omega_N, error


def cluster_lse_estimates(f_estimates, spacing=OMEGA_SPACING, cluster_tol=0.15):
    """
    Cluster MPM frequency estimates to identify stable harmonic lines.

    Assigns each estimate to the nearest harmonic and computes the
    median + std per harmonic for plotting.

    Returns DataFrame: {N, omega_est_median, omega_est_std, count}
    """
    results = {}
    for f in f_estimates:
        N, omega_N, err = map_to_harmonic(f, spacing)
        # Only include if within ±cluster_tol of a harmonic
        if abs(err) < cluster_tol and 1 <= N <= 40:
            if N not in results:
                results[N] = []
            results[N].append(f)

    rows = []
    for N, vals in results.items():
        vals = np.array(vals)
        rows.append({
            'N': N,
            'omega_est': np.median(vals),
            'omega_std': np.std(vals),
            'count': len(vals),
        })
    df = pd.DataFrame(rows).sort_values('N').reset_index(drop=True)
    return df


# ============================================================================
# MAIN
# ============================================================================

print("=" * 70)
print("Figure AI-7: Low Frequency Line Spectrum")
print("=" * 70)
print()

# --- Load Fourier analysis points (nominal model from Phase 3) ---
print("Loading nominal model (Fourier analysis points)...")
nm = pd.read_csv(NOMINAL_MODEL_PATH)
print(f"  {len(nm)} spectral lines, "
      f"freq range: {nm.frequency.min():.2f} - {nm.frequency.max():.2f} rad/yr")

# Map nominal model lines to harmonic numbers
fourier_N     = []
fourier_omega = []
for _, row in nm.iterrows():
    N, omega_N, err = map_to_harmonic(row['frequency'])
    if N <= 34:
        fourier_N.append(N)
        fourier_omega.append(row['frequency'])
print(f"  Mapped {len(fourier_N)} lines to N=1..34")
print()

# --- Load AI-6 MPM Prony estimates ---
print("Loading AI-6 MPM Prony estimates (digital filter analysis)...")
if os.path.exists(LSE_DATA_PATH):
    lse = np.load(LSE_DATA_PATH)
    f_est_all = lse['f_estimates']
    f_centers = lse['filter_centers']
    print(f"  {len(f_est_all)} total estimates, "
          f"range: {f_est_all.min():.2f} - {f_est_all.max():.2f} rad/yr")

    df_clusters = cluster_lse_estimates(f_est_all, cluster_tol=0.15)
    print(f"  {len(df_clusters)} harmonic clusters identified")
    print()
    print("  Harmonic clusters (N, median_omega, count):")
    for _, row in df_clusters.iterrows():
        print(f"    N={int(row['N']):>2d}  omega={row['omega_est']:.3f} rad/yr  "
              f"(expected {int(row['N'])*OMEGA_SPACING:.3f})  count={int(row['count'])}")
    print()
else:
    print(f"  WARNING: AI-6 LSE data not found at {LSE_DATA_PATH}")
    print(f"  Run fig_AI6_lse_analysis.py first to generate it.")
    df_clusters = pd.DataFrame()
    f_est_all   = np.array([])

# ============================================================================
# PLOT
# ============================================================================

fig, ax = plt.subplots(figsize=(11, 11))

# N axis range: 0 to 34 (matching Hurst's figure)
N_max = 34
N_line = np.linspace(0, N_max, 200)
omega_line = OMEGA_SPACING * N_line

# Reference line: omega_N = 0.3676 * N
ax.plot(N_line, omega_line, '-', color='black', linewidth=0.8, zorder=1)
ax.text(16, 0.3676 * 16 + 0.2, f'$\\omega_n = {OMEGA_SPACING}\\, N$',
        fontsize=11, rotation=np.degrees(np.arctan(OMEGA_SPACING * (11/N_max))),
        rotation_mode='anchor')

# Fourier analysis points (circles with cross, like Hurst's 'x' marks)
if len(fourier_N) > 0:
    ax.scatter(fourier_N, fourier_omega, marker='x', s=60, linewidths=1.5,
               color='black', zorder=4, label='Fourier Analysis')

# Digital filter analysis points (filled dots) from AI-6 clusters
if len(df_clusters) > 0:
    # Only show harmonic numbers that are in the comb bank range (N~20..33)
    # The comb bank covers 7.6-12.0 rad/yr → N = 7.6/0.3676..12.0/0.3676 ≈ 21..33
    ax.scatter(df_clusters.N, df_clusters.omega_est,
               marker='.', s=80, color='black', zorder=4,
               label='Digital Filter Analysis')

    # Error bars (std of cluster)
    ax.errorbar(df_clusters.N, df_clusters.omega_est,
                yerr=df_clusters.omega_std,
                fmt='none', color='gray', linewidth=0.8, capsize=2,
                alpha=0.5, zorder=3)

# Hurst's figure specific annotations
# The tick marks on the diagonal line (little x symbols from the book)
for N in range(1, N_max + 1):
    omega_exact = N * OMEGA_SPACING
    if omega_exact <= 12.5:
        ax.plot([N - 0.3, N + 0.3], [omega_exact - 0.111, omega_exact + 0.111],
                '-', color='black', linewidth=0.6, alpha=0.4)

ax.set_xlim(0, N_max)
ax.set_ylim(0, 12.5)
ax.set_xlabel('N  -->', fontsize=12)
ax.set_ylabel('$\\omega_n$  --  RADIANS/YEAR', fontsize=12)

# Grid matching Hurst's figure
ax.set_xticks(np.arange(0, N_max + 1, 2))
ax.set_yticks(np.arange(0, 13, 1))
ax.grid(True, alpha=0.25)

# Title
ax.set_title(
    'LOW FREQUENCY LINE SPECTRUM\nDOW-JONES INDUSTRIAL AVERAGE\n'
    f'$\\omega_n = {OMEGA_SPACING}\\, N$',
    fontsize=12, fontweight='bold', pad=10
)

# Legend (bottom right, as in Hurst's figure)
handle_fourier = mlines.Line2D([], [], marker='x', color='black',
                                linestyle='None', markersize=7,
                                label='Fourier Analysis')
handle_digital = mlines.Line2D([], [], marker='.', color='black',
                                linestyle='None', markersize=8,
                                label='Digital Filter Analysis')
ax.legend(handles=[handle_fourier, handle_digital],
          loc='lower right', fontsize=9, framealpha=0.9)

# Figure label
ax.text(0.97, 0.03, 'FIGURE AI-7', transform=ax.transAxes,
        fontsize=10, ha='right', va='bottom', color='gray')

fig.tight_layout()
out_path = os.path.join(SCRIPT_DIR, 'fig_AI7_line_spectrum.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_path}")
print()
print("Done.")
