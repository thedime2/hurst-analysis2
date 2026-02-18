# -*- coding: utf-8 -*-
"""
Figure AI-2: Idealized Comb Filter Frequency Response
Appendix A, Figure AI-2 Reproduction

Reproduces Hurst's "Idealized Comb Filter" figure showing the frequency
response of all 23 overlapping band-pass filters.

Two subplots:
  Top:    Weekly data (fs=52) - Ormsby vs CMW vs Ideal trapezoidal
  Bottom: Daily data  (fs~251) - Ormsby vs CMW vs Ideal trapezoidal

X-axis zoomed to 7.0-13.0 rad/yr to match Hurst's figure.

Reference: J.M. Hurst, Appendix A, Figure AI-2, p.192
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from utils_ai import (
    design_comb_bank, make_ormsby_kernels, daily_nw,
    ormsby_frequency_response, cmw_frequency_response, idealized_response,
    load_daily_data, FS_WEEKLY, NW_WEEKLY,
    print_comb_bank_summary,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NFFT = 65536
XMIN, XMAX = 7.0, 13.0   # rad/yr zoom range for plot


# ============================================================================
# MAIN
# ============================================================================

print("=" * 70)
print("Figure AI-2: Comb Filter Frequency Response")
print("Ormsby vs CMW, Weekly vs Daily")
print("=" * 70)
print()

# --- Weekly filter bank ---
print("Weekly (fs=52):")
specs_w = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
print_comb_bank_summary(specs_w)
filters_w = make_ormsby_kernels(specs_w, fs=FS_WEEKLY)
freqs_w, H_orm_w = ormsby_frequency_response(filters_w, fs=FS_WEEKLY, nfft=NFFT)
freqs_cmw_w, H_cmw_w = cmw_frequency_response(specs_w, fs=FS_WEEKLY, nfft=NFFT)
print()

# --- Daily filter bank ---
print("Daily (fs~251):")
_, _, fs_daily = load_daily_data()
nw_daily = daily_nw(fs_daily)
print(f"  fs_daily = {fs_daily:.1f}, nw_daily = {nw_daily}")
specs_d = design_comb_bank(fs=fs_daily, nw=nw_daily)
print_comb_bank_summary(specs_d)
filters_d = make_ormsby_kernels(specs_d, fs=fs_daily)
freqs_d, H_orm_d = ormsby_frequency_response(filters_d, fs=fs_daily, nfft=NFFT)
freqs_cmw_d, H_cmw_d = cmw_frequency_response(specs_d, fs=fs_daily, nfft=NFFT)
print()

# ============================================================================
# PLOT
# ============================================================================

# Color palette: 23 distinct colours cycling through tab20 + tab20b
cmap = plt.cm.get_cmap('tab20', 20)
colors = [cmap(i % 20) for i in range(23)]

fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True, sharey=True)

for ax_idx, (ax, specs, freqs_orm, H_orm, freqs_cmw, H_cmw, fs, title_fs) in enumerate([
    (axes[0], specs_w, freqs_w, H_orm_w, freqs_cmw_w, H_cmw_w, FS_WEEKLY, 52),
    (axes[1], specs_d, freqs_d, H_orm_d, freqs_cmw_d, H_cmw_d, fs_daily, int(round(fs_daily))),
]):
    # Mask to zoom range
    mask_orm = (freqs_orm >= XMIN - 0.5) & (freqs_orm <= XMAX + 0.5)
    mask_cmw = (freqs_cmw >= XMIN - 0.5) & (freqs_cmw <= XMAX + 0.5)

    # Frequency axes for zoom
    f_orm = freqs_orm[mask_orm]
    f_cmw = freqs_cmw[mask_cmw]

    # Dense frequency grid for idealized trapezoid
    f_ideal = np.linspace(XMIN - 0.5, XMAX + 0.5, 4000)

    for i, spec in enumerate(specs):
        color = colors[i]

        # Idealized trapezoid (thin gray, drawn first)
        H_ideal = idealized_response(spec, f_ideal)
        ax.plot(f_ideal, H_ideal, '-', color='lightgray', linewidth=0.6,
                alpha=0.7, zorder=1)

        # Ormsby actual response (solid)
        ax.plot(f_orm, H_orm[i][mask_orm], '-', color=color,
                linewidth=1.0, alpha=0.85, zorder=3)

        # CMW response (dashed)
        ax.plot(f_cmw, H_cmw[i][mask_cmw], '--', color=color,
                linewidth=0.9, alpha=0.75, zorder=2)

        # Filter number label at top of idealized passband
        f_label = (spec['f2'] + spec['f3']) / 2.0
        if XMIN <= f_label <= XMAX:
            ax.text(f_label, 1.03, str(i + 1), ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold', color=color,
                    zorder=4)

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(0.0, 1.13)
    ax.set_ylabel('Amplitude Ratio', fontsize=11)
    ax.set_title(
        f'{"Weekly" if ax_idx == 0 else "Daily"}  '
        f'(fs = {title_fs} samples/yr,  nw = {specs[0]["nw"]})',
        fontsize=11, fontweight='bold'
    )
    ax.grid(True, alpha=0.25)
    ax.axhline(1.0, color='silver', linewidth=0.5, linestyle=':')

axes[1].set_xlabel('Angular Frequency (rad/yr)', fontsize=12)

# Legend (shared)
patch_ideal  = mpatches.Patch(facecolor='lightgray', label='Ideal trapezoid')
line_ormsby  = mlines.Line2D([], [], color='gray', linestyle='-',  linewidth=1.5, label='Ormsby FIR (actual)')
line_cmw     = mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5, label='CMW Gaussian (actual)')
axes[0].legend(handles=[patch_ideal, line_ormsby, line_cmw],
               loc='upper right', fontsize=9, framealpha=0.9)

fig.suptitle(
    'IDEALIZED COMB FILTER  -  Figure AI-2\n'
    'Ormsby FIR vs Complex Morlet Wavelet  |  Weekly vs Daily',
    fontsize=13, fontweight='bold', y=0.98
)

fig.tight_layout(rect=[0, 0, 1, 0.97])
out_path = os.path.join(SCRIPT_DIR, 'fig_AI2_filter_response.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_path}")

# ============================================================================
# SECOND FIGURE: Detailed view of 4 selected filters showing filter quality
# ============================================================================

# Show filters FC-1, FC-6 (near 8.6), FC-11 (near 10.0), FC-21 (near 11.8)
highlight_idx = [0, 5, 10, 20]
highlight_labels = ['FC-1 (7.6 r/y)', 'FC-6 (8.6 r/y)',
                    'FC-11 (9.6 r/y)', 'FC-21 (11.6 r/y)']

fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9), sharex=False, sharey=True)

for row_idx, (specs, freqs_orm, H_orm, freqs_cmw, H_cmw, fs, row_label) in enumerate([
    (specs_w, freqs_w, H_orm_w, freqs_cmw_w, H_cmw_w, FS_WEEKLY, 'Weekly'),
    (specs_d, freqs_d, H_orm_d, freqs_cmw_d, H_cmw_d, fs_daily, 'Daily'),
]):
    for col_idx, (filt_i, flabel) in enumerate(zip(highlight_idx, highlight_labels)):
        ax = axes2[row_idx, col_idx]
        spec = specs[filt_i]
        color = colors[filt_i]

        # Zoom to this filter's extent + 0.5 margin
        xlim = (spec['f1'] - 0.4, spec['f4'] + 0.4)
        f_ideal = np.linspace(xlim[0], xlim[1], 2000)

        # Ideal
        H_id = idealized_response(spec, f_ideal)
        ax.plot(f_ideal, H_id, '-', color='lightgray', linewidth=1.5, label='Ideal')

        # Ormsby
        mask = (freqs_orm >= xlim[0]) & (freqs_orm <= xlim[1])
        ax.plot(freqs_orm[mask], H_orm[filt_i][mask], '-', color=color,
                linewidth=1.5, label='Ormsby')

        # CMW
        mask_c = (freqs_cmw >= xlim[0]) & (freqs_cmw <= xlim[1])
        ax.plot(freqs_cmw[mask_c], H_cmw[filt_i][mask_c], '--', color=color,
                linewidth=1.5, label='CMW')

        # 0.5 line
        ax.axhline(0.5, color='silver', linewidth=0.6, linestyle=':')

        ax.set_xlim(xlim)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3)

        if row_idx == 0:
            ax.set_title(flabel, fontsize=10, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(f'{row_label}\nAmplitude', fontsize=9)
        if row_idx == 1:
            ax.set_xlabel('rad/yr', fontsize=9)

        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=8, loc='upper left')

fig2.suptitle('Filter Shape Detail: 4 Selected Ormsby Comb Filters\n'
              'Ormsby (solid) vs CMW (dashed) vs Ideal (gray)',
              fontsize=12, fontweight='bold')
fig2.tight_layout()
out2 = os.path.join(SCRIPT_DIR, 'fig_AI2_filter_detail.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: {out2}")
print()
print("Done.")
