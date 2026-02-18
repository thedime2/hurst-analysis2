# -*- coding: utf-8 -*-
"""
Figure AI-8: Low-Frequency Portion: Spectral Model
Appendix A, Figure AI-8 Reproduction

Reproduces Hurst's table of all 34 harmonic spectral lines:
  N | omega_n | T_Y | T_M | T_W | T_NOM

omega_n = n * 0.3676  rad/yr   (Hurst's fundamental spacing)
T_Y  = 2*pi / omega_n          (period in years)
T_M  = T_Y * 12               (period in months)
T_W  = 2*pi / omega_n * 52    (period in weeks)
T_NOM = Hurst's nominal label for selected harmonics

Group boundaries (horizontal dividers) follow Hurst's original layout:
  N = 1-2     (18Y, 9Y)
  N = 3-7     (4.3Y, 3.0Y)
  N = 8-14    (18M)
  N = 15-19   (12M)
  N = 20-26   (9M)
  N = 27-34   (6M)

Reference: J.M. Hurst, Appendix A, Figure AI-8, p.199
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OMEGA_SPACING = 0.3676   # rad/yr per harmonic number N

# ============================================================================
# COMPUTE TABLE DATA
# ============================================================================

# T_NOM entries: {N: label}  (matches Hurst's original)
T_NOM_LABELS = {
    1:  '18.0 Y',
    2:  '9.0 Y',
    4:  '4.3 Y',
    6:  '3.0 Y',
    10: '18.0 M',
    15: '12.0 M',
    23: '9.0 M',
    34: '6.0 M',
}

# Group boundaries: list of (N_start, N_end, line_weight)
# Heavy (1.2pt) lines drawn after each group; thin (0.6pt) sub-dividers within groups
GROUPS = [
    (1,  2,  1.2),    # 18Y, 9Y
    (3,  7,  1.2),    # 4.3Y, 3.0Y
    (8,  14, 1.2),    # 18M  (thin sub-divider after N=12)
    (15, 19, 1.2),    # 12M
    (20, 26, 1.2),    # 9M
    (27, 34, 0.0),    # 6M  (no line after — last group)
]
# Thin sub-dividers WITHIN a group (drawn after this N)
SUBDIV = {12}   # thin line within the 8-14 group

rows = []
for n in range(1, 35):
    omega_n = n * OMEGA_SPACING
    T_Y = 2 * np.pi / omega_n                # years
    T_M = T_Y * 12                            # months
    T_W = 2 * np.pi / omega_n * 52            # weeks

    # Hurst shows T_Y for N=1-14 only
    # N=1-9: 1 decimal; N=10-14: 2 decimals (matches Hurst's table precision)
    if n <= 9:
        ty_str = f'{T_Y:.1f}'
    elif n <= 14:
        ty_str = f'{T_Y:.2f}'
    else:
        ty_str = ''

    # Hurst shows T_M for N=3-34
    if n >= 3:
        tm_str = f'{T_M:.1f}' if T_M >= 10 else f'{T_M:.2f}'
    else:
        tm_str = ''

    # Hurst shows T_W for N=30-34 only
    tw_str = f'{T_W:.1f}' if n >= 30 else ''

    # omega_n formatting (match Hurst: 4 decimal places)
    om_str = f'{omega_n:.4f}'

    # T_NOM
    tnom_str = T_NOM_LABELS.get(n, '')

    rows.append({
        'N': n,
        'omega': om_str,
        'T_Y': ty_str,
        'T_M': tm_str,
        'T_W': tw_str,
        'T_NOM': tnom_str,
    })

print(f"Table: {len(rows)} harmonic rows computed.")
print()

# ============================================================================
# PLOT AS STYLED TABLE FIGURE
# ============================================================================

# Layout: 34 data rows + 1 header row
N_ROWS   = 35   # 1 header + 34 data
N_COLS   = 6    # N | omega | T_Y | T_M | T_W | T_NOM

# Figure size calibrated to give a clean table
FIG_W, FIG_H = 10, 14

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, N_COLS)
# Leave half-row padding at bottom (-0.7) and top (N_ROWS+1.2)
ax.set_ylim(-0.7, N_ROWS + 1.2)
ax.axis('off')

# Column x-centres and widths
COL_X = [0.42, 1.42, 2.52, 3.62, 4.72, 5.62]   # centre x of each column
# Approximate column right-edges (for drawing vertical lines)
COL_EDGES = [0, 0.82, 2.02, 3.12, 4.22, 5.22, 6.0]

HEADER_Y = N_ROWS - 0.3    # y position of header text (shifted up slightly)
DATA_Y0  = N_ROWS - 1.5    # y of first data row (N=1)

# Header text
headers = ['N', r'$\omega_n$', r'$T_Y$', r'$T_M$', r'$T_W$', 'T, NOM.']
for j, (cx, hdr) in enumerate(zip(COL_X, headers)):
    ax.text(cx, HEADER_Y, hdr,
            ha='center', va='center', fontsize=10.5, fontweight='bold',
            fontfamily='DejaVu Serif')

# Outer border
for x in [COL_EDGES[0], COL_EDGES[-1]]:
    ax.axvline(x, color='black', linewidth=1.2, ymin=0, ymax=1)
ax.axhline(DATA_Y0 + 0.5, color='black', linewidth=1.5,
           xmin=0, xmax=1)   # below header (half-row below header centre)
ax.axhline(DATA_Y0 - 34 + 0.5, color='black', linewidth=1.5,
           xmin=0, xmax=1)   # bottom (half row below N=34)

# Vertical column dividers (thin)
for x in COL_EDGES[1:-1]:
    ax.plot([x, x], [DATA_Y0 - 34 + 0.5, DATA_Y0 + 0.5],
            color='black', linewidth=0.6)

# Header top border
ax.axhline(HEADER_Y + 0.7, color='black', linewidth=1.5, xmin=0, xmax=1)

# ---- Data rows ----
# Group separator: heavy line after each group
group_end_rows = {g[1] for g in GROUPS}
# Caption note position update
CAPTION_Y = DATA_Y0 - 34 + 0.5 - 0.9   # below the bottom border

for i, row in enumerate(rows):
    n = row['N']
    y = DATA_Y0 - i    # y decreases going down

    # Row background: alternate very light shading for readability
    # (Hurst's original has no shading; skip for authentic look)

    # N column
    ax.text(COL_X[0], y, f'{n}.',
            ha='center', va='center', fontsize=8.5, fontfamily='DejaVu Serif')

    # omega_n column
    ax.text(COL_X[1], y, row['omega'],
            ha='center', va='center', fontsize=8.5, fontfamily='DejaVu Serif')

    # T_Y column
    ax.text(COL_X[2], y, row['T_Y'],
            ha='center', va='center', fontsize=8.5, fontfamily='DejaVu Serif')

    # T_M column
    ax.text(COL_X[3], y, row['T_M'],
            ha='center', va='center', fontsize=8.5, fontfamily='DejaVu Serif')

    # T_W column
    ax.text(COL_X[4], y, row['T_W'],
            ha='center', va='center', fontsize=8.5, fontfamily='DejaVu Serif')

    # T_NOM column (placed at group midpoint - simpler: just put on the row)
    if row['T_NOM']:
        ax.text(COL_X[5], y, row['T_NOM'],
                ha='center', va='center', fontsize=8.5, fontweight='bold',
                fontfamily='DejaVu Serif')

    # Group separator line after each group
    if n in group_end_rows and n < 34:
        y_line = y - 0.5
        lw = {g[1]: g[2] for g in GROUPS}.get(n, 0.9)
        if lw > 0:
            ax.plot([COL_EDGES[0], COL_EDGES[-1]], [y_line, y_line],
                    color='black', linewidth=lw)

    # Thin sub-divider within a group
    if n in SUBDIV:
        y_sub = y - 0.5
        ax.plot([COL_EDGES[0], COL_EDGES[-1]], [y_sub, y_sub],
                color='black', linewidth=0.45, linestyle='--')

# ---- Title and caption ----
ax.text(N_COLS / 2, N_ROWS + 0.4,
        'FIGURE A I-8',
        ha='center', va='bottom', fontsize=12, fontweight='bold',
        fontfamily='DejaVu Serif')

ax.text(N_COLS / 2, CAPTION_Y - 0.15,
        'Low-Frequency Portion: Spectral Model',
        ha='center', va='top', fontsize=10, fontstyle='italic',
        fontfamily='DejaVu Serif')

# ---- Nominal model note ----
ax.text(0.01, CAPTION_Y - 0.65,
        r'$\omega_n = n \times 0.3676$ rad/yr,  $n = 1 \ldots 34$',
        ha='left', va='top', fontsize=8, color='dimgray')

fig.tight_layout(rect=[0, 0.02, 1, 0.99])
out_path = os.path.join(SCRIPT_DIR, 'fig_AI8_spectral_table.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_path}")
print()
print("Done.")
