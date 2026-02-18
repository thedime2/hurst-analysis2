# -*- coding: utf-8 -*-
"""
Figure AI-4 Comb Filter Bank Spectrogram / 3D Surface

Builds a time-frequency-amplitude visualization from Hurst's 23-filter
comb bank outputs, comparable to a discrete spectrogram.

For each filter k (center fc_k):
  - Row in heatmap  = filter center frequency fc_k
  - Column in heatmap = time (week within display window)
  - Color/Height     = envelope amplitude |z_k(t)|

The overlapping filter skirts cause neighboring rows to bleed into each other,
creating smooth amplitude ridges at the spectral line positions -- exactly
as in a proper spectrogram. No normalization trick needed; the overlap
just produces a blended surface between adjacent bands.

To get a CONTINUOUS (not 23-row) surface, we interpolate linearly between
filter centers in the frequency axis. This is equivalent to assuming linear
amplitude variation between adjacent filter outputs.

Outputs:
  fig_AI4_spec_heatmap.png    -- 2D heatmaps: Ormsby vs CMW
  fig_AI4_spec_3d.png         -- 3D surface: Ormsby vs CMW
  fig_AI4_spec_overlay.png    -- Heatmap + FVT dot overlay (Ormsby, best method)
  fig_AI4_spec_difference.png -- CMW minus Ormsby amplitude difference

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figure AI-4 (Frequency vs Time)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (needed for 3d projection)
from scipy.signal import find_peaks

from utils_ai import (
    load_weekly_data,
    design_comb_bank, make_ormsby_kernels, apply_comb_bank,
    get_window, FS_WEEKLY, NW_WEEKLY,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
)
from src.time_frequency.cmw import ormsby_spec_to_cmw_params, apply_cmw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

N_FILTERS   = 23
YMIN, YMAX  = 7.4, 12.6
CLIP_FRAC   = 0.30
SMOOTH_N5   = 5

# Fine frequency grid for interpolated continuous heatmap
F_FINE_N    = 300    # number of interpolated frequency rows


# ============================================================================
# PEAK / TROUGH + MEASUREMENT UTILITIES (minimal, for FVT overlay)
# ============================================================================

def parabolic_peak(y, idx):
    if idx <= 0 or idx >= len(y) - 1: return float(idx)
    y0, y1, y2 = float(y[idx-1]), float(y[idx]), float(y[idx+1])
    denom = y0 - 2.0*y1 + y2
    if abs(denom) < 1e-14: return float(idx)
    return idx + np.clip(0.5*(y0-y2)/denom, -1.0, 1.0)

def find_peaks_sub(signal, f_center, fs, min_dist_frac=0.55):
    T_samp = 2*np.pi / f_center * fs
    min_d  = max(3, int(T_samp * min_dist_frac))
    idx, _ = find_peaks(signal, distance=min_d)
    return np.array([parabolic_peak(signal, i) for i in idx])

def find_troughs_sub(signal, f_center, fs):
    return find_peaks_sub(-signal, f_center, fs)

def clip_to_window(times, freqs, s_idx, e_idx, f_center):
    if len(times) == 0: return np.array([]), np.array([])
    mask  = (times >= s_idx) & (times < e_idx)
    t = times[mask] - s_idx
    f = freqs[mask]
    valid = ((f >= f_center*(1-CLIP_FRAC)) & (f <= f_center*(1+CLIP_FRAC)) &
             (f >= YMIN) & (f <= YMAX))
    return t[valid], f[valid]

def smooth_ma(t, f, n):
    if n <= 1 or len(f) < n: return t, f
    f_sm = np.convolve(f, np.ones(n)/n, mode='valid')
    pad  = (len(f) - len(f_sm)) // 2
    return t[pad: pad+len(f_sm)], f_sm

def measure_pttp5(z_analytic, f_center, fs, s_idx, e_idx):
    sig_real = z_analytic.real
    pk = find_peaks_sub(sig_real, f_center, fs)
    tr = find_troughs_sub(sig_real, f_center, fs)
    events = ([(t,'P') for t in pk] + [(t,'T') for t in tr])
    events.sort(key=lambda x: x[0])
    if len(events) < 2: return np.array([]), np.array([])
    times, freqs = [], []
    for k in range(len(events)-1):
        t1,_ = events[k]; t2,_ = events[k+1]
        dt = (t2-t1)/fs
        if dt > 0: freqs.append(np.pi/dt); times.append(t2)
    t_cl, f_cl = clip_to_window(np.array(times), np.array(freqs), s_idx, e_idx, f_center)
    return smooth_ma(t_cl, f_cl, SMOOTH_N5)


# ============================================================================
# LOAD DATA AND APPLY FILTER BANKS
# ============================================================================

print("Loading weekly DJIA data...")
close, dates = load_weekly_data()
s_idx, e_idx = get_window(dates)
n_weeks = e_idx - s_idx
t_axis  = np.arange(n_weeks)     # weeks 0..n_weeks-1

print(f"Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} weeks)")

specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
f_centers = np.array([s['f_center'] for s in specs])

print("\nApplying Ormsby comb bank...")
filters        = make_ormsby_kernels(specs, fs=FS_WEEKLY)
ormsby_outputs = apply_comb_bank(close, filters, fs=FS_WEEKLY)

print("Applying CMW bank...")
cmw_params = [ormsby_spec_to_cmw_params(s) for s in specs]
cmw_outputs = []
for params in cmw_params:
    out = apply_cmw(close, params['f0'], params['fwhm'], FS_WEEKLY, analytic=True)
    cmw_outputs.append(out)


# ============================================================================
# BUILD AMPLITUDE MATRICES  (N_FILTERS x n_weeks)
# ============================================================================

def build_amp_matrix(outputs, s_idx, e_idx):
    """Raw envelope amplitude for display window. Shape: (N_FILTERS, n_weeks)."""
    n = e_idx - s_idx
    Z = np.zeros((N_FILTERS, n))
    for i, out in enumerate(outputs):
        seg = out['signal'][s_idx:e_idx]
        Z[i] = np.abs(seg)
    return Z

Z_orm = build_amp_matrix(ormsby_outputs, s_idx, e_idx)
Z_cmw = build_amp_matrix(cmw_outputs,    s_idx, e_idx)

# Global peak (for consistent colour scale across Ormsby and CMW)
Z_peak_orm = np.percentile(Z_orm, 99)
Z_peak_cmw = np.percentile(Z_cmw, 99)
Z_peak = max(Z_peak_orm, Z_peak_cmw)

print(f"\nAmplitude matrix shape: {Z_orm.shape}")
print(f"  Ormsby peak (99th pct): {Z_peak_orm:.2f}")
print(f"  CMW    peak (99th pct): {Z_peak_cmw:.2f}")


# ============================================================================
# INTERPOLATED CONTINUOUS HEATMAP GRID
# ============================================================================

# Fine frequency axis (interpolated between filter centers)
f_fine = np.linspace(YMIN, YMAX, F_FINE_N)

def interpolate_amp(Z_discrete, f_centers, f_fine):
    """
    Interpolate amplitude from 23 filter centers to fine frequency grid.
    At each time step, do 1D linear interpolation over the 23 discrete rows.
    Returns shape: (F_FINE_N, n_weeks).
    """
    n = Z_discrete.shape[1]
    Z_interp = np.zeros((len(f_fine), n))
    for t in range(n):
        Z_interp[:, t] = np.interp(f_fine, f_centers, Z_discrete[:, t])
    return Z_interp

print("Interpolating amplitude grids to fine frequency axis...")
ZI_orm = interpolate_amp(Z_orm, f_centers, f_fine)
ZI_cmw = interpolate_amp(Z_cmw, f_centers, f_fine)


# ============================================================================
# FVT MEASUREMENTS (Ormsby PT+TP+5pt MA) for overlay
# ============================================================================

print("Computing FVT measurements for overlay...")
fvt_times, fvt_freqs = [], []
for i, out in enumerate(ormsby_outputs):
    fc = specs[i]['f_center']
    z  = out['signal']
    t, f = measure_pttp5(z, fc, FS_WEEKLY, s_idx, e_idx)
    fvt_times.append(t)
    fvt_freqs.append(f)

# Flatten for scatter
fvt_t_all = np.concatenate(fvt_times) if fvt_times else np.array([])
fvt_f_all = np.concatenate(fvt_freqs) if fvt_freqs else np.array([])


# ============================================================================
# COLOUR MAP HELPERS
# ============================================================================

def amp_cmap():
    """Sequential colormap good for amplitude: black -> blue -> green -> yellow."""
    return 'viridis'

def amp_cmap_hot():
    return 'hot'

def sqrt_norm(Z, vmax=None):
    """Square-root normalised array (stretches low amplitudes)."""
    if vmax is None:
        vmax = np.max(Z)
    return np.clip(np.sqrt(Z / (vmax + 1e-12)), 0, 1)


# ============================================================================
# FIGURE 1: 2D HEATMAPS -- Ormsby vs CMW (interpolated, sqrt-scaled)
# ============================================================================

print("\nGenerating Figure 1: 2D heatmaps...")

fig1, axes1 = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'wspace': 0.08})

extent = [0, n_weeks, YMIN, YMAX]
norm_kwargs = dict(vmin=0, vmax=1, cmap=amp_cmap(), aspect='auto',
                   origin='lower', extent=extent, interpolation='bilinear')

im0 = axes1[0].imshow(sqrt_norm(ZI_orm, Z_peak), **norm_kwargs)
axes1[0].set_title(
    f'Ormsby FIR Comb Bank  |  sqrt-scaled amplitude\n'
    f'23 filters, passband=0.2 rad/yr, skirt=0.3 rad/yr',
    fontsize=10, fontweight='bold')
axes1[0].set_xlabel('Weeks', fontsize=10)
axes1[0].set_ylabel('Radians/Year', fontsize=10)
axes1[0].set_xticks(np.arange(0, n_weeks+1, 25))
axes1[0].set_yticks([8, 9, 10, 11, 12])
axes1[0].text(0.01, 0.99, DATE_DISPLAY_START, transform=axes1[0].transAxes,
              fontsize=7, va='top', color='white', alpha=0.8)

im1 = axes1[1].imshow(sqrt_norm(ZI_cmw, Z_peak), **norm_kwargs)
axes1[1].set_title(
    f'CMW Gaussian Bank  |  sqrt-scaled amplitude\n'
    f'23 filters, FWHM=0.50 rad/yr (Gaussian spectral response)',
    fontsize=10, fontweight='bold')
axes1[1].set_xlabel('Weeks', fontsize=10)
axes1[1].set_xticks(np.arange(0, n_weeks+1, 25))
axes1[1].set_yticks([8, 9, 10, 11, 12])
axes1[1].text(0.99, 0.99, DATE_DISPLAY_END, transform=axes1[1].transAxes,
              fontsize=7, va='top', ha='right', color='white', alpha=0.8)

# Add horizontal lines for filter centers
for ax in axes1:
    for fc in f_centers:
        ax.axhline(fc, color='white', linewidth=0.25, alpha=0.3, linestyle=':')

plt.colorbar(im0, ax=axes1[0], fraction=0.03, label='Normalised amplitude (sqrt scaled)')
plt.colorbar(im1, ax=axes1[1], fraction=0.03, label='Normalised amplitude (sqrt scaled)')

fig1.suptitle(
    'Comb Filter Bank Spectrogram  |  DJIA Weekly  |  '
    f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}\n'
    'Color = filter envelope amplitude interpolated between 23 centers',
    fontsize=11, fontweight='bold')

out1 = os.path.join(SCRIPT_DIR, 'fig_AI4_spec_heatmap.png')
fig1.savefig(out1, dpi=140, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved: {out1}")


# ============================================================================
# FIGURE 2: 3D SURFACE -- Ormsby vs CMW
# ============================================================================

print("Generating Figure 2: 3D surfaces...")

# Subsample time for 3D (every 2 weeks to keep surface manageable)
t_step = 2
t_sub  = t_axis[::t_step]
f_sub  = f_fine[::3]             # every 3rd frequency row

ZI_orm_sub = ZI_orm[::3, ::t_step]
ZI_cmw_sub = ZI_cmw[::3, ::t_step]

T_grid, F_grid = np.meshgrid(t_sub, f_sub)   # shape (F_FINE_N//3, n_weeks//2)

# Shared colour scaling for both surfaces
z_max3d = np.percentile(np.maximum(ZI_orm_sub, ZI_cmw_sub), 99)
Z_orm_n  = np.clip(ZI_orm_sub / (z_max3d + 1e-12), 0, 1)
Z_cmw_n  = np.clip(ZI_cmw_sub / (z_max3d + 1e-12), 0, 1)

cmap3d = plt.colormaps['plasma']

fig2 = plt.figure(figsize=(20, 9))
fig2.subplots_adjust(wspace=0.01)

for col, (label, Z_n) in enumerate([('Ormsby FIR', Z_orm_n), ('CMW Gaussian', Z_cmw_n)]):
    ax = fig2.add_subplot(1, 2, col+1, projection='3d')

    surf = ax.plot_surface(T_grid, F_grid, Z_n,
                           facecolors=cmap3d(Z_n),
                           linewidth=0, antialiased=True, shade=True, alpha=0.9)

    ax.set_xlabel('Weeks', fontsize=9, labelpad=6)
    ax.set_ylabel('Frequency\n(rad/yr)', fontsize=9, labelpad=6)
    ax.set_zlabel('Norm. amplitude', fontsize=9, labelpad=6)
    ax.set_xlim(0, n_weeks)
    ax.set_ylim(YMIN, YMAX)
    ax.set_zlim(0, 1)
    ax.set_xticks(np.arange(0, n_weeks+1, 50))
    ax.set_yticks([8, 9, 10, 11, 12])
    ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.view_init(elev=28, azim=-55)
    ax.set_title(
        f'{label}  |  Time-Frequency-Amplitude\n'
        f'{"passband=0.2, skirt=0.3 rad/yr" if "Ormsby" in label else "FWHM=0.50 rad/yr Gaussian"}',
        fontsize=10, fontweight='bold', pad=12)

    # Draw spectral line positions as thin vertical planes (amplitude ridges)
    # These are just visual guides at the nominal Hurst harmonic positions
    hurst_lines = np.arange(1, 33) * 0.3676    # n*0.3676 rad/yr
    hurst_lines = hurst_lines[(hurst_lines >= YMIN) & (hurst_lines <= YMAX)]
    for hl in hurst_lines:
        ax.plot([0, n_weeks], [hl, hl], [0, 0],
                color='cyan', linewidth=0.6, alpha=0.4, linestyle='--')

fig2.suptitle(
    '3D Comb Filter Bank Spectrogram  |  DJIA Weekly  |  '
    f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}\n'
    'Ridges = spectral lines; dashed cyan lines = Hurst nominal harmonics (n x 0.3676 rad/yr)',
    fontsize=11, fontweight='bold')

out2 = os.path.join(SCRIPT_DIR, 'fig_AI4_spec_3d.png')
fig2.savefig(out2, dpi=130, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: {out2}")


# ============================================================================
# FIGURE 3: HEATMAP + FVT DOT OVERLAY (Ormsby)
# Shows WHERE in the amplitude pattern each frequency measurement falls
# ============================================================================

print("Generating Figure 3: heatmap + FVT overlay...")

fig3, ax3 = plt.subplots(figsize=(14, 7))

im3 = ax3.imshow(sqrt_norm(ZI_orm, Z_peak_orm),
                 vmin=0, vmax=1, cmap='magma', aspect='auto',
                 origin='lower', extent=extent, interpolation='bilinear')

# Overlay FVT dots
ax3.scatter(fvt_t_all, fvt_f_all,
            s=6, c='cyan', alpha=0.75, linewidths=0, zorder=5,
            label='FVT measurements (Ormsby PT+TP+5pt MA)')

# Filter center guide lines
for fc in f_centers:
    ax3.axhline(fc, color='white', linewidth=0.25, alpha=0.25, linestyle=':')

# Hurst nominal harmonics
hurst_in_range = [n*0.3676 for n in range(1, 33)
                  if YMIN <= n*0.3676 <= YMAX]
for hl in hurst_in_range:
    ax3.axhline(hl, color='lime', linewidth=0.5, alpha=0.5, linestyle='--')

ax3.set_xlabel('Weeks', fontsize=11)
ax3.set_ylabel('Frequency (rad/yr)', fontsize=11)
ax3.set_xlim(0, n_weeks)
ax3.set_ylim(YMIN, YMAX)
ax3.set_xticks(np.arange(0, n_weeks+1, 25))
ax3.set_yticks([8, 9, 10, 11, 12])
ax3.grid(True, axis='x', alpha=0.2, color='white', linewidth=0.5)
ax3.legend(fontsize=9, loc='upper right')

plt.colorbar(im3, ax=ax3, fraction=0.025, label='Filter envelope amplitude (sqrt-scaled)')

ax3.set_title(
    f'Comb Filter Spectrogram + FVT Measurements  |  Ormsby  |  '
    f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}\n'
    'Color = |z(t)| (envelope amplitude, sqrt-scaled)  |  '
    'Cyan dots = frequency measurements  |  Green dashes = Hurst nominal harmonics',
    fontsize=10, fontweight='bold')

out3 = os.path.join(SCRIPT_DIR, 'fig_AI4_spec_overlay.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"  Saved: {out3}")


# ============================================================================
# FIGURE 4: CMW MINUS ORMSBY AMPLITUDE DIFFERENCE
# Shows where CMW captures more or less energy than Ormsby
# ============================================================================

print("Generating Figure 4: CMW - Ormsby amplitude difference...")

# Normalize each bank independently to its own peak
ZI_orm_n = ZI_orm / (Z_peak_orm + 1e-12)
ZI_cmw_n = ZI_cmw / (Z_peak_cmw + 1e-12)
Z_diff    = ZI_cmw_n - ZI_orm_n   # positive = CMW has more energy here

fig4, ax4 = plt.subplots(figsize=(14, 6))

divnorm = mcolors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
im4 = ax4.imshow(Z_diff,
                 norm=divnorm, cmap='RdBu_r', aspect='auto',
                 origin='lower', extent=extent, interpolation='bilinear')

for fc in f_centers:
    ax4.axhline(fc, color='black', linewidth=0.2, alpha=0.3, linestyle=':')

for hl in hurst_in_range:
    ax4.axhline(hl, color='black', linewidth=0.5, alpha=0.5, linestyle='--')

ax4.set_xlabel('Weeks', fontsize=11)
ax4.set_ylabel('Frequency (rad/yr)', fontsize=11)
ax4.set_xlim(0, n_weeks)
ax4.set_ylim(YMIN, YMAX)
ax4.set_xticks(np.arange(0, n_weeks+1, 25))
ax4.set_yticks([8, 9, 10, 11, 12])
ax4.grid(True, axis='x', alpha=0.25, color='black', linewidth=0.5)

plt.colorbar(im4, ax=ax4, fraction=0.025, label='CMW - Ormsby normalised amplitude')

ax4.set_title(
    f'CMW minus Ormsby Amplitude  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}\n'
    'Red = CMW has MORE energy  |  Blue = Ormsby has MORE energy\n'
    'Dashed black = Hurst nominal harmonics  |  Each bank normalised to own peak',
    fontsize=10, fontweight='bold')

out4 = os.path.join(SCRIPT_DIR, 'fig_AI4_spec_difference.png')
fig4.savefig(out4, dpi=140, bbox_inches='tight')
plt.close(fig4)
print(f"  Saved: {out4}")


# ============================================================================
# FIGURE 5: FOUR-PANEL -- Linear | Sqrt | Log | Discrete-row heatmap (Ormsby)
# Compares scaling choices for the same data
# ============================================================================

print("Generating Figure 5: scaling comparison for Ormsby...")

fig5, axes5 = plt.subplots(2, 2, figsize=(20, 12), gridspec_kw={'wspace': 0.1, 'hspace': 0.3})

# --- Linear scaling ---
ax = axes5[0, 0]
im = ax.imshow(ZI_orm / (Z_peak_orm + 1e-12), vmin=0, vmax=1,
               cmap='hot', aspect='auto', origin='lower',
               extent=extent, interpolation='bilinear')
ax.set_title('Ormsby  |  Linear amplitude', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.03)

# --- Sqrt scaling ---
ax = axes5[0, 1]
im = ax.imshow(sqrt_norm(ZI_orm, Z_peak_orm), vmin=0, vmax=1,
               cmap='hot', aspect='auto', origin='lower',
               extent=extent, interpolation='bilinear')
ax.set_title('Ormsby  |  Sqrt amplitude (stretches low values)', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.03)

# --- Log scaling ---
ax = axes5[1, 0]
eps = Z_peak_orm * 0.01    # 1% noise floor
Z_log = np.log10(ZI_orm / Z_peak_orm + eps / Z_peak_orm + 1e-9)
Z_log = (Z_log - Z_log.min()) / (Z_log.max() - Z_log.min() + 1e-12)
im = ax.imshow(Z_log, vmin=0, vmax=1,
               cmap='hot', aspect='auto', origin='lower',
               extent=extent, interpolation='bilinear')
ax.set_title('Ormsby  |  Log amplitude (suppresses weak filters)', fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.03)

# --- Discrete rows (no interpolation) ---
ax = axes5[1, 1]
im = ax.imshow(sqrt_norm(Z_orm, Z_peak_orm), vmin=0, vmax=1,
               cmap='hot', aspect='auto', origin='lower',
               extent=[0, n_weeks, 0.5, N_FILTERS + 0.5],
               interpolation='nearest')
ax.set_title('Ormsby  |  23 Discrete filter rows  (no freq interpolation)', fontsize=10, fontweight='bold')
ax.set_ylabel('Filter index (1=lowest fc)', fontsize=9)
ax.set_yticks(range(1, N_FILTERS+1, 2))
plt.colorbar(im, ax=ax, fraction=0.03)

for ax in axes5.flat:
    ax.set_xlabel('Weeks', fontsize=9)
    ax.set_xticks(np.arange(0, n_weeks+1, 25))

fig5.suptitle(
    'Amplitude Scaling Comparison  |  Ormsby Comb Bank  |  '
    f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}',
    fontsize=11, fontweight='bold')

out5 = os.path.join(SCRIPT_DIR, 'fig_AI4_spec_scaling.png')
fig5.savefig(out5, dpi=120, bbox_inches='tight')
plt.close(fig5)
print(f"  Saved: {out5}")


# ============================================================================
# PRINT SUMMARY
# ============================================================================

print()
print("=" * 60)
print("Spectrogram Summary")
print("=" * 60)
print(f"  Display window : {n_weeks} weeks ({DATE_DISPLAY_START} to {DATE_DISPLAY_END})")
print(f"  Filters        : {N_FILTERS} x {f_centers[0]:.1f}..{f_centers[-1]:.1f} rad/yr")
print(f"  Freq interp    : {F_FINE_N} rows (linear between filter centers)")
print()
print("Filter overlap analysis:")
print(f"  Ormsby passband: 0.2 rad/yr  |  skirt: 0.3 rad/yr each side")
print(f"  Filter spacing : 0.2 rad/yr  |  adjacent skirt overlap: 0.2 rad/yr")
print(f"  Overlap type   : TOTAL skirt overlap (adjacent filters share full skirt)")
print(f"  CMW FWHM       : 0.50 rad/yr  sigma={cmw_params[0]['sigma_f']:.3f} rad/yr")
print(f"  CMW overlap at : +/-sigma from center = +/-{cmw_params[0]['sigma_f']:.3f} rad/yr")
print()
print("Output files:")
for out in [out1, out2, out3, out4, out5]:
    print(f"  {os.path.basename(out)}")
print()
print("Done.")
