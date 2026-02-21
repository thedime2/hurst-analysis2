# -*- coding: utf-8 -*-
"""
Figure AI-4 v3: Decimated Output (spacing parameter) Experiment

Tests whether the built-in `spacing` parameter of apply_filter_bank()
reproduces Hurst's FVT point density and visual pattern.

Background
----------
apply_filter_bank(signal, filters, spacing=N) decimates the input to every
Nth sample, designs the kernel for fs/N, and places output back into a
full-length array with NaNs in the gaps. For analytic (complex) mode this
works correctly — the bandpass centre frequencies (7.6-12.0 rad/yr) are well
below the decimated Nyquist (π × 52/N rad/yr).

Why this may reproduce Hurst
-----------------------------
With spacing=7 (weekly data):
  - Filter kernels: ~3501//7 = 500 taps (same time-span, fewer weights)
  - Output: one computed value every 7 weeks, NaN elsewhere
  - FC-1 (fc=7.6 r/y, period≈43 wk): ~43/7 ≈ 6 samples/cycle
    → peak detection yields ~269/43 ≈ 6 peaks in the display window
  - FC-10 (fc=9.4 r/y, period≈34 wk): ~34/7 ≈ 5 samples/cycle → ~8 peaks
  - This matches Hurst's observed ~7 measurements per filter!

Measurement approach for spaced output
---------------------------------------
1. Extract non-NaN indices from the full-length output array
2. Compact these into a dense sub-array
3. Run peak/trough detection on the compact sub-array
4. Map peak sample positions back to full-length (week) coordinates
5. Measure P→T half-period: frequency = π / Δt_yr, placed at second event

Spacings tested: 1 (baseline), 4, 7, 14

Figures produced:
  fig_AI4_v3_spacing_grid.png       — 4-panel: one per spacing value
  fig_AI4_v3_spacing_compare.png    — all spacings overlaid (with reference)
  fig_AI4_v3_spacing_density.png    — point count per filter vs spacing
  fig_AI4_v3_spacing_waveforms.png  — compact waveforms for 4 filters, spacing=7

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figure AI-4, p.194
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks

from src.filters import design_hurst_comb_bank, create_filter_kernels, apply_filter_bank

from utils_ai import (
    load_weekly_data,
    get_window, FS_WEEKLY, NW_WEEKLY,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_IMAGE  = os.path.join(SCRIPT_DIR, '../../references/appendix_a/figure_AI4_v2.png')

YMIN, YMAX  = 7.5, 12.5
CLIP_FRAC   = 0.30
X_MAX_LABEL = 275

# Decimation factors to test
# spacing=14 exceeds Nyquist (π×52/14=11.65 r/y < f4_max=12.4 r/y) → use 10
SPACINGS    = [1, 4, 7, 10]
SPACING_LBL = ['spacing=1 (no decim)', 'spacing=4 (~monthly)',
               'spacing=7 (~7wk step)', 'spacing=10 (~10wk step)']


# ============================================================================
# MEASUREMENT ON SPACED (NaN-gapped) OUTPUT
# ============================================================================

def parabolic_peak(y, idx):
    """3-point parabolic sub-sample peak (on compact non-NaN array)."""
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    y0, y1, y2 = float(y[idx-1]), float(y[idx]), float(y[idx+1])
    d = y0 - 2.0*y1 + y2
    if abs(d) < 1e-14:
        return float(idx)
    return idx + np.clip(0.5*(y0-y2)/d, -1.0, 1.0)


def measure_pp_spaced(analytic_full, f_center, fs, s_idx, e_idx, spacing):
    """
    Peak-to-peak full-period measurement on spaced output.
    One measurement per full cycle → matches Hurst's ~7 pts/filter density.
    Point placed at the SECOND PEAK.
    """
    seg = analytic_full[s_idx:e_idx].real
    valid_mask = ~np.isnan(seg)
    valid_idx  = np.where(valid_mask)[0]
    if len(valid_idx) < 4:
        return np.array([]), np.array([])

    compact_real = seg[valid_idx]
    fs_compact   = fs / spacing
    T_compact    = 2 * np.pi / f_center * fs_compact
    min_d        = max(2, int(T_compact * 0.55))

    peak_ci, _ = find_peaks(compact_real, distance=min_d)
    if len(peak_ci) < 2:
        return np.array([]), np.array([])

    peak_cf = np.array([parabolic_peak(compact_real, i) for i in peak_ci])

    def compact_to_weeks(cf):
        i0   = np.clip(np.floor(cf).astype(int), 0, len(valid_idx)-1)
        i1   = np.clip(i0 + 1, 0, len(valid_idx)-1)
        frac = cf - i0
        return valid_idx[i0] + frac * (valid_idx[i1] - valid_idx[i0])

    peak_wk = compact_to_weeks(peak_cf)
    if len(peak_wk) < 2:
        return np.array([]), np.array([])

    dt_yr = np.diff(peak_wk) / fs           # Δt in years (full sample basis)
    freqs = 2 * np.pi / np.where(dt_yr > 0, dt_yr, np.nan)
    times = peak_wk[1:]                      # point at SECOND peak

    t_arr = np.array(times, dtype=float)
    f_arr = np.array(freqs, dtype=float)
    ok    = np.isfinite(f_arr)
    t_arr, f_arr = t_arr[ok], f_arr[ok]

    if len(t_arr) == 0:
        return t_arr, f_arr
    valid = ((f_arr >= f_center*(1-CLIP_FRAC)) & (f_arr <= f_center*(1+CLIP_FRAC)) &
             (f_arr >= YMIN) & (f_arr <= YMAX))
    return t_arr[valid], f_arr[valid]


def measure_pt_spaced(analytic_full, f_center, fs, s_idx, e_idx, spacing):
    """
    Measure PT half-period frequencies from a spaced (NaN-gapped) filter output.

    Steps:
      1. Slice to display window
      2. Find non-NaN positions → compact sub-array (effective fs = fs/spacing)
      3. Detect peaks and troughs on compact real part
      4. Map compact indices back to full-window week coordinates
      5. P→T or T→P: frequency = π / Δt_yr, point at second event (in weeks)

    Returns (t_weeks, f_radyr) arrays.
    """
    seg = analytic_full[s_idx:e_idx]          # slice to display window
    real_seg = seg.real

    # 1. Non-NaN mask
    valid_mask = ~np.isnan(real_seg)
    valid_idx  = np.where(valid_mask)[0]      # positions within display window

    if len(valid_idx) < 4:
        return np.array([]), np.array([])

    compact_real = real_seg[valid_idx]         # dense non-NaN values
    # Compact sample spacing in years (one compact sample = spacing full samples)
    fs_compact = fs / spacing                  # samples per year for compact

    # 2. Peak/trough detection on compact array
    T_compact = 2 * np.pi / f_center * fs_compact   # expected period in compact samples
    min_d = max(2, int(T_compact * 0.45))

    peak_ci,  _ = find_peaks( compact_real, distance=min_d)
    trough_ci, _ = find_peaks(-compact_real, distance=min_d)

    # Sub-sample refinement
    peak_cf  = np.array([parabolic_peak(compact_real, i) for i in peak_ci])
    trough_cf = np.array([parabolic_peak(-compact_real, i) for i in trough_ci])

    # Map compact fractional indices back to full-window week positions
    # compact_cf → week position = valid_idx interpolated
    def compact_to_weeks(cf):
        """Map fractional compact index to weeks within display window."""
        # Linear interpolation between neighbouring valid_idx entries
        i0 = np.clip(np.floor(cf).astype(int), 0, len(valid_idx)-1)
        i1 = np.clip(i0 + 1, 0, len(valid_idx)-1)
        frac = cf - i0
        return valid_idx[i0] + frac * (valid_idx[i1] - valid_idx[i0])

    peak_wk  = compact_to_weeks(peak_cf)   if len(peak_cf) > 0  else np.array([])
    trough_wk = compact_to_weeks(trough_cf) if len(trough_cf) > 0 else np.array([])

    # 3. Merge and sort chronologically
    events = ([(t, 'P') for t in peak_wk] +
              [(t, 'T') for t in trough_wk])
    events.sort(key=lambda x: x[0])

    if len(events) < 2:
        return np.array([]), np.array([])

    times, freqs = [], []
    for k in range(len(events) - 1):
        t1, e1 = events[k]
        t2, e2 = events[k+1]
        if e1 == e2:
            continue              # same type → skip (missed event)
        dt_yr = (t2 - t1) / fs   # Δt in years (full-sample basis)
        if dt_yr <= 0:
            continue
        freq = np.pi / dt_yr     # half-period → frequency
        times.append(t2)         # point at second event (in weeks)
        freqs.append(freq)

    t_arr = np.array(times, dtype=float)
    f_arr = np.array(freqs, dtype=float)

    # 4. Clip to ±CLIP_FRAC of filter centre and to YMIN..YMAX
    if len(t_arr) == 0:
        return t_arr, f_arr
    valid = ((f_arr >= f_center*(1-CLIP_FRAC)) & (f_arr <= f_center*(1+CLIP_FRAC)) &
             (f_arr >= YMIN) & (f_arr <= YMAX))
    return t_arr[valid], f_arr[valid]


# ============================================================================
# LOAD DATA AND DESIGN FILTER BANK
# ============================================================================

print("Loading weekly DJIA data...")
close, dates_dt = load_weekly_data()
s_idx, e_idx = get_window(dates_dt)
n_weeks = e_idx - s_idx
print(f"Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} weeks)")

# Design once (nw=3501 for spacing=1; apply_filter_bank auto-scales for spacing>1)
specs_base  = design_hurst_comb_bank(
    n_filters=23, w1_start=7.2, w_step=0.2,
    passband_width=0.2, skirt_width=0.3,
    nw=NW_WEEKLY, fs=FS_WEEKLY
)
filters_base = create_filter_kernels(specs_base, fs=FS_WEEKLY,
                                      filter_type='modulate', analytic=True)

f_centers = np.array([s['f_center'] for s in specs_base])

# Colour ramp
_cmap  = plt.colormaps['coolwarm']
COLORS = [_cmap(i / 22) for i in range(23)]


# ============================================================================
# APPLY AND MEASURE FOR EACH SPACING
# ============================================================================

all_results_pt = {}   # spacing -> list of (t_wk, f_radyr, fc) per filter  [PT]
all_results_pp = {}   # spacing -> list of (t_wk, f_radyr, fc) per filter  [PP]

for sp in SPACINGS:
    print(f"\nspacing={sp}")

    result  = apply_filter_bank(close, filters_base, fs=FS_WEEKLY,
                                 mode='reflect', spacing=sp, startidx=0,
                                 interp='none')
    outputs = result['filter_outputs']

    meas_pt, meas_pp = [], []
    for i, out in enumerate(outputs):
        fc = specs_base[i]['f_center']
        z  = out['signal']        # complex, full-length, NaNs at non-computed positions
        t_pt, f_pt = measure_pt_spaced(z, fc, FS_WEEKLY, s_idx, e_idx, sp)
        t_pp, f_pp = measure_pp_spaced(z, fc, FS_WEEKLY, s_idx, e_idx, sp)
        meas_pt.append((t_pt, f_pt, fc))
        meas_pp.append((t_pp, f_pp, fc))

    n_pt = sum(len(t) for t, f, _ in meas_pt)
    n_pp = sum(len(t) for t, f, _ in meas_pp)
    print(f"  PT measurements: {n_pt}  (avg {n_pt/23:.1f}/filter)")
    print(f"  PP measurements: {n_pp}  (avg {n_pp/23:.1f}/filter)")
    all_results_pt[sp] = meas_pt
    all_results_pp[sp] = meas_pp

# Convenience alias for backward compat in figure code below
all_results = all_results_pt


# ============================================================================
# PLOT HELPERS
# ============================================================================

def plot_panel(ax, meas, title, connect=True, lw=0.75, dot_size=2.0, alpha=0.85):
    for i, (t, f, fc) in enumerate(meas):
        if len(t) == 0:
            continue
        c = COLORS[i]
        if connect:
            ax.plot(t, f, '-', color=c, linewidth=lw, alpha=alpha, zorder=3)
        ax.plot(t, f, 'o', color=c, markersize=dot_size, alpha=alpha, zorder=4)
        ax.text(t[0]  - 2, f[0],  str(i+1), fontsize=5.0, color=c,
                ha='right', va='center', zorder=5)
        ax.text(t[-1] + 2, f[-1], str(i+1), fontsize=5.0, color=c,
                ha='left',  va='center', zorder=5)
    for ref in range(8, 13):
        ax.axhline(ref, color='#BBBBBB', linewidth=0.5, zorder=1)
    for wk in range(0, n_weeks + 50, 25):
        ax.axvline(wk, color='#DDDDDD', linewidth=0.35, zorder=1)
    ax.set_xlim(0, max(n_weeks, X_MAX_LABEL))
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel('Weeks', fontsize=8)
    ax.set_ylabel('rad/yr', fontsize=8)
    ax.set_xticks(range(0, X_MAX_LABEL + 1, 50))
    ax.set_yticks([8, 9, 10, 11, 12])
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8, fontweight='bold')


# ============================================================================
# FIGURE 1: 4-PANEL (one per spacing)
# ============================================================================

print("\nGenerating Figure 1: 4-panel spacing comparison...")

fig1, axes1 = plt.subplots(2, 2, figsize=(22, 16),
                            gridspec_kw={'wspace': 0.15, 'hspace': 0.35})

for ax, sp, lbl in zip(axes1.flat, SPACINGS, SPACING_LBL):
    meas = all_results[sp]
    n_pts = sum(len(t) for t, f, _ in meas)
    plot_panel(ax, meas,
               f'{lbl}  |  {n_pts} pts  ({n_pts/23:.1f} avg/filter)\n'
               f'PT measurements, point at 2nd event, no smoothing',
               connect=True, dot_size=1.8)

fig1.suptitle(
    f'AI-4  |  Decimated-Output Spacing Comparison\n'
    f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}  |  Ormsby FIR analytic  |  '
    f'apply_filter_bank(spacing=N, interp="none")',
    fontsize=11, fontweight='bold'
)
out1 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_spacing_grid.png')
fig1.savefig(out1, dpi=130, bbox_inches='tight')
plt.close(fig1)
print(f"  Saved: {out1}")


# ============================================================================
# FIGURE 2: SPACING=7 WITH REFERENCE UNDERLAY
# ============================================================================

print("Generating Figure 2: spacing=7 with reference underlay...")

fig2, axes2 = plt.subplots(1, 2, figsize=(22, 10),
                            gridspec_kw={'wspace': 0.06})

meas7_pt = all_results_pt[7]
meas7_pp = all_results_pp[7]
meas7    = meas7_pp          # use PP for reference overlay (matches Hurst density)
n7       = sum(len(t) for t, f, _ in meas7)

# Left: our spacing=7 result
plot_panel(axes2[0], meas7,
           f'spacing=7  |  {n7} pts ({n7/23:.1f} avg/filter)\n'
           f'PT measurements  |  No smoothing',
           connect=True, dot_size=2.0)
axes2[0].set_xticks(range(0, X_MAX_LABEL + 1, 25))
axes2[0].set_xticklabels([str(x) if x <= 250 else '' for x in range(0, X_MAX_LABEL+1, 25)],
                          fontsize=7)

# Right: reference underlay + our result
ax_r = axes2[1]
if os.path.exists(REF_IMAGE):
    ref_img = mpimg.imread(REF_IMAGE)
    ax_r.imshow(ref_img, extent=[0, X_MAX_LABEL, YMIN, YMAX],
                aspect='auto', alpha=0.50, zorder=1)
plot_panel(ax_r, meas7,
           f'Reference Underlay + spacing=7 Overlay',
           connect=True, dot_size=2.0, alpha=0.88)
ax_r.set_xticks(range(0, X_MAX_LABEL + 1, 25))
ax_r.set_xticklabels([str(x) if x <= 250 else '' for x in range(0, X_MAX_LABEL+1, 25)],
                      fontsize=7)

n7_pt = sum(len(t) for t, f, _ in meas7_pt)
fig2.suptitle(
    f'AI-4  |  spacing=7 (decimated output, NaN gaps)  |  PP measurement (1/cycle)\n'
    f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}  |  Taps: {NW_WEEKLY}//7={NW_WEEKLY//7}  |  '
    f'PP: {n7} pts ({n7/23:.1f}/f)  PT: {n7_pt} pts ({n7_pt/23:.1f}/f)  '
    f'|  Reference at 50% opacity (right)',
    fontsize=10, fontweight='bold'
)
out2 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_spacing_s7_overlay.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: {out2}")


# ============================================================================
# FIGURE 3: POINT DENSITY DIAGNOSTIC
# ============================================================================

print("Generating Figure 3: point density per filter...")

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6),
                            gridspec_kw={'wspace': 0.25})

ax_l, ax_r2 = axes3

# Per-filter point counts for each spacing (solid=PP, dashed=PT)
for sp, lbl, clr in zip(SPACINGS, SPACING_LBL,
                         ['steelblue', 'forestgreen', 'firebrick', 'darkorange']):
    meas_pp = all_results_pp[sp]
    meas_pt = all_results_pt[sp]
    counts_pp = [len(t) for t, f, _ in meas_pp]
    counts_pt = [len(t) for t, f, _ in meas_pt]
    ax_l.plot(f_centers, counts_pp, '-o', color=clr, linewidth=1.2, markersize=3,
              label=f'spacing={sp} PP', alpha=0.85)
    ax_l.plot(f_centers, counts_pt, '--', color=clr, linewidth=0.8,
              label=f'spacing={sp} PT', alpha=0.55)

ax_l.axhline(7, color='firebrick', linestyle=':', linewidth=1.5,
             label='Hurst target (~7 pts/filter)')
ax_l.set_xlabel('Filter centre (rad/yr)', fontsize=10)
ax_l.set_ylabel('Measurement count', fontsize=10)
ax_l.set_xticks(f_centers[::2])
ax_l.set_xticklabels([f'{fc:.1f}' for fc in f_centers[::2]], fontsize=7, rotation=45)
ax_l.legend(fontsize=8)
ax_l.grid(True, alpha=0.25)
ax_l.set_title('Measurements per Filter vs Spacing', fontsize=10, fontweight='bold')

# Total count bar chart — grouped: PP (solid) and PT (hatched) side by side
totals_pp = [sum(len(t) for t, f, _ in all_results_pp[sp]) for sp in SPACINGS]
totals_pt = [sum(len(t) for t, f, _ in all_results_pt[sp]) for sp in SPACINGS]
bar_x     = np.arange(len(SPACINGS))
clrs      = ['steelblue', 'forestgreen', 'firebrick', 'darkorange']
w         = 0.38
bars_pp = ax_r2.bar(bar_x - w/2, totals_pp, width=w,
                    color=clrs, alpha=0.90, label='PP (full cycle)')
bars_pt = ax_r2.bar(bar_x + w/2, totals_pt, width=w,
                    color=clrs, alpha=0.45, hatch='//', label='PT (half cycle)')
ax_r2.axhline(7*23, color='black', linestyle=':', linewidth=1.5,
              label=f'Hurst target (7 x 23 = {7*23})')
ax_r2.set_xticks(bar_x)
ax_r2.set_xticklabels([f'spacing={sp}' for sp in SPACINGS], fontsize=9)
ax_r2.set_ylabel('Total measurement points', fontsize=10)
ax_r2.legend(fontsize=8)
ax_r2.grid(True, axis='y', alpha=0.3)
ax_r2.set_title('Total Measurements vs Spacing  (PP solid / PT hatched)',
                fontsize=10, fontweight='bold')

for bar, total in zip(bars_pp, totals_pp):
    ax_r2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{total}\n({total/23:.1f}/f)', ha='center', va='bottom', fontsize=7.5)

fig3.suptitle('AI-4 Point Density: Decimated Output (spacing) vs Baseline',
              fontsize=11, fontweight='bold')
out3 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_spacing_density.png')
fig3.savefig(out3, dpi=130, bbox_inches='tight')
plt.close(fig3)
print(f"  Saved: {out3}")


# ============================================================================
# FIGURE 4: WAVEFORMS FOR 4 FILTERS (spacing=1 vs spacing=7)
# ============================================================================

print("Generating Figure 4: compact waveforms for spacing=1 vs spacing=7...")

SHOW_FILTERS = [0, 4, 9, 15]   # FC-1, FC-5, FC-10, FC-16 (indices)

# Reapply for spacing=1 and spacing=7 (already have all_results, but need waveforms)
result1 = apply_filter_bank(close, filters_base, fs=FS_WEEKLY,
                             mode='reflect', spacing=1, interp='none')
result7 = apply_filter_bank(close, filters_base, fs=FS_WEEKLY,
                             mode='reflect', spacing=7, interp='none')

fig4, axes4 = plt.subplots(len(SHOW_FILTERS), 2, figsize=(20, 14),
                            gridspec_kw={'hspace': 0.45, 'wspace': 0.15},
                            sharey='row')

weeks = np.arange(n_weeks)

for row, fidx in enumerate(SHOW_FILTERS):
    fc   = specs_base[fidx]['f_center']
    T_wk = 2 * np.pi / fc * FS_WEEKLY

    for col, (result, sp, clr) in enumerate([(result1, 1, 'steelblue'),
                                              (result7, 7, 'firebrick')]):
        ax  = axes4[row, col]
        out = result['filter_outputs'][fidx]
        sig = out['signal'][s_idx:e_idx].real

        # Replace NaN with 0 for plotting (NaN shows as gap in line plot)
        sig_plot = np.where(np.isnan(sig), np.nan, sig)
        ax.plot(weeks, sig_plot, color=clr, linewidth=0.6, alpha=0.85)

        # Mark computed positions (non-NaN) as dots
        valid = ~np.isnan(sig)
        ax.plot(weeks[valid], sig[valid], 'o', color=clr, markersize=2.0, alpha=0.5)

        # Mark peaks
        t_meas, f_meas = all_results[sp][fidx][:2]
        n_pts_f = len(t_meas)
        ax.axhline(0, color='gray', linewidth=0.4)

        non_nan_count = np.sum(~np.isnan(sig))
        ax.set_title(
            f'FC-{fidx+1}  fc={fc:.1f}r/y  T={T_wk:.0f}wk  |  '
            f'spacing={sp}  |  {non_nan_count} computed pts  |  {n_pts_f} FVT meas.',
            fontsize=8, fontweight='bold'
        )
        ax.set_xlabel('Weeks' if row == len(SHOW_FILTERS)-1 else '')
        ax.set_ylabel('Amplitude', fontsize=7)
        ax.grid(True, axis='x', alpha=0.2)
        ax.set_xlim(0, n_weeks)
        ax.tick_params(labelsize=7)

fig4.suptitle(
    f'Waveforms: spacing=1 (left, dense) vs spacing=7 (right, NaN-gapped)\n'
    f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}  |  Dots = computed positions',
    fontsize=11, fontweight='bold'
)
out4 = os.path.join(SCRIPT_DIR, 'fig_AI4_v3_spacing_waveforms.png')
fig4.savefig(out4, dpi=130, bbox_inches='tight')
plt.close(fig4)
print(f"  Saved: {out4}")


# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 65)
print("Decimated-Output Spacing Summary")
print("=" * 65)
print(f"{'spacing':>8}  {'nw_dec':>7}  {'meth':>6}  {'total_pts':>10}  {'avg/filter':>10}  {'vs_hurst_7':>10}")
for sp in SPACINGS:
    nw_dec = NW_WEEKLY // sp
    for label, meas in [('PP', all_results_pp[sp]), ('PT', all_results_pt[sp])]:
        n_pts  = sum(len(t) for t, f, _ in meas)
        ratio  = n_pts / (7 * 23)
        print(f"{sp:>8d}  {nw_dec:>7d}  {label:>6}  {n_pts:>10d}  {n_pts/23:>10.1f}  {ratio:>9.2f}x")
print()
print("Note: spacing=7 with ~500 taps targets Hurst's ~7 pts/filter density")
print()
print("Output files:")
for f in [out1, out2, out3, out4]:
    print(f"  {os.path.basename(f)}")
print("\nDone.")
