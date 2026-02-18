# -*- coding: utf-8 -*-
"""
Figure AI-4: Frequency vs Time
Appendix A, Figure AI-4 Reproduction

Shows instantaneous frequency measured from each comb filter output.

Hurst's original figure: all 23 filters overlaid in one panel, with filter
numbers labeled at both left and right sides, x-axis 0-275 weeks, y-axis
7.5-12.5+ rad/yr.  Lines show slow sinusoidal variation (the "incredible
frequency-separation effect").

Methods implemented:
  1. Zero-crossing half-period
  2. Peak+trough interleaved (zig-zag, closest to Hurst's original)
  3. 1-mode MPM on each half-cycle segment (sub-sample resolution)
  4. Wrapped phase peak/trough detection (NEW)

Figures produced:
  fig_AI4_hurst_style.png  - Single-panel, all 23 filters, matching Hurst's layout
  fig_AI4_weekly_methods.png  - Weekly data: 4 methods in 2x2 grid
  fig_AI4_daily_methods.png   - Daily data:  4 methods in 2x2 grid

Display window: 1934-12-07 to 1940-01-26

Reference: J.M. Hurst, Appendix A, Figure AI-4, p.194
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.signal import find_peaks

from utils_ai import (
    load_weekly_data, load_daily_data,
    design_comb_bank, make_ormsby_kernels, apply_comb_bank,
    get_window, daily_nw, FS_WEEKLY, NW_WEEKLY,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
    measure_zerocross_halfperiod, measure_phase_halfperiod,
    mpm_1mode,
    print_comb_bank_summary,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

N_DISPLAY  = 23     # all 23 filters
YMIN       = 7.4    # slightly below 7.6 rad/yr centre of FC-1
YMAX       = 12.6   # slightly above 12.0 rad/yr centre of FC-23
CLIP_FRAC  = 0.28   # keep measurements within ±28 % of filter centre
              # → only admit freq if fc*(1-CLIP) <= f <= fc*(1+CLIP)

# Colour ramp from dark-blue (low freq) to dark-red (high freq)
_cmap  = plt.colormaps['coolwarm']
COLORS = [_cmap(i / (N_DISPLAY - 1)) for i in range(N_DISPLAY)]


# ============================================================================
# MEASUREMENT FUNCTIONS
# ============================================================================

def measure_peak_period(signal_real, fs, f_center):
    """
    Peak-to-peak full period → frequency, placed at 2nd peak.

    Uses distance-based peak detection (not prominence) so it works correctly
    even when the global signal amplitude varies widely across the full record.
    """
    T_samples = 2 * np.pi / f_center * fs      # expected full period in samples
    min_dist  = max(3, int(T_samples * 0.5))    # peaks no closer than 0.5 period
    peaks, _  = find_peaks(signal_real, distance=min_dist)
    if len(peaks) < 2:
        return np.array([]), np.array([])
    dt = np.diff(peaks) / fs        # years
    freqs = 2 * np.pi / dt          # rad/yr (full period)
    times = peaks[1:].astype(float)
    return times, freqs


def measure_trough_period(signal_real, fs, f_center):
    """Trough-to-trough full period → frequency, placed at 2nd trough."""
    T_samples = 2 * np.pi / f_center * fs
    min_dist  = max(3, int(T_samples * 0.5))
    troughs, _ = find_peaks(-signal_real, distance=min_dist)
    if len(troughs) < 2:
        return np.array([]), np.array([])
    dt = np.diff(troughs) / fs
    freqs = 2 * np.pi / dt
    times = troughs[1:].astype(float)
    return times, freqs


def interleave_peaks_troughs(t_pk, f_pk, t_tr, f_tr):
    """Merge peak and trough measurements chronologically (zig-zag pattern)."""
    if len(t_pk) == 0 and len(t_tr) == 0:
        return np.array([]), np.array([])
    all_t = np.concatenate([t_pk, t_tr])
    all_f = np.concatenate([f_pk, f_tr])
    order = np.argsort(all_t)
    return all_t[order], all_f[order]


def measure_halfcycle_mpm(analytic_signal, fs, min_seg_len=6):
    """1-mode MPM on each zero-crossing segment for sub-sample frequency."""
    real = analytic_signal.real
    signs = np.sign(real)
    sign_idx = np.where(signs != 0)[0]
    if len(sign_idx) < 2:
        return np.array([]), np.array([])

    crossings = []
    for i in range(len(sign_idx) - 1):
        a, b = sign_idx[i], sign_idx[i + 1]
        if signs[a] != signs[b]:
            frac = abs(real[a]) / (abs(real[a]) + abs(real[b]))
            crossings.append(a + frac)

    crossings = np.array(crossings)
    if len(crossings) < 2:
        return np.array([]), np.array([])

    times, freqs = [], []
    for i in range(len(crossings) - 1):
        t1 = int(np.floor(crossings[i]))
        t2 = int(np.ceil(crossings[i + 1]))
        if t2 - t1 < min_seg_len:
            continue
        seg = analytic_signal[t1:t2]
        omega = mpm_1mode(seg)
        if np.isnan(omega):
            continue
        freq_radyr = abs(omega) * fs
        half_exp = np.pi / (freq_radyr / fs)
        if abs((t2 - t1) - half_exp) > 3 * half_exp:
            continue
        times.append((crossings[i] + crossings[i + 1]) / 2.0)
        freqs.append(freq_radyr)

    return np.array(times), np.array(freqs)


def filter_window(times, freqs, s_idx, e_idx, f_center, clip=CLIP_FRAC,
                  samp_per_week=1.0):
    """
    Clip measurements to display window and to ±clip fraction of filter centre.
    Returns (times_in_weeks, freqs_in_radyr).
    """
    if len(times) == 0:
        return np.array([]), np.array([])
    mask = (times >= s_idx) & (times < e_idx)
    t = (times[mask] - s_idx) / samp_per_week    # convert to weeks
    f = freqs[mask]
    valid = ((f >= f_center * (1 - clip)) & (f <= f_center * (1 + clip)) &
             (f >= YMIN) & (f <= YMAX))
    return t[valid], f[valid]


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

def plot_fvt_panel(ax, all_meas, title, n_weeks, connect=True,
                   show_labels_right=False):
    """
    Plot frequency-vs-time for all filters on one axis.

    all_meas : list of dicts with keys 'times', 'freqs', 'center', 'idx'
    """
    for meas in all_meas:
        t, f = meas['times'], meas['freqs']
        idx  = meas['idx']
        color = COLORS[idx]
        label_txt = str(idx + 1)

        if len(t) == 0:
            continue

        if connect:
            ax.plot(t, f, '-o', color=color, markersize=1.8,
                    linewidth=0.7, alpha=0.85, zorder=3)
        else:
            ax.plot(t, f, 'o', color=color, markersize=2.0,
                    alpha=0.8, zorder=3)

        # Label at start
        ax.text(t[0] - 2, f[0], label_txt, fontsize=5.5,
                color=color, ha='right', va='center')
        # Label at end
        if show_labels_right:
            ax.text(t[-1] + 2, f[-1], label_txt, fontsize=5.5,
                    color=color, ha='left', va='center')

    # Horizontal gridlines at integer rad/yr
    for ref in range(int(YMIN) + 1, int(YMAX) + 1):
        ax.axhline(ref, color='silver', linewidth=0.5, zorder=1)

    ax.set_xlim(0, n_weeks)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel('Weeks', fontsize=10)
    ax.set_ylabel('Radians/Year', fontsize=10)
    ax.grid(True, axis='x', alpha=0.2)
    ax.set_title(title, fontsize=10, fontweight='bold')


# ============================================================================
# BUILD ALL MEASUREMENTS FOR ONE DATA SET
# ============================================================================

def build_measurements(close, dates_dt, fs, specs, filters, outputs):
    """Compute all 4 methods for every filter; return lists of meas dicts."""
    s_idx, e_idx = get_window(dates_dt)
    samp_pw = fs / FS_WEEKLY          # samples per week

    meas_zc   = []   # zero-crossing
    meas_pt   = []   # peak+trough interleaved
    meas_mpm  = []   # MPM
    meas_ph   = []   # wrapped phase

    for i, out in enumerate(outputs):
        spec     = specs[i]
        fc       = spec['f_center']
        sig_real = out['signal'].real
        sig_cx   = out['signal']

        kw = dict(s_idx=s_idx, e_idx=e_idx, f_center=fc, samp_per_week=samp_pw)

        # Method 1: zero-crossing half-period
        t1, f1 = measure_zerocross_halfperiod(sig_real, fs)
        t1, f1 = filter_window(t1, f1, **kw)
        meas_zc.append({'times': t1, 'freqs': f1, 'center': fc, 'idx': i})

        # Method 2: peak+trough interleaved (Hurst-style zig-zag)
        tpk, fpk = measure_peak_period(sig_real, fs, fc)
        ttr, ftr = measure_trough_period(sig_real, fs, fc)
        t2, f2   = interleave_peaks_troughs(tpk, fpk, ttr, ftr)
        t2, f2   = filter_window(t2, f2, **kw)
        meas_pt.append({'times': t2, 'freqs': f2, 'center': fc, 'idx': i})

        # Method 3: MPM half-cycle
        t3, f3 = measure_halfcycle_mpm(sig_cx, fs)
        t3, f3 = filter_window(t3, f3, **kw)
        meas_mpm.append({'times': t3, 'freqs': f3, 'center': fc, 'idx': i})

        # Method 4: wrapped phase
        t4_raw, f4_raw, _ = measure_phase_halfperiod(sig_cx, fs)
        t4, f4 = filter_window(t4_raw, f4_raw, **kw)
        meas_ph.append({'times': t4, 'freqs': f4, 'center': fc, 'idx': i})

    n_weeks = (e_idx - s_idx) / samp_pw
    return meas_zc, meas_pt, meas_mpm, meas_ph, n_weeks


# ============================================================================
# MAIN: WEEKLY DATA
# ============================================================================

print("=" * 70)
print("AI-4: Weekly data")
print("=" * 70)

close_w, dates_w = load_weekly_data()
fs_w = FS_WEEKLY
nw_w = NW_WEEKLY
print(f"  Loaded {len(close_w)} points (ALL CSV), fs={fs_w}")

specs_w   = design_comb_bank(fs=fs_w, nw=nw_w)
filters_w = make_ormsby_kernels(specs_w, fs=fs_w)
print_comb_bank_summary(specs_w)
outputs_w = apply_comb_bank(close_w, filters_w, fs=fs_w)

mzc_w, mpt_w, mmpm_w, mph_w, nwk_w = build_measurements(
    close_w, dates_w, fs_w, specs_w, filters_w, outputs_w)

print(f"  Display: {nwk_w:.0f} weeks")
for label, ml in [('ZC', mzc_w), ('P+T', mpt_w), ('MPM', mmpm_w), ('Phase', mph_w)]:
    total = sum(len(m['times']) for m in ml)
    print(f"    {label}: {total} points total across 23 filters")
print()

# ---- Hurst-style main figure: single panel, zero-crossing method ----
# Zero-crossing gives the densest, most reliable measurement set for weekly data.
fig_hs, ax_hs = plt.subplots(figsize=(13, 9))
plot_fvt_panel(ax_hs, mzc_w,
               'FREQUENCY VERSUS TIME  -  Figure AI-4\n'
               'Ormsby FIR  |  Weekly DJIA  |  Zero-Crossing Half-Period',
               nwk_w, connect=True, show_labels_right=True)
ax_hs.set_xticks(np.arange(0, nwk_w + 1, 25))
ax_hs.set_yticks([8, 9, 10, 11, 12])
# Right-axis labels
ax_right = ax_hs.twinx()
ax_right.set_ylim(YMIN, YMAX)
ax_right.set_yticks([8, 9, 10, 11, 12])
ax_right.tick_params(labelsize=9)
ax_right.set_ylabel('Radians/Year', fontsize=10)

# Date annotations
ax_hs.text(0.01, 0.98, DATE_DISPLAY_START, transform=ax_hs.transAxes,
           fontsize=7, va='top', color='gray')
ax_hs.text(0.99, 0.98, DATE_DISPLAY_END, transform=ax_hs.transAxes,
           fontsize=7, va='top', ha='right', color='gray')

fig_hs.tight_layout()
path_hs = os.path.join(SCRIPT_DIR, 'fig_AI4_hurst_style.png')
fig_hs.savefig(path_hs, dpi=150, bbox_inches='tight')
plt.close(fig_hs)
print(f"  Saved: {path_hs}")

# ---- 4-method comparison figure (weekly) ----
fig_w, axes_w = plt.subplots(2, 2, figsize=(20, 14),
                              gridspec_kw={'hspace': 0.38, 'wspace': 0.28})
ax_flat = axes_w.flatten()

method_info = [
    (mzc_w,  'Method 1: Zero-Crossing Half-Period'),
    (mpt_w,  'Method 2: Peak+Trough Interleaved  (Hurst)'),
    (mmpm_w, 'Method 3: MPM per Half-Cycle  (sub-sample)'),
    (mph_w,  'Method 4: Wrapped Phase Peak/Trough'),
]
for ax, (ml, title) in zip(ax_flat, method_info):
    ax.set_xlim(0, nwk_w)
    plot_fvt_panel(ax, ml, title, nwk_w, connect=True, show_labels_right=False)
    ax.set_xticks(np.arange(0, nwk_w + 1, 25))
    ax.set_yticks([8, 9, 10, 11, 12])

fig_w.suptitle(
    f'FREQUENCY VERSUS TIME  -  Figure AI-4\n'
    f'Weekly DJIA  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  |  All 23 Filters',
    fontsize=12, fontweight='bold'
)
path_mw = os.path.join(SCRIPT_DIR, 'fig_AI4_weekly_methods.png')
fig_w.savefig(path_mw, dpi=150, bbox_inches='tight')
plt.close(fig_w)
print(f"  Saved: {path_mw}")
print()

# ============================================================================
# DAILY DATA
# ============================================================================

print("=" * 70)
print("AI-4: Daily data")
print("=" * 70)

close_d, dates_d, fs_d = load_daily_data()
nw_d = daily_nw(fs_d)
print(f"  Loaded {len(close_d)} points (ALL CSV), fs={fs_d:.1f}, nw={nw_d}")

specs_d   = design_comb_bank(fs=fs_d, nw=nw_d)
filters_d = make_ormsby_kernels(specs_d, fs=fs_d)
outputs_d = apply_comb_bank(close_d, filters_d, fs=fs_d)

mzc_d, mpt_d, mmpm_d, mph_d, nwk_d = build_measurements(
    close_d, dates_d, fs_d, specs_d, filters_d, outputs_d)

print(f"  Display: {nwk_d:.0f} weeks")
for label, ml in [('ZC', mzc_d), ('P+T', mpt_d), ('MPM', mmpm_d), ('Phase', mph_d)]:
    total = sum(len(m['times']) for m in ml)
    print(f"    {label}: {total} points total")
print()

fig_d, axes_d = plt.subplots(2, 2, figsize=(20, 14),
                              gridspec_kw={'hspace': 0.38, 'wspace': 0.28})
ax_flat_d = axes_d.flatten()

method_info_d = [
    (mzc_d,  'Method 1: Zero-Crossing Half-Period'),
    (mpt_d,  'Method 2: Peak+Trough Interleaved  (Hurst)'),
    (mmpm_d, 'Method 3: MPM per Half-Cycle  (sub-sample)'),
    (mph_d,  'Method 4: Wrapped Phase Peak/Trough'),
]
for ax, (ml, title) in zip(ax_flat_d, method_info_d):
    ax.set_xlim(0, nwk_d)
    plot_fvt_panel(ax, ml, title, nwk_d, connect=True, show_labels_right=False)
    ax.set_xticks(np.arange(0, nwk_d + 1, 25))
    ax.set_yticks([8, 9, 10, 11, 12])

fig_d.suptitle(
    f'FREQUENCY VERSUS TIME  -  Figure AI-4\n'
    f'Daily DJIA  (fs={fs_d:.0f})  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  |  All 23 Filters',
    fontsize=12, fontweight='bold'
)
path_md = os.path.join(SCRIPT_DIR, 'fig_AI4_daily_methods.png')
fig_d.savefig(path_md, dpi=150, bbox_inches='tight')
plt.close(fig_d)
print(f"  Saved: {path_md}")
print()
print("Done.")
