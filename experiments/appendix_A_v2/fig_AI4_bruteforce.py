# -*- coding: utf-8 -*-
"""
Figure AI-4 Brute-Force Measurement Scheme Comparison
Appendix A, Figure AI-4

Systematically tests every plausible measurement scheme to find which best
matches Hurst's original AI-4 "Frequency Versus Time" figure.

Key insight: AI-3 peaks/troughs ARE the AI-4 measurement events.
- The FVT plot measures the period between events and places a frequency
  value AT the event time.
- We test all 6 schemes:
    A. PP  : peak → peak    (full period),  placed at 2nd peak
    B. TT  : trough → trough (full period), placed at 2nd trough
    C. PT  : peak → trough  (half period × 2 = full-period estimate), placed at trough
    D. TP  : trough → peak  (half period × 2 = full-period estimate), placed at peak
    E. PT+TP interleaved: all events, half-periods → same as zero-crossing but with peaks not ZC
    F. PP+TT interleaved: peaks & troughs both as full-period estimators, merged

- Sub-sample parabolic interpolation at every peak/trough for precision.

Outputs:
  fig_AI4_brute_6schemes.png   – 2×3 grid: 6 schemes for all 23 filters
  fig_AI4_brute_aligned.png    – AI-3 waveform (FC-20..23) + AI-4 side-by-side
  fig_AI4_brute_best.png       – Best matching scheme only, vs Hurst reference
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

from utils_ai import (
    load_weekly_data,
    design_comb_bank, make_ormsby_kernels, apply_comb_bank,
    get_window, FS_WEEKLY, NW_WEEKLY,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
REF_IMAGE  = os.path.join(BASE_DIR, 'references/appendix_a/figure_AI4_v2.png')

N_FILTERS = 23
YMIN, YMAX = 7.4, 12.6
CLIP_FRAC  = 0.30   # slightly wider to not over-clip

_cmap  = plt.colormaps['coolwarm']
COLORS = [_cmap(i / (N_FILTERS - 1)) for i in range(N_FILTERS)]


# ============================================================================
# SUB-SAMPLE PARABOLIC INTERPOLATION
# ============================================================================

def parabolic_peak(y, idx):
    """
    Sub-sample peak position via 3-point parabolic fit.
    Returns fractional sample index of the true peak.
    """
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    y0, y1, y2 = float(y[idx-1]), float(y[idx]), float(y[idx+1])
    denom = y0 - 2.0*y1 + y2
    if abs(denom) < 1e-14:
        return float(idx)
    delta = 0.5 * (y0 - y2) / denom
    return idx + np.clip(delta, -1.0, 1.0)


def find_peaks_interp(signal, f_center, fs, min_dist_frac=0.55):
    """
    Find peaks with distance guard, then apply parabolic sub-sample interpolation.
    Returns array of fractional sample positions.
    """
    T_samp = 2 * np.pi / f_center * fs
    min_d  = max(3, int(T_samp * min_dist_frac))
    idx, _ = find_peaks(signal, distance=min_d)
    return np.array([parabolic_peak(signal, i) for i in idx])


def find_troughs_interp(signal, f_center, fs, min_dist_frac=0.55):
    """Find troughs with sub-sample interpolation."""
    return find_peaks_interp(-signal, f_center, fs, min_dist_frac)


# ============================================================================
# MEASUREMENT SCHEMES  (all return times in SAMPLES, freqs in RAD/YR)
# ============================================================================

def scheme_PP(peaks, fs):
    """Peak-to-peak full period; placed at 2nd peak."""
    if len(peaks) < 2:
        return np.array([]), np.array([])
    dt    = np.diff(peaks) / fs          # years
    freqs = 2 * np.pi / dt
    times = peaks[1:]
    return times, freqs


def scheme_TT(troughs, fs):
    """Trough-to-trough full period; placed at 2nd trough."""
    return scheme_PP(troughs, fs)


def scheme_PT(peaks, troughs, fs):
    """
    Peak → next trough (half period); placed at trough.
    Frequency = 2*pi / (2 * half_period) = pi / half_period.
    """
    times, freqs = [], []
    ti = 0
    for p in peaks:
        # find first trough AFTER this peak
        while ti < len(troughs) and troughs[ti] <= p:
            ti += 1
        if ti >= len(troughs):
            break
        t = troughs[ti]
        half_T = (t - p) / fs          # years
        if half_T > 0:
            freqs.append(np.pi / half_T)
            times.append(t)
    return np.array(times), np.array(freqs)


def scheme_TP(peaks, troughs, fs):
    """
    Trough → next peak (half period); placed at peak.
    """
    times, freqs = [], []
    pi_idx = 0
    for tr in troughs:
        while pi_idx < len(peaks) and peaks[pi_idx] <= tr:
            pi_idx += 1
        if pi_idx >= len(peaks):
            break
        p = peaks[pi_idx]
        half_T = (p - tr) / fs
        if half_T > 0:
            freqs.append(np.pi / half_T)
            times.append(p)
    return np.array(times), np.array(freqs)


def scheme_PT_TP_interleaved(peaks, troughs, fs):
    """
    PT and TP interleaved: all peaks and troughs in chronological order,
    consecutive half-period → frequency = pi / half_period.
    Placed at the 2nd event of each pair.
    """
    events = ([(t, 'P') for t in peaks] +
              [(t, 'T') for t in troughs])
    events.sort(key=lambda x: x[0])
    if len(events) < 2:
        return np.array([]), np.array([])
    times, freqs = [], []
    for k in range(len(events) - 1):
        t1, _ = events[k]
        t2, _ = events[k + 1]
        dt = (t2 - t1) / fs
        if dt > 0:
            freqs.append(np.pi / dt)
            times.append(t2)
    return np.array(times), np.array(freqs)


def scheme_PP_TT_interleaved(peaks, troughs, fs):
    """
    PP and TT merged (both full-period estimators), sorted chronologically.
    Gives ~ same density as PP+TT merged, with the zig-zag where peak
    period ≠ trough period due to amplitude modulation.
    """
    tp, fp = scheme_PP(peaks, fs)
    tt, ft = scheme_TT(troughs, fs)
    if len(tp) == 0 and len(tt) == 0:
        return np.array([]), np.array([])
    all_t = np.concatenate([tp, tt])
    all_f = np.concatenate([fp, ft])
    order = np.argsort(all_t)
    return all_t[order], all_f[order]


# ============================================================================
# CLIP TO DISPLAY WINDOW + FREQUENCY BOUNDS
# ============================================================================

def clip_to_window(times, freqs, s_idx, e_idx, f_center):
    """Convert to weeks-from-display-start and clip to plausible freq range."""
    if len(times) == 0:
        return np.array([]), np.array([])
    mask = (times >= s_idx) & (times < e_idx)
    t = times[mask] - s_idx          # weeks from display start (weekly: 1 sample = 1 week)
    f = freqs[mask]
    valid = ((f >= f_center * (1 - CLIP_FRAC)) & (f <= f_center * (1 + CLIP_FRAC)) &
             (f >= YMIN) & (f <= YMAX))
    return t[valid], f[valid]


# ============================================================================
# OPTIONAL SMOOTHING
# ============================================================================

def smooth(t, f, n=2):
    """n-point centred moving average on f (only if enough points)."""
    if len(f) <= n:
        return t, f
    f_sm = np.convolve(f, np.ones(n) / n, mode='valid')
    pad  = (len(f) - len(f_sm)) // 2
    # keep matching t indices
    t_sm = t[pad: pad + len(f_sm)]
    return t_sm, f_sm


# ============================================================================
# PLOT ONE PANEL
# ============================================================================

def plot_panel(ax, all_meas, title, n_weeks, smooth_n=1,
               show_labels_right=True):
    """Plot all 23 filters on one axis. smooth_n=1 means no smoothing."""
    for meas in all_meas:
        t, f = meas['times'], meas['freqs']
        idx  = meas['idx']
        fc   = meas['center']
        if len(t) == 0:
            continue
        if smooth_n > 1:
            t, f = smooth(t, f, smooth_n)
        if len(t) == 0:
            continue
        color = COLORS[idx]
        ax.plot(t, f, '-o', color=color, markersize=2.0,
                linewidth=0.75, alpha=0.9, zorder=3)
        ax.text(t[0] - 2, f[0], str(idx + 1), fontsize=5,
                color=color, ha='right', va='center')
        if show_labels_right:
            ax.text(t[-1] + 2, f[-1], str(idx + 1), fontsize=5,
                    color=color, ha='left', va='center')

    for ref in [8, 9, 10, 11, 12]:
        ax.axhline(ref, color='silver', linewidth=0.5, zorder=1)

    ax.set_xlim(0, n_weeks)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(np.arange(0, n_weeks + 1, 25))
    ax.set_yticks([8, 9, 10, 11, 12])
    ax.set_xlabel('Weeks', fontsize=8)
    ax.set_ylabel('Rad/Yr', fontsize=8)
    ax.grid(True, axis='x', alpha=0.2)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)


# ============================================================================
# LOAD DATA AND APPLY FILTERS
# ============================================================================

print("Loading weekly data and applying 23 comb filters (ALL CSV)...")
close, dates = load_weekly_data()
specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)
outputs = apply_comb_bank(close, filters, fs=FS_WEEKLY)

s_idx, e_idx = get_window(dates)
n_weeks = e_idx - s_idx   # weekly: 1 sample = 1 week
print(f"Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} weeks)")
print()

# ============================================================================
# COMPUTE ALL 6 SCHEMES FOR EVERY FILTER
# ============================================================================

scheme_names = [
    ('A: Peak→Peak\n(full period, at 2nd peak)', 'PP', 1),
    ('B: Trough→Trough\n(full period, at 2nd trough)', 'TT', 1),
    ('C: Peak→Trough\n(half-period×2, at trough)', 'PT', 1),
    ('D: Trough→Peak\n(half-period×2, at peak)', 'TP', 1),
    ('E: PT+TP interleaved\n(half-period, at each event)', 'PT_TP', 1),
    ('F: PP+TT interleaved\n(full period, peaks & troughs merged)', 'PP_TT', 1),
]

# Store: {scheme_key: list of {times, freqs, center, idx}}
all_schemes = {key: [] for _, key, _ in scheme_names}

for i, out in enumerate(outputs):
    spec     = specs[i]
    fc       = spec['f_center']
    sig_real = out['signal'].real

    pk = find_peaks_interp(sig_real, fc, FS_WEEKLY)
    tr = find_troughs_interp(sig_real, fc, FS_WEEKLY)

    kw = dict(s_idx=s_idx, e_idx=e_idx, f_center=fc)

    t, f = scheme_PP(pk, FS_WEEKLY);               t,f = clip_to_window(t,f,**kw)
    all_schemes['PP'].append({'times':t,'freqs':f,'center':fc,'idx':i})

    t, f = scheme_TT(tr, FS_WEEKLY);               t,f = clip_to_window(t,f,**kw)
    all_schemes['TT'].append({'times':t,'freqs':f,'center':fc,'idx':i})

    t, f = scheme_PT(pk, tr, FS_WEEKLY);            t,f = clip_to_window(t,f,**kw)
    all_schemes['PT'].append({'times':t,'freqs':f,'center':fc,'idx':i})

    t, f = scheme_TP(pk, tr, FS_WEEKLY);            t,f = clip_to_window(t,f,**kw)
    all_schemes['TP'].append({'times':t,'freqs':f,'center':fc,'idx':i})

    t, f = scheme_PT_TP_interleaved(pk, tr, FS_WEEKLY); t,f = clip_to_window(t,f,**kw)
    all_schemes['PT_TP'].append({'times':t,'freqs':f,'center':fc,'idx':i})

    t, f = scheme_PP_TT_interleaved(pk, tr, FS_WEEKLY); t,f = clip_to_window(t,f,**kw)
    all_schemes['PP_TT'].append({'times':t,'freqs':f,'center':fc,'idx':i})

for _, key, _ in scheme_names:
    total = sum(len(m['times']) for m in all_schemes[key])
    print(f"  {key:8s}: {total:4d} pts total  ({total//N_FILTERS:2d} avg/filter)")

print()

# ============================================================================
# FIGURE 1: 2×3 GRID OF ALL 6 SCHEMES
# ============================================================================

fig6, axes6 = plt.subplots(2, 3, figsize=(24, 13),
                            gridspec_kw={'hspace': 0.40, 'wspace': 0.22})
axes_flat = axes6.flatten()

for ax, (title, key, sm) in zip(axes_flat, scheme_names):
    plot_panel(ax, all_schemes[key], title, n_weeks,
               smooth_n=sm, show_labels_right=True)

fig6.suptitle(
    f'FIGURE AI-4: All 6 Measurement Schemes  |  Weekly Ormsby  |  23 Filters\n'
    f'{DATE_DISPLAY_START} to {DATE_DISPLAY_END}  '
    f'|  Parabolic sub-sample interpolation at every peak/trough',
    fontsize=11, fontweight='bold'
)
out6 = os.path.join(SCRIPT_DIR, 'fig_AI4_brute_6schemes.png')
fig6.savefig(out6, dpi=140, bbox_inches='tight')
plt.close(fig6)
print(f"Saved: {out6}")


# ============================================================================
# FIGURE 2: AI-3 WAVEFORM + AI-4 FVT  aligned on same x-axis (FC-18 to FC-23)
# ============================================================================
# Shows the waveforms (top) and where the frequency measurements land (bottom)
# for the 6 highest-frequency filters (which show the most variation in AI-4)

SEL = list(range(17, 23))   # FC-18..FC-23 (0-indexed 17..22)
N_SEL = len(SEL)

weeks_disp = np.arange(n_weeks)

fig_al, axes_al = plt.subplots(2, 1, figsize=(16, 10),
                                gridspec_kw={'hspace': 0.08, 'height_ratios': [1, 1]},
                                sharex=True)

ax_wave = axes_al[0]
ax_fvt  = axes_al[1]

# ---- Top panel: waveforms (AI-3 style) ----
SPACING   = 3.5
TRACK_AMP = 1.2
rms_vals = [np.sqrt(np.mean(outputs[i]['signal'][s_idx:e_idx].real**2))
            for i in SEL]
rms_vals = [r for r in rms_vals if r > 0]
global_rms = np.median(rms_vals) if rms_vals else 1.0
scale = TRACK_AMP / (3.0 * global_rms)

for j, i in enumerate(SEL):
    offset = (N_SEL - 1 - j) * SPACING
    sig = outputs[i]['signal'][s_idx:e_idx].real * scale
    color = COLORS[i]
    ax_wave.axhline(offset, color='silver', linewidth=0.4)
    ax_wave.plot(weeks_disp, sig + offset, '-', color=color,
                 linewidth=0.7, alpha=0.9)
    env = outputs[i]['envelope'][s_idx:e_idx] * scale if outputs[i]['envelope'] is not None else None
    if env is not None:
        env_sm = uniform_filter1d(env, size=3)
        ax_wave.plot(weeks_disp,  env_sm + offset, '--', color=color,
                     linewidth=0.6, alpha=0.5)
        ax_wave.plot(weeks_disp, -env_sm + offset, '--', color=color,
                     linewidth=0.6, alpha=0.5)
    ax_wave.text(-3, offset, f'FC-{i+1}\n{specs[i]["f_center"]:.1f}r/y',
                 fontsize=7, ha='right', va='center', color=color)

ax_wave.set_xlim(0, n_weeks)
ax_wave.set_ylim(-SPACING * 0.5, N_SEL * SPACING)
ax_wave.set_ylabel('Filter Output (normalized)', fontsize=9)
ax_wave.set_title(
    f'AI-3 Waveforms (FC-18..FC-23)  aligned with  AI-4 FVT below\n'
    f'Peaks/troughs in top panel = measurement events in bottom panel',
    fontsize=10, fontweight='bold'
)
ax_wave.grid(True, axis='x', alpha=0.2)
ax_wave.set_yticks([])

# ---- Bottom panel: FVT for selected filters ----
# Show 3 best candidate schemes side-by-side on same axes
scheme_overlay = [
    ('PP',    '-o', 2.2, 0.85, 'A: Peak→Peak'),
    ('PP_TT', '-^', 1.8, 0.70, 'F: PP+TT merged'),
    ('PT_TP', '-s', 1.4, 0.55, 'E: PT+TP interleaved'),
]

for scheme_key, marker, ms, alpha, slabel in scheme_overlay:
    for j, i in enumerate(SEL):
        meas = all_schemes[scheme_key][i]
        t, f = meas['times'], meas['freqs']
        if len(t) == 0:
            continue
        color = COLORS[i]
        lbl = slabel if j == 0 else None
        ax_fvt.plot(t, f, marker, color=color, markersize=ms,
                    linewidth=0.75, alpha=alpha, label=lbl, zorder=3)

for ref in [8, 9, 10, 11, 12]:
    if YMIN <= ref <= YMAX:
        ax_fvt.axhline(ref, color='silver', linewidth=0.5)

ax_fvt.set_xlim(0, n_weeks)
ax_fvt.set_ylim(YMIN, YMAX)
ax_fvt.set_xticks(np.arange(0, n_weeks + 1, 25))
ax_fvt.set_yticks([8, 9, 10, 11, 12])
ax_fvt.set_xlabel('Weeks', fontsize=10)
ax_fvt.set_ylabel('Radians/Year', fontsize=10)
ax_fvt.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax_fvt.grid(True, axis='x', alpha=0.2)
ax_fvt.set_title('FVT: Three Candidate Schemes Overlaid (FC-18..FC-23)',
                 fontsize=10, fontweight='bold')

fig_al.text(0.01, 0.99, DATE_DISPLAY_START, fontsize=7, va='top', color='gray')
fig_al.text(0.99, 0.99, DATE_DISPLAY_END, fontsize=7, va='top', ha='right', color='gray')

out_al = os.path.join(SCRIPT_DIR, 'fig_AI4_brute_aligned.png')
fig_al.savefig(out_al, dpi=150, bbox_inches='tight')
plt.close(fig_al)
print(f"Saved: {out_al}")


# ============================================================================
# FIGURE 3: BEST SCHEME vs HURST REFERENCE (side by side)
# ============================================================================
# "Best" = PP+TT interleaved (scheme F) as it gives the right density
# and creates the zig-zag between peak and trough period estimates.

fig_best, (ax_hurst, ax_pp, ax_f) = plt.subplots(1, 3, figsize=(27, 9),
    gridspec_kw={'wspace': 0.10})

# Panel 1: Hurst reference
if os.path.exists(REF_IMAGE):
    ax_hurst.imshow(mpimg.imread(REF_IMAGE))
ax_hurst.axis('off')
ax_hurst.set_title("Hurst's Original AI-4\n(reference)", fontsize=10, fontweight='bold')

# Panel 2: Scheme A (PP - one point per cycle)
plot_panel(ax_pp, all_schemes['PP'],
           'Scheme A: Peak→Peak\n(1 pt/cycle - densest match to Hurst)',
           n_weeks, smooth_n=1, show_labels_right=True)

# Panel 3: Scheme F (PP+TT interleaved)
plot_panel(ax_f, all_schemes['PP_TT'],
           'Scheme F: PP+TT Interleaved\n(zig-zag, 2 pts/cycle)',
           n_weeks, smooth_n=1, show_labels_right=True)

fig_best.suptitle(
    'FIGURE AI-4  |  Hurst Reference vs Best Candidate Schemes\n'
    'Weekly DJIA  |  Ormsby FIR  |  Parabolic sub-sample interpolation',
    fontsize=11, fontweight='bold', y=1.01
)
out_best = os.path.join(SCRIPT_DIR, 'fig_AI4_brute_best.png')
fig_best.savefig(out_best, dpi=130, bbox_inches='tight')
plt.close(fig_best)
print(f"Saved: {out_best}")

print("\nDone.")
