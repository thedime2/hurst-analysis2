# -*- coding: utf-8 -*-
"""
Figure AI-4 Phase-Derivative Experiments

Three new measurement methods beyond the brute-force peak/trough scheme comparison:

  Method PD:  Instantaneous phase derivative  d(unwrap(angle(z)))/dt
              -- continuous, sub-sample precision, no inter-peak discretisation
              -- sampled at peak times (one value per cycle)

  Method PD-S: Phase derivative smoothed + subsampled at peaks
              -- heavy Gaussian smooth of omega_inst, then sample at peaks
              -- removes numerical noise while preserving genuine FM

  Method AG:  Amplitude-gated reproduction of Hurst's "missing" filters
              -- gate: only show filter k if its display-window RMS exceeds
                 30% of the median RMS across all filters
              -- expected to suppress FC-8 (9.0), FC-12 (9.8), FC-16 (10.6)
                 confirming Hurst's "filters 8, 12, 16 discarded as meaningless"

Also generates a 5-panel diagnostic figure for one selected filter (FC-20)
showing: waveform | wrapped phase | unwrapped phase | omega_inst | FVT comparison.

Outputs:
  fig_AI4_phase_deriv_4panel.png  -- PD vs PD-S vs PT_TP-best vs Reference
  fig_AI4_phase_deriv_filter.png  -- 5-panel diagnostic for one filter
  fig_AI4_amplitude_gated.png     -- Hurst-style panel with amplitude gating
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from utils_ai import (
    load_weekly_data,
    design_comb_bank, make_ormsby_kernels, apply_comb_bank,
    get_window, FS_WEEKLY, NW_WEEKLY,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
REF_IMAGE  = os.path.join(BASE_DIR, 'references/appendix_a/figure_AI4_v2.png')

N_FILTERS  = 23
YMIN, YMAX = 7.4, 12.6
CLIP_FRAC  = 0.30

_cmap  = plt.colormaps['coolwarm']
COLORS = [_cmap(i / (N_FILTERS - 1)) for i in range(N_FILTERS)]

# Amplitude gate: filter must have display-window RMS >= AMP_THRESH * median_RMS
AMP_THRESH = 0.30


# ============================================================================
# UTILITIES: PEAK FINDING (reused from final script)
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
    if len(times) == 0:
        return np.array([]), np.array([])
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


# ============================================================================
# METHOD PD: INSTANTANEOUS PHASE DERIVATIVE
# ============================================================================

def instantaneous_freq(z_analytic, fs):
    """
    Compute instantaneous frequency from the unwrapped phase derivative.

    omega_inst[n] = d(phi)/dt  at sample n,
    where phi = unwrap(angle(z_analytic)).

    Returns: omega array of length len(z_analytic)-1, in rad/yr.
    """
    phi = np.unwrap(np.angle(z_analytic))
    omega = np.diff(phi) * fs          # rad/sample * samples/yr = rad/yr
    return omega


def sample_at_peaks(omega_inst, peaks_frac, n_samples):
    """
    Sample the instantaneous frequency array at fractional peak positions.
    Uses linear interpolation between adjacent integer samples.

    omega_inst: length N-1 array (from np.diff)
    peaks_frac: fractional sample indices of peaks (may be non-integer)
    n_samples:  length of original signal

    Returns: frequency values at each peak.
    """
    # Clip to valid range for omega_inst (0..N-2)
    pk = peaks_frac.copy()
    pk = pk[(pk >= 0) & (pk < len(omega_inst) - 1)]
    if len(pk) == 0:
        return pk, np.array([])
    idx_lo = np.floor(pk).astype(int)
    idx_hi = np.minimum(idx_lo + 1, len(omega_inst) - 1)
    frac   = pk - idx_lo
    freqs  = omega_inst[idx_lo] * (1 - frac) + omega_inst[idx_hi] * frac
    return pk, freqs


def measure_pd(z_analytic, f_center, fs, s_idx, e_idx,
               gauss_sigma=0.0, use_troughs=True):
    """
    Phase-derivative method: compute omega_inst, optionally smooth,
    then sample at peak (and optionally trough) times.

    gauss_sigma : Gaussian smooth sigma in SAMPLES (0 = no smoothing)
    use_troughs : if True, sample at peaks + troughs (2x density)

    Returns: (times_display, freqs) with times in weeks from display start.
    """
    sig_real = z_analytic.real
    omega_inst = instantaneous_freq(z_analytic, fs)

    # Optional Gaussian smoothing of omega_inst
    if gauss_sigma > 0:
        omega_smooth = gaussian_filter1d(omega_inst, sigma=gauss_sigma)
    else:
        omega_smooth = omega_inst

    # Find peaks (and optionally troughs) of real part (sub-sample)
    pk = find_peaks_sub(sig_real, f_center, fs)
    if use_troughs:
        tr = find_troughs_sub(sig_real, f_center, fs)
        events = np.sort(np.concatenate([pk, tr]))
    else:
        events = pk

    if len(events) == 0:
        return np.array([]), np.array([])

    # Sample omega at event positions
    _, freqs = sample_at_peaks(omega_smooth, events, len(sig_real))
    times = events[:len(freqs)]

    return clip_to_window(times, freqs, s_idx, e_idx, f_center)


def measure_pdc(z_analytic, f_center, fs, s_idx, e_idx):
    """
    Cycle-averaged phase derivative: average omega_inst over exactly one full
    cycle [peak_k - T/2, peak_k + T/2] for each peak k.

    This gives one frequency estimate per cycle (PP-like density) but uses
    ALL samples in the cycle rather than just the inter-peak interval.
    Less sensitive to instantaneous phase noise than point sampling.
    """
    sig_real   = z_analytic.real
    omega_inst = instantaneous_freq(z_analytic, fs)   # length N-1

    pk_idx = find_peaks_sub(sig_real, f_center, fs)   # fractional positions
    if len(pk_idx) < 2:
        return np.array([]), np.array([])

    T_samp_half = int(round(np.pi / f_center * fs))   # half-cycle in samples

    times, freqs = [], []
    for pk in pk_idx:
        lo = max(0, int(round(pk)) - T_samp_half)
        hi = min(len(omega_inst), int(round(pk)) + T_samp_half)
        if hi <= lo:
            continue
        seg = omega_inst[lo:hi]
        # Clip outliers before averaging (robust to phase spikes)
        median_f = np.median(seg)
        mad      = np.median(np.abs(seg - median_f))
        inliers  = seg[np.abs(seg - median_f) < 5 * max(mad, 0.5)]
        if len(inliers) == 0:
            inliers = seg
        freqs.append(float(np.mean(inliers)))
        times.append(pk)

    return clip_to_window(np.array(times), np.array(freqs), s_idx, e_idx, f_center)


# ============================================================================
# METHOD PT_TP (reference - reused from fig_AI4_final.py)
# ============================================================================

def scheme_PP(peaks, fs):
    if len(peaks) < 2: return np.array([]), np.array([])
    dt = np.diff(peaks) / fs
    return peaks[1:], 2*np.pi / dt


def scheme_PT_TP(peaks, troughs, fs):
    events = ([(t,'P') for t in peaks] + [(t,'T') for t in troughs])
    events.sort(key=lambda x: x[0])
    if len(events) < 2: return np.array([]), np.array([])
    times, freqs = [], []
    for k in range(len(events)-1):
        t1, _ = events[k]; t2, _ = events[k+1]
        dt = (t2-t1)/fs
        if dt > 0:
            freqs.append(np.pi/dt); times.append(t2)
    return np.array(times), np.array(freqs)


# ============================================================================
# LOAD DATA AND APPLY FILTERS
# ============================================================================

print("Loading weekly DJIA data...")
close, dates = load_weekly_data()
specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)
outputs = apply_comb_bank(close, filters, fs=FS_WEEKLY)

s_idx, e_idx = get_window(dates)
n_weeks = e_idx - s_idx
print(f"Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} weeks)")

# ============================================================================
# COMPUTE ALL METHODS FOR EVERY FILTER
# ============================================================================

# Compute display-window RMS for amplitude gating
rms_per_filter = []
for i, out in enumerate(outputs):
    seg = out['signal'][s_idx:e_idx].real
    rms_per_filter.append(np.sqrt(np.mean(seg**2)))
rms_median = np.median(rms_per_filter)
print(f"\nRMS: median={rms_median:.4f}, "
      f"threshold={AMP_THRESH*rms_median:.4f}")

active_filters = []
for i, rms in enumerate(rms_per_filter):
    fc = specs[i]['f_center']
    flag = 'ACTIVE' if rms >= AMP_THRESH * rms_median else 'GATED (low amplitude)'
    status = '*' if rms < AMP_THRESH * rms_median else ' '
    print(f"  {status}FC-{i+1:2d} ({fc:.1f} r/y)  RMS={rms:.4f}  {flag}")
    active_filters.append(rms >= AMP_THRESH * rms_median)

print()

# Gaussian sigma for PD-S: smooth over ~half a cycle
GAUSS_SIGMA_PD  = 0.0    # PD: no extra smoothing (raw at peaks)
# PD-S: adaptive sigma = T_cycle/4 samples (quarter-cycle smoothing per filter)
# computed per-filter in the loop below
SMOOTH_N5 = 5            # 5-pt MA for PT_TP reference

meas_pd    = []   # Method PD:  phase deriv at peaks+troughs, no gauss
meas_pds   = []   # Method PD-S: phase deriv + adaptive Gauss smooth
meas_pdc   = []   # Method PDC: cycle-averaged phase deriv at each peak (1 pt/cycle)
meas_pttp  = []   # Method PT_TP + 5pt MA (from fig_AI4_final)
meas_ag    = []   # Amplitude-gated PT_TP + 5pt MA

for i, out in enumerate(outputs):
    fc       = specs[i]['f_center']
    z        = out['signal']
    sig_real = z.real

    pk = find_peaks_sub(sig_real, fc, FS_WEEKLY)
    tr = find_troughs_sub(sig_real, fc, FS_WEEKLY)

    # Adaptive Gaussian sigma = T_cycle/4 samples (quarter-cycle smooth)
    T_cycle_samp = 2 * np.pi / fc * FS_WEEKLY   # samples per cycle
    sigma_adaptive = max(2.0, T_cycle_samp / 4.0)

    # PD: instantaneous phase derivative at peaks+troughs (no smoothing)
    t, f = measure_pd(z, fc, FS_WEEKLY, s_idx, e_idx,
                      gauss_sigma=0.0, use_troughs=True)
    meas_pd.append({'times_sm': t, 'freqs_sm': f, 'center': fc, 'idx': i,
                    'active': active_filters[i]})

    # PD-S: phase derivative + adaptive Gaussian smooth, sampled at peaks+troughs
    t, f = measure_pd(z, fc, FS_WEEKLY, s_idx, e_idx,
                      gauss_sigma=sigma_adaptive, use_troughs=True)
    meas_pds.append({'times_sm': t, 'freqs_sm': f, 'center': fc, 'idx': i,
                     'active': active_filters[i]})

    # PDC: cycle-averaged phase derivative at each peak (1 pt/cycle, robust)
    t, f = measure_pdc(z, fc, FS_WEEKLY, s_idx, e_idx)
    meas_pdc.append({'times_sm': t, 'freqs_sm': f, 'center': fc, 'idx': i,
                     'active': active_filters[i]})

    # PT_TP + 5pt MA  (reference best from brute-force)
    t_pt, f_pt = scheme_PT_TP(pk, tr, FS_WEEKLY)
    t_pt, f_pt = clip_to_window(t_pt, f_pt, s_idx, e_idx, fc)
    t_sm, f_sm = smooth_ma(t_pt, f_pt, SMOOTH_N5)
    meas_pttp.append({'times_sm': t_sm, 'freqs_sm': f_sm, 'center': fc, 'idx': i,
                      'active': active_filters[i]})
    meas_ag.append({'times_sm': t_sm, 'freqs_sm': f_sm, 'center': fc, 'idx': i,
                    'active': active_filters[i]})

for tag, lst in [('PD',meas_pd), ('PD-S',meas_pds), ('PDC',meas_pdc), ('PT_TP-5',meas_pttp)]:
    tot = sum(len(m['times_sm']) for m in lst)
    print(f"  {tag:8s}: {tot:4d} pts total  ({tot//N_FILTERS:2d} avg/filter)")


# ============================================================================
# DRAW HELPER
# ============================================================================

def draw_fvt(ax, meas_list, use_gating=False,
             markersize=1.8, linewidth=0.75, alpha=0.88,
             show_labels=True):
    for m in meas_list:
        t = m['times_sm']
        f = m['freqs_sm']
        if len(t) == 0:
            continue
        if use_gating and not m['active']:
            continue   # suppress low-amplitude filters
        color = COLORS[m['idx']]
        ax.plot(t, f, '-o', color=color,
                markersize=markersize, linewidth=linewidth, alpha=alpha, zorder=3)
        if show_labels and len(t):
            ax.text(t[0]-2,  f[0],  str(m['idx']+1), fontsize=5.5,
                    color=color, ha='right', va='center')
            ax.text(t[-1]+2, f[-1], str(m['idx']+1), fontsize=5.5,
                    color=color, ha='left',  va='center')

    for ref in [8, 9, 10, 11, 12]:
        ax.axhline(ref, color='silver', linewidth=0.5, zorder=1)
    ax.set_xlim(0, n_weeks)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xticks(np.arange(0, n_weeks+1, 25))
    ax.set_yticks([8, 9, 10, 11, 12])
    ax.set_xlabel('Weeks', fontsize=10)
    ax.set_ylabel('Radians/Year', fontsize=10)
    ax.grid(True, axis='x', alpha=0.2)
    ax.text(0.01, 0.99, DATE_DISPLAY_START, transform=ax.transAxes,
            fontsize=7, va='top', color='gray')
    ax.text(0.99, 0.99, DATE_DISPLAY_END, transform=ax.transAxes,
            fontsize=7, va='top', ha='right', color='gray')


# ============================================================================
# FIGURE 1: FOUR-PANEL COMPARISON
# Reference | PD at peaks | PD-S (Gauss smooth) | PT_TP 5pt MA (reference best)
# ============================================================================

fig1, axes1 = plt.subplots(1, 4, figsize=(36, 9), gridspec_kw={'wspace': 0.12})

if os.path.exists(REF_IMAGE):
    axes1[0].imshow(mpimg.imread(REF_IMAGE))
axes1[0].axis('off')
axes1[0].set_title("Hurst's Original AI-4", fontsize=10, fontweight='bold')

n_pdc_active = sum(len(m['times_sm']) for m in meas_pdc if m['active'])
draw_fvt(axes1[1], meas_pdc, use_gating=False)
axes1[1].set_title(
    f'PDC: Cycle-avg phase deriv at peaks\n(all filters, ~{sum(len(m["times_sm"]) for m in meas_pdc)//N_FILTERS} pts/filter)',
    fontsize=9.5, fontweight='bold')

draw_fvt(axes1[2], meas_pdc, use_gating=True)
n_active = sum(1 for m in meas_pdc if m['active'])
axes1[2].set_title(
    f'PDC + amplitude gate ({n_active}/23 filters)\n(gap filters removed: FC-17, FC-20)',
    fontsize=9.5, fontweight='bold')

draw_fvt(axes1[3], meas_pttp)
axes1[3].set_title(
    f'PT+TP + {SMOOTH_N5}-pt MA (best from brute-force)\n(~{sum(len(m["times_sm"]) for m in meas_pttp)//N_FILTERS} pts/filter)',
    fontsize=9.5, fontweight='bold')

fig1.suptitle(
    'FIGURE AI-4  |  Phase-Derivative Methods vs Hurst Reference\n'
    'PDC = cycle-avg d(phi)/dt at peaks  |  PD-S = Gauss-smoothed at peaks+troughs  |  Weekly Ormsby FIR',
    fontsize=11, fontweight='bold', y=1.01)
out1 = os.path.join(SCRIPT_DIR, 'fig_AI4_phase_deriv_4panel.png')
fig1.savefig(out1, dpi=130, bbox_inches='tight')
plt.close(fig1)
print(f"\nSaved: {out1}")


# ============================================================================
# FIGURE 2: 5-PANEL DIAGNOSTIC FOR ONE FILTER (FC-20, ~11.4 rad/yr)
# Waveform | Wrapped phase | Unwrapped phase | omega_inst | FVT comparison
# ============================================================================

DIAG_IDX   = 19   # FC-20 (0-indexed)
fc_d       = specs[DIAG_IDX]['f_center']
z_d        = outputs[DIAG_IDX]['signal']
sigma_diag = max(2.0, 2*np.pi / fc_d * FS_WEEKLY / 4.0)   # adaptive sigma for FC-20

seg_start = max(0, s_idx - 50)
seg_end   = min(len(z_d), e_idx + 50)
z_seg     = z_d[seg_start:seg_end]
t_seg     = (np.arange(len(z_seg)) + seg_start - s_idx)   # weeks from display start

phi_wrapped   = np.angle(z_seg)
phi_unwrapped = np.unwrap(np.angle(z_seg))
omega_inst_d  = np.diff(phi_unwrapped) * FS_WEEKLY    # rad/yr
omega_sm_d    = gaussian_filter1d(omega_inst_d, sigma=sigma_diag)

fig2, axes2 = plt.subplots(5, 1, figsize=(14, 18), sharex=False,
                            gridspec_kw={'hspace': 0.45})

# Panel 1: Waveform
ax = axes2[0]
ax.plot(t_seg, z_seg.real, 'b-', linewidth=0.7, label='real')
ax.plot(t_seg, np.abs(z_seg), 'r--', linewidth=0.6, alpha=0.6, label='envelope')
ax.plot(t_seg, -np.abs(z_seg), 'r--', linewidth=0.6, alpha=0.6)
ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
ax.axvline(n_weeks, color='gray', linewidth=0.5, linestyle=':')
ax.set_ylabel('Amplitude', fontsize=9)
ax.legend(fontsize=8, loc='upper right')
ax.set_title(f'FC-{DIAG_IDX+1} ({fc_d:.1f} rad/yr)  Waveform + Envelope', fontsize=10)
ax.grid(True, alpha=0.2)

# Panel 2: Wrapped phase
ax = axes2[1]
ax.plot(t_seg, np.degrees(phi_wrapped), 'g-', linewidth=0.6)
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('Wrapped phase (deg)', fontsize=9)
ax.set_title('Wrapped Phase  angle(z)', fontsize=10)
ax.set_ylim(-200, 200)
ax.grid(True, alpha=0.2)

# Panel 3: Unwrapped phase (should be nearly linear slope = fc)
ax = axes2[2]
# Remove linear trend for clarity
slope = fc_d / FS_WEEKLY   # expected rad/sample
n_arr = np.arange(len(phi_unwrapped))
phi_detrended = phi_unwrapped - slope * n_arr
ax.plot(t_seg, phi_detrended, 'm-', linewidth=0.6)
ax.set_ylabel('Phase residual (rad)', fontsize=9)
ax.set_title(f'Unwrapped phase minus linear trend (slope={slope:.4f} rad/sample)', fontsize=10)
ax.grid(True, alpha=0.2)

# Panel 4: Instantaneous frequency
ax = axes2[3]
ax.plot(t_seg[:-1], omega_inst_d, 'k-', linewidth=0.5, alpha=0.5, label='raw omega_inst')
ax.plot(t_seg[:-1], omega_sm_d,   'b-', linewidth=1.0, alpha=0.9, label=f'Gauss(sigma={sigma_diag:.1f})')
ax.axhline(fc_d, color='red', linewidth=0.6, linestyle='--', label=f'fc={fc_d:.1f}')
ax.set_ylabel('omega_inst (rad/yr)', fontsize=9)
ax.set_ylim(fc_d*(1-CLIP_FRAC*0.8), fc_d*(1+CLIP_FRAC*0.8))
ax.legend(fontsize=8)
ax.set_title('Instantaneous Frequency  d(unwrap(phi))/dt', fontsize=10)
ax.grid(True, alpha=0.2)
ax.set_xlim(t_seg[0], t_seg[-1])

# Panel 5: FVT comparison (PD vs PT_TP-best)
ax = axes2[4]
m_pd   = meas_pd[DIAG_IDX]
m_pds  = meas_pds[DIAG_IDX]
m_pt   = meas_pttp[DIAG_IDX]
if len(m_pd['times_sm']):
    ax.plot(m_pd['times_sm'],  m_pd['freqs_sm'],  '-o', color='navy',   markersize=4, linewidth=1.0, label='PD raw')
if len(m_pds['times_sm']):
    ax.plot(m_pds['times_sm'], m_pds['freqs_sm'], '-s', color='blue',   markersize=3, linewidth=0.8, label=f'PD-S(adaptive sigma)')
if len(m_pt['times_sm']):
    ax.plot(m_pt['times_sm'],  m_pt['freqs_sm'],  '-^', color='orange', markersize=3, linewidth=0.8, label='PT+TP 5pt MA')
ax.axhline(fc_d, color='red', linewidth=0.6, linestyle='--', label=f'fc={fc_d:.1f}')
ax.set_ylim(YMIN, YMAX)
ax.set_xlim(0, n_weeks)
ax.set_xlabel('Weeks', fontsize=9)
ax.set_ylabel('Frequency (rad/yr)', fontsize=9)
ax.set_title('FVT Comparison: PD vs PD-S vs PT+TP-5pt', fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
ax.set_xticks(np.arange(0, n_weeks+1, 25))

fig2.suptitle(
    f'FC-{DIAG_IDX+1} Diagnostic: Phase Derivative Analysis\n'
    f'Center: {fc_d:.2f} rad/yr  |  Weekly Ormsby  |  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}',
    fontsize=11, fontweight='bold')
out2 = os.path.join(SCRIPT_DIR, 'fig_AI4_phase_deriv_filter.png')
fig2.savefig(out2, dpi=130, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: {out2}")


# ============================================================================
# FIGURE 3: AMPLITUDE-GATED HURST-STYLE PANEL
# Same as fig_AI4_final.png but with low-amplitude filters suppressed
# Expected: FC-8, FC-12, FC-16 (or similar) disappear
# ============================================================================

fig3, axes3 = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw={'wspace': 0.12})

# Left: ungated (all filters)
draw_fvt(axes3[0], meas_ag, use_gating=False)
axes3[0].set_title(
    'All 23 Filters (no amplitude gating)\nPT+TP + 5pt MA',
    fontsize=10, fontweight='bold')

# Right: gated (suppress low-RMS filters)
draw_fvt(axes3[1], meas_ag, use_gating=True)
n_gated = sum(1 for m in meas_ag if not m['active'])
axes3[1].set_title(
    f'Amplitude-Gated (threshold={AMP_THRESH*100:.0f}% median RMS)\n'
    f'{n_gated} filters suppressed  (expect FC-8, FC-12, FC-16)',
    fontsize=10, fontweight='bold')

# Annotate which filters were gated
for m in meas_ag:
    if not m['active']:
        fc = m['center']
        # Mark on y-axis
        axes3[1].axhline(fc, color=COLORS[m['idx']], linewidth=0.8,
                         linestyle=':', alpha=0.5)
        axes3[1].text(n_weeks * 0.5, fc + 0.08,
                      f'FC-{m["idx"]+1} ({fc:.1f}) GATED',
                      fontsize=6.5, color=COLORS[m['idx']], ha='center', va='bottom')

fig3.suptitle(
    f'FIGURE AI-4  |  Amplitude Gating to Reproduce Hurst\'s Missing Filters (8, 12, 16)\n'
    f'Threshold = {AMP_THRESH*100:.0f}% of median RMS  |  Weekly DJIA  |  Ormsby FIR',
    fontsize=11, fontweight='bold', y=1.01)
out3 = os.path.join(SCRIPT_DIR, 'fig_AI4_amplitude_gated.png')
fig3.savefig(out3, dpi=130, bbox_inches='tight')
plt.close(fig3)
print(f"Saved: {out3}")

print("\nDone.")
print()
print("Summary of gated filters:")
for i, (m, active) in enumerate(zip(meas_ag, active_filters)):
    if not active:
        print(f"  FC-{i+1:2d} ({specs[i]['f_center']:.1f} r/y)  RMS={rms_per_filter[i]:.4f}"
              f"  ({rms_per_filter[i]/rms_median*100:.0f}% of median) -- GATED")
