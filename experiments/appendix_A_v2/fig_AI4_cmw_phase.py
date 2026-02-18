# -*- coding: utf-8 -*-
"""
Figure AI-4 CMW vs Ormsby Phase-Derivative Comparison

Applies the same measurement schemes (PT+TP + 5pt MA, PDC + amplitude gate)
to Complex Morlet Wavelet (CMW) filter outputs and compares with:
  - Hurst's original reference
  - Ormsby PT+TP + 5pt MA (our current best)

Key difference:
  - Ormsby: rectangular passband 0.2 rad/yr, skirt 0.3 rad/yr total
  - CMW:    Gaussian in freq domain, FWHM = 0.5 rad/yr (matched to Ormsby skirt midpoints)

The wider Gaussian CMW admits more energy from adjacent spectral lines, so the
filter output may carry more AM/FM from beating -- potentially explaining the
larger oscillation amplitude in Hurst's top cluster (FC-18..23).

Outputs:
  fig_AI4_cmw_4panel.png  -- Reference | Ormsby PT+TP-5 | CMW PT+TP-5 | CMW PDC+gate
  fig_AI4_cmw_vs_ormsby.png -- Side-by-side: Ormsby vs CMW (both best methods)
  fig_AI4_cmw_best.png    -- Single CMW best-method panel (Hurst style)

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figure AI-4 (Frequency vs Time)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks

from utils_ai import (
    load_weekly_data,
    design_comb_bank, make_ormsby_kernels, apply_comb_bank,
    get_window, FS_WEEKLY, NW_WEEKLY,
    DATE_DISPLAY_START, DATE_DISPLAY_END,
)
from src.time_frequency.cmw import ormsby_spec_to_cmw_params, apply_cmw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
REF_IMAGE  = os.path.join(BASE_DIR, 'references/appendix_a/figure_AI4_v2.png')

N_FILTERS  = 23
YMIN, YMAX = 7.4, 12.6
CLIP_FRAC  = 0.30
AMP_THRESH = 0.30   # amplitude gate: 30% of median RMS
SMOOTH_N5  = 5      # 5-pt MA for PT_TP

_cmap  = plt.colormaps['coolwarm']
COLORS = [_cmap(i / (N_FILTERS - 1)) for i in range(N_FILTERS)]


# ============================================================================
# PEAK / TROUGH UTILITIES (same as phase_deriv script)
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
# MEASUREMENT METHODS
# ============================================================================

def instantaneous_freq(z_analytic, fs):
    """d(unwrap(angle(z)))/dt in rad/yr. Returns N-1 length array."""
    phi = np.unwrap(np.angle(z_analytic))
    return np.diff(phi) * fs


def scheme_PT_TP(peaks, troughs, fs):
    """Half-period between every adjacent peak/trough (interleaved)."""
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


def measure_pttp5(z_analytic, f_center, fs, s_idx, e_idx):
    """PT+TP interleaved + 5-pt moving average."""
    sig_real = z_analytic.real
    pk = find_peaks_sub(sig_real, f_center, fs)
    tr = find_troughs_sub(sig_real, f_center, fs)
    t_pt, f_pt = scheme_PT_TP(pk, tr, fs)
    t_cl, f_cl = clip_to_window(t_pt, f_pt, s_idx, e_idx, f_center)
    return smooth_ma(t_cl, f_cl, SMOOTH_N5)


def sample_omega_at_events(omega_inst, events):
    """Linear interpolate omega_inst (length N-1) at fractional positions."""
    ev = events[(events >= 0) & (events < len(omega_inst) - 1)]
    if len(ev) == 0:
        return ev, np.array([])
    idx_lo = np.floor(ev).astype(int)
    idx_hi = np.minimum(idx_lo + 1, len(omega_inst) - 1)
    frac   = ev - idx_lo
    freqs  = omega_inst[idx_lo] * (1-frac) + omega_inst[idx_hi] * frac
    return ev, freqs


def measure_pdc(z_analytic, f_center, fs, s_idx, e_idx):
    """Cycle-averaged phase derivative: robust mean over ±T/2 per peak."""
    sig_real   = z_analytic.real
    omega_inst = instantaneous_freq(z_analytic, fs)
    pk_idx = find_peaks_sub(sig_real, f_center, fs)
    if len(pk_idx) < 2:
        return np.array([]), np.array([])
    T_samp_half = int(round(np.pi / f_center * fs))
    times, freqs = [], []
    for pk in pk_idx:
        lo = max(0, int(round(pk)) - T_samp_half)
        hi = min(len(omega_inst), int(round(pk)) + T_samp_half)
        if hi <= lo: continue
        seg = omega_inst[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        inliers = seg[np.abs(seg - med) < 5 * max(mad, 0.5)]
        if len(inliers) == 0: inliers = seg
        freqs.append(float(np.mean(inliers)))
        times.append(pk)
    return clip_to_window(np.array(times), np.array(freqs), s_idx, e_idx, f_center)


# ============================================================================
# LOAD DATA AND DESIGN FILTERS
# ============================================================================

print("Loading weekly DJIA data...")
close, dates = load_weekly_data()
s_idx, e_idx = get_window(dates)
n_weeks = e_idx - s_idx
print(f"Display window: {DATE_DISPLAY_START} to {DATE_DISPLAY_END}  ({n_weeks} weeks)")

# Ormsby comb bank specs and outputs
print("\nApplying Ormsby comb bank (23 filters, 7.6-12.0 rad/yr)...")
specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)
ormsby_outputs = apply_comb_bank(close, filters, fs=FS_WEEKLY)

# CMW matched parameters (FWHM from Ormsby skirt midpoints)
print("\nComputing CMW filter parameters (matched FWHM)...")
cmw_params = [ormsby_spec_to_cmw_params(s) for s in specs]
for i, (p, s) in enumerate(zip(cmw_params, specs)):
    if i == 0 or i == 22:
        print(f"  FC-{i+1:2d}: f0={p['f0']:.2f}  FWHM={p['fwhm']:.3f}  "
              f"sigma={p['sigma_f']:.3f} rad/yr  "
              f"(Ormsby passband={s['bandwidth']:.3f}, skirt={s['skirt_width']:.3f} rad/yr)")

# Apply CMW bank
print("\nApplying CMW bank (23 filters)...")
cmw_outputs = []
for i, params in enumerate(cmw_params):
    out = apply_cmw(close, params['f0'], params['fwhm'], FS_WEEKLY, analytic=True)
    out['spec'] = params
    out['index'] = i
    cmw_outputs.append(out)
    if (i+1) % 5 == 0 or i == 22:
        print(f"  Applied CMW filter {i+1}/{N_FILTERS}")

print()


# ============================================================================
# COMPUTE RMS FOR AMPLITUDE GATING (both Ormsby and CMW)
# ============================================================================

def compute_rms(outputs, key='signal'):
    """Display-window RMS for each filter."""
    rms = []
    for out in outputs:
        seg = out[key][s_idx:e_idx]
        if np.iscomplexobj(seg):
            seg = seg.real
        rms.append(float(np.sqrt(np.mean(seg**2))))
    return np.array(rms)

rms_ormsby = compute_rms(ormsby_outputs)
rms_cmw    = compute_rms(cmw_outputs)

med_ormsby = np.median(rms_ormsby)
med_cmw    = np.median(rms_cmw)

active_ormsby = rms_ormsby >= AMP_THRESH * med_ormsby
active_cmw    = rms_cmw    >= AMP_THRESH * med_cmw

print("Amplitude gating results:")
for i in range(N_FILTERS):
    fc = specs[i]['f_center']
    flg_o = '' if active_ormsby[i] else ' *GATED*'
    flg_c = '' if active_cmw[i]    else ' *GATED*'
    if not active_ormsby[i] or not active_cmw[i]:
        print(f"  FC-{i+1:2d} ({fc:.1f})  "
              f"Ormsby={rms_ormsby[i]/med_ormsby*100:.0f}%{flg_o}  "
              f"CMW={rms_cmw[i]/med_cmw*100:.0f}%{flg_c}")

n_gated_ormsby = sum(~active_ormsby)
n_gated_cmw    = sum(~active_cmw)
print(f"  Ormsby: {n_gated_ormsby} filters gated,  CMW: {n_gated_cmw} filters gated")
print()


# ============================================================================
# APPLY ALL MEASUREMENT SCHEMES TO BOTH FILTER BANKS
# ============================================================================

meas_ormsby_pttp5  = []
meas_cmw_pttp5     = []
meas_cmw_pdc       = []

print("Measuring frequencies...")
for i in range(N_FILTERS):
    fc = specs[i]['f_center']

    # Ormsby: PT+TP + 5pt MA
    z_orm = ormsby_outputs[i]['signal']
    t, f  = measure_pttp5(z_orm, fc, FS_WEEKLY, s_idx, e_idx)
    meas_ormsby_pttp5.append({'times_sm': t, 'freqs_sm': f, 'center': fc, 'idx': i,
                              'active': active_ormsby[i]})

    # CMW: PT+TP + 5pt MA
    z_cmw = cmw_outputs[i]['signal']
    t, f  = measure_pttp5(z_cmw, fc, FS_WEEKLY, s_idx, e_idx)
    meas_cmw_pttp5.append({'times_sm': t, 'freqs_sm': f, 'center': fc, 'idx': i,
                           'active': active_cmw[i]})

    # CMW: PDC (cycle-averaged phase derivative)
    t, f  = measure_pdc(z_cmw, fc, FS_WEEKLY, s_idx, e_idx)
    meas_cmw_pdc.append({'times_sm': t, 'freqs_sm': f, 'center': fc, 'idx': i,
                         'active': active_cmw[i]})

for tag, lst in [('Ormsby PT+TP-5', meas_ormsby_pttp5),
                 ('CMW   PT+TP-5', meas_cmw_pttp5),
                 ('CMW   PDC', meas_cmw_pdc)]:
    tot = sum(len(m['times_sm']) for m in lst)
    print(f"  {tag:16s}: {tot:4d} pts total  ({tot//N_FILTERS:2d} avg/filter)")

print()


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
            continue
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
# Reference | Ormsby PT+TP-5 | CMW PT+TP-5 | CMW PDC + gate
# ============================================================================

fig1, axes1 = plt.subplots(1, 4, figsize=(36, 9), gridspec_kw={'wspace': 0.12})

if os.path.exists(REF_IMAGE):
    axes1[0].imshow(mpimg.imread(REF_IMAGE))
axes1[0].axis('off')
axes1[0].set_title("Hurst's Original AI-4", fontsize=10, fontweight='bold')

n_pts_o = sum(len(m['times_sm']) for m in meas_ormsby_pttp5)
draw_fvt(axes1[1], meas_ormsby_pttp5)
axes1[1].set_title(
    f'Ormsby  PT+TP + {SMOOTH_N5}-pt MA\n'
    f'(passband=0.2, skirt=0.3 rad/yr | ~{n_pts_o//N_FILTERS} pts/filter)',
    fontsize=9.5, fontweight='bold')

n_pts_c = sum(len(m['times_sm']) for m in meas_cmw_pttp5)
draw_fvt(axes1[2], meas_cmw_pttp5)
fwhm_val = cmw_params[0]['fwhm']
axes1[2].set_title(
    f'CMW  PT+TP + {SMOOTH_N5}-pt MA\n'
    f'(FWHM={fwhm_val:.2f} rad/yr Gaussian | ~{n_pts_c//N_FILTERS} pts/filter)',
    fontsize=9.5, fontweight='bold')

n_pdc_active = sum(len(m['times_sm']) for m in meas_cmw_pdc if m['active'])
n_active = sum(1 for m in meas_cmw_pdc if m['active'])
n_pdc_all = sum(len(m['times_sm']) for m in meas_cmw_pdc)
draw_fvt(axes1[3], meas_cmw_pdc, use_gating=True)
axes1[3].set_title(
    f'CMW  PDC + amplitude gate\n'
    f'({n_active}/23 filters | ~{n_pdc_all//N_FILTERS} pts/filter before gate)',
    fontsize=9.5, fontweight='bold')

fig1.suptitle(
    'FIGURE AI-4  |  CMW vs Ormsby: Frequency-vs-Time Measurement Comparison\n'
    f'CMW FWHM={fwhm_val:.2f} rad/yr (Gaussian) vs Ormsby passband=0.2 rad/yr (rectangular)  |  Weekly DJIA',
    fontsize=11, fontweight='bold', y=1.01)

out1 = os.path.join(SCRIPT_DIR, 'fig_AI4_cmw_4panel.png')
fig1.savefig(out1, dpi=130, bbox_inches='tight')
plt.close(fig1)
print(f"Saved: {out1}")


# ============================================================================
# FIGURE 2: SIDE-BY-SIDE Ormsby vs CMW (both PT+TP-5)
# Cleaner focus on filter bandwidth effect
# ============================================================================

fig2, axes2 = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw={'wspace': 0.12})

draw_fvt(axes2[0], meas_ormsby_pttp5)
axes2[0].set_title(
    f'Ormsby FIR  |  PT+TP + {SMOOTH_N5}-pt MA\n'
    f'Passband=0.2 rad/yr  |  ~{n_pts_o//N_FILTERS} pts/filter\n'
    f'[flat-top rectangular + cosine skirts]',
    fontsize=10, fontweight='bold')

draw_fvt(axes2[1], meas_cmw_pttp5)
axes2[1].set_title(
    f'CMW Gaussian  |  PT+TP + {SMOOTH_N5}-pt MA\n'
    f'FWHM={fwhm_val:.2f} rad/yr  |  ~{n_pts_c//N_FILTERS} pts/filter\n'
    f'[Gaussian spectral response, wider = more adjacent line bleed]',
    fontsize=10, fontweight='bold')

fig2.suptitle(
    'FIGURE AI-4  |  Filter Type Comparison: Rectangular vs Gaussian Passband\n'
    'Same data, same measurement scheme, different spectral response shapes',
    fontsize=11, fontweight='bold', y=1.01)

out2 = os.path.join(SCRIPT_DIR, 'fig_AI4_cmw_vs_ormsby.png')
fig2.savefig(out2, dpi=130, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: {out2}")


# ============================================================================
# FIGURE 3: BEST CMW SINGLE PANEL (Hurst style)
# Use CMW PDC + amplitude gate as cleanest method
# ============================================================================

fig3, ax3 = plt.subplots(1, 1, figsize=(10, 7))

draw_fvt(ax3, meas_cmw_pdc, use_gating=True)
ax3.set_title(
    f'FIGURE AI-4  |  CMW Gaussian Filter Bank  |  PDC + Amplitude Gate\n'
    f'FWHM={fwhm_val:.2f} rad/yr  |  {n_active}/23 filters shown  |  '
    f'Weekly DJIA  {DATE_DISPLAY_START} to {DATE_DISPLAY_END}',
    fontsize=10, fontweight='bold')

out3 = os.path.join(SCRIPT_DIR, 'fig_AI4_cmw_best.png')
fig3.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"Saved: {out3}")


# ============================================================================
# FIGURE 4: AMPLITUDE ENVELOPE COMPARISON (CMW vs Ormsby for selected filters)
# Shows WHY CMW might have different FM: more adjacent line energy
# Plots the display-window waveform for FC-22 (top cluster, fc~11.8 rad/yr)
# ============================================================================

DIAG_FILTERS = [0, 10, 16, 20]   # FC-1, FC-11, FC-17, FC-21

fig4, axes4 = plt.subplots(len(DIAG_FILTERS), 2, figsize=(16, 4*len(DIAG_FILTERS)),
                            gridspec_kw={'wspace': 0.12, 'hspace': 0.45})

t_axis = np.arange(n_weeks)

for row, fi in enumerate(DIAG_FILTERS):
    fc = specs[fi]['f_center']

    # Ormsby
    z_o   = ormsby_outputs[fi]['signal'][s_idx:e_idx]
    env_o = np.abs(z_o)
    ax = axes4[row, 0]
    ax.fill_between(t_axis, -env_o, env_o, alpha=0.25, color=COLORS[fi])
    ax.plot(t_axis, z_o.real, color=COLORS[fi], linewidth=0.7)
    ax.set_title(f'FC-{fi+1} ({fc:.1f} r/y)  ORMSBY', fontsize=9)
    ax.set_ylim(-1.1*env_o.max(), 1.1*env_o.max())
    ax.set_ylabel('Amplitude', fontsize=8)
    ax.set_xticks(np.arange(0, n_weeks+1, 25))
    ax.grid(True, axis='x', alpha=0.2)

    # CMW
    z_c   = cmw_outputs[fi]['signal'][s_idx:e_idx]
    env_c = np.abs(z_c)
    ax = axes4[row, 1]
    ax.fill_between(t_axis, -env_c, env_c, alpha=0.25, color=COLORS[fi])
    ax.plot(t_axis, z_c.real, color=COLORS[fi], linewidth=0.7)
    ax.set_title(f'FC-{fi+1} ({fc:.1f} r/y)  CMW  FWHM={cmw_params[fi]["fwhm"]:.2f}', fontsize=9)
    ax.set_ylim(-1.1*env_c.max(), 1.1*env_c.max())
    ax.set_xticks(np.arange(0, n_weeks+1, 25))
    ax.grid(True, axis='x', alpha=0.2)

    if row == len(DIAG_FILTERS)-1:
        axes4[row, 0].set_xlabel('Weeks from display start', fontsize=9)
        axes4[row, 1].set_xlabel('Weeks from display start', fontsize=9)

fig4.suptitle(
    'Ormsby vs CMW Waveform Comparison (selected filters)\n'
    'Wider CMW FWHM -> more adjacent line energy -> different AM envelope',
    fontsize=11, fontweight='bold')

out4 = os.path.join(SCRIPT_DIR, 'fig_AI4_cmw_waveforms.png')
fig4.savefig(out4, dpi=120, bbox_inches='tight')
plt.close(fig4)
print(f"Saved: {out4}")


# ============================================================================
# PRINT SUMMARY TABLE
# ============================================================================

print()
print("=" * 65)
print("CMW vs Ormsby  --  AI-4 FVT Method Summary")
print("=" * 65)
print(f"{'Method':<28} {'pts/filter':>10}  {'notes'}")
print("-" * 65)

methods = [
    ('Ormsby PT+TP+5pt MA', meas_ormsby_pttp5, False),
    ('CMW   PT+TP+5pt MA', meas_cmw_pttp5,    False),
    ('CMW   PDC (all)', meas_cmw_pdc, False),
    ('CMW   PDC + amp gate', meas_cmw_pdc, True),
]
for label, lst, gate in methods:
    if gate:
        tot = sum(len(m['times_sm']) for m in lst if m['active'])
        n_shown = sum(1 for m in lst if m['active'])
        note = f'{n_shown}/23 filters'
    else:
        tot = sum(len(m['times_sm']) for m in lst)
        note = '23/23 filters'
    print(f"  {label:<26} {tot//N_FILTERS:>10}   {note}")

print("-" * 65)
print(f"  Hurst original (visual estimate)           ~7   ~21/23 filters")
print("=" * 65)
print()
print("CMW filter parameters (first / last):")
p0  = cmw_params[0];  s0  = specs[0]
p22 = cmw_params[22]; s22 = specs[22]
print(f"  FC-1  ({s0['f_center']:.1f} r/y):  FWHM={p0['fwhm']:.3f} rad/yr  "
      f"sigma_f={p0['sigma_f']:.3f}  (Ormsby BW={s0['bandwidth']:.1f} skirt={s0['skirt_width']:.1f})")
print(f"  FC-23 ({s22['f_center']:.1f} r/y):  FWHM={p22['fwhm']:.3f} rad/yr  "
      f"sigma_f={p22['sigma_f']:.3f}  (Ormsby BW={s22['bandwidth']:.1f} skirt={s22['skirt_width']:.1f})")
print()
print("Gated filters:")
for i in range(N_FILTERS):
    if not active_ormsby[i]:
        fc = specs[i]['f_center']
        print(f"  Ormsby: FC-{i+1:2d} ({fc:.1f}) gated  "
              f"({rms_ormsby[i]/med_ormsby*100:.0f}% of median RMS)")
for i in range(N_FILTERS):
    if not active_cmw[i]:
        fc = specs[i]['f_center']
        print(f"  CMW:    FC-{i+1:2d} ({fc:.1f}) gated  "
              f"({rms_cmw[i]/med_cmw*100:.0f}% of median RMS)")

print()
print("Done.")
