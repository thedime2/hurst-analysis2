# -*- coding: utf-8 -*-
"""
Envelope Diagnosis: Ormsby vs CMW -- what causes the visual difference?

KEY FINDING (from diagnose_envelope.py):
  - Pure tone at fc:     Ormsby CV = 0.0%,  CMW CV = 0.0%  --> BOTH CORRECT
  - Two tones fc+adj:    Ormsby CV = 1.9%,  CMW CV = 4.3%  --> CMW leaks MORE
  - DJIA signal:         Ormsby CV = 32-53%, CMW CV = 33-54% --> SIMILAR TOTAL

Conclusion: Neither filter has an implementation bug. The difference in VISUAL
smoothness comes from the SPECTRAL RESPONSE SHAPE:

  Ormsby (flat-top 0.2 rad/yr passband + cosine skirts):
    - Captures ALL energy within passband with EQUAL weight
    - May straddle 2 Hurst lines (adjacent lines often partially inside passband+skirt)
    - Multiple beat pairs at Deltaw, 2*Deltaw, ... rad/yr --> IRREGULAR envelope texture
    - Flat top = no frequency preference within band

  CMW (Gaussian FWHM=0.5 rad/yr):
    - Weights energy by DISTANCE from center (Gaussian decay)
    - Only the NEAREST spectral line dominates (others attenuated exponentially)
    - Primarily ONE beat pair (fc vs nearest line) --> SMOOTH sinusoidal AM envelope
    - Gaussian = maximum time-frequency uncertainty reduction (Heisenberg optimal)

Both correctly compute the analytic signal. CMW envelopes LOOK smoother because
Gaussian weighting creates clean single-beat-frequency AM rather than multi-beat noise.

Outputs:
  fig_AI4_env_freq_response.png  -- Ormsby vs CMW passband detail + spectral line positions
  fig_AI4_env_multitone.png      -- Synthetic multi-tone input: envelope comparison
  fig_AI4_env_summary.png        -- Summary: pure tone, two-tone, DJIA envelopes
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from utils_ai import design_comb_bank, make_ormsby_kernels, load_weekly_data, get_window, FS_WEEKLY, NW_WEEKLY
from src.filters.funcOrmsby import apply_ormsby_filter
from src.time_frequency.cmw import ormsby_spec_to_cmw_params, apply_cmw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)
close, dates = load_weekly_data()
s_idx, e_idx = get_window(dates)
N   = len(close)
t_n = np.arange(N)
t_disp = np.arange(e_idx - s_idx)   # 0..268 weeks

# Pick FC-8 (fc=9.0) as demo filter -- sits in a spectral gap
IDX   = 7
fc    = specs[IDX]['f_center']     # rad/yr
h     = filters[IDX]['kernel']
p     = ormsby_spec_to_cmw_params(specs[IDX])

print(f"Demo filter: FC-{IDX+1}  fc={fc:.1f} rad/yr")
print(f"  Ormsby: passband [{specs[IDX]['f2']:.1f}, {specs[IDX]['f3']:.1f}]  skirt [{specs[IDX]['f1']:.1f}, {specs[IDX]['f4']:.1f}]")
print(f"  CMW:    FWHM={p['fwhm']:.3f}  sigma={p['sigma_f']:.3f} rad/yr")


# ============================================================================
# FIGURE 1: FREQUENCY RESPONSE -- showing which Hurst lines each filter captures
# ============================================================================

NFFT = 131072
H_orm_full = np.fft.fft(h, n=NFFT)
freqs_rad   = np.fft.fftfreq(NFFT, d=1.0/FS_WEEKLY) * 2*np.pi
pos_mask    = (freqs_rad >= 0) & (freqs_rad <= 13)
f_pos       = freqs_rad[pos_mask]

mag_orm = np.abs(H_orm_full[pos_mask])
pk_orm  = np.max(mag_orm)
mag_orm_norm = mag_orm / pk_orm

sigma_f = p['sigma_f']
mag_cmw = np.exp(-0.5 * ((f_pos - fc) / sigma_f)**2)

# Ormsby response in dB (clipped at -80 dB for clarity)
mag_orm_db = np.clip(20*np.log10(mag_orm_norm + 1e-10), -80, 3)
mag_cmw_db = np.clip(20*np.log10(mag_cmw + 1e-10), -80, 3)

# Hurst spectral lines in range
hurst_n = np.arange(1, 40)
hurst_freqs = hurst_n * 0.3676
hurst_in_range = hurst_freqs[(hurst_freqs >= 6.5) & (hurst_freqs <= 13)]

fig1, axes1 = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'wspace': 0.12})

# --- Left: linear scale (passband detail) ---
ax = axes1[0]
f_zoom = (f_pos >= fc - 1.2) & (f_pos <= fc + 1.2)
f_z = f_pos[f_zoom]
ax.plot(f_z, mag_orm_norm[f_zoom], 'b-', linewidth=2.0, label='Ormsby (flat-top + linear skirts)')
ax.plot(f_z, mag_cmw[f_zoom],      'r-', linewidth=2.0, label=f'CMW Gaussian (FWHM={p["fwhm"]:.2f} rad/yr)')
ax.fill_between(f_z, 0, mag_orm_norm[f_zoom], alpha=0.12, color='blue')
ax.fill_between(f_z, 0, mag_cmw[f_zoom],      alpha=0.12, color='red')

for hl in hurst_in_range:
    if fc - 1.2 <= hl <= fc + 1.2:
        g_orm = np.interp(hl, f_pos, mag_orm_norm)
        g_cmw = np.interp(hl, f_pos, mag_cmw)
        ax.axvline(hl, color='green', linewidth=1.0, linestyle='--', alpha=0.7)
        ax.plot(hl, g_orm, 'bs', markersize=7, zorder=5)
        ax.plot(hl, g_cmw, 'r^', markersize=7, zorder=5)
        n_idx = int(round(hl / 0.3676))
        ax.text(hl, -0.08, f'n={n_idx}\n{hl:.2f}', fontsize=7, ha='center', color='darkgreen')
        ax.text(hl + 0.02, g_orm + 0.03, f'{g_orm:.2f}', fontsize=7.5, color='blue')
        ax.text(hl + 0.02, g_cmw - 0.07, f'{g_cmw:.2f}', fontsize=7.5, color='red')

ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlim(fc - 1.2, fc + 1.2)
ax.set_ylim(-0.15, 1.12)
ax.set_xlabel('Frequency (rad/yr)', fontsize=10)
ax.set_ylabel('Normalised gain', fontsize=10)
ax.set_title(f'FC-{IDX+1} (fc={fc:.1f} rad/yr) -- Linear Scale\nGreen dashes = Hurst spectral lines (n x 0.3676)', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

# --- Right: dB scale (full picture) ---
ax = axes1[1]
f_zoom2 = (f_pos >= 6.5) & (f_pos <= 13)
ax.plot(f_pos[f_zoom2], mag_orm_db[f_zoom2], 'b-', linewidth=1.5, label='Ormsby')
ax.plot(f_pos[f_zoom2], mag_cmw_db[f_zoom2], 'r-', linewidth=1.5, label='CMW Gaussian')
for hl in hurst_in_range:
    ax.axvline(hl, color='green', linewidth=0.7, linestyle='--', alpha=0.5)
for other_fc in [s['f_center'] for s in specs]:
    ax.axvline(other_fc, color='gray', linewidth=0.3, alpha=0.3)

ax.axhline(-3, color='orange', linewidth=0.8, linestyle=':', alpha=0.7, label='-3 dB')
ax.axhline(-20, color='purple', linewidth=0.8, linestyle=':', alpha=0.7, label='-20 dB')
ax.set_xlim(6.5, 13)
ax.set_ylim(-82, 5)
ax.set_xlabel('Frequency (rad/yr)', fontsize=10)
ax.set_ylabel('Gain (dB)', fontsize=10)
ax.set_title(f'FC-{IDX+1} -- dB Scale (0 to 13 rad/yr)\nGreen = Hurst lines, Gray = other filter centers', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

fig1.suptitle(
    f'Ormsby vs CMW Frequency Response  |  FC-{IDX+1} fc={fc:.1f} rad/yr\n'
    'Key: How many Hurst spectral lines fall within each filter\'s response?',
    fontsize=11, fontweight='bold')

out1 = os.path.join(SCRIPT_DIR, 'fig_AI4_env_freq_response.png')
fig1.savefig(out1, dpi=130, bbox_inches='tight')
plt.close(fig1)
print(f"Saved: {out1}")


# ============================================================================
# FIGURE 2: SYNTHETIC MULTI-TONE -- envelope texture comparison
# Ormsby: multiple beats → irregular; CMW: one dominant beat → smooth
# ============================================================================

# Build a realistic synthetic DJIA-like signal:
# Multiple Hurst lines in the 7.6-12 rad/yr range with 1/omega amplitude scaling
np.random.seed(42)
syn_phases = np.random.uniform(0, 2*np.pi, 30)
syn_signal = np.zeros(N)
for k, (hn, ph) in enumerate(zip(hurst_n, syn_phases)):
    hf = hn * 0.3676
    amp = 1.0 / hf    # k/omega amplitude law
    syn_signal += amp * np.cos(hf / FS_WEEKLY * t_n + ph)

# Apply both filters
z_o_syn = apply_ormsby_filter(syn_signal, h, mode='reflect', fs=FS_WEEKLY)['signal']
z_c_syn = apply_cmw(syn_signal, p['f0'], p['fwhm'], FS_WEEKLY, analytic=True)['signal']

env_o_syn = np.abs(z_o_syn[s_idx:e_idx])
env_c_syn = np.abs(z_c_syn[s_idx:e_idx])

z_o_djia = apply_ormsby_filter(close, h, mode='reflect', fs=FS_WEEKLY)['signal']
z_c_djia = apply_cmw(close, p['f0'], p['fwhm'], FS_WEEKLY, analytic=True)['signal']

env_o_djia = np.abs(z_o_djia[s_idx:e_idx])
env_c_djia = np.abs(z_c_djia[s_idx:e_idx])

fig2, axes2 = plt.subplots(2, 2, figsize=(18, 10), gridspec_kw={'wspace': 0.12, 'hspace': 0.4})

def cv(e):
    m = np.mean(e)
    return np.std(e)/m*100 if m > 1e-12 else 0

for col, (label, eo, ec) in enumerate([
        (f'Synthetic multi-tone (all n*0.3676 lines, 1/omega amp)',
         env_o_syn, env_c_syn),
        ('DJIA signal (1934-1940)', env_o_djia, env_c_djia)]):

    ax_top = axes2[0, col]
    ax_bot = axes2[1, col]

    # Top: waveform + envelope
    sig_o = z_o_djia[s_idx:e_idx].real if col else z_o_syn[s_idx:e_idx].real
    sig_c = z_c_djia[s_idx:e_idx].real if col else z_c_syn[s_idx:e_idx].real

    ax_top.plot(t_disp, sig_o, 'b-', linewidth=0.5, alpha=0.6)
    ax_top.plot(t_disp, eo,   'b-', linewidth=1.5, label=f'Ormsby CV={cv(eo):.1f}%')
    ax_top.plot(t_disp, -eo,  'b-', linewidth=1.5, alpha=0.3)
    ax_top.plot(t_disp, sig_c, 'r-', linewidth=0.5, alpha=0.6)
    ax_top.plot(t_disp, ec,   'r-', linewidth=1.5, label=f'CMW CV={cv(ec):.1f}%')
    ax_top.plot(t_disp, -ec,  'r-', linewidth=1.5, alpha=0.3)
    ax_top.set_title(f'FC-{IDX+1} ({fc:.1f} r/y) -- {label}\nWaveform (thin) + Envelope (thick)', fontsize=9.5, fontweight='bold')
    ax_top.set_ylabel('Amplitude', fontsize=9)
    ax_top.set_xticks(np.arange(0, len(t_disp)+1, 25))
    ax_top.legend(fontsize=9)
    ax_top.grid(True, axis='x', alpha=0.2)

    # Bottom: envelope only (zoomed to compare texture)
    ax_bot.plot(t_disp, eo / np.max(eo), 'b-', linewidth=1.3, label=f'Ormsby CV={cv(eo):.1f}%')
    ax_bot.plot(t_disp, ec / np.max(ec), 'r-', linewidth=1.3, label=f'CMW CV={cv(ec):.1f}%')
    ax_bot.set_title('Normalised envelope (shape comparison)', fontsize=9.5, fontweight='bold')
    ax_bot.set_xlabel('Weeks', fontsize=9)
    ax_bot.set_ylabel('Norm. envelope', fontsize=9)
    ax_bot.set_xticks(np.arange(0, len(t_disp)+1, 25))
    ax_bot.legend(fontsize=9)
    ax_bot.grid(True, axis='x', alpha=0.2)
    ax_bot.set_ylim(0, 1.05)

fig2.suptitle(
    f'Envelope Texture: Ormsby (flat-top) vs CMW (Gaussian)  |  FC-{IDX+1} fc={fc:.1f} rad/yr\n'
    'Ormsby: multiple beat pairs -> irregular envelope  |  CMW: dominant-line weighting -> smoother AM',
    fontsize=11, fontweight='bold')

out2 = os.path.join(SCRIPT_DIR, 'fig_AI4_env_multitone.png')
fig2.savefig(out2, dpi=130, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: {out2}")


# ============================================================================
# FIGURE 3: SUMMARY TABLE -- all 23 filters, CV for pure tone / DJIA
# ============================================================================

print("\nComputing summary for all 23 filters...")
cv_orm_pure  = []
cv_cmw_pure  = []
cv_orm_djia  = []
cv_cmw_djia  = []

for i in range(len(specs)):
    fc_i  = specs[i]['f_center']
    h_i   = filters[i]['kernel']
    p_i   = ormsby_spec_to_cmw_params(specs[i])

    pure = np.cos(fc_i / FS_WEEKLY * t_n)

    zo_p = apply_ormsby_filter(pure,  h_i, mode='reflect', fs=FS_WEEKLY)['signal'][s_idx:e_idx]
    zc_p = apply_cmw(pure,  p_i['f0'], p_i['fwhm'], FS_WEEKLY, analytic=True)['signal'][s_idx:e_idx]
    zo_d = apply_ormsby_filter(close, h_i, mode='reflect', fs=FS_WEEKLY)['signal'][s_idx:e_idx]
    zc_d = apply_cmw(close, p_i['f0'], p_i['fwhm'], FS_WEEKLY, analytic=True)['signal'][s_idx:e_idx]

    cv_orm_pure.append(cv(np.abs(zo_p)))
    cv_cmw_pure.append(cv(np.abs(zc_p)))
    cv_orm_djia.append(cv(np.abs(zo_d)))
    cv_cmw_djia.append(cv(np.abs(zc_d)))

fc_all = [s['f_center'] for s in specs]

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'wspace': 0.12})

ax = axes3[0]
ax.bar(np.array(range(23)) - 0.2, cv_orm_pure, 0.35, color='blue', alpha=0.7, label='Ormsby')
ax.bar(np.array(range(23)) + 0.2, cv_cmw_pure, 0.35, color='red',  alpha=0.7, label='CMW')
ax.set_xticks(range(23))
ax.set_xticklabels([f'{f:.1f}' for f in fc_all], fontsize=6.5, rotation=45)
ax.set_xlabel('Filter center frequency (rad/yr)', fontsize=9)
ax.set_ylabel('Envelope CV%', fontsize=9)
ax.set_title('Pure sinusoid at fc\n(correct implementation = ~0% CV)', fontsize=10, fontweight='bold')
ax.axhline(1.0, color='gray', linewidth=0.7, linestyle='--', label='1% threshold')
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.25)
ax.set_ylim(0, 5)

ax = axes3[1]
ax.bar(np.array(range(23)) - 0.2, cv_orm_djia, 0.35, color='blue', alpha=0.7, label='Ormsby')
ax.bar(np.array(range(23)) + 0.2, cv_cmw_djia, 0.35, color='red',  alpha=0.7, label='CMW')
ax.set_xticks(range(23))
ax.set_xticklabels([f'{f:.1f}' for f in fc_all], fontsize=6.5, rotation=45)
ax.set_xlabel('Filter center frequency (rad/yr)', fontsize=9)
ax.set_ylabel('Envelope CV%', fontsize=9)
ax.set_title('DJIA signal 1934-1940\n(similar total variability; visual difference is texture)', fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.25)

# Add hurst spectral line markers
for ax in axes3:
    for hl in hurst_in_range:
        if 7.6 <= hl <= 12.0:
            ax.axvline([f['f_center'] for f in specs].index(
                min([s['f_center'] for s in specs], key=lambda f: abs(f-hl))),
                color='green', linewidth=0.5, alpha=0.4)

fig3.suptitle(
    'Ormsby vs CMW -- Envelope Variability (CV%) for All 23 Filters\n'
    'Pure tone: both ~0% (correct implementations) | DJIA: both similarly variable (real signal content)',
    fontsize=11, fontweight='bold')

out3 = os.path.join(SCRIPT_DIR, 'fig_AI4_env_summary.png')
fig3.savefig(out3, dpi=130, bbox_inches='tight')
plt.close(fig3)
print(f"Saved: {out3}")

print()
print("CONCLUSION:")
print("  - Neither filter has an implementation bug")
print("  - Pure tone at fc: both give CV < 0.5% (correct analytic signal)")
print("  - DJIA: both have similar TOTAL envelope variability (CV 30-55%)")
print("  - VISUAL difference (CMW looks smoother) is due to spectral TEXTURE:")
print("    * Ormsby flat-top: equal weight to all in-band energy")
print("      -> multiple Hurst lines captured -> many beat pairs -> irregular texture")
print("    * CMW Gaussian: exponentially decreasing weight with distance from center")
print("      -> one dominant line per filter -> single beat pair -> regular smooth AM")
print()
print("  The CMW Gaussian response IS the minimum time-frequency uncertainty")
print("  representation (Heisenberg-optimal). This naturally creates smooth envelopes.")
print("  Hurst may have used wider/smoother filters, which is why his AI-3 figure")
print("  shows smooth envelopes similar to CMW rather than the Ormsby texture.")
