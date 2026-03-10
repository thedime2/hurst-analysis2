"""
Sinusoidal recovery diagnostics for complex analytic filter outputs.

Tests whether multi-mode MPM can recover the original frequencies,
amplitudes, and phases from a bandpass filter output, using:
  - SV profile (Hankel matrix) to count in-band components
  - Envelope beat spectrum to identify pairwise beat frequencies
  - Multi-mode MPM to extract {freq, amplitude, phase} per mode
  - Least-squares reconstruction to validate the model

Three signals tested per filter:
  1. Pure tone at fc                    → expect 1 mode, perfect recovery
  2. Two tones: fc and fc+0.3676 rad/yr → expect 2 modes, beat at 0.3676
  3. DJIA 1921-1965                     → unknown structure, explored here

Reference: Hurst Appendix A, comb filter analysis.
"""

import sys, os
sys.path.insert(0, os.path.abspath('../..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils_ai import (
    design_comb_bank, make_ormsby_kernels, load_weekly_data,
    get_window, FS_WEEKLY, NW_WEEKLY,
    DATE_ANALYSIS_START, DATE_ANALYSIS_END,
    sv_profile, estimate_n_modes, mpm_multimode,
    envelope_beat_spectrum, recover_band_sinusoids,
)
from src.filters import apply_ormsby_filter

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)
close, dates = load_weekly_data()

# Display window (for plotting)
s_idx, e_idx = get_window(dates)
t_dates = dates[s_idx:e_idx]

# Analysis window (Hurst 1921-1965) — used for MPM on DJIA
a_idx, b_idx = get_window(dates, DATE_ANALYSIS_START, DATE_ANALYSIS_END)

N   = len(close)
t_n = np.arange(N)

DELTA_F    = 0.3676    # rad/yr — spacing between the two synthetic tones
FILTER_IDX = [3, 11, 19]

# ---------------------------------------------------------------------------
# Helper: choose the signal segment to use for MPM.
# For both synthetic and DJIA we use the Hurst analysis window length
# (≈2297 samples = 44 yrs) for consistent frequency resolution and bounded
# computation.  The beat period for 0.3676 rad/yr is 17 yr = 884 samples,
# so ≈2297 samples gives ~2.5 beat cycles — enough for reliable separation.
# ---------------------------------------------------------------------------
_HURST_LEN = b_idx - a_idx   # ~2297 samples

def analysis_slice(is_djia):
    if is_djia:
        return (a_idx, b_idx)
    # Synthetic: centre a Hurst-length window in the middle of the record
    mid  = N // 2
    half = _HURST_LEN // 2
    return (max(0, mid - half), min(N, mid + half))


def fmt_component(c, true_freq=None):
    f = c['freq_radyr']
    T = abs(2 * np.pi / f) if abs(f) > 1e-6 else np.inf
    err = f'  df={f - true_freq:+.4f}' if true_freq is not None else ''
    return (f"  f={f:+7.3f} rad/yr  T={T:.3f}yr  "
            f"amp={c['amplitude']:.4f}  ph={np.degrees(c['phase_rad']):+7.1f}deg"
            f"  damp={c['damping']:.4f}{err}")


# ===========================================================================
# Main loop — one figure per filter
# ===========================================================================
for i in FILTER_IDX:
    spec    = specs[i]
    fc      = spec['f_center']
    h       = filters[i]['kernel']
    f1, f2, f3, f4 = spec['f1'], spec['f2'], spec['f3'], spec['f4']
    omega_s = fc / FS_WEEKLY    # rad/sample

    # Build test signals (full record length)
    pure1 = np.cos(omega_s * t_n)
    pure2 = (np.cos(omega_s * t_n)
             + 0.5 * np.cos((fc + DELTA_F) / FS_WEEKLY * t_n))

    tests = [
        # (label, signal, is_djia, true_freqs_radyr, true_amps)
        ('Pure tone at fc',            pure1, False, [fc],                [1.0]),
        (f'Two tones fc & fc+{DELTA_F}', pure2, False, [fc, fc + DELTA_F], [1.0, 0.5]),
        ('DJIA 1921-1965',             close, True,  None,                None),
    ]

    print(f'\n{"="*72}')
    print(f'FILTER {i+1}  fc={fc:.2f} rad/yr  T={2*np.pi/fc:.3f} yr')
    print(f'  Stopband  [{f1:.2f}, {f4:.2f}]  rad/yr')
    print(f'  Passband  [{f2:.2f}, {f3:.2f}]  rad/yr')
    print(f'{"="*72}')

    # -----------------------------------------------------------------------
    # Figure layout: 3 rows (signals) × 3 cols (beat / SV / recon)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f'Filter {i+1}  fc={fc:.2f} rad/yr  (T={2*np.pi/fc:.3f} yr)  —  '
        f'Sinusoidal Recovery Diagnostics',
        fontsize=11)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    for row, (label, sig, is_djia, true_freqs, true_amps) in enumerate(tests):

        # Full filter output
        z_full = apply_ormsby_filter(sig, h, mode='reflect',
                                     fs=FS_WEEKLY)['signal']

        # Segment used for MPM (short for DJIA stationarity, full for synth)
        pa, pb = analysis_slice(is_djia)
        z_mpm  = z_full[pa:pb]

        # Auto-estimate modes (cap at 6 for DJIA)
        n_est = estimate_n_modes(z_mpm, max_modes=6)

        # Full recovery
        result = recover_band_sinusoids(z_mpm, spec, FS_WEEKLY,
                                        n_modes=n_est, max_modes=6)

        # Envelope beat spectrum (full signal for best resolution)
        beat_f, beat_p = envelope_beat_spectrum(z_full, FS_WEEKLY)

        # ---- Col 0: Beat spectrum ----------------------------------------
        ax0 = fig.add_subplot(gs[row, 0])
        # Show only up to 1.5× filter width
        bw = f4 - f1
        ax0.semilogy(beat_f, beat_p + 1e-8, 'k-', lw=0.7)
        ax0.set_xlim(0, bw * 1.5)
        ax0.set_ylim(1e-6, 2)
        if true_freqs is not None and len(true_freqs) == 2:
            df_beat = abs(true_freqs[1] - true_freqs[0])
            ax0.axvline(df_beat, color='r', lw=1.5, ls='--',
                        label=f'Beat {df_beat:.4f}')
            ax0.legend(fontsize=7)
        ax0.set_xlabel('Beat freq (rad/yr)', fontsize=8)
        ax0.set_ylabel('Power (norm)', fontsize=8)
        ax0.set_title(f'{label}\nBeat spectrum', fontsize=8)
        ax0.grid(True, alpha=0.3, which='both')
        ax0.tick_params(labelsize=7)

        # ---- Col 1: SV profile -------------------------------------------
        ax1 = fig.add_subplot(gs[row, 1])
        sv  = sv_profile(z_mpm)
        sv_norm = sv / (sv[0] + 1e-30)
        n_show  = min(len(sv_norm), 12)
        ax1.semilogy(np.arange(1, n_show + 1), sv_norm[:n_show],
                     'o-', ms=5, lw=1.2)
        ax1.axvline(n_est + 0.5, color='r', lw=1.5, ls='--',
                    label=f'{n_est} modes')
        # Mark expected true modes for synthetic
        if true_freqs is not None:
            ax1.axvline(len(true_freqs) + 0.5, color='g', lw=1, ls=':',
                        label=f'{len(true_freqs)} true')
        ax1.legend(fontsize=7)
        ax1.set_xlabel('Mode index', fontsize=8)
        ax1.set_ylabel('Norm. singular value', fontsize=8)
        ax1.set_title(f'SV profile  (est. {n_est} modes)', fontsize=8)
        ax1.set_xticks(np.arange(1, n_show + 1))
        ax1.grid(True, alpha=0.3, which='both')
        ax1.tick_params(labelsize=7)

        # ---- Col 2: Reconstruction (display window) ----------------------
        ax2 = fig.add_subplot(gs[row, 2])
        z_win     = z_full[s_idx:e_idx]
        # Shift reconstruction indices if MPM was on a sub-slice
        rec_full = np.zeros(N, dtype=complex)
        n_vec_pa = np.arange(pb - pa)
        for c in result['components']:
            omega_samp = c['freq_radyr'] / FS_WEEKLY
            seg = c['amplitude'] * np.exp(
                1j * (omega_samp * n_vec_pa + c['phase_rad']))
            rec_full[pa:pb] += seg
        z_rec_win = rec_full[s_idx:e_idx]

        ax2.plot(t_dates, z_win.real,     'k-',  lw=0.8, alpha=0.7,
                 label='Filter output')
        ax2.plot(t_dates, z_rec_win.real, 'r--', lw=1.2,
                 label=f'MPM ({n_est}-mode)  SNR={result["snr_db"]:.0f}dB')
        ax2.plot(t_dates, np.abs(z_win),  'b-',  lw=1.0, alpha=0.6,
                 label='Envelope')
        ymax = np.percentile(np.abs(z_win), 99) * 1.3
        ax2.set_ylim(-ymax, ymax)
        ax2.legend(fontsize=7, loc='upper right')
        ax2.set_xlabel('Date', fontsize=8)
        ax2.set_ylabel('Amplitude', fontsize=8)
        ax2.set_title(f'Reconstruction  SNR={result["snr_db"]:.1f} dB',
                      fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=7)

        # ---- Console output ----------------------------------------------
        print(f'\n  [{label}]  est. modes={n_est}  '
              f'SNR={result["snr_db"]:.1f} dB  '
              f'(MPM on {pb-pa} samples)')
        for k, c in enumerate(result['components']):
            in_b = 'IN ' if f1 <= abs(c['freq_radyr']) <= f4 else 'OUT'
            tf = true_freqs[k] if (true_freqs and k < len(true_freqs)) else None
            print(f'    Mode {k+1} [{in_b}] {fmt_component(c, tf)}')

        if true_freqs:
            print(f'    True freqs: {[f"{f:.4f}" for f in true_freqs]} rad/yr  '
                  f'True amps: {true_amps}')
            # Show expected filter gain at each true frequency
            for tf, ta in zip(true_freqs, true_amps):
                if tf < f1 or tf > f4:
                    gain = 0.0
                elif tf < f2:
                    gain = (tf - f1) / (f2 - f1)
                elif tf <= f3:
                    gain = 1.0
                else:
                    gain = (f4 - tf) / (f4 - f3)
                print(f'    Filter gain at {tf:.4f} rad/yr = {gain:.4f}  '
                      f'=> expected output amp = {ta * gain:.4f}')

            # If n_est < len(true_freqs), also try forced n_modes to recover all
            if n_est < len(true_freqs):
                forced = recover_band_sinusoids(
                    z_mpm, spec, FS_WEEKLY,
                    n_modes=len(true_freqs), max_modes=len(true_freqs))
                print(f'    [Forced {len(true_freqs)}-mode fit]  '
                      f'SNR={forced["snr_db"]:.1f} dB')
                for k, (c, tf, ta) in enumerate(
                        zip(forced['components'], true_freqs, true_amps)):
                    df = abs(c['freq_radyr']) - tf
                    print(f'      Mode {k+1}: f={c["freq_radyr"]:+.4f}  '
                          f'df={df:+.5f}  amp={c["amplitude"]:.4f}  '
                          f'(exp {ta:.4f})')

    plt.savefig(f'fig_recovery_filter{i+1}.png', dpi=110, bbox_inches='tight')

plt.show()
