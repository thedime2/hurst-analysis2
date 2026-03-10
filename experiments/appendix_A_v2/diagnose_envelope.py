import sys, os
sys.path.insert(0, os.path.abspath('../..'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin2
from scipy.signal import hilbert as scipy_hilbert
from utils_ai import design_comb_bank, make_ormsby_kernels, load_weekly_data, get_window, FS_WEEKLY, NW_WEEKLY
from src.filters import apply_ormsby_filter, create_filter_kernels
from src.time_frequency.cmw import ormsby_spec_to_cmw_params, apply_cmw

specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)
close, dates = load_weekly_data()
s_idx, e_idx = get_window(dates)
N   = len(close)
t_n = np.arange(N)
t_dates = dates[s_idx:e_idx]   # date axis for display window

def cv(e):
    m = float(np.mean(e))
    return (float(np.std(e)) / m * 100.0) if m > 1e-12 else 0.0

print('='*68)
print('Envelope CV% -- pure tone, two-tone, DJIA (lower = smoother)')
print('Correct pure tone: omega = fc/FS rad/sample (NO extra 2pi)')
print('='*68)

for i in [3, 11, 19]:
    fc   = specs[i]['f_center']     # rad/yr
    h    = filters[i]['kernel']
    p    = ormsby_spec_to_cmw_params(specs[i])

    omega_s = fc / FS_WEEKLY        # rad/sample (correct digital freq)
    pure1   = np.cos(omega_s * t_n)                                  # single tone at fc
    pure2   = np.cos(omega_s * t_n) + 0.5 * np.cos((fc + 0.3676) / FS_WEEKLY * t_n)  # two tones
    pure2   = np.cos(omega_s * t_n) + 0.5 * np.cos((fc + 0.676) / FS_WEEKLY * t_n)  # two tones DF smaller gap
    

    tests = [('Pure tone at fc', pure1),
             ('Two tones fc + fc+0.3676', pure2),
             ('DJIA signal', close)]

    print(f'FC-{i+1} fc={fc:.1f} rad/yr  omega_sample={omega_s:.4f} rad/sample')

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f'Filter {i+1}  fc={fc:.1f} rad/yr  '
                 f'(T={2*np.pi/fc:.2f} yr)   ω_sample={omega_s:.4f} rad/sample',
                 fontsize=11)

    for row, (label, sig) in enumerate(tests):
        zo = apply_ormsby_filter(sig, h, mode='reflect', fs=FS_WEEKLY)['signal'][s_idx:e_idx]
        zc = apply_cmw(sig, p['f0'], p['fwhm'], FS_WEEKLY, analytic=True)['signal'][s_idx:e_idx]
        eo = np.abs(zo)
        ec = np.abs(zc)

        print(f'  {label}')
        print(f'    Ormsby  CV={cv(eo):6.1f}%  mean={np.mean(eo):.4f}')
        print(f'    CMW     CV={cv(ec):6.1f}%  mean={np.mean(ec):.4f}')

        ax = axes[row]
        # real parts (faint, thin)
        ax.plot(t_dates, zo.real, color='C0', alpha=0.25, lw=0.7)
        ax.plot(t_dates, zc.real, color='C1', alpha=0.25, lw=0.7)
        # positive envelopes (solid, bold)
        ax.plot(t_dates, eo,  color='C0', lw=1.6,
                label=f'Ormsby env  CV={cv(eo):.1f}%  mean={np.mean(eo):.3f}')
        ax.plot(t_dates, ec,  color='C1', lw=1.6,
                label=f'CMW env     CV={cv(ec):.1f}%  mean={np.mean(ec):.3f}')
        # negative envelopes (mirror, dashed)
        ax.plot(t_dates, -eo, color='C0', lw=0.8, ls='--', alpha=0.5)
        ax.plot(t_dates, -ec, color='C1', lw=0.8, ls='--', alpha=0.5)

        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylabel('Amplitude')

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    print()

# ============================================================
# Second figure set: Envelope method comparison
#   Ormsby complex  vs  Ormsby real+Hilbert
#   FIR2 complex    vs  FIR2 real+Hilbert
#   CMW
# ============================================================

# Real-valued Ormsby kernels (analytic=False) for Hilbert comparison
filters_real = create_filter_kernels(specs, fs=FS_WEEKLY, filter_type='modulate', analytic=False)

print('='*68)
print('Envelope method comparison  (lower CV% = smoother)')
print('  Ormsby cplx | Ormsby+Hilbert | FIR2 cplx | FIR2+Hilbert | CMW')
print('='*68)

for i in [3, 11, 19]:
    spec  = specs[i]
    fc    = spec['f_center']          # rad/yr
    h_o_cplx = filters[i]['kernel']  # complex analytic Ormsby (existing)
    h_o_real = filters_real[i]['kernel']  # real Ormsby kernel
    p     = ormsby_spec_to_cmw_params(spec)

    # FIR2 trapezoidal real kernel — same f1..f4 edges, converted to [0,1]
    # firwin2 expects frequencies normalized by Nyquist (1 = Nyquist in cycles/yr)
    nyq_cyc = FS_WEEKLY / 2.0                        # 26 cycles/yr
    f_norm = [0.0,
              spec['f1'] / (2*np.pi) / nyq_cyc,
              spec['f2'] / (2*np.pi) / nyq_cyc,
              spec['f3'] / (2*np.pi) / nyq_cyc,
              spec['f4'] / (2*np.pi) / nyq_cyc,
              1.0]
    gains = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    h_f2_real = firwin2(NW_WEEKLY, f_norm, gains)

    # FIR2 complex analytic kernel: zero out negative frequencies
    nk    = len(h_f2_real)
    H     = np.fft.fft(h_f2_real)
    H_a   = np.zeros(nk, dtype=complex)
    H_a[0] = H[0]                          # DC
    H_a[1 : nk//2] = 2.0 * H[1 : nk//2]  # positive freqs × 2
    if nk % 2 == 0:
        H_a[nk//2] = H[nk//2]             # Nyquist (if even)
    h_f2_cplx = np.fft.ifft(H_a)

    omega_s = fc / FS_WEEKLY
    pure1   = np.cos(omega_s * t_n)
    pure2   = np.cos(omega_s * t_n) + 0.5 * np.cos((fc + 0.3676) / FS_WEEKLY * t_n)
    tests   = [('Pure tone at fc', pure1),
               ('Two tones fc + fc+0.3676', pure2),
               ('DJIA signal', close)]

    print(f'\nFC-{i+1}  fc={fc:.1f} rad/yr  (T={2*np.pi/fc:.2f} yr)')

    fig2, axes2 = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig2.suptitle(
        f'Filter {i+1}  fc={fc:.1f} rad/yr  (T={2*np.pi/fc:.2f} yr) — '
        f'Envelope Method Comparison',
        fontsize=11)

    for row, (label, sig) in enumerate(tests):
        # 1. Ormsby complex analytic → envelope directly
        zo_c = apply_ormsby_filter(sig, h_o_cplx, mode='reflect', fs=FS_WEEKLY)['signal'][s_idx:e_idx]
        eo_c = np.abs(zo_c)

        # 2. Ormsby real → scipy Hilbert → envelope
        # Apply Hilbert to the FULL filtered signal, then slice — avoids edge artifacts
        zo_r_full = apply_ormsby_filter(sig, h_o_real, mode='reflect', fs=FS_WEEKLY)['signal']
        eo_h = np.abs(scipy_hilbert(zo_r_full.real))[s_idx:e_idx]

        # 3. FIR2 complex analytic → envelope directly
        zf_c = apply_ormsby_filter(sig, h_f2_cplx, mode='reflect', fs=FS_WEEKLY)['signal'][s_idx:e_idx]
        ef_c = np.abs(zf_c)

        # 4. FIR2 real → scipy Hilbert → envelope
        # Apply Hilbert to the FULL filtered signal, then slice — avoids edge artifacts
        zf_r_full = apply_ormsby_filter(sig, h_f2_real, mode='reflect', fs=FS_WEEKLY)['signal']
        ef_h = np.abs(scipy_hilbert(zf_r_full.real))[s_idx:e_idx]

        # 5. CMW
        zc  = apply_cmw(sig, p['f0'], p['fwhm'], FS_WEEKLY, analytic=True)['signal'][s_idx:e_idx]
        ec  = np.abs(zc)

        results = [('Ormsby cplx   ', eo_c),
                   ('Ormsby+Hilbert', eo_h),
                   ('FIR2 cplx     ', ef_c),
                   ('FIR2+Hilbert  ', ef_h),
                   ('CMW           ', ec)]
        print(f'  {label}')
        for name, e in results:
            print(f'    {name}  CV={cv(e):6.1f}%  mean={np.mean(e):.4f}')

        ax = axes2[row]
        ax.plot(t_dates, eo_c, color='C0', lw=1.8,
                label=f'Ormsby complex    CV={cv(eo_c):.1f}%')
        ax.plot(t_dates, eo_h, color='C0', lw=1.0, ls='--',
                label=f'Ormsby+Hilbert    CV={cv(eo_h):.1f}%')
        ax.plot(t_dates, ef_c, color='C2', lw=1.8,
                label=f'FIR2 complex      CV={cv(ef_c):.1f}%')
        ax.plot(t_dates, ef_h, color='C2', lw=1.0, ls='--',
                label=f'FIR2+Hilbert      CV={cv(ef_h):.1f}%')
        ax.plot(t_dates, ec,  color='C1', lw=1.8,
                label=f'CMW               CV={cv(ec):.1f}%')
        ax.axhline(0, color='k', lw=0.4, ls=':')
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=7.5, loc='upper right', ncol=2)
        ax.set_ylabel('Envelope')

        # Clip y-axis to 99th percentile of all envelopes to suppress any residual spikes
        all_env = np.concatenate([eo_c, eo_h, ef_c, ef_h, ec])
        ymax = np.percentile(all_env[np.isfinite(all_env)], 99) * 1.15
        ax.set_ylim(-ymax * 0.15, ymax)

    axes2[-1].set_xlabel('Date')
    plt.tight_layout()

plt.show()
