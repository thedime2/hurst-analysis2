import sys, os
sys.path.insert(0, os.path.abspath('../..'))
import numpy as np
from utils_ai import design_comb_bank, make_ormsby_kernels, load_weekly_data, get_window, FS_WEEKLY, NW_WEEKLY
from src.filters.funcOrmsby import apply_ormsby_filter
from src.time_frequency.cmw import ormsby_spec_to_cmw_params, apply_cmw

specs   = design_comb_bank(fs=FS_WEEKLY, nw=NW_WEEKLY)
filters = make_ormsby_kernels(specs, fs=FS_WEEKLY)
close, dates = load_weekly_data()
s_idx, e_idx = get_window(dates)
N   = len(close)
t_n = np.arange(N)

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

    tests = [('Pure tone at fc', pure1),
             ('Two tones fc + fc+0.3676', pure2),
             ('DJIA signal', close)]

    print(f'FC-{i+1} fc={fc:.1f} rad/yr  omega_sample={omega_s:.4f} rad/sample')
    for label, sig in tests:
        zo = apply_ormsby_filter(sig, h, mode='reflect', fs=FS_WEEKLY)['signal'][s_idx:e_idx]
        zc = apply_cmw(sig, p['f0'], p['fwhm'], FS_WEEKLY, analytic=True)['signal'][s_idx:e_idx]
        eo = np.abs(zo)
        ec = np.abs(zc)
        print(f'  {label}')
        print(f'    Ormsby  CV={cv(eo):6.1f}%  mean={np.mean(eo):.4f}')
        print(f'    CMW     CV={cv(ec):6.1f}%  mean={np.mean(ec):.4f}')
    print()
