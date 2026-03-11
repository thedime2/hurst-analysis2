# -*- coding: utf-8 -*-
"""
Automated Filter Design — Stage 9

Designs the 6-filter decomposition (LP + 5 BP) from group boundaries,
producing both Ormsby and CMW specifications.

The 6 filters follow Hurst's Principle of Harmonicity (~2:1 period ratios):
  LP-1: Trend (18yr+)
  BP-2: 54-month group
  BP-3: 18-month group
  BP-4: 40-week group
  BP-5: 20-week group
  BP-6: 10-week group

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing (1970),
           Page 152 and Cyclitec Cycles Course
"""

import numpy as np
from src.time_frequency.cmw import ormsby_spec_to_cmw_params


# Hurst/Cyclitec nominal group boundaries (rad/yr)
# These are the "ideal" boundaries from the Nominal Model
CYCLITEC_BOUNDARIES = {
    'trend_54mo': 0.93,   # Between trend and 54-month
    '54mo_18mo':  2.09,   # Between 54-month and 18-month
    '18mo_40wk':  3.93,   # Between 18-month and 40-week
    '40wk_20wk':  6.98,   # Between 40-week and 20-week
    '20wk_10wk':  12.82,  # Between 20-week and 10-week
}


def design_analysis_filters(group_boundaries=None, w0=0.3676, fs=52,
                             transition_bw=0.35, nw_cycles=7):
    """
    Design 6-filter specifications from group boundaries.

    Parameters
    ----------
    group_boundaries : array or None
        Trough frequencies defining group edges (rad/yr).
        If None, uses Cyclitec nominal boundaries.
    w0 : float
        Fundamental spacing (rad/yr)
    fs : float
        Sampling rate
    transition_bw : float
        Ormsby transition bandwidth (rad/yr)
    nw_cycles : int
        Filter length in cycles of center frequency (default 7)

    Returns
    -------
    dict with:
        'ormsby_specs': list of 6 Ormsby filter specs
        'cmw_specs': list of 6 CMW parameter sets
        'summary': human-readable summary
    """
    # Use provided boundaries or Cyclitec defaults
    if group_boundaries is not None and len(group_boundaries) >= 4:
        bounds = np.sort(group_boundaries)
        # Take the most significant boundaries (deepest troughs)
        # Map to the 5 inter-group boundaries
        b = _select_best_boundaries(bounds, w0)
    else:
        b = list(CYCLITEC_BOUNDARIES.values())

    # Ensure we have 5 boundaries
    while len(b) < 5:
        # Extend by 2:1 Harmonicity
        b.append(b[-1] * 2)
    b = sorted(b[:5])

    # Design 6 filters
    ormsby_specs = []
    cmw_specs = []

    # Filter 1: Lowpass (trend)
    f_pass = b[0] - transition_bw / 2
    f_stop = b[0] + transition_bw / 2
    f_pass = max(0.3, f_pass)  # Floor

    nw_lp = int(nw_cycles * (2 * np.pi / (b[0] / 2) * fs))
    nw_lp = min(nw_lp, 1393)  # Cap at Hurst's standard

    lp_spec = {
        'type': 'lp',
        'f_pass': float(f_pass),
        'f_stop': float(f_stop),
        'nw': nw_lp,
        'label': 'LP-1 Trend',
        'index': 0,
    }
    ormsby_specs.append(lp_spec)
    cmw_specs.append(ormsby_spec_to_cmw_params(lp_spec))

    # Filters 2-6: Bandpass
    bp_names = ['BP-2 54mo', 'BP-3 18mo', 'BP-4 40wk', 'BP-5 20wk', 'BP-6 10wk']
    for i in range(5):
        w_low = b[i]
        w_high = b[i + 1] if i + 1 < len(b) else b[i] * 2

        f1 = w_low - transition_bw / 2
        f2 = w_low + transition_bw / 2
        f3 = w_high - transition_bw / 2
        f4 = w_high + transition_bw / 2

        f1 = max(0.1, f1)

        # Filter length: 7 cycles of center frequency
        f_center = (f2 + f3) / 2
        period_samples = 2 * np.pi / f_center * fs
        nw = int(nw_cycles * period_samples)
        nw = max(199, min(nw, 2000))  # Reasonable range

        bp_spec = {
            'type': 'bp',
            'f1': float(f1),
            'f2': float(f2),
            'f3': float(f3),
            'f4': float(f4),
            'nw': nw,
            'label': bp_names[i] if i < len(bp_names) else f'BP-{i+2}',
            'index': i + 1,
        }
        ormsby_specs.append(bp_spec)
        cmw_specs.append(ormsby_spec_to_cmw_params(bp_spec))

    # Summary
    summary_lines = ["=== 6-Filter Design ==="]
    for spec in ormsby_specs:
        if spec['type'] == 'lp':
            summary_lines.append(
                f"  {spec['label']}: LP f_pass={spec['f_pass']:.2f} "
                f"f_stop={spec['f_stop']:.2f} nw={spec['nw']}"
            )
        else:
            summary_lines.append(
                f"  {spec['label']}: BP [{spec['f1']:.2f}, {spec['f2']:.2f}, "
                f"{spec['f3']:.2f}, {spec['f4']:.2f}] nw={spec['nw']}"
            )
    summary = '\n'.join(summary_lines)

    return {
        'ormsby_specs': ormsby_specs,
        'cmw_specs': cmw_specs,
        'boundaries_used': b,
        'summary': summary,
    }


def _select_best_boundaries(trough_freqs, w0, target_boundaries=None):
    """
    Select the 5 best group boundaries from detected troughs.

    Maps troughs to the nearest Cyclitec boundary and picks the best match.
    """
    if target_boundaries is None:
        target_boundaries = list(CYCLITEC_BOUNDARIES.values())

    selected = []
    for target in target_boundaries:
        if len(trough_freqs) == 0:
            selected.append(target)
            continue
        dists = np.abs(trough_freqs - target)
        best_idx = np.argmin(dists)
        if dists[best_idx] < target * 0.3:  # Within 30% of target
            selected.append(float(trough_freqs[best_idx]))
        else:
            selected.append(target)  # Use default

    return selected
