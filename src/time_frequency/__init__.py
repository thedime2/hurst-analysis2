# -*- coding: utf-8 -*-
"""
Time-Frequency Analysis Module

Complex Morlet Wavelets (CMW) with FWHM-based frequency-domain design,
matched to Ormsby filter specifications. Scalogram computation and
ridge detection for continuous time-frequency analysis.
"""

from .cmw import (
    ormsby_spec_to_cmw_params,
    cmw_freq_domain,
    apply_cmw,
    apply_cmw_bank,
    FWHM_TO_SIGMA,
)
from .morse import (
    morse_freq_domain,
    apply_morse,
)

from .scalogram import compute_scalogram
from .ridge_detection import (
    detect_ridges,
    match_ridges_to_nominal,
    compute_ridge_statistics,
)
from .hypothesis_tests import (
    test_drift_rate_distribution,
    test_envelope_wobble_spectrum,
    test_fm_am_coupling,
    test_synthetic_beating,
)

__all__ = [
    'ormsby_spec_to_cmw_params',
    'cmw_freq_domain',
    'apply_cmw',
    'apply_cmw_bank',
    'FWHM_TO_SIGMA',
    'morse_freq_domain',
    'apply_morse',
    'compute_scalogram',
    'detect_ridges',
    'match_ridges_to_nominal',
    'compute_ridge_statistics',
    'test_drift_rate_distribution',
    'test_envelope_wobble_spectrum',
    'test_fm_am_coupling',
    'test_synthetic_beating',
]
