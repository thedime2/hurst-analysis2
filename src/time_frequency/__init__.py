# -*- coding: utf-8 -*-
"""
Time-Frequency Analysis Module

Complex Morlet Wavelets (CMW) with FWHM-based frequency-domain design,
matched to Ormsby filter specifications.
"""

from .cmw import (
    ormsby_spec_to_cmw_params,
    cmw_freq_domain,
    apply_cmw,
    apply_cmw_bank,
    FWHM_TO_SIGMA,
)

__all__ = [
    'ormsby_spec_to_cmw_params',
    'cmw_freq_domain',
    'apply_cmw',
    'apply_cmw_bank',
    'FWHM_TO_SIGMA',
]
