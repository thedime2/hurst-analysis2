# -*- coding: utf-8 -*-
"""
Automated Nominal Model Derivation Pipeline

End-to-end pipeline that derives the Nominal Model from raw price data,
automating Hurst's methodology from Appendix A.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing (1970)
"""

from .derive_nominal_model import derive_nominal_model, NominalModelResult
from .comb_bank import design_extended_cmw_bank, design_narrowband_cmw_bank
from .validation import validate_model
from .filter_design import design_analysis_filters

__all__ = [
    'derive_nominal_model',
    'NominalModelResult',
    'design_extended_cmw_bank',
    'design_narrowband_cmw_bank',
    'validate_model',
    'design_analysis_filters',
]
