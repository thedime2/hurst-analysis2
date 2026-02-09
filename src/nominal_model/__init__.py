"""
Nominal Model derivation modules for Hurst's spectral analysis.

Implements the derivation of discrete line frequencies and the period
hierarchy from comb filter frequency-vs-time measurements.

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figures AI-5 and AI-6
"""

from .sideband_analysis import (
    group_filters_into_lines,
    compute_sideband_envelopes
)
from .lse_smoothing import (
    smooth_frequency_trace,
    fit_frequency_line
)
from .derivation import (
    identify_line_frequencies,
    compute_line_spacings,
    build_nominal_model,
    validate_against_fourier
)

__all__ = [
    'group_filters_into_lines',
    'compute_sideband_envelopes',
    'smooth_frequency_trace',
    'fit_frequency_line',
    'identify_line_frequencies',
    'compute_line_spacings',
    'build_nominal_model',
    'validate_against_fourier'
]
