"""
Spectral analysis modules including Lanczos spectrum, peak detection, and envelope fitting.
"""

from .lanczos import lanczos_spectrum, nextpow2, nextpow2b
from .peak_detection import (
    find_spectral_peaks,
    find_spectral_troughs,
    filter_peaks_by_frequency_range,
    detect_fine_structure_spacing
)
from .envelopes import (
    fit_power_law_envelope,
    fit_upper_envelope,
    fit_lower_envelope,
    envelope_model,
    compute_fit_quality,
    fit_dual_envelope
)

__all__ = [
    'lanczos_spectrum',
    'nextpow2',
    'nextpow2b',
    'find_spectral_peaks',
    'find_spectral_troughs',
    'filter_peaks_by_frequency_range',
    'detect_fine_structure_spacing',
    'fit_power_law_envelope',
    'fit_upper_envelope',
    'fit_lower_envelope',
    'envelope_model',
    'compute_fit_quality',
    'fit_dual_envelope'
]
