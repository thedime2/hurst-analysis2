"""
Ormsby filter and filter bank modules for spectral analysis.

Implements real and complex Ormsby bandpass filters, filter bank design
(both Q-based and uniform-step comb banks), and filter application utilities.
"""

from .funcOrmsby import (
    ormsby_filter,
    apply_ormsby_filter,
    funcOrmsby3,
    ormsby_derivative_filter
)
from .funcDesignFilterBank import (
    design_ormsby_filter_bank,
    design_hurst_comb_bank,
    create_filter_kernels,
    apply_filter_bank,
    plot_filter_bank_response,
    plot_idealized_comb_response,
    create_time_frequency_heatmap,
    print_filter_specs
)
from .decimation import (
    decimate_signal,
    interpolate_sparse,
    interpolate_3point,
    interpolate_output_dict
)

__all__ = [
    'ormsby_filter',
    'apply_ormsby_filter',
    'funcOrmsby3',
    'ormsby_derivative_filter',
    'design_ormsby_filter_bank',
    'design_hurst_comb_bank',
    'create_filter_kernels',
    'apply_filter_bank',
    'plot_filter_bank_response',
    'plot_idealized_comb_response',
    'create_time_frequency_heatmap',
    'print_filter_specs',
    'decimate_signal',
    'interpolate_sparse',
    'interpolate_3point',
    'interpolate_output_dict'
]
