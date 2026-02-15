# -*- coding: utf-8 -*-
"""
Visualization utilities for Hurst Spectral Analysis.

Import setup_plotting() at the top of experiment scripts to avoid
plt.show() blocking issues when running from the command line.

Usage:
    from src.visualization import setup_plotting
    setup_plotting()
"""

import matplotlib
import matplotlib.pyplot as plt


def setup_plotting(backend=None):
    """
    Configure matplotlib for non-blocking operation.

    Call this once at the top of experiment scripts, BEFORE any plt calls.

    Parameters
    ----------
    backend : str, optional
        Force a specific backend. If None, uses 'Agg' (file-only, no display).
        Use 'TkAgg' or 'Qt5Agg' if you need interactive windows.
    """
    if backend is not None:
        matplotlib.use(backend)
    else:
        # Agg is safe for headless / CLI usage -- saves to file only
        matplotlib.use('Agg')

    # Also turn on interactive mode so any accidental plt.show() returns
    plt.ion()
