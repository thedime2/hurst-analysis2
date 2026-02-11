# -*- coding: utf-8 -*-
"""
Decimation and interpolation utilities for filter pipelines.

Supports decimating signals (every Nth sample with configurable offset),
then interpolating gaps back using:
  - 'none'   : NaN fill (sparse output)
  - '3point' : Hurst's page 213 parabolic interpolation (Lagrange degree-2)
  - 'cubic'  : scipy cubic spline
  - 'linear' : linear interpolation

Reference: J.M. Hurst, "The Profit Magic of Stock Transaction Timing", p.213
"""

import numpy as np
from scipy.interpolate import interp1d

VALID_METHODS = ('none', '3point', 'cubic', 'linear')


def decimate_signal(signal, spacing, offset=1):
    """
    Extract every Nth sample from signal with a configurable start offset.

    Parameters
    ----------
    signal : ndarray, shape (L,)
        Input signal.
    spacing : int
        Decimation factor. 1 = no decimation.
    offset : int
        1-based starting index (1 through spacing).
        offset=1 starts at index 0, offset=3 starts at index 2.

    Returns
    -------
    decimated : ndarray
        Extracted samples.
    indices : ndarray of int
        Original indices of the extracted samples.
    """
    if spacing < 1:
        raise ValueError(f"spacing must be >= 1, got {spacing}")
    if not (1 <= offset <= spacing):
        raise ValueError(f"offset must be in [1, {spacing}], got {offset}")

    start_idx = offset - 1  # convert 1-based to 0-based
    indices = np.arange(start_idx, len(signal), spacing)
    return signal[indices], indices


def interpolate_sparse(values, indices, full_length, method='none'):
    """
    Fill gaps in a sparse signal using the specified interpolation method.

    Parameters
    ----------
    values : ndarray
        Sparse sample values (real or complex).
    indices : ndarray of int
        Original indices where values were sampled.
    full_length : int
        Length of the full (non-decimated) output array.
    method : str
        'none', '3point', 'cubic', or 'linear'.

    Returns
    -------
    result : ndarray, shape (full_length,)
        Interpolated signal. NaN at positions not covered by the method.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'")

    is_complex = np.iscomplexobj(values)

    if method == 'none':
        if is_complex:
            result = np.full(full_length, np.nan + 0j, dtype=np.complex128)
        else:
            result = np.full(full_length, np.nan, dtype=np.float64)
        result[indices] = values
        return result

    if method == '3point':
        if is_complex:
            real_part = interpolate_3point(values.real, indices, full_length)
            imag_part = interpolate_3point(values.imag, indices, full_length)
            return real_part + 1j * imag_part
        return interpolate_3point(values, indices, full_length)

    # 'linear' or 'cubic' via scipy
    if is_complex:
        f_real = interp1d(indices, values.real, kind=method,
                          fill_value='extrapolate', assume_sorted=True)
        f_imag = interp1d(indices, values.imag, kind=method,
                          fill_value='extrapolate', assume_sorted=True)
        x_full = np.arange(full_length)
        return f_real(x_full) + 1j * f_imag(x_full)
    else:
        f = interp1d(indices, values, kind=method,
                     fill_value='extrapolate', assume_sorted=True)
        return f(np.arange(full_length))


def interpolate_3point(values, indices, full_length):
    """
    Hurst page 213: fit quadratic through each consecutive triplet, evaluate at gaps.

    For each triplet (x[i-1], y[i-1]), (x[i], y[i]), (x[i+1], y[i+1]):
      - Fit Lagrange degree-2 polynomial exactly through all 3 points
      - Evaluate at integer positions in the interval
      - Overlapping: use parabola centered at i for first half, i+1 for second half

    Edge positions use linear interpolation/extrapolation from nearest 2 points.

    Parameters
    ----------
    values : ndarray
        Sparse sample values (real).
    indices : ndarray of int
        Original indices.
    full_length : int
        Length of full output.

    Returns
    -------
    result : ndarray, shape (full_length,)
    """
    n = len(values)

    # Fallback: not enough points for parabola
    if n < 3:
        if n < 2:
            result = np.full(full_length, np.nan)
            if n == 1:
                result[indices[0]] = values[0]
            return result
        # Linear fallback for 2 points
        f = interp1d(indices, values, kind='linear',
                     fill_value='extrapolate', assume_sorted=True)
        return f(np.arange(full_length))

    result = np.full(full_length, np.nan)

    # Place known values
    result[indices] = values

    # For each consecutive pair of known points, interpolate the gap
    for k in range(n - 1):
        x_left = indices[k]
        x_right = indices[k + 1]

        # Gap positions (exclusive of endpoints which are already set)
        gap_x = np.arange(x_left + 1, x_right)
        if len(gap_x) == 0:
            continue

        # Midpoint of the gap for switching parabolas
        mid = (x_left + x_right) / 2.0

        # Left parabola: centered at k, uses triplet [k-1, k, k+1]
        if k > 0:
            tri_left = np.array([indices[k - 1], indices[k], indices[k + 1]])
            val_left = np.array([values[k - 1], values[k], values[k + 1]])
            coeff_left = np.polyfit(tri_left, val_left, 2)
        else:
            # At left edge, only right parabola available
            coeff_left = None

        # Right parabola: centered at k+1, uses triplet [k, k+1, k+2]
        if k + 2 < n:
            tri_right = np.array([indices[k], indices[k + 1], indices[k + 2]])
            val_right = np.array([values[k], values[k + 1], values[k + 2]])
            coeff_right = np.polyfit(tri_right, val_right, 2)
        else:
            # At right edge, only left parabola available
            coeff_right = None

        for x in gap_x:
            if coeff_left is not None and coeff_right is not None:
                # Use left parabola for first half, right for second half
                if x <= mid:
                    result[x] = np.polyval(coeff_left, x)
                else:
                    result[x] = np.polyval(coeff_right, x)
            elif coeff_left is not None:
                result[x] = np.polyval(coeff_left, x)
            elif coeff_right is not None:
                result[x] = np.polyval(coeff_right, x)

    # Extrapolate edges (before first index, after last index)
    if indices[0] > 0:
        # Linear extrapolation from first two points
        f_edge = interp1d(indices[:2], values[:2], kind='linear',
                          fill_value='extrapolate', assume_sorted=True)
        edge_x = np.arange(0, indices[0])
        result[edge_x] = f_edge(edge_x)

    if indices[-1] < full_length - 1:
        # Linear extrapolation from last two points
        f_edge = interp1d(indices[-2:], values[-2:], kind='linear',
                          fill_value='extrapolate', assume_sorted=True)
        edge_x = np.arange(indices[-1] + 1, full_length)
        result[edge_x] = f_edge(edge_x)

    return result


def interpolate_phase_wrapped(phasew_sparse, indices, full_length, method='3point'):
    """
    Interpolate wrapped phase using unwrap-interpolate-rewrap strategy.

    Avoids artifacts at +/-pi discontinuities by unwrapping before
    interpolation, then re-wrapping the result.

    Parameters
    ----------
    phasew_sparse : ndarray
        Wrapped phase values at sparse indices.
    indices : ndarray of int
        Original indices.
    full_length : int
        Length of full output.
    method : str
        Interpolation method ('none', '3point', 'cubic', 'linear').

    Returns
    -------
    result : ndarray, shape (full_length,)
        Wrapped phase in [-pi, pi].
    """
    if method == 'none':
        result = np.full(full_length, np.nan)
        result[indices] = phasew_sparse
        return result

    unwrapped = np.unwrap(phasew_sparse)
    interp_unwrapped = interpolate_sparse(unwrapped, indices, full_length, method)
    return np.angle(np.exp(1j * interp_unwrapped))


def interpolate_output_dict(output_dict, indices, full_length, method='none'):
    """
    Apply interpolation to all arrays in a filter output dict.

    The output dict follows the structure from apply_ormsby_filter() / apply_cmw():
        'signal', 'envelope', 'phase', 'phasew', 'frequency'

    Parameters
    ----------
    output_dict : dict
        Filter output with arrays of decimated length.
    indices : ndarray of int
        Original indices of the decimated samples.
    full_length : int
        Length of the original (non-decimated) signal.
    method : str
        Interpolation method.

    Returns
    -------
    result : dict
        Same keys, arrays expanded to full_length.
    """
    result = {}

    for key, val in output_dict.items():
        # Pass through non-array fields (metadata like 'spec', 'index')
        if val is None or not isinstance(val, np.ndarray):
            result[key] = val
            continue

        if key == 'phasew':
            # Wrapped phase needs special handling
            result[key] = interpolate_phase_wrapped(val, indices, full_length, method)
        elif key == 'signal' and np.iscomplexobj(val):
            # Complex signal: interpolate real/imag independently
            result[key] = interpolate_sparse(val, indices, full_length, method)
        elif key == 'envelope':
            interped = interpolate_sparse(val, indices, full_length, method)
            # Clip to non-negative (cubic can overshoot)
            if method == 'cubic' and not np.all(np.isnan(interped)):
                interped = np.maximum(interped, 0.0)
            result[key] = interped
        else:
            # signal (real), phase (unwrapped), frequency
            result[key] = interpolate_sparse(val, indices, full_length, method)

    return result
