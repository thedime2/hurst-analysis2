# -*- coding: utf-8 -*-
"""
Modulation sideband analysis for identifying line frequencies.

Groups overlapping comb filter frequency traces into "line families" and
computes the modulation sideband envelope for each family.

This implements the analysis shown in Hurst's Figure AI-5: "Modulation
Sidebands -- The 'Line' Frequency Phenomena."

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
           Appendix A, Figure AI-5
"""

import numpy as np
from sklearn.cluster import KMeans


def group_filters_into_lines(freq_measurements, n_lines=6, method='kmeans'):
    """
    Group comb filter frequency traces into line families.

    Each comb filter's median measured frequency is used to assign it to
    the nearest line frequency. Filters whose measured frequency clusters
    together are considered part of the same "line family."

    Parameters
    ----------
    freq_measurements : list of dict
        Each dict has:
            'filter_index': int
            'center_freq': float (filter design center, rad/yr)
            'peaks': dict from measure_freq_at_peaks
            'troughs': dict from measure_freq_at_troughs
            'zero_crossings': dict from measure_freq_at_zero_crossings
    n_lines : int
        Number of line frequencies to identify (default 6 for Hurst's
        comb bank range)
    method : str
        Clustering method: 'kmeans' or 'equal_spacing'

    Returns
    -------
    dict with:
        line_frequencies : array of shape (n_lines,) - nominal line
            frequencies in rad/yr, sorted ascending
        line_periods_weeks : array of shape (n_lines,) - corresponding
            periods in weeks
        assignments : array of shape (n_filters,) - line index for each
            filter
        groups : list of list of int - filter indices in each line family
        median_freqs : array - median measured frequency per filter
    """
    # Compute median measured frequency for each filter (peak-to-peak method)
    median_freqs = np.array([
        _median_measured_freq(fm) for fm in freq_measurements
    ])

    if method == 'kmeans':
        # K-means clustering on 1D frequency values
        km = KMeans(n_clusters=n_lines, random_state=42, n_init=10)
        assignments = km.fit_predict(median_freqs.reshape(-1, 1))
        line_frequencies = np.sort(km.cluster_centers_.flatten())

        # Re-assign with sorted labels so line 0 = lowest freq
        sorted_centers = np.sort(km.cluster_centers_.flatten())
        center_to_idx = {c: i for i, c in enumerate(sorted_centers)}
        assignments = np.array([
            center_to_idx[km.cluster_centers_[a][0]] for a in assignments
        ])
    elif method == 'equal_spacing':
        # Simple approach: divide frequency range equally
        f_min = np.min(median_freqs)
        f_max = np.max(median_freqs)
        boundaries = np.linspace(f_min, f_max, n_lines + 1)
        line_frequencies = (boundaries[:-1] + boundaries[1:]) / 2
        assignments = np.digitize(median_freqs, boundaries[1:-1])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build groups (list of filter indices per line)
    groups = []
    for i in range(n_lines):
        group_mask = assignments == i
        groups.append(np.where(group_mask)[0].tolist())

    # Refine line frequencies as median of member filter medians
    refined_freqs = np.zeros(n_lines)
    for i in range(n_lines):
        if groups[i]:
            refined_freqs[i] = np.median(median_freqs[groups[i]])
        else:
            refined_freqs[i] = line_frequencies[i]

    line_periods_weeks = 2 * np.pi / refined_freqs * 52

    return {
        'line_frequencies': refined_freqs,
        'line_periods_weeks': line_periods_weeks,
        'assignments': assignments,
        'groups': groups,
        'median_freqs': median_freqs
    }


def compute_sideband_envelopes(freq_measurements, grouping, time_range=None,
                                fs=52, measurement='peaks_period'):
    """
    Compute modulation sideband envelopes for each line family.

    For each group of filters assigned to a line, collects all frequency
    measurements over time and computes the upper and lower envelope
    (max/min frequency at each time step).

    Parameters
    ----------
    freq_measurements : list of dict
        Frequency measurements per filter (from Phase 2)
    grouping : dict
        Output from group_filters_into_lines()
    time_range : tuple of (start_idx, end_idx), optional
        Sample index range to analyze. If None, uses full range.
    fs : float
        Sampling rate (samples/year)
    measurement : str
        Which measurement to use: 'peaks_period', 'peaks_phase', 'zero_crossings'

    Returns
    -------
    list of dict, one per line family:
        'line_freq': float - nominal line frequency
        'line_period_weeks': float - nominal period in weeks
        'time_grid': array - regular time grid (sample indices)
        'upper_envelope': array - max frequency at each time step
        'lower_envelope': array - min frequency at each time step
        'mean_trace': array - mean frequency at each time step
        'n_filters': int - number of filters in this family
    """
    groups = grouping['groups']
    line_freqs = grouping['line_frequencies']

    # Determine time grid
    all_times = []
    for fm in freq_measurements:
        times, _ = _get_measurement_data(fm, measurement)
        if len(times) > 0:
            all_times.extend(times)

    if not all_times:
        return []

    t_min = int(np.min(all_times)) if time_range is None else time_range[0]
    t_max = int(np.max(all_times)) if time_range is None else time_range[1]

    # Regular time grid with spacing proportional to typical measurement interval
    grid_spacing = max(1, int((t_max - t_min) / 200))
    time_grid = np.arange(t_min, t_max + 1, grid_spacing)
    n_grid = len(time_grid)

    envelopes = []
    for line_idx, (group, line_freq) in enumerate(zip(groups, line_freqs)):
        if not group:
            envelopes.append(None)
            continue

        # Collect all frequency measurements from filters in this group
        all_t = []
        all_f = []
        for filt_idx in group:
            fm = freq_measurements[filt_idx]
            times, freqs = _get_measurement_data(fm, measurement)
            if len(times) == 0:
                continue

            # Filter to time range and remove extreme outliers
            if time_range is not None:
                mask = (times >= time_range[0]) & (times <= time_range[1])
                times = times[mask]
                freqs = freqs[mask]

            # Remove outliers beyond 2x center frequency
            center = fm['center_freq']
            valid = (freqs > center * 0.5) & (freqs < center * 2.0)
            all_t.extend(times[valid])
            all_f.extend(freqs[valid])

        if not all_t:
            envelopes.append(None)
            continue

        all_t = np.array(all_t)
        all_f = np.array(all_f)

        # Compute envelopes on the time grid using windowed max/min
        window_half = grid_spacing * 2
        upper = np.full(n_grid, np.nan)
        lower = np.full(n_grid, np.nan)
        mean_trace = np.full(n_grid, np.nan)

        for i, t in enumerate(time_grid):
            mask = np.abs(all_t - t) <= window_half
            if np.any(mask):
                vals = all_f[mask]
                upper[i] = np.max(vals)
                lower[i] = np.min(vals)
                mean_trace[i] = np.mean(vals)

        envelopes.append({
            'line_freq': line_freq,
            'line_period_weeks': 2 * np.pi / line_freq * 52,
            'time_grid': time_grid,
            'upper_envelope': upper,
            'lower_envelope': lower,
            'mean_trace': mean_trace,
            'n_filters': len(group)
        })

    return envelopes


def _median_measured_freq(fm):
    """Get median peak-to-peak measured frequency for a filter."""
    freqs = fm['peaks']['freqs_period']
    if len(freqs) == 0:
        return fm['center_freq']

    center = fm['center_freq']
    valid = (freqs > center * 0.5) & (freqs < center * 2.0)
    if np.sum(valid) == 0:
        return fm['center_freq']

    return np.median(freqs[valid])


def _get_measurement_data(fm, measurement):
    """Extract times and frequencies from a filter measurement dict."""
    if measurement == 'peaks_period':
        return fm['peaks']['times'], fm['peaks']['freqs_period']
    elif measurement == 'peaks_phase':
        times = fm['peaks']['times']
        freqs = fm['peaks'].get('freqs_phase')
        if freqs is None:
            return np.array([]), np.array([])
        return times, freqs
    elif measurement == 'zero_crossings':
        return fm['zero_crossings']['times'], fm['zero_crossings']['freqs']
    else:
        raise ValueError(f"Unknown measurement: {measurement}")
