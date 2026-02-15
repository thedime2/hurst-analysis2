# -*- coding: utf-8 -*-
"""
Ridge Detection -- Extract Frequency Ridges from Scalograms

Identifies ridges (continuous curves of local energy maxima in the
time-frequency plane) from a CMW scalogram matrix. Ridges correspond
to Hurst's spectral lines and nominal model frequencies.

Algorithm:
  1. At each time column, find local peaks along the frequency axis
  2. Chain peaks across adjacent time columns by nearest-frequency matching
  3. Discard ridges shorter than a minimum duration threshold
  4. Compute per-ridge statistics (mean freq, drift rate, stability)

All frequencies are in radians per year (rad/yr).
"""

import numpy as np
from scipy.signal import find_peaks


def detect_ridges(scalogram_matrix, frequencies,
                  min_prominence=0.1, max_freq_jump=None,
                  min_duration_samples=26):
    """
    Extract frequency ridges from a scalogram matrix.

    Parameters
    ----------
    scalogram_matrix : ndarray (n_scales, n_samples)
        Envelope amplitude matrix from compute_scalogram().
    frequencies : ndarray (n_scales,)
        Center frequencies in rad/yr (from scalogram output).
    min_prominence : float
        Minimum prominence as fraction of column maximum for a
        frequency-axis peak to qualify. Default 0.1 (10%).
    max_freq_jump : float or None
        Maximum frequency change (rad/yr) between adjacent time steps
        for ridge continuity. If None, defaults to median spacing
        between adjacent frequency bins.
    min_duration_samples : int
        Minimum ridge length in samples to keep. Default 26 (~6 months).

    Returns
    -------
    ridges : list of dict
        Each ridge has:
        - 'time_indices': ndarray of sample indices
        - 'freq_indices': ndarray of frequency bin indices
        - 'frequencies': ndarray of frequencies (rad/yr) at each time
        - 'amplitudes': ndarray of envelope amplitudes along ridge
        - 'duration_samples': int
        - 'duration_years': float
        - 'mean_freq': float (rad/yr)
        - 'std_freq': float
        - 'mean_period_weeks': float
        - 'drift_rate': float (rad/yr per year, from linear fit)
        - 'r_squared': float (quality of linear drift fit)
        - 'ridge_id': int
    """
    n_scales, n_samples = scalogram_matrix.shape

    # Default max_freq_jump: median spacing between frequency bins
    if max_freq_jump is None:
        spacings = np.diff(frequencies)
        max_freq_jump = np.median(spacings) * 3.0

    # Step 1: Find peaks at each time column
    # Store as list of lists: peaks_per_col[t] = list of (freq_idx, amplitude)
    peaks_per_col = []
    for t in range(n_samples):
        col = scalogram_matrix[:, t]
        col_max = col.max()
        if col_max <= 0:
            peaks_per_col.append([])
            continue

        prom_threshold = col_max * min_prominence
        peak_idx, props = find_peaks(col, prominence=prom_threshold)

        peaks = [(pi, col[pi]) for pi in peak_idx]
        peaks_per_col.append(peaks)

    # Step 2: Chain peaks across time by nearest-frequency matching
    # Active ridges: list of dicts with 'time_indices', 'freq_indices', 'amplitudes'
    active_ridges = []
    finished_ridges = []

    for t in range(n_samples):
        peaks = peaks_per_col[t]
        used_peaks = set()

        if active_ridges and peaks:
            # For each active ridge, find closest peak within max_freq_jump
            # Sort ridges by last amplitude (brightest first) to give priority
            ridge_order = sorted(range(len(active_ridges)),
                                 key=lambda i: active_ridges[i]['amplitudes'][-1],
                                 reverse=True)

            for ri in ridge_order:
                ridge = active_ridges[ri]
                last_freq = frequencies[ridge['freq_indices'][-1]]
                best_dist = max_freq_jump
                best_pi = None

                for pi_idx, (pi, amp) in enumerate(peaks):
                    if pi_idx in used_peaks:
                        continue
                    dist = abs(frequencies[pi] - last_freq)
                    if dist < best_dist:
                        best_dist = dist
                        best_pi = pi_idx

                if best_pi is not None:
                    pi, amp = peaks[best_pi]
                    ridge['time_indices'].append(t)
                    ridge['freq_indices'].append(pi)
                    ridge['amplitudes'].append(amp)
                    used_peaks.add(best_pi)

        # Terminate ridges that didn't get extended this step
        still_active = []
        for ridge in active_ridges:
            if ridge['time_indices'][-1] == t:
                still_active.append(ridge)
            else:
                finished_ridges.append(ridge)
        active_ridges = still_active

        # Start new ridges from unmatched peaks
        for pi_idx, (pi, amp) in enumerate(peaks):
            if pi_idx not in used_peaks:
                active_ridges.append({
                    'time_indices': [t],
                    'freq_indices': [pi],
                    'amplitudes': [amp],
                })

    # Finalize remaining active ridges
    finished_ridges.extend(active_ridges)

    # Step 3: Filter by minimum duration and compute statistics
    ridges = []
    ridge_id = 0
    fs_assumed = 52  # weekly data

    for r in finished_ridges:
        t_arr = np.array(r['time_indices'])
        f_arr = np.array(r['freq_indices'])
        a_arr = np.array(r['amplitudes'])
        duration = len(t_arr)

        if duration < min_duration_samples:
            continue

        freq_values = frequencies[f_arr]
        duration_years = duration / fs_assumed

        # Linear drift fit: freq = a + b * time
        t_years = t_arr / fs_assumed
        if len(t_years) > 1:
            coeffs = np.polyfit(t_years, freq_values, 1)
            drift_rate = coeffs[0]  # rad/yr per year
            # R-squared
            predicted = np.polyval(coeffs, t_years)
            ss_res = np.sum((freq_values - predicted) ** 2)
            ss_tot = np.sum((freq_values - freq_values.mean()) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        else:
            drift_rate = 0.0
            r_squared = 1.0

        mean_freq = float(np.mean(freq_values))
        ridges.append({
            'time_indices': t_arr,
            'freq_indices': f_arr,
            'frequencies': freq_values,
            'amplitudes': a_arr,
            'duration_samples': duration,
            'duration_years': duration_years,
            'mean_freq': mean_freq,
            'std_freq': float(np.std(freq_values)),
            'mean_period_weeks': float(2.0 * np.pi / mean_freq * 52.0),
            'drift_rate': drift_rate,
            'r_squared': r_squared,
            'ridge_id': ridge_id,
        })
        ridge_id += 1

    # Sort by mean frequency
    ridges.sort(key=lambda r: r['mean_freq'])

    return ridges


def match_ridges_to_nominal(ridges, nominal_frequencies, tolerance=0.5):
    """
    Match detected ridges to nominal model line frequencies.

    For each ridge, finds the nearest nominal line by mean frequency.

    Parameters
    ----------
    ridges : list of dict
        Output from detect_ridges().
    nominal_frequencies : ndarray
        Nominal model line frequencies (rad/yr).
    tolerance : float
        Maximum distance (rad/yr) to consider a match.

    Returns
    -------
    matches : list of dict
        Each match has:
        - 'ridge_id': int
        - 'nominal_line': int (0-indexed)
        - 'nominal_freq': float
        - 'ridge_mean_freq': float
        - 'distance': float
        - 'drift_rate': float
    unmatched_ridges : list of int
        Ridge IDs with no nominal match.
    unmatched_nominal : list of int
        Nominal line indices with no ridge match.
    """
    matches = []
    matched_ridge_ids = set()
    matched_nominal_ids = set()

    for ridge in ridges:
        dists = np.abs(nominal_frequencies - ridge['mean_freq'])
        nearest = int(np.argmin(dists))
        dist = dists[nearest]

        if dist <= tolerance:
            matches.append({
                'ridge_id': ridge['ridge_id'],
                'nominal_line': nearest,
                'nominal_freq': float(nominal_frequencies[nearest]),
                'ridge_mean_freq': ridge['mean_freq'],
                'distance': float(dist),
                'drift_rate': ridge['drift_rate'],
            })
            matched_ridge_ids.add(ridge['ridge_id'])
            matched_nominal_ids.add(nearest)

    unmatched_ridges = [r['ridge_id'] for r in ridges
                        if r['ridge_id'] not in matched_ridge_ids]
    unmatched_nominal = [i for i in range(len(nominal_frequencies))
                         if i not in matched_nominal_ids]

    return matches, unmatched_ridges, unmatched_nominal


def compute_ridge_statistics(ridges):
    """
    Compute aggregate statistics across all ridges.

    Parameters
    ----------
    ridges : list of dict
        Output from detect_ridges().

    Returns
    -------
    stats : dict
        - 'n_ridges': int
        - 'mean_duration_years': float
        - 'median_duration_years': float
        - 'total_coverage': float (fraction of max possible time covered)
        - 'drift_rates': ndarray of all drift rates
        - 'mean_drift_rate': float
        - 'std_drift_rate': float
        - 'mean_frequencies': ndarray
        - 'freq_range': tuple (min, max)
    """
    if not ridges:
        return {
            'n_ridges': 0,
            'mean_duration_years': 0.0,
            'median_duration_years': 0.0,
            'total_coverage': 0.0,
            'drift_rates': np.array([]),
            'mean_drift_rate': 0.0,
            'std_drift_rate': 0.0,
            'mean_frequencies': np.array([]),
            'freq_range': (0.0, 0.0),
        }

    durations = np.array([r['duration_years'] for r in ridges])
    drifts = np.array([r['drift_rate'] for r in ridges])
    mean_freqs = np.array([r['mean_freq'] for r in ridges])

    # Coverage: fraction of time axis covered by any ridge
    all_times = set()
    for r in ridges:
        all_times.update(r['time_indices'].tolist())
    max_time = max(r['time_indices'].max() for r in ridges) + 1
    coverage = len(all_times) / max_time if max_time > 0 else 0.0

    return {
        'n_ridges': len(ridges),
        'mean_duration_years': float(np.mean(durations)),
        'median_duration_years': float(np.median(durations)),
        'total_coverage': coverage,
        'drift_rates': drifts,
        'mean_drift_rate': float(np.mean(drifts)),
        'std_drift_rate': float(np.std(drifts)),
        'mean_frequencies': mean_freqs,
        'freq_range': (float(mean_freqs.min()), float(mean_freqs.max())),
    }
