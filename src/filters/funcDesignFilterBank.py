# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 21:35:59 2026

@author: Dime
"""

"""
Ormsby Filter Bank - Functional Implementation
Integrates with existing OrmsbyComplexWithFilterBank.py functions
"""
from .funcOrmsby import ormsby_filter, apply_ormsby_filter

"""
Ormsby Filter Bank - Functional Implementation
Integrates with existing OrmsbyComplexWithFilterBank.py functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd


def design_ormsby_filter_bank(fs=52,
                               nw_base=199*7,
                               f_lp_pass=0.85,
                               f_lp_stop=1.25,
                               Q_factor=2.0,
                               spacing_mode='balanced',  # 'constant_q', 'balanced', or 'overlap_3db'
                               max_freq=None):
    """
    Design a bank of Ormsby filters that sweep the spectrum.
    
    Parameters:
    -----------
    fs : float
        Sampling rate (data points per year)
    nw_base : int
        Base filter length (will be adjusted per filter based on bandwidth)
    f_lp_pass : float
        Lowpass passband edge in rad/year
    f_lp_stop : float
        Lowpass stopband edge in rad/year
    Q_factor : float
        Quality factor Q = f_center / bandwidth (used as starting point)
        Higher Q = narrower filters
        Typical values: 1-4 for audio, 2-8 for analysis
    spacing_mode : str
        'constant_q': Strict constant-Q, overlap at -3dB points
        'balanced': Adjust bandwidth to make filters meet at skirt edges (better unity gain)
        'overlap_3db': Overlap at -3dB points (50% gain)
    max_freq : float or None
        Maximum center frequency (default: 0.8 * Nyquist)
        
    Returns:
    --------
    filter_specs : list of dict
        List of filter specifications
    """
    nyquist = np.pi * fs
    
    if max_freq is None:
        max_freq = 0.8 * nyquist
    
    filter_specs = []
    
    # First filter: lowpass
    lp_spec = {
        'type': 'lp',
        'f_pass': f_lp_pass,
        'f_stop': f_lp_stop,
        'f_center': (f_lp_pass + f_lp_stop) / 2,
        'bandwidth': f_lp_stop - f_lp_pass,
        'f_3db_right': (f_lp_pass + f_lp_stop) / 2,  # Midpoint of transition
        'transition_bw': f_lp_stop - f_lp_pass,
        'nw': nw_base,
        'index': 0
    }
    filter_specs.append(lp_spec)
    
    # Calculate transition bandwidth ratio from LP
    transition_bw_base = f_lp_stop - f_lp_pass
    
    # Design bandpass filters
    filter_index = 1
    
    # For balanced mode, we need to plan ahead
    # Let's use a geometric progression for center frequencies
    if spacing_mode == 'balanced':
        # Use constant-Q to get initial center frequencies
        temp_centers = []
        temp_f_center = f_lp_stop
        while temp_f_center < max_freq:
            temp_centers.append(temp_f_center)
            bw = temp_f_center / Q_factor
            trans_ratio = transition_bw_base / lp_spec['bandwidth']
            trans_bw = bw * trans_ratio
            # Next center using -3dB overlap approximation
            denominator = 1 - (1 + trans_ratio) / (2 * Q_factor)
            if denominator > 0.1:
                f_3db_right = temp_f_center + bw/2 + trans_bw/2
                temp_f_center = f_3db_right / denominator
            else:
                temp_f_center = temp_f_center + bw
            if len(temp_centers) > 100:
                break
        
        # Now design filters with adjusted bandwidths to fill gaps
        prev_f4 = f_lp_stop  # Previous filter's f4 (stopband edge)
        
        for i, current_f_center in enumerate(temp_centers):
            if current_f_center >= max_freq:
                break
                
            # Calculate ideal bandwidth from constant-Q
            bandwidth_ideal = current_f_center / Q_factor
            trans_ratio = transition_bw_base / lp_spec['bandwidth']
            trans_bw = bandwidth_ideal * trans_ratio
            
            # BALANCED MODE: Adjust f2 to start where previous filter's f4 ended
            # This ensures continuous coverage
            f2 = prev_f4  # Start passband where previous stopband ended
            
            # Keep center frequency, solve for f3
            # f_center = (f2 + f3) / 2
            # f3 = 2 * f_center - f2
            f3 = 2 * current_f_center - f2
            
            # Recalculate actual bandwidth
            bandwidth = f3 - f2
            
            # Add transition bands
            f1 = f2 - trans_bw
            f4 = f3 + trans_bw
            
            # Ensure f1 >= minimum
            if f1 < 0.1:
                f1 = 0.1
                f2 = f1 + trans_bw
                f3 = f2 + bandwidth
                f4 = f3 + trans_bw
                current_f_center = (f2 + f3) / 2
            
            # Stop if exceeding max frequency
            if f4 > max_freq * 1.2:
                break
            
            # Calculate actual Q for this filter
            actual_Q = current_f_center / bandwidth if bandwidth > 0 else Q_factor
            
            # Scale filter length
            nw_scale = max(0.3, 1.0 / (current_f_center / lp_spec['f_center']))
            nw = int(nw_base * nw_scale)
            if nw % 2 == 0:
                nw += 1
            
            bp_spec = {
                'type': 'bp',
                'f1': f1,
                'f2': f2,
                'f3': f3,
                'f4': f4,
                'f_center': current_f_center,
                'bandwidth': bandwidth,
                'Q': actual_Q,
                'Q_target': Q_factor,
                'f_3db_left': (f1 + f2) / 2,
                'f_3db_right': (f3 + f4) / 2,
                'transition_bw': trans_bw,
                'nw': nw,
                'index': filter_index
            }
            filter_specs.append(bp_spec)
            
            prev_f4 = f4  # Update for next iteration
            filter_index += 1
            
            if filter_index > 100:
                print("Warning: Reached 100 filters, stopping")
                break
    
    else:  # constant_q or overlap_3db modes
        current_f_center = f_lp_stop
        
        while current_f_center < max_freq:
            # Calculate bandwidth from Q factor
            bandwidth = current_f_center / Q_factor
            
            # Calculate transition bands
            trans_ratio = transition_bw_base / lp_spec['bandwidth']
            trans_bw = bandwidth * trans_ratio
            
            # Four corner frequencies
            f2 = current_f_center - bandwidth / 2
            f3 = current_f_center + bandwidth / 2
            f1 = f2 - trans_bw
            f4 = f3 + trans_bw
            
            # Calculate -3dB points
            f_3db_left = (f1 + f2) / 2
            f_3db_right = (f3 + f4) / 2
            
            # Ensure f1 >= minimum
            if f1 < 0.1:
                f1 = 0.1
                f2 = f1 + trans_bw
                f3 = f2 + bandwidth
                f4 = f3 + trans_bw
                current_f_center = (f2 + f3) / 2
                f_3db_left = (f1 + f2) / 2
                f_3db_right = (f3 + f4) / 2
            
            # Stop if exceeding max frequency
            if f4 > max_freq * 1.2:
                break
            
            # Scale filter length
            nw_scale = max(0.3, 1.0 / (current_f_center / lp_spec['f_center']))
            nw = int(nw_base * nw_scale)
            if nw % 2 == 0:
                nw += 1
            
            bp_spec = {
                'type': 'bp',
                'f1': f1,
                'f2': f2,
                'f3': f3,
                'f4': f4,
                'f_center': current_f_center,
                'bandwidth': bandwidth,
                'Q': Q_factor,
                'Q_target': Q_factor,
                'f_3db_left': f_3db_left,
                'f_3db_right': f_3db_right,
                'transition_bw': trans_bw,
                'nw': nw,
                'index': filter_index
            }
            filter_specs.append(bp_spec)
            
            # Calculate next center frequency
            if spacing_mode == 'overlap_3db':
                # Overlap at -3dB points (50% gain)
                denominator = 1 - (1 + trans_ratio) / (2 * Q_factor)
                if denominator > 0.1:
                    next_f_center = f_3db_right / denominator
                else:
                    next_f_center = current_f_center + bandwidth
            else:  # constant_q
                # Simple constant-Q spacing
                next_f_center = current_f_center + bandwidth
            
            current_f_center = next_f_center
            filter_index += 1
            
            if filter_index > 100:
                print("Warning: Reached 100 filters, stopping")
                break
    
    print(f"Generated {len(filter_specs)} filters (mode: {spacing_mode})")
    print(f"Frequency range: {f_lp_pass:.2f} to {filter_specs[-1]['f_center']:.2f} rad/year")
    
    return filter_specs


def design_hurst_comb_bank(
    n_filters=23,
    w1_start=7.2,
    w_step=0.2,
    passband_width=0.2,
    skirt_width=0.3,
    nw=1393,
    fs=52
):
    """
    Design Hurst's uniform-step overlapping comb filter bank.

    Reproduces the 23-filter comb bank from Appendix A, Figure AI-2.
    All filters have identical bandwidth and skirt width, with center
    frequencies uniformly spaced by w_step rad/year.

    Parameters
    ----------
    n_filters : int
        Number of comb filters (default: 23, per Hurst)
    w1_start : float
        Lower skirt edge (w1) of first filter, in rad/year (default: 7.2)
    w_step : float
        Frequency step between successive filters, in rad/year (default: 0.2)
        Each filter's four edges are shifted by this amount from the previous.
    passband_width : float
        Width of flat passband (w3 - w2), in rad/year (default: 0.2)
    skirt_width : float
        Width of each transition band (w2 - w1 = w4 - w3), in rad/year (default: 0.3)
    nw : int
        Filter length in samples. If None, auto-computed from skirt_width and fs.
        Default: 1393 (199*7, consistent with existing project convention).
    fs : float
        Sampling rate in samples per year (default: 52 for weekly data)

    Returns
    -------
    filter_specs : list of dict
        List of filter specifications, each containing:
        - 'type': 'bp'
        - 'f1', 'f2', 'f3', 'f4': Corner frequencies in rad/year
        - 'f_center': Center frequency (f2 + f3) / 2 in rad/year
        - 'bandwidth': Passband width (f3 - f2) in rad/year
        - 'skirt_width': Transition bandwidth in rad/year
        - 'Q', 'Q_target': Quality factor (f_center / bandwidth)
        - 'nw': Filter length
        - 'index': Filter number (0-based)
        - 'label': Human-readable label (e.g., "FC-1")

    Notes
    -----
    Hurst's specification (Appendix A, p. 192):
    - Filter 1:  w1=7.2, w2=7.5, w3=7.7, w4=8.0 rad/year
    - Filter 2:  w1=7.4, w2=7.7, w3=7.9, w4=8.2 rad/year
    - ...
    - Filter 23: w1=11.6, w2=11.9, w3=12.1, w4=12.4 rad/year

    Each subsequent filter shifts all four edges by w_step (0.2 rad/yr).
    Center frequencies: 7.6, 7.8, 8.0, ..., 12.0 rad/year.
    Total span per filter: passband_width + 2*skirt_width = 0.8 rad/year.

    Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing,
               Appendix A, Figure AI-2, p. 192.
    """
    # Auto-compute nw if not provided
    if nw is None:
        f_skirt_cycles = skirt_width / (2 * np.pi)
        n_cycles = 4  # standard for good Ormsby rolloff
        nw = int(np.ceil(n_cycles * fs / f_skirt_cycles))
        if nw % 2 == 0:
            nw += 1

    nyquist_rad = np.pi * fs

    filter_specs = []

    for i in range(n_filters):
        w1 = w1_start + i * w_step
        w2 = w1 + skirt_width
        w3 = w2 + passband_width
        w4 = w3 + skirt_width
        f_center = (w2 + w3) / 2.0

        if w4 > nyquist_rad:
            raise ValueError(
                f"Filter {i+1} f4={w4:.2f} rad/yr exceeds Nyquist "
                f"({nyquist_rad:.2f} rad/yr)"
            )

        Q = f_center / passband_width

        spec = {
            'type': 'bp',
            'f1': w1,
            'f2': w2,
            'f3': w3,
            'f4': w4,
            'f_center': f_center,
            'bandwidth': passband_width,
            'skirt_width': skirt_width,
            'Q': Q,
            'Q_target': Q,
            'nw': nw,
            'index': i,
            'label': f"FC-{i+1}"
        }
        filter_specs.append(spec)

    print(f"Hurst Comb Bank: {n_filters} filters")
    print(f"  Center freq range: {filter_specs[0]['f_center']:.1f} - "
          f"{filter_specs[-1]['f_center']:.1f} rad/year")
    print(f"  Passband width: {passband_width} rad/year")
    print(f"  Skirt width: {skirt_width} rad/year")
    print(f"  Total span per filter: {passband_width + 2*skirt_width} rad/year")
    print(f"  Filter length: {nw} samples")

    return filter_specs


def create_filter_kernels(filter_specs, fs=52, filter_type='modulate', analytic=True):
    """
    Create filter coefficient arrays from specifications.
    Uses ormsby_filter() from your existing code.
    
    Parameters:
    -----------
    filter_specs : list of dict
        Output from design_ormsby_filter_bank()
    fs : float
        Sampling rate
    filter_type : str
        'modulate' or 'subtract' (for BP filters)
    analytic : bool
        If True, create complex analytic filters
        
    Returns:
    --------
    filters : list of dict
        Each containing:
        - kernel: filter coefficients (array)
        - spec: filter specification
        - nw: filter length
    """
    twopi = 2 * np.pi
    filters = []
    
    for spec in filter_specs:
        if spec['type'] == 'lp':
            f_edges = np.array([spec['f_pass'], spec['f_stop']]) / twopi
            h = ormsby_filter(
                nw=spec['nw'],
                f_edges=f_edges,
                fs=fs,
                filter_type='lp',
                analytic=analytic
            )
        else:  # bandpass
            f_edges = np.array([spec['f1'], spec['f2'], spec['f3'], spec['f4']]) / twopi
            h = ormsby_filter(
                nw=spec['nw'],
                f_edges=f_edges,
                fs=fs,
                filter_type='bp',
                method=filter_type,
                analytic=analytic
            )
        
        filters.append({
            'kernel': h,
            'spec': spec,
            'nw': spec['nw']
        })
    
    return filters


def apply_filter_bank(signal, filters, fs=52, mode='reflect',
                      spacing=1, offset=1, interp='none'):
    """
    Apply all filters in the bank to a signal.
    Uses apply_ormsby_filter() from your existing code.

    Parameters:
    -----------
    signal : array-like
        Input signal
    filters : list of dict
        Output from create_filter_kernels()
    fs : float
        Sampling rate (samples per year)
    mode : str
        Boundary handling: 'reflect', 'zeropad', or 'valid'
    spacing : int
        Decimation factor. 1 = no decimation (default). N = every Nth sample.
    offset : int
        1-based starting index (1 through spacing). Default 1.
    interp : str
        Gap-filling method: 'none', '3point', 'cubic', 'linear'

    Returns:
    --------
    results : dict containing:
        - filter_outputs: list of filter output dicts
        - filter_specs: list of specifications
        - signal: original input signal
    """
    from .decimation import decimate_signal, interpolate_output_dict, VALID_METHODS

    signal = np.asarray(signal, dtype=np.float64)
    full_length = len(signal)

    results = {
        'filter_outputs': [],
        'filter_specs': [f['spec'] for f in filters],
        'signal': signal
    }

    # No decimation: original path
    if spacing == 1:
        for i, filt in enumerate(filters):
            output = apply_ormsby_filter(
                signal=signal,
                h=filt['kernel'],
                mode=mode,
                fs=fs
            )
            output['spec'] = filt['spec']
            output['index'] = i
            results['filter_outputs'].append(output)
        return results

    # --- Decimated path ---
    if interp not in VALID_METHODS:
        raise ValueError(f"interp must be one of {VALID_METHODS}, got '{interp}'")

    signal_dec, indices = decimate_signal(signal, spacing, offset)
    fs_dec = fs / spacing
    nyq_dec = np.pi * fs_dec  # Nyquist in rad/yr

    twopi = 2 * np.pi

    for i, filt in enumerate(filters):
        spec = filt['spec']

        # Nyquist check: highest frequency edge must be below decimated Nyquist
        if spec['type'] == 'lp':
            f_max_rad = spec['f_stop']
        else:
            f_max_rad = spec['f4']
        if f_max_rad > nyq_dec:
            raise ValueError(
                f"Filter {i} ({spec.get('label', '')}) has f_max={f_max_rad:.2f} rad/yr "
                f"which exceeds decimated Nyquist={nyq_dec:.2f} rad/yr "
                f"(spacing={spacing}, fs_dec={fs_dec:.1f})"
            )

        # Redesign kernel for decimated sampling rate
        nw_dec = max(51, filt['nw'] // spacing)
        if nw_dec % 2 == 0:
            nw_dec += 1  # keep odd

        # Determine filter type and analytic from original kernel
        is_analytic = np.iscomplexobj(filt['kernel'])

        if spec['type'] == 'lp':
            f_edges = np.array([spec['f_pass'], spec['f_stop']]) / twopi
            h_dec = ormsby_filter(
                nw=nw_dec, f_edges=f_edges, fs=fs_dec,
                filter_type='lp', analytic=is_analytic
            )
        else:
            f_edges = np.array([spec['f1'], spec['f2'], spec['f3'], spec['f4']]) / twopi
            # Infer method from spec if available, default to 'modulate'
            method = spec.get('method', 'modulate')
            h_dec = ormsby_filter(
                nw=nw_dec, f_edges=f_edges, fs=fs_dec,
                filter_type='bp', method=method, analytic=is_analytic
            )

        # Apply to decimated signal
        output = apply_ormsby_filter(
            signal=signal_dec, h=h_dec, mode=mode, fs=fs_dec
        )

        # Interpolate back to full length
        output = interpolate_output_dict(output, indices, full_length, method=interp)

        output['spec'] = spec
        output['index'] = i
        results['filter_outputs'].append(output)

    return results


def plot_filter_bank_response(filters, fs=52, nfft=4096, normalize_analytic=True):
    """
    Plot frequency response of all filters in the bank.
    
    Parameters:
    -----------
    filters : list of dict
        Output from create_filter_kernels()
    fs : float
        Sampling rate
    nfft : int
        FFT length for frequency analysis
    normalize_analytic : bool
        If True, divide analytic filter responses by 2 for unity gain display
        
    Returns:
    --------
    fig : matplotlib figure
    """
    from scipy.fft import fft
    
    nyquist = np.pi * fs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    
    # Frequency axis in rad/year (freqs_norm * fs gives cycles/yr, * 2pi -> rad/yr)
    freqs_norm = np.arange(nfft) / nfft
    pos_mask = freqs_norm <= 0.5
    freqs_rad = freqs_norm[pos_mask] * fs * 2 * np.pi
    
    # Check if filters are analytic (complex)
    is_analytic = np.iscomplexobj(filters[0]['kernel'])
    scale_factor = 0.5 if (is_analytic and normalize_analytic) else 1.0
    
    # Plot individual responses
    sum_response = np.zeros(len(freqs_rad))
    
    for filt in filters:
        H = fft(filt['kernel'], n=nfft)
        H_mag = np.abs(H[pos_mask]) * scale_factor
        
        spec = filt['spec']
        label = f"F{spec['index']}: {spec['f_center']:.2f} rad/yr"
        
        # Linear magnitude
        ax1.plot(freqs_rad, H_mag, alpha=0.6, label=label)
        
        # dB magnitude
        H_db = 20 * np.log10(H_mag + 1e-10)
        ax2.plot(freqs_rad, H_db, alpha=0.6)
        
        # Sum for reconstruction (power for analytic)
        sum_response += H_mag**2
    
    # Plot sum response
    sum_response = np.sqrt(sum_response)
    ax1.plot(freqs_rad, sum_response, 'k-', linewidth=2.5, label='Sum')
    ax1.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Unity')
    ax1.set_xlabel('Frequency (rad/year)')
    ax1.set_ylabel('Magnitude')
    title_suffix = ' (Normalized)' if (is_analytic and normalize_analytic) else ''
    ax1.set_title(f'Filter Bank - Linear Magnitude{title_suffix}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.set_xlim(0, min(max(freqs_rad), nyquist * 0.9))
    ax1.set_ylim(0, 1.5)
    
    # dB plot
    sum_db = 20 * np.log10(sum_response + 1e-10)
    ax2.plot(freqs_rad, sum_db, 'k-', linewidth=2.5, label='Sum')
    ax2.axhline(0, color='r', linestyle='--', alpha=0.5, label='0 dB')
    ax2.axhline(-3, color='orange', linestyle=':', alpha=0.5, label='-3 dB')
    ax2.set_xlabel('Frequency (rad/year)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Filter Bank - dB Magnitude')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-40, 5)
    ax2.set_xlim(0, min(max(freqs_rad), nyquist * 0.9))
    ax2.legend()
    
    plt.tight_layout()
    return fig


def plot_idealized_comb_response(filter_specs, filters=None, fs=52, nfft=8192,
                                  figsize=(14, 8)):
    """
    Plot idealized trapezoidal frequency response of a comb filter bank,
    reproducing Hurst's Figure AI-2 style.

    For each filter, draws a trapezoid: (f1, 0) -> (f2, 1) -> (f3, 1) -> (f4, 0).
    Optionally overlays actual FFT-computed frequency response if filter kernels
    are provided.

    Parameters
    ----------
    filter_specs : list of dict
        Output from design_hurst_comb_bank() or similar. Each dict must have
        'f1', 'f2', 'f3', 'f4' in rad/year.
    filters : list of dict, optional
        Output from create_filter_kernels(). If provided, overlays actual
        frequency response.
    fs : float
        Sampling rate (samples per year)
    nfft : int
        FFT length for actual response computation
    figsize : tuple
        Figure size

    Returns
    -------
    fig : matplotlib Figure

    Reference: J.M. Hurst, Appendix A, Figure AI-2, p. 192.
    """
    from scipy.fft import fft

    fig, ax = plt.subplots(figsize=figsize)

    n_filters = len(filter_specs)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_filters, 20)))

    # Draw idealized trapezoidal shapes
    for i, spec in enumerate(filter_specs):
        f1, f2, f3, f4 = spec['f1'], spec['f2'], spec['f3'], spec['f4']
        color = colors[i % len(colors)]

        # Trapezoid vertices
        trap_f = [f1, f2, f3, f4]
        trap_a = [0.0, 1.0, 1.0, 0.0]

        ax.plot(trap_f, trap_a, '-', color=color, linewidth=1.2, alpha=0.8)
        ax.fill(trap_f, trap_a, color=color, alpha=0.05)

        # Label at top
        label_x = (f2 + f3) / 2.0
        ax.text(label_x, 1.02, str(i + 1), ha='center', va='bottom',
                fontsize=7, fontweight='bold')

    # Overlay actual FFT response if kernels provided
    if filters is not None:
        freqs_norm = np.arange(nfft) / nfft
        pos_mask = freqs_norm <= 0.5
        # Convert to rad/year: normalized freq * fs * 2*pi
        freqs_rad = freqs_norm[pos_mask] * fs * 2 * np.pi

        is_analytic = np.iscomplexobj(filters[0]['kernel'])
        scale_factor = 0.5 if is_analytic else 1.0

        for i, filt in enumerate(filters):
            H = fft(filt['kernel'], n=nfft)
            H_mag = np.abs(H[pos_mask]) * scale_factor
            color = colors[i % len(colors)]
            ax.plot(freqs_rad, H_mag, '--', color=color, linewidth=0.6,
                    alpha=0.5)

    # Axis formatting to match Hurst's Figure AI-2
    f1_min = filter_specs[0]['f1']
    f4_max = filter_specs[-1]['f4']
    margin = 0.2
    ax.set_xlim(f1_min - margin, f4_max + margin)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('Angular Frequency (Rad./Yr.)')
    ax.set_ylabel('Amplitude Ratio')
    ax.set_title('IDEALIZED COMB FILTER\nFIGURE AI-2 Reproduction')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_time_frequency_heatmap(results, dates=None, vmin=None, vmax=None,
                                   figsize=(14, 8), cmap='jet'):
    """
    Create time-frequency heatmap of filter bank outputs.
    
    Parameters:
    -----------
    results : dict
        Output from apply_filter_bank()
    dates : array-like, optional
        Time axis labels
    vmin, vmax : float, optional
        Color scale limits
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
        
    Returns:
    --------
    fig : matplotlib figure
    tf_matrix : ndarray
        Time-frequency matrix (filters × time)
    """
    n_filters = len(results['filter_outputs'])
    n_samples = len(results['signal'])
    
    # Create matrix: rows = filters, columns = time
    tf_matrix = np.zeros((n_filters, n_samples))
    center_freqs = np.zeros(n_filters)
    
    for i, output in enumerate(results['filter_outputs']):
        if output['envelope'] is not None:
            # Use envelope for analytic filters
            tf_matrix[i, :] = output['envelope']
        else:
            # Use absolute value for real filters
            tf_matrix[i, :] = np.abs(output['signal'])
        
        center_freqs[i] = output['spec']['f_center']
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if dates is not None:
        extent = [0, len(dates)-1, center_freqs[-1], center_freqs[0]]
    else:
        extent = [0, n_samples-1, center_freqs[-1], center_freqs[0]]
    
    im = ax.imshow(tf_matrix, aspect='auto', origin='upper', 
                  cmap=cmap, extent=extent,
                  vmin=vmin, vmax=vmax,
                  interpolation='bilinear')
    
    ax.set_xlabel('Time (samples)' if dates is None else 'Date')
    ax.set_ylabel('Center Frequency (rad/year)')
    ax.set_title('Time-Frequency Analysis (Filter Bank Output)')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Amplitude (Envelope)')
    
    # Date ticks if provided
    if dates is not None:
        n_ticks = 10
        tick_indices = np.linspace(0, len(dates)-1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([str(dates[i])[:10] for i in tick_indices], rotation=45)
    
    plt.tight_layout()
    
    return fig, tf_matrix


def optimize_filter_bank_overlap(fs=52, nw_base=199*7, 
                                  f_lp_pass=0.85, f_lp_stop=1.25,
                                  Q_factor=2.0, max_freq=None,
                                  target_gain=1.0, tolerance=0.1):
    """
    Find optimal overlap_factor to minimize gain ripple.
    
    Parameters:
    -----------
    All standard parameters plus:
    target_gain : float
        Target sum gain (1.0 for unity)
    tolerance : float
        Acceptable deviation from target (±0.1 = ±10%)
        
    Returns:
    --------
    best_overlap_factor : float
    best_ripple : float (max deviation from target)
    """
    from scipy.fft import fft
    
    print("Searching for optimal overlap factor...")
    best_factor = 0.85
    best_ripple = float('inf')
    
    # Search range for overlap factor
    test_factors = np.linspace(0.5, 1.2, 15)
    
    for factor in test_factors:
        # Design filter bank
        specs = design_ormsby_filter_bank(
            fs=fs, nw_base=nw_base,
            f_lp_pass=f_lp_pass, f_lp_stop=f_lp_stop,
            Q_factor=Q_factor, overlap_factor=factor,
            max_freq=max_freq
        )
        
        # Create kernels
        filters = create_filter_kernels(specs, fs=fs, 
                                       filter_type='modulate', 
                                       analytic=True)
        
        # Compute sum response
        nfft = 4096
        freqs_norm = np.arange(nfft) / nfft
        pos_mask = freqs_norm <= 0.5
        
        sum_response = np.zeros(np.sum(pos_mask))
        for filt in filters:
            H = fft(filt['kernel'], n=nfft)
            H_mag = np.abs(H[pos_mask]) * 0.5  # Normalize analytic
            sum_response += H_mag**2
        sum_response = np.sqrt(sum_response)
        
        # Measure ripple in passband (exclude edges)
        valid_range = (sum_response > 0.3)  # Only in passband
        if np.any(valid_range):
            ripple = np.max(np.abs(sum_response[valid_range] - target_gain))
            
            if ripple < best_ripple:
                best_ripple = ripple
                best_factor = factor
                
            print(f"  overlap_factor={factor:.2f}: ripple={ripple:.3f}")
    
    print(f"\nOptimal overlap_factor: {best_factor:.2f}")
    print(f"Minimum ripple: ±{best_ripple:.3f} ({best_ripple*100:.1f}%)")
    
    if best_ripple > tolerance:
        print(f"Warning: Ripple exceeds tolerance of ±{tolerance}")
        print("Consider: (1) adjusting Q_factor, (2) accepting ripple, or (3) using equalization")
    
    return best_factor, best_ripple


def print_filter_specs(filter_specs):
    """Pretty print filter specifications."""
    print("=" * 80)
    print("FILTER BANK SPECIFICATIONS")
    print("=" * 80)
    
    for spec in filter_specs:
        if spec['type'] == 'lp':
            print(f"Filter {spec['index']:2d} (LOWPASS):")
            print(f"  Passband edge:  {spec['f_pass']:.3f} rad/yr")
            print(f"  Stopband edge:  {spec['f_stop']:.3f} rad/yr")
            print(f"  Bandwidth:      {spec['bandwidth']:.3f} rad/yr")
            print(f"  Filter length:  {spec['nw']} taps")
        else:
            print(f"Filter {spec['index']:2d} (BANDPASS):")
            print(f"  Corners: f1={spec['f1']:.3f}, f2={spec['f2']:.3f}, "
                  f"f3={spec['f3']:.3f}, f4={spec['f4']:.3f} rad/yr")
            print(f"  Center freq:    {spec['f_center']:.3f} rad/yr")
            print(f"  Bandwidth:      {spec['bandwidth']:.3f} rad/yr")
            print(f"  Q actual:       {spec['Q']:.2f} (target: {spec.get('Q_target', spec['Q']):.2f})")
            print(f"  Filter length:  {spec['nw']} taps")
        print()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    import sys
    sys.path.append('.')  # Ensure imports work
    
    # Design filter bank
    print("Designing filter bank...")
    specs = design_ormsby_filter_bank(
        fs=52,
        nw_base=199*7,
        f_lp_pass=0.85,
        f_lp_stop=1.25,
        Q_factor=2.0,
        spacing_mode='balanced',  # Try: 'constant_q', 'balanced', 'overlap_3db'
        max_freq=None
    )
    
    print("\nTry different modes:")
    print("  'balanced'    : Adjusts bandwidth for continuous coverage → best unity gain")
    print("  'overlap_3db' : Overlaps at -3dB points → good for analysis")
    print("  'constant_q'  : Strict constant-Q → best frequency resolution")
    
    # Print specifications
    print_filter_specs(specs)
    
    # Create filter kernels
    print("Creating filter kernels...")
    filters = create_filter_kernels(
        filter_specs=specs,
        fs=52,
        filter_type='modulate',
        analytic=True
    )
    print(f"Created {len(filters)} filter kernels\n")
    
    # Plot frequency responses
    print("Plotting frequency responses...")
    fig = plot_filter_bank_response(filters, fs=52)
    plt.show()
    
    # Example: Apply to synthetic signal
    print("\nExample with synthetic signal:")
    t = np.arange(0, 1000) / 52  # 1000 weeks
    signal = (np.sin(2 * np.pi * 1.5 * t) +      # 1.5 rad/yr component
              0.5 * np.sin(2 * np.pi * 5.0 * t) + # 5.0 rad/yr component
              0.3 * np.random.randn(len(t)))      # noise
    
    # Apply filter bank
    results = apply_filter_bank(signal, filters, fs=52, mode='reflect')
    
    # Create heatmap
    fig, tf_matrix = create_time_frequency_heatmap(results)
    plt.show()
    
    print("\n" + "=" * 80)
    print("Filter bank demonstration complete!")
    print("=" * 80)