# -*- coding: utf-8 -*-
"""
Page 152: 6-Filter CMW Projection v5 -- Cycle Statistics Method
================================================================

Instead of mathematical models (sine waves, EKF, MPM), this approach measures
ACTUAL cycle statistics from the bandpass filter outputs:
  - Peak detection → cycle period distribution (peak-to-peak)
  - Envelope amplitude distribution per half-cycle
  - Median cycle template extraction via phase-normalized resampling
  - Forward projection by tiling median template with ±1σ timing uncertainty

Layout inspired by TradingHurst charts:
  Row 0: Price + composite + overlaid cycles with yellow envelopes + projection band
  Rows 1-5: Individual BP filter oscillators + projections + stats panels
  Row 6: Heatmap (amplitude per filter band over time) + spectral cross-section

Compared against v1 (static sinusoid) baseline for projection accuracy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.loaders import getStooq
from src.time_frequency.cmw import apply_cmw


# ============================================================================
# Configuration
# ============================================================================

FS = 52
TWOPI = 2 * np.pi

DISPLAY_START = '1935-01-01'
DISPLAY_END = '1954-02-01'
PROJECTION_WEEKS = 100

# Cycle statistics parameters
MIN_CYCLES_FOR_STATS = 3
N_SIGMA = 1.0
TEMPLATE_POINTS = 100  # phase grid resolution for median template

# Heatmap
HEATMAP_CMAP = 'inferno'

FILTER_SPECS = [
    {'type': 'lp', 'f_pass': 0.85, 'f_stop': 1.25,
     'label': 'LP-1: Trend (>5 yr)', 'color': '#4fc3f7'},
    {'type': 'bp', 'f1': 0.85, 'f2': 1.25, 'f3': 2.05, 'f4': 2.45,
     'label': 'BP-2: ~3.8 yr', 'color': '#ff7f0e'},
    {'type': 'bp', 'f1': 3.20, 'f2': 3.55, 'f3': 6.35, 'f4': 6.70,
     'label': 'BP-3: ~1.3 yr', 'color': '#2ca02c'},
    {'type': 'bp', 'f1': 7.25, 'f2': 7.55, 'f3': 9.55, 'f4': 9.85,
     'label': 'BP-4: ~0.7 yr', 'color': '#d62728'},
    {'type': 'bp', 'f1': 13.65, 'f2': 13.95, 'f3': 19.35, 'f4': 19.65,
     'label': 'BP-5: ~0.4 yr', 'color': '#9467bd'},
    {'type': 'bp', 'f1': 28.45, 'f2': 28.75, 'f3': 35.95, 'f4': 36.25,
     'label': 'BP-6: ~0.2 yr', 'color': '#8c564b'},
]


def spec_to_cmw(spec):
    if spec['type'] == 'lp':
        return 0.0, spec['f_pass'] + spec['f_stop']
    f0 = (spec['f2'] + spec['f3']) / 2.0
    lower = (spec['f1'] + spec['f2']) / 2.0
    upper = (spec['f3'] + spec['f4']) / 2.0
    return f0, upper - lower


# ============================================================================
# Cycle Statistics
# ============================================================================

def measure_cycle_statistics(sig_real, envelope, phase, fs, disp_s, disp_e):
    """
    Measure cycle statistics from a BP filter output within the display window.

    Uses peak-to-peak intervals for period measurement and envelope peaks
    for amplitude. Builds a median cycle template by phase-normalizing each
    complete cycle and taking the pointwise median.

    Returns None if insufficient cycles are detected.
    """
    sig_w = sig_real[disp_s:disp_e]
    env_w = envelope[disp_s:disp_e]
    phase_w = phase[disp_s:disp_e]
    N = len(sig_w)

    # Detect peaks and troughs
    peaks, _ = find_peaks(sig_w)
    troughs, _ = find_peaks(-sig_w)

    if len(peaks) < MIN_CYCLES_FOR_STATS + 1:
        return None

    # Full cycle periods (peak-to-peak)
    peak_periods = np.diff(peaks).astype(float)

    # Half-cycle amplitudes: envelope value at each peak
    peak_amps = env_w[peaks]

    # Also measure trough-to-trough for comparison
    if len(troughs) >= 2:
        trough_periods = np.diff(troughs).astype(float)
        all_periods = np.concatenate([peak_periods, trough_periods])
    else:
        all_periods = peak_periods

    period_median = np.median(all_periods)
    period_std = np.std(all_periods)
    amp_median = np.median(peak_amps)
    amp_std = np.std(peak_amps)

    # Build median cycle template: extract each peak-to-peak cycle,
    # resample to common phase grid, take median
    phase_grid = np.linspace(0, TWOPI, TEMPLATE_POINTS, endpoint=False)
    templates = []

    for i in range(len(peaks) - 1):
        s, e = peaks[i], peaks[i + 1]
        if e - s < 4:
            continue
        cycle_sig = sig_w[s:e]
        cycle_phase = phase_w[s:e]

        # Normalize phase to [0, 2π)
        ph0 = cycle_phase[0]
        cycle_phase_norm = cycle_phase - ph0
        # Should span approximately 2π; handle wrapping
        span = cycle_phase_norm[-1]
        if span < np.pi or span > 3 * np.pi:
            continue  # irregular cycle, skip
        # Scale to exactly [0, 2π)
        cycle_phase_norm = cycle_phase_norm / span * TWOPI

        try:
            interp_fn = interp1d(cycle_phase_norm, cycle_sig,
                                 kind='linear', bounds_error=False,
                                 fill_value=(cycle_sig[0], cycle_sig[-1]))
            templates.append(interp_fn(phase_grid))
        except Exception:
            continue

    if len(templates) < MIN_CYCLES_FOR_STATS:
        # Fallback: pure cosine at median amplitude
        median_template = amp_median * np.cos(phase_grid)
    else:
        median_template = np.median(np.array(templates), axis=0)

    # Current phase at display end
    current_phase = phase[disp_e - 1]

    # Envelope template (for heatmap projection)
    env_template = np.abs(median_template)

    return {
        'period_median': period_median,
        'period_std': period_std,
        'amplitude_median': amp_median,
        'amplitude_std': amp_std,
        'n_cycles': len(peaks) - 1,
        'n_templates': len(templates),
        'median_template': median_template,
        'env_template': env_template,
        'phase_grid': phase_grid,
        'current_phase': current_phase,
        'peak_indices': peaks + disp_s,  # absolute indices
        'trough_indices': troughs + disp_s,
        'peak_periods': peak_periods,
        'peak_amps': peak_amps,
    }


def project_with_uncertainty(stats, n_proj, n_sigma=1.0):
    """
    Project forward using the median cycle template with ±1σ timing bands.

    The center projection tiles the template at the median phase rate.
    Upper/lower bands use faster/slower phase rates (±1σ of period std).
    """
    template = stats['median_template']
    phase_grid = stats['phase_grid']
    current_phase = stats['current_phase']
    period_med = stats['period_median']
    period_std = stats['period_std']

    dphi = TWOPI / period_med  # radians per sample at median period

    t = np.arange(1, n_proj + 1)
    phase_center = current_phase + dphi * t

    # Template interpolator (periodic via modular phase)
    interp_fn = interp1d(phase_grid, template, kind='linear',
                         bounds_error=False,
                         fill_value=(template[0], template[-1]))

    center = interp_fn(np.mod(phase_center, TWOPI))

    # Timing uncertainty: vary period by ±σ
    period_fast = max(period_med - n_sigma * period_std, period_med * 0.5)
    period_slow = min(period_med + n_sigma * period_std, period_med * 2.0)

    phase_fast = current_phase + (TWOPI / period_fast) * t
    phase_slow = current_phase + (TWOPI / period_slow) * t

    upper = interp_fn(np.mod(phase_fast, TWOPI))
    lower = interp_fn(np.mod(phase_slow, TWOPI))

    # Envelope of the band: at each t, take min/max of the three traces
    band_upper = np.maximum(center, np.maximum(upper, lower))
    band_lower = np.minimum(center, np.minimum(upper, lower))

    # Predicted peak/trough sample indices (relative to projection start)
    # Peaks occur where template phase ≈ phase_of_template_max
    tmpl_peak_phase = phase_grid[np.argmax(template)]
    tmpl_trough_phase = phase_grid[np.argmin(template)]

    def find_phase_crossings(phase_series, target_phase):
        """Find sample indices where phase crosses target (mod 2π)."""
        wrapped = np.mod(phase_series - target_phase + np.pi, TWOPI) - np.pi
        crossings = []
        for j in range(len(wrapped) - 1):
            if wrapped[j] <= 0 < wrapped[j + 1] or wrapped[j] >= 0 > wrapped[j + 1]:
                crossings.append(j)
        return np.array(crossings)

    peak_times = find_phase_crossings(phase_center, tmpl_peak_phase)
    trough_times = find_phase_crossings(phase_center, tmpl_trough_phase)

    # Timing uncertainty for peaks/troughs (±σ in weeks)
    timing_sigma_weeks = period_std / TWOPI * np.pi  # half-cycle uncertainty

    return {
        'center': center,
        'upper': band_upper,
        'lower': band_lower,
        'phase_center': phase_center,
        'peak_times': peak_times,
        'trough_times': trough_times,
        'timing_sigma': timing_sigma_weeks,
    }


def project_lowpass(signal, disp_s, disp_e, n_proj):
    """LP filter: linear extrapolation from last 52 weeks."""
    lookback = min(52, disp_e - disp_s)
    segment = np.real(signal[disp_e - lookback:disp_e])
    t_seg = np.arange(lookback, dtype=float)
    coeffs = np.polyfit(t_seg, segment, 1)
    t_fwd = np.arange(lookback, lookback + n_proj, dtype=float)
    proj = np.polyval(coeffs, t_fwd)
    return {
        'center': proj,
        'upper': None,
        'lower': None,
        'slope_per_yr': coeffs[0] * FS,
    }


# ============================================================================
# Heatmap Matrix
# ============================================================================

def build_heatmap_matrix(filter_outputs, projections, cycle_stats_list,
                         disp_s, disp_e, n_proj):
    """
    Build time-frequency amplitude heatmap.

    Each row = one filter's envelope amplitude (normalized 0-1).
    Rows ordered by period (longest at top).
    Historical part uses actual envelope; projection tiles the median envelope.
    """
    n_filters = len(filter_outputs)
    n_disp = disp_e - disp_s
    n_total = n_disp + n_proj

    matrix = np.zeros((n_filters, n_total))
    freq_labels = []
    f0_values = []

    for i, (out, proj, stats) in enumerate(
            zip(filter_outputs, projections, cycle_stats_list)):
        spec = out['spec']
        f0 = out['f0']
        f0_values.append(f0)

        # Historical envelope
        if out['envelope'] is not None:
            env_hist = out['envelope'][disp_s:disp_e]
        else:
            sig = out['signal']
            env_hist = np.abs(sig[disp_s:disp_e])
        matrix[i, :n_disp] = np.abs(env_hist)

        # Projection envelope
        if stats is not None and stats['env_template'] is not None:
            env_tmpl = stats['env_template']
            # Tile from current phase position
            dphi = TWOPI / stats['period_median']
            current_phase = stats['current_phase']
            t_proj = np.arange(1, n_proj + 1)
            phase_proj = current_phase + dphi * t_proj
            phase_wrapped = np.mod(phase_proj, TWOPI)

            interp_fn = interp1d(stats['phase_grid'], env_tmpl,
                                 kind='linear', bounds_error=False,
                                 fill_value=np.mean(env_tmpl))
            env_proj = interp_fn(phase_wrapped)
            matrix[i, n_disp:] = env_proj
        elif proj is not None:
            # LP or fallback: use abs of projection
            matrix[i, n_disp:] = np.abs(proj['center'])
        else:
            matrix[i, n_disp:] = matrix[i, n_disp - 1]

        # Label
        if f0 > 0:
            T_wk = TWOPI / f0 * FS
            if T_wk >= 52:
                freq_labels.append(f'{T_wk/52:.1f} yr')
            else:
                freq_labels.append(f'{T_wk:.0f} wk')
        else:
            freq_labels.append('Trend')

    # Sort by period (longest = lowest freq at top)
    sort_idx = np.argsort(f0_values)  # ascending freq → descending period
    matrix = matrix[sort_idx]
    freq_labels = [freq_labels[i] for i in sort_idx]
    f0_sorted = [f0_values[i] for i in sort_idx]

    # Normalize each row independently
    for i in range(n_filters):
        row = matrix[i]
        vmin = np.percentile(row, 2)
        vmax = np.percentile(row, 98)
        if vmax > vmin:
            matrix[i] = (row - vmin) / (vmax - vmin)
        matrix[i] = np.clip(matrix[i], 0, 1)

    return {
        'matrix': matrix,
        'freq_labels': freq_labels,
        'f0_sorted': f0_sorted,
        'n_disp': n_disp,
        'sort_idx': sort_idx,
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_results(dates, close, filter_outputs, projections, cycle_stats_list,
                 heatmap_data, disp_s, disp_e, proj_e, composite):
    """TradingHurst-style dark theme figure."""

    n_proj = proj_e - disp_e
    dates_disp = dates[disp_s:disp_e]
    dates_proj = dates[disp_e:proj_e]
    actual_proj = close[disp_e:proj_e]

    # --- Dark theme ---
    plt.rcParams.update({
        'figure.facecolor': '#0a0a1a',
        'axes.facecolor': '#0a0a1a',
        'axes.edgecolor': '#333355',
        'axes.labelcolor': '#ccccdd',
        'text.color': '#ccccdd',
        'xtick.color': '#999999',
        'ytick.color': '#999999',
        'grid.color': '#222244',
        'grid.alpha': 0.4,
    })

    fig = plt.figure(figsize=(22, 28))
    height_ratios = [3.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3]
    gs = gridspec.GridSpec(7, 2, figure=fig,
                           width_ratios=[4, 1],
                           height_ratios=height_ratios,
                           hspace=0.35, wspace=0.12,
                           left=0.06, right=0.97, top=0.95, bottom=0.03)

    # ==== Row 0: Price + composite + projection ====
    ax0 = fig.add_subplot(gs[0, :])

    ax0.plot(dates_disp, close[disp_s:disp_e], color='#ddddee', lw=1.2,
             label='DJIA Close', zorder=5)

    # Overlay longer cycle components with yellow envelopes
    for i, out in enumerate(filter_outputs[:3]):  # LP, BP-2, BP-3
        sig = out['signal']
        sig_r = sig.real if np.iscomplexobj(sig) else sig
        ax0.plot(dates_disp, sig_r[disp_s:disp_e],
                 color=out['spec']['color'], lw=0.8, alpha=0.6)
        if out['envelope'] is not None:
            env = out['envelope']
            ax0.plot(dates_disp, env[disp_s:disp_e],
                     color='#ffdd44', lw=0.5, alpha=0.4)
            ax0.plot(dates_disp, -env[disp_s:disp_e],
                     color='#ffdd44', lw=0.5, alpha=0.4)

    ax0.plot(dates_disp, composite[disp_s:disp_e], color='cyan', lw=0.8,
             alpha=0.5, label='Composite')

    # Composite projection
    comp_center = np.sum([p['center'] for p in projections], axis=0)
    comp_upper_parts = []
    comp_lower_parts = []
    for p in projections:
        comp_upper_parts.append(p['upper'] if p['upper'] is not None else p['center'])
        comp_lower_parts.append(p['lower'] if p['lower'] is not None else p['center'])
    comp_upper = np.sum(comp_upper_parts, axis=0)
    comp_lower = np.sum(comp_lower_parts, axis=0)

    ax0.plot(dates_proj, actual_proj, color='#ddddee', lw=1, alpha=0.3,
             label='Actual (holdout)')
    ax0.plot(dates_proj, comp_center, color='#ff44ff', lw=2, alpha=0.9,
             label='v5 Projection')
    ax0.fill_between(dates_proj, comp_lower, comp_upper,
                     color='#ff44ff', alpha=0.15, label='±1σ timing band')
    ax0.axvline(dates[disp_e], color='#666688', ls='--', lw=1, alpha=0.7)

    # Add dominant cycle (BP-2, BP-3) peak/trough zones on price chart
    for bp_idx in [1, 2]:  # BP-2 and BP-3
        p = projections[bp_idx]
        if 'peak_times' in p and 'timing_sigma' in p:
            sigma = p['timing_sigma']
            for pt in p.get('trough_times', []):
                if pt < n_proj:
                    pt = int(pt)
                    lo = max(0, int(pt - sigma))
                    hi = min(n_proj - 1, int(pt + sigma))
                    ax0.axvspan(dates_proj[lo], dates_proj[hi],
                                color='#ff4444', alpha=0.04)
            for pt in p.get('peak_times', []):
                if pt < n_proj:
                    pt = int(pt)
                    lo = max(0, int(pt - sigma))
                    hi = min(n_proj - 1, int(pt + sigma))
                    ax0.axvspan(dates_proj[lo], dates_proj[hi],
                                color='#44ff44', alpha=0.04)

    # Metrics
    corr = np.corrcoef(actual_proj, comp_center)[0, 1]
    rmse = np.sqrt(np.mean((actual_proj - comp_center)**2))
    ss_r = np.sum((actual_proj - comp_center)**2)
    ss_t = np.sum((actual_proj - np.mean(actual_proj))**2)
    r2 = 1 - ss_r / ss_t if ss_t > 0 else 0

    ax0.set_title(f'v5: Cycle Statistics Projection — {PROJECTION_WEEKS}-Week Forecast  '
                  f'(R²={r2:.3f}  r={corr:.3f}  RMSE={rmse:.1f})',
                  fontsize=13, fontweight='bold', pad=12)
    ax0.set_ylabel('Price', fontsize=10)
    ax0.legend(loc='upper left', fontsize=8, framealpha=0.5)
    ax0.grid(True)
    ax0.xaxis.set_major_locator(mdates.YearLocator(2))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ==== Rows 1-5: BP filters ====
    bp_outputs = [(out, proj, stats) for out, proj, stats
                  in zip(filter_outputs, projections, cycle_stats_list)
                  if out['spec']['type'] == 'bp']

    for row, (out, proj, stats) in enumerate(bp_outputs):
        spec = out['spec']
        color = spec['color']

        # Left: oscillator + projection
        ax_osc = fig.add_subplot(gs[row + 1, 0])
        sig_r = out['signal'].real if np.iscomplexobj(out['signal']) else out['signal']

        ax_osc.plot(dates_disp, sig_r[disp_s:disp_e], color=color, lw=0.7)
        if out['envelope'] is not None:
            env = out['envelope']
            ax_osc.plot(dates_disp, env[disp_s:disp_e],
                        color='#ffdd44', lw=0.5, alpha=0.6)
            ax_osc.plot(dates_disp, -env[disp_s:disp_e],
                        color='#ffdd44', lw=0.5, alpha=0.6)

        # Actual holdout (faint)
        if proj_e <= len(sig_r):
            ax_osc.plot(dates_proj, sig_r[disp_e:proj_e],
                        color=color, lw=0.4, alpha=0.25)

        # Projection
        if proj is not None:
            ax_osc.plot(dates_proj, proj['center'],
                        color=color, lw=1.5, ls='--', alpha=0.9)
            if proj['upper'] is not None:
                ax_osc.fill_between(dates_proj, proj['lower'], proj['upper'],
                                    color=color, alpha=0.15)

            # Peak/trough zone markers (limit to avoid clutter on high-freq)
            if 'peak_times' in proj and 'timing_sigma' in proj:
                sigma_wk = proj['timing_sigma']
                max_markers = 8  # limit markers for readability

                pk_times = proj.get('peak_times', [])
                tr_times = proj.get('trough_times', [])
                pk_times = pk_times[pk_times < n_proj][:max_markers]
                tr_times = tr_times[tr_times < n_proj][:max_markers]

                for pt in pk_times:
                    pt = int(pt)
                    lo = max(0, int(pt - sigma_wk))
                    hi = min(n_proj - 1, int(pt + sigma_wk))
                    ax_osc.axvspan(dates_proj[lo], dates_proj[hi],
                                   color='#44ff44', alpha=0.06)
                    ax_osc.plot(dates_proj[pt], proj['center'][pt],
                                'v', color='#44ff44', markersize=4, alpha=0.7)

                for tt in tr_times:
                    tt = int(tt)
                    lo = max(0, int(tt - sigma_wk))
                    hi = min(n_proj - 1, int(tt + sigma_wk))
                    ax_osc.axvspan(dates_proj[lo], dates_proj[hi],
                                   color='#ff4444', alpha=0.06)
                    ax_osc.plot(dates_proj[tt], proj['center'][tt],
                                '^', color='#ff4444', markersize=4, alpha=0.7)

        ax_osc.axvline(dates[disp_e], color='#666688', ls='--', lw=0.7, alpha=0.5)
        ax_osc.axhline(0, color='#444466', lw=0.3)
        ax_osc.set_ylabel(spec['label'], fontsize=7, rotation=0,
                          labelpad=90, ha='left')
        ax_osc.grid(True)
        ax_osc.tick_params(axis='both', labelsize=7)
        ax_osc.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_osc.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Right: stats panel
        ax_st = fig.add_subplot(gs[row + 1, 1])
        ax_st.axis('off')

        if stats is not None:
            T_wk = stats['period_median']
            T_yr = T_wk / FS
            lines = [
                spec['label'],
                '',
                f"Period: {T_wk:.1f} ± {stats['period_std']:.1f} wk",
                f"       ({T_yr:.2f} ± {stats['period_std']/FS:.2f} yr)",
                f"Amplitude: {stats['amplitude_median']:.1f} ± {stats['amplitude_std']:.1f}",
                f"Cycles: {stats['n_cycles']} ({stats['n_templates']} templated)",
            ]

            # Per-filter projection correlation
            if proj_e <= len(sig_r):
                af = sig_r[disp_e:proj_e]
                fc = np.corrcoef(af, proj['center'])[0, 1] if np.std(proj['center']) > 0 else 0
                lines.append(f"Proj corr: {fc:+.3f}")

            txt = '\n'.join(lines)
            ax_st.text(0.05, 0.5, txt, fontsize=7, color='#ccccdd',
                       va='center', family='monospace',
                       transform=ax_st.transAxes)
        else:
            ax_st.text(0.05, 0.5, f'{spec["label"]}\nInsufficient cycles',
                       fontsize=7, color='#666688', va='center',
                       transform=ax_st.transAxes)

    # ==== Row 6: Heatmap + spectral cross-section ====
    ax_heat = fig.add_subplot(gs[6, 0])
    ax_spec = fig.add_subplot(gs[6, 1])

    mat = heatmap_data['matrix']
    n_disp = heatmap_data['n_disp']
    labels = heatmap_data['freq_labels']
    n_filters = mat.shape[0]

    # Use dates for x-axis
    all_dates = np.concatenate([dates_disp, dates_proj])
    extent_left = mdates.date2num(pd.Timestamp(dates_disp[0]))
    extent_right = mdates.date2num(pd.Timestamp(dates_proj[-1]))

    im = ax_heat.imshow(mat, aspect='auto', cmap=HEATMAP_CMAP,
                        extent=[extent_left, extent_right,
                                n_filters - 0.5, -0.5],
                        interpolation='bilinear')
    ax_heat.xaxis_date()
    ax_heat.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_heat.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Projection boundary
    proj_date_num = mdates.date2num(pd.Timestamp(dates[disp_e]))
    ax_heat.axvline(proj_date_num, color='white', ls='--', lw=1, alpha=0.6)

    # Y-axis labels
    ax_heat.set_yticks(range(n_filters))
    ax_heat.set_yticklabels(labels, fontsize=7)
    ax_heat.set_title('Time-Frequency Amplitude Heatmap', fontsize=10,
                      fontweight='bold', pad=8)
    ax_heat.tick_params(axis='x', labelsize=7)

    # Spectral cross-section at projection boundary
    profile = mat[:, n_disp - 1]
    y_pos = np.arange(n_filters)
    bars = ax_spec.barh(y_pos, profile, color='magenta', alpha=0.7, height=0.7)
    ax_spec.set_yticks(range(n_filters))
    ax_spec.set_yticklabels(labels, fontsize=6)
    ax_spec.set_xlim(0, 1.1)
    ax_spec.set_xlabel('Amplitude', fontsize=7)
    ax_spec.set_title('Current\nSpectrum', fontsize=8, fontweight='bold')
    ax_spec.invert_yaxis()
    ax_spec.grid(True, axis='x')
    ax_spec.tick_params(axis='both', labelsize=6)

    # Suptitle
    fig.suptitle("Page 152: 6-Filter CMW — Cycle Statistics Projection (v5)\n"
                 "Measured cycle periods + median template + ±1σ timing uncertainty",
                 fontsize=12, fontweight='bold', y=0.98, color='#ddddee')

    return fig, comp_center, comp_upper, comp_lower


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Page 152: 6-Filter CMW Projection v5 -- Cycle Statistics")
    print("=" * 70)

    # --- Load data ---
    df = getStooq('^dji', 'w')
    df = df.sort_values('Date').reset_index(drop=True)
    dates = pd.to_datetime(df['Date']).values
    close = df['Close'].values.astype(np.float64)

    print(f"\nData: {len(close)} weekly samples")
    print(f"  {pd.Timestamp(dates[0]).strftime('%Y-%m-%d')} to "
          f"{pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}")

    # --- Display window ---
    disp_s = np.searchsorted(dates, np.datetime64(DISPLAY_START))
    disp_e = np.searchsorted(dates, np.datetime64(DISPLAY_END))
    proj_e = min(disp_e + PROJECTION_WEEKS, len(close))
    n_proj = proj_e - disp_e

    print(f"\nDisplay: {DISPLAY_START} to {DISPLAY_END} ({disp_e - disp_s} weeks)")
    print(f"Projection: {n_proj} weeks to "
          f"{pd.Timestamp(dates[proj_e-1]).strftime('%Y-%m-%d')}")

    # --- Apply 6 CMW filters ---
    print(f"\n--- Applying 6 CMW filters ---")
    filter_outputs = []

    for spec in FILTER_SPECS:
        f0, fwhm = spec_to_cmw(spec)
        analytic = (spec['type'] != 'lp')
        result = apply_cmw(close, f0, fwhm, FS, analytic=analytic)
        result['spec'] = spec
        result['f0'] = f0
        result['fwhm'] = fwhm
        filter_outputs.append(result)

        if analytic:
            print(f"  {spec['label']:20s}  f0={f0:.2f}  FWHM={fwhm:.2f}  "
                  f"T={TWOPI/f0:.2f}yr ({TWOPI/f0*FS:.0f}wk)")
        else:
            print(f"  {spec['label']:20s}  f0={f0:.2f}  FWHM={fwhm:.2f}  (lowpass)")

    # --- Composite ---
    composite = np.zeros_like(close)
    for out in filter_outputs:
        sig = out['signal']
        composite += sig.real if np.iscomplexobj(sig) else sig

    # --- Measure cycle statistics ---
    print(f"\n--- Cycle Statistics ---")
    cycle_stats_list = []

    for out in filter_outputs:
        spec = out['spec']
        if spec['type'] == 'lp':
            cycle_stats_list.append(None)
            continue

        sig_r = out['signal'].real if np.iscomplexobj(out['signal']) else out['signal']
        stats = measure_cycle_statistics(sig_r, out['envelope'], out['phase'],
                                         FS, disp_s, disp_e)
        cycle_stats_list.append(stats)

        if stats is not None:
            T_yr = stats['period_median'] / FS
            print(f"  {spec['label']:20s}  T={stats['period_median']:.1f} "
                  f"±{stats['period_std']:.1f} wk  ({T_yr:.2f} yr)  "
                  f"A={stats['amplitude_median']:.1f}  "
                  f"cycles={stats['n_cycles']}  templates={stats['n_templates']}")
        else:
            print(f"  {spec['label']:20s}  ** insufficient cycles **")

    # --- Projections ---
    print(f"\n--- Computing projections ---")
    projections = []

    for i, (out, stats) in enumerate(zip(filter_outputs, cycle_stats_list)):
        spec = out['spec']

        if spec['type'] == 'lp':
            proj = project_lowpass(out['signal'], disp_s, disp_e, n_proj)
            projections.append(proj)
            print(f"  {spec['label']:20s}  LP linear extrap, "
                  f"slope={proj['slope_per_yr']:.2f}/yr")
            continue

        if stats is not None:
            proj = project_with_uncertainty(stats, n_proj, N_SIGMA)
            projections.append(proj)
            print(f"  {spec['label']:20s}  median template projection")
        else:
            # Fallback: constant zero
            proj = {
                'center': np.zeros(n_proj),
                'upper': np.zeros(n_proj),
                'lower': np.zeros(n_proj),
            }
            projections.append(proj)
            print(f"  {spec['label']:20s}  ** fallback: zero **")

    # --- Build heatmap ---
    heatmap_data = build_heatmap_matrix(filter_outputs, projections,
                                         cycle_stats_list,
                                         disp_s, disp_e, n_proj)

    # --- v1 baseline for comparison ---
    proj_v1 = []
    for out, stats in zip(filter_outputs, cycle_stats_list):
        spec = out['spec']
        if spec['type'] == 'lp':
            proj_v1.append(projections[0]['center'])
            continue
        env = out['envelope']
        phase = out['phase']
        A = env[disp_e - 1]
        f0 = out['f0']
        period_samples = TWOPI / f0 * FS
        n_lb = int(3 * period_samples)
        phase_w = phase[max(disp_s, disp_e - n_lb):disp_e]
        dphi = np.median(np.diff(phase_w))
        phi0 = phase[disp_e - 1]
        t_fwd = np.arange(1, n_proj + 1)
        pv1 = A * np.cos(phi0 + dphi * t_fwd)
        proj_v1.append(pv1)

    comp_v1 = np.sum(proj_v1, axis=0)
    comp_v5 = np.sum([p['center'] for p in projections], axis=0)
    actual_proj = close[disp_e:proj_e]

    # --- Metrics ---
    def metrics(actual, pred, name):
        corr = np.corrcoef(actual, pred)[0, 1]
        ss_r = np.sum((actual - pred)**2)
        ss_t = np.sum((actual - np.mean(actual))**2)
        r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
        rmse = np.sqrt(np.mean((actual - pred)**2))
        err_end = (pred[-1] / actual[-1] - 1) * 100
        return corr, r2, rmse, err_end

    c1, r1, rm1, e1 = metrics(actual_proj, comp_v1, 'v1')
    c5, r5, rm5, e5 = metrics(actual_proj, comp_v5, 'v5')

    print(f"\n{'='*65}")
    print(f"PROJECTION COMPARISON ({n_proj} weeks)")
    print(f"{'='*65}")
    print(f"  {'Metric':<20s}  {'v1 static':>12s}  {'v5 cyclestats':>12s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}")
    print(f"  {'Correlation':<20s}  {c1:>12.4f}  {c5:>12.4f}")
    print(f"  {'R²':<20s}  {r1:>12.4f}  {r5:>12.4f}")
    print(f"  {'RMSE':<20s}  {rm1:>12.1f}  {rm5:>12.1f}")
    print(f"  {'End price':<20s}  {comp_v1[-1]:>12.1f}  {comp_v5[-1]:>12.1f}")
    print(f"  {'End price error':<20s}  {e1:>+11.1f}%  {e5:>+11.1f}%")
    print(f"  {'Actual end price':<20s}  {actual_proj[-1]:>12.1f}")

    # --- Plot ---
    fig, _, _, _ = plot_results(dates, close, filter_outputs, projections,
                                cycle_stats_list, heatmap_data,
                                disp_s, disp_e, proj_e, composite)

    outpath = os.path.join(os.path.dirname(__file__),
                           'cmw_6filter_projection_v5_cyclestats.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nFigure saved: {outpath}")

    try:
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    except Exception:
        plt.close()


if __name__ == '__main__':
    main()
