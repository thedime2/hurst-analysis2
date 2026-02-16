# -*- coding: utf-8 -*-
"""
Phase 2 Comb Filter: AI-2 and AI-3 Reproduction
Modulate vs Subtract methods, Weekly vs Daily data

Reproduces Figures AI-2 and AI-3 from Hurst's Appendix A (p.192-193):
- AI-2: Actual FFT frequency response of 23 comb filters
        Compares modulate vs subtract, real vs complex, different NW
- AI-3: Time-domain outputs FC-1..FC-10 on single axis with vertical offsets,
        smoothed envelopes, matching Hurst's layout

Hurst's AI-3 window: 5-24-40 to 3-29-46 (~306 weeks)

Data range: 1921-04-29 to 1965-05-21
Reference: J.M. Hurst, Appendix A, Figures AI-2 and AI-3
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.filters import (
    design_hurst_comb_bank,
    create_filter_kernels,
    apply_filter_bank,
)


# ============================================================================
# HELPERS
# ============================================================================

def get_window_indices(dates_dt, date_start, date_end):
    """Return start_idx, end_idx for a date window."""
    mask = (dates_dt >= pd.to_datetime(date_start)) & \
           (dates_dt <= pd.to_datetime(date_end))
    if not mask.any():
        return 0, len(dates_dt)
    indices = np.where(mask)[0]
    return indices[0], indices[-1] + 1


def compute_fft_response(kernel, nfft, fs):
    """
    Compute frequency response properly for real or complex kernels.
    For complex analytic kernels, use full FFT and return positive freqs.
    Returns (freqs_radyr, magnitude_normalized).
    """
    if np.iscomplexobj(kernel):
        # Full FFT for complex kernel -- one-sided spectrum
        H = np.fft.fft(kernel, n=nfft)
        freqs_hz = np.fft.fftfreq(nfft, d=1.0 / fs)
        # Take positive frequencies only
        pos = freqs_hz >= 0
        freqs_radyr = freqs_hz[pos] * 2 * np.pi
        H_mag = np.abs(H[pos])
    else:
        # rfft for real kernel
        H = np.fft.rfft(kernel, n=nfft)
        freqs_hz = np.fft.rfftfreq(nfft, d=1.0 / fs)
        freqs_radyr = freqs_hz * 2 * np.pi
        H_mag = np.abs(H)

    H_norm = H_mag / np.max(H_mag) if np.max(H_mag) > 0 else H_mag
    return freqs_radyr, H_norm


# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

# Data paths
csv_weekly = os.path.join(script_dir, '../../data/raw/^dji_w.csv')
csv_daily = os.path.join(script_dir, '../../data/raw/^dji_d.csv')

# Hurst's analysis window
DATE_START = '1921-04-29'
DATE_END = '1965-05-21'

# Hurst's AI-3 display window (editorial error in book header corrected)
AI3_DATE_START = '1934-12-07'
AI3_DATE_END = '1940-01-26'

# Sampling rates
FS_WEEKLY = 52  # samples/year

# Comb filter bank parameters (Hurst, Appendix A, p.192)
N_FILTERS = 23
W1_START = 7.2         # rad/year - lower skirt edge of first filter
W_STEP = 0.2           # rad/year - step between successive filters
PASSBAND_WIDTH = 0.2   # rad/year - flat passband width
SKIRT_WIDTH = 0.3      # rad/year - transition band width
NW_WEEKLY = 3501       # Filter length for weekly data
NW_DAILY = 5001        # Filter length for daily data

# Number of filters to display in AI-3
N_DISPLAY = 10



# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("Phase 2 Comb Filter: AI-2 & AI-3 Reproduction")
print("Modulate vs Subtract, Weekly vs Daily, Real vs Complex")
print("=" * 80)
print()

# Weekly data
print("Loading DJIA weekly data...")
df_w = pd.read_csv(csv_weekly)
df_w['Date'] = pd.to_datetime(df_w['Date'])
df_hw = df_w[df_w.Date.between(DATE_START, DATE_END)]
close_weekly = df_hw.Close.values
dates_weekly = pd.to_datetime(df_hw.Date.values)
print(f"  Points: {len(close_weekly)}, Range: {DATE_START} to {DATE_END}")

# Daily data
print("Loading DJIA daily data...")
df_d = pd.read_csv(csv_daily)
df_d['Date'] = pd.to_datetime(df_d['Date'])
df_hd = df_d[df_d.Date.between(DATE_START, DATE_END)]
close_daily = df_hd.Close.values
dates_daily = pd.to_datetime(df_hd.Date.values)
total_years = (dates_daily[-1] - dates_daily[0]).days / 365.25
FS_DAILY = len(close_daily) / total_years
print(f"  Points: {len(close_daily)}, Effective fs: {FS_DAILY:.1f} trading days/year")
print()


# ============================================================================
# DESIGN & APPLY FILTER CONFIGURATIONS
# ============================================================================

# 6 configs: weekly/daily x modulate/subtract, all complex analytic
# Plus 2 real (non-complex) weekly configs for AI-2 comparison
configs = [
    {'label': 'Weekly Modulate (complex)',  'fs': FS_WEEKLY, 'nw': NW_WEEKLY,
     'method': 'modulate', 'analytic': True,
     'signal': close_weekly, 'dates': dates_weekly, 'tag': 'weekly_mod_cx'},
    {'label': 'Weekly Subtract (complex)',  'fs': FS_WEEKLY, 'nw': NW_WEEKLY,
     'method': 'subtract', 'analytic': True,
     'signal': close_weekly, 'dates': dates_weekly, 'tag': 'weekly_sub_cx'},
    {'label': 'Weekly Modulate (real)',     'fs': FS_WEEKLY, 'nw': NW_WEEKLY,
     'method': 'modulate', 'analytic': False,
     'signal': close_weekly, 'dates': dates_weekly, 'tag': 'weekly_mod_re'},
    {'label': 'Weekly Subtract (real)',     'fs': FS_WEEKLY, 'nw': NW_WEEKLY,
     'method': 'subtract', 'analytic': False,
     'signal': close_weekly, 'dates': dates_weekly, 'tag': 'weekly_sub_re'},
    {'label': 'Daily Modulate (complex)',   'fs': FS_DAILY,  'nw': NW_DAILY,
     'method': 'modulate', 'analytic': True,
     'signal': close_daily,  'dates': dates_daily,  'tag': 'daily_mod_cx'},
    {'label': 'Daily Subtract (complex)',   'fs': FS_DAILY,  'nw': NW_DAILY,
     'method': 'subtract', 'analytic': True,
     'signal': close_daily,  'dates': dates_daily,  'tag': 'daily_sub_cx'},
]

all_results = {}
all_filters = {}
all_specs = {}

for cfg in configs:
    print(f"--- {cfg['label']} ---")
    specs = design_hurst_comb_bank(
        n_filters=N_FILTERS,
        w1_start=W1_START,
        w_step=W_STEP,
        passband_width=PASSBAND_WIDTH,
        skirt_width=SKIRT_WIDTH,
        nw=cfg['nw'],
        fs=cfg['fs']
    )

    filters = create_filter_kernels(
        filter_specs=specs,
        fs=cfg['fs'],
        filter_type=cfg['method'],
        analytic=cfg['analytic']
    )

    print(f"  Applying {N_FILTERS} filters (nw={cfg['nw']}) to {len(cfg['signal'])} samples...")
    results = apply_filter_bank(
        signal=cfg['signal'],
        filters=filters,
        fs=cfg['fs'],
        mode='reflect'
    )

    all_results[cfg['tag']] = results
    all_filters[cfg['tag']] = filters
    all_specs[cfg['tag']] = specs
    print(f"  Done.")

print()


# ============================================================================
# FIGURE AI-2: FREQUENCY RESPONSE - 4-panel comparison
# Complex vs Real, Modulate vs Subtract
# ============================================================================

print("Generating AI-2: Frequency response (4-panel comparison)...")

NFFT = 65536  # Large FFT for fine frequency resolution in passband

fig_ai2, axes_ai2 = plt.subplots(2, 2, figsize=(18, 12), sharey=True, sharex=True)

panel_configs = [
    (axes_ai2[0, 0], 'weekly_mod_cx', 'Modulate - Complex Analytic'),
    (axes_ai2[0, 1], 'weekly_sub_cx', 'Subtract - Complex Analytic'),
    (axes_ai2[1, 0], 'weekly_mod_re', 'Modulate - Real'),
    (axes_ai2[1, 1], 'weekly_sub_re', 'Subtract - Real'),
]

for ax, tag, title in panel_configs:
    filters = all_filters[tag]
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(filters), 20)))

    for i, f in enumerate(filters):
        freqs_radyr, H_norm = compute_fft_response(f['kernel'], NFFT, FS_WEEKLY)
        ax.plot(freqs_radyr, H_norm, color=colors[i % len(colors)],
                linewidth=0.8, alpha=0.8)

    ax.set_xlim(6.5, 13.0)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Passband reference lines
    for spec in all_specs[tag]:
        ax.axvline(spec['f2'], color='gray', linewidth=0.2, alpha=0.3)
        ax.axvline(spec['f3'], color='gray', linewidth=0.2, alpha=0.3)

axes_ai2[1, 0].set_xlabel('Angular Frequency (rad/yr)')
axes_ai2[1, 1].set_xlabel('Angular Frequency (rad/yr)')
axes_ai2[0, 0].set_ylabel('Amplitude Ratio')
axes_ai2[1, 0].set_ylabel('Amplitude Ratio')

fig_ai2.suptitle(f'IDEALIZED COMB FILTER - Figure AI-2\n'
                  f'Actual FFT Response (nfft={NFFT}, nw={NW_WEEKLY})',
                  fontsize=13, fontweight='bold')
fig_ai2.tight_layout()
ai2_path = os.path.join(script_dir, 'phase2_comb_AI2_modulate_vs_subtract.png')
fig_ai2.savefig(ai2_path, dpi=150, bbox_inches='tight')
plt.close(fig_ai2)
print(f"  Saved: {ai2_path}")

# Also generate a zoomed view of a single filter to inspect passband flatness
print("Generating AI-2 passband zoom (single filter detail)...")
fig_zoom, axes_zoom = plt.subplots(2, 2, figsize=(16, 10), sharey=True, sharex=True)

zoom_filter_idx = 11  # FC-12, middle of bank
zoom_spec = all_specs['weekly_mod_cx'][zoom_filter_idx]
zoom_xlim = (zoom_spec['f1'] - 0.3, zoom_spec['f4'] + 0.3)

for ax, tag, title in [
    (axes_zoom[0, 0], 'weekly_mod_cx', 'Modulate - Complex'),
    (axes_zoom[0, 1], 'weekly_sub_cx', 'Subtract - Complex'),
    (axes_zoom[1, 0], 'weekly_mod_re', 'Modulate - Real'),
    (axes_zoom[1, 1], 'weekly_sub_re', 'Subtract - Real'),
]:
    kernel = all_filters[tag][zoom_filter_idx]['kernel']
    freqs_radyr, H_norm = compute_fft_response(kernel, NFFT, FS_WEEKLY)
    ax.plot(freqs_radyr, H_norm, 'b-', linewidth=1.5)

    # Ideal trapezoidal response
    spec = all_specs[tag][zoom_filter_idx]
    ideal_f = [spec['f1'], spec['f2'], spec['f3'], spec['f4']]
    ideal_h = [0, 1, 1, 0]
    ax.plot(ideal_f, ideal_h, 'r--', linewidth=1.0, alpha=0.7, label='Ideal')

    ax.set_xlim(zoom_xlim)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Annotate passband ripple
    mask_pb = (freqs_radyr >= spec['f2']) & (freqs_radyr <= spec['f3'])
    if np.any(mask_pb):
        pb_vals = H_norm[mask_pb]
        ripple_db = 20 * np.log10(np.max(pb_vals) / np.min(pb_vals)) if np.min(pb_vals) > 0 else 0
        ax.text(0.02, 0.85, f'Passband ripple: {ripple_db:.2f} dB',
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes_zoom[1, 0].set_xlabel('Angular Frequency (rad/yr)')
axes_zoom[1, 1].set_xlabel('Angular Frequency (rad/yr)')
axes_zoom[0, 0].set_ylabel('Amplitude Ratio')
axes_zoom[1, 0].set_ylabel('Amplitude Ratio')

fig_zoom.suptitle(f'PASSBAND DETAIL: FC-{zoom_filter_idx+1} '
                   f'(center={zoom_spec["f_center"]:.1f} rad/yr)\n'
                   f'nfft={NFFT}, nw={NW_WEEKLY}',
                   fontsize=12, fontweight='bold')
fig_zoom.tight_layout()
zoom_path = os.path.join(script_dir, 'phase2_comb_AI2_passband_zoom.png')
fig_zoom.savefig(zoom_path, dpi=150, bbox_inches='tight')
plt.close(fig_zoom)
print(f"  Saved: {zoom_path}")
print()


# ============================================================================
# FIGURE AI-3: SINGLE-AXIS STACKED LAYOUT (Hurst style)
# All FC-1..FC-10 on one axis with vertical offsets
# ============================================================================

def plot_AI3_stacked(results, dates_dt, date_start, date_end, fs,
                     n_display, save_path, title_suffix):
    """
    Plot comb filter outputs on a single axis with vertical offsets,
    matching Hurst's original AI-3 layout.

    Each filter is offset vertically. Horizontal lines mark zeros.
    Envelopes are smoothed for clean display.
    """
    s_idx, e_idx = get_window_indices(dates_dt, date_start, date_end)
    n_samples = e_idx - s_idx
    weeks = np.arange(n_samples) / fs * 52  # samples to weeks

    # Collect signals and envelopes for the display window
    signals = []
    envelopes = []
    center_freqs = []
    for i in range(n_display):
        output = results['filter_outputs'][i]
        sig_seg = output['signal'][s_idx:e_idx].real
        signals.append(sig_seg)
        center_freqs.append(output['spec']['f_center'])

        if output['envelope'] is not None:
            envelopes.append(output['envelope'][s_idx:e_idx])
        else:
            envelopes.append(None)

    # Use uniform amplitude scale: global max across all filters
    global_max = max(np.max(np.abs(s)) for s in signals)
    # Vertical spacing between filter zero-lines
    spacing = 2.2 * global_max  # leave ~10% gap between adjacent filter ranges

    fig, ax = plt.subplots(figsize=(16, 14))

    ytick_positions = []
    ytick_labels = []

    for i in range(n_display):
        # Offset: FC-1 at top, FC-10 at bottom
        offset = (n_display - 1 - i) * spacing

        # Zero reference line
        ax.axhline(offset, color='gray', linewidth=0.4, linestyle='-')

        # Plot signal
        ax.plot(weeks, signals[i] + offset, 'b-', linewidth=0.5)

        # Plot smoothed envelope
        if envelopes[i] is not None:
            ax.plot(weeks, envelopes[i] + offset, 'r-', linewidth=0.8, alpha=0.7)
            ax.plot(weeks, -envelopes[i] + offset, 'r-', linewidth=0.8, alpha=0.7)

        # Y-tick label
        cf = center_freqs[i]
        period_wk = 2 * np.pi / cf * 52
        ytick_positions.append(offset)
        ytick_labels.append(f"FC-{i+1}  {cf:.1f} r/y  ({period_wk:.0f}wk)")

        # +/- amplitude reference ticks
        for val, label in [(global_max, f'+{global_max:.0f}'), (-global_max, f'-{global_max:.0f}')]:
            ax.plot([-2, 0], [val + offset, val + offset], 'k-', linewidth=0.3)

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=8, family='monospace')
    ax.set_xlabel('Weeks', fontsize=11)
    ax.set_xlim(0, weeks[-1])
    ax.set_ylim(-spacing * 0.5, (n_display - 0.5) * spacing)
    ax.grid(True, axis='x', alpha=0.2)

    ax.set_title(f'COMB OUTPUT EXAMPLE - Figure AI-3\n{title_suffix}',
                  fontsize=13, fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


print("Generating AI-3 figures (single-axis stacked layout)...")

ai3_configs = [
    ('weekly_mod_cx', 'Weekly Modulate (complex)'),
    ('weekly_sub_cx', 'Weekly Subtract (complex)'),
    ('daily_mod_cx',  'Daily Modulate (complex)'),
    ('daily_sub_cx',  'Daily Subtract (complex)'),
]

for tag, label in ai3_configs:
    # Find matching config for dates/fs
    cfg = next(c for c in configs if c['tag'] == tag)
    save_path = os.path.join(script_dir, f'phase2_comb_AI3_{tag}.png')
    plot_AI3_stacked(
        all_results[tag], cfg['dates'],
        AI3_DATE_START, AI3_DATE_END,
        cfg['fs'], N_DISPLAY, save_path,
        f'{label} ({AI3_DATE_START} to {AI3_DATE_END})'
    )

print()


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("=" * 80)
print("SUMMARY: Modulate vs Subtract Comparison (Weekly, Complex)")
print("=" * 80)
print()

s_idx_w, e_idx_w = get_window_indices(dates_weekly, AI3_DATE_START, AI3_DATE_END)

print(f"AI-3 window: {AI3_DATE_START} to {AI3_DATE_END}")
print(f"  Weekly: indices {s_idx_w} to {e_idx_w} ({e_idx_w - s_idx_w} weeks)")
print()

print(f"{'FC':>4s}  {'Center':>8s}  {'Mod RMS':>9s}  {'Sub RMS':>9s}  {'Ratio':>7s}  "
      f"{'Mod MaxEnv':>10s}  {'Sub MaxEnv':>10s}")
print("-" * 70)

for i in range(N_DISPLAY):
    out_mod = all_results['weekly_mod_cx']['filter_outputs'][i]
    out_sub = all_results['weekly_sub_cx']['filter_outputs'][i]
    center = out_mod['spec']['f_center']

    seg_mod = out_mod['signal'][s_idx_w:e_idx_w].real
    seg_sub = out_sub['signal'][s_idx_w:e_idx_w].real
    rms_mod = np.sqrt(np.mean(seg_mod**2))
    rms_sub = np.sqrt(np.mean(seg_sub**2))
    ratio = rms_mod / rms_sub if rms_sub > 0 else float('inf')

    env_mod = np.max(out_mod['envelope'][s_idx_w:e_idx_w]) if out_mod['envelope'] is not None else 0
    env_sub = np.max(out_sub['envelope'][s_idx_w:e_idx_w]) if out_sub['envelope'] is not None else 0

    print(f"  {i+1:>2d}  {center:8.2f}  {rms_mod:9.4f}  {rms_sub:9.4f}  {ratio:7.3f}  "
          f"{env_mod:10.4f}  {env_sub:10.4f}")

print()

# Passband flatness summary
print("PASSBAND FLATNESS (nfft=65536):")
print(f"{'Config':>25s}  {'Mean Ripple(dB)':>15s}  {'Max Ripple(dB)':>15s}")
print("-" * 60)

for tag, label in [
    ('weekly_mod_cx', 'Modulate Complex'),
    ('weekly_sub_cx', 'Subtract Complex'),
    ('weekly_mod_re', 'Modulate Real'),
    ('weekly_sub_re', 'Subtract Real'),
]:
    ripples = []
    for f_obj, spec in zip(all_filters[tag], all_specs[tag]):
        freqs_radyr, H_norm = compute_fft_response(f_obj['kernel'], NFFT, FS_WEEKLY)
        mask_pb = (freqs_radyr >= spec['f2']) & (freqs_radyr <= spec['f3'])
        if np.any(mask_pb):
            pb = H_norm[mask_pb]
            if np.min(pb) > 0:
                ripples.append(20 * np.log10(np.max(pb) / np.min(pb)))

    if ripples:
        print(f"  {label:>23s}  {np.mean(ripples):15.3f}  {np.max(ripples):15.3f}")

print()
print("Figures generated:")
print(f"  phase2_comb_AI2_modulate_vs_subtract.png  -- 4-panel freq response")
print(f"  phase2_comb_AI2_passband_zoom.png          -- Single filter passband detail")
for tag, label in ai3_configs:
    print(f"  phase2_comb_AI3_{tag}.png  -- AI-3 {label}")
print()
print("Done.")
