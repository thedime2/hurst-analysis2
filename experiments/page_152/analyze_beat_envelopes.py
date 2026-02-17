# -*- coding: utf-8 -*-
"""
Page 152: Beat Envelope Analysis

For each BP filter (2-6), compute:
  1. Filter output over full 1921-1965 window
  2. Amplitude envelope via Hilbert transform
  3. Lanczos spectrum of envelope to find dominant modulation period
  4. Cross-correlation between all pairs of BP envelopes
  5. Variance explained by each filter

Reference: J.M. Hurst, The Profit Magic of Stock Transaction Timing, p. 152
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from src.filters import ormsby_filter, apply_ormsby_filter
from src.spectral import lanczos_spectrum

FS = 52
TWOPI = 2 * np.pi

# Filter specifications (all frequencies in rad/year)
FILTER_SPECS = [
    {"type": "lp", "f_pass": 0.85, "f_stop": 1.25, "nw": 1393, "label": "1"},
    {"type": "bp", "f1": 0.85, "f2": 1.25, "f3": 2.05, "f4": 2.45, "nw": 1393, "label": "2"},
    {"type": "bp", "f1": 3.20, "f2": 3.55, "f3": 6.35, "f4": 6.70, "nw": 1245, "label": "3"},
    {"type": "bp", "f1": 7.25, "f2": 7.55, "f3": 9.55, "f4": 9.85, "nw": 1745, "label": "4"},
    {"type": "bp", "f1": 13.65, "f2": 13.95, "f3": 19.35, "f4": 19.65, "nw": 1299, "label": "5"},
    {"type": "bp", "f1": 28.45, "f2": 28.75, "f3": 35.95, "f4": 36.25, "nw": 1299, "label": "6"},
]

# ============================================================================
# LOAD DATA
# ============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '../../data/raw/^dji_w.csv')

print("=" * 70)
print("Page 152: Beat Envelope Analysis")
print("=" * 70)
print()

print("Loading weekly DJIA data...")
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'])
df_hurst = df[df.Date.between('1921-04-29', '1965-05-21')].copy()
close_prices = df_hurst.Close.values.astype(np.float64)
dates = df_hurst.Date.values
n_points = len(close_prices)
print(f"  Loaded {n_points} samples from 1921-04-29 to 1965-05-21")
print()

total_var = np.var(close_prices)
print(f"  Total price variance: {total_var:.2f}")
print()

# ============================================================================
# COMPUTE FILTER OUTPUTS
# ============================================================================

print("Computing filter outputs...")
print("-" * 70)

filter_outputs = {}
envelopes = {}

for spec in FILTER_SPECS:
    label = spec["label"]
    nw = spec["nw"]

    if spec["type"] == "lp":
        # Low-pass filter
        f_edges_rad_yr = [spec["f_pass"], spec["f_stop"]]
        f_edges_cw = [f / TWOPI for f in f_edges_rad_yr]
        h = ormsby_filter(nw=nw, f_edges=f_edges_cw, fs=FS,
                          filter_type='lp', analytic=False)
        result = apply_ormsby_filter(close_prices, h, mode='reflect', fs=FS)
        sig = result['signal']
        filter_outputs[label] = sig
        var_explained = np.var(sig) / total_var * 100
        print(f"  Filter {label} (LP):  var={np.var(sig):.2f}  "
              f"({var_explained:.1f}% of total)")
    else:
        # Band-pass filter
        f_edges_rad_yr = [spec["f1"], spec["f2"], spec["f3"], spec["f4"]]
        f_edges_cw = [f / TWOPI for f in f_edges_rad_yr]
        h = ormsby_filter(nw=nw, f_edges=f_edges_cw, fs=FS,
                          filter_type='bp', method='modulate', analytic=False)
        result = apply_ormsby_filter(close_prices, h, mode='reflect', fs=FS)
        sig = result['signal']
        filter_outputs[label] = sig

        # Compute Hilbert envelope
        analytic_sig = hilbert(sig)
        env = np.abs(analytic_sig)
        envelopes[label] = env

        var_explained = np.var(sig) / total_var * 100
        fc = (spec["f2"] + spec["f3"]) / 2
        period_yr = TWOPI / fc
        print(f"  Filter {label} (BP):  fc={fc:.2f} rad/yr  T={period_yr:.2f} yr  "
              f"var={np.var(sig):.2f}  ({var_explained:.1f}% of total)")

# Residual
reconstructed = np.zeros_like(close_prices)
for label, sig in filter_outputs.items():
    reconstructed += sig
residual = close_prices - reconstructed
residual_var = np.var(residual) / total_var * 100

print()
print(f"  Sum of filter variances: "
      f"{sum(np.var(s) for s in filter_outputs.values()):.2f}  "
      f"({sum(np.var(s) for s in filter_outputs.values()) / total_var * 100:.1f}%)")
print(f"  Residual variance:       {np.var(residual):.2f}  ({residual_var:.1f}%)")
print()

# ============================================================================
# VARIANCE DECOMPOSITION SUMMARY
# ============================================================================

print("=" * 70)
print("VARIANCE DECOMPOSITION")
print("=" * 70)
print()
print(f"  {'Filter':<12s} {'Variance':>12s} {'% of Total':>12s} {'RMS':>12s}")
print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

for spec in FILTER_SPECS:
    label = spec["label"]
    sig = filter_outputs[label]
    v = np.var(sig)
    pct = v / total_var * 100
    rms = np.sqrt(np.mean(sig**2))
    ftype = "LP" if spec["type"] == "lp" else "BP"
    print(f"  {ftype}-{label:<9s} {v:12.2f} {pct:11.1f}% {rms:12.2f}")

print(f"  {'Residual':<12s} {np.var(residual):12.2f} "
      f"{np.var(residual)/total_var*100:11.1f}% {np.sqrt(np.mean(residual**2)):12.2f}")
print()

# Cross-terms (covariance between filters)
print("  Cross-term covariances (showing filters contribute correlated energy):")
cross_total = 0.0
for i, spec_i in enumerate(FILTER_SPECS):
    for j, spec_j in enumerate(FILTER_SPECS):
        if j <= i:
            continue
        li, lj = spec_i["label"], spec_j["label"]
        cov = np.mean(filter_outputs[li] * filter_outputs[lj])
        if abs(cov) > 0.1:
            cross_total += 2 * cov
            print(f"    Cov({li},{lj}) = {cov:.2f}")
print(f"  Total cross-term contribution: {cross_total:.2f} "
      f"({cross_total/total_var*100:.1f}%)")
print()

# ============================================================================
# ENVELOPE MODULATION ANALYSIS
# ============================================================================

print("=" * 70)
print("ENVELOPE MODULATION ANALYSIS (Hilbert Transform)")
print("=" * 70)
print()

bp_labels = ["2", "3", "4", "5", "6"]

for label in bp_labels:
    env = envelopes[label]
    spec = [s for s in FILTER_SPECS if s["label"] == label][0]
    fc = (spec["f2"] + spec["f3"]) / 2
    carrier_period_yr = TWOPI / fc

    # Remove DC from envelope before spectral analysis
    env_centered = env - np.mean(env)

    # Compute Lanczos spectrum of the envelope
    # dataspacing=1 (weekly), datapointsperyr=52
    w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(
        env_centered, 1, 52
    )

    # Convert to rad/year
    omega_yr = w * 52

    # Find dominant modulation frequency (skip DC region, look for peaks)
    # Skip very low frequencies (< 0.1 rad/yr) to avoid DC leakage
    mask = omega_yr > 0.2
    omega_masked = omega_yr[mask]
    amp_masked = amp[mask]

    # Find top 3 peaks
    from scipy.signal import find_peaks
    peaks, props = find_peaks(amp_masked, distance=3)

    if len(peaks) > 0:
        # Sort by amplitude
        sorted_idx = np.argsort(amp_masked[peaks])[::-1]
        top_peaks = peaks[sorted_idx[:5]]

        print(f"  BP-{label} (carrier T={carrier_period_yr:.2f} yr, "
              f"fc={fc:.2f} rad/yr)")
        print(f"    Envelope mean: {np.mean(env):.2f}, "
              f"std: {np.std(env):.2f}, "
              f"modulation depth: {np.std(env)/np.mean(env)*100:.1f}%")
        print(f"    Top modulation frequencies:")
        for rank, pk in enumerate(top_peaks):
            om = omega_masked[pk]
            period_yr = TWOPI / om if om > 0 else float('inf')
            a = amp_masked[pk]
            print(f"      #{rank+1}: omega={om:.3f} rad/yr  "
                  f"T={period_yr:.2f} yr  amp={a:.2f}")
        # Dominant
        dom_omega = omega_masked[top_peaks[0]]
        dom_period = TWOPI / dom_omega
        print(f"    --> Dominant modulation period: {dom_period:.2f} years")
    else:
        print(f"  BP-{label}: No clear modulation peaks found")
    print()

# ============================================================================
# CROSS-CORRELATION OF ENVELOPES
# ============================================================================

print("=" * 70)
print("ENVELOPE CROSS-CORRELATION (zero-lag, normalized)")
print("=" * 70)
print()

# Normalize envelopes (subtract mean, divide by std)
env_normed = {}
for label in bp_labels:
    e = envelopes[label]
    env_normed[label] = (e - np.mean(e)) / np.std(e)

# Compute cross-correlation matrix
print(f"  {'':>8s}", end="")
for label in bp_labels:
    print(f"  BP-{label:>4s}", end="")
print()
print(f"  {'':>8s}", end="")
for label in bp_labels:
    print(f"  {'----':>7s}", end="")
print()

for i, li in enumerate(bp_labels):
    print(f"  BP-{li:>4s}", end="")
    for j, lj in enumerate(bp_labels):
        corr = np.mean(env_normed[li] * env_normed[lj])
        print(f"  {corr:7.3f}", end="")
    print()
print()

# Highlight strong correlations
print("  Strong envelope correlations (|r| > 0.3):")
found_strong = False
for i, li in enumerate(bp_labels):
    for j, lj in enumerate(bp_labels):
        if j <= i:
            continue
        corr = np.mean(env_normed[li] * env_normed[lj])
        if abs(corr) > 0.3:
            found_strong = True
            print(f"    BP-{li} vs BP-{lj}: r = {corr:.3f}")
if not found_strong:
    print("    (none found)")
print()

# ============================================================================
# PREDICTABILITY ANALYSIS
# ============================================================================

print("=" * 70)
print("PREDICTABILITY ANALYSIS: 96% energy vs 23% predictable")
print("=" * 70)
print()

# The "96% energy" means filters capture 96% of total variance
# But predictability requires the cycles to be stationary/regular

# For each BP filter, compute:
# 1. Total variance explained
# 2. "Predictable" variance = variance of a pure sinusoid at the carrier freq
#    vs the actual variance (which includes amplitude modulation)

print("  For each BP filter, how stable is the amplitude?")
print(f"  {'Filter':<10s} {'Mean Amp':>10s} {'Std Amp':>10s} {'CV (%)':>10s} "
      f"{'Min/Max':>10s}")
print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for label in bp_labels:
    env = envelopes[label]
    mean_a = np.mean(env)
    std_a = np.std(env)
    cv = std_a / mean_a * 100
    ratio = np.min(env) / np.max(env)
    print(f"  BP-{label:<7s} {mean_a:10.2f} {std_a:10.2f} {cv:9.1f}% {ratio:10.3f}")

print()
print("  CV = Coefficient of Variation (std/mean * 100)")
print("  High CV means amplitude varies a lot -> less predictable")
print("  Min/Max ratio close to 0 means envelope goes nearly to zero -> beating")
print()

# Compute fraction of time each filter has "useful" amplitude
# (above some threshold of mean)
print("  Fraction of time envelope > 50% of mean (active cycles):")
for label in bp_labels:
    env = envelopes[label]
    threshold = 0.5 * np.mean(env)
    frac_active = np.mean(env > threshold) * 100
    print(f"    BP-{label}: {frac_active:.1f}% of time active")

print()
print("=" * 70)
print("DONE")
print("=" * 70)
