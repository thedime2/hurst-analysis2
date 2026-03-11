# -*- coding: utf-8 -*-
"""
Validate Inter-Group Coupling Coefficients Across Multiple Eras

Tests whether the coupling relationships discovered in the 1935-1954 display
window hold on modern data. This is the critical gap identified in
prd/trading_methodology.md.

Analyses:
  1. Envelope cross-correlation matrix (all 5 filter pairs)
  2. Phase synchronization (F3/F6 amplification at F2 troughs)
  3. Amplitude growth transmission (F4 -> F2 leading indicator)
  4. Cycle asymmetry (bull/bear duration ratios)

Datasets tested:
  - DJIA 1921-1965 (Hurst era, baseline)
  - DJIA 1965-2005 (Post-Hurst)
  - DJIA 1985-2025 (Modern)
  - SPX  1985-2025 (Cross-market)

Reference: experiments/phase7_unified/fig_hidden_relationships.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.stats import pearsonr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.filters import ormsby_filter, apply_ormsby_filter

# ---------- constants ----------
FS = 52
TWOPI = 2 * np.pi

# 6-filter specs (rad/year) — same as fig_hidden_relationships.py
FILTER_SPECS = [
    {"label": "F1 (trend)", "type": "lp", "f_pass": 0.85, "f_stop": 1.25, "nw": 1393},
    {"label": "F2 (3.8yr)", "type": "bp", "f1": 0.85, "f2": 1.25, "f3": 2.05, "f4": 2.45, "nw": 1393},
    {"label": "F3 (1.3yr)", "type": "bp", "f1": 3.20, "f2": 3.55, "f3": 6.35, "f4": 6.70, "nw": 1245},
    {"label": "F4 (0.7yr)", "type": "bp", "f1": 7.25, "f2": 7.55, "f3": 9.55, "f4": 9.85, "nw": 1745},
    {"label": "F5 (20wk)", "type": "bp", "f1": 13.65, "f2": 13.95, "f3": 19.35, "f4": 19.65, "nw": 1299},
    {"label": "F6 (9wk)", "type": "bp", "f1": 28.45, "f2": 28.75, "f3": 35.95, "f4": 36.25, "nw": 1299},
]

# Datasets to test
DATASETS = [
    {'label': 'DJIA 1921-1965 (Hurst)', 'symbol': 'djia', 'start': '1921-04-29', 'end': '1965-05-21'},
    {'label': 'DJIA 1965-2005', 'symbol': 'djia', 'start': '1965-05-22', 'end': '2005-05-21'},
    {'label': 'DJIA 1985-2025', 'symbol': 'djia', 'start': '1985-01-01', 'end': '2026-01-31'},
    {'label': 'SPX 1985-2025', 'symbol': 'spx', 'start': '1985-01-01', 'end': '2026-01-31'},
]


def load_weekly(symbol, start, end):
    """Load weekly price data."""
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    fname = '^dji_w.csv' if symbol == 'djia' else '^spx_w.csv'
    df = pd.read_csv(os.path.join(base, fname))
    df['Date'] = pd.to_datetime(df['Date'])
    mask = df['Date'].between(start, end)
    df = df[mask].sort_values('Date').reset_index(drop=True)
    return df['Close'].values, df['Date'].values


def apply_filter(signal, spec):
    """Apply one Ormsby filter."""
    nw = spec["nw"]
    if spec["type"] == "lp":
        f_edges = np.array([spec["f_pass"], spec["f_stop"]], dtype=float) / TWOPI
        h = ormsby_filter(nw=nw, f_edges=f_edges, fs=FS, filter_type="lp", analytic=False)
    else:
        f_edges = np.array([spec["f1"], spec["f2"], spec["f3"], spec["f4"]], dtype=float) / TWOPI
        h = ormsby_filter(nw=nw, f_edges=f_edges, fs=FS, filter_type="bp",
                         method="modulate", analytic=False)
    result = apply_ormsby_filter(signal, h, mode="reflect", fs=FS)
    return result["signal"].astype(float)


def get_envelope_and_phase(bp_signal):
    """Compute Hilbert envelope and instantaneous phase."""
    analytic = hilbert(bp_signal)
    envelope = np.abs(analytic)
    phase = np.angle(analytic)
    # Shift so 0=trough, pi=peak
    phase_shifted = (phase + np.pi / 2) % TWOPI
    return envelope, phase_shifted


def run_coupling_analysis(close, dates, label, trim_frac=0.15):
    """
    Run all 4 coupling analyses on one dataset.
    trim_frac: fraction of data to trim from edges (filter startup artifacts).
    """
    n = len(close)
    log_prices = np.log(close)

    # Apply all 6 filters
    outputs = []
    for spec in FILTER_SPECS:
        sig = apply_filter(log_prices, spec)
        outputs.append(sig)

    # Envelopes and phases for bandpass filters (F2-F6 = indices 1-5)
    envelopes = [None]  # placeholder for F1
    phases = [None]
    for i in range(1, 6):
        env, ph = get_envelope_and_phase(outputs[i])
        envelopes.append(env)
        phases.append(ph)

    # Trim edges to avoid filter artifacts
    si = int(n * trim_frac)
    ei = int(n * (1 - trim_frac))

    results = {
        'label': label,
        'n_samples': ei - si,
        'years': (ei - si) / FS,
    }

    # --- Analysis 1: Envelope correlation matrix ---
    corr_matrix = np.zeros((5, 5))
    pval_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            env_i = envelopes[i + 1][si:ei]
            env_j = envelopes[j + 1][si:ei]
            r, p = pearsonr(env_i, env_j)
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p
    results['corr_matrix'] = corr_matrix
    results['pval_matrix'] = pval_matrix

    # Key pairs
    results['r_F2_F3'] = corr_matrix[0, 1]
    results['r_F2_F4'] = corr_matrix[0, 2]
    results['r_F2_F6'] = corr_matrix[0, 4]
    results['r_F3_F6'] = corr_matrix[1, 4]

    # --- Analysis 2: Phase synchronization (F3/F6 at F2 troughs) ---
    long_ph = phases[1][si:ei]  # F2 phase
    trough_mask = (long_ph < np.pi / 3) | (long_ph > 5 * np.pi / 3)
    peak_mask = (long_ph > 2 * np.pi / 3) & (long_ph < 4 * np.pi / 3)

    phase_sync = {}
    for j, fname in [(2, 'F3'), (3, 'F4'), (4, 'F5'), (5, 'F6')]:
        short_env = envelopes[j][si:ei]
        amp_trough = np.mean(short_env[trough_mask]) if np.sum(trough_mask) > 10 else np.nan
        amp_peak = np.mean(short_env[peak_mask]) if np.sum(peak_mask) > 10 else np.nan
        ratio = amp_trough / amp_peak if amp_peak > 0 else np.nan
        phase_sync[fname] = {
            'amp_at_trough': amp_trough,
            'amp_at_peak': amp_peak,
            'ratio': ratio,
            'amplified': ratio > 1.1 if not np.isnan(ratio) else None,
        }
    results['phase_sync'] = phase_sync

    # --- Analysis 3: Amplitude growth transmission (F4 -> F2) ---
    window = 52  # 1-year rolling
    env_changes = {}
    for i in range(1, 6):
        ec = np.zeros(n)
        for t in range(window, n):
            if envelopes[i][t - window] > 1e-10:
                ec[t] = (envelopes[i][t] - envelopes[i][t - window]) / envelopes[i][t - window]
        env_changes[i] = ec

    # Search for best lag between all pairs
    lead_lag = {}
    pairs_to_test = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
    filter_names = {1: 'F2', 2: 'F3', 3: 'F4', 4: 'F5', 5: 'F6'}

    for (a, b) in pairs_to_test:
        ec_a = env_changes[a][si:ei]
        ec_b = env_changes[b][si:ei]
        best_lag = 0
        best_r = 0
        for lag in range(-26, 27, 2):
            if lag > 0:
                x = ec_a[lag:]
                y = ec_b[:-lag]
            elif lag < 0:
                x = ec_a[:lag]
                y = ec_b[-lag:]
            else:
                x, y = ec_a, ec_b
            mn = min(len(x), len(y))
            if mn > 100:
                valid = (np.abs(x[:mn]) < 10) & (np.abs(y[:mn]) < 10)
                if np.sum(valid) > 50:
                    r_lag, _ = pearsonr(x[:mn][valid], y[:mn][valid])
                    if abs(r_lag) > abs(best_r):
                        best_r = r_lag
                        best_lag = lag
        pair_name = f'{filter_names[a]}->{filter_names[b]}'
        lead_lag[pair_name] = {'lag_wk': best_lag, 'r': best_r}

    results['lead_lag'] = lead_lag

    # --- Analysis 4: Cycle asymmetry (bull/bear duration) ---
    asymmetry = {}
    for i in range(1, 6):
        sig = outputs[i][si:ei]
        # Find zero crossings
        signs = np.sign(sig)
        zc = np.where(np.diff(signs))[0]
        if len(zc) < 4:
            asymmetry[filter_names[i]] = {'up_down_ratio': np.nan, 'n_cycles': 0}
            continue

        up_durations = []
        down_durations = []
        for k in range(len(zc) - 1):
            duration = zc[k + 1] - zc[k]
            # Check if this half-cycle is up or down
            mid = (zc[k] + zc[k + 1]) // 2
            if sig[mid] > 0:
                up_durations.append(duration)
            else:
                down_durations.append(duration)

        mean_up = np.mean(up_durations) if up_durations else np.nan
        mean_down = np.mean(down_durations) if down_durations else np.nan
        ratio = mean_up / mean_down if mean_down > 0 else np.nan

        asymmetry[filter_names[i]] = {
            'mean_up_wk': mean_up,
            'mean_down_wk': mean_down,
            'up_down_ratio': ratio,
            'n_cycles': len(up_durations) + len(down_durations),
        }
    results['asymmetry'] = asymmetry

    return results


def print_comparison(all_results):
    """Print comparative tables across all datasets."""
    print("=" * 80)
    print("COUPLING COEFFICIENT VALIDATION ACROSS ERAS")
    print("=" * 80)

    # --- Envelope correlations ---
    print("\n--- 1. Envelope Correlations (key pairs) ---")
    print(f"{'Dataset':<30} {'F2-F3':>8} {'F2-F4':>8} {'F2-F6':>8} {'F3-F6':>8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['label']:<30} {r['r_F2_F3']:>8.3f} {r['r_F2_F4']:>8.3f} "
              f"{r['r_F2_F6']:>8.3f} {r['r_F3_F6']:>8.3f}")

    # --- Phase synchronization ---
    print("\n--- 2. Phase Synchronization (amplitude ratio at F2 trough vs peak) ---")
    print(f"{'Dataset':<30} {'F3':>8} {'F4':>8} {'F5':>8} {'F6':>8}")
    print("-" * 70)
    for r in all_results:
        ps = r['phase_sync']
        vals = [ps[f]['ratio'] for f in ['F3', 'F4', 'F5', 'F6']]
        print(f"{r['label']:<30}", end="")
        for v in vals:
            if np.isnan(v):
                print(f"{'N/A':>8}", end="")
            else:
                marker = '*' if v > 1.1 else (' ' if v > 0.9 else '-')
                print(f"{v:>7.2f}{marker}", end="")
        print()
    print("  * = amplified at F2 trough (>1.1), - = suppressed (<0.9)")

    # --- Leading indicator ---
    print("\n--- 3. F4 -> F2 Leading Indicator ---")
    print(f"{'Dataset':<30} {'Lag(wk)':>8} {'r':>8}")
    print("-" * 50)
    for r in all_results:
        ll = r['lead_lag'].get('F2->F4', {'lag_wk': 0, 'r': 0})
        print(f"{r['label']:<30} {ll['lag_wk']:>8d} {ll['r']:>8.3f}")

    # Top 3 coupled pairs per era
    print("\n--- 3b. Top Coupled Pairs Per Era ---")
    for r in all_results:
        pairs = sorted(r['lead_lag'].items(), key=lambda x: abs(x[1]['r']), reverse=True)
        print(f"\n  {r['label']}:")
        for name, val in pairs[:5]:
            marker = " ***" if abs(val['r']) > 0.3 else ""
            print(f"    {name:<12} lag={val['lag_wk']:>4d}wk  r={val['r']:>+.3f}{marker}")

    # --- Asymmetry ---
    print("\n--- 4. Cycle Asymmetry (up/down duration ratio) ---")
    print(f"{'Dataset':<30} {'F2':>8} {'F3':>8} {'F4':>8} {'F5':>8} {'F6':>8}")
    print("-" * 80)
    for r in all_results:
        a = r['asymmetry']
        print(f"{r['label']:<30}", end="")
        for f in ['F2', 'F3', 'F4', 'F5', 'F6']:
            v = a[f]['up_down_ratio']
            if np.isnan(v):
                print(f"{'N/A':>8}", end="")
            else:
                print(f"{v:>8.3f}", end="")
        print()
    print("  F2>1 = bull longer, F3-F5<1 = bear longer (expected)")

    # --- Stability assessment ---
    print("\n" + "=" * 80)
    print("STABILITY ASSESSMENT")
    print("=" * 80)

    # Check which findings are stable across eras
    hurst = all_results[0]
    for r in all_results[1:]:
        print(f"\n  {r['label']} vs Hurst era:")

        # Correlation stability
        for pair, key in [('F2-F3', 'r_F2_F3'), ('F2-F6', 'r_F2_F6'), ('F3-F6', 'r_F3_F6')]:
            h_val = hurst[key]
            m_val = r[key]
            stable = abs(m_val - h_val) < 0.2
            print(f"    {pair}: {h_val:.3f} -> {m_val:.3f}  {'STABLE' if stable else 'CHANGED'}")

        # Phase sync stability
        for f in ['F3', 'F6']:
            h_ratio = hurst['phase_sync'][f]['ratio']
            m_ratio = r['phase_sync'][f]['ratio']
            if not np.isnan(h_ratio) and not np.isnan(m_ratio):
                both_amp = h_ratio > 1.1 and m_ratio > 1.1
                print(f"    {f}@F2trough: {h_ratio:.2f} -> {m_ratio:.2f}  "
                      f"{'STABLE' if both_amp else 'CHANGED'}")

        # Asymmetry stability
        h_f2 = hurst['asymmetry']['F2']['up_down_ratio']
        m_f2 = r['asymmetry']['F2']['up_down_ratio']
        if not np.isnan(h_f2) and not np.isnan(m_f2):
            both_bull = h_f2 > 1.0 and m_f2 > 1.0
            print(f"    F2 bull>bear: {h_f2:.3f} -> {m_f2:.3f}  "
                  f"{'STABLE' if both_bull else 'CHANGED'}")


def plot_comparison(all_results, save_path):
    """Create comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Coupling Coefficient Stability Across Eras', fontsize=14, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels_short = [r['label'].split('(')[0].strip() if '(' in r['label']
                    else r['label'][:15] for r in all_results]

    # 1. Envelope correlations bar chart
    ax = axes[0, 0]
    pairs = ['F2-F3', 'F2-F4', 'F2-F6', 'F3-F6']
    keys = ['r_F2_F3', 'r_F2_F4', 'r_F2_F6', 'r_F3_F6']
    x = np.arange(len(pairs))
    width = 0.18
    for i, r in enumerate(all_results):
        vals = [r[k] for k in keys]
        ax.bar(x + i * width, vals, width, label=labels_short[i], color=colors[i], alpha=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(pairs)
    ax.set_ylabel('Pearson r')
    ax.set_title('Envelope Correlations')
    ax.legend(fontsize=8)
    ax.axhline(0, color='k', linewidth=0.5)

    # 2. Phase synchronization
    ax = axes[0, 1]
    filters = ['F3', 'F4', 'F5', 'F6']
    x = np.arange(len(filters))
    for i, r in enumerate(all_results):
        vals = [r['phase_sync'][f]['ratio'] for f in filters]
        ax.bar(x + i * width, vals, width, label=labels_short[i], color=colors[i], alpha=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(filters)
    ax.set_ylabel('Trough/Peak Amplitude Ratio')
    ax.set_title('Phase Sync: Amplitude at F2 Trough vs Peak')
    ax.axhline(1.0, color='k', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=8)

    # 3. Asymmetry ratios
    ax = axes[1, 0]
    filters = ['F2', 'F3', 'F4', 'F5', 'F6']
    x = np.arange(len(filters))
    for i, r in enumerate(all_results):
        vals = [r['asymmetry'][f]['up_down_ratio'] for f in filters]
        ax.bar(x + i * width, vals, width, label=labels_short[i], color=colors[i], alpha=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(filters)
    ax.set_ylabel('Up/Down Duration Ratio')
    ax.set_title('Cycle Asymmetry')
    ax.axhline(1.0, color='k', linewidth=0.5, linestyle='--')
    ax.legend(fontsize=8)

    # 4. Full correlation matrices (Hurst vs Modern)
    ax = axes[1, 1]
    # Show difference: Modern - Hurst
    diff = all_results[2]['corr_matrix'] - all_results[0]['corr_matrix']
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='equal')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    labels_filt = ['F2', 'F3', 'F4', 'F5', 'F6']
    ax.set_xticklabels(labels_filt)
    ax.set_yticklabels(labels_filt)
    ax.set_title('Correlation Change: DJIA Modern - Hurst')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{diff[i,j]:+.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    plt.close()


# ---------- main ----------
if __name__ == '__main__':
    all_results = []

    for ds in DATASETS:
        print(f"\nProcessing {ds['label']}...")
        close, dates = load_weekly(ds['symbol'], ds['start'], ds['end'])
        print(f"  {len(close)} samples, {len(close)/FS:.1f} years")

        if len(close) < 1000:
            print(f"  SKIP: too few samples")
            continue

        results = run_coupling_analysis(close, dates, ds['label'])
        all_results.append(results)

    print_comparison(all_results)

    fig_path = os.path.join(os.path.dirname(__file__), 'fig_coupling_validation.png')
    plot_comparison(all_results, fig_path)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
