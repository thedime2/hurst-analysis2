# -*- coding: utf-8 -*-
"""
Walk-Forward Backtest: Adaptive Parameter Optimization

Instead of fixed thresholds, this script:
  1. Splits data into rolling train/test windows
  2. On each train window, optimizes synchronicity thresholds via grid search
  3. Applies optimized params to the test window (no lookahead)
  4. Also re-estimates coupling coefficients per window to detect regime changes
  5. Compares walk-forward vs fixed-parameter performance

Key insight from validate_coupling_modern.py:
  - F2 asymmetry is STABLE and strengthening (1.53 -> 2.2-3.6)
  - F4->F2 leading indicator REVERSED on modern data (r=+0.37 -> -0.17)
  - F3/F6 amplification at F2 troughs REVERSED (1.3 -> 0.8)
  => Strategies 2 and 4 need adaptive weighting; Strategies 1,3,5 are robust

Reference: prd/trading_methodology.md, Phase C
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Reuse the full backtest machinery
from backtest_trading_methodology import (
    FS, TWOPI, FILTER_SPECS, NOMINAL_FREQS, ASYMMETRY_RATIOS,
    load_weekly, apply_filter, compute_state_vector,
    strategy_buy_hold, strategy_synchronicity, strategy_integrated,
    evaluate_strategy,
)
from scipy.stats import pearsonr


def estimate_coupling(state, start_idx, end_idx):
    """
    Estimate coupling coefficients on a window of the state vector.
    Returns dict of key coefficients that strategies depend on.
    """
    sl = slice(start_idx, end_idx)
    n = end_idx - start_idx

    # F4->F2 envelope growth correlation (Strategy 2 dependency)
    f4_ec = state['f4_env_change'][sl]
    f2_env = state['envelopes'][1][sl]
    # F2 envelope change (13-week)
    f2_ec = np.zeros(n)
    for t in range(13, n):
        if state['envelopes'][1][start_idx + t - 13] > 1e-10:
            f2_ec[t] = (f2_env[t] - state['envelopes'][1][start_idx + t - 13]) / \
                        state['envelopes'][1][start_idx + t - 13]

    valid = (np.abs(f4_ec) < 10) & (np.abs(f2_ec) < 10) & (np.arange(n) > 52)
    if np.sum(valid) > 100:
        r_f4_f2, _ = pearsonr(f4_ec[valid], f2_ec[valid])
    else:
        r_f4_f2 = 0.0

    # F3/F6 amplification at F2 troughs (Strategy 4 dependency)
    f2_phase = state['phases'][1][sl]
    trough_mask = (f2_phase < np.pi / 3) | (f2_phase > 5 * np.pi / 3)
    peak_mask = (f2_phase > 2 * np.pi / 3) & (f2_phase < 4 * np.pi / 3)

    f3_ratio = np.nan
    f6_ratio = np.nan
    for i, name in [(2, 'F3'), (5, 'F6')]:
        env = state['envelopes'][i][sl]
        amp_t = np.mean(env[trough_mask]) if np.sum(trough_mask) > 10 else np.nan
        amp_p = np.mean(env[peak_mask]) if np.sum(peak_mask) > 10 else np.nan
        ratio = amp_t / amp_p if amp_p > 0 else np.nan
        if name == 'F3':
            f3_ratio = ratio
        else:
            f6_ratio = ratio

    # F2 asymmetry (Strategy 5 dependency)
    f2_sig = state['outputs'][1][sl]
    signs = np.sign(f2_sig)
    zc = np.where(np.diff(signs))[0]
    f2_asym = 1.5  # default
    if len(zc) >= 4:
        up_d, down_d = [], []
        for k in range(len(zc) - 1):
            dur = zc[k + 1] - zc[k]
            mid = (zc[k] + zc[k + 1]) // 2
            if f2_sig[mid] > 0:
                up_d.append(dur)
            else:
                down_d.append(dur)
        if up_d and down_d:
            f2_asym = np.mean(up_d) / np.mean(down_d)

    return {
        'r_f4_f2': r_f4_f2,
        'f3_trough_ratio': f3_ratio,
        'f6_trough_ratio': f6_ratio,
        'f2_asymmetry': f2_asym,
    }


def optimize_sync_thresholds(state, log_returns, start_idx, end_idx):
    """
    Grid search for optimal synchronicity thresholds on a training window.
    Returns (threshold_long, threshold_short) that maximize Sharpe.
    """
    best_sharpe = -999
    best_params = (-0.3, 0.3)
    sw = state['sync_weighted']

    for tl in np.arange(-0.6, -0.05, 0.05):
        for ts in np.arange(0.05, 0.65, 0.05):
            # Build position
            n_full = len(sw)
            position = np.zeros(n_full)
            for t in range(start_idx, end_idx):
                if sw[t] < tl:
                    position[t] = 1.0
                elif sw[t] > ts:
                    position[t] = -1.0
                else:
                    position[t] = 0.0

            # Evaluate on window
            sr = position[start_idx:end_idx - 1] * log_returns[start_idx + 1:end_idx]
            if len(sr) < 52:
                continue
            ann_ret = np.mean(sr) * FS
            ann_vol = np.std(sr) * np.sqrt(FS)
            sharpe = ann_ret / ann_vol if ann_vol > 0.01 else 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (tl, ts)

    return best_params, best_sharpe


def strategy_adaptive_integrated(state, coupling, sync_params):
    """
    Integrated strategy with adaptive weights based on measured coupling.
    Strategies with weak/reversed coupling get downweighted.
    """
    n = len(state['sync_score'])
    position = np.zeros(n)
    tl, ts = sync_params

    # Adaptive weights for each modifier based on coupling strength
    # F4 leading: weight by |r_f4_f2| — if reversed or weak, dampen
    f4_weight = max(0.0, coupling['r_f4_f2'])  # Only use if positive correlation
    f4_weight = min(f4_weight / 0.3, 1.0)  # Normalize: 0.3+ = full weight

    # Phase sync: weight by trough amplification ratio
    f3_amp = coupling['f3_trough_ratio']
    f6_amp = coupling['f6_trough_ratio']
    phase_sync_active = (not np.isnan(f3_amp) and f3_amp > 1.1) or \
                        (not np.isnan(f6_amp) and f6_amp > 1.1)

    # Asymmetry: always active (confirmed stable), but use measured ratio
    asym_ratio = coupling['f2_asymmetry'] if not np.isnan(coupling['f2_asymmetry']) else 1.5

    # Pre-compute F3/F6 envelope relative to median (for phase sync)
    env_rel = {}
    for i in [2, 5]:
        med = np.ones(n)
        for t in range(52, n):
            med[t] = np.median(state['envelopes'][i][max(0, t - 260):t])
        med[:52] = med[52]
        env_rel[i] = state['envelopes'][i] / np.maximum(med, 1e-10)

    for t in range(1, n):
        # Step 1: Synchronicity with optimized thresholds
        sw = state['sync_weighted'][t]
        if sw < tl:
            base = np.clip(-sw / abs(tl), 0, 1)
        elif sw > ts:
            base = np.clip(-sw / ts, -1, 0)
        else:
            base = 0.0

        # Step 2: F4 leading (adaptive weight)
        f4_mod = 1.0
        if f4_weight > 0.1:
            if state['f4_env_change'][t] > 0.5:
                f4_mod = 1.0 + 0.3 * f4_weight
            elif state['f4_env_change'][t] < -0.3:
                f4_mod = 1.0 - 0.3 * f4_weight

        # Step 3: Amplitude regime
        n_active = sum(1 for i in range(1, 6) if state['amp_regime'][i][t] != 'WEAK')
        amp_mod = 0.2 + 0.8 * (n_active / 5.0)

        # Step 4: Phase sync (adaptive — only if confirmed in this era)
        f2z = state['f2_zone'][t]
        phase_sync_mod = 1.0
        if phase_sync_active and f2z == 'trough':
            f3_a = env_rel[2][t] > 1.2
            f6_a = env_rel[5][t] > 1.2
            if f3_a and f6_a:
                phase_sync_mod = 1.4
            elif f3_a or f6_a:
                phase_sync_mod = 1.2
        elif f2z == 'peak':
            phase_sync_mod = 0.9

        # Step 5: Asymmetry (always active, uses measured ratio)
        asym_mod = 1.0
        if f2z in ('trough', 'rising') and base > 0:
            asym_mod = min(asym_ratio / 1.5, 1.5)  # Scale by actual asymmetry
        elif f2z in ('trough', 'rising') and base < 0:
            asym_mod = 0.7
        elif f2z in ('peak', 'falling') and base < 0:
            asym_mod = 1.1
        elif f2z in ('peak', 'falling') and base > 0:
            asym_mod = 0.5

        # Step 6: Band calibration
        n_nominal = sum(1 for i in range(1, 6) if 0.8 <= state['freq_ratio'][i][t] <= 1.2)
        n_slow = sum(1 for i in range(1, 6) if state['freq_ratio'][i][t] > 1.2)
        calib_mod = (0.6 + 0.4 * n_nominal / 5.0) * (1.0 + 0.1 * n_slow)

        position[t] = np.clip(base * f4_mod * amp_mod * phase_sync_mod * asym_mod * calib_mod, -1, 1)

    return position


def run_walkforward(symbol, start, end, label,
                    train_years=15, test_years=5, warmup=520):
    """
    Walk-forward optimization.

    Slides a train_years window forward by test_years increments.
    On each step:
      1. Estimate coupling on train window
      2. Optimize sync thresholds on train window
      3. Apply adaptive strategy on test window
      4. Collect OOS results
    """
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD: {label}")
    print(f"  Train={train_years}yr, Test={test_years}yr")
    print(f"{'='*70}")

    close, dates = load_weekly(symbol, start, end)
    n = len(close)
    total_years = n / FS
    print(f"Data: {n} samples, {total_years:.1f} years")

    log_prices = np.log(close)
    log_returns = np.diff(log_prices)
    log_returns = np.concatenate([[0], log_returns])

    # Compute state vector on full data
    print("Computing state vector...")
    state = compute_state_vector(log_prices)

    train_wk = int(train_years * FS)
    test_wk = int(test_years * FS)

    # Walk-forward loop
    wf_position = np.zeros(n)
    fixed_position = np.zeros(n)
    bh_position = np.ones(n)
    window_results = []

    t_start = warmup  # First possible train start
    window_num = 0

    while t_start + train_wk + test_wk <= n:
        train_end = t_start + train_wk
        test_end = min(train_end + test_wk, n)

        window_num += 1
        train_date_start = pd.Timestamp(dates[t_start])
        train_date_end = pd.Timestamp(dates[train_end - 1])
        test_date_start = pd.Timestamp(dates[train_end])
        test_date_end = pd.Timestamp(dates[test_end - 1])

        # 1. Estimate coupling on train window
        coupling = estimate_coupling(state, t_start, train_end)

        # 2. Optimize sync thresholds on train window
        sync_params, train_sharpe = optimize_sync_thresholds(
            state, log_returns, t_start, train_end
        )

        # 3. Apply adaptive integrated on TEST window
        adaptive_pos = strategy_adaptive_integrated(state, coupling, sync_params)
        wf_position[train_end:test_end] = adaptive_pos[train_end:test_end]

        # Fixed-parameter comparison
        fixed_pos = strategy_synchronicity(state, threshold_long=-0.3, threshold_short=0.3)
        fixed_position[train_end:test_end] = fixed_pos[train_end:test_end]

        # Evaluate test window
        test_ret = log_returns[train_end + 1:test_end]
        wf_ret = wf_position[train_end:test_end - 1] * test_ret
        fix_ret = fixed_position[train_end:test_end - 1] * test_ret
        bh_ret = test_ret

        def quick_sharpe(returns):
            if len(returns) < 10:
                return 0.0
            ar = np.mean(returns) * FS
            av = np.std(returns) * np.sqrt(FS)
            return ar / av if av > 0.01 else 0.0

        wr = {
            'window': window_num,
            'train': f"{train_date_start.strftime('%Y')}-{train_date_end.strftime('%Y')}",
            'test': f"{test_date_start.strftime('%Y')}-{test_date_end.strftime('%Y')}",
            'sync_tl': sync_params[0],
            'sync_ts': sync_params[1],
            'train_sharpe': train_sharpe,
            'r_f4_f2': coupling['r_f4_f2'],
            'f3_ratio': coupling['f3_trough_ratio'],
            'f6_ratio': coupling['f6_trough_ratio'],
            'f2_asym': coupling['f2_asymmetry'],
            'wf_sharpe': quick_sharpe(wf_ret),
            'fix_sharpe': quick_sharpe(fix_ret),
            'bh_sharpe': quick_sharpe(bh_ret),
        }
        window_results.append(wr)

        print(f"  W{window_num}: train={wr['train']} test={wr['test']} "
              f"sync=[{sync_params[0]:.2f},{sync_params[1]:.2f}] "
              f"r_f4f2={coupling['r_f4_f2']:+.2f} "
              f"WF={wr['wf_sharpe']:.2f} Fix={wr['fix_sharpe']:.2f} BH={wr['bh_sharpe']:.2f}")

        t_start += test_wk

    # Overall performance
    print(f"\n--- Walk-Forward Summary ({len(window_results)} windows) ---")
    wf_sharpes = [w['wf_sharpe'] for w in window_results]
    fix_sharpes = [w['fix_sharpe'] for w in window_results]
    bh_sharpes = [w['bh_sharpe'] for w in window_results]

    print(f"  Walk-Forward avg Sharpe: {np.mean(wf_sharpes):.3f} (std={np.std(wf_sharpes):.3f})")
    print(f"  Fixed-Param  avg Sharpe: {np.mean(fix_sharpes):.3f} (std={np.std(fix_sharpes):.3f})")
    print(f"  Buy & Hold   avg Sharpe: {np.mean(bh_sharpes):.3f} (std={np.std(bh_sharpes):.3f})")

    # Coupling evolution
    print(f"\n--- Coupling Evolution ---")
    print(f"  {'Window':<8} {'Test':<12} {'r_F4F2':>8} {'F3@trgh':>8} {'F6@trgh':>8} {'F2asym':>8}")
    for w in window_results:
        f3r = f"{w['f3_ratio']:.2f}" if not np.isnan(w['f3_ratio']) else "N/A"
        f6r = f"{w['f6_ratio']:.2f}" if not np.isnan(w['f6_ratio']) else "N/A"
        print(f"  W{w['window']:<7} {w['test']:<12} {w['r_f4_f2']:>+8.3f} "
              f"{f3r:>8} {f6r:>8} {w['f2_asym']:>8.2f}")

    return window_results, wf_position, fixed_position, dates, log_prices, state


def plot_walkforward(window_results, wf_pos, fix_pos, dates, log_prices,
                     state, label, save_path, warmup=520):
    """Plot walk-forward results."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    fig.suptitle(f'Walk-Forward Backtest: {label}', fontsize=14, fontweight='bold')

    n = len(log_prices)
    log_returns = np.diff(log_prices)
    log_returns = np.concatenate([[0], log_returns])

    # Find first non-zero position (start of WF period)
    wf_start = warmup
    for t in range(warmup, n):
        if abs(wf_pos[t]) > 0.01:
            wf_start = t
            break

    # 1. Cumulative returns comparison
    ax = axes[0]
    wf_ret = wf_pos[wf_start:-1] * log_returns[wf_start + 1:]
    fix_ret = fix_pos[wf_start:-1] * log_returns[wf_start + 1:]
    bh_ret = log_returns[wf_start + 1:]

    cum_wf = np.cumsum(wf_ret)
    cum_fix = np.cumsum(fix_ret)
    cum_bh = np.cumsum(bh_ret)
    t_dates = dates[wf_start + 1:wf_start + 1 + len(cum_wf)]

    ax.plot(t_dates, cum_bh, 'gray', linewidth=1.5, label='Buy & Hold')
    ax.plot(t_dates, cum_fix, '#1f77b4', linewidth=1.5, alpha=0.8,
            label='Fixed Sync')
    ax.plot(t_dates, cum_wf, '#d62728', linewidth=2.5,
            label='Walk-Forward Adaptive')
    ax.set_ylabel('Cumulative Log Return')
    ax.set_title('Walk-Forward vs Fixed Parameters')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Per-window Sharpe comparison
    ax = axes[1]
    windows = [w['window'] for w in window_results]
    wf_s = [w['wf_sharpe'] for w in window_results]
    fix_s = [w['fix_sharpe'] for w in window_results]
    bh_s = [w['bh_sharpe'] for w in window_results]
    x = np.arange(len(windows))
    w = 0.25
    ax.bar(x - w, bh_s, w, label='Buy & Hold', color='gray', alpha=0.7)
    ax.bar(x, fix_s, w, label='Fixed Sync', color='#1f77b4', alpha=0.7)
    ax.bar(x + w, wf_s, w, label='Walk-Forward', color='#d62728', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([w['test'] for w in window_results], rotation=45, fontsize=8)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Per-Window Out-of-Sample Sharpe')
    ax.legend(fontsize=9)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # 3. Coupling coefficient evolution
    ax = axes[2]
    r_f4 = [w['r_f4_f2'] for w in window_results]
    f3_r = [w['f3_ratio'] if not np.isnan(w['f3_ratio']) else 0 for w in window_results]
    f6_r = [w['f6_ratio'] if not np.isnan(w['f6_ratio']) else 0 for w in window_results]
    f2_a = [w['f2_asym'] for w in window_results]

    ax.plot(x, r_f4, 'o-', label='r(F4->F2)', color='#ff7f0e')
    ax.plot(x, f3_r, 's-', label='F3 trough ratio', color='#2ca02c')
    ax.plot(x, f6_r, '^-', label='F6 trough ratio', color='#9467bd')
    ax.axhline(1.0, color='k', linewidth=0.5, linestyle='--')
    ax.axhline(0.0, color='k', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([w['test'] for w in window_results], rotation=45, fontsize=8)
    ax.set_ylabel('Coefficient')
    ax.set_title('Coupling Coefficients Over Time (estimated on train window)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Optimized threshold evolution
    ax = axes[3]
    tl = [w['sync_tl'] for w in window_results]
    ts = [w['sync_ts'] for w in window_results]
    ax2 = ax.twinx()
    ax.plot(x, tl, 'v-', label='Threshold Long', color='green', markersize=8)
    ax.plot(x, ts, '^-', label='Threshold Short', color='red', markersize=8)
    ax2.plot(x, f2_a, 'D-', label='F2 Asymmetry', color='#8c564b', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([w['test'] for w in window_results], rotation=45, fontsize=8)
    ax.set_ylabel('Sync Threshold')
    ax2.set_ylabel('F2 Asymmetry Ratio')
    ax.set_title('Optimized Parameters & Asymmetry Over Time')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.close()


# ---------- main ----------
if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)

    datasets = [
        # Long DJIA history — most windows
        {'symbol': 'djia', 'start': '1921-04-29', 'end': '2005-12-31',
         'label': 'DJIA 1921-2005 (Full Walk-Forward)'},
        # SPX modern
        {'symbol': 'spx', 'start': '1950-01-01', 'end': '2026-01-31',
         'label': 'SPX 1950-2025 (Walk-Forward)'},
    ]

    for ds in datasets:
        wr, wf_pos, fix_pos, dates, log_prices, state = run_walkforward(
            ds['symbol'], ds['start'], ds['end'], ds['label'],
            train_years=15, test_years=5
        )

        fig_name = f"fig_walkforward_{ds['symbol']}.png"
        fig_path = os.path.join(base_dir, fig_name)
        plot_walkforward(wr, wf_pos, fix_pos, dates, log_prices, state,
                         ds['label'], fig_path)

    print("\n" + "=" * 70)
    print("WALK-FORWARD COMPLETE")
    print("=" * 70)
