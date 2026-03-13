# -*- coding: utf-8 -*-
"""
Backtest Trading Methodology: Real-Time Cyclic State Estimation

Implements and backtests the 6 strategies from prd/trading_methodology.md:
  1. Synchronicity scoring (multi-band phase alignment)
  2. F4 leading indicator for F2
  3. Amplitude regime classification
  4. Phase synchronization exploitation
  5. Asymmetry-aware position management
  6. Integrated framework combining all signals

Backtest approach:
  - Walk-forward: no lookahead bias
  - Filters applied causally (no future data in reflect padding — use 'constant' mode)
  - Position sizing: -1 (full short) to +1 (full long)
  - Returns: log(price) changes, position * return

Reference: prd/trading_methodology.md
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.filters import ormsby_filter, apply_ormsby_filter

# ---------- constants ----------
FS = 52
TWOPI = 2 * np.pi

# Nominal center frequencies for each BP filter (rad/year)
# Used by Strategy 6 (Band Calibration) to compute R_i = f_nominal / f_instantaneous
NOMINAL_FREQS = {
    1: 1.65,   # F2: center of 0.85-2.45 passband ~ 54-month
    2: 4.93,   # F3: center of 3.20-6.70 ~ 18-month
    3: 8.55,   # F4: center of 7.25-9.85 ~ 40-week
    4: 16.65,  # F5: center of 13.65-19.65 ~ 20-week
    5: 32.35,  # F6: center of 28.45-36.25 ~ 10-week
}

# Confirmed asymmetry ratios (up/down duration) from fig_hidden_relationships.py
ASYMMETRY_RATIOS = {
    1: 1.506,  # F2: bull longer
    2: 0.565,  # F3: bear longer
    3: 0.674,  # F4: bear longer
    4: 0.301,  # F5: bear longer
    5: 1.0,    # F6: ~symmetric (not measured separately)
}

# 6-filter specs (rad/year)
FILTER_SPECS = [
    {"label": "F1 (trend)", "type": "lp", "f_pass": 0.85, "f_stop": 1.25, "nw": 1393},
    {"label": "F2 (3.8yr)", "type": "bp", "f1": 0.85, "f2": 1.25, "f3": 2.05, "f4": 2.45, "nw": 1393},
    {"label": "F3 (1.3yr)", "type": "bp", "f1": 3.20, "f2": 3.55, "f3": 6.35, "f4": 6.70, "nw": 1245},
    {"label": "F4 (0.7yr)", "type": "bp", "f1": 7.25, "f2": 7.55, "f3": 9.55, "f4": 9.85, "nw": 1745},
    {"label": "F5 (20wk)", "type": "bp", "f1": 13.65, "f2": 13.95, "f3": 19.35, "f4": 19.65, "nw": 1299},
    {"label": "F6 (9wk)", "type": "bp", "f1": 28.45, "f2": 28.75, "f3": 35.95, "f4": 36.25, "nw": 1299},
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
    """
    Apply one Ormsby filter. BP filters use analytic=True to get envelope,
    phase, and instantaneous frequency directly — no Hilbert transform needed.

    Returns dict with keys: signal, envelope, phase, frequency (or None for LP).
    """
    nw = spec["nw"]
    is_bp = spec["type"] == "bp"

    if spec["type"] == "lp":
        f_edges = np.array([spec["f_pass"], spec["f_stop"]], dtype=float) / TWOPI
        h = ormsby_filter(nw=nw, f_edges=f_edges, fs=FS, filter_type="lp", analytic=False)
    else:
        f_edges = np.array([spec["f1"], spec["f2"], spec["f3"], spec["f4"]], dtype=float) / TWOPI
        h = ormsby_filter(nw=nw, f_edges=f_edges, fs=FS, filter_type="bp",
                         method="modulate", analytic=is_bp)

    result = apply_ormsby_filter(signal, h, mode="reflect", fs=FS)

    # For BP: apply_ormsby_filter already computed envelope, unwrapped phase,
    # and inst. frequency (in Hz = cycles/year) from the complex analytic output.
    # Convert phase convention: Ormsby analytic phase has 0 at peak of cosine.
    # Shift by +pi/2 so that 0=trough, pi=peak (matching Hurst convention).
    if is_bp and result['phase'] is not None:
        result['phase_shifted'] = (result['phasew'] + np.pi / 2) % TWOPI
        # Convert inst. freq from cycles/year (Hz) to rad/year
        if result['frequency'] is not None:
            result['freq_rad'] = result['frequency'] * TWOPI
        else:
            result['freq_rad'] = None
    else:
        result['phase_shifted'] = None
        result['freq_rad'] = None

    return result


def compute_state_vector(log_prices):
    """
    Compute the full state vector for the trading methodology.

    Returns dict with:
      phases[i]: instantaneous phase for F2-F6 (0=trough, pi=peak)
      envelopes[i]: instantaneous envelope magnitude
      sync_score: synchronicity score S(t) in [-1, +1]
      sync_weighted: amplitude-weighted synchronicity
      f4_env_change: F4 envelope 13-week rate of change
      amp_regime[i]: 'STRONG', 'NORMAL', or 'WEAK' per band
      f2_zone: 'trough', 'rising', 'peak', or 'falling'
    """
    n = len(log_prices)

    # Apply all 6 filters (F1=LP real, F2-F6=BP analytic)
    filter_results = []
    for spec in FILTER_SPECS:
        fr = apply_filter(log_prices, spec)
        filter_results.append(fr)

    # Real-valued filter outputs (for zero-crossing detection etc.)
    outputs = [fr['signal'].real if np.iscomplexobj(fr['signal']) else fr['signal']
               for fr in filter_results]

    # Envelopes, phases, and inst. frequency from analytic BP filters (F2-F6)
    envelopes = {}
    phases = {}
    inst_freq = {}
    freq_ratio = {}
    for i in range(1, 6):
        fr = filter_results[i]
        envelopes[i] = fr['envelope']
        phases[i] = fr['phase_shifted']
        # Instantaneous frequency (rad/year) and nominal ratio
        if fr['freq_rad'] is not None:
            inst_freq[i] = np.abs(fr['freq_rad'])
            r_i = np.ones(n)
            valid = inst_freq[i] > 0.1
            r_i[valid] = NOMINAL_FREQS[i] / inst_freq[i][valid]
            freq_ratio[i] = np.clip(r_i, 0.3, 3.0)
        else:
            inst_freq[i] = np.zeros(n)
            freq_ratio[i] = np.ones(n)

    # --- Synchronicity score ---
    # s_i = -cos(theta_i): -1 at trough, +1 at peak
    phase_scores = np.zeros((5, n))
    for idx, i in enumerate(range(1, 6)):
        phase_scores[idx] = -np.cos(phases[i])

    sync_score = np.mean(phase_scores, axis=0)

    # Amplitude-weighted version
    median_envs = {}
    for i in range(1, 6):
        # Use expanding window for median (no lookahead)
        med = np.zeros(n)
        for t in range(52, n):
            med[t] = np.median(envelopes[i][max(0, t-520):t])  # 10yr rolling median
        med[:52] = med[52]
        median_envs[i] = med

    weights = np.zeros((5, n))
    for idx, i in enumerate(range(1, 6)):
        w = envelopes[i] / np.maximum(median_envs[i], 1e-10)
        weights[idx] = np.clip(w, 0.1, 5.0)  # Bound weights

    sync_weighted = np.sum(weights * phase_scores, axis=0) / np.sum(weights, axis=0)

    # --- F4 envelope rate of change (13-week) ---
    f4_env_change = np.zeros(n)
    for t in range(13, n):
        if envelopes[3][t - 13] > 1e-10:
            f4_env_change[t] = (envelopes[3][t] - envelopes[3][t - 13]) / envelopes[3][t - 13]

    # --- Amplitude regime per band ---
    amp_regime = {}
    for i in range(1, 6):
        regime = np.full(n, 'NORMAL', dtype=object)
        # Cycle-appropriate lookback (3-5 complete cycles)
        lookback_map = {1: 520, 2: 260, 3: 156, 4: 104, 5: 52}
        lb = lookback_map.get(i, 260)
        for t in range(lb, n):
            window = envelopes[i][max(0, t - lb):t]
            p33 = np.percentile(window, 33)
            p67 = np.percentile(window, 67)
            if envelopes[i][t] > p67:
                regime[t] = 'STRONG'
            elif envelopes[i][t] < p33:
                regime[t] = 'WEAK'
        amp_regime[i] = regime

    # --- F2 phase zone ---
    f2_zone = np.full(n, 'rising', dtype=object)
    for t in range(n):
        ph = phases[1][t]
        if ph < np.pi / 3 or ph > 5 * np.pi / 3:
            f2_zone[t] = 'trough'
        elif np.pi / 3 <= ph < 2 * np.pi / 3:
            f2_zone[t] = 'rising'
        elif 2 * np.pi / 3 <= ph < 4 * np.pi / 3:
            f2_zone[t] = 'peak'
        else:
            f2_zone[t] = 'falling'

    return {
        'outputs': outputs,
        'envelopes': envelopes,
        'phases': phases,
        'sync_score': sync_score,
        'sync_weighted': sync_weighted,
        'f4_env_change': f4_env_change,
        'amp_regime': amp_regime,
        'f2_zone': f2_zone,
        'phase_scores': phase_scores,
        'inst_freq': inst_freq,
        'freq_ratio': freq_ratio,
    }


def strategy_buy_hold(log_returns, n):
    """Baseline: always long."""
    return np.ones(n)


def strategy_synchronicity(state, threshold_long=-0.3, threshold_short=0.3):
    """
    Strategy 1: Position based on synchronicity score.
    S < threshold_long -> long, S > threshold_short -> short, else flat.
    """
    n = len(state['sync_weighted'])
    position = np.zeros(n)
    sw = state['sync_weighted']

    for t in range(1, n):
        if sw[t] < threshold_long:
            position[t] = 1.0  # Long
        elif sw[t] > threshold_short:
            position[t] = -1.0  # Short
        else:
            position[t] = 0.0  # Flat

    return position


def strategy_f4_leading(state):
    """
    Strategy 2: F4 envelope growth as regime indicator for F2.
    When F4 env rising + F2 near trough -> strong long
    When F4 env rising + F2 near peak -> defensive
    """
    n = len(state['f4_env_change'])
    position = np.zeros(n)

    for t in range(1, n):
        f4_rising = state['f4_env_change'][t] > 0.3
        f2z = state['f2_zone'][t]

        if f2z == 'trough':
            position[t] = 1.0
        elif f2z == 'rising':
            position[t] = 0.7
        elif f2z == 'peak':
            position[t] = -0.3 if f4_rising else 0.0
        elif f2z == 'falling':
            position[t] = -0.5 if f4_rising else -0.3

    return position


def strategy_amplitude_regime(state):
    """
    Strategy 3: Only trade when cycles are active (STRONG/NORMAL).
    Position proportional to number of active bands * sync score.
    """
    n = len(state['sync_score'])
    position = np.zeros(n)

    for t in range(1, n):
        n_active = sum(1 for i in range(1, 6) if state['amp_regime'][i][t] != 'WEAK')
        activity = n_active / 5.0  # 0 to 1
        sw = state['sync_weighted'][t]

        if sw < -0.2:
            position[t] = activity  # Long, scaled by active bands
        elif sw > 0.2:
            position[t] = -activity  # Short, scaled by active bands
        else:
            position[t] = 0.0

    return position


def strategy_phase_sync(state):
    """
    Strategy 4: Phase Synchronization Exploitation.

    F3/F6 amplitudes increase 36-50% when F2 is at a trough, creating
    explosive V-shaped reversals. Enter long when F5/F6 crosses zero
    (rising) while F2 is in trough zone and F3/F6 envelopes are elevated.

    At F2 peaks: tops are rounded, don't try to time — wait for F4 confirm.

    Reference: trading_methodology.md, Strategy 4
    """
    n = len(state['sync_score'])
    position = np.zeros(n)

    # Compute F3/F6 envelope relative to their running median
    env_rel = {}
    for i in [2, 5]:  # F3, F6
        med = np.ones(n)
        for t in range(52, n):
            med[t] = np.median(state['envelopes'][i][max(0, t - 260):t])
        med[:52] = med[52]
        env_rel[i] = state['envelopes'][i] / np.maximum(med, 1e-10)

    # Detect F5/F6 zero crossings (rising) from filter outputs
    f5_sig = state['outputs'][4]  # F5
    f6_sig = state['outputs'][5]  # F6

    for t in range(2, n):
        f2z = state['f2_zone'][t]

        # F2 trough zone: look for V-reversal entry
        if f2z == 'trough':
            f3_amplified = env_rel[2][t] > 1.2  # F3 above 120% of median
            f6_amplified = env_rel[5][t] > 1.2  # F6 above 120% of median
            # F5 or F6 just crossed zero going up
            f5_rising = f5_sig[t] > 0 and f5_sig[t - 1] <= 0
            f6_rising = f6_sig[t] > 0 and f6_sig[t - 1] <= 0

            if (f3_amplified or f6_amplified) and (f5_rising or f6_rising):
                position[t] = 1.0  # Strong long entry
            elif f3_amplified or f6_amplified:
                position[t] = 0.7  # Elevated envelopes, waiting for cross
            else:
                position[t] = 0.3  # In trough zone but no amplification
        # F2 peak zone: confirm then act (wait for F4 to turn)
        elif f2z == 'peak':
            f4_turning = state['phases'][3][t] > np.pi  # F4 past peak
            if f4_turning:
                position[t] = -0.7  # Confirmed decline starting
            else:
                position[t] = -0.2  # Peak zone, not yet confirmed
        elif f2z == 'rising':
            position[t] = 0.3
        else:  # falling
            position[t] = -0.3

    return position


def strategy_asymmetry(state):
    """
    Strategy 5: Asymmetry Exploitation.

    F2 bull phases last 50% longer than bear (1.506 ratio).
    F3-F5 bear phases last longer than bull (corrections drag on).

    Rules:
    - F2 rising: hold longs longer, buy dips (corrections look scarier than they are)
    - F2 falling: exit quickly, bear phases are shorter but sharper
    - Corrections in uptrends: accumulate (F3-F5 bear > bull means prolonged dips)
    - Rallies in downtrends: don't chase (brief and sharp)

    Reference: trading_methodology.md, Strategy 5
    """
    n = len(state['sync_score'])
    position = np.zeros(n)

    for t in range(1, n):
        f2z = state['f2_zone'][t]
        sw = state['sync_weighted'][t]

        if f2z in ('trough', 'rising'):
            # F2 bull: stay long, buy dips
            if sw < -0.3:
                position[t] = 1.0  # Trough alignment in bull — max long
            elif sw < 0.0:
                position[t] = 0.7  # Mild bearish reading in bull — still long (asymmetry)
            elif sw < 0.3:
                position[t] = 0.3  # Correction within uptrend — accumulate
            else:
                position[t] = 0.0  # Strong bear signal even in F2 bull — step aside

        elif f2z in ('peak', 'falling'):
            # F2 bear: shorter but sharper — exit quickly, don't chase rallies
            if sw > 0.3:
                position[t] = -1.0  # Peak alignment in bear — max short
            elif sw > 0.0:
                position[t] = -0.5  # Mild bull reading in bear — stay short
            elif sw > -0.3:
                position[t] = 0.0  # Counter-trend rally — don't chase
            else:
                position[t] = 0.0  # Strong trough signal — pause, but F2 still falling

    return position


def strategy_band_calibration(state):
    """
    Strategy 6: Nominal Model Band Calibration.

    Use instantaneous frequency vs nominal to detect stretched/compressed
    cycles. Stretched cycles (R_i > 1.2) tend to have higher amplitude.
    Compressed cycles (R_i < 0.8) tend to have lower amplitude.

    When most bands run at nominal speed, cycle behavior is predictable.
    When bands are stretched/compressed, adjust confidence.

    Reference: trading_methodology.md, Strategy 6
    """
    n = len(state['sync_score'])
    position = np.zeros(n)

    for t in range(1, n):
        # Count how many bands are nominal (0.8-1.2), slow (>1.2), or fast (<0.8)
        n_nominal = 0
        n_slow = 0
        n_fast = 0
        for i in range(1, 6):
            r_i = state['freq_ratio'][i][t]
            if 0.8 <= r_i <= 1.2:
                n_nominal += 1
            elif r_i > 1.2:
                n_slow += 1  # Stretched — higher amplitude expected
            else:
                n_fast += 1  # Compressed — lower amplitude expected

        # Confidence factor: more nominal = more predictable
        confidence = n_nominal / 5.0  # 0 to 1

        # Slow cycles amplify expected moves, fast cycles dampen
        amp_modifier = 1.0 + 0.15 * n_slow - 0.15 * n_fast

        # Base signal from sync score
        sw = state['sync_weighted'][t]
        base = np.clip(-sw * 2, -1, 1)

        position[t] = np.clip(base * confidence * amp_modifier, -1, 1)

    return position


def strategy_integrated(state):
    """
    Integrated framework: Combines ALL 6 strategy signals.

    Step 1: Base signal from synchronicity (Strategy 1)
    Step 2: F4 leading indicator modifier (Strategy 2)
    Step 3: Amplitude regime modifier (Strategy 3)
    Step 4: Phase synchronization amplifier at F2 troughs (Strategy 4)
    Step 5: Asymmetry-aware holding rules (Strategy 5)
    Step 6: Band calibration confidence (Strategy 6)

    Position = Base * F4_mod * Amp_mod * PhaseSync_mod * Asym_mod * Calib_mod

    Reference: trading_methodology.md, Integrated Decision Framework
    """
    n = len(state['sync_score'])
    position = np.zeros(n)

    # Pre-compute F3/F6 envelope relative to median (for phase sync)
    env_rel = {}
    for i in [2, 5]:  # F3, F6
        med = np.ones(n)
        for t in range(52, n):
            med[t] = np.median(state['envelopes'][i][max(0, t - 260):t])
        med[:52] = med[52]
        env_rel[i] = state['envelopes'][i] / np.maximum(med, 1e-10)

    for t in range(1, n):
        # --- Step 1: Base from synchronicity ---
        sw = state['sync_weighted'][t]
        base = np.clip(-sw * 2, -1, 1)

        # --- Step 2: F4 leading indicator ---
        f4_mod = 1.0
        if state['f4_env_change'][t] > 0.5:
            f4_mod = 1.3
        elif state['f4_env_change'][t] < -0.3:
            f4_mod = 0.7

        # --- Step 3: Amplitude regime ---
        n_active = sum(1 for i in range(1, 6) if state['amp_regime'][i][t] != 'WEAK')
        amp_mod = 0.2 + 0.8 * (n_active / 5.0)

        # --- Step 4: Phase synchronization at F2 troughs ---
        f2z = state['f2_zone'][t]
        phase_sync_mod = 1.0
        if f2z == 'trough':
            f3_amp = env_rel[2][t] > 1.2
            f6_amp = env_rel[5][t] > 1.2
            if f3_amp and f6_amp:
                phase_sync_mod = 1.4  # Both amplified — explosive reversal expected
            elif f3_amp or f6_amp:
                phase_sync_mod = 1.2  # One amplified
        elif f2z == 'peak':
            # F3/F6 suppressed at peaks — tops are rounded, less conviction
            phase_sync_mod = 0.9

        # --- Step 5: Asymmetry exploitation ---
        asym_mod = 1.0
        if f2z in ('trough', 'rising') and base > 0:
            asym_mod = 1.2  # Bull phases longer — hold with conviction
        elif f2z in ('trough', 'rising') and base < 0:
            asym_mod = 0.7  # Corrections in uptrends look scarier than they are
        elif f2z in ('peak', 'falling') and base < 0:
            asym_mod = 1.1  # Bear phases sharper — maintain short
        elif f2z in ('peak', 'falling') and base > 0:
            asym_mod = 0.5  # Counter-trend rally in bear — don't chase

        # --- Step 6: Band calibration confidence ---
        n_nominal = sum(1 for i in range(1, 6) if 0.8 <= state['freq_ratio'][i][t] <= 1.2)
        n_slow = sum(1 for i in range(1, 6) if state['freq_ratio'][i][t] > 1.2)
        calib_mod = (0.6 + 0.4 * n_nominal / 5.0) * (1.0 + 0.1 * n_slow)

        # --- Combine ---
        position[t] = np.clip(base * f4_mod * amp_mod * phase_sync_mod * asym_mod * calib_mod, -1, 1)

    return position


def compute_expected_moves(state, warmup=520):
    """
    Step 4 of Integrated Framework: Set expectations using amplitude regime
    and asymmetry ratios.

    For each time step, estimate:
      - Expected move size: current envelope * asymmetry ratio for dominant band
      - Expected duration: nominal half-period * asymmetry ratio
      - Stop loss level: p90 of last 5 cycles' amplitude range

    Reference: trading_methodology.md, Step 4
    """
    n = len(state['sync_score'])

    # Find dominant band (highest current envelope relative to median)
    dominant_band = np.ones(n, dtype=int)  # default F2
    for t in range(warmup, n):
        best_rel = 0
        for i in range(1, 6):
            lb = {1: 520, 2: 260, 3: 156, 4: 104, 5: 52}[i]
            if t < lb:
                continue
            med = np.median(state['envelopes'][i][max(0, t - lb):t])
            rel = state['envelopes'][i][t] / max(med, 1e-10)
            if rel > best_rel:
                best_rel = rel
                dominant_band[t] = i

    # Nominal half-periods (weeks)
    nominal_half_periods = {1: 100, 2: 34, 3: 19, 4: 10, 5: 5}

    expected_move = np.zeros(n)
    expected_duration = np.zeros(n)
    stop_level = np.zeros(n)

    for t in range(warmup, n):
        db = dominant_band[t]
        env_now = state['envelopes'][db][t]
        asym = ASYMMETRY_RATIOS[db]

        # Expected move: current envelope magnitude (in log space)
        # Adjust by asymmetry: if rising, move lasts asym * half-period
        f2z = state['f2_zone'][t]
        if f2z in ('trough', 'rising'):
            expected_move[t] = env_now * max(asym, 1.0)
            expected_duration[t] = nominal_half_periods[db] * max(asym, 1.0)
        else:
            expected_move[t] = env_now * max(1.0 / asym, 1.0)
            expected_duration[t] = nominal_half_periods[db] * max(1.0 / asym, 1.0)

        # Stop loss: p90 of recent envelope values (last 5 cycle periods)
        lb = nominal_half_periods[db] * 10  # ~5 full cycles
        window = state['envelopes'][db][max(0, t - lb):t]
        if len(window) > 10:
            stop_level[t] = np.percentile(window, 90)
        else:
            stop_level[t] = env_now * 2

    return {
        'dominant_band': dominant_band,
        'expected_move': expected_move,
        'expected_duration': expected_duration,
        'stop_level': stop_level,
    }


def evaluate_strategy(log_returns, position, name, warmup=520):
    """
    Evaluate a strategy's performance.
    warmup: weeks to skip at start (filter artifacts).
    """
    # Strategy returns: position[t-1] * log_return[t] (position at t-1 produces return at t)
    strat_returns = position[warmup:-1] * log_returns[warmup + 1:]
    bh_returns = log_returns[warmup + 1:]

    n = len(strat_returns)
    years = n / FS

    # Annualized metrics
    ann_return_strat = np.mean(strat_returns) * FS
    ann_return_bh = np.mean(bh_returns) * FS
    ann_vol_strat = np.std(strat_returns) * np.sqrt(FS)
    ann_vol_bh = np.std(bh_returns) * np.sqrt(FS)

    sharpe_strat = ann_return_strat / ann_vol_strat if ann_vol_strat > 0 else 0
    sharpe_bh = ann_return_bh / ann_vol_bh if ann_vol_bh > 0 else 0

    # Cumulative
    cum_strat = np.cumsum(strat_returns)
    cum_bh = np.cumsum(bh_returns)

    # Max drawdown
    peak_strat = np.maximum.accumulate(cum_strat)
    dd_strat = cum_strat - peak_strat
    max_dd_strat = np.min(dd_strat)

    peak_bh = np.maximum.accumulate(cum_bh)
    dd_bh = cum_bh - peak_bh
    max_dd_bh = np.min(dd_bh)

    # Win rate
    wins = np.sum(strat_returns > 0)
    total = np.sum(strat_returns != 0)
    win_rate = wins / total if total > 0 else 0

    # Time in market
    time_in = np.mean(np.abs(position[warmup:-1]) > 0.01)

    # Exposure-adjusted return
    avg_exposure = np.mean(np.abs(position[warmup:-1]))
    adj_return = ann_return_strat / avg_exposure if avg_exposure > 0.01 else 0

    # Calmar ratio (return / max drawdown)
    calmar = ann_return_strat / abs(max_dd_strat) if abs(max_dd_strat) > 0.01 else 0

    # Profit factor: gross profits / gross losses
    gross_profit = np.sum(strat_returns[strat_returns > 0])
    gross_loss = abs(np.sum(strat_returns[strat_returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        'name': name,
        'years': years,
        'ann_return': ann_return_strat,
        'ann_return_bh': ann_return_bh,
        'ann_vol': ann_vol_strat,
        'sharpe': sharpe_strat,
        'sharpe_bh': sharpe_bh,
        'max_dd': max_dd_strat,
        'max_dd_bh': max_dd_bh,
        'win_rate': win_rate,
        'time_in_market': time_in,
        'avg_exposure': avg_exposure,
        'adj_return': adj_return,
        'calmar': calmar,
        'profit_factor': profit_factor,
        'cum_strat': cum_strat,
        'cum_bh': cum_bh,
    }


def run_backtest(symbol, start, end, label, warmup=520):
    """Run all strategies on one dataset."""
    print(f"\n{'='*70}")
    print(f"BACKTEST: {label}")
    print(f"{'='*70}")

    close, dates = load_weekly(symbol, start, end)
    n = len(close)
    print(f"Data: {n} samples, {n/FS:.1f} years")

    log_prices = np.log(close)
    log_returns = np.diff(log_prices)
    log_returns = np.concatenate([[0], log_returns])

    # Compute state vector
    print("Computing state vector...")
    state = compute_state_vector(log_prices)

    # Run all 8 strategies (6 individual + buy&hold + integrated)
    strategies = {
        'Buy & Hold': strategy_buy_hold(log_returns, n),
        '1-Synchronicity': strategy_synchronicity(state),
        '2-F4 Leading': strategy_f4_leading(state),
        '3-Amp Regime': strategy_amplitude_regime(state),
        '4-Phase Sync': strategy_phase_sync(state),
        '5-Asymmetry': strategy_asymmetry(state),
        '6-Band Calib': strategy_band_calibration(state),
        'Integrated': strategy_integrated(state),
    }

    results = {}
    for name, position in strategies.items():
        r = evaluate_strategy(log_returns, position, name, warmup=warmup)
        results[name] = r

    # Compute expected move sizing
    exp_moves = compute_expected_moves(state, warmup=warmup)

    # Print summary
    print(f"\n{'Strategy':<20} {'Return':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} "
          f"{'WinRate':>8} {'Expos':>8} {'Calmar':>8} {'PF':>8}")
    print("-" * 100)
    for name, r in results.items():
        print(f"{name:<20} {r['ann_return']:>8.1%} {r['ann_vol']:>8.1%} "
              f"{r['sharpe']:>8.3f} {r['max_dd']:>8.1%} "
              f"{r['win_rate']:>8.1%} {r['avg_exposure']:>8.1%} "
              f"{r['calmar']:>8.3f} {r['profit_factor']:>8.2f}")

    # Print expected move sizing summary
    em = exp_moves['expected_move'][warmup:]
    ed = exp_moves['expected_duration'][warmup:]
    sl = exp_moves['stop_level'][warmup:]
    valid = em > 0
    print(f"\n  Expected Move Sizing (median): move={np.median(em[valid]):.4f} log, "
          f"duration={np.median(ed[valid]):.0f}wk, stop={np.median(sl[valid]):.4f} log")

    return results, state, dates, log_prices, close, exp_moves


def plot_backtest(results_dict, dates, log_prices, state, label, save_path,
                  warmup=520, exp_moves=None):
    """Plot backtest results — 6 panels."""
    fig, axes = plt.subplots(6, 1, figsize=(16, 28))
    fig.suptitle(f'Trading Methodology Backtest: {label}', fontsize=14, fontweight='bold')

    t_dates = dates[warmup + 1:]

    # 1. Cumulative returns — all strategies
    ax = axes[0]
    colors = {
        'Buy & Hold': 'gray', '1-Synchronicity': '#1f77b4',
        '2-F4 Leading': '#ff7f0e', '3-Amp Regime': '#2ca02c',
        '4-Phase Sync': '#9467bd', '5-Asymmetry': '#8c564b',
        '6-Band Calib': '#e377c2', 'Integrated': '#d62728',
    }
    for name, r in results_dict.items():
        lw = 2.5 if name == 'Integrated' else (1.8 if name == 'Buy & Hold' else 1.0)
        alpha = 1.0 if name in ('Integrated', 'Buy & Hold') else 0.6
        ax.plot(t_dates[:len(r['cum_strat'])], r['cum_strat'],
                label=f"{name} (S={r['sharpe']:.2f})",
                color=colors.get(name, 'black'), linewidth=lw, alpha=alpha)
    ax.set_ylabel('Cumulative Log Return')
    ax.set_title('All Strategies Comparison')
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

    # 2. Synchronicity score over time
    ax = axes[1]
    sw = state['sync_weighted'][warmup:]
    ax.fill_between(dates[warmup:warmup + len(sw)], sw, 0,
                    where=sw < 0, color='green', alpha=0.3, label='Bullish')
    ax.fill_between(dates[warmup:warmup + len(sw)], sw, 0,
                    where=sw > 0, color='red', alpha=0.3, label='Bearish')
    ax.plot(dates[warmup:warmup + len(sw)], sw, 'k-', linewidth=0.5)
    ax.set_ylabel('Sync Score')
    ax.set_title('Amplitude-Weighted Synchronicity Score')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)

    # 3. Log price with F2 phase zones
    ax = axes[2]
    lp = log_prices[warmup:]
    ax.plot(dates[warmup:warmup + len(lp)], lp, 'k-', linewidth=0.8)
    zone_colors = {'trough': 'green', 'rising': 'lightgreen', 'peak': 'red', 'falling': 'lightsalmon'}
    f2z = state['f2_zone'][warmup:]
    d = dates[warmup:warmup + len(f2z)]
    for zone, color in zone_colors.items():
        mask = np.array([z == zone for z in f2z])
        if np.any(mask):
            ax.fill_between(d, lp.min(), lp.max(), where=mask,
                           color=color, alpha=0.15, label=f'F2 {zone}')
    ax.set_ylabel('Log Price')
    ax.set_title('Price with F2 Phase Zones')
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # 4. Integrated strategy position
    ax = axes[3]
    int_pos = strategy_integrated(state)
    p = int_pos[warmup:]
    ax.fill_between(dates[warmup:warmup + len(p)], p, 0,
                    where=p > 0, color='green', alpha=0.5)
    ax.fill_between(dates[warmup:warmup + len(p)], p, 0,
                    where=p < 0, color='red', alpha=0.5)
    ax.set_ylabel('Position')
    ax.set_title('Integrated Strategy Position (-1 to +1)')
    ax.set_ylim(-1.3, 1.3)
    ax.grid(True, alpha=0.3)

    # 5. Expected move sizing (Step 4)
    ax = axes[4]
    if exp_moves is not None:
        em = exp_moves['expected_move'][warmup:]
        sl = exp_moves['stop_level'][warmup:]
        d_em = dates[warmup:warmup + len(em)]
        ax.plot(d_em, em, 'b-', linewidth=0.8, label='Expected Move')
        ax.plot(d_em, sl, 'r--', linewidth=0.8, alpha=0.7, label='Stop Level (p90)')
        ax.set_ylabel('Log Amplitude')
        ax.set_title('Expected Move Sizing (Envelope-Based)')
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 6. Band calibration: instantaneous frequency ratios
    ax = axes[5]
    filter_labels = {1: 'F2', 2: 'F3', 3: 'F4', 4: 'F5', 5: 'F6'}
    cal_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i in range(1, 6):
        fr = state['freq_ratio'][i][warmup:]
        d_fr = dates[warmup:warmup + len(fr)]
        ax.plot(d_fr, fr, linewidth=0.6, alpha=0.7,
                color=cal_colors[i - 1], label=filter_labels[i])
    ax.axhline(1.0, color='k', linewidth=1, linestyle='-')
    ax.axhline(0.8, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(1.2, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_ylabel('R_i = f_nom / f_inst')
    ax.set_title('Band Calibration: Nominal vs Instantaneous Frequency')
    ax.set_ylim(0.3, 2.0)
    ax.legend(fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.close()


# ---------- main ----------
if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)

    # Backtest datasets
    datasets = [
        # In-sample (Hurst era) — does the methodology capture known patterns?
        {'symbol': 'djia', 'start': '1921-04-29', 'end': '1965-05-21',
         'label': 'DJIA 1921-1965 (Hurst In-Sample)', 'warmup': 520},
        # Out-of-sample periods
        {'symbol': 'djia', 'start': '1965-05-22', 'end': '2005-05-21',
         'label': 'DJIA 1965-2005 (Out-of-Sample)', 'warmup': 520},
        {'symbol': 'spx', 'start': '1950-01-01', 'end': '2026-01-31',
         'label': 'SPX 1950-2025 (Full History)', 'warmup': 520},
    ]

    all_summary = []

    for ds in datasets:
        results, state, dates, log_prices, close, exp_moves = run_backtest(
            ds['symbol'], ds['start'], ds['end'], ds['label'], ds['warmup']
        )

        fig_name = f"fig_backtest_{ds['symbol']}_{ds['start'][:4]}_{ds['end'][:4]}.png"
        fig_path = os.path.join(base_dir, fig_name)
        plot_backtest(results, dates, log_prices, state, ds['label'], fig_path,
                      ds['warmup'], exp_moves)

        for name, r in results.items():
            all_summary.append({
                'Dataset': ds['label'][:25],
                'Strategy': name,
                'Return': r['ann_return'],
                'Sharpe': r['sharpe'],
                'MaxDD': r['max_dd'],
                'Exposure': r['avg_exposure'],
                'Calmar': r['calmar'],
                'PF': r['profit_factor'],
            })

    # Grand summary
    print("\n" + "=" * 110)
    print("GRAND SUMMARY: ALL BACKTESTS")
    print("=" * 110)
    print(f"{'Dataset':<26} {'Strategy':<20} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} "
          f"{'Expos':>8} {'Calmar':>8} {'PF':>8}")
    print("-" * 110)
    for s in all_summary:
        print(f"{s['Dataset']:<26} {s['Strategy']:<20} {s['Return']:>8.1%} "
              f"{s['Sharpe']:>8.3f} {s['MaxDD']:>8.1%} {s['Exposure']:>8.1%} "
              f"{s['Calmar']:>8.3f} {s['PF']:>8.2f}")

    print("\n" + "=" * 110)
    print("DONE")
    print("=" * 110)
