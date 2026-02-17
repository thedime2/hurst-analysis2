# -*- coding: utf-8 -*-
"""
Page 152: 6-Filter Projection -- codex_v2
=========================================

Upgrades over baseline projection:
  1) Wavelet mode switch: Complex Morlet (CMW) or generalized Morse.
  2) Demodulated state-space projection per BP filter:
     - track log-envelope dynamics
     - track frequency offset dynamics (carrier removed)
  3) Beat-aware two-tone fallback when narrowband beating is detected.
  4) Diagnostics per filter: beat depth, frequency drift, confidence.

Note on FFT boundary handling:
  - This script intentionally DOES NOT modify FFT padding internals.
  - Existing CMW/Morse paths keep current FFT-then-trim behavior.
  - Reflect-mode convolution is available in Ormsby filtering code and is
    reported separately for parity checks.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import medfilt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.loaders import getStooq
from src.time_frequency.cmw import apply_cmw
from src.time_frequency.morse import apply_morse


# ============================================================================
# Configuration
# ============================================================================

FS = 52
TWOPI = 2 * np.pi

DISPLAY_START = "1935-01-01"
DISPLAY_END = "1954-02-01"
PROJECTION_WEEKS = 100

# Wavelet mode: "cmw" or "morse"
WAVELET_MODE = "cmw"
MORSE_GAMMA = 3.0

# Demodulated state model parameters
LOOKBACK_CYCLES = 8
IF_MEDIAN_KERNEL = 9
LOW_ENV_QUANTILE = 0.20
RHO_LOGA = 0.985
RHO_DW = 0.96

# Beat / two-tone selection
ENABLE_TWO_TONE = True
BEAT_DEPTH_MIN = 0.22
SECONDARY_RATIO_MIN = 0.35
MIN_TWO_TONE_SEPARATION = 0.06  # rad/yr offset separation
TWO_TONE_CONF_FLOOR = 0.45

DEFAULT_BP_MODEL_PARAMS = {
    "rho_loga": RHO_LOGA,
    "rho_dw": RHO_DW,
    "lookback_cycles": LOOKBACK_CYCLES,
    "beat_depth_min": BEAT_DEPTH_MIN,
    "secondary_ratio_min": SECONDARY_RATIO_MIN,
    "two_tone_conf_floor": TWO_TONE_CONF_FLOOR,
    "force_model": "auto",  # auto | state_demod | two_tone
}

# Guardrail to avoid unstable forced two-tone choices on BP-5.
BP5_FORCE_STATE_IF_WEAK_TWO_TONE = True
BP5_TWO_TONE_MIN_MEDIAN_R2 = 0.05

FILTER_SPECS = [
    {"type": "lp", "f_pass": 0.85, "f_stop": 1.25, "label": "LP-1: Trend (>5 yr)", "color": "#1f77b4"},
    {"type": "bp", "f1": 0.85, "f2": 1.25, "f3": 2.05, "f4": 2.45, "label": "BP-2: ~3.8 yr", "color": "#ff7f0e"},
    {"type": "bp", "f1": 3.20, "f2": 3.55, "f3": 6.35, "f4": 6.70, "label": "BP-3: ~1.3 yr", "color": "#2ca02c"},
    {"type": "bp", "f1": 7.25, "f2": 7.55, "f3": 9.55, "f4": 9.85, "label": "BP-4: ~0.7 yr", "color": "#d62728"},
    {"type": "bp", "f1": 13.65, "f2": 13.95, "f3": 19.35, "f4": 19.65, "label": "BP-5: ~0.4 yr", "color": "#9467bd"},
    {"type": "bp", "f1": 28.45, "f2": 28.75, "f3": 35.95, "f4": 36.25, "label": "BP-6: ~0.2 yr", "color": "#8c564b"},
]


# ============================================================================
# Utilities
# ============================================================================

def spec_to_wavelet_params(spec: dict) -> tuple[float, float]:
    if spec["type"] == "lp":
        return 0.0, spec["f_pass"] + spec["f_stop"]
    f0 = (spec["f2"] + spec["f3"]) / 2.0
    lower = (spec["f1"] + spec["f2"]) / 2.0
    upper = (spec["f3"] + spec["f4"]) / 2.0
    return f0, upper - lower


def _odd_kernel(kernel: int, n: int) -> int:
    k = int(max(1, kernel))
    if k % 2 == 0:
        k += 1
    if k > n:
        k = n if n % 2 == 1 else max(1, n - 1)
    return max(1, k)


def _interp_nan(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    idx = np.arange(len(x))
    ok = np.isfinite(x)
    if np.all(ok):
        return x
    if not np.any(ok):
        return np.zeros_like(x)
    out = x.copy()
    out[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return out


def _robust_unwrapped_frequency(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    env = np.abs(z)
    phase = np.unwrap(np.angle(z))
    w_raw = np.gradient(phase) * FS  # rad/yr

    thr = float(np.quantile(env, LOW_ENV_QUANTILE)) if len(env) > 8 else float(np.min(env))
    w_masked = w_raw.copy()
    w_masked[env <= thr] = np.nan
    w_filled = _interp_nan(w_masked)

    k = _odd_kernel(IF_MEDIAN_KERNEL, len(w_filled))
    if k > 1:
        w_smooth = medfilt(w_filled, kernel_size=k)
    else:
        w_smooth = w_filled

    return w_smooth, phase, env, thr


def _scalar_kalman(y: np.ndarray, q_frac: float, r_frac: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    valid = y[np.isfinite(y)]
    if len(valid) == 0:
        return np.zeros_like(y)

    var = float(np.var(valid) + 1e-8)
    q = max(1e-8, q_frac * var)
    r = max(1e-8, r_frac * var)

    x = float(np.median(valid))
    p = var
    out = np.zeros_like(y)

    for i, yi in enumerate(y):
        # Predict
        x_pred = x
        p_pred = p + q

        # Update only if observation exists
        if np.isfinite(yi):
            k = p_pred / (p_pred + r)
            x = x_pred + k * (yi - x_pred)
            p = (1.0 - k) * p_pred
        else:
            x = x_pred
            p = p_pred

        out[i] = x

    return out


def _one_tone_confidence(c: np.ndarray, dw_ref: float) -> float:
    n = np.arange(len(c), dtype=float)
    basis = np.exp(1j * dw_ref * n / FS)
    denom = np.vdot(basis, basis)
    if np.abs(denom) < 1e-12:
        return 0.0
    a = np.vdot(basis, c) / denom
    fit = a * basis
    resid = np.linalg.norm(c - fit) / (np.linalg.norm(c) + 1e-12)
    return float(np.clip(1.0 - resid, 0.0, 1.0))


def _fit_two_tone(c: np.ndarray, max_offset: float) -> dict | None:
    n = len(c)
    if n < 32:
        return None

    win = np.hanning(n)
    c0 = (c - np.mean(c)) * win
    C = np.fft.fft(c0)
    f = np.fft.fftfreq(n, d=1.0 / FS) * TWOPI  # rad/yr offsets

    mask = (np.abs(f) > 0.02) & (np.abs(f) <= max_offset)
    if np.sum(mask) < 4:
        return None

    f_sel = f[mask]
    m_sel = np.abs(C[mask])
    rank = np.argsort(m_sel)[::-1]
    if len(rank) < 2:
        return None

    i1 = rank[0]
    f1 = float(f_sel[i1])
    p1 = float(m_sel[i1])

    f2 = None
    p2 = None
    for ii in rank[1:]:
        cand = float(f_sel[ii])
        if abs(cand - f1) >= MIN_TWO_TONE_SEPARATION:
            f2 = cand
            p2 = float(m_sel[ii])
            break

    if f2 is None or p2 is None:
        return None

    n_idx = np.arange(n, dtype=float)
    B = np.column_stack([np.exp(1j * f1 * n_idx / FS), np.exp(1j * f2 * n_idx / FS)])
    coeffs, _, _, _ = np.linalg.lstsq(B, c, rcond=None)
    fit = B @ coeffs

    resid = np.linalg.norm(c - fit) / (np.linalg.norm(c) + 1e-12)
    conf = float(np.clip(1.0 - resid, 0.0, 1.0))
    ratio = float(p2 / max(p1, 1e-12))

    return {
        "dw1": f1,
        "dw2": f2,
        "a1": coeffs[0],
        "a2": coeffs[1],
        "secondary_ratio": ratio,
        "fit_confidence": conf,
        "fit_residual": resid,
    }


def _linear_lp_projection(sig: np.ndarray, idx_end: int, n_forward: int) -> tuple[np.ndarray, float]:
    lookback = min(52, idx_end + 1)
    segment = np.asarray(sig[idx_end - lookback + 1 : idx_end + 1], dtype=float)
    t = np.arange(lookback, dtype=float)
    coeffs = np.polyfit(t, segment, 1)
    t_fwd = np.arange(lookback, lookback + n_forward, dtype=float)
    return np.polyval(coeffs, t_fwd), float(coeffs[0])


def _state_projection(
    sig_complex: np.ndarray,
    idx_end: int,
    n_forward: int,
    w_nom: float,
    n_lookback: int,
    rho_loga: float = RHO_LOGA,
    rho_dw: float = RHO_DW,
) -> dict:
    h_start = max(0, idx_end - n_lookback + 1)
    idx_hist = np.arange(h_start, idx_end + 1, dtype=float)
    z_hist = sig_complex[h_start : idx_end + 1]

    carrier_hist = np.exp(-1j * w_nom * idx_hist / FS)
    c_hist = z_hist * carrier_hist

    dw_hist, phi_c_hist, env_hist, _thr = _robust_unwrapped_frequency(c_hist)
    loga_hist = np.log(np.maximum(env_hist, 1e-10))

    loga_est = _scalar_kalman(loga_hist, q_frac=0.025, r_frac=0.20)
    dw_est = _scalar_kalman(dw_hist, q_frac=0.04, r_frac=0.30)

    tail = min(52, len(loga_est))
    mu_loga = float(np.median(loga_est[-tail:]))
    mu_dw = float(np.median(dw_est[-tail:]))

    # Drift estimate in rad/yr per year.
    if len(dw_est) > 8:
        t_yr = idx_hist / FS
        slope_dw = float(np.polyfit(t_yr, dw_est, 1)[0])
    else:
        slope_dw = 0.0
    slope_dw_step = slope_dw / FS

    beat_p95 = float(np.percentile(env_hist, 95))
    beat_p05 = float(np.percentile(env_hist, 5))
    beat_depth = (beat_p95 - beat_p05) / max(beat_p95 + beat_p05, 1e-12)

    dw_ref = float(np.median(dw_est[-tail:]))
    one_tone_conf = _one_tone_confidence(c_hist, dw_ref=dw_ref)

    # Forward simulation in demodulated domain.
    loga = float(loga_est[-1])
    dw = float(dw_est[-1])
    phi_c = float(phi_c_hist[-1])

    c_fwd = np.zeros(n_forward, dtype=complex)
    env_fwd = np.zeros(n_forward, dtype=float)
    dw_fwd = np.zeros(n_forward, dtype=float)

    for t in range(n_forward):
        loga = mu_loga + rho_loga * (loga - mu_loga)
        dw = mu_dw + rho_dw * (dw - mu_dw) + slope_dw_step
        phi_c = phi_c + dw / FS
        c_t = np.exp(loga + 1j * phi_c)
        c_fwd[t] = c_t
        env_fwd[t] = float(np.abs(c_t))
        dw_fwd[t] = dw

    idx_fwd = np.arange(idx_end + 1, idx_end + 1 + n_forward, dtype=float)
    z_fwd = c_fwd * np.exp(1j * w_nom * idx_fwd / FS)

    return {
        "projection": z_fwd.real,
        "c_hist": c_hist,
        "env_hist": env_hist,
        "dw_hist": dw_est,
        "env_pred": env_fwd,
        "dw_pred": dw_fwd,
        "hist_start": int(h_start),
        "beat_depth": float(beat_depth),
        "freq_drift": float(slope_dw),
        "single_tone_confidence": float(one_tone_conf),
        "method": "state_demod",
    }


def _two_tone_projection(
    sig_complex: np.ndarray,
    idx_end: int,
    n_forward: int,
    w_nom: float,
    n_lookback: int,
    max_offset: float,
) -> dict | None:
    h_start = max(0, idx_end - n_lookback + 1)
    idx_hist = np.arange(h_start, idx_end + 1, dtype=float)
    z_hist = sig_complex[h_start : idx_end + 1]
    c_hist = z_hist * np.exp(-1j * w_nom * idx_hist / FS)

    fit = _fit_two_tone(c_hist, max_offset=max_offset)
    if fit is None:
        return None

    dw1, dw2 = fit["dw1"], fit["dw2"]
    a1, a2 = fit["a1"], fit["a2"]

    n_hist = len(c_hist)
    n_idx_hist = np.arange(n_hist, dtype=float)
    c_last_model = a1 * np.exp(1j * dw1 * (n_hist - 1) / FS) + a2 * np.exp(1j * dw2 * (n_hist - 1) / FS)
    if np.abs(c_last_model) > 1e-12:
        alpha = c_hist[-1] / c_last_model
        a1 = a1 * alpha
        a2 = a2 * alpha

    n_idx_fwd = np.arange(n_hist, n_hist + n_forward, dtype=float)
    c_fwd = a1 * np.exp(1j * dw1 * n_idx_fwd / FS) + a2 * np.exp(1j * dw2 * n_idx_fwd / FS)
    idx_fwd_abs = np.arange(idx_end + 1, idx_end + 1 + n_forward, dtype=float)
    z_fwd = c_fwd * np.exp(1j * w_nom * idx_fwd_abs / FS)

    env_hist = np.abs(c_hist)
    env_fwd = np.abs(c_fwd)
    p95 = float(np.percentile(env_hist, 95))
    p05 = float(np.percentile(env_hist, 5))
    beat_depth = (p95 - p05) / max(p95 + p05, 1e-12)

    return {
        "projection": z_fwd.real,
        "c_hist": c_hist,
        "env_hist": env_hist,
        "env_pred": env_fwd,
        "hist_start": int(h_start),
        "dw1": float(dw1),
        "dw2": float(dw2),
        "secondary_ratio": float(fit["secondary_ratio"]),
        "fit_confidence": float(fit["fit_confidence"]),
        "beat_depth": float(beat_depth),
        "method": "two_tone",
    }


def _project_bp_with_models(out: dict, idx_end: int, n_forward: int, model_params: dict | None = None) -> dict:
    """
    Project one BP filter with baseline, state, two-tone, and selected output.
    """
    if model_params is None:
        model_params = DEFAULT_BP_MODEL_PARAMS

    params = dict(DEFAULT_BP_MODEL_PARAMS)
    params.update(model_params)

    f0 = float(out["f0"])
    sig_complex = out["signal"]
    env = out["envelope"]
    phase = out["phase"]

    period_samples = TWOPI / f0 * FS

    # Baseline v1
    n_lookback_phase = int(max(24, 3 * period_samples))
    phase_window = phase[max(0, idx_end - n_lookback_phase + 1) : idx_end + 1]
    dphi = np.median(np.diff(phase_window))
    A = float(env[idx_end])
    phi0 = float(phase[idx_end])
    t_fwd = np.arange(1, n_forward + 1, dtype=float)
    pred_v1 = A * np.cos(phi0 + dphi * t_fwd)

    # State model
    n_lookback = int(max(80, params["lookback_cycles"] * period_samples))
    state = _state_projection(
        sig_complex=sig_complex,
        idx_end=idx_end,
        n_forward=n_forward,
        w_nom=f0,
        n_lookback=n_lookback,
        rho_loga=float(params["rho_loga"]),
        rho_dw=float(params["rho_dw"]),
    )
    pred_state = state["projection"]

    # Two-tone
    pred_two = None
    tone2 = None
    if ENABLE_TWO_TONE:
        max_offset = max(1.0, float(out["fwhm"]))
        tone2 = _two_tone_projection(
            sig_complex=sig_complex,
            idx_end=idx_end,
            n_forward=n_forward,
            w_nom=f0,
            n_lookback=n_lookback,
            max_offset=max_offset,
        )
        if tone2 is not None:
            pred_two = tone2["projection"]

    force_model = str(params.get("force_model", "auto"))
    if force_model == "state_demod":
        selected = pred_state
        method = "state_demod"
    elif force_model == "two_tone":
        if pred_two is not None:
            selected = pred_two
            method = "two_tone"
        else:
            selected = pred_state
            method = "state_demod"
    else:
        use_two_tone = False
        if tone2 is not None:
            use_two_tone = (
                state["beat_depth"] >= float(params["beat_depth_min"])
                and tone2["secondary_ratio"] >= float(params["secondary_ratio_min"])
                and tone2["fit_confidence"] >= max(float(params["two_tone_conf_floor"]), state["single_tone_confidence"])
            )
        selected = pred_two if use_two_tone and pred_two is not None else pred_state
        method = "two_tone" if use_two_tone and pred_two is not None else "state_demod"

    confidence = tone2["fit_confidence"] if method == "two_tone" and tone2 is not None else state["single_tone_confidence"]

    return {
        "projection_v1": pred_v1,
        "projection_state": pred_state,
        "projection_two_tone": pred_two,
        "projection_selected": selected,
        "state": state,
        "two_tone": tone2,
        "method_selected": method,
        "confidence": float(confidence),
        "params": params,
    }


def _candidate_model_param_grid() -> list[dict]:
    grid = []
    for rho_loga, rho_dw, lookback_cycles, beat_depth_min, secondary_ratio_min, conf_floor, force_model in itertools.product(
        [0.980, 0.985],
        [0.94, 0.96],
        [6, 8],
        [0.18, 0.24],
        [0.30, 0.40],
        [0.40, 0.50],
        ["auto", "state_demod", "two_tone"],
    ):
        grid.append(
            {
                "rho_loga": float(rho_loga),
                "rho_dw": float(rho_dw),
                "lookback_cycles": int(lookback_cycles),
                "beat_depth_min": float(beat_depth_min),
                "secondary_ratio_min": float(secondary_ratio_min),
                "two_tone_conf_floor": float(conf_floor),
                "force_model": force_model,
            }
        )
    return grid


def _tuning_end_indices(disp_s: int, disp_e: int, horizon: int, n_windows: int) -> np.ndarray:
    earliest = disp_s + max(4 * FS, horizon + 26)
    latest = disp_e - horizon - 1
    if latest <= earliest:
        return np.array([], dtype=int)
    return np.unique(np.linspace(earliest, latest, int(max(2, n_windows))).astype(int))


def _tune_bp_model_params(outputs: list[dict], disp_s: int, disp_e: int, horizon: int, n_windows: int = 6) -> dict:
    """
    Tune BP model-selection parameters on rolling historical windows.
    """
    ends = _tuning_end_indices(disp_s=disp_s, disp_e=disp_e, horizon=horizon, n_windows=n_windows)
    if len(ends) == 0:
        return {}

    grid = _candidate_model_param_grid()
    tuned = {}

    print(f"\nAuto-tuning BP parameters: {len(grid)} candidates x {len(ends)} rolling windows")
    for out in outputs:
        spec = out["spec"]
        if spec["type"] != "bp":
            continue

        sig = out["signal"].real if np.iscomplexobj(out["signal"]) else out["signal"]
        best_score = -np.inf
        best_params = dict(DEFAULT_BP_MODEL_PARAMS)
        best_r2 = []
        best_score_by_force = {k: -np.inf for k in ("auto", "state_demod", "two_tone")}
        best_params_by_force = {}
        best_r2_by_force = {}

        for params in grid:
            r2_scores = []
            for idx_end in ends:
                idx_stop = idx_end + 1 + horizon
                if idx_stop > len(sig):
                    continue
                pack = _project_bp_with_models(out=out, idx_end=idx_end, n_forward=horizon, model_params=params)
                target = sig[idx_end + 1 : idx_stop]
                met = _metrics(target, pack["projection_selected"])
                if np.isfinite(met["r2"]):
                    r2_scores.append(float(met["r2"]))

            if len(r2_scores) == 0:
                continue
            score = float(np.median(r2_scores))
            force_model = str(params.get("force_model", "auto"))
            if force_model in best_score_by_force and score > best_score_by_force[force_model]:
                best_score_by_force[force_model] = score
                best_params_by_force[force_model] = dict(params)
                best_r2_by_force[force_model] = list(r2_scores)
            if score > best_score:
                best_score = score
                best_params = dict(params)
                best_r2 = r2_scores

        guard_applied = "none"
        if (
            spec["label"] == "BP-5: ~0.4 yr"
            and BP5_FORCE_STATE_IF_WEAK_TWO_TONE
            and best_params.get("force_model") == "two_tone"
        ):
            two_score = best_score_by_force.get("two_tone", -np.inf)
            state_score = best_score_by_force.get("state_demod", -np.inf)
            if (
                np.isfinite(two_score)
                and two_score < BP5_TWO_TONE_MIN_MEDIAN_R2
                and np.isfinite(state_score)
                and state_score >= two_score
            ):
                best_params = dict(best_params_by_force.get("state_demod", best_params))
                best_score = float(state_score)
                best_r2 = list(best_r2_by_force.get("state_demod", best_r2))
                guard_applied = "bp5_force_state_if_weak_two_tone"

        tuned[spec["label"]] = {
            **best_params,
            "tune_score_r2_median": float(best_score),
            "tune_score_r2_mean": float(np.mean(best_r2)) if len(best_r2) else float("nan"),
            "tune_windows": int(len(ends)),
            "best_score_auto": float(best_score_by_force.get("auto", np.nan)),
            "best_score_state_demod": float(best_score_by_force.get("state_demod", np.nan)),
            "best_score_two_tone": float(best_score_by_force.get("two_tone", np.nan)),
            "guard_applied": guard_applied,
        }
        print(
            f"  {spec['label']:20s}  best median R2={best_score:+.3f}  "
            f"model={best_params['force_model']}  rhoA={best_params['rho_loga']:.3f}  "
            f"rhoW={best_params['rho_dw']:.3f}  L={best_params['lookback_cycles']}"
        )
        if guard_applied != "none":
            print(
                f"    guard: {guard_applied} "
                f"(two_tone={best_score_by_force.get('two_tone', np.nan):+.3f}, "
                f"state={best_score_by_force.get('state_demod', np.nan):+.3f})"
            )

    return tuned


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(actual) != len(pred) or len(actual) == 0:
        return {"corr": np.nan, "r2": np.nan, "rmse": np.nan, "dir": np.nan, "end_err": np.nan}

    corr = float(np.corrcoef(actual, pred)[0, 1]) if np.std(pred) > 0 else np.nan
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))

    if len(actual) > 1:
        dir_match = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(pred))))
    else:
        dir_match = np.nan
    end_err = float((pred[-1] / actual[-1] - 1.0) * 100.0) if actual[-1] != 0 else np.nan

    return {"corr": corr, "r2": r2, "rmse": rmse, "dir": dir_match, "end_err": end_err}


def _print_fft_padding_report() -> None:
    print("\nFFT / boundary report (no code change applied):")
    print("  - CMW/Morse filtering currently uses FFT multiply + trim to original length.")
    print("  - This keeps current calibration and historical comparability.")
    print("  - Reflect boundary mode exists in Ormsby convolution path, not in CMW/Morse core.")
    print("  - Per request, FFT padding internals were reviewed but left unchanged in this version.")


# ============================================================================
# Main
# ============================================================================

def main(
    wavelet_mode: str = WAVELET_MODE,
    morse_gamma: float = MORSE_GAMMA,
    auto_tune: bool = False,
    tune_windows: int = 6,
    save_tuned_json: bool = True,
) -> None:
    wavelet_mode = wavelet_mode.lower().strip()
    if wavelet_mode not in {"cmw", "morse"}:
        raise ValueError(f"wavelet_mode must be 'cmw' or 'morse', got '{wavelet_mode}'")

    print("=" * 80)
    print("Page 152: 6-Filter Projection codex_v2")
    print(
        f"Wavelet mode: {wavelet_mode.upper()}"
        + (f" (gamma={morse_gamma:.1f})" if wavelet_mode == "morse" else "")
    )
    print("=" * 80)

    df = getStooq("^dji", "w")
    df = df.sort_values("Date").reset_index(drop=True)
    dates = pd.to_datetime(df["Date"]).values
    close = df["Close"].values.astype(float)

    disp_s = int(np.searchsorted(dates, np.datetime64(DISPLAY_START)))
    disp_e = int(np.searchsorted(dates, np.datetime64(DISPLAY_END)))
    proj_e = min(disp_e + PROJECTION_WEEKS, len(close))
    n_proj = proj_e - disp_e

    print(f"\nData: {len(close)} weekly samples")
    print(f"Display: {DISPLAY_START} -> {DISPLAY_END} ({disp_e - disp_s} samples)")
    print(
        f"Projection horizon: {n_proj} weeks "
        f"({pd.Timestamp(dates[disp_e]).strftime('%Y-%m-%d')} -> "
        f"{pd.Timestamp(dates[proj_e - 1]).strftime('%Y-%m-%d')})"
    )

    _print_fft_padding_report()

    print("\nApplying filter bank...")
    outputs = []
    for spec in FILTER_SPECS:
        f0, fwhm = spec_to_wavelet_params(spec)
        analytic = spec["type"] != "lp"

        if wavelet_mode == "cmw":
            out = apply_cmw(close, f0, fwhm, FS, analytic=analytic)
        elif wavelet_mode == "morse":
            out = apply_morse(close, f0, fwhm, FS, analytic=analytic, gamma=morse_gamma)
        else:
            raise ValueError(f"Unknown wavelet_mode='{wavelet_mode}'. Use 'cmw' or 'morse'.")

        out["spec"] = spec
        out["f0"] = f0
        out["fwhm"] = fwhm
        outputs.append(out)

        if analytic:
            print(f"  {spec['label']:20s}  f0={f0:6.2f}  fwhm={fwhm:5.2f}  T={TWOPI/f0:5.2f}y")
        else:
            print(f"  {spec['label']:20s}  f0={f0:6.2f}  fwhm={fwhm:5.2f}  (lowpass)")

    # In-sample composite check
    composite = np.zeros_like(close)
    for out in outputs:
        sig = out["signal"]
        composite += sig.real if np.iscomplexobj(sig) else sig
    rms_orig = np.sqrt(np.mean(close[disp_s:disp_e] ** 2))
    rms_err = np.sqrt(np.mean((close[disp_s:disp_e] - composite[disp_s:disp_e]) ** 2))
    pct_energy = (1.0 - rms_err / max(rms_orig, 1e-12)) * 100.0
    print(f"\nComposite energy capture in display window: {pct_energy:.1f}%")

    # Optional auto-tuning (BP only) on historical rolling windows.
    tuned_bp_params = {}
    if auto_tune:
        tuned_bp_params = _tune_bp_model_params(
            outputs=outputs,
            disp_s=disp_s,
            disp_e=disp_e,
            horizon=PROJECTION_WEEKS,
            n_windows=tune_windows,
        )
        if tuned_bp_params and save_tuned_json:
            tuned_path = os.path.join(
                os.path.dirname(__file__),
                f"cmw_6filter_projection_codex_v2_params_{wavelet_mode}.json",
            )
            payload = {
                "wavelet_mode": wavelet_mode,
                "morse_gamma": morse_gamma if wavelet_mode == "morse" else None,
                "display_start": DISPLAY_START,
                "display_end": DISPLAY_END,
                "projection_weeks": PROJECTION_WEEKS,
                "tune_windows": tune_windows,
                "bp_params": tuned_bp_params,
            }
            with open(tuned_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved tuned BP params: {tuned_path}")

    # Projections
    proj_baseline = []
    proj_state = []
    proj_selected = []
    proj_two_tone = []
    diag_rows = []

    for out in outputs:
        spec = out["spec"]
        f0 = out["f0"]

        if spec["type"] == "lp":
            sig_lp = out["signal"]
            lp_pred, slope_wk = _linear_lp_projection(sig_lp, idx_end=disp_e - 1, n_forward=n_proj)
            proj_baseline.append(lp_pred.copy())
            proj_state.append(lp_pred.copy())
            proj_selected.append(lp_pred.copy())
            proj_two_tone.append(None)
            diag_rows.append(
                {
                    "method": "lp_linear",
                    "beat_depth": np.nan,
                    "freq_drift": np.nan,
                    "confidence": np.nan,
                    "extra": f"slope/yr={slope_wk * FS:.3f}",
                    "state": None,
                    "two_tone": None,
                }
            )
            print(f"\n  {spec['label']:20s}  LP linear slope={slope_wk * FS:.3f}/yr")
            continue

        bp_params = dict(DEFAULT_BP_MODEL_PARAMS)
        if spec["label"] in tuned_bp_params:
            bp_params.update(tuned_bp_params[spec["label"]])

        pack = _project_bp_with_models(
            out=out,
            idx_end=disp_e - 1,
            n_forward=n_proj,
            model_params=bp_params,
        )
        state = pack["state"]
        tone2 = pack["two_tone"]
        method = pack["method_selected"]

        proj_baseline.append(pack["projection_v1"])
        proj_state.append(pack["projection_state"])
        proj_two_tone.append(pack["projection_two_tone"])
        proj_selected.append(pack["projection_selected"])

        if tone2 is not None:
            tone_txt = f"dw1={tone2['dw1']:.3f}, dw2={tone2['dw2']:.3f}, r2={tone2['secondary_ratio']:.2f}"
        else:
            tone_txt = "two-tone: n/a"
        extra = (
            f"{tone_txt}; force={bp_params['force_model']} "
            f"rhoA={bp_params['rho_loga']:.3f} rhoW={bp_params['rho_dw']:.3f} "
            f"L={bp_params['lookback_cycles']}"
        )
        diag_rows.append(
            {
                "method": method,
                "beat_depth": state["beat_depth"],
                "freq_drift": state["freq_drift"],
                "confidence": pack["confidence"],
                "extra": extra,
                "state": state,
                "two_tone": tone2,
            }
        )

        print(f"\n  {spec['label']}")
        print(
            f"    beat_depth={state['beat_depth']:.3f}  drift={state['freq_drift']:+.4f} rad/yr/yr  "
            f"state_conf={state['single_tone_confidence']:.3f}"
        )
        if tone2 is not None:
            print(
                f"    2tone_conf={tone2['fit_confidence']:.3f}  secondary_ratio={tone2['secondary_ratio']:.3f}  "
                f"dw=[{tone2['dw1']:+.3f}, {tone2['dw2']:+.3f}]"
            )
        print(
            f"    selected: {method} | force={bp_params['force_model']} "
            f"rhoA={bp_params['rho_loga']:.3f} rhoW={bp_params['rho_dw']:.3f} "
            f"L={bp_params['lookback_cycles']}"
        )

    comp_v1 = np.sum(proj_baseline, axis=0)
    comp_sel = np.sum(proj_selected, axis=0)
    actual_comp_holdout = composite[disp_e:proj_e]
    actual_price_holdout = close[disp_e:proj_e]

    m_v1 = _metrics(actual_comp_holdout, comp_v1)
    m_sel = _metrics(actual_comp_holdout, comp_sel)

    print(f"\n{'=' * 70}")
    print(f"Composite holdout metrics ({n_proj} weeks)")
    print(f"{'=' * 70}")
    print("  Target series: actual holdout composite (sum of true filter outputs)")
    print(f"  {'Metric':18s}  {'Baseline v1':>12s}  {'codex_v2':>12s}")
    print(f"  {'Correlation':18s}  {m_v1['corr']:>12.4f}  {m_sel['corr']:>12.4f}")
    print(f"  {'R2':18s}  {m_v1['r2']:>12.4f}  {m_sel['r2']:>12.4f}")
    print(f"  {'RMSE':18s}  {m_v1['rmse']:>12.2f}  {m_sel['rmse']:>12.2f}")
    print(f"  {'Direction match':18s}  {m_v1['dir'] * 100:>11.1f}%  {m_sel['dir'] * 100:>11.1f}%")
    print(f"  {'End error':18s}  {m_v1['end_err']:>+11.1f}%  {m_sel['end_err']:>+11.1f}%")
    print(f"  {'Actual comp end':18s}  {actual_comp_holdout[-1]:>12.1f}")
    print(f"  {'Actual price end':18s}  {actual_price_holdout[-1]:>12.1f}")

    # Per-filter holdout correlation
    print("\nPer-filter holdout correlation (actual filter output vs selected projection):")
    for out, pred, diag in zip(outputs, proj_selected, diag_rows):
        spec = out["spec"]
        sig = out["signal"].real if np.iscomplexobj(out["signal"]) else out["signal"]
        af = sig[disp_e:proj_e]
        c = np.corrcoef(af, pred)[0, 1] if np.std(pred) > 0 else np.nan
        print(f"  {spec['label']:20s}  corr={c:+.3f}  method={diag['method']}")

    # =========================================================================
    # Plotting
    # =========================================================================
    dates_disp = dates[disp_s:disp_e]
    dates_proj = dates[disp_e:proj_e]

    fig = plt.figure(figsize=(20, 32))
    gs = fig.add_gridspec(len(FILTER_SPECS) + 1, 2, width_ratios=[3.4, 1.6], hspace=0.36, wspace=0.18)

    # Composite row
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(dates_disp, close[disp_s:disp_e], color="#666666", lw=0.9, alpha=0.45, label="DJIA price (in-sample context)")
    ax0.plot(
        dates_disp,
        composite[disp_s:disp_e],
        color="#1a3b8f",
        lw=1.1,
        alpha=0.8,
        label=f"True composite in-sample ({pct_energy:.1f}% vs price)",
    )
    ax0.plot(
        dates_proj,
        actual_comp_holdout,
        color="black",
        lw=1.4,
        alpha=0.85,
        label="True holdout composite target",
    )
    ax0.plot(
        dates_proj,
        actual_price_holdout,
        color="#888888",
        lw=0.8,
        alpha=0.35,
        ls=":",
        label="Holdout price (context)",
    )
    ax0.plot(dates_proj, comp_v1, color="#2aa1c0", lw=1.2, alpha=0.7, label=f"Baseline v1 (R2={m_v1['r2']:.3f})")
    ax0.plot(dates_proj, comp_sel, color="#d11f5e", lw=1.8, alpha=0.9, label=f"codex_v2 selected (R2={m_sel['r2']:.3f})")
    ax0.axvline(dates[disp_e], color="gray", ls="--", lw=1, alpha=0.5)
    ax0.set_title(
        f"Page 152 Projection codex_v2 | Mode={wavelet_mode.upper()} | "
        "Baseline vs State/Beat-Aware Selection",
        fontsize=12,
        fontweight="bold",
    )
    ax0.set_ylabel("Price")
    ax0.grid(True, alpha=0.2)
    ax0.legend(fontsize=8, loc="upper left")
    ax0.xaxis.set_major_locator(mdates.YearLocator(2))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Filter rows
    for i, (out, p_v1, p_state, p_sel, p2, diag) in enumerate(
        zip(outputs, proj_baseline, proj_state, proj_selected, proj_two_tone, diag_rows), start=1
    ):
        spec = out["spec"]
        color = spec["color"]

        ax_sig = fig.add_subplot(gs[i, 0])
        sig = out["signal"].real if np.iscomplexobj(out["signal"]) else out["signal"]
        ax_sig.plot(dates_disp, sig[disp_s:disp_e], color=color, lw=0.7, alpha=0.8, label="In-sample filter signal")

        if out["envelope"] is not None:
            env_disp = out["envelope"][disp_s:disp_e]
            ax_sig.plot(dates_disp, env_disp, color=color, lw=0.9, alpha=0.5, label="In-sample envelope +")
            ax_sig.plot(dates_disp, -env_disp, color=color, lw=0.9, alpha=0.5, label="_nolegend_")

        hold_sig = sig[disp_e:proj_e]
        ax_sig.plot(
            dates_proj,
            hold_sig,
            color="black",
            lw=1.0,
            alpha=0.75,
            label="Holdout target signal",
        )
        if out["envelope"] is not None:
            hold_env = out["envelope"][disp_e:proj_e]
            ax_sig.plot(
                dates_proj,
                hold_env,
                color="black",
                lw=0.9,
                alpha=0.45,
                ls="--",
                label="Holdout target envelope +",
            )
            ax_sig.plot(
                dates_proj,
                -hold_env,
                color="black",
                lw=0.9,
                alpha=0.45,
                ls="--",
                label="_nolegend_",
            )

        # Projections
        ax_sig.plot(dates_proj, p_v1, color="#2aa1c0", lw=1.0, alpha=0.55, ls="--", label="v1")
        ax_sig.plot(dates_proj, p_state, color="#f18f01", lw=1.1, alpha=0.7, label="state")
        if p2 is not None:
            ax_sig.plot(dates_proj, p2, color="#a71d5d", lw=1.1, alpha=0.55, ls="-.", label="2-tone")
        ax_sig.plot(dates_proj, p_sel, color="#d11f5e", lw=1.6, alpha=0.9, label="selected")

        ax_sig.axvline(dates[disp_e], color="gray", ls="--", lw=0.8, alpha=0.5)
        ax_sig.axhline(0, color="gray", lw=0.3)
        ax_sig.set_ylabel(spec["label"], fontsize=8, rotation=0, labelpad=80, ha="left")
        ax_sig.grid(True, alpha=0.15)
        ax_sig.tick_params(axis="y", labelsize=7)
        ax_sig.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_sig.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_sig.legend(fontsize=6.5, loc="upper right", ncol=2)

        ax_diag = fig.add_subplot(gs[i, 1])
        if spec["type"] == "lp":
            ax_diag.text(0.04, 0.92, "Method: LP linear extrapolation", transform=ax_diag.transAxes, fontsize=8, va="top")
            ax_diag.text(0.04, 0.78, diag["extra"], transform=ax_diag.transAxes, fontsize=8, va="top")
            ax_diag.axis("off")
            continue

        state = diag["state"]
        hist_start = state["hist_start"]
        hist_dates = dates[hist_start:disp_e]
        env_hist = state["env_hist"]
        env_pred = state["env_pred"]
        dw_hist = state["dw_hist"]
        dw_pred = state["dw_pred"]

        ax_diag.plot(hist_dates, env_hist, color=color, lw=0.8, alpha=0.6)
        ax_diag.plot(dates_proj, env_pred, color="#f18f01", lw=1.0, alpha=0.7, ls="--")
        if diag["two_tone"] is not None:
            env_pred_2 = diag["two_tone"]["env_pred"]
            ax_diag.plot(dates_proj, env_pred_2, color="#a71d5d", lw=1.0, alpha=0.6, ls="-.")

        ax_diag.set_ylabel("|c|", fontsize=7)
        ax_diag.grid(True, alpha=0.15)
        ax_diag.tick_params(axis="both", labelsize=6)
        ax_diag.xaxis.set_major_locator(mdates.YearLocator(4))
        ax_diag.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax_w = ax_diag.twinx()
        ax_w.plot(hist_dates, dw_hist, color="#6b1d1d", lw=0.6, alpha=0.55)
        ax_w.plot(dates_proj, dw_pred, color="#b22222", lw=0.8, alpha=0.7, ls="--")
        ax_w.set_ylabel("dw (rad/yr)", fontsize=7, color="#6b1d1d")
        ax_w.tick_params(axis="y", labelsize=6, colors="#6b1d1d")

        txt = (
            f"method={diag['method']}\n"
            f"beat={diag['beat_depth']:.3f}\n"
            f"drift={diag['freq_drift']:+.3f}/yr\n"
            f"conf={diag['confidence']:.3f}\n"
            f"{diag['extra']}"
        )
        ax_diag.text(
            0.98,
            0.98,
            txt,
            transform=ax_diag.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
        )

    fig.suptitle(
        "Page 152 codex_v2: Demodulated State Model + Beat-Aware Two-Tone Projection\n"
        f"Wavelet mode={wavelet_mode.upper()} | FFT boundary behavior unchanged by request",
        fontsize=12,
        fontweight="bold",
        y=1.0,
    )

    outpath = os.path.join(os.path.dirname(__file__), f"cmw_6filter_projection_codex_v2_{wavelet_mode}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {outpath}")

    try:
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    except Exception:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Page 152 codex_v2 projection with CMW or Morse mode.")
    parser.add_argument(
        "--wavelet-mode",
        default=WAVELET_MODE,
        choices=["cmw", "morse"],
        help="Filtering mode for BP/LP wavelets.",
    )
    parser.add_argument(
        "--morse-gamma",
        default=MORSE_GAMMA,
        type=float,
        help="Gamma parameter for generalized Morse filter (used only in morse mode).",
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Auto-tune BP model-selection params on rolling historical windows before final projection.",
    )
    parser.add_argument(
        "--tune-windows",
        default=6,
        type=int,
        help="Number of rolling windows used for auto-tuning.",
    )
    parser.add_argument(
        "--no-save-tuned-json",
        action="store_true",
        help="Do not write tuned parameter JSON when --auto-tune is enabled.",
    )
    args = parser.parse_args()
    main(
        wavelet_mode=args.wavelet_mode,
        morse_gamma=args.morse_gamma,
        auto_tune=args.auto_tune,
        tune_windows=args.tune_windows,
        save_tuned_json=not args.no_save_tuned_json,
    )
