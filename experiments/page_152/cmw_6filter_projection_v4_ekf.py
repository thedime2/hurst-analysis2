# -*- coding: utf-8 -*-
"""
Page 152: 6-Filter CMW Projection v4 -- Extended Kalman Filter
===============================================================

Models each BP filter output as a sum of M complex oscillators with
slowly-varying amplitude and frequency, tracked by an EKF:

    z(t) = sum_k  c_k(t) * exp(j * phi_k(t))

State per mode:  [Re(c_k), Im(c_k), omega_k]   (3 states)
M modes per filter => 3M state dimension

Transition:  rotation by omega_k with random walk on omega_k
Observation: Re(sum_k c_k) + Im(sum_k c_k) from the analytic CMW output

Initialization: MPM pole extraction (frequency-gated)
Prediction: deterministic propagation of Kalman state from cutoff

Also includes:
  v4b: Phase-space nearest-neighbor prediction (Takens embedding)
       -- a nonlinear dynamics approach for comparison

Compared against:
  v1:  Static amplitude + median phase rate
  v3b: MPM envelope + v1 phase (hybrid)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.linalg import svd

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

# EKF parameters
EKF_FIT_CYCLES = 8       # run EKF over last N cycles for convergence
EKF_MODES_PER_FILTER = 2  # number of oscillator modes per BP filter
SIGMA_Z = 0.02            # process noise on complex amplitude (relative)
SIGMA_OMEGA = 0.005       # process noise on frequency (rad/yr per step)
SIGMA_OBS = 0.01           # observation noise (relative)

# MPM parameters for initialization
MPM_MAX_ORDER = 4
SV_THRESHOLD = 0.05

# Phase-space embedding parameters
PSNN_EMBED_DIM = 6        # embedding dimension
PSNN_N_NEIGHBORS = 5      # number of nearest neighbors
PSNN_FIT_CYCLES = 10      # history for neighbor search

FILTER_SPECS = [
    {'type': 'lp', 'f_pass': 0.85, 'f_stop': 1.25,
     'label': 'LP-1: Trend (>5 yr)', 'color': '#1f77b4'},
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


def get_passband(spec):
    """Return (low, high) passband in rad/yr with margin."""
    if spec['type'] == 'lp':
        return (0.0, spec['f_stop'] * 1.5)
    return (spec['f1'] * 0.5, spec['f4'] * 1.5)


# ============================================================================
# MPM for EKF initialization
# ============================================================================

def matrix_pencil(y, L=None, max_order=4, sv_thresh=0.05):
    """MPM: decompose y[n] into sum_k c_k * z_k^n."""
    N = len(y)
    if L is None:
        L = N // 3

    rows = N - L
    H = np.zeros((rows, L + 1), dtype=complex)
    for col in range(L + 1):
        H[:, col] = y[col:col + rows]

    Y0, Y1 = H[:, :-1], H[:, 1:]
    U, s, Vh = svd(Y0, full_matrices=False)
    sv_ratio = s / s[0]

    M = 1
    for i in range(1, min(len(s), max_order)):
        if sv_ratio[i] > sv_thresh:
            M = i + 1
        else:
            break

    S_inv = np.diag(1.0 / s[:M])
    A = S_inv @ (U[:, :M].conj().T @ Y1 @ Vh[:M, :].conj().T)
    poles = np.linalg.eigvals(A)

    # Solve for amplitudes
    n_idx = np.arange(N)
    Z = np.column_stack([poles[k] ** n_idx for k in range(M)])
    amps, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)

    return poles, amps, M


def get_initial_modes(sig_complex, n_fit, spec, n_modes=2):
    """
    Use MPM to extract initial mode parameters for EKF.
    Returns list of (omega_rad_yr, amplitude_complex) tuples.
    """
    y = sig_complex[-n_fit:]
    L = max(n_fit // 3, MPM_MAX_ORDER + 2)

    try:
        poles, amps, order = matrix_pencil(y, L=L, max_order=MPM_MAX_ORDER,
                                            sv_thresh=SV_THRESHOLD)
    except Exception:
        # Fallback: use nominal center frequency
        f0 = (spec['f2'] + spec['f3']) / 2.0
        return [(f0, sig_complex[-1] / 2)]

    # Frequency gate
    pb_low, pb_high = get_passband(spec)
    freqs = np.abs(np.angle(poles) * FS)

    valid = [(poles[i], amps[i], freqs[i]) for i in range(len(poles))
             if pb_low <= freqs[i] <= pb_high and freqs[i] > 0.1]

    if not valid:
        f0 = (spec['f2'] + spec['f3']) / 2.0
        return [(f0, sig_complex[-1] / 2)]

    # Sort by amplitude magnitude, take top n_modes
    valid.sort(key=lambda x: np.abs(x[1]), reverse=True)
    valid = valid[:n_modes]

    # Convert discrete pole to omega (rad/yr)
    modes = []
    for pole, amp, freq in valid:
        omega = freq  # already in rad/yr
        # The amplitude at the END of the fit window
        c = amp * pole ** (n_fit - 1)
        modes.append((omega, c))

    # If we have fewer than n_modes, duplicate the dominant at slightly offset freq
    while len(modes) < n_modes:
        w0, c0 = modes[0]
        offset = 0.2 * (len(modes))  # small offset
        modes.append((w0 + offset, c0 * 0.1))

    return modes


# ============================================================================
# Extended Kalman Filter -- multi-mode complex oscillator
# ============================================================================

class MultiModeEKF:
    """
    EKF tracking M complex oscillator modes.

    State: [z1_re, z1_im, w1, z2_re, z2_im, w2, ...]  (3M states)
    Observation: [Re(sum_k z_k), Im(sum_k z_k)]  (2 observations)
    """

    def __init__(self, modes, sigma_z=0.02, sigma_omega=0.005, sigma_obs=0.01):
        """
        modes: list of (omega_rad_yr, c_complex) tuples
        """
        self.M = len(modes)
        self.n_state = 3 * self.M
        self.fs = FS

        # Initialize state
        self.x = np.zeros(self.n_state)
        for k, (omega, c) in enumerate(modes):
            self.x[3*k] = c.real
            self.x[3*k + 1] = c.imag
            self.x[3*k + 2] = omega

        # Initial covariance
        self.P = np.eye(self.n_state)
        for k in range(self.M):
            amp = np.abs(modes[k][1])
            self.P[3*k, 3*k] = (amp * 0.5)**2
            self.P[3*k+1, 3*k+1] = (amp * 0.5)**2
            self.P[3*k+2, 3*k+2] = 1.0**2  # omega uncertainty: 1 rad/yr

        # Process noise
        self.Q_base = np.zeros((self.n_state, self.n_state))
        for k in range(self.M):
            amp = max(np.abs(modes[k][1]), 1e-6)
            self.Q_base[3*k, 3*k] = (sigma_z * amp)**2
            self.Q_base[3*k+1, 3*k+1] = (sigma_z * amp)**2
            self.Q_base[3*k+2, 3*k+2] = (sigma_omega)**2

        # Observation noise
        self.R = sigma_obs**2 * np.eye(2)

        # Observation matrix H (2 x 3M): sum of all modes' Re and Im
        self.H = np.zeros((2, self.n_state))
        for k in range(self.M):
            self.H[0, 3*k] = 1.0      # Re(z_k) contributes to observed Re
            self.H[1, 3*k + 1] = 1.0  # Im(z_k) contributes to observed Im

    def _transition(self, x):
        """Nonlinear state transition: rotate each mode by its omega."""
        x_new = np.copy(x)
        for k in range(self.M):
            zr = x[3*k]
            zi = x[3*k + 1]
            w = x[3*k + 2]

            c = np.cos(w / self.fs)
            s = np.sin(w / self.fs)

            x_new[3*k] = zr * c - zi * s
            x_new[3*k + 1] = zr * s + zi * c
            # omega stays the same (random walk via process noise)

        return x_new

    def _jacobian(self, x):
        """Jacobian of transition function."""
        F = np.zeros((self.n_state, self.n_state))

        for k in range(self.M):
            zr = x[3*k]
            zi = x[3*k + 1]
            w = x[3*k + 2]

            c = np.cos(w / self.fs)
            s = np.sin(w / self.fs)

            i = 3 * k
            F[i, i] = c
            F[i, i+1] = -s
            F[i, i+2] = (-zr * s - zi * c) / self.fs

            F[i+1, i] = s
            F[i+1, i+1] = c
            F[i+1, i+2] = (zr * c - zi * s) / self.fs

            F[i+2, i+2] = 1.0

        return F

    def predict(self):
        """EKF prediction step."""
        F = self._jacobian(self.x)
        self.x = self._transition(self.x)
        self.P = F @ self.P @ F.T + self.Q_base
        return self.x.copy()

    def update(self, y_obs):
        """EKF update step with observation y_obs = [Re(z_sum), Im(z_sum)]."""
        y_pred = self.H @ self.x
        innovation = y_obs - y_pred

        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x.copy()

        self.x = self.x + K @ innovation
        I_KH = np.eye(self.n_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form

        return self.x.copy()

    def step(self, y_obs):
        """Predict + update."""
        self.predict()
        return self.update(y_obs)

    def get_prediction(self, n_forward):
        """
        Propagate current state forward n_forward steps (no observations).
        Returns real part of sum of modes at each step.
        """
        x = self.x.copy()
        predictions_re = np.zeros(n_forward)
        predictions_im = np.zeros(n_forward)

        for t in range(n_forward):
            x = self._transition(x)
            z_sum = 0.0 + 0.0j
            for k in range(self.M):
                z_sum += x[3*k] + 1j * x[3*k + 1]
            predictions_re[t] = z_sum.real
            predictions_im[t] = z_sum.imag

        return predictions_re

    def get_mode_info(self):
        """Return current mode parameters."""
        modes = []
        for k in range(self.M):
            zr = self.x[3*k]
            zi = self.x[3*k + 1]
            w = self.x[3*k + 2]
            amp = np.sqrt(zr**2 + zi**2)
            phi = np.arctan2(zi, zr)
            modes.append({'omega': w, 'amplitude': amp, 'phase': phi,
                          'z_re': zr, 'z_im': zi})
        return modes


# ============================================================================
# Phase-Space Nearest Neighbor Prediction (Takens embedding)
# ============================================================================

def psnn_predict(signal_real, n_forward, embed_dim=6, n_neighbors=5, n_history=500):
    """
    Phase-space nearest-neighbor prediction.

    1. Embed signal in delay-coordinate space (Takens theorem)
    2. Find k nearest neighbors to the current state
    3. Average their future trajectories weighted by inverse distance

    This captures nonlinear dynamics that sinusoidal models miss.
    """
    sig = signal_real[-n_history:]
    N = len(sig)

    if N < embed_dim + n_forward + 10:
        return np.full(n_forward, sig[-1])  # fallback

    # Determine optimal delay from first zero-crossing of autocorrelation
    autocorr = np.correlate(sig - np.mean(sig), sig - np.mean(sig), mode='full')
    autocorr = autocorr[N-1:]
    autocorr = autocorr / autocorr[0]

    delay = 1
    for i in range(1, min(N // 4, 50)):
        if autocorr[i] < 0:
            delay = i
            break

    # Build embedding matrix
    n_vectors = N - (embed_dim - 1) * delay
    if n_vectors < n_forward + n_neighbors + 1:
        delay = max(1, delay // 2)
        n_vectors = N - (embed_dim - 1) * delay

    if n_vectors < n_forward + n_neighbors + 1:
        return np.full(n_forward, sig[-1])  # fallback

    embedding = np.zeros((n_vectors, embed_dim))
    for d in range(embed_dim):
        embedding[:, d] = sig[d * delay:d * delay + n_vectors]

    # Current state = last embedding vector
    current = embedding[-1:]

    # Find k nearest neighbors (excluding last n_forward vectors to avoid look-ahead)
    max_idx = n_vectors - n_forward - 1
    if max_idx < n_neighbors:
        return np.full(n_forward, sig[-1])

    candidates = embedding[:max_idx]
    distances = np.sqrt(np.sum((candidates - current)**2, axis=1))

    # Get top-k neighbors
    nn_idx = np.argsort(distances)[:n_neighbors]
    nn_dists = distances[nn_idx]

    # Avoid division by zero
    nn_dists = np.maximum(nn_dists, 1e-10)
    weights = 1.0 / nn_dists
    weights /= weights.sum()

    # Average future trajectories of neighbors
    # For each neighbor at index i, its future is sig[i + (embed_dim-1)*delay + 1 : ...]
    prediction = np.zeros(n_forward)
    for j, (idx, w) in enumerate(zip(nn_idx, weights)):
        # The neighbor's embedding starts at sig index = idx
        # The "present" of this neighbor in the original signal is at
        # idx + (embed_dim - 1) * delay
        present_idx = idx + (embed_dim - 1) * delay
        future_start = present_idx + 1
        future_end = future_start + n_forward

        if future_end > N:
            # Not enough future data, use what we have + extrapolate
            available = N - future_start
            if available > 0:
                prediction[:available] += w * sig[future_start:N]
                prediction[available:] += w * sig[N - 1]
            else:
                prediction += w * sig[-1]
        else:
            prediction += w * sig[future_start:future_end]

    return prediction


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Page 152: 6-Filter CMW Projection v4")
    print("  v4a: Extended Kalman Filter (multi-mode complex oscillator)")
    print("  v4b: Phase-space nearest-neighbor (Takens embedding)")
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

    # --- Filter with CMW ---
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
            print(f"  {spec['label']:20s}  f0={f0:.2f}  FWHM={fwhm:.2f}  T={TWOPI/f0:.2f}yr")
        else:
            print(f"  {spec['label']:20s}  f0={f0:.2f}  FWHM={fwhm:.2f}  (lowpass)")

    # --- Composite reconstruction ---
    composite = np.zeros_like(close)
    for out in filter_outputs:
        sig = out['signal']
        composite += sig.real if np.iscomplexobj(sig) else sig

    disp_prices = close[disp_s:disp_e]
    disp_composite = composite[disp_s:disp_e]
    rms_orig = np.sqrt(np.mean(disp_prices**2))
    rms_err = np.sqrt(np.mean((disp_prices - disp_composite)**2))
    pct_energy = (1 - rms_err / rms_orig) * 100
    print(f"\nComposite energy captured: {pct_energy:.1f}%")

    # ========================================================================
    # PROJECTIONS
    # ========================================================================

    proj_v1 = []     # v1: static A + median dphi
    proj_v3b = []    # v3b: MPM envelope + v1 phase (from v3)
    proj_v4a = []    # v4a: EKF
    proj_v4b = []    # v4b: phase-space NN
    ekf_diag = []    # diagnostics

    for i, out in enumerate(filter_outputs):
        spec = out['spec']
        f0 = out['f0']

        if spec['type'] == 'lp':
            # LP: linear extrapolation (same for all methods)
            sig = out['signal']
            lookback = min(52, disp_e - disp_s)
            segment = np.array(sig[disp_e - lookback:disp_e], dtype=np.float64)
            t_seg = np.arange(lookback)
            coeffs = np.polyfit(t_seg, segment, 1)
            t_fwd = np.arange(lookback, lookback + n_proj)
            lp_proj = np.polyval(coeffs, t_fwd)

            proj_v1.append(lp_proj)
            proj_v3b.append(lp_proj)
            proj_v4a.append(lp_proj)
            proj_v4b.append(lp_proj)
            ekf_diag.append(None)
            print(f"\n  {spec['label']:20s}  LP linear extrap, slope={coeffs[0]*FS:.2f}/yr")
            continue

        # ---- BP filter ----
        period_samples = TWOPI / f0 * FS
        env = out['envelope']
        phase = out['phase']
        sig_complex = out['signal']
        sig_real = sig_complex.real if np.iscomplexobj(sig_complex) else sig_complex

        # --- v1: static amplitude + median phase rate ---
        A_v1 = env[disp_e - 1]
        n_lookback = int(3 * period_samples)
        phase_window = phase[max(disp_s, disp_e - n_lookback):disp_e]
        dphi_v1 = np.median(np.diff(phase_window))
        phi_start = phase[disp_e - 1]
        w_eff_v1 = dphi_v1 * FS

        t_fwd = np.arange(1, n_proj + 1)
        pv1 = A_v1 * np.cos(phi_start + dphi_v1 * t_fwd)
        proj_v1.append(pv1)

        # --- v3b: MPM envelope + v1 phase (simple replication) ---
        # Use median envelope of last 2 cycles as modulated amplitude
        n_2cyc = int(2 * period_samples)
        env_recent = env[max(0, disp_e - n_2cyc):disp_e]
        A_med = np.median(env_recent)
        pv3b = A_med * np.cos(phi_start + dphi_v1 * t_fwd)
        proj_v3b.append(pv3b)

        # --- v4a: Extended Kalman Filter ---
        print(f"\n  {spec['label']}")

        # Initialize from MPM
        n_init = int(EKF_FIT_CYCLES * period_samples)
        n_init = min(n_init, disp_e)
        n_init = max(n_init, 80)

        modes = get_initial_modes(sig_complex[:disp_e], n_init, spec,
                                  n_modes=EKF_MODES_PER_FILTER)

        # Scale observation noise to signal level
        sig_rms = np.sqrt(np.mean(np.abs(sig_complex[disp_e - n_init:disp_e])**2))
        obs_noise = SIGMA_OBS * sig_rms

        ekf = MultiModeEKF(modes,
                           sigma_z=SIGMA_Z,
                           sigma_omega=SIGMA_OMEGA,
                           sigma_obs=obs_noise)

        # Run EKF through the last n_init samples
        ekf_track = np.zeros(n_init, dtype=complex)
        sig_window = sig_complex[disp_e - n_init:disp_e]

        for t in range(n_init):
            y_obs = np.array([sig_window[t].real, sig_window[t].imag])
            x_est = ekf.step(y_obs)
            z_sum = sum(x_est[3*k] + 1j * x_est[3*k+1]
                       for k in range(ekf.M))
            ekf_track[t] = z_sum

        # Get mode info at cutoff
        mode_info = ekf.get_mode_info()
        for k, m in enumerate(mode_info):
            print(f"    Mode {k+1}: w={m['omega']:.3f} rad/yr  "
                  f"A={m['amplitude']:.1f}  phi={m['phase']:.2f} rad")

        # EKF boundary fit
        n_bnd = max(n_init // 5, 10)
        bnd_actual = sig_window[-n_bnd:]
        bnd_ekf = ekf_track[-n_bnd:]
        bnd_corr = np.abs(np.corrcoef(bnd_actual.real, bnd_ekf.real)[0, 1])
        print(f"    Boundary correlation: {bnd_corr:.4f}")

        # Project forward
        pv4a = ekf.get_prediction(n_proj)
        proj_v4a.append(pv4a)

        ekf_diag.append({
            'ekf_track': ekf_track,
            'n_init': n_init,
            'mode_info': mode_info,
            'boundary_corr': bnd_corr,
        })

        # --- v4b: Phase-space nearest neighbor ---
        n_psnn_history = min(int(PSNN_FIT_CYCLES * period_samples),
                             disp_e - disp_s)
        n_psnn_history = max(n_psnn_history, 200)

        sig_history = sig_real[disp_e - n_psnn_history:disp_e]
        pv4b = psnn_predict(sig_history, n_proj,
                            embed_dim=PSNN_EMBED_DIM,
                            n_neighbors=PSNN_N_NEIGHBORS,
                            n_history=n_psnn_history)
        proj_v4b.append(pv4b)

        print(f"    v1:  A={A_v1:.1f}  w_eff={w_eff_v1:.3f}")
        print(f"    v4a: EKF {EKF_MODES_PER_FILTER} modes")
        print(f"    v4b: PSNN embed={PSNN_EMBED_DIM} nn={PSNN_N_NEIGHBORS}")

    # --- Composite projections ---
    comp_v1 = np.sum(proj_v1, axis=0)
    comp_v3b = np.sum(proj_v3b, axis=0)
    comp_v4a = np.sum(proj_v4a, axis=0)
    comp_v4b = np.sum(proj_v4b, axis=0)

    actual_proj = close[disp_e:proj_e]

    # --- Metrics ---
    def r2(a, p):
        ss_r = np.sum((a - p)**2)
        ss_t = np.sum((a - np.mean(a))**2)
        return 1 - ss_r / ss_t if ss_t > 0 else 0

    metrics = {}
    for name, comp in [('v1', comp_v1), ('v3b', comp_v3b),
                        ('v4a', comp_v4a), ('v4b', comp_v4b)]:
        corr = np.corrcoef(actual_proj, comp)[0, 1]
        r2_val = r2(actual_proj, comp)
        rmse = np.sqrt(np.mean((actual_proj - comp)**2))
        direction = np.mean(np.sign(np.diff(actual_proj)) == np.sign(np.diff(comp)))
        err = (comp[-1] / actual_proj[-1] - 1) * 100
        metrics[name] = {'corr': corr, 'r2': r2_val, 'rmse': rmse,
                         'dir': direction, 'end_err': err, 'end_price': comp[-1]}

    print(f"\n{'='*75}")
    print(f"PROJECTION COMPARISON ({n_proj} weeks)")
    print(f"{'='*75}")
    hdr = f"  {'Metric':<22}  {'v1 static':>10}  {'v3b hybrid':>10}  {'v4a EKF':>10}  {'v4b PSNN':>10}"
    print(hdr)
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for key, label in [('corr', 'Correlation'), ('r2', 'R2'),
                        ('rmse', 'RMSE'), ('dir', 'Direction match'),
                        ('end_price', 'End price'), ('end_err', 'End price err')]:
        vals = [metrics[n][key] for n in ['v1', 'v3b', 'v4a', 'v4b']]
        if key == 'dir':
            print(f"  {label:22s}  {vals[0]*100:>9.1f}%  {vals[1]*100:>9.1f}%  "
                  f"{vals[2]*100:>9.1f}%  {vals[3]*100:>9.1f}%")
        elif key == 'end_err':
            print(f"  {label:22s}  {vals[0]:>+9.1f}%  {vals[1]:>+9.1f}%  "
                  f"{vals[2]:>+9.1f}%  {vals[3]:>+9.1f}%")
        elif key in ('rmse', 'end_price'):
            print(f"  {label:22s}  {vals[0]:>10.1f}  {vals[1]:>10.1f}  "
                  f"{vals[2]:>10.1f}  {vals[3]:>10.1f}")
        else:
            print(f"  {label:22s}  {vals[0]:>10.4f}  {vals[1]:>10.4f}  "
                  f"{vals[2]:>10.4f}  {vals[3]:>10.4f}")
    print(f"  {'Actual end price':22s}  {actual_proj[-1]:>10.1f}")

    # --- Per-filter comparison ---
    print(f"\n--- Per-filter correlation with actual holdout ---")
    for i, (out, pv1, pv3b, pv4a, pv4b) in enumerate(
            zip(filter_outputs, proj_v1, proj_v3b, proj_v4a, proj_v4b)):
        spec = out['spec']
        sig_real = out['signal'].real if np.iscomplexobj(out['signal']) else out['signal']
        if proj_e <= len(sig_real):
            af = sig_real[disp_e:proj_e]
            corrs = []
            for pv in [pv1, pv3b, pv4a, pv4b]:
                c = np.corrcoef(af, pv)[0, 1] if np.std(pv) > 0 else 0
                corrs.append(c)
            print(f"  {spec['label']:20s}  v1={corrs[0]:+.3f}  v3b={corrs[1]:+.3f}  "
                  f"v4a={corrs[2]:+.3f}  v4b={corrs[3]:+.3f}")

    # ========================================================================
    # PLOTTING
    # ========================================================================

    dates_disp = dates[disp_s:disp_e]
    dates_proj = dates[disp_e:proj_e]

    n_filters = len(FILTER_SPECS)
    fig = plt.figure(figsize=(20, 34))
    gs = fig.add_gridspec(n_filters + 1, 2, width_ratios=[3, 1],
                          hspace=0.35, wspace=0.2)

    # === Row 0: Composite ===
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(dates_disp, close[disp_s:disp_e], 'k-', lw=1.2, label='DJIA Close')
    ax0.plot(dates_disp, composite[disp_s:disp_e], 'b-', lw=0.8, alpha=0.3,
             label=f'Composite ({pct_energy:.1f}%)')
    ax0.plot(dates_proj, actual_proj, 'k-', lw=1.2, alpha=0.3, label='Actual (holdout)')
    ax0.plot(dates_proj, comp_v1, 'c-', lw=1, alpha=0.5,
             label=f'v1 static (R2={metrics["v1"]["r2"]:.3f})')
    ax0.plot(dates_proj, comp_v4a, 'r-', lw=2, alpha=0.8,
             label=f'v4a EKF (R2={metrics["v4a"]["r2"]:.3f})')
    ax0.plot(dates_proj, comp_v4b, 'g-', lw=1.5, alpha=0.6,
             label=f'v4b PSNN (R2={metrics["v4b"]["r2"]:.3f})')
    ax0.axvline(dates[disp_e], color='grey', ls='--', lw=1, alpha=0.5)
    ax0.set_ylabel('Price', fontsize=9)
    ax0.set_title('v4: EKF + Phase-Space NN vs v1 Baseline -- 100-Week Projection',
                  fontsize=12, fontweight='bold')
    ax0.legend(loc='upper left', fontsize=8)
    ax0.grid(True, alpha=0.2)
    ax0.xaxis.set_major_locator(mdates.YearLocator(2))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # === Rows 1-6: Individual filters ===
    for i, (out, pv1, pv4a, pv4b, diag) in enumerate(
            zip(filter_outputs, proj_v1, proj_v4a, proj_v4b, ekf_diag)):
        spec = out['spec']
        color = spec['color']

        # --- Left: signal + projections ---
        ax_sig = fig.add_subplot(gs[i + 1, 0])

        sig_real = out['signal'].real if np.iscomplexobj(out['signal']) else out['signal']
        ax_sig.plot(dates_disp, sig_real[disp_s:disp_e], color=color, lw=0.5, alpha=0.6)

        if out['envelope'] is not None:
            env_disp = out['envelope'][disp_s:disp_e]
            ax_sig.plot(dates_disp, env_disp, color=color, lw=1, alpha=0.4)
            ax_sig.plot(dates_disp, -env_disp, color=color, lw=1, alpha=0.4)

        # EKF tracking (last portion of display)
        if diag is not None:
            n_init = diag['n_init']
            track_s = disp_e - n_init
            if track_s >= disp_s:
                track_dates = dates[track_s:disp_e]
                ax_sig.plot(track_dates, diag['ekf_track'].real, 'k-', lw=1,
                           alpha=0.6, label=f'EKF track (r={diag["boundary_corr"]:.3f})')

        # Actual holdout (faint)
        if proj_e <= len(sig_real):
            ax_sig.plot(dates_proj, sig_real[disp_e:proj_e], color=color,
                       lw=0.5, alpha=0.25)

        # Projections
        ax_sig.plot(dates_proj, pv1, 'c-', lw=1, alpha=0.4, label='v1')
        ax_sig.plot(dates_proj, pv4a, 'r-', lw=1.5, alpha=0.8, label='v4a EKF')
        ax_sig.plot(dates_proj, pv4b, 'g-', lw=1, alpha=0.6, label='v4b PSNN')

        ax_sig.axvline(dates[disp_e], color='grey', ls='--', lw=0.8, alpha=0.5)
        ax_sig.axhline(0, color='grey', lw=0.3)
        ax_sig.set_ylabel(spec['label'], fontsize=8, rotation=0, labelpad=80, ha='left')
        ax_sig.grid(True, alpha=0.15)
        ax_sig.tick_params(axis='y', labelsize=7)
        ax_sig.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_sig.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        if i == 0:
            ax_sig.legend(fontsize=7, loc='upper right')

        # --- Right: EKF mode diagram or tracking quality ---
        ax_r = fig.add_subplot(gs[i + 1, 1])

        if diag is not None:
            # Show EKF tracking error over time
            n_init = diag['n_init']
            sig_window = sig_real[disp_e - n_init:disp_e]
            ekf_real = diag['ekf_track'].real
            err = np.abs(sig_window - ekf_real)
            t_track = np.arange(n_init)

            ax_r.semilogy(t_track, err + 1e-10, color=color, lw=0.5, alpha=0.5)
            ax_r.axhline(np.median(err), color='red', ls='--', lw=0.8, alpha=0.7)
            ax_r.set_title(f'EKF tracking error', fontsize=7)
            ax_r.set_ylabel('|error|', fontsize=7)

            # Annotate modes
            mi = diag['mode_info']
            mode_str = '\n'.join([f'M{k+1}: w={m["omega"]:.2f} A={m["amplitude"]:.1f}'
                                  for k, m in enumerate(mi)])
            ax_r.text(0.95, 0.95, mode_str, transform=ax_r.transAxes,
                     fontsize=6, va='top', ha='right', family='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            ax_r.text(0.5, 0.5, 'LP\n(linear)', transform=ax_r.transAxes,
                     ha='center', va='center', fontsize=8)
        ax_r.tick_params(axis='both', labelsize=6)

    fig.suptitle("Page 152: 6-Filter CMW + Extended Kalman Filter Projection\n"
                 "Left: signal + EKF track (black) + projections "
                 "(cyan=v1, red=EKF, green=PSNN)\n"
                 "Right: EKF tracking error + mode parameters",
                 fontsize=11, fontweight='bold', y=1.0)

    outpath = os.path.join(os.path.dirname(__file__), 'cmw_6filter_projection_v4_ekf.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {outpath}")

    try:
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    except Exception:
        plt.close()


if __name__ == '__main__':
    main()
