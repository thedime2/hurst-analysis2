# -*- coding: utf-8 -*-
"""
Generalized Morse Wavelet Filtering (frequency-domain).

Implements a narrowband analytic Morse filter with user-facing parameters:
  - center frequency `f0` (rad/year)
  - spectral width `fwhm` (rad/year)

For bandpass filters, beta is solved numerically to match target FWHM/f0
at fixed gamma (default gamma=3). This keeps the interface aligned with the
existing CMW implementation.
"""

from functools import lru_cache

import numpy as np

from .cmw import FWHM_TO_SIGMA


def _morse_bp_response(freqs_rad: np.ndarray, f0: float, beta: float, gamma: float) -> np.ndarray:
    """
    Unit-peak Morse response centered at f0 for positive frequencies.

    H(w) = (w/f0)^beta * exp((beta/gamma) * (1 - (w/f0)^gamma))
    """
    w = np.asarray(freqs_rad, dtype=float)
    x = np.maximum(w / max(f0, 1e-12), 1e-18)
    log_h = beta * np.log(x) + (beta / gamma) * (1.0 - np.power(x, gamma))
    h = np.exp(log_h)
    h[w <= 0] = 0.0
    return h


@lru_cache(maxsize=1024)
def _relative_fwhm_ratio(beta: float, gamma: float) -> float:
    """Return FWHM/f0 for normalized Morse response at given beta,gamma."""
    x = np.logspace(-4, 1, 20000)
    y = _morse_bp_response(x, f0=1.0, beta=beta, gamma=gamma)

    # Peak is at x=1 for this parameterization.
    i_peak = int(np.argmin(np.abs(x - 1.0)))
    y_peak = y[i_peak]
    if y_peak <= 0:
        return np.inf
    y_half = 0.5 * y_peak

    left = y[:i_peak]
    right = y[i_peak:]

    li = np.where(left < y_half)[0]
    if len(li) == 0:
        x_low = x[0]
    else:
        j = li[-1]
        x1, x2 = x[j], x[j + 1]
        y1, y2 = y[j], y[j + 1]
        den = y2 - y1
        if abs(den) < 1e-12:
            x_low = 0.5 * (x1 + x2)
        else:
            x_low = x1 + (y_half - y1) * (x2 - x1) / den

    ri = np.where(right < y_half)[0]
    if len(ri) == 0:
        x_high = x[-1]
    else:
        j = i_peak + ri[0]
        x1, x2 = x[j - 1], x[j]
        y1, y2 = y[j - 1], y[j]
        den = y2 - y1
        if abs(den) < 1e-12:
            x_high = 0.5 * (x1 + x2)
        else:
            x_high = x1 + (y_half - y1) * (x2 - x1) / den

    return float(max(0.0, x_high - x_low))


def _solve_beta_for_fwhm_ratio(target_ratio: float, gamma: float) -> float:
    """
    Solve beta such that Morse relative width matches target_ratio=FWHM/f0.
    """
    target_ratio = float(max(target_ratio, 1e-4))

    beta_lo = 0.2
    beta_hi = 200.0
    r_lo = _relative_fwhm_ratio(beta_lo, gamma)
    r_hi = _relative_fwhm_ratio(beta_hi, gamma)

    # If target is outside practical range, clamp to boundary.
    if target_ratio >= r_lo:
        return beta_lo
    if target_ratio <= r_hi:
        return beta_hi

    for _ in range(40):
        beta_mid = 0.5 * (beta_lo + beta_hi)
        r_mid = _relative_fwhm_ratio(beta_mid, gamma)
        if r_mid > target_ratio:
            beta_lo = beta_mid
        else:
            beta_hi = beta_mid
    return 0.5 * (beta_lo + beta_hi)


def morse_freq_domain(
    f0: float,
    fwhm: float,
    fs: float,
    nfft: int,
    analytic: bool = True,
    gamma: float = 3.0,
    beta: float | None = None,
):
    """
    Create a Morse filter response in the frequency domain.

    For f0<=0 (LP/DC case), a Gaussian lowpass fallback is used so the
    interface remains compatible with the CMW pipeline.
    """
    freqs_rad = np.fft.fftfreq(nfft, d=1.0 / fs) * (2.0 * np.pi)
    f0 = float(f0)
    fwhm = float(max(fwhm, 1e-6))

    if f0 <= 0.0:
        # Lowpass fallback compatible with CMW LP design.
        sigma_f = fwhm * FWHM_TO_SIGMA
        if analytic:
            H = np.zeros(nfft, dtype=float)
            pos = freqs_rad >= 0
            H[pos] = np.exp(-0.5 * (freqs_rad[pos] / sigma_f) ** 2)
            H[pos] *= 2.0
            H[0] /= 2.0
        else:
            H = np.exp(-0.5 * (freqs_rad / sigma_f) ** 2)
        return {
            "H": H,
            "freqs_rad": freqs_rad,
            "f0": f0,
            "fwhm": fwhm,
            "gamma": gamma,
            "beta": None,
        }

    if beta is None:
        beta = _solve_beta_for_fwhm_ratio(target_ratio=fwhm / f0, gamma=gamma)
    beta = float(max(beta, 1e-6))

    if analytic:
        H = np.zeros(nfft, dtype=float)
        pos = freqs_rad >= 0
        H[pos] = _morse_bp_response(freqs_rad[pos], f0=f0, beta=beta, gamma=gamma)
        peak = float(np.max(H)) if np.max(H) > 0 else 1.0
        H[pos] /= peak
        # Same one-sided convention as CMW.
        H[pos] *= 2.0
        H[0] /= 2.0
    else:
        H_pos = np.zeros(nfft, dtype=float)
        pos = freqs_rad >= 0
        H_pos[pos] = _morse_bp_response(freqs_rad[pos], f0=f0, beta=beta, gamma=gamma)

        H_neg = np.zeros(nfft, dtype=float)
        neg = freqs_rad <= 0
        H_neg[neg] = _morse_bp_response(-freqs_rad[neg], f0=f0, beta=beta, gamma=gamma)

        H = H_pos + H_neg
        peak = float(np.max(H)) if np.max(H) > 0 else 1.0
        H /= peak

    return {
        "H": H,
        "freqs_rad": freqs_rad,
        "f0": f0,
        "fwhm": fwhm,
        "gamma": gamma,
        "beta": beta,
    }


def apply_morse(
    signal,
    f0: float,
    fwhm: float,
    fs: float,
    analytic: bool = True,
    gamma: float = 3.0,
    beta: float | None = None,
    spacing: int = 1,
    startidx: int = 0,
    interp: str = "none",
):
    """
    Apply Morse filter to a signal via FFT multiplication.

    Output dict matches apply_cmw() for compatibility:
      signal, envelope, phase, phasew, frequency
    """
    signal = np.asarray(signal, dtype=np.float64)
    full_length = len(signal)

    if spacing > 1:
        from src.filters.decimation import decimate_signal, interpolate_output_dict, VALID_METHODS

        if interp not in VALID_METHODS:
            raise ValueError(f"interp must be one of {VALID_METHODS}, got '{interp}'")

        signal_dec, indices = decimate_signal(signal, spacing, offset=startidx + 1)
        fs_dec = fs / spacing

        nyq_dec = np.pi * fs_dec
        if f0 + fwhm / 2 > nyq_dec:
            raise ValueError(
                f"Morse f0={f0:.2f} + fwhm/2={fwhm/2:.2f} = {f0 + fwhm/2:.2f} rad/yr "
                f"exceeds decimated Nyquist={nyq_dec:.2f} rad/yr (spacing={spacing})"
            )

        out = _apply_morse_core(signal_dec, f0=f0, fwhm=fwhm, fs=fs_dec, analytic=analytic, gamma=gamma, beta=beta)
        return interpolate_output_dict(out, indices, full_length, method=interp)

    return _apply_morse_core(signal, f0=f0, fwhm=fwhm, fs=fs, analytic=analytic, gamma=gamma, beta=beta)


def _apply_morse_core(signal, f0: float, fwhm: float, fs: float, analytic: bool, gamma: float, beta: float | None):
    L = len(signal)
    nfft = int(2 ** np.ceil(np.log2(max(L, 2))))

    design = morse_freq_domain(f0=f0, fwhm=fwhm, fs=fs, nfft=nfft, analytic=analytic, gamma=gamma, beta=beta)
    H = design["H"]

    signal_fft = np.fft.fft(signal, n=nfft)
    y_full = np.fft.ifft(H * signal_fft)
    y = y_full[:L]

    out = {}
    if analytic:
        phi = np.angle(y)
        phi_unwrapped = np.unwrap(phi)
        out["signal"] = y
        out["envelope"] = np.abs(y)
        out["phase"] = phi_unwrapped
        out["phasew"] = phi
        out["frequency"] = np.gradient(phi_unwrapped, 1.0 / fs) / (2.0 * np.pi)
    else:
        out["signal"] = y.real
        out["envelope"] = None
        out["phase"] = None
        out["phasew"] = None
        out["frequency"] = None
    return out
