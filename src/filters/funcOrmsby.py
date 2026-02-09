# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 21:37:12 2026
@author: Dime
"""

"""
Ormsby Filter Bank - Functional Implementation
Integrates with existing OrmsbyComplexWithFilterBank.py functions
"""

import numpy as np
import time


def ormsby_filter(
    nw,
    f_edges,
    fs,
    filter_type='bp',
    method='modulate',      # 'subtract' or 'modulate' (for bp only)
    analytic=False         # True → complex quadrature filter 
    #constrain_dc=False      # Optional: enforce H(0)=1 for LP - DF Dropped this else we get half amplitude on low pass
):
    
    """
    (Recommendation Use the modulate method for analytic bandpass filters - it's Cleaner (no wobble)
     More symmetric
     Better phase characteristics
     Standard practice in signal processing
     The subtract method works and can be useful for understanding the construction, but modulate is superior for production use.
    """
    
    if nw % 2 == 0:
        nw += 1
    N = nw
    M = N // 2
    n = np.arange(-M, M + 1)

    #fs = 1.0 / dt  # ← Keep this, remove the hardcoded fs=52
    #fs = 52
    nyq = fs / 2.0

    def norm_freq(f):
        lam = np.array(f) / fs
        if np.any(lam < 0) or np.any(lam > 0.5):
            raise ValueError("Frequencies must be in [0, fs/2]")
        return lam
            
    def ormsby_lp(lc, lr, n, analytic=False):
        n = np.asarray(n)
        h = np.zeros_like(n, dtype=np.float64)
        nz = n != 0
    
        if analytic:
            
            # FIX: Add negative sign for proper Hilbert transform
            if nz.any():
                h[nz] = -(np.sin(2 * np.pi * lr * n[nz]) - np.sin(2 * np.pi * lc * n[nz])) \
                        / (2 * np.pi * (lr - lc) * (np.pi * n[nz])**2)
            h[~nz] = 0.0    
            # if nz.any():
            #     h[nz] = (np.sin(2 * np.pi * lr * n[nz]) - np.sin(2 * np.pi * lc * n[nz])) \
            #              / (2 * np.pi * (lr - lc) * (np.pi * n[nz])**2)
            # h[~nz] = 0.0  # Quadrature part has zero DC component
        else:
            if nz.any():
                h[nz] = (np.cos(2 * np.pi * lc * n[nz]) - np.cos(2 * np.pi * lr * n[nz])) \
                        / (2 * np.pi * (lr - lc) * (np.pi * n[nz])**2)
            h[~nz] = (lc + lr) / np.pi  # Correct center tap from L'Hôpital's rule
        
        return h
    
    if filter_type == 'lp':
        if len(f_edges) != 2:
            raise ValueError("LP requires [f_pass, f_stop]")
        f_pass, f_stop = f_edges
        if not (0 <= f_pass < f_stop <= nyq):
            raise ValueError("For LP: 0 ≤ f_pass < f_stop ≤ Nyquist")
        lam_c, lam_r = norm_freq([f_pass, f_stop])
        h_real = ormsby_lp(lam_c, lam_r, n, analytic=False)
        if analytic:
            h_quad = ormsby_lp(lam_c, lam_r, n, analytic=True)
            h = h_real + 1j * h_quad
        else:
            h = h_real
        if True: #constrain_dc:  DF: always use constrain DC
            # Apply DC constraint correctly
            if analytic:
                # Only normalize by the real part's DC gain
                dc_gain = np.sum(h_real)
                if dc_gain != 0:
                    h = h / dc_gain
            else:
                h = h / np.sum(h)
            
    elif filter_type == 'hp':
        if len(f_edges) != 2:
            raise ValueError("HP requires [f_stop, f_pass]")
        f_stop, f_pass = f_edges
        if not (0 <= f_stop < f_pass <= nyq):
            raise ValueError("For HP: 0 ≤ f_stop < f_pass ≤ Nyquist")
        lam_c, lam_r = norm_freq([f_stop, f_pass])
        lp = ormsby_lp(lam_c, lam_r, n, analytic=False)
        delta = np.zeros_like(lp)
        delta[M] = 1.0
        h_real = delta - lp
        if analytic:
            lpq = ormsby_lp(lam_c, lam_r, n, analytic=True)
            h_quad = -lpq
            h = h_real + 1j * h_quad
        else:
            h = h_real

    elif filter_type == 'bp':
        if len(f_edges) != 4:
            raise ValueError("BP requires [f1, f2, f3, f4]")
        f1, f2, f3, f4 = f_edges
        if not (0 <= f1 < f2 <= f3 < f4 <= nyq):
            raise ValueError("For BP: 0 ≤ f1 < f2 ≤ f3 < f4 ≤ Nyquist")
        lam1, lam2, lam3, lam4 = norm_freq([f1, f2, f3, f4])

        if method == 'subtract':
            lp_high = ormsby_lp(lam3, lam4, n, analytic=False)
            lp_low  = ormsby_lp(lam1, lam2, n, analytic=False)
            
            # Normalize each LP filter to unit DC gain
            dc_high = np.sum(lp_high)
            dc_low = np.sum(lp_low)
            if dc_high != 0:
                lp_high = lp_high / dc_high
            if dc_low != 0:
                lp_low = lp_low / dc_low
                
            h_real = lp_high - lp_low
    
            if analytic:
                lpq_high = ormsby_lp(lam3, lam4, n, analytic=True)
                lpq_low  = ormsby_lp(lam1, lam2, n, analytic=True)
                # For quadrature parts, use same normalization factors
                if dc_high != 0:
                    lpq_high = lpq_high / dc_high
                if dc_low != 0:
                    lpq_low = lpq_low / dc_low
                h_quad = lpq_high - lpq_low
                h = h_real + 1j * h_quad
            else:
                h = h_real

        elif method == 'modulate':
            f0 = (f2 + f3) / 2.0
            lam0 = f0 / fs
            df = (f3 - f2) / 2.0
            lam_c = df / fs
            lam_r = (df + min(f2 - f1, f4 - f3)) / fs
            if lam_r <= lam_c:
                lam_r = lam_c + 0.001
            
            # Use the unified approach for both analytic and non-analytic
            base_lp_real = ormsby_lp(lam_c, lam_r, n, analytic=False)
            dc_base = np.sum(base_lp_real)
            if dc_base != 0:
                base_lp_real = base_lp_real / dc_base
            
            h_real = 2 * base_lp_real * np.cos(2 * np.pi * lam0 * n)
            
            if analytic:
                # base_lp_imag = ormsby_lp(lam_c, lam_r, n, analytic=True)
                # if dc_base != 0:
                #     base_lp_imag = base_lp_imag / dc_base
                # h_quad = 2 * base_lp_imag * np.sin(2 * np.pi * lam0 * n)
                
                h_quad = 2 * base_lp_real * np.sin(2 * np.pi * lam0 * n)
                
                h = h_real + 1j * h_quad
            else:
                h = h_real
                
        else:
            raise ValueError("method must be 'subtract' or 'modulate'")

    elif filter_type == 'bs':
        if len(f_edges) != 4:
            raise ValueError("BS requires [f1, f2, f3, f4]")
        f1, f2, f3, f4 = f_edges
        if not (0 <= f1 < f2 <= f3 < f4 <= nyq):
            raise ValueError("For BS: 0 ≤ f1 < f2 ≤ f3 < f4 ≤ Nyquist")
        lam1, lam2, lam3, lam4 = norm_freq([f1, f2, f3, f4])
        
        # Normalize the low-pass filters for consistency
        lp_low = ormsby_lp(lam1, lam2, n, analytic=False)
        lp_high = ormsby_lp(lam3, lam4, n, analytic=False)
        
        dc_low = np.sum(lp_low)
        dc_high = np.sum(lp_high)
        if dc_low != 0:
            lp_low = lp_low / dc_low
        if dc_high != 0:
            lp_high = lp_high / dc_high
            
        delta = np.zeros_like(lp_low)
        delta[M] = 1.0
        hp_high = delta - lp_high
        h_real = lp_low + hp_high
        
        if analytic:
            lpq_low = ormsby_lp(lam1, lam2, n, analytic=True)
            lpq_high = ormsby_lp(lam3, lam4, n, analytic=True)
            if dc_low != 0:
                lpq_low = lpq_low / dc_low
            if dc_high != 0:
                lpq_high = lpq_high / dc_high
            h_quad = lpq_low - lpq_high
            h = h_real + 1j * h_quad
        else:
            h = h_real
            
    elif filter_type == 'deriv':
        return ormsby_derivative_filter(nw, f_edges[0], f_edges[1], dt, constrain=True)
    else:
        raise ValueError("filter_type must be 'lp', 'hp', 'bp', 'deriv' or 'bs'")

    return h

def funcOrmsby3(nw,w1,w2,w3,w4,datapointsperyr,spacing,type_s):
    tstart = time.time()
    pi = np.pi # lookup once for further use as pi
    if nw % 2 == 0: # if nw was even
        nw=nw+1 # % make it odd
    wr  = w1/datapointsperyr*spacing
    wc  = w2/datapointsperyr*spacing
    wc2 = w3/datapointsperyr*spacing
    wr2 = w4/datapointsperyr*spacing
    wn = (2*pi*datapointsperyr)/spacing


    st = int(((nw-1) / 2)*-1)    # start time = ((oddnw-1) / 2) * -1
    et = int(((nw-1) / 2))       # end time = ((oddnw-1) / 2) 
    cw = int(((nw+1) / 2)-1)    # start weight = ((oddnw+1) / 2) * -1

    tn  = np.arange(st,et+1)

    #tn[cw]=1  #temp fudge to avoid div/o error
    np.seterr(divide='ignore', invalid='ignore') # ignore div 0 errors or div NaN
    
    hA = (np.cos(wc*tn)  - np.cos(wr*tn))  / ((wr-wc)   *tn ** 2*pi)  #% ormsby (eq.5)  #python change ./ to /  and .^2 to **2
    hB = (np.cos(wc2*tn) - np.cos(wr2*tn)) / ((wr2-wc2) *tn ** 2*pi)  #% ormsby (eq.5)
    
    #%chA = (exp(cos(wc*tn*1i))  - exp(cos(wr*tn*1i)))  ./ ((wr-wc)*tn ** 2*pi)    #% ormsby (eq.5) ./ to /  and .^2 to **2
    #%chB = (exp(cos(wc2*tn*1i)) - exp(cos(wr2*tn*1i))) ./ ((wr2-wc2)*tn ** 2*pi) #% ormsby (eq.5)
    
    hAq = (np.sin(wc*tn)  - np.sin(wr*tn))  / ((wr-wc)*tn ** 2*pi)    #% ormsby (eq.5) ./ to /  and .^2 to **2
    hBq = (np.sin(wc2*tn) - np.sin(wr2*tn)) / ((wr2-wc2)*tn ** 2*pi)  #% ormsby (eq.5) 
 
    #tn[cw]=0 # put back 0 in cw   
 
    hA[cw] = (w2/wn) + (w1/wn) #% fill in the centre value for the individual LOW pass
    hB[cw] = (w3/wn) + (w4/wn)
    
    hAq[cw] = 0 #%(w2/wn)+(w1/wn); #% fill in the centre value for the individual LOW pass
    hBq[cw] = 0 #%(w3/wn)+(w4/wn);
    
    sumA = sum(hA)    #%sum individual LP
    sumB = sum(hB)  
    
    sumAq = sum(hAq)  #%sum individual LP -2022.08.04 DF Quad normalzn.
    sumBq = sum(hBq)  
    
    hA = hA / sumA # % divide by sum
    hB = hB / sumB # % divide by sum
    
    #% test sum to 0  (rather than sum to 1)
    hAq = hAq - sumAq / len(hAq)   #np.numel(hAq)
    hBq = hBq - sumBq / len(hBq)   #np.numel(hBq)
    
    if (type_s == "lp"): #% if lowpass
       #% hC = hA*-1; % high pass
       #% hC(cw) = hA(cw)+1; % high pass (add 1 to centre weight after inverting each lowpass value)..
       hC = hA
    
    if (type_s == "bp"):  # %(type_s=="bp") % if bandpass
        hC = hB-hA  #% bandpass
    
    if (type_s == "bpq"): #% if bandpass quad
        hC = (hB-hA) + (1j * (hBq-hAq)) # % bandpass  #python imaginary is 1j not 1i

    # hA is short LP   
    # hB is longer LP  
    # hC is longer LP - shorter (bandpass)  
    # hD is shorter - longer (bandstop)
    # hE is you reverse the sign of lowpass you get high pass
    
    #   a           b                   c             d             e 
    # ---         ------               ---       ---      ---       /---
    #    \              \             /   \         \    /         /
    #     \----          \----    ---/     \---      \--/      ---/
    
    # rounding is divide all weights by sum of all weights sum of lp = 1.
    
    w = hC
    tend = time.time() - tstart 
    print('Ormsby Filter Computed '+type_s+' : ' + str(tend))
    
    return w #return in python

def ormsby_derivative_filter(nw, f_pass, f_stop, fs, constrain=True, min_gain=1e-6):
    """
    First-order derivative filter with smoothing (Ormsby, Appendix B).
    
    Parameters
    ----------
    nw : int
        Number of weights (will be made odd; center = 0)
    f_pass : float
        Passband edge (Hz) — up to here, H(f) ≈ j*2πf
    f_stop : float
        Roll-off termination (Hz) — H(f) = 0 for f > f_stop
    fs: frequency sampling
    constrain : bool
        If True, scale weights so dH/df|₀ = j*2π (correct derivative gain)
    min_gain : float
        Minimum allowed gain before disabling constraint

    Returns
    -------
    h : ndarray (real, anti-symmetric)
    """
    if nw % 2 == 0:
        nw += 1
    N = nw // 2  # half-length
    #fs = 1.0 / dt
    dt = 1/fs
    if not (0 < f_pass < f_stop <= fs / 2):
        raise ValueError("Require 0 < f_pass < f_stop <= Nyquist")
    
    lam_c = f_pass / fs
    lam_R = (f_stop - f_pass) / fs
    
    # Ensure sufficient resolution: need at least a few samples per cycle at f_pass
    min_nw = int(np.ceil(4.0 / (f_pass * dt)))  # ~4 pts per cycle
    if nw < min_nw:
        print(f"Warning: nw={nw} too small for f_pass={f_pass:.3f} Hz. Using nw={min_nw}")
        return ormsby_derivative_filter(min_nw, f_pass, f_stop, fs, constrain, min_gain)
    
    # Index vector: [-N, ..., -1, 0, 1, ..., N]
    k = np.arange(-N, N + 1)
    h = np.zeros_like(k, dtype=np.float64)
    
    nz = k != 0
    k_nz = k[nz]
    
    # Unconstrained weights (Appendix B)
    term1 = np.sin(2 * np.pi * lam_c * k_nz)
    term2 = (lam_c / lam_R) * (
        np.sin(2 * np.pi * (lam_c + lam_R) * k_nz) - term1
    )
    h[nz] = (term1 + term2) / k_nz
    
    # Enforce exact anti-symmetry
    h = 0.5 * (h - h[::-1])
    h[N] = 0.0  # center must be zero
    
    if constrain:
        # Compute gain: G = sum_{k=1}^N 2*k*h[k]  (should equal 0.5 for ideal differentiator)
        k_pos = np.arange(1, N + 1)
        h_pos = h[N + 1:]  # positive lags
        G = 2.0 * np.sum(k_pos * h_pos)
        
        if abs(G) < min_gain:
            print(f"Warning: derivative gain {G:.2e} too small. Returning unconstrained filter.")
            return h
        
        scale = 0.5 / G
        h *= scale
    
    return h

def apply_ormsby_filter(signal, h, mode='reflect', fs=None):
    """
    Apply a real or complex FIR filter to a real signal and return analytic components.
    
    Parameters
    ----------
    signal : array-like, shape (L,)
        Real-valued input (e.g., log-price, returns).
    h : array-like, shape (N,) — N must be odd
        Filter coefficients (real or complex). Assumed zero-phase and centered.
    mode : {'valid', 'zeropad', 'reflect'}
        Boundary handling method.
    fs : float, optional
        fs Sampling interval (seconds). Required if you want instantaneous frequency.
    
    Returns
    -------
    result : dict
        {
            'signal': ndarray,          # filtered output (same length as input)
            'envelope': ndarray or None,
            'phase': ndarray or None,
            'frequency': ndarray or None  # in Hz, if dt provided
        }
    """
    signal = np.asarray(signal, dtype=np.float64)
    h = np.asarray(h)
    
    if h.ndim != 1:
        raise ValueError("Filter h must be 1D")
    if signal.ndim != 1:
        raise ValueError("Signal must be 1D")
    if len(h) % 2 == 0:
        raise ValueError("Filter length must be odd (for centered design)")
    
    L = len(signal)
    N = len(h)
    M = N // 2

    # --- Convolution with boundary handling ---
    if mode == 'valid':
        y_full = np.convolve(signal, h, mode='full')
        start = M
        end = start + L
        y_centered = y_full[start:end]
        y = np.full(L, np.nan, dtype=h.dtype)
        valid_start = M
        valid_end = L - M
        if valid_start < valid_end:
            y[valid_start:valid_end] = y_centered[valid_start:valid_end]

    elif mode == 'zeropad':
        y = np.convolve(signal, h, mode='same').astype(h.dtype)

    elif mode == 'reflect':
        left_pad = signal[:M][::-1] if M > 0 else np.array([])
        right_pad = signal[-M:][::-1] if M > 0 else np.array([])
        signal_padded = np.concatenate([left_pad, signal, right_pad])
        y = np.convolve(signal_padded, h, mode='valid').astype(h.dtype)
        if len(y) != L:
            raise RuntimeError("Padding mismatch in reflect mode")

    else:
        raise ValueError("mode must be 'valid', 'zeropad', or 'reflect'")

    # --- Extract analytic components if complex ---
    out = {'signal': y}

    if np.iscomplexobj(h):
        A = np.abs(y)
        phi = np.angle(y)
        phi_unwrapped = np.unwrap(phi)
        out['envelope'] = A
        out['phase'] = phi_unwrapped
        out['phasew'] = phi
        if fs is not None:
            f_inst = np.gradient(phi_unwrapped, 1/fs) / (2 * np.pi)
            out['frequency'] = f_inst
        else:
            out['frequency'] = None
    else:
        out['envelope'] = None
        out['phase'] = None
        out['frequency'] = None

    return out
