# Supplementary PRD: Parametric Spectral Methods for Phase 2 Enhancement

## Motivation

Phase 2 comb filter analysis has revealed three interrelated limitations that prevent
faithful reproduction of Hurst's Figures AI-3 through AI-5:

### Problem 1: Frequency Quantization in Discrete Period Measurement

With weekly-sampled data, peak-to-peak period measurement yields only integer N-week
periods. The resulting frequency quantization omega = 2*pi*52/N produces discrete levels
spaced 0.2-0.4 rad/yr apart in the 7.6-12.0 rad/yr range -- coincidentally almost
identical to the nominal line spacing of ~0.37 rad/yr. This means:

- Figure AI-4 shows horizontal dot bands instead of smooth frequency traces
- The apparent "frequency separation" is partly quantization artifact
- Any period-based measurement on integer-sample data has this floor

Hurst's smooth AI-4 traces imply he used either sub-sample interpolation or a continuous
frequency estimation method (possibly analogue filtering with continuous readout).

### Problem 2: Envelope Wobble from Unresolved Multi-Component Beating

The comb filter envelopes (AI-3) show rapid amplitude modulation not present in Hurst's
smooth dotted envelopes. This beating occurs when two or more closely-spaced spectral
components fall within a single filter's passband (0.8 rad/yr total width). The beat
frequency equals the component separation: if two lines at omega_1 and omega_2 are both
captured, the envelope oscillates at |omega_1 - omega_2|.

This is diagnostic information -- the wobble tells us multiple lines exist within each
band -- but the current Ormsby filter approach cannot decompose them.

### Problem 3: Filter Design Trade-off (Selectivity vs. Energy Capture)

Current design: passband 0.2 rad/yr, skirt 0.3 rad/yr, NW=1999, Q~38.

The ultra-narrow passband rejects legitimate spectral energy. Comparing our AI-2 with
Hurst's original, his triangular filter shapes suggest broader, less selective filters
(likely fewer coefficients). Our 1999-tap filters also have ~38-week transient response,
contributing to edge artifacts and slow envelope settling.

### The Fundamental Insight

The comb filter bank is a **coarse channelizer** -- its job is to isolate spectral
neighborhoods, not to resolve individual lines. The fine resolution should come from
**parametric methods** applied within each channel. This two-stage approach:

1. Broad filter isolates a spectral neighborhood (captures all energy)
2. Parametric method resolves individual components within the neighborhood

...is standard practice in modern spectral analysis and likely what Hurst approximated
with his careful manual measurements.

---

## Proposed Enhancement: Two-Stage Parametric Decomposition

### Stage 1: Broadened Comb Filters (Modified Phase 2)

Redesign the comb bank with wider passbands to capture full spectral neighborhoods:

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| Passband width | 0.2 rad/yr | 0.6-0.8 rad/yr | Capture 2-3 nominal lines per band |
| Skirt width | 0.3 rad/yr | 0.2-0.3 rad/yr | Moderate transition |
| NW (taps) | 1999 | 501-999 | Shorter = broader response, less ringing |
| N_filters | 23 | 12-15 | Wider bands need fewer filters for coverage |
| Overlap | ~75% | ~50% | Sufficient for cross-validation |

Alternative: Keep existing narrow comb bank for validation but add a second "wide comb"
bank specifically for parametric decomposition.

### Stage 2: Parametric Spectral Estimation Within Each Band

Apply parametric methods to the broadened filter outputs to resolve individual components.

---

## Parametric Methods to Implement

### Method 1: Matrix Pencil Method (MPM) -- PRIMARY

**What it does**: Decomposes a signal segment into a sum of K complex exponentials:

```
x(n) = sum_{k=1}^{K} c_k * z_k^n
where z_k = exp((alpha_k + j*omega_k) / fs)
```

Each component has: frequency omega_k, amplitude |c_k|, phase arg(c_k), damping alpha_k.

**Algorithm**:
1. Form Hankel matrix Y from signal samples
2. Compute SVD: Y = U * S * V^H
3. Threshold singular values to determine model order K
4. Compute signal poles z_k from pencil parameter eigenvalues
5. Solve for amplitudes c_k via least squares

**Why it's preferred**:
- SVD provides natural noise rejection and model order selection
- More numerically stable than Prony
- Well-suited to short data segments (100-400 samples)
- Gives both frequency and damping (damping indicates stationarity)

**Implementation plan**:
```
src/parametric/
    __init__.py
    matrix_pencil.py    -- Core MPM algorithm
    sliding_window.py   -- Time-varying MPM analysis
    model_order.py      -- Singular value thresholding, MDL/AIC criteria
```

**Key function signatures**:
```python
def matrix_pencil(x, K=None, pencil_parameter=None, threshold='mdl'):
    """
    Matrix Pencil Method for exponential decomposition.

    Parameters:
        x: ndarray, input signal segment (real or complex)
        K: int or None, number of exponentials (auto-detected if None)
        pencil_parameter: int or None, pencil parameter L (default: len(x)//3)
        threshold: str, model order selection ('mdl', 'aic', 'svd_gap', 'manual')

    Returns:
        frequencies: ndarray, frequencies in natural units (rad/sample)
        amplitudes: ndarray, complex amplitudes
        damping: ndarray, damping factors (negative = decaying)
        residual: float, reconstruction error
    """

def sliding_mpm(x, window_size, hop_size, fs=52, K=None, **kwargs):
    """
    Time-varying MPM analysis via sliding window.

    Returns:
        times: ndarray, center times of each window
        freq_tracks: list of ndarray, frequency tracks over time
        amp_tracks: list of ndarray, amplitude tracks over time
        damping_tracks: list of ndarray, damping tracks over time
    """
```

### Method 2: Prony's Method -- VALIDATION

**What it does**: Same exponential decomposition, different algorithm.

**Algorithm**:
1. Fit linear prediction model of order p to the signal
2. Find roots of characteristic polynomial
3. Select roots inside/near unit circle
4. Solve for amplitudes via least squares

**Why include it**:
- Classical method (1795) -- well-documented and understood
- Different numerical pathway validates MPM results
- Educational: makes the exponential decomposition concept concrete
- Simpler to implement (good for initial prototyping)

**Implementation**:
```python
def prony(x, p, fs=None):
    """
    Prony's method for exponential decomposition.

    Parameters:
        x: ndarray, input signal
        p: int, model order (number of exponentials)
        fs: float, sampling rate

    Returns:
        frequencies, amplitudes, damping, residual
    """
```

### Method 3: ESPRIT -- ADVANCED (Optional)

**What it does**: Estimates frequencies via eigenvalues of a rotation matrix derived
from signal subspace shift-invariance.

**Why include it**:
- Naturally pairs with MPM (uses same SVD decomposition)
- Often gives cleaner frequency estimates than MPM
- Standard reference method in spectral estimation literature

**Implementation**: Lower priority; implement after MPM and Prony are validated.

---

## Sub-Sample Peak Interpolation (Quick Win)

Before implementing full parametric methods, add sub-sample interpolation to the existing
period-based frequency measurement. This partially breaks the quantization floor.

### Parabolic (Quadratic) Interpolation

For each detected peak at integer index n_peak, fit a parabola through the 3 points
(n-1, x[n-1]), (n, x[n]), (n+1, x[n+1]):

```
delta = 0.5 * (x[n-1] - x[n+1]) / (x[n-1] - 2*x[n] + x[n+1])
n_refined = n_peak + delta
```

This gives fractional-sample peak locations, so periods become fractional weeks.

Expected improvement: frequency resolution improves from ~0.3 rad/yr to ~0.05 rad/yr
(approximate 6x improvement for typical SNR).

### Sinc Interpolation (Better)

For higher accuracy, use sinc (Whittaker-Shannon) interpolation around each peak:

```
x_interp(t) = sum_n x[n] * sinc((t - n) / T)
```

Then find the maximum of x_interp via golden section search or Newton's method.

---

## Integration With Existing Pipeline

### Modified Phase 2 Workflow

```
                    DJIA Weekly Data (2298 samples)
                              |
                    +---------+---------+
                    |                   |
            Narrow Comb Bank      Wide Comb Bank
            (existing, 23x)       (new, 12-15x)
                    |                   |
            Existing AI-2,3,4     Parametric Stage
            (for validation)            |
                                  +-----+-----+
                                  |           |
                              MPM/Prony   Sub-sample
                              (per band)  interpolation
                                  |           |
                              Resolved    Refined
                              components  freq traces
                                  |           |
                                  +-----+-----+
                                        |
                                  Enhanced AI-4
                                  (smooth traces)
                                        |
                                  Enhanced AI-5
                                  (true sidebands)
                                        |
                                  Phase 3 validation
                                  (nominal model)
```

### New Figures to Generate

| Figure | Description | Method |
|--------|-------------|--------|
| AI-3b | Filter outputs with smooth envelopes (wider filters) | Wide comb + analytic envelope |
| AI-4b | Frequency vs time with sub-sample interpolation | Parabolic/sinc interpolation |
| AI-4c | Frequency vs time from sliding-window MPM | MPM tracks |
| AI-4d | Component amplitude vs time from MPM | MPM amplitude tracks |
| AI-5b | Enhanced modulation sidebands from MPM decomposition | MPM within grouped bands |
| D-1 | MPM model order (singular values) per band | SVD analysis |
| D-2 | MPM reconstruction quality (residual vs. K) | Validation |
| D-3 | Prony vs MPM frequency comparison | Cross-validation |

### Validation Strategy

1. **Synthetic test**: Generate known multi-component signal, verify MPM recovers
   exact frequencies, amplitudes, and damping rates
2. **Narrow comb cross-check**: MPM frequencies within wide bands should match
   the discrete measurements from existing narrow comb filters
3. **Fourier cross-check**: MPM-detected frequencies should appear as peaks in
   the Phase 1 Lanczos spectrum
4. **Phase 3 consistency**: Enhanced frequency estimates should confirm or refine
   the 0.3719 rad/yr nominal line spacing

---

## Implementation Phases

### Phase 2A-0: Daily Data Reproduction (FIRST PRIORITY)

**Goal**: Reproduce Hurst's smooth AI-4 frequency traces by switching to daily data,
which naturally provides ~5x finer frequency resolution without any new algorithms.

**Rationale**: Comparing our AI-4 output with Hurst's original, overlaying AI-3 bandpass
outputs on top of AI-4 frequency dots reveals that measured frequency points align with
the peaks (blue) and troughs (red) of the filtered oscillation. This is FM-AM coupling
from multi-component beating -- the envelope modulation drives the frequency modulation.
With weekly data, these frequency values snap to quantized integer-period levels
(step ~0.3 rad/yr), producing horizontal bands. With daily data (step ~0.06 rad/yr),
the continuous trajectory between these levels becomes visible -- matching what Hurst drew.

**Key observation**: Weekly quantization step (0.31-0.35 rad/yr) is nearly identical to
the nominal line spacing (0.37 rad/yr). This means the "frequency separation effect" in
our current AI-4 is partly quantization artifact. Daily data resolves this ambiguity.

**Available data**: `data/raw/^dji_d.csv` contains 12,131 daily DJIA points covering the
full Hurst window (1921-04-29 to 1965-05-21), mean spacing 1.33 days (trading days only),
with only 1 gap exceeding 5 days.

**Quantization comparison**:

| Sampling | Period ~32 wk | Freq Step | vs. Line Spacing (0.37) |
|----------|---------------|-----------|-------------------------|
| Weekly (52/yr) | N=32 weeks | 0.31 rad/yr | 84% -- nearly indistinguishable |
| Daily (~252/yr) | N=160 days | 0.06 rad/yr | 16% -- well resolved |
| Daily + interp | N=160.3 days | ~0.01 rad/yr | 3% -- excellent |

**Tasks**:

- [ ] Determine effective daily sampling rate (trading days/year, handle gaps)
- [ ] Re-run existing 23-filter comb bank on daily data (fs~252, scale NW proportionally)
- [ ] Measure peak-to-peak and trough-to-trough periods in days
- [ ] Plot AI-4 with connected lines (not scatter dots) -- match Hurst's drawing style
- [ ] Overlay AI-3 envelope on AI-4 frequency traces to confirm FM-AM coupling
- [ ] Compare daily-data AI-3 envelopes with weekly -- are they smoother?
- [ ] Generate comparison figure: weekly quantized vs daily smooth traces side by side

**Filter scaling for daily data**:
- fs: 52 -> ~252 (trading days/year)
- NW: 1999 -> ~9688 (proportional scaling for same frequency resolution)
  - OR: use NW=4001-5001 for slightly broader response (recommended)
- Passband/skirt widths: unchanged (still in rad/yr units)
- All frequency specifications stay in rad/yr (no conversion needed)

**Success criteria**:
- AI-4 traces show smooth, slowly-varying frequency curves matching Hurst's original
- FM-AM coupling visible: frequency peaks/troughs align with envelope troughs/peaks
- Daily and weekly median frequencies agree within 0.1 rad/yr per filter

### Phase 2A: Sub-Sample Interpolation (1-2 sessions)

**Goal**: Further improve frequency measurement with interpolation on daily data.

- [ ] Implement parabolic sub-sample peak interpolation in `frequency_measurement.py`
- [ ] Add sinc interpolation option for higher accuracy
- [ ] Regenerate AI-4 with interpolated frequencies
- [ ] Compare quantization improvement quantitatively
- [ ] Experiment with shorter/broader filter variants (NW=501, passband=0.5)

### Phase 2B: Matrix Pencil Method (2-3 sessions)

**Goal**: Core parametric decomposition capability.

- [ ] Implement MPM in `src/parametric/matrix_pencil.py`
- [ ] Implement model order selection (MDL criterion + SVD gap)
- [ ] Validate on synthetic signals (known frequencies, amplitudes, damping)
- [ ] Apply to individual comb filter outputs (narrow bank)
- [ ] Implement sliding-window MPM in `src/parametric/sliding_window.py`
- [ ] Generate enhanced AI-4c (smooth frequency tracks from MPM)

### Phase 2C: Prony and Cross-Validation (1-2 sessions)

**Goal**: Validate MPM results with independent method.

- [ ] Implement Prony's method in `src/parametric/prony.py`
- [ ] Compare Prony vs MPM on same filter outputs
- [ ] Document agreement/disagreement and investigate discrepancies
- [ ] Generate cross-validation figure D-3

### Phase 2D: Wide Comb Bank + Full Integration (2-3 sessions)

**Goal**: Complete two-stage pipeline.

- [ ] Design wide comb bank (12-15 filters, 0.6-0.8 rad/yr passband)
- [ ] Apply wide comb to DJIA data
- [ ] Run MPM within each wide band (sliding window)
- [ ] Generate enhanced AI-3b, AI-4c, AI-5b figures
- [ ] Cross-validate against narrow comb and Fourier results
- [ ] Update Phase 3 nominal model with refined frequency estimates
- [ ] Document improvements to nominal line spacing estimate

---

## Technical Notes

### MPM Pencil Parameter Selection

The pencil parameter L controls the trade-off between frequency resolution and noise
sensitivity. Rule of thumb: L = N/3 to N/2 where N is the window length.

For our data:
- Window 200 weeks: L = 67-100
- Window 400 weeks: L = 133-200

### Model Order Selection

Three approaches to determine K (number of exponentials):

1. **SVD gap**: Look for a sharp drop in singular values
2. **MDL (Minimum Description Length)**: Information-theoretic criterion
3. **AIC (Akaike Information Criterion)**: Similar but tends to overestimate K

Recommendation: Use MDL as primary, validate with SVD gap inspection.

### Expected Number of Components Per Band

Based on the nominal line spacing (~0.37 rad/yr) and the current filter total width
(0.8 rad/yr), we expect 1-3 spectral lines per narrow comb filter. With wider filters
(1.5-2.0 rad/yr total), expect 3-6 components per band.

### Damping Interpretation

- alpha_k ~ 0: Stable sinusoidal component (true spectral line)
- alpha_k < 0: Decaying component (transient or edge artifact)
- alpha_k > 0: Growing component (numerical artifact, reject)

Components with significant negative damping are likely:
- Filter transient artifacts (reject)
- Non-stationary amplitude modulation (characterize)
- Noise (reject if amplitude is small)

### Computational Considerations

MPM requires SVD of an (N-L) x L matrix per window per band:
- N=200, L=67: SVD of 133x67 matrix (~0.01s)
- 15 bands x 40 windows = 600 SVDs total (~6s)
- Well within interactive computation budget

---

## Dependencies

All methods use standard numpy/scipy:
- `numpy.linalg.svd` -- SVD for MPM
- `numpy.linalg.eig` -- Eigenvalue decomposition for MPM/ESPRIT
- `numpy.roots` -- Polynomial roots for Prony
- `numpy.linalg.lstsq` -- Amplitude estimation
- `scipy.signal.find_peaks` -- Peak detection (existing)
- `scipy.interpolate` -- Sub-sample interpolation (optional)

No additional packages required.

---

## Success Criteria

1. **AI-4 improvement**: Frequency traces show smooth, continuous variation
   (matching Hurst's original character) rather than quantized horizontal bands
2. **Envelope explanation**: MPM identifies 2-3 components per filter band,
   and their beating frequency matches the observed envelope wobble period
3. **Frequency accuracy**: MPM/Prony frequencies agree within 0.05 rad/yr of
   narrow comb filter median measurements
4. **Nominal model refinement**: Updated line spacing estimate with uncertainty
   bounds (target: 0.37 +/- 0.02 rad/yr)
5. **Reconstruction quality**: MPM model explains >90% of variance in each
   filter output (R^2 > 0.9)

---

## References

- Hua, Y. and Sarkar, T.K., "Matrix Pencil Method for Estimating Parameters
  of Exponentially Damped/Undamped Sinusoids in Noise," IEEE Trans. ASSP, 1990
- Prony, R., "Essai experimental et analytique," J. de l'Ecole Polytechnique, 1795
- Roy, R. and Kailath, T., "ESPRIT - Estimation of Signal Parameters via
  Rotational Invariance Techniques," IEEE Trans. ASSP, 1989
- Hurst, J.M., The Profit Magic of Stock Transaction Timing, Chapter 10 and
  Appendix A, 1970
