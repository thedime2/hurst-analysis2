# Hurst Spectral Analysis Research PRD

## 1. Project Purpose

This project aims to **faithfully reproduce, understand, validate, and extend**
the spectral market analysis framework developed by **J.M. Hurst** in  
*The Profit Magic of Stock Transaction Timing*, with a primary focus on the
**Dow Jones Industrial Average (DJIA)**.

The guiding principle is **reproduction first**, followed by structured
investigation of assumptions and derivations, and only then extension using
modern time–frequency tools.

---

## 2. Primary Objectives

### 2.1 Reproduction Objectives

- Reproduce Appendix A figures (AI-1 through AI-6)
- Reproduce the six-filter decomposition on page 152
- Match:
  - Data spacing and sampling assumptions
  - Frequency units (radians per year)
  - Frequency resolution
  - Visual and numerical characteristics
- Verify reconstruction accuracy and energy conservation

### 2.2 Understanding Objectives

- Determine how Hurst derived the **Nominal Model**
- Clarify the relationship between:
  - Fourier–Lanczos fine structure
  - Line spectra
  - Overlapping band-pass filter outputs
  - Envelopes and modulation sidebands
- Understand Hurst’s classification of certain frequencies as “meaningless”

### 2.3 Extension Objectives

- Investigate finer spectral structure using:
  - Complex Ormsby filters
  - Complex Morlet Wavelets (CMW)
  - Ridge detection and tracking
- Explore:
  - Minimum resolvable line spacing
  - Beating versus slow frequency drift
  - Envelope interactions across scales

---

## 3. Scope and Constraints

### In Scope
- DJIA weekly data (primary reproduction dataset)
- DJIA daily data (secondary, for extensions)
- Fourier–Lanczos spectral analysis
- Overlapping Ormsby comb filter banks
- Nominal Model derivation and testing

### Out of Scope (Initial Phases)
- Trading execution systems
- Risk management optimization
- Parameter optimization for profitability

---

## 4. Data Specifications

- Source: **stooq.com**
- Format: CSV (OHLCV)
- Data is cached locally for repeatability
- Optional preprocessing:
  - Interpolation of missing weekdays
  - Weekend filling to allow 365-period filtering

No adjusted/unadjusted price corrections are applied; analysis is performed on
the raw price series as loaded.

---

## 5. Phase Breakdown

### Phase 1 — Fourier–Lanczos Spectral Ground Truth ✅ COMPLETE

**Goal:** Reproduce Appendix A, Figure AI-1

**Status:** Complete (2026-02-09)

Tasks:
- ✅ Load DJIA weekly data (2298 samples, 1921–1965)
- ✅ Compute Fourier–Lanczos spectrum
- ✅ Express frequency in radians per year
- ✅ Identify broad spectral lobes and fine frequency structure
- ✅ Fit peak-to-peak envelope: a(w) = k / w

Deliverables:
- ✅ `figure_AI1_reproduction.png` — Fourier spectrum with fitted envelopes
- ✅ `data/processed/phase1_results.txt` — Peak/trough frequency lists
- ✅ Envelope fit parameters (upper k=53.96, R²=0.959; lower k=24.40, R²=0.925)

Key results:
- 11 major peaks, 10 troughs detected (min_distance=6, prominence=1.2)
- Upper/lower envelope ratio: 2.21
- Frequency resolution: 0.14 rad/year (full 44-year record)

Implementation:
- `src/spectral/lanczos.py` — Core Fourier–Lanczos spectrum
- `src/spectral/peak_detection.py` — Peak/trough detection with scipy.signal
- `src/spectral/envelopes.py` — Power-law envelope fitting (log-log regression)
- `experiments/appendix_A/phase1_complete.py` — Reproducible pipeline

### Replication Note: Lanczos Spectrum Data Length and Frequency Resolution

During replication of Appendix A (Figure AI-1), an inconsistency was identified
between the textual description in *The Profit Magic of Stock Transaction Timing*
and the empirical requirements of the plotted Fourier–Lanczos spectrum.

Hurst states that the analysis used "2229 data points" providing a frequency
resolution of **0.568 radians per year**. However, when applied to weekly DJIA
data spanning **29 April 1921 through mid-1965** (~44 years, ~2297–2299 samples),
the theoretical frequency resolution is:

\[
\Delta \omega \approx \frac{2\pi}{T} \approx 0.14 \text{ radians per year}
\]

A resolution of **0.568 radians per year corresponds to an effective record
length of approximately 11 years**, which is incompatible with:

- The presence of spectral structure below ~0.6 radians per year in Figure AI-1
- The identification of fine frequency spacing at **0.3676 radians per year**
- The later derivation of the Nominal Model based on sub-rad/year structure

Empirical replication confirms that the Lanczos spectrum shown in Appendix A
must have been computed over the **full ~44-year DJIA record**. The stated
resolution value is therefore interpreted as an **editorial or explanatory
error**, rather than a methodological one.

Importantly, this discrepancy does not invalidate Hurst's conclusions. The fine
spectral structure and minimum line spacing are not inferred from discrete
Fourier bin spacing alone, but from envelope curvature, regular trough spacing,
and subsequent validation using overlapping comb filter banks. These methods
permit reliable inference of sub-bin spectral organization in quasi-stationary
data.

Accordingly, this project treats the Appendix A Fourier–Lanczos spectrum as a
**full-record analysis**, consistent with signal-processing theory and with
Hurst's downstream results in Appendix A and Appendix B.


---

### Phase 2 — Overlapping Comb Filter Analysis ✅ COMPLETE

**Goal:** Reproduce Figures AI-2 through AI-4

**Status:** Complete (2026-02-09)

Tasks:
- ✅ Implement Ormsby band-pass filters (real and complex, modulate method)
- ✅ Construct 23 overlapping combs with uniform 0.2 rad/yr spacing
- ✅ Apply filters to DJIA data (7.6–12.0 rad/yr, nw=1999)
- ✅ Extract instantaneous frequency vs time (peaks, troughs, zero crossings)
- ✅ Identify frequency clustering and gaps

Deliverables:
- ✅ `figure_AI2_reproduction.png` — Idealized + actual comb filter response
- ✅ `figure_AI3_reproduction.png` — Comb filter time-domain outputs with envelopes
- ✅ `figure_AI4_peaks.png` — Frequency vs time (peak-to-peak period method)
- ✅ `data/processed/phase2_results.txt` — Filter specs, envelope amplitudes, freq measurements

Key results:
- 23 filters, centers 7.6–12.0 rad/yr (periods 27–43 weeks)
- Passband 0.2 rad/yr, skirt 0.3 rad/yr, step 0.2 rad/yr
- Complex analytic filters with modulate method
- Clear frequency separation visible in AI-4 reproduction

Implementation:
- `src/filters/funcOrmsby.py` — Real and complex Ormsby filter kernels
- `src/filters/funcDesignFilterBank.py` — Hurst comb bank design, filter application
- `experiments/appendix_A/phase2_figure_AI2.py` — Reproducible pipeline

---

### Phase 3 — Line Spectrum and Nominal Model Derivation ✅ COMPLETE

**Goal:** Reproduce Figures AI-5, AI-6 and derive nominal spacing

**Status:** Complete (2026-02-09)

Tasks:
- ✅ Group comb filter traces into 6 line families (KMeans clustering)
- ✅ Compute modulation sideband envelopes for each line family
- ✅ Design medium-frequency comb bank (15 filters, 3.5–7.8 rad/yr)
- ✅ Apply LSE smoothing (median + Savitzky-Golay) to all frequency traces
- ✅ Aggregate frequency data across three bands (HF comb, MF comb, Fourier)
- ✅ Identify and merge 27 distinct nominal line frequencies
- ✅ Compute line spacings and compare with Fourier fine structure
- ✅ Build nominal period hierarchy table

Deliverables:
- ✅ `figure_AI5_reproduction.png` — Modulation sidebands (6 line families)
- ✅ `figure_AI6_reproduction.png` — LSE frequency vs time analysis (27 lines)
- ✅ `data/processed/phase3_results.txt` — Full nominal model, spacings, validation
- ✅ `data/processed/nominal_model.csv` — Machine-readable period hierarchy

Key results:
- **Mean line spacing: 0.3719 rad/yr** (Hurst: 0.3676, delta = 1.2%)
- 27 nominal lines from 2.28–11.95 rad/yr (periods 27–144 weeks)
- 6 HF line frequencies match Hurst's AI-5 within 0.0–0.3 rad/yr
- 67% of nominal lines match Phase 1 Fourier peaks
- Three-band approach: HF comb (7.6–12), MF comb (3.5–7.8), Fourier (<3.5)

Implementation:
- `src/spectral/frequency_measurement.py` — Peak/trough/zero-crossing freq measurement
- `src/nominal_model/sideband_analysis.py` — Line grouping and sideband envelopes
- `src/nominal_model/lse_smoothing.py` — Robust frequency trace smoothing
- `src/nominal_model/derivation.py` — Line identification, spacing, nominal model
- `experiments/appendix_A/phase3_nominal_model.py` — Reproducible pipeline

---

### Phase 4 — Page 45 & Page 152 Filter Reproduction

**Goal:** Reproduce Chapter II filter demonstrations and the six-filter structural decomposition

**Status:** In progress

#### Part A: Page 45 — Figures II-9 & II-10

Single bandpass filter demonstrating "Time-Persistence of Cyclicality" (II-9)
and "Principle of Variation at Work" (II-10, with amplitude envelope).

Filter specification (rad/year, initial estimates from visual inspection):
- Ormsby bandpass: w1=3.20, w2=3.55, w3=6.35, w4=6.70
- nw = 359 × 5 = 1795
- Display window: 1935-01-01 to 1954-02-01

Comparison of two bandpass construction methods:
1. Modulated (cosine-shift of baseband lowpass) — recommended
2. Subtract (LP_high minus LP_low) — classic approach

Tasks:
- Apply filter using both methods with analytic (complex) mode
- Compare envelope extraction quality between methods
- Visual match to Hurst's Figures II-9 and II-10

Deliverables:
- `experiments/page_45/reproduce_II9_II10.py`
- Comparison plot: `experiments/page_45/figure_II9_II10_comparison.png`

#### Part B: Page 152 — Six-Filter Decomposition

Six-filter structural decomposition of DJIA into frequency bands.

Filter specifications (rad/year, initial estimates from visual inspection):

| # | Type | w1 | w2 | w3 | w4 | Center | Period | nw |
|---|------|----|----|----|----|----|-----|----|
| 1 | LP   | —  | —  | 0.85 | 1.25 | — | >5 yr | 1393 |
| 2 | BP   | 0.85 | 1.25 | 2.05 | 2.45 | 1.65 | 3.81 yr | 1393 |
| 3 | BP   | 3.20 | 3.55 | 6.35 | 6.70 | 4.95 | 1.27 yr | 1245 |
| 4 | BP   | 7.25 | 7.55 | 9.55 | 9.85 | 8.55 | 0.74 yr | 1745 |
| 5 | BP   | 13.65 | 13.95 | 19.35 | 19.65 | 16.65 | 0.38 yr | 1299 |
| 6 | BP   | 28.45 | 28.75 | 35.95 | 36.25 | 32.35 | 0.19 yr | 1299 |

Note: These are initial estimates subject to refinement. The focus is
reproduction first, then understanding WHY these particular filters were chosen.

Three rendering modes:
1. Real-valued filters (no envelopes) — direct comparison to Hurst's figure
2. Complex modulated bandpass with analytic envelopes
3. Complex subtract bandpass with analytic envelopes

Tasks:
- Apply all 6 filters in 3 modes
- Verify summed reconstruction accuracy
- Compare envelope quality between modulate and subtract methods
- Relate bands to nominal model layers (later)
- Explain why apparent spectral gaps do not violate energy conservation (later)

Deliverables:
- `experiments/page_152/reproduce_decomposition.py`
- `experiments/page_152/page152_real.png`
- `experiments/page_152/page152_complex_modulate.png`
- `experiments/page_152/page152_complex_subtract.png`
- Reconstruction error metrics

#### Results (Phase 4)

**Page 45**: Both modulate and subtract methods produce identical filtered signals
and envelopes for this wide passband. Amplitude pattern matches Hurst's Figure II-9.

**Page 152**: All 3 rendering modes verified. Reconstruction captures 96.2% of
signal energy (3.8% residual from spectral gaps between filter passbands).

#### Part C: CMW (Complex Morlet Wavelet) Comparison

FWHM-matched Complex Morlet Wavelets designed in the frequency domain, using
Mike X Cohen's FWHM parameterization (Cohen 2019, NeuroImage 199:81-86).

Matching rule: CMW center frequency = Ormsby (w2+w3)/2, CMW FWHM half-gain
points align with Ormsby skirt midpoints (w1+w2)/2 and (w3+w4)/2.

Module: `src/time_frequency/cmw.py`

Comparison scripts:
- `experiments/page_45/compare_ormsby_vs_cmw.py` — single bandpass
- `experiments/page_152/compare_ormsby_vs_cmw.py` — 6-filter decomposition
- `experiments/appendix_A/compare_comb_ormsby_vs_cmw.py` — 23-filter comb bank

Results: CMW envelopes track similarly to Ormsby envelopes. CMW reconstruction
captures 96.6% energy (vs 96.2% for Ormsby). The Gaussian frequency response
produces smoother envelopes with no sidelobe artifacts, but lacks the flat
passband of the Ormsby trapezoid.

---

### Phase 5 — Modern Extensions

**Goal:** Test and extend Hurst's assumptions

**Status:** In progress (CMW infrastructure complete)

Tasks:
- ✅ Implement CMW frequency-domain design with FWHM matching
- ✅ Produce Ormsby vs CMW comparisons for page 45, 152, and comb bank
- Compute CMW scalograms
- Perform ridge detection
- Compare ridge continuity to filter-derived line spectra
- Test beating versus drift hypotheses
- Analyze envelope cross-scale influence

Deliverables:
- ✅ CMW module: `src/time_frequency/cmw.py`
- ✅ Comparison plots (page 45, page 152, comb bank)
- Scalograms
- Ridge plots
- Comparative diagnostics

---

## 6. Key Hypotheses to Test

1. ✅ **Market spectra consist of discrete, slowly drifting lines** —
   Confirmed: 27 quasi-horizontal lines identified in LSE frequency-vs-time
   analysis (Figure AI-6). Lines show excellent stability over ~275 weeks.
2. ⬜ Fourier fine structure reflects time-averaged drift — Partially addressed:
   mean line spacing (0.3719 rad/yr) matches Hurst's fine structure (0.3676)
   within 1.2%. Full drift analysis deferred to Phase 5.
3. ⬜ "Meaningless" frequencies arise from beating or filter mismatch — Not yet
   tested systematically. Some unmatched nominal lines may correspond to these.
4. ✅ **Nominal model periods emerge from stable line spacing** — Confirmed:
   mean spacing 0.3719 rad/yr with 27 lines spanning 2.28–11.95 rad/yr.
5. ⬜ Envelope behavior is driven by multi-line interference — Deferred to
   Phase 5 (CMW, ridge detection).

---

## 7. Success Criteria

- ✅ Visual and numerical agreement with Hurst's figures (AI-1 through AI-6)
- ✅ Reproducible derivation of nominal spacing (0.3719 vs 0.3676 rad/yr)
- ⬜ Clear explanation of page 152 filter choices (Phase 4)
- ⬜ Demonstrated transferability (or limits) to modern data (Phase 5)

---

## 8. Documentation and Governance

- Every reproduced figure maps to:
  - One script in `experiments/`
  - One or more reusable functions in `src/`
- Reproduction code paths must remain untouched by extensions
- All assumptions must be explicit and testable
