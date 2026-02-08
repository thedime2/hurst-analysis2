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

### Phase 1 — Fourier–Lanczos Spectral Ground Truth

**Goal:** Reproduce Appendix A, Figure AI-1

Tasks:
- Load DJIA weekly data
- Compute Fourier–Lanczos spectrum
- Express frequency in radians per year
- Identify:
  - Broad spectral lobes
  - Fine frequency structure
- Fit peak-to-peak envelope:
  - \( a(\omega) = k / \omega \)

Deliverables:
- Fourier spectrum plot
- Peak and trough frequency lists
- Envelope fit parameters

## Replication Note: Lanczos Spectrum Data Length and Frequency Resolution

During replication of Appendix A (Figure AI-1), an inconsistency was identified
between the textual description in *The Profit Magic of Stock Transaction Timing*
and the empirical requirements of the plotted Fourier–Lanczos spectrum.

Hurst states that the analysis used “2229 data points” providing a frequency
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

Importantly, this discrepancy does not invalidate Hurst’s conclusions. The fine
spectral structure and minimum line spacing are not inferred from discrete
Fourier bin spacing alone, but from envelope curvature, regular trough spacing,
and subsequent validation using overlapping comb filter banks. These methods
permit reliable inference of sub-bin spectral organization in quasi-stationary
data.

Accordingly, this project treats the Appendix A Fourier–Lanczos spectrum as a
**full-record analysis**, consistent with signal-processing theory and with
Hurst’s downstream results in Appendix A and Appendix B.


---

### Phase 2 — Overlapping Comb Filter Analysis

**Goal:** Reproduce Figures AI-2 through AI-5

Tasks:
- Implement Ormsby band-pass filters (real and complex)
- Construct overlapping combs with uniform frequency spacing
- Apply filters to DJIA data
- Extract instantaneous frequency vs time
- Identify frequency clustering and gaps
- Shade modulation sidebands

Deliverables:
- Filter bank definitions
- Time-domain filter outputs
- Frequency-vs-time plots
- Sideband grouping visualizations

---

### Phase 3 — Line Spectrum and Nominal Model Derivation

**Goal:** Reproduce Figure AI-6 and nominal spacing derivation

Tasks:
- Aggregate frequency-vs-time data across filters
- Perform least-squares line fitting
- Estimate minimum line spacing
- Compare with Fourier fine structure
- Reconstruct nominal period hierarchy

Deliverables:
- Line spacing vs index plot
- Derived nominal periods
- Comparison with Hurst’s published tables

---

### Phase 4 — Page 152 Filter Decomposition

**Goal:** Reproduce the six-filter structural decomposition

Tasks:
- Reconstruct low-pass and band-pass filters from page 152
- Verify summed reconstruction accuracy
- Relate bands to nominal model layers
- Explain why apparent spectral gaps do not violate energy conservation

Deliverables:
- Filter definitions
- Component plots
- Reconstruction error metrics

---

### Phase 5 — Modern Extensions

**Goal:** Test and extend Hurst’s assumptions

Tasks:
- Compute CMW scalograms
- Perform ridge detection
- Compare ridge continuity to filter-derived line spectra
- Test beating versus drift hypotheses
- Analyze envelope cross-scale influence

Deliverables:
- Scalograms
- Ridge plots
- Comparative diagnostics

---

## 6. Key Hypotheses to Test

1. Market spectra consist of discrete, slowly drifting lines
2. Fourier fine structure reflects time-averaged drift
3. “Meaningless” frequencies arise from beating or filter mismatch
4. Nominal model periods emerge from stable line spacing
5. Envelope behavior is driven by multi-line interference

---

## 7. Success Criteria

- Visual and numerical agreement with Hurst’s figures
- Reproducible derivation of nominal spacing
- Clear explanation of page 152 filter choices
- Demonstrated transferability (or limits) to modern data

---

## 8. Documentation and Governance

- Every reproduced figure maps to:
  - One script in `experiments/`
  - One or more reusable functions in `src/`
- Reproduction code paths must remain untouched by extensions
- All assumptions must be explicit and testable
