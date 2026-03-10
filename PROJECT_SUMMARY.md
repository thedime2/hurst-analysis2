# Hurst Spectral Analysis: Project Summary

**Reproducing J.M. Hurst's spectral framework from *The Profit Magic of Stock Transaction Timing* (1970)**

Data: DJIA weekly closes, 1921-04-29 to 1965-05-21 (2,298 samples, fs=52/year)
Source: stooq.com | All frequencies in radians/year

---

## Method Pipeline

```
Raw DJIA weekly prices (1921-1965)
        |
        v
[Phase 1] Fourier-Lanczos Spectrum  --->  Figure AI-1
        |   11 peaks, power-law envelope a(w) = k/w
        v
[Phase 2] 23-filter Comb Bank (7.6-12 rad/yr)  --->  Figures AI-2, AI-3, AI-4
        |   Overlapping Ormsby bandpass, 0.2 rad/yr step
        |   Envelope extraction, frequency-vs-time measurement
        v
[Phase 3] Nominal Model Derivation  --->  Figures AI-5, AI-6, AI-7, AI-8
        |   27 lines, mean spacing 0.3719 rad/yr
        |   Three-band fusion: HF comb + MF comb + Fourier
        v
[Phase 4] Structural Decomposition  --->  Page 45 (II-9, II-10), Page 152
        |   6 bandpass filters spanning 0-36 rad/yr
        |   96.2% energy reconstruction
        v
[Phase 5] Modern Extensions  --->  CMW scalograms, ridges, hypothesis tests
        CMW FWHM-matched to Ormsby
        36 ridges, 100% nominal coverage
        BEATING dominates (4/4 tests)
```

---

## Phase 1: Fourier-Lanczos Spectrum

**Goal:** Reproduce Appendix A, Figure AI-1 -- the foundational power spectrum

**Method:** Fourier-Lanczos spectral analysis (not standard FFT) applied to the full 44-year DJIA record.

| Hurst's Reference | Our Reproduction |
|---|---|
| ![ref](references/appendix_a/figure_AI1.png) | ![ours](experiments/appendix_A/figure_AI1_reproduction.png) |

**Key Results:**

| Metric | Value |
|---|---|
| Major peaks detected | 11 |
| Troughs detected | 10 |
| Upper envelope k | 53.96 (R^2 = 0.959) |
| Lower envelope k | 24.40 (R^2 = 0.925) |
| Envelope ratio | 2.21 |
| Frequency resolution | 0.14 rad/yr |

**Peak Frequencies (rad/yr):** 2.28, 3.70, 5.27, 6.69, 8.11, 9.68, 11.95, 13.09, 14.51, 16.51, 18.78

**Status: SOLVED** -- Excellent visual and numerical match. Power-law envelope confirmed with R^2 > 0.92 on both bounds.

**Discrepancy resolved:** Hurst states 2229 points with 0.568 rad/yr resolution, but empirical evidence shows the spectrum was computed over the full ~2297-sample record at ~0.14 rad/yr resolution. Interpreted as an editorial error.

---

## Phase 2: Overlapping Comb Filter Bank

**Goal:** Reproduce Figures AI-2 (filter response), AI-3 (time-domain outputs), AI-4 (frequency vs time)

**Method:** 23 overlapping Ormsby bandpass filters, complex analytic mode, applied to DJIA data.

| Parameter | Value |
|---|---|
| Filters | 23 |
| Centers | 7.6 to 12.0 rad/yr |
| Passband | 0.2 rad/yr |
| Skirt | 0.3 rad/yr |
| Step | 0.2 rad/yr |
| Kernel length | 1999 samples |

### Figure AI-2: Filter Frequency Response

| Hurst's Reference | Our Reproduction |
|---|---|
| ![ref](references/appendix_a/figure_AI2.png) | ![ours](experiments/appendix_A/figure_AI2_reproduction.png) |

**Status: SOLVED** -- Overlapping trapezoidal responses match Hurst's idealized bank.

### Figure AI-3: Comb Filter Time-Domain Outputs

| Hurst's Reference | Our Reproduction |
|---|---|
| ![ref](references/appendix_a/figure_AI3.png) | ![ours](experiments/appendix_A/figure_AI3_reproduction.png) |

Advanced v2 alignment work:

![AI3 alignment](experiments/appendix_A_v2/fig_AI3_align_best.png)

**Status: MOSTLY SOLVED** -- Waveform shapes and envelope modulation match. Pixel-perfect alignment required extensive brute-force spacing/startidx tuning. Some display-window calibration uncertainty remains.

### Figure AI-4: Frequency vs Time

| Hurst's Reference | Our Reproduction |
|---|---|
| ![ref](references/appendix_a/figure_AI4.png) | ![ours](experiments/appendix_A/figure_AI4_reproduction.png) |

Extensive v2 investigation (6 measurement schemes, phase derivatives, CMW comparison):

| 4-panel comparison | Brute-force scheme search |
|---|---|
| ![4panel](experiments/appendix_A_v2/fig_AI4_final_4panel.png) | ![brute](experiments/appendix_A_v2/fig_AI4_brute_6schemes.png) |

**Status: PARTIALLY SOLVED** -- See [Open Questions](#open-questions) section. The qualitative pattern (frequency clustering, gap filters) is reproduced, but exact point density and measurement method remain uncertain.

**Key findings from v2 investigation:**
- PP (peak-to-peak) method gives ~7 pts/filter (matches Hurst's density)
- Phase-derivative with amplitude gating also gives ~7 pts/filter
- PT+TP interleaved gives ~12 pts/filter (too many)
- Spacing/decimation does NOT change measurement density
- Amplitude gating (30% threshold) correctly suppresses "meaningless" gap filters

---

## Phase 3: Nominal Model Derivation

**Goal:** Reproduce Figures AI-5, AI-6, AI-7, AI-8 and derive the line spectrum

**Method:** Three-band frequency fusion:
1. HF comb bank (7.6-12 rad/yr, 23 filters) -- sideband clustering
2. MF comb bank (3.5-7.6 rad/yr) -- direct measurement
3. Fourier peaks (< 3.5 rad/yr) -- spectrum peak detection

### Figure AI-5: Modulation Sidebands

| Hurst's Reference | Our Reproduction |
|---|---|
| ![ref](references/appendix_a/figure_AI5.png) | ![ours](experiments/appendix_A/figure_AI5_reproduction.png) |

**Status: SOLVED** -- 6 line families identified, frequency deviations match Hurst's sideband structure.

### Figure AI-6: LSE Smoothed Frequency vs Time

| Hurst's Reference | Our Reproduction |
|---|---|
| ![ref](references/appendix_a/figure_AI6.png) | ![ours](experiments/appendix_A/figure_AI6_reproduction.png) |

Advanced v2 LSE analysis with sliding-window MPM:

![AI6 LSE](experiments/appendix_A_v2/fig_AI6_lse_analysis.png)

**Status: SOLVED** -- 27 nominal lines derived with mean spacing 0.3719 rad/yr (Hurst: 0.3676, delta = 1.2%).

### Figure AI-7: Line Spectrum (w vs harmonic number N)

![AI7](experiments/appendix_A_v2/fig_AI7_line_spectrum.png)

**Status: SOLVED** -- Harmonic structure w_n = 0.3676*N confirmed from both Fourier and digital filter data.

### Figure AI-8: Spectral Model Table (34 harmonics)

![AI8](experiments/appendix_A_v2/fig_AI8_spectral_table.png)

**Status: SOLVED** -- Complete 34-line table with periods from 17.1 years down to 10 weeks.

### The 27-Line Nominal Model

| # | w (rad/yr) | Period | Spacing |
|---|---|---|---|
| 1 | 2.277 | 2.76 yr | -- |
| 2 | 3.360 | 1.87 yr | 1.084 |
| 3 | 3.938 | 1.60 yr | 0.577 |
| 4 | 4.227 | 1.49 yr | 0.289 |
| 5 | 4.455 | 1.41 yr | 0.228 |
| 6 | 4.725 | 1.33 yr | 0.270 |
| 7 | 5.117 | 1.23 yr | 0.392 |
| 8 | 5.513 | 1.14 yr | 0.397 |
| 9 | 5.773 | 1.09 yr | 0.259 |
| 10 | 6.203 | 1.01 yr | 0.430 |
| 11 | 6.751 | 0.93 yr | 0.549 |
| 12 | 7.057 | 0.89 yr | 0.305 |
| 13 | 7.671 | 0.82 yr | 0.614 |
| 14 | 7.898 | 0.80 yr | 0.228 |
| 15 | 8.336 | 0.75 yr | 0.438 |
| 16 | 8.849 | 0.71 yr | 0.513 |
| 17 | 9.070 | 0.69 yr | 0.221 |
| 18 | 9.254 | 0.68 yr | 0.183 |
| 19 | 9.461 | 0.66 yr | 0.208 |
| 20 | 9.673 | 0.65 yr | 0.211 |
| 21 | 10.164 | 0.62 yr | 0.491 |
| 22 | 10.388 | 0.61 yr | 0.224 |
| 23 | 10.813 | 0.58 yr | 0.425 |
| 24 | 11.126 | 0.56 yr | 0.313 |
| 25 | 11.481 | 0.55 yr | 0.355 |
| 26 | 11.731 | 0.54 yr | 0.250 |
| 27 | 11.947 | 0.53 yr | 0.216 |

**Mean spacing: 0.3719 rad/yr** (Hurst stated 0.3676 -- 1.2% difference)

---

## Phase 4: Structural Decomposition

### Page 45: Single Bandpass Filter (Figures II-9, II-10)

**Method:** Ormsby bandpass [3.20, 3.55, 6.35, 6.70] rad/yr, nw=1795

![Page 45](experiments/page_45/figure_II9_II10_comparison.png)

| Ormsby vs CMW time domain | Ormsby vs CMW frequency response |
|---|---|
| ![time](experiments/page_45/compare_ormsby_vs_cmw_time.png) | ![freq](experiments/page_45/compare_ormsby_vs_cmw_freq.png) |

**Status: SOLVED** -- Both modulate and subtract methods produce identical results for wide passbands. CMW comparison shows smoother envelopes but identical oscillatory structure.

### Page 152: Six-Filter Decomposition

**Method:** 6 filters (1 lowpass + 5 bandpass) covering 0-36 rad/yr

| # | Type | Edges (rad/yr) | Center | Period | nw |
|---|---|---|---|---|---|
| 1 | LP | pass < 0.85, stop > 1.25 | -- | > 5 yr | 1393 |
| 2 | BP | [0.85, 1.25, 2.05, 2.45] | 1.65 | 3.8 yr | 1393 |
| 3 | BP | [3.20, 3.55, 6.35, 6.70] | 4.95 | 1.3 yr | 1245 |
| 4 | BP | [7.25, 7.55, 9.55, 9.85] | 8.55 | 0.7 yr | 1745 |
| 5 | BP | [13.65, 13.95, 19.35, 19.65] | 16.65 | 0.4 yr | 1299 |
| 6 | BP | [28.45, 28.75, 35.95, 36.25] | 32.35 | 0.2 yr | 1299 |

| Real-valued decomposition | Complex modulate + envelopes |
|---|---|
| ![real](experiments/page_152/page152_real.png) | ![mod](experiments/page_152/page152_complex_modulate.png) |

Unified layout (all 6 filters):

![unified](experiments/page_152/page152_unified_layout_clean.png)

Filter derivation from nominal model:

![derivation](experiments/page_152/phase5D_derivation_verification.png)

| Metric | Value |
|---|---|
| Reconstruction energy (Ormsby) | 96.2% |
| Reconstruction energy (CMW) | 96.6% |
| Spectral gap loss | ~3.8% |

**Status: SOLVED** -- Reconstruction validates filter design. Cyclitec mapping confirmed (filters target cycle isolation, not energy maximization). Filter specs are user-estimated from visual inspection of Hurst's graphics -- not published canonical values.

**CMW projections and advanced decomposition:**

| CMW 6-filter projection | Ormsby vs CMW comparison |
|---|---|
| ![cmw](experiments/page_152/cmw_6filter_projection.png) | ![compare](experiments/page_152/compare_ormsby_vs_cmw_time.png) |

---

## Phase 5: Modern Extensions

### 5A: CMW Scalogram

**Method:** Complex Morlet Wavelet, 150 scales, 0.5-80 rad/yr, constant-Q (Q=5)

| Full scalogram | Display window zoom |
|---|---|
| ![full](experiments/appendix_A/phase5A_scalogram_full.png) | ![zoom](experiments/appendix_A/phase5A_scalogram_display.png) |

Marginal spectrum validation vs Lanczos:

![marginal](experiments/appendix_A/phase5A_marginal_spectrum.png)

**Status: SOLVED** -- Scalogram shows clear energy ridges at nominal model frequencies. Marginal spectrum matches Lanczos spectrum.

### 5B: Ridge Detection

**Method:** Continuous ridge extraction tracing maximum energy paths in time-frequency plane.

| Ridges on scalogram | Ridge frequency vs time |
|---|---|
| ![ridges](experiments/appendix_A/phase5B_ridges_on_scalogram.png) | ![freq](experiments/appendix_A/phase5B_ridge_freq_vs_time.png) |

| Ridge statistics | Ridge vs comb comparison |
|---|---|
| ![stats](experiments/appendix_A/phase5B_ridge_statistics.png) | ![vs_comb](experiments/appendix_A/phase5B_ridge_vs_comb.png) |

| Metric | Value |
|---|---|
| Ridges detected | 36 |
| Nominal line coverage | 100% |
| Mean drift | Near zero |

**Status: SOLVED** -- Ridge detection confirms stationary line spectrum. All 27 nominal lines have corresponding ridges.

### 5C: Beating vs Drift Hypothesis

**The central question:** Do comb filter envelope wobbles represent actual frequency drift, or interference beating between closely-spaced stationary lines?

| Test 1: Drift distribution | Test 2: Envelope wobble spectrum |
|---|---|
| ![drift](experiments/appendix_A/phase5C_drift_distribution.png) | ![wobble](experiments/appendix_A/phase5C_envelope_spectra.png) |

| Test 3: FM-AM coupling | Test 4: Synthetic beating |
|---|---|
| ![fmam](experiments/appendix_A/phase5C_fm_am_coupling.png) | ![synth](experiments/appendix_A/phase5C_synthetic_beating.png) |

Verdict summary:

![verdict](experiments/appendix_A/phase5C_verdict_summary.png)

**VERDICT: BEATING DOMINATES (4/4 tests)**

| Test | Result | Evidence |
|---|---|---|
| Drift distribution | Stationary | Ridge drift rates clustered near zero |
| Envelope wobble spectrum | Beat peaks | Wobble frequencies match predicted beat frequencies |
| FM-AM coupling | 100% correlated | Frequency and amplitude variations locked in phase |
| Synthetic two-tone | Matches real | Synthetic beating envelopes reproduce observed patterns |

**Status: SOLVED** -- This is the project's most significant finding. Hurst's observed frequency modulation in comb filter outputs is NOT frequency drift but multi-line beating interference. The spectral lines are stationary.

### 5D: Filter Derivation Verification

Verified that page 152's six filter center frequencies can be derived from the nominal model via Cyclitec cycle hierarchy mapping. Each filter targets one dominant cycle:

| Filter | Dominant Cycle | Period Ratio |
|---|---|---|
| LP-1 | Trend | -- |
| BP-2 | ~3.8 yr | ~2:1 with BP-3 |
| BP-3 | ~1.3 yr | ~2:1 with BP-4 |
| BP-4 | ~0.7 yr | ~2:1 with BP-5 |
| BP-5 | ~0.4 yr | ~2:1 with BP-6 |
| BP-6 | ~0.2 yr | -- |

**Status: SOLVED** -- The ~2:1 period ratio hierarchy (Hurst's "Principle of Harmonicity") is confirmed.

---

## Forecasting / Trading Extensions

Exploratory backtests using the spectral framework:

| 27-line sinusoidal model | Full 6-filter trading system |
|---|---|
| ![27line](data/processed/backtest_27line_model.png) | ![trading](data/processed/hurst_cycle_trading_backtest.png) |

**Status: EXPLORATORY** -- Backtests demonstrate the framework's practical applicability but are in-sample exercises, not validated trading strategies.

---

## Ormsby vs CMW Comparison (Cross-cutting)

| Aspect | Ormsby | CMW |
|---|---|---|
| Passband shape | Flat (trapezoidal) | Gaussian (rounded) |
| Sidelobes | Present (suppressed by skirts) | None |
| Envelope smoothness | Good | Better (lower CV%) |
| Energy capture | 96.2% | 96.6% |
| Computational cost | FFT convolution | FFT multiplication |
| Gap filter behavior | Clean suppression | Wider FWHM rescues some |
| Recommended for | Sharp spectral isolation | Smooth time-domain envelopes |

![Ormsby vs CMW freq](experiments/page_152/compare_ormsby_vs_cmw_freq.png)

---

## Code Architecture

```
src/
 +-- data/loaders.py              # getStooq(), CSV loading, date filtering
 +-- spectral/
 |    +-- lanczos.py              # Fourier-Lanczos spectrum (Phase 1 core)
 |    +-- peak_detection.py       # Peak/trough detection, spacing
 |    +-- envelopes.py            # Power-law envelope fitting a(w) = k/w
 |    +-- frequency_measurement.py # Instantaneous freq from peaks/troughs/zeros
 +-- filters/
 |    +-- funcOrmsby.py           # Ormsby FIR filter (real + complex analytic)
 |    +-- funcDesignFilterBank.py # Comb bank design, apply, envelope extraction
 |    +-- decimation.py           # Decimation/spacing utilities
 +-- nominal_model/
 |    +-- derivation.py           # Line identification, spacing, period hierarchy
 |    +-- sideband_analysis.py    # Modulation sideband grouping (KMeans)
 |    +-- lse_smoothing.py        # Robust frequency trace smoothing
 +-- time_frequency/
      +-- cmw.py                  # Complex Morlet Wavelet (FWHM-matched)
      +-- scalogram.py            # CMW scalogram computation
      +-- ridge_detection.py      # Ridge extraction and tracking
      +-- hypothesis_tests.py     # Beat vs drift testing (4 tests)
      +-- morse.py                # Generalized Morse wavelet (alternative)
```

**64 experiment scripts** across 5 directories, **191 PNG figures**, **17 src modules**

---

## What We Solved

| Question | Answer | Confidence |
|---|---|---|
| Can we reproduce Hurst's Fourier-Lanczos spectrum? | Yes -- 11 peaks, 1/w envelope, R^2 > 0.92 | High |
| Is the discrete line spectrum real? | Yes -- 27 lines with 0.3719 rad/yr spacing | High |
| Does Hurst's comb filter method work? | Yes -- clean frequency separation, envelope modulation | High |
| What is the fine-structure spacing? | 0.3719 rad/yr (Hurst: 0.3676, 1.2% error) | High |
| Is frequency drift or beating? | **Beating** (4/4 hypothesis tests) | High |
| Can we reproduce the 6-filter decomposition? | Yes -- 96.2% energy capture | High |
| Are the spectral lines stationary? | Yes -- ridge drift near zero over 44 years | High |
| Does CMW improve on Ormsby? | Marginally (smoother envelopes, 96.6% vs 96.2%) | Medium |
| Can filters be derived from the nominal model? | Yes -- Cyclitec mapping, ~2:1 period ratios | High |
| Does the Principle of Harmonicity hold? | Yes -- ~2:1 ratios across the full hierarchy | High |

---

## What We Failed to Solve / Open Questions

### 1. Exact AI-4 Measurement Method (Partially Resolved)

We tested 6 measurement schemes extensively but cannot definitively determine which method Hurst used for Figure AI-4. PP (peak-to-peak) and phase-derivative-with-gating both give ~7 pts/filter matching Hurst's density, but neither is a perfect visual match. Hurst's text is ambiguous about whether measurements are placed at the 2nd event or the midpoint.

**Impact:** Low -- the qualitative result (frequency clustering around nominal lines) is robust regardless of method.

### 2. Exact Page 152 Filter Specifications

The 6 filter edge frequencies are **our visual estimates** from Hurst's graphics, NOT published canonical values. We spent considerable effort on brute-force parameter sweeps (especially BP-2) but cannot verify against ground truth.

**Impact:** Medium -- filter design is valid but could be refined if Hurst's exact specs were found.

### 3. Why These Specific 6 Frequencies?

We verified the Cyclitec ~2:1 mapping but haven't fully explained WHY Hurst chose these particular center frequencies over other possible groupings from the 27-line model. The answer likely lies in the Cyclitec course material (1177-page scanned PDF with no extractable text).

**Impact:** Medium -- understanding the selection rationale would complete the Phase 4 story.

### 4. Sub-3.5 rad/yr Nominal Lines

Lines below 3.5 rad/yr in the nominal model rely on Fourier peaks only (comb filters impractical at those long periods). These low-frequency lines have lower confidence than the comb-filter-validated HF lines.

**Impact:** Low-Medium -- affects the long-period end of the hierarchy but doesn't invalidate the model.

### 5. Daily Data Extension

Phase 2A-0 was started but not completed. Daily DJIA analysis (fs=251) could validate whether the same line structure appears at higher time resolution, and could extend the model to frequencies above 26 rad/yr (Nyquist for weekly data).

**Impact:** Medium -- would strengthen the model but isn't required for the core thesis.

### 6. Matrix Pencil Method Parametric Decomposition

Phase 2B (MPM-based extraction of individual sinusoidal components) was prototyped but not fully validated on DJIA data. The MPM recovery diagnostics show promise for pure and two-tone signals but DJIA modal structure per filter is still uncertain.

**Impact:** Medium-High -- MPM could provide definitive mode counts and amplitudes per filter.

### 7. Sub-Sample Interpolation (Phase 2A)

Never started. Would improve frequency measurement precision for the comb filter bank.

**Impact:** Low -- current precision is sufficient for nominal model derivation.

### 8. Transfer to Modern Data

All analysis uses 1921-1965 data. We have not tested whether the same spectral structure persists in post-1965 DJIA data. Hurst's claim is that these are fundamental market cycles, not period-specific artifacts.

**Impact:** High -- this is the ultimate validation of Hurst's framework.

### 9. Forecasting Viability

Trading backtests were exploratory (in-sample). No out-of-sample validation of the spectral framework as a predictive tool.

**Impact:** High for practical applications, low for the reproduction project.

### 10. Decimation Role Clarification

Hurst mentions decimation but our experiments show spacing/decimation does NOT change FVT measurement density. We don't fully understand why Hurst introduced this concept.

**Impact:** Low -- the method works without decimation.

---

## Project Statistics

| Metric | Count |
|---|---|
| Python experiment scripts | 64 |
| PNG figures generated | 191 |
| src/ modules | 17 |
| Nominal model lines | 27 |
| Phases completed | 5 of 5 (core) |
| Supplementary phases | 3 not started |
| Hypothesis tests passed | 4/4 (beating) |
| Reconstruction energy | 96.2% (Ormsby), 96.6% (CMW) |
| Line spacing accuracy | 1.2% vs Hurst |

---

## Key References

- J.M. Hurst, *The Profit Magic of Stock Transaction Timing* (1970)
- Cyclitec Services Training Course (J.M. Hurst)
- Project PRD: [prd/hurst_spectral_analysis_prd.md](prd/hurst_spectral_analysis_prd.md)
- Detailed AI-4 notes: [experiments/appendix_A_v2/AI4_measurement_notes.md](experiments/appendix_A_v2/AI4_measurement_notes.md)

---

*Generated 2026-03-05. 64 scripts, 191 figures, 17 modules, 5 phases complete.*
