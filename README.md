# Hurst Analysis Project

This project is a systematic attempt to **reproduce, understand, validate, and extend**
the spectral market analysis techniques developed by **J.M. Hurst** in  
*The Profit Magic of Stock Transaction Timing*.

The primary focus is on the **Dow Jones Industrial Average (DJIA)** and the material
presented in **Appendix A** and **page 152**, including Fourier–Lanczos analysis,
overlapping comb filters, Ormsby band-pass filters, and the derivation of the
**Nominal Model**.

This is a **research project**, not a trading system (yet).

---

## Core Objectives

1. **Faithful Reproduction**
   - Reproduce key figures and analyses from Hurst’s book
   - Match assumptions, data spacing, frequency units, and methodology
   - Avoid reinterpretation until reproduction is verified

2. **Understanding the Nominal Model**
   - Determine how Hurst moved from spectral observations to a structured model
   - Investigate the relationship between:
     - Fourier fine structure
     - Line spectra
     - Band-pass filter outputs
     - Envelopes and modulation

3. **Validation of Assumptions**
   - Test claims of:
     - Discrete line spectra
     - Slow frequency drift
     - Meaningless frequencies outside filter passbands
   - Quantify reconstruction accuracy and energy conservation

4. **Extension with Modern Tools**
   - Complex Ormsby filters and analytic envelopes
   - Complex Morlet Wavelet (CMW) scalograms
   - Ridge detection and time-frequency tracking
   - Envelope cross-scale influence and beating diagnostics

5. **Application to Modern Data**
   - Apply derived models to recent DJIA data
   - Evaluate stability, transferability, and predictive implications

---

## What This Project Is *Not*

- ❌ A black-box trading strategy  
- ❌ A reinterpretation of Hurst before verification  
- ❌ An optimization or curve-fitting exercise  
- ❌ A claim that Hurst “works” or “doesn’t work” a priori  

---

## Data

- Source: **stooq.com**
- Format: CSV (OHLCV)
- Frequency:
  - Weekly (primary, for reproduction)
  - Daily (for later extensions)
- Data is cached locally to ensure repeatability

No adjustment between adjusted/unadjusted prices is applied;
analysis is performed on the raw series as loaded.

---

## Project Structure

```text
hurst-analysis2/
├── CLAUDE.md                  # Technical conventions and agent instructions
├── README.md
├── prd/
│   ├── hurst_spectral_analysis_prd.md          # Main project PRD
│   ├── supplementary_parametric_methods.md     # Phase 2 enhancement plans
│   ├── page152_filter_derivation.md            # Filter derivation research
│   └── ...
│
├── data/
│   ├── raw/                   # Cached CSV data from stooq
│   └── processed/             # Analysis results and nominal model
│       └── nominal_model.csv  # 27-line period hierarchy
│
├── src/
│   ├── data/
│   │   └── loaders.py                  # getStooq() data loader
│   │
│   ├── spectral/
│   │   ├── lanczos.py                  # Fourier-Lanczos spectrum
│   │   ├── peak_detection.py           # Peak/trough detection
│   │   ├── envelopes.py                # Power-law envelope fitting
│   │   └── frequency_measurement.py    # Instantaneous frequency
│   │
│   ├── filters/
│   │   ├── funcOrmsby.py               # Ormsby filter (real + complex)
│   │   ├── funcDesignFilterBank.py     # Comb bank design
│   │   └── decimation.py              # Decimation utilities
│   │
│   ├── nominal_model/
│   │   ├── sideband_analysis.py        # KMeans line grouping
│   │   ├── lse_smoothing.py            # Frequency trace smoothing
│   │   └── derivation.py              # Line spacing and model builder
│   │
│   └── time_frequency/
│       └── cmw.py                      # Complex Morlet Wavelet (FWHM)
│
├── experiments/
│   ├── appendix_A/            # Phases 1-3: Figures AI-1 through AI-6
│   ├── page_45/               # Phase 4A: Figures II-9 and II-10
│   └── page_152/              # Phase 4B: Six-filter decomposition
│
└── references/
    ├── appendix_a/            # Scanned book figures AI-1 through AI-8
    └── page_152/              # Scanned page 152 filter decomposition
```

---

## Methodological Principles

- **Reproduce first, explain second**
- **Time-local analysis is essential** (markets are quasi-stationary)
- **Overlapping filters > single spectral estimates**
- **Visual diagnostics matter**
- **Every claim must be testable**

---

## Current Status

| Phase | Status | Key Result |
|-------|--------|------------|
| 1. Fourier-Lanczos | COMPLETE | 11 peaks, a(w)=k/w envelope, editorial error in Hurst's stated resolution identified |
| 2. Comb Filter Bank | COMPLETE | 23 filters (7.6-12 rad/yr), frequency clustering confirmed |
| 3. Nominal Model | COMPLETE | 27 lines, spacing 0.3719 rad/yr (1.2% match to Hurst's 0.3676) |
| 4. Page 45 & 152 | COMPLETE | 96.2% reconstruction energy, CMW comparison (96.6%) |
| 5. Extensions | IN PROGRESS | CMW module done, Ormsby vs CMW comparisons done |

---

## Key Results

- **Line Spectrum Confirmed**: DJIA exhibits discrete spectral peaks, not continuous noise
- **Nominal Line Spacing**: 0.3719 rad/yr (Hurst's value: 0.3676, only 1.2% difference)
- **27 Nominal Lines**: Spanning 2.28-11.95 rad/yr (periods 27-144 weeks)
- **Amplitude-Frequency Law**: a(w) = k/w confirmed (all cycles have equal max rate of change)
- **Reconstruction**: 6-filter decomposition captures 96.2% of signal energy

---

## Outstanding Work

1. **Filter Derivation**: How did Hurst select the 6 page-152 filter frequencies from the nominal model?
2. **Phase 5**: CMW scalograms, ridge detection, beating vs drift hypothesis
3. **Parametric Methods**: Matrix Pencil Method, Prony, daily data reproduction
4. **Modern Data**: Transfer framework to post-1965 DJIA data

---

## References

- J.M. Hurst, *The Profit Magic of Stock Transaction Timing* (1970)
- J.M. Hurst, *Cycles Course* (Cyclitec Services, 1973-1975)
- Ormsby, J.F.A., *Design Methods for Sampled Data Filters* (1960)
- Lanczos, C., *Applied Analysis* (1956)
