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
hurst-analysis/
├── README.md
├── prd/
│   └── hurst_spectral_analysis_prd.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── src/
│   ├── data/
│   │   ├── loaders.py
│   │   └── preprocessing.py
│   │
│   ├── spectral/
│   │   ├── lanczos.py
│   │   ├── envelopes.py
│   │   └── spacing.py
│   │
│   ├── filters/
│   │   ├── ormsby.py
│   │   ├── combs.py
│   │   └── reconstruction.py
│   │
│   ├── nominal_model/
│   │   ├── clustering.py
│   │   ├── fitting.py
│   │   └── model.py
│   │
│   ├── time_frequency/
│   │   ├── cmw.py
│   │   ├── ridges.py
│   │   └── comparisons.py
│   │
│   └── visualization/
│       ├── fourier_plots.py
│       ├── filter_outputs.py
│       └── scalograms.py
│
├── experiments/
│   ├── appendix_A/
│   ├── p152_filters/
│   └── modern_extensions/
│
├── notebooks/
│   ├── reproduction_walkthrough.ipynb
│   └── hypothesis_tests.ipynb
│
└── app/
    └── streamlit_app.py
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

- Project structure defined
- Research PRD completed
- Preparing Phase 1:
  - Fourier–Lanczos spectrum
  - Fine-structure detection
  - Envelope fitting

---

## Next Steps

1. Integrate existing data loader (`getStooq`)
2. Stabilize Lanczos spectral analysis
3. Reproduce Appendix A Figure AI-1
4. Implement overlapping Ormsby comb filters
5. Derive and test the Nominal Model

---

## References

- J.M. Hurst, *The Profit Magic of Stock Transaction Timing*
- Appendix A, Figures AI-1 through AI-6
- Page 152 band-pass filter analysis
