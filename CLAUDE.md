# Claude Code Project Instructions: Hurst Spectral Analysis

## Project Purpose

This is a **research project** to faithfully reproduce, understand, validate, and extend the spectral market analysis framework developed by **J.M. Hurst** in *The Profit Magic of Stock Transaction Timing*.

**Primary focus**: Dow Jones Industrial Average (DJIA), specifically:
- Appendix A (Figures AI-1 through AI-6)
- Page 152 six-filter decomposition
- Derivation of the Nominal Model

**Philosophy**: **Reproduce first, understand second, extend third**

This is NOT a trading system or optimization exercise. It is a systematic attempt to verify Hurst's methodology before extending it with modern tools.

---

## Key Technical Conventions

### Frequency Units
- **All frequencies in radians per year (ω)**
- Conversion: `ω (rad/year) = ω (rad/week) × 52`
- Period: `T (years) = 2π / ω`

### Data Assumptions
- **Weekly spacing**: 52 data points per year
- **Primary time range**: 1921-04-29 to 1965-05-21 (matches Hurst's analysis window)
- **Data source**: stooq.com (cached locally in `data/raw/`)
- **No price adjustments**: Analysis uses raw Close prices as loaded

### Spectral Method
- **Fourier-Lanczos spectrum** (not standard FFT)
- Implemented in [src/spectral/lanczos.py](src/spectral/lanczos.py)
- Returns: `w, wRad, cosprt, sinprt, amp, phRad, phGrad`

---

## Important Technical Notes

### Lanczos Resolution Discrepancy
The Lanczos spectrum implementation includes a detailed replication note (see [src/spectral/lanczos.py](src/spectral/lanczos.py) docstring).

**Summary**: Hurst's text states 2229 data points with 0.568 rad/year resolution, but empirical evidence shows the spectrum must have been computed over the **full ~44-year record** (~2297 samples) with actual resolution ~0.14 rad/year. This is consistent with:
- Low-frequency content in Figure AI-1
- Fine frequency spacing at ~0.3676 rad/year
- Subsequent Nominal Model derivation

This is interpreted as an **editorial error**, not a methodological one. Fine spectral structure is inferred through envelope curvature, trough regularity, and overlapping comb filter validation, not from discrete Fourier bin spacing alone.

### "Meaningless Frequencies"
The PRD and Hurst's work mention certain frequencies as "meaningless" - these are NOT errors but hypotheses to test:
- May arise from beating between closely-spaced lines
- May represent filter mismatch artifacts
- May indicate sidebands of amplitude modulation

Do NOT dismiss or filter these until the hypothesis is tested in Phase 2-3.

---

## Code Organization

```
hurst-analysis2/
├── src/
│   ├── data/          # Data loaders and preprocessing
│   │   └── loaders.py # getStooq() function
│   │
│   ├── spectral/      # Core spectral analysis algorithms
│   │   ├── lanczos.py         # Fourier-Lanczos spectrum
│   │   ├── peak_detection.py  # Peak/trough detection
│   │   └── envelopes.py       # Envelope fitting (a(ω) = k/ω)
│   │
│   ├── filters/       # Ormsby filters, comb banks (Phase 2)
│   ├── nominal_model/ # Nominal Model derivation (Phase 3)
│   ├── time_frequency/ # CMW freq-domain FWHM design (Phase 5)
│   │   └── cmw.py             # ormsby_spec_to_cmw_params, apply_cmw, apply_cmw_bank
│   └── visualization/ # Plotting utilities
│
├── experiments/
│   ├── appendix_A/    # Reproduction scripts for Appendix A figures
│   │   ├── test_lanczos_djia.py    # Basic spectrum test
│   │   ├── phase1_complete.py      # Phase 1 deliverables
│   │   └── compare_comb_ormsby_vs_cmw.py  # Comb bank CMW comparison
│   ├── page_45/       # Page 45 Figures II-9/II-10
│   │   ├── reproduce_II9_II10.py          # Ormsby modulate vs subtract
│   │   └── compare_ormsby_vs_cmw.py       # Ormsby vs CMW comparison
│   └── page_152/      # Page 152 six-filter decomposition
│       ├── reproduce_decomposition.py     # 3 rendering modes
│       └── compare_ormsby_vs_cmw.py       # Ormsby vs CMW comparison
│
├── data/
│   ├── raw/           # Cached CSV data from stooq
│   └── processed/     # Analysis results, deliverables
│
├── prd/
│   └── hurst_spectral_analysis_prd.md  # Full project requirements
│
├── references/        # Reference materials from Hurst's book
│   ├── appendix_a/    # Appendix A figures (AI-1 through AI-8)
│   └── page_152/      # Page 152 filter decomposition
│
└── README.md          # Project overview
```

---

## Reference Materials from Hurst's Book

The project includes reference figures extracted from *The Profit Magic of Stock Transaction Timing*.
These are used to validate reproductions and understand Hurst's original analysis.

**Location**: [references/](references/)

### Appendix A Figures
Located in [references/appendix_a/](references/appendix_a/)

| Figure | File | Description |
|--------|------|-------------|
| AI-1 | `figure_AI1.png` | Fourier-Lanczos spectrum with power-law envelope |
| AI-2 | `figure_AI2.png` | Overlapping comb filter bank outputs |
| AI-3 | `figure_AI3.png` | Instantaneous frequency vs time (comb filter results) |
| AI-4 | `figure_AI4.png` | Line spectrum derivation |
| AI-5 | `figure_AI5.png` | Frequency clustering and sideband structure |
| AI-6 | `figure_AI6.png` | Nominal model and period hierarchy |
| AI-7 | `figure_AI7.png` | Extended frequency analysis |
| AI-8 | `figure_AI8.png` | Reconstruction validation |

**Key Reference Points:**
- Figure AI-1 shows the Fourier-Lanczos spectrum with fitted envelopes a(ω) = k/ω
- Fine structure spacing at ~0.3676 rad/year (period ≈ 17.1 years)
- Peak-to-peak amplitude envelope on log-log axes

### Page 152: Six-Filter Decomposition
Located in [references/page_152/](references/page_152/)

| Element | File | Description |
|---------|------|-------------|
| Filter decomposition | `filter_decomposition.png` | Page 152 structural decomposition into 6 filters |

**Key Reference Points:**
- Low-pass and band-pass filter specifications
- Frequency ranges and filter responses
- Reconstruction accuracy and energy conservation

---

## Using Reference Materials in Development

**When reproducing Phase N figures:**
1. Open the corresponding reference figure from `references/`
2. Compare your generated figure visually and numerically
3. Document any differences in code comments
4. If discrepancies exist, investigate:
   - Data source differences (we use modern stooq data)
   - Methodological interpretation
   - Parameter values and assumptions

**Example comparison:**
```python
# Phase 1: Figure AI-1 Reproduction
# Reference: references/appendix_a/figure_AI1.png
# Our reproduction: experiments/appendix_A/figure_AI1_reproduction.png
#
# Comparison notes:
# - Envelope fits well to data
# - Fine structure visible at ~0.73 rad/year spacing (vs Hurst's 0.3676)
# - May indicate different data period or sampling
```

---

## Project Phases

### Phase 1: Fourier-Lanczos Spectral Ground Truth (COMPLETE)
**Goal**: Reproduce Appendix A, Figure AI-1

### Phase 2: Overlapping Comb Filter Analysis (COMPLETE)
**Goal**: Reproduce Figures AI-2 through AI-5

### Phase 3: Line Spectrum and Nominal Model (COMPLETE)
**Goal**: Reproduce Figure AI-6 and derive nominal spacing

### Phase 4: Page 45 & Page 152 Filter Reproduction (COMPLETE)
**Goal**: Reproduce six-filter structural decomposition
- ✅ Page 45 reproduction (modulate and subtract methods)
- ✅ Page 152 six-filter decomposition (3 rendering modes, 96.2% energy)
- ✅ CMW comparison (96.6% energy)
- ✅ Explained WHY these 6 filter frequencies were chosen (Phase 7, Part 15)

### Phase 5: Modern Extensions (COMPLETE)
**Goal**: CMW scalograms, ridge detection, hypothesis testing
- ✅ CMW frequency-domain FWHM design (`src/time_frequency/cmw.py`)
- ✅ Ormsby vs CMW comparisons for page 45, page 152, and comb bank
- ✅ Spacing/startidx integration into core APIs (`funcOrmsby.py`, `cmw.py`, `funcDesignFilterBank.py`)
- ✅ CMW scalograms (`src/time_frequency/scalogram.py`)
- ✅ Ridge detection and tracking (`src/time_frequency/ridge_detection.py`)
- ✅ Beating vs drift hypothesis testing (`src/time_frequency/hypothesis_tests.py`) -- **BEATING dominates (4/4 tests)**
- ✅ Page 152 filter derivation from nominal model -- Cyclitec mapping verified
- ✅ Algorithmic filter derivation from spectral troughs -- 98.1% reconstruction
- ✅ CMW envelope analysis of derived filters -- inter-cycle coupling confirmed

### Phase 6: Modern Data Validation (COMPLETE)
**Goal**: Confirm harmonic structure across 130 years and multiple indices
- ✅ 6A: Daily DJIA 1921-1965 — spacing=0.3675, harmonics to N=56, R²=0.9998
- ✅ 6B: 3 DJIA eras + 2 SPX periods — spacing within 0.7% of 0.3676, R²≥0.999
- ✅ 6C: 130-year sliding window — 23 windows, spacing mean=0.3668, std=0.0015
- ✅ 6E: Daily modern DJIA & SPX 1953-2025 — harmonics to N=79, R²=0.9999
- **Verdict**: Harmonic structure is a permanent, market-wide phenomenon

### Phase 7: Unified Theory Synthesis (COMPLETE)
**Goal**: Explain the complete framework from spectrum to trading filters
- ✅ 6-filter overlay on daily+weekly spectrum (energy partition 87/10/2/0.5/0.3/0.1%)
- ✅ Trough dividers as natural group boundaries (half-integer harmonic indices)
- ✅ Modulation model: 3 mechanisms explaining 1/w envelope
- ✅ Inter-cycle coupling: envelope correlations 0.3-0.6, F4 leads F2 by 14 weeks
- ✅ Algorithmic filter derivation from spectrum alone (98.1% reconstruction)
- ✅ Cross-validation: 8 periods, 2 indices — w0 universal
- See: `prd/hurst_unified_theory_v2.md`

### Phase 10: Automated Nominal Model Pipeline (COMPLETE)
**Goal**: End-to-end derivation from raw prices to nominal model
- ✅ 10-stage pipeline in `src/pipeline/` (4 modules)
- ✅ 3-method w0 estimation with sub-harmonic correction
- ✅ Narrowband CMW resolves individual harmonics (79/79 confirmed in daily data)
- ✅ 4-test validation suite (spectral, reconstruction, cycle count, envelope)
- ✅ Automated 6-filter design from trough dividers
- ✅ 6 visualization types including 3D time-frequency surfaces
- See: `prd/nominal_model_pipeline.md`, `experiments/pipeline/`

---

## Coding Conventions

### Style
- Follow existing code style in [src/spectral/lanczos.py](src/spectral/lanczos.py)
- Use numpy for array operations
- Docstrings should reference Hurst's book and page numbers where relevant
- Comment any deviations from Hurst's methodology

### Naming
- Use `omega` or `w` for frequency in radians
- Use `amp` for amplitude
- Use `dataspacing` for sample interval
- Use `datapointsperyr` for sampling rate (typically 52 for weekly)

### Function Signatures
- Return multiple values as tuples (following lanczos_spectrum pattern)
- Accept numpy arrays, not lists
- Use keyword arguments with defaults for optional parameters

### Documentation
- Every reproduction script should have:
  - Header comment stating which figure/analysis it reproduces
  - Data range used
  - Key assumptions
  - Reference to Hurst's original (page number, figure number)

---

## Reproduction Requirements

When reproducing Hurst's figures:
1. **Visual match**: Generated figure should match Hurst's original in:
   - Axis ranges and scales (linear/log)
   - Peak/trough locations
   - Overall spectral envelope shape

2. **Numerical match**: Key parameters should be within 5-10% unless explained:
   - Envelope slopes
   - Peak frequencies
   - Filter center frequencies

3. **Documentation**: Any differences must be:
   - Noted in code comments
   - Explained as either:
     - Data difference (we use updated stooq data)
     - Interpretation difference (document reasoning)
     - Calculation difference (explain why)

4. **Testability**: Every claim about the spectrum should be:
   - Quantifiable (not just "looks similar")
   - Reproducible (script can be re-run)
   - Verifiable (compare to Hurst's published values)

---

## Testing and Verification

### Before Committing Code
1. Run the affected script to verify it executes without errors
2. Visually inspect output for reasonableness
3. Check that results are consistent with PRD expectations
4. Ensure no hardcoded paths (use relative paths or os.path.join)

### Phase Completion Criteria
Each phase is complete when:
- All deliverables match PRD specification
- Visual output matches Hurst's original figure
- Results are saved to `data/processed/`
- Script is documented and can be re-run deterministically

---

## Common Patterns

### Loading DJIA Data
```python
from src.data import getStooq
import pandas as pd

# Load weekly DJIA data
df = pd.read_csv('data/raw/^dji_w.csv')
df.Date = df.Date.apply(pd.to_datetime)

# Filter to Hurst's time window
df_hurst = df[df.Date.between('1921-04-29', '1965-05-21')]
close_prices = df_hurst.Close.values
```

### Computing Lanczos Spectrum
```python
from src.spectral import lanczos_spectrum

dataspacing = 1  # Weekly data, no gaps
datapointsperyr = 52

w, wRad, cosprt, sinprt, amp, phRad, phGrad = lanczos_spectrum(
    close_prices, dataspacing, datapointsperyr
)

# Convert to rad/year
omega_yr = w * 52
```

### Plotting Spectrum
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(omega_yr, amp, 'b.-', markersize=4)
plt.xscale('linear')
plt.yscale('log')
plt.xlim(-0.1, 22)
plt.xlabel("Angular Frequency ω (radians per year)")
plt.ylabel("Amplitude (log scaled)")
plt.title("Lanczos Spectrum – DJIA 1921–1965")
plt.grid(True)
plt.show()
```

---

## Current Development Focus

**All core phases COMPLETE** (Phases 1-7, 10). The project has achieved its primary goal: faithful reproduction, validation, and extension of Hurst's spectral framework.

**Key milestone (March 2026)**: Narrowband CMW resolves **79 individual harmonics** (N=2 to N=80) in daily DJIA data, extending the Nominal Model far beyond Hurst's 27-34 comb-derived lines.

**Next steps** (not yet started):
- Improve reconstruction R² by using CMW-confirmed lines (target >0.70)
- Run pipeline on DJIA 1965-2025 and SPX 1985-2025 (multi-period validation)
- Stage 10 CMW envelope analysis: modulation periods, inter-filter coupling
- Investigate whether harmonics extend beyond N=80 with higher-frequency daily data
- Backtest: can CMW-derived amplitudes and phases predict turning points?

**Supplementary Work** (deferred):
- Sub-sample interpolation (Phase 2A)
- Matrix Pencil Method parametric decomposition (Phase 2B)

---

## References

- J.M. Hurst, *The Profit Magic of Stock Transaction Timing* (1970)
- Project PRD: [prd/hurst_spectral_analysis_prd.md](prd/hurst_spectral_analysis_prd.md)
- Unified Theory: [prd/hurst_unified_theory_v2.md](prd/hurst_unified_theory_v2.md)
- Pipeline PRD: [prd/nominal_model_pipeline.md](prd/nominal_model_pipeline.md)
- Insights Summary: [prd/insights_summary.md](prd/insights_summary.md)
- Project README: [README.md](README.md)

---

## Notes for Future Sessions

- The page 152 six-filter specifications in `experiments/page_152/reproduce_decomposition.py` are user-estimated from visual inspection of Hurst's graphics, NOT published values from the book. The automated filter design in `src/pipeline/filter_design.py` now derives these from trough dividers.

- The Cyclitec PDF (`Dropbox/ebooks/jm-hurst-cycles-coursecyclitec-services-training-course_compress.pdf`) is a 1177-page scanned document with no extractable text. OCR would be needed for automated searching.

- The original 27-line nominal model in `data/processed/nominal_model.csv` spans 2.28-11.95 rad/yr (weekly comb bank + Fourier peaks). The narrowband CMW model extends this to 79 lines spanning 0.71-28.58 rad/yr.

- The `src/pipeline/` module provides `derive_nominal_model()` for single-call pipeline execution and `design_narrowband_cmw_bank()` for individual harmonic resolution.

- The project uses Python 3.13+ - ensure all code is compatible
