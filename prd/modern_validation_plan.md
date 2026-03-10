# Plan: Modern Data Validation & Daily Extension

## What You Need to Provide

### Data Files (download from stooq.com)
1. **S&P 500 daily**: `data/raw/^spx_d.csv` — stooq ticker `^spx`
2. **S&P 500 weekly**: `data/raw/^spx_w.csv` — stooq ticker `^spx`

The DJIA daily/weekly files already exist and cover 1896-2026.

### That's it — everything else is code.

---

## Data Sampling Decision: Trading Days vs Calendar Days

### The Problem

Daily DJIA data has **non-uniform sampling**:
- Pre-1953: ~284 trading days/year (NYSE had Saturday half-sessions)
- 1953-2025: ~252 trading days/year (Mon-Fri only)
- Two large gaps: 137 days (Jul-Dec 1914, WWI closure), 12 days (Mar 1933, bank holiday)
- Hurst era (1921-1965): 12,367 trading days with 1,086 Saturdays (8.8%)

The Lanczos spectrum assumes **uniform spacing**. We have three options:

### Option A: Trading Days Only (Recommended for primary analysis)
- Use data as-is, compute empirical `fs` per era
- **Pre-1953**: fs ≈ 284 days/yr → Nyquist ≈ 446 rad/yr
- **Post-1953**: fs ≈ 252 days/yr → Nyquist ≈ 396 rad/yr
- **Pros**: No interpolation artifacts, uses real prices only
- **Cons**: Non-uniform fs across eras, Saturday-to-Monday gap is ~1.7× a normal gap
- **Mitigation**: Analyze each era separately with its own fs; for cross-era comparison, use overlapping windows

### Option B: Interpolate to Calendar Days (365/yr)
- Fill weekends + holidays with cubic interpolation
- **Pros**: Uniform sampling, fs=365 exactly, Nyquist=573 rad/yr
- **Cons**: ~30% of data points are synthetic; introduces smoothing artifacts at high frequency; interpolated weekends inject false low-frequency energy
- **Not recommended** as primary — useful as sensitivity check only

### Option C: Interpolate to 252/yr (uniform trading days)
- Resample pre-1953 data to 252 days/yr by removing Saturday sessions
- **Pros**: Uniform fs=252 across all eras
- **Cons**: Discards real price information (Saturday closes); requires choosing which Saturday to drop
- **Compromise**: Could work but loses data unnecessarily

### Recommendation

**Use Option A (trading days) as the primary analysis**, with per-era fs computation. Run Option B as a secondary check to confirm results are robust to interpolation. Skip Option C.

For cross-era comparisons, the key metric (line spacing in rad/yr) is unit-independent — it doesn't matter whether fs=252 or fs=284, because we convert to rad/yr at the end.

---

## Experiment Plan

### Phase 6A: Daily DJIA — Hurst Era Reproduction

**Goal**: Reproduce the Hurst analysis on daily data to validate against weekly results and extend to higher frequencies.

**Script**: `experiments/modern_validation/daily_hurst_era.py`

| Step | Method | Expected Result |
|------|--------|----------------|
| 1. Load daily DJIA 1921-04-29 to 1965-05-21 | Trading days only, compute fs empirically (~277 days/yr average across the mixed era) | ~12,100 data points |
| 2. Compute Lanczos spectrum | `lanczos_spectrum(data, 1, fs_daily)` | Same peaks as weekly but extending to ~400 rad/yr |
| 3. Detect peaks (fine mode) | `find_peaks(amp, min_distance=2)` | Should find ~34 harmonics (vs 11 lobes from weekly) |
| 4. Fit 1/ω envelope | `fit_power_law_envelope()` | Same k slope as weekly |
| 5. Map peaks to ω_n = 0.3676·N | `map_to_harmonic()` | Confirm N=1..34 from weekly, discover N=35..~90 |
| 6. Apply 23-filter comb bank (scaled nw) | `design_hurst_comb_bank(nw=auto, fs=fs_daily)` | Same frequency clustering as weekly |
| 7. Plot side-by-side: daily vs weekly spectrum | Overlay comparison | Peaks should align; daily adds high-freq detail |

**Key question answered**: Does the harmonic series ω_n = 0.3676·N extend beyond N=34?

---

### Phase 6B: Modern Weekly DJIA — Transfer Test

**Goal**: Test whether the 0.3676 rad/yr harmonic structure persists in post-1965 data.

**Script**: `experiments/modern_validation/modern_weekly_djia.py`

| Step | Method | Expected Result |
|------|--------|----------------|
| 1. Load weekly DJIA for 3 eras | 1921-1965 (Hurst), 1965-2005, 1985-2025 | ~2300, ~2100, ~2100 samples each |
| 2. Compute Lanczos spectrum per era | Same pipeline as Phase 1 | Side-by-side spectra |
| 3. Detect peaks per era | Fine mode (min_distance=2) | Compare peak locations |
| 4. Fit 1/ω envelopes per era | Power-law fit | Compare slopes k |
| 5. Measure line spacing per era | `compute_line_spacings()` | Is it still ~0.37 rad/yr? |
| 6. Apply comb bank per era | Same 23-filter specs | Does frequency clustering persist? |
| 7. Run 4 beating tests per era | `hypothesis_tests.py` | Are lines still stationary? |
| 8. Compute similarity score | Custom metric (see below) | Quantify transfer quality |

**Hurst Similarity Score** (new function):
```
score = weighted_mean([
    1.0 if |spacing - 0.3676| < 0.05 else 0.0,   # spacing match (40%)
    R² of 1/ω envelope fit,                        # envelope shape (20%)
    fraction of N=1..34 harmonics detected,        # harmonic coverage (20%)
    1.0 if beating_tests >= 3/4 else 0.0,          # stationarity (20%)
])
```

**Key question answered**: Does the spectral structure transfer to modern markets?

---

### Phase 6C: Sliding-Window Spectral Evolution

**Goal**: Track how the spectrum evolves over the full 130-year DJIA record.

**Script**: `experiments/modern_validation/sliding_window_evolution.py`

| Step | Method | Expected Result |
|------|--------|----------------|
| 1. Load full weekly DJIA (1896-2026) | ~6,750 samples | Full record |
| 2. Define 20-year sliding windows, 5-year step | 1896-1916, 1901-1921, ..., 2006-2026 | ~22 windows |
| 3. Per window: Lanczos spectrum + peak detection | Fine mode | Peak frequency lists |
| 4. Per window: measure mean line spacing | `compute_line_spacings()` | Spacing vs time plot |
| 5. Per window: fit 1/ω envelope | Power-law fit | Slope k vs time plot |
| 6. Per window: Hurst similarity score | Combined metric | Score vs time plot |
| 7. Create waterfall/heatmap plot | Spectra stacked by start year | Visual evolution |

**Output**: A single figure showing the spectral "DNA" of the DJIA over 130 years — whether the harmonic structure is permanent, transient, or evolving.

---

### Phase 6D: S&P 500 Cross-Market Comparison

**Goal**: Test whether the 0.3676 rad/yr fundamental appears in a different index.

**Script**: `experiments/modern_validation/cross_market_spx.py`

**Requires**: S&P 500 daily + weekly from stooq (^spx)

| Step | Method | Expected Result |
|------|--------|----------------|
| 1. Load S&P 500 weekly, full history | stooq data (1928-2026?) | As much as available |
| 2. Compute spectrum for overlapping period | 1928-1965 (overlap with Hurst) | Compare to DJIA spectrum |
| 3. Compute spectrum for modern period | 1965-2025 | Does SPX show same structure? |
| 4. Detect peaks and measure spacing | Fine mode | Is spacing ~0.37 rad/yr? |
| 5. Apply comb bank | Same 23-filter specs | Does clustering occur? |
| 6. Compare DJIA vs SPX side-by-side | Overlay spectra, overlay spacing | Correlation analysis |

**Key question answered**: Is the harmonic structure universal (market-wide) or DJIA-specific?

---

### Phase 6E: Daily Modern DJIA — Full Extension

**Goal**: Apply complete pipeline to daily post-1953 data (uniform fs≈252).

**Script**: `experiments/modern_validation/daily_modern_djia.py`

| Step | Method | Expected Result |
|------|--------|----------------|
| 1. Load daily DJIA 1953-2025 | ~18,200 trading days, fs≈252 | Clean uniform sampling |
| 2. Lanczos spectrum | Full range to Nyquist (~396 rad/yr) | High-frequency structure |
| 3. Fine peak detection | min_distance=2 | N=1..~100 harmonics? |
| 4. CMW scalogram (full range) | 200 scales, 0.5-300 rad/yr | Time-frequency evolution |
| 5. Ridge detection | Same algorithm as Phase 5B | Ridge count, drift, coverage |
| 6. Beating tests | 4 tests on new data | Still stationary? |
| 7. Page 152 decomposition | Same 6 filters | Does reconstruction still work? |
| 8. Extended filter bank | Add BP-7 (40-day), BP-8 (20-day) | Capture daily-scale cycles |

**Extended filters** (new, for daily data):

| Filter | Target | Approx Edges (rad/yr) | Period |
|--------|--------|-----------------------|--------|
| BP-7 | 40-day cycle | [50, 52, 64, 66] | ~5 weeks |
| BP-8 | 20-day cycle | [100, 104, 128, 132] | ~2.5 weeks |

These follow the 2:1 cascade from BP-6 (80-day) and are only possible with daily data.

---

### Phase 6F: Calendar-Day Interpolation Sensitivity Check

**Goal**: Verify that results from Option A (trading days) are robust.

**Script**: `experiments/modern_validation/interpolation_sensitivity.py`

| Step | Method | Expected Result |
|------|--------|----------------|
| 1. Take daily DJIA 1953-2025 (trading days) | fs≈252 | Baseline |
| 2. Interpolate to calendar days (365/yr) | Cubic interpolation | fs=365 |
| 3. Compute spectrum for both | Same pipeline | Compare peaks |
| 4. Measure spacing for both | Same pipeline | Should agree within 5% |
| 5. Flag any artifacts from interpolation | High-freq differences | Document |

---

## Implementation Order (Priority)

```
Phase 6B: Modern Weekly DJIA (FIRST — highest impact, easiest)
    ↓  uses only existing weekly data + existing code
Phase 6C: Sliding-Window Evolution (SECOND — compelling visualization)
    ↓  uses only existing weekly data + new wrapper
Phase 6A: Daily Hurst Era (THIRD — validates daily approach)
    ↓  uses existing daily data, needs fs computation
Phase 6E: Daily Modern DJIA (FOURTH — full daily extension)
    ↓  needs Phase 6A validation first
Phase 6D: S&P 500 Cross-Market (FIFTH — needs new data download)
    ↓  needs you to download ^spx from stooq
Phase 6F: Interpolation Sensitivity (LAST — confirmatory only)
```

---

## New Code Needed

### 1. Data Loader Enhancement (`src/data/loaders.py`)

```python
def load_daily_djia(start, end, fill_method=None):
    """Load daily DJIA with computed fs.

    fill_method: None (trading days only), 'calendar' (interpolate to 365/yr)
    Returns: close_prices, fs, date_index
    """
```

### 2. Hurst Similarity Score (`src/validation/similarity.py`)

```python
def hurst_similarity_score(spectrum_results, line_spacing_results,
                           beating_results, reference_spacing=0.3676):
    """Compute 0-1 score measuring how 'Hurst-like' a spectrum is."""
```

### 3. Sliding Window Wrapper (`src/validation/sliding_window.py`)

```python
def sliding_window_analysis(data, fs, window_years=20, step_years=5):
    """Run full Hurst pipeline on overlapping windows.
    Returns: list of per-window results dicts.
    """
```

### 4. Extended Filter Bank for Daily Data (`src/filters/funcDesignFilterBank.py`)

Add BP-7 and BP-8 specs to handle 40-day and 20-day cycles.

### 5. Cross-Era Comparison Plotter (`src/visualization/era_comparison.py`)

Side-by-side spectrum plots, spacing evolution, similarity score timeline.

---

## Expected Outcomes

### If Hurst is Right (cycles are fundamental)
- Phase 6B: Spacing ≈ 0.37 rad/yr in all eras, similarity score > 0.8
- Phase 6C: Waterfall shows stable horizontal ridges across 130 years
- Phase 6D: S&P 500 shows same spacing (universal market structure)
- Phase 6E: Harmonics extend to N=60+ on daily data, 2:1 cascade continues

### If Hurst is Partially Right (structure exists but evolves)
- Phase 6B: Spacing ≈ 0.37 ± 0.10 rad/yr, similarity score 0.5-0.8
- Phase 6C: Waterfall shows ridges that drift or appear/disappear
- Phase 6D: S&P 500 shows similar but not identical spacing
- Phase 6E: Low harmonics (N<20) persist, high harmonics disrupted

### If Hurst is Wrong (artifact of the era)
- Phase 6B: No coherent spacing in modern data, similarity score < 0.3
- Phase 6C: Waterfall shows structure only in 1921-1965 window
- Phase 6D: S&P 500 shows completely different structure
- Phase 6E: Daily spectrum is featureless noise above ~12 rad/yr

---

## Time Estimate

| Phase | New Code | Experiment Script | Figures | Complexity |
|-------|----------|------------------|---------|------------|
| 6B | ~50 lines (similarity fn) | ~200 lines | 4-6 panels | Low |
| 6C | ~80 lines (sliding window) | ~150 lines | 2-3 panels | Low-Medium |
| 6A | ~30 lines (daily loader) | ~250 lines | 6-8 panels | Medium |
| 6E | ~50 lines (extended filters) | ~300 lines | 8-10 panels | Medium |
| 6D | ~20 lines (SPX loader) | ~200 lines | 4-6 panels | Low (needs data) |
| 6F | ~40 lines (interpolation) | ~100 lines | 2 panels | Low |

---

## Questions For You Before Starting

1. **Start with Phase 6B** (modern weekly DJIA)? It's the highest-impact, lowest-effort test.
2. **Download S&P 500 data** from stooq when convenient — needed for Phase 6D.
3. **Daily data approach**: Are you OK with Option A (trading days, per-era fs) as primary? Or do you prefer to interpolate to 365?
4. **How many eras** for the modern test? I suggested 3 (Hurst, 1965-2005, 1985-2025) but could do more overlapping windows.
