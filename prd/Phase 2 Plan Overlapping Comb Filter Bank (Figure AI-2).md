# Phase 2 Plan: Overlapping Comb Filter Bank (Figure AI-2)

## Context

Phase 1 (Fourier-Lanczos spectrum) is complete. Phase 2 reproduces Figures AI-2 through AI-5 from Hurst's Appendix A. This plan focuses on the comb filter bank design and Figure AI-2 reproduction, with infrastructure that enables AI-3 (time-domain outputs), AI-4 (frequency-vs-time), and AI-5 (modulation sidebands) as follow-on work.

Hurst specified 23 overlapping bandpass filters with uniform 0.2 rad/yr spacing, constant bandwidth, and constant skirt widths. The existing `design_ormsby_filter_bank()` uses Q-based spacing and is **not suitable** for this; we need a new dedicated function.

The existing `ormsby_filter(method='modulate', analytic=True)` in `funcOrmsby.py` is the right tool for creating each filter kernel.

---

## Hurst's Comb Bank Specification (p. 192)

| Parameter | Value |
|-----------|-------|
| Number of filters | 23 |
| Filter 1 edges (w1,w2,w3,w4) | 7.2, 7.5, 7.7, 8.0 rad/yr |
| Step per filter | +0.2 rad/yr on all 4 edges |
| Filter 23 edges | 11.6, 11.9, 12.1, 12.4 rad/yr |
| Center frequencies | 7.6, 7.8, 8.0, ..., 12.0 rad/yr |
| Passband width (w3-w2) | 0.2 rad/yr (constant) |
| Skirt width (w2-w1 = w4-w3) | 0.3 rad/yr (symmetric, constant) |
| Total span per filter | 0.8 rad/yr |

---

## Implementation Steps

### Step 1: Clean up `src/filters/funcOrmsby.py`

- **Remove** duplicate `apply_filter_bank()` function (lines 18-63) - it's a copy from `funcDesignFilterBank.py` with broken imports
- **Remove** unused imports at top: `matplotlib.pyplot`, `LogNorm`, `pandas` (only `numpy` and `time` are used)
- Functions that remain: `ormsby_filter()`, `funcOrmsby3()`, `ormsby_derivative_filter()`, `apply_ormsby_filter()`

### Step 2: Fix imports in `src/filters/funcDesignFilterBank.py`

Three broken imports reference `OrmsbyComplexWithFilterBank` (the experiment file) instead of `funcOrmsby`:

| Line | Current (broken) | Fix |
|------|-------------------|-----|
| 12 | `from funcOrmsby import ...` | `from .funcOrmsby import ormsby_filter, apply_ormsby_filter` |
| 291 | `from OrmsbyComplexWithFilterBank import ormsby_filter` | Remove (use module-level import) |
| 349 | `from OrmsbyComplexWithFilterBank import apply_ormsby_filter` | Remove (use module-level import) |

Also remove unused import `ormsby_derivative_filter` from line 12.

### Step 3: Create `src/filters/__init__.py`

Make `src/filters/` a proper Python package. Expose key functions:
- From `funcOrmsby`: `ormsby_filter`, `apply_ormsby_filter`, `funcOrmsby3`, `ormsby_derivative_filter`
- From `funcDesignFilterBank`: `design_hurst_comb_bank` (new), `create_filter_kernels`, `apply_filter_bank`, `plot_filter_bank_response`, `create_time_frequency_heatmap`, `print_filter_specs`

### Step 4: Add `design_hurst_comb_bank()` to `funcDesignFilterBank.py`

Insert after `design_ormsby_filter_bank()`, before `create_filter_kernels()`.

**Signature:**
```python
def design_hurst_comb_bank(
    n_filters=23,
    w1_start=7.2,
    w_step=0.2,
    passband_width=0.2,
    skirt_width=0.3,
    nw=1393,
    fs=52
) -> list[dict]:
```

**Logic:** Simple loop generating `w1 = w1_start + i*w_step`, then `w2 = w1 + skirt_width`, `w3 = w2 + passband_width`, `w4 = w3 + skirt_width` for i in 0..22.

**Returns** list of spec dicts with keys: `type`, `f1`-`f4`, `f_center`, `bandwidth`, `skirt_width`, `Q`, `Q_target`, `nw`, `index`, `label`. The Q and Q_target fields ensure compatibility with existing `print_filter_specs()`.

**Frequency convention:** All specs in rad/year (consistent with existing `design_ormsby_filter_bank()`). The downstream `create_filter_kernels()` already divides by 2pi before calling `ormsby_filter()`.

### Step 5: Add `plot_idealized_comb_response()` to `funcDesignFilterBank.py`

Draws Hurst-style trapezoidal filter shapes for visual comparison with Figure AI-2. For each filter: trapezoid vertices at `(f1, 0) -> (f2, 1) -> (f3, 1) -> (f4, 0)`. Optionally overlays actual FFT response from filter kernels for validation.

### Step 6: Create `experiments/appendix_A/phase2_figure_AI2.py`

Main experiment script following Phase 1 pattern:

1. **Config**: Hurst parameters, paths, date ranges
2. **Design**: Call `design_hurst_comb_bank()` with Hurst's exact specs
3. **Build kernels**: `create_filter_kernels(specs, fs=52, filter_type='modulate', analytic=True)`
4. **Plot AI-2**: Idealized + actual frequency response, save as `figure_AI2_reproduction.png`
5. **Load DJIA**: Weekly data, 1921-1965 window
6. **Apply filters**: `apply_filter_bank(close_prices, filters, fs=52, mode='reflect')`
7. **Plot AI-3 preview**: First 10 filter outputs with envelopes (time window ~1940-1946, matching Hurst)
8. **Plot AI-4 preview**: Instantaneous frequency vs time for all 23 filters
9. **Save results**: Structured text to `data/processed/phase2_results.txt`

**Frequency unit note:** `apply_ormsby_filter()` returns instantaneous frequency in cycles/year. Multiply by 2pi to get rad/year for Hurst-style plots.

---

## Files Modified/Created

| File | Action |
|------|--------|
| [src/filters/funcOrmsby.py](src/filters/funcOrmsby.py) | Clean: remove duplicate function + unused imports |
| [src/filters/funcDesignFilterBank.py](src/filters/funcDesignFilterBank.py) | Fix imports, add `design_hurst_comb_bank()`, add `plot_idealized_comb_response()` |
| `src/filters/__init__.py` | **Create**: package init exposing key functions |
| `experiments/appendix_A/phase2_figure_AI2.py` | **Create**: main Phase 2 experiment script |

## Existing Functions Reused

- `ormsby_filter()` in [src/filters/funcOrmsby.py:65](src/filters/funcOrmsby.py#L65) - creates filter kernels
- `apply_ormsby_filter()` in [src/filters/funcOrmsby.py:427](src/filters/funcOrmsby.py#L427) - applies filter, extracts envelope/phase/freq
- `create_filter_kernels()` in [src/filters/funcDesignFilterBank.py:267](src/filters/funcDesignFilterBank.py#L267) - batch kernel creation
- `apply_filter_bank()` in [src/filters/funcDesignFilterBank.py:326](src/filters/funcDesignFilterBank.py#L326) - batch filter application
- `plot_filter_bank_response()` in [src/filters/funcDesignFilterBank.py:374](src/filters/funcDesignFilterBank.py#L374) - actual FFT response plot
- `print_filter_specs()` in [src/filters/funcDesignFilterBank.py:607](src/filters/funcDesignFilterBank.py#L607) - console output

---

## Verification

1. **Filter specs**: Confirm Filter 1 = [7.2, 7.5, 7.7, 8.0] and Filter 23 = [11.6, 11.9, 12.1, 12.4]
2. **Visual match**: Compare `figure_AI2_reproduction.png` against `references/appendix_a/figure_AI2.png`
3. **Passband gain**: Actual FFT response should show ~1.0 gain in passband (within 10%)
4. **Frequency chain**: Verify conversion: rad/yr -> cycles/yr (divide 2pi) -> normalized (divide fs) -> filter kernel
5. **Run script**: Execute `phase2_figure_AI2.py` end-to-end without errors
6. **Filter outputs**: Verify filtered signals show amplitude modulation with reasonable envelopes
7. **Instantaneous freq**: Check that frequency tracks cluster near filter center frequencies
