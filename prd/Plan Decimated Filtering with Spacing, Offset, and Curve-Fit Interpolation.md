# Plan: Decimated Filtering with Spacing, Offset, and Curve-Fit Interpolation

## Context

The current filtering infrastructure (Ormsby and CMW) always operates at the full sampling rate (fs=52 weekly). The user wants to add **decimation** support: apply filters to every Nth data point (with a configurable starting offset), then optionally **interpolate** the gaps using Hurst's 3-point parabola method (page 213) or modern scipy methods. This applies to both Ormsby and CMW filters, including their envelope and phase outputs.

The legacy `funcOrmsby3()` ([funcOrmsby.py:225](src/filters/funcOrmsby.py#L225)) already has a `spacing` parameter that adjusts filter design frequencies, demonstrating the concept. The new implementation separates concerns more cleanly: decimate signal, adjust fs, apply filter, interpolate back.

---

## New Parameters

All three parameters added to filter application functions:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spacing` | int | 1 | Decimation factor. 1 = no decimation (current behavior). N = every Nth sample. |
| `offset` | int | 1 | 1-based starting index (1 through `spacing`). offset=3 with spacing=5 starts at index 2 (0-based). |
| `interp` | str | `'none'` | Gap-filling method: `'none'` (NaN), `'3point'` (Hurst parabola), `'cubic'` (spline), `'linear'` |

---

## Interpolation Methods

### `'none'` — NaN fill
Place computed values at their original indices, NaN everywhere else.

### `'3point'` — Hurst's Page 213 Parabolic Interpolation
For each consecutive **triplet** of computed points (x[i-1], y[i-1]), (x[i], y[i]), (x[i+1], y[i+1]):
- Fit quadratic y = ax^2 + bx + c exactly through all 3 points (Lagrange degree-2)
- Evaluate at all integer positions in the interval
- Overlapping parabolas: for the gap between x[i] and x[i+1], use the parabola centered at i for the first half, centered at i+1 for the second half
- Edge positions: linear extrapolation from nearest 2 computed points

### `'cubic'` — Scipy cubic spline
`scipy.interpolate.interp1d(kind='cubic', fill_value='extrapolate')`

### `'linear'` — Linear interpolation
`scipy.interpolate.interp1d(kind='linear', fill_value='extrapolate')`

### Phase-aware handling
Wrapped phase (`phasew`) uses unwrap-interpolate-rewrap strategy to avoid artifacts at +/-pi discontinuities. All other arrays (signal, envelope, phase unwrapped, frequency) interpolate directly.

### Complex signal handling
For complex analytic outputs, interpolate real and imaginary parts independently, then recombine.

---

## Architecture

```
src/filters/decimation.py          (NEW — core utility functions)
    |
    +-- decimate_signal()          Extract every Nth sample
    +-- interpolate_sparse()       Route to method-specific interpolation
    +-- interpolate_3point()       Hurst parabola triplet method
    +-- interpolate_phase_wrapped() Unwrap-interpolate-rewrap for phasew
    +-- interpolate_output_dict()  Apply interpolation to full filter output dict
```

Integration points (existing files, new params only):
```
src/filters/funcDesignFilterBank.py
    apply_filter_bank(... spacing=1, offset=1, interp='none')
        -> internally: decimate signal, redesign kernels for fs/spacing, apply, interpolate

src/time_frequency/cmw.py
    apply_cmw(... spacing=1, offset=1, interp='none')
        -> internally: decimate signal, use fs/spacing, apply FFT filter, interpolate
    apply_cmw_bank(... spacing=1, offset=1, interp='none')
        -> passes through to apply_cmw()
```

Low-level `apply_ormsby_filter()` is **NOT modified** — it takes a prebuilt kernel and stays as a clean primitive.

---

## Files to Create/Modify

| Action | File | Description |
|--------|------|-------------|
| **CREATE** | `src/filters/decimation.py` | Decimation + interpolation utilities (5 functions) |
| **MODIFY** | `src/filters/funcDesignFilterBank.py` | Add spacing/offset/interp to `apply_filter_bank()` |
| **MODIFY** | `src/time_frequency/cmw.py` | Add spacing/offset/interp to `apply_cmw()` and `apply_cmw_bank()` |
| **MODIFY** | `src/filters/__init__.py` | Export new decimation functions |
| **CREATE** | `experiments/appendix_A/test_decimation.py` | Demo/validation script |

---

## Implementation Details

### 1. `src/filters/decimation.py` (NEW)

```python
def decimate_signal(signal, spacing, offset=1):
    """Extract every Nth sample. Returns (decimated_array, original_indices)."""
    start_idx = offset - 1  # convert 1-based to 0-based
    indices = np.arange(start_idx, len(signal), spacing)
    return signal[indices], indices

def interpolate_sparse(values, indices, full_length, method='none'):
    """Fill gaps. Routes to method-specific implementation."""
    # 'none' -> NaN array with values at indices
    # 'linear'/'cubic' -> scipy.interpolate.interp1d
    # '3point' -> interpolate_3point()

def interpolate_3point(values, indices, full_length):
    """Hurst page 213: fit quadratic through each triplet, evaluate at gaps."""
    # For each triplet [i-1, i, i+1]: np.polyfit(x_triplet, y_triplet, 2)
    # Evaluate polynomial at integer positions between triplet endpoints
    # Edge handling: linear from nearest 2 points

def interpolate_phase_wrapped(phasew_sparse, indices, full_length, method):
    """Unwrap sparse -> interpolate -> rewrap."""
    unwrapped = np.unwrap(phasew_sparse)
    interp_unwrapped = interpolate_sparse(unwrapped, indices, full_length, method)
    return np.angle(np.exp(1j * interp_unwrapped))

def interpolate_output_dict(output_dict, indices, full_length, method='none'):
    """Apply interpolation to all arrays in a filter output dict."""
    # signal: if complex, interpolate real/imag separately
    # envelope: interpolate directly
    # phase: interpolate directly (already unwrapped)
    # phasew: use interpolate_phase_wrapped()
    # frequency: interpolate directly
    # None fields: leave as None
```

### 2. `apply_filter_bank()` modifications ([funcDesignFilterBank.py:443](src/filters/funcDesignFilterBank.py#L443))

```python
def apply_filter_bank(signal, filters, fs=52, mode='reflect',
                      spacing=1, offset=1, interp='none'):
```

When `spacing > 1`:
1. Decimate signal: `signal_dec, indices = decimate_signal(signal, spacing, offset)`
2. Set `fs_dec = fs / spacing`
3. **Nyquist check**: Verify all filter edges < pi * fs / spacing rad/yr
4. **Redesign each kernel** for `fs_dec`:
   - Scale `nw` by `1/spacing` (same time duration, fewer samples): `nw_dec = max(51, nw // spacing) | 1`
   - Call `ormsby_filter()` with `fs=fs_dec` and same frequency edges (rad/yr unchanged)
5. Apply redesigned kernel to decimated signal via `apply_ormsby_filter()`
6. Interpolate output back to `len(signal)` via `interpolate_output_dict()`

### 3. `apply_cmw()` modifications ([cmw.py:153](src/time_frequency/cmw.py#L153))

```python
def apply_cmw(signal, f0, fwhm, fs, analytic=True,
              spacing=1, offset=1, interp='none'):
```

When `spacing > 1`:
1. Store `original_length = len(signal)`
2. Decimate: `signal_dec, indices = decimate_signal(signal, spacing, offset)`
3. **Nyquist check**: `f0 + fwhm/2 < pi * fs / spacing`
4. Apply CMW to `signal_dec` with `fs_dec = fs/spacing` (the Gaussian auto-adjusts since `cmw_freq_domain` is parametric in fs)
5. Interpolate output back via `interpolate_output_dict()`

### 4. `apply_cmw_bank()` modifications ([cmw.py:227](src/time_frequency/cmw.py#L227))

Pass `spacing`, `offset`, `interp` through to each `apply_cmw()` call.

### 5. `src/filters/__init__.py` update

Add exports: `decimate_signal`, `interpolate_sparse`, `interpolate_3point`, `interpolate_output_dict`

---

## Demo Script: `experiments/appendix_A/test_decimation.py`

1. Load DJIA weekly data (Hurst window)
2. Pick 3 representative comb filters (FC-1, FC-12, FC-23)
3. Apply **baseline**: `spacing=1` (full resolution)
4. Apply with `spacing=5, offset=1`:
   - `interp='none'` (sparse dots)
   - `interp='3point'` (Hurst parabola)
   - `interp='cubic'` (modern spline)
5. Apply with `spacing=5, offset=3` (different phase)
6. Repeat for CMW bank

**Plot 1** (per filter): Overlay baseline + sparse + 3point + cubic on same axes
**Plot 2**: Envelope comparison — baseline vs interpolated
**Plot 3**: RMS error table printed to console

**Validation checks**:
- `spacing=1` output identical to existing behavior (regression test)
- All interpolated arrays have length == original signal length
- Envelope remains non-negative after interpolation
- RMS error between decimated+interpolated and baseline < 15% for spacing=5

---

## Edge Cases and Constraints

1. **Nyquist violation**: If filter center freq exceeds decimated Nyquist (`pi*fs/spacing` rad/yr), raise `ValueError` with clear message
2. **Offset validation**: Must satisfy `1 <= offset <= spacing`
3. **Spacing=1**: All new code is bypassed, zero behavioral change (backward compatible)
4. **Filter length floor**: `nw_dec = max(51, nw // spacing)`, always odd
5. **Envelope non-negativity**: After interpolation, clip envelope to `max(0, ...)` if cubic overshoot occurs
6. **Short signals**: If decimated signal has < 3 points, fall back to linear for `'3point'` method

---

## Verification

1. Run `experiments/appendix_A/test_decimation.py` — check plots and RMS errors
2. Run all existing experiment scripts unchanged — confirm no regressions (spacing defaults to 1)
3. Visually compare 3-point parabola vs cubic spline vs baseline to assess interpolation quality
