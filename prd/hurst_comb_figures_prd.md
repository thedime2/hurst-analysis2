# PRD: Hurst Appendix A Comb Filter Figures AI-2 through AI-7
## Product Requirements Document

**Author:** Project
**Date:** 2026-02-18  (updated)
**Status:** Active
**Location:** `experiments/appendix_A_v2/`

---

## 1. Overview & Goals

This PRD covers a fresh, focused reproduction of Hurst's Appendix A comb filter analysis
figures AI-2 through AI-7. It supersedes earlier Phase 2/3 scripts by adding:

- **Ormsby vs CMW comparison** in AI-2 and AI-3
- **Daily data comparison** throughout (fs = 251 trading days/yr)
- **Multiple frequency measurement methods** for AI-4, including MPM
- **New figures AI-5 and AI-6** not previously implemented

Reference: J.M. Hurst, *The Profit Magic of Stock Transaction Timing* (1970), Appendix A.

---

## 2. Data

| Parameter | Value |
|-----------|-------|
| Weekly CSV | `data/raw/^dji_w.csv` |
| Daily CSV | `data/raw/^dji_d.csv` |
| Analysis window | 1921-04-29 to 1965-05-21 |
| Display window | **1934-12-07 to 1940-01-26** (~267 weeks) |
| Weekly fs | 52 samples/year |
| Daily fs | ~258.8 trading days/year (computed empirically; pre-1952 NYSE had 6-day weeks) |

**Important**: ALL CSV rows are loaded for filtering (no date restriction). This avoids edge
effects from filter startup/shutdown at the boundaries of the analysis window. Only the
display window 1934-12-07 to 1940-01-26 is shown in the figures.

The display window 1934-12-07 to 1940-01-26 matches Hurst's AI-3 "example" period
(editorial note: the printed header "5-24-40 / 3-29-46" in the book is a transcription error).

---

## 3. Comb Filter Bank Specification

Hurst's uniform-step bank (Appendix A, p. 192):

| Parameter | Value |
|-----------|-------|
| Number of filters | 23 |
| First filter lower skirt (w1) | 7.2 rad/yr |
| Step between filters | 0.2 rad/yr |
| Passband width (w3 - w2) | 0.2 rad/yr |
| Skirt width (w2 - w1 = w4 - w3) | 0.3 rad/yr |
| Total span per filter | 0.8 rad/yr |
| Center frequencies | 7.6, 7.8, 8.0, ..., 12.0 rad/yr (FC-1..FC-23) |

### Filter Edge Formula

Filter k (0-indexed):
```
w1 = 7.2 + k*0.2
w2 = w1 + 0.3   (skirt edge)
w3 = w2 + 0.2   (passband end)
w4 = w3 + 0.3   (outer skirt)
fc = (w2 + w3) / 2
```

### Filter Kernel Lengths

| Sampling | Weekly (fs=52) | Daily (fs~251) |
|----------|----------------|----------------|
| Ormsby nw | 3501 | ~16900 (auto-scaled) |
| Method | modulate, complex analytic | same |

Rule: `nw_daily = int(nw_weekly * fs_daily / 52)` rounded to odd.

### CMW Parameters (matched to Ormsby)

Derived from `src.time_frequency.cmw.ormsby_spec_to_cmw_params()`:
- Center frequency f0 = (w2 + w3) / 2
- FWHM = (w3 + w4)/2 - (w1 + w2)/2 = 0.5 rad/yr for all filters

---

## 4. Figure AI-2: Filter Frequency Response

### Description
Two subplots showing the actual FFT frequency response of all 23 comb filters,
comparing Ormsby FIR vs CMW responses against the idealized trapezoidal shape.

### Layout
```
Figure (width=16, height=10)
+------------------------------------------+
|  Subplot 1: Weekly (fs=52)                |
|  Ormsby (solid), CMW (dashed),            |
|  Ideal trapezoid (thin gray)              |
|  23 filters, color-coded 1-23            |
|  X: 7.0-13.0 rad/yr  Y: 0.0-1.1         |
|  Legend: Ormsby | CMW | Ideal            |
+------------------------------------------+
|  Subplot 2: Daily (fs~251)                |
|  Same layout as Subplot 1                |
+------------------------------------------+
```

### Technical Requirements
- NFFT = 65536 for fine spectral resolution
- Ormsby: full FFT of kernel, normalize to peak = 1.0
- CMW: frequency-domain Gaussian with FWHM matched to Ormsby skirt midpoints
  - CMW response: `H_cmw(f) = exp(-(f - f0)^2 / (2 * sigma_f^2))`
  - sigma_f from `ormsby_spec_to_cmw_params(spec)['sigma_f']`
  - Normalize CMW peak to 1.0
- Idealized trapezoid: piecewise linear through [w1,0]-[w2,1]-[w3,1]-[w4,0]
- Filter labels (1-23) displayed at top of each filter's trapezoid
- X-axis zoom: 7.0 to 13.0 rad/yr (slightly wider than filter bank)
- Grid: yes, alpha=0.3
- Both subplots share X and Y axes

### Output
`experiments/appendix_A_v2/fig_AI2_filter_response.png`

---

## 5. Figure AI-3: Comb Filter Time-Domain Outputs

### Description
Shows FC-1 through FC-10 filter outputs over the display window, comparing
Ormsby vs CMW and weekly vs daily data.

### Layout
```
Figure A: Weekly (2-column, same figure)
+---------------------+---------------------+
| Ormsby Weekly       | CMW Weekly          |
| FC-1: waveform+env  | FC-1: waveform+env  |
| FC-2: ...           | FC-2: ...           |
| ...                 | ...                 |
| FC-10               | FC-10               |
| X: weeks 0-267      | X: weeks 0-267      |
+---------------------+---------------------+

Figure B: Daily (same layout, daily data)
```

### Technical Requirements

**Vertical layout (Hurst style):**
- FC-1 at top, FC-10 at bottom
- Each trace normalized to global RMS amplitude (median across all filters)
- Target half-amplitude = TRACK_AMP = 1.5 normalized units
- Vertical offset spacing = SPACING = 4.0 (normalized) between filter zero-lines
- Zero reference line (thin gray) for each filter
- Waveform: real part of analytic output, blue solid line
- Envelope: abs of analytic output, red dashed line (both +/-)
- **Wrapped phase overlay**: thin gray line on each track, scaled ±π → ±0.6 track units
  (computed as `np.angle(signal)` for both Ormsby and CMW outputs)

**X-axis:**
- Show weeks from 0 at display window start
- Grid lines at every 25 weeks
- Label "Weeks"

**Y-axis:**
- Filter labels on left: "FC-1", "FC-2" etc. with center frequency
- +2 / 0 / -2 annotation marks per filter (optional)

**Alternative layout (simpler, subplots):**
- n_display × 1 subplots (one per filter) with sharex=True
- Matches the earlier phase2_figure_AI2.py approach

Use the **single-axis stacked layout** (offsets) to match Hurst's original.

### Output
- `fig_AI3_weekly.png` - Weekly data, 2-column (Ormsby + CMW)
- `fig_AI3_daily.png` - Daily data, 2-column (Ormsby + CMW)

---

## 6. Figure AI-4: Frequency vs Time

### Hurst's Own Description (Appendix A)

> "Each sine curve in time was frequency analyzed to permit study of frequency variations
> as a function of time. Results for the 23 filters of this example are shown in Figure
> AI-4. Here, it is noted that the output of filters 1, 2, and 3 are clustered in a narrow
> frequency band, then there is a 'leap' in frequency to the output of filters 4, 5, 6, and
> 7. Outputs from several filters in the comb (such as 8, 12, and 16), whose response curves
> straddled the frequency gaps shown, fell completely outside the possible response
> pass-bands of the respective filter and were discarded as meaningless."

Key phrases:
- **"frequency analyzed"**: some form of period measurement applied to each filter output
- **"filters straddling frequency gaps"**: filters 8, 12, 16 (our FC-8=9.0, FC-12=9.8,
  FC-16=10.6 rad/yr) fall between spectral line clusters. Their output contains low-energy
  beating between adjacent harmonics, causing the measured frequency to drift OUTSIDE the
  filter's ±30% passband — so they appear "meaningless."
- **"discarded as meaningless"**: in our reproduction ALL filters show data because our
  passband clipping is ±30%; Hurst may have used tighter clipping or amplitude gating.

### Missing Filters 8, 12, 16 — Physical Explanation

Filters at 9.0, 9.8, 10.6 rad/yr fall in the gaps between spectral line clusters:
- Cluster A: ~7.6–8.8 rad/yr (harmonics n=21..24)
- Gap at ~9.0 rad/yr → **FC-8 missing**
- Cluster B: ~9.2–9.6 rad/yr (harmonic n=25..26)
- Gap at ~9.8 rad/yr → **FC-12 missing**
- Cluster C: ~10.0–10.4 rad/yr (harmonic n=27..28)
- Gap at ~10.6 rad/yr → **FC-16 missing**

The measured frequency of a filter in a gap drifts widely because the filter captures
two adjacent spectral lines beating against each other. The envelope oscillates and the
instantaneous frequency swings across the full beat range — outside the filter's passband.
This is NOT measurement error; it is physically correct but Hurst called it "meaningless"
because it identifies anti-node regions rather than spectral lines.

### Brute-Force Measurement Scheme Results (2026-02-18)

Six schemes were tested with parabolic sub-sample interpolation at every peak/trough
([fig_AI4_brute_6schemes.png](fig_AI4_brute_6schemes.png)):

| Scheme | Density | Frequency Variation | Match to Hurst |
|--------|---------|---------------------|----------------|
| A: PP (peak→peak full period at 2nd peak) | 7 avg/filter | Flat — misses drift | Density ✓, Shape ✗ |
| B: TT (trough→trough) | 8 avg/filter | Flat | Same as A |
| C: PT (half-period at trough) | 8 avg/filter | Moderate | Better |
| D: TP (half-period at peak) | 7 avg/filter | Moderate | Better |
| E: PT+TP interleaved (half-period between every adjacent event) | 16 avg/filter | Zig-zag + drift ✓ | Best variation |
| F: PP+TT merged chronologically | 16 avg/filter | Moderate zigzag | Dense |

**Best match overall:** Scheme E (PT+TP interleaved) with 5-pt moving average:
- 12 avg/filter (close to Hurst's estimated 7–10)
- Shows frequency drift visible in Hurst's figure (e.g., FC-17 drifting 10.7→10.0)
- Reproduces filter crossings (FC-15 vs FC-17 around week 110)
- Output: `fig_AI4_final.png`, `fig_AI4_final_4panel.png`

**Remaining difference:** Hurst's top cluster (FC-18..23, near 12 rad/yr) shows larger
oscillation amplitude (~±0.5 rad/yr) than ours (~±0.15). Likely cause: Hurst's analog/
early-digital filters had wider effective bandwidth, allowing adjacent harmonics to beat
and produce more frequency modulation.

### Measurement Methods Implemented

#### Method 1: Zero-Crossing Half-Period
Find all zero-crossings; half-period between consecutive crossings → frequency = π/T_half.
Gives ~13 pts/filter. Noisy zig-zag from consecutive rising/falling asymmetry.

#### Method 2: Peak-to-Peak (PP) / Method 3: Trough-to-Trough (TT)
Distance-based peak detection + parabolic sub-sample interpolation.
Gives ~7–8 pts/filter. Too flat — filters near spectral lines show near-constant frequency.

#### Method 4: PT+TP Interleaved (BEST for Hurst match)
All peaks and troughs in chronological order; consecutive half-period between every
adjacent event → 2 measurements per cycle. Apply 5-pt moving average.
Gives ~12 pts/filter. Captures drift and filter crossings.

#### Method 5 (TO TRY): Instantaneous Phase Derivative
```python
phi_unwrapped = np.unwrap(np.angle(z_analytic))  # analytic signal from Ormsby
omega_inst    = np.diff(phi_unwrapped) * fs       # rad/yr, at every sample
```
Then sample at peaks or smooth to desired density. Gives true continuous instantaneous
frequency without discretization artifacts. See `fig_AI4_phase_deriv.py`.

#### Method 6 (TO TRY): Phase Derivative Sampled at Peaks
Compute instantaneous phase derivative (Method 5), then sample its value only at
the peak times of the filter output. Effectively gives PP scheme but the frequency
estimate uses the local gradient rather than the inter-peak interval.

### Parabolic Interpolation at Peaks (Hurst's Appendix)
Hurst describes fitting a 3-point parabola through consecutive peak amplitudes for
interpolation. For frequency measurement, we apply this to peak TIMES:
```python
def parabolic_peak(y, idx):
    y0, y1, y2 = y[idx-1], y[idx], y[idx+1]
    denom = y0 - 2*y1 + y2
    delta = 0.5 * (y0 - y2) / denom
    return idx + clip(delta, -1, 1)
```
Already implemented in `fig_AI4_bruteforce.py` and `fig_AI4_final.py`.

### Peak Detection
Use distance-based `find_peaks(signal, distance=T_half_samples*0.55)`.
Prominence-based detection FAILS when signal amplitude varies widely across the
full 6747-sample record (Great Depression low amplitude vs 1950s high amplitude).

### Frequency Clipping
Keep measurements within ±30% of filter centre frequency AND within [7.4, 12.6] rad/yr.
Hurst's "missing" filters 8, 12, 16 had their entire outputs fall outside this window.

### Output Files
- `fig_AI4_final.png` — Best Hurst-style single panel (PT+TP + 5pt MA) **[CURRENT BEST]**
- `fig_AI4_final_4panel.png` — Reference vs PP raw vs PP smoothed vs PT+TP 5pt
- `fig_AI4_brute_6schemes.png` — All 6 schemes side-by-side
- `fig_AI4_brute_aligned.png` — AI-3 waveforms aligned with AI-4 FVT (same x-axis)
- `fig_AI4_phase_deriv.png` — Instantaneous phase derivative comparison (TO DO)

---

## 7. Figure AI-5: Modulation Sidebands

### Description
Shows 6 frequency bands as shaded regions, illustrating the amplitude of
instantaneous frequency modulation around each spectral line. Reproduces
Hurst's "The Line Frequency Phenomena" figure.

### Center Frequencies (from Hurst AI-5)
```
11.8 rad/yr  (27.7 weeks period)
11.0 rad/yr  (29.7 weeks period)
10.2 rad/yr  (32.0 weeks period)
 9.4 rad/yr  (34.7 weeks period)
 8.6 rad/yr  (38.0 weeks period)
 7.8 rad/yr  (41.9 weeks period)
```

### Method
For each center frequency f0 (corresponding to a filter FC-k):
1. Take the AI-4 frequency measurements (zero-crossing method) for that filter
2. At each time point, the measurement gives an instantaneous frequency f(t)
3. The deviation is δf(t) = f(t) - f0
4. Plot:
   - Horizontal center line at f0 (thin solid)
   - Shaded region: f0 + δf(t) above and below, as a filled area between
     `f0 + smoothed_upper(δf)` and `f0 + smoothed_lower(δf)`
   - Use a slow-moving envelope of δf (e.g., interpolated peak envelope)
5. Alternatively: shade between the raw f(t) trace and f0 (simpler)

### Layout
```
Single figure, tall format (width=12, height=16)
6 bands stacked vertically, each centered on its f0
Band width displayed: ±2 rad/yr around f0
Left Y-axis: frequency in rad/yr
Right Y-axis: period in weeks (T = 2π/f * 52)
X-axis: weeks (display window)
Hatch pattern: diagonal (matches Hurst's original)
```

### Output
`fig_AI5_sidebands.png`

---

## 8. Figure AI-6: LSE Frequency vs Time Analysis

### Description
Applies sliding-window Prony/LSE analysis to each filter output to produce
frequency vs time estimates displayed as short horizontal line segments.
Reproduces Hurst's "LSE, Frequency vs Time Analysis" figure.

### Method (Prony / 1-mode LSE)

For each filter FC-k:
1. Extract complex analytic output y[n]
2. For each window centered at time t_c with half-length W:
   - Extract segment: y[t_c - W : t_c + W]
   - Apply 1-mode MPM to segment (same as AI-4 Method 4)
   - Get frequency estimate f_est (rad/yr)
3. Draw horizontal line segment from (t_c - W/2, f_est) to (t_c + W/2, f_est)
4. Step by W/2 (50% overlap between windows)

**Window sizing:**
- Use `W = int(round(fs / f_c * 1.5))` samples (1.5 periods per window)
- Minimum W = 20 samples
- For higher frequencies, windows are shorter → more data points, less smoothing

### Layout
```
Single figure (width=14, height=10)
X: Weeks (0 to 300 relative to analysis start, matching Hurst's AI-6)
Y: 0 to 12 rad/yr
Grid: yes
Each filter's segments: thin horizontal black line
Multiple filters overlap → dense clustering visible
Title: "LSE, FREQUENCY VS TIME ANALYSIS"
```

### Analysis Window for AI-6
Hurst's AI-6 shows weeks 20-300 (280-week span), suggesting he used a wider
window than AI-3. Use the full 1921-1965 analysis window but display only the
first 300 weeks (from analysis start).

### Output
`fig_AI6_lse_analysis.png`

---

## 9. Figure AI-7: Line Spectrum with Digital Filter Points

### Description
Updates the Phase 3 nominal model line spectrum (w_n = 0.3676*N) to add
the digital filter analysis points from AI-6, matching Hurst's "Best Estimate
of Line-Spacing" figure.

### Method
1. Load nominal model from `data/processed/nominal_model.csv`
2. From AI-6 LSE results: for each filter FC-k, collect all frequency estimates
3. Cluster estimates → identify stable clusters → get median frequency per cluster
4. Map cluster centers to nearest harmonic N of 0.3676 rad/yr
5. Plot:
   - Straight line: w = 0.3676 * N (Hurst's nominal model)
   - Fourier analysis points (from existing Phase 3): circles
   - Digital filter points (from AI-6): squares
   - Error bars showing ±1 std of each cluster

### Layout
```
Single figure (width=10, height=10)
X: N (harmonic number, 0-34)
Y: omega (0-12 rad/yr)
Legend: Fourier Analysis (o) | Digital Filter Analysis (.)
Reference line: w = 0.3676*N
Title: "LOW FREQUENCY LINE SPECTRUM - DOW JONES INDUSTRIAL AVERAGE"
```

### Output
`fig_AI7_line_spectrum.png`

---

## 9.5 Figure AI-8: Low-Frequency Portion: Spectral Model (Table)

### Description
Hurst's AI-8 is a **table** (not a graph) showing all 34 harmonic spectral lines
with columns N | ω_n | T_Y | T_M | T_W | T_NOM.

### Nominal Model
`ω_n = n × 0.3676 rad/yr`  for n = 1..34

### Columns
| Column | Contents | Shown for |
|--------|----------|-----------|
| N | Harmonic number 1-34 | All rows |
| ω_n | n × 0.3676 (4 dp) | All rows |
| T_Y | Period in years (1 dp for N≤9, 2 dp for N=10-14) | N=1-14 |
| T_M | Period in months (1 dp if ≥10, 2 dp if <10) | N=3-34 |
| T_W | Period in weeks (1 dp) | N=30-34 |
| T_NOM | Nominal label | Selected rows (see below) |

### T_NOM Labels
| N | T_NOM |
|---|-------|
| 1 | 18.0 Y |
| 2 | 9.0 Y |
| 4 | 4.3 Y |
| 6 | 3.0 Y |
| 10 | 18.0 M |
| 15 | 12.0 M |
| 23 | 9.0 M |
| 34 | 6.0 M |

### Group Boundaries (horizontal dividing lines)
| Group | N range | T_NOM | Notes |
|-------|---------|-------|-------|
| 1 | 1-2 | 18Y, 9Y | Thick border after N=2 |
| 2 | 3-7 | 4.3Y, 3.0Y | Thick border after N=7 |
| 3 | 8-14 | 18M | Thick border after N=14; thin dashed sub-divider after N=12 |
| 4 | 15-19 | 12M | Thick border after N=19 |
| 5 | 20-26 | 9M | Thick border after N=26 |
| 6 | 27-34 | 6M | Final group, no trailing border |

### Output
`fig_AI8_spectral_table.png`

---

## 10. Implementation Plan

### Directory Structure
```
experiments/appendix_A_v2/
├── fig_AI2_filter_response.py   # AI-2: Filter freq response comparison
├── fig_AI3_comb_outputs.py      # AI-3: FC-1..10 stacked, 4 panels + wrapped phase
├── fig_AI4_freq_vs_time.py      # AI-4: Frequency vs time, 4 methods + Hurst-style panel
├── fig_AI5_sidebands.py         # AI-5: Modulation sideband shading
├── fig_AI6_lse_analysis.py      # AI-6: Prony sliding-window LSE
├── fig_AI7_line_spectrum.py     # AI-7: Line spectrum + filter points
├── fig_AI8_spectral_table.py    # AI-8: 34-harmonic spectral model table (NEW)
└── utils_ai.py                  # Shared utilities (data load, filter design, phase measurement)
```

### Shared Utilities (utils_ai.py)
- `load_weekly_data()` → (close, dates_dt)
- `load_daily_data()` → (close, dates_dt, fs_daily)
- `design_comb_bank(fs, nw)` → filter_specs
- `make_ormsby_kernels(specs, fs)` → list of complex kernels
- `make_cmw_response(specs, fs, nfft)` → list of (freqs, H_mag) arrays
- `apply_all_filters(signal, kernels, fs, mode='reflect')` → list of analytic outputs
- `get_display_window(dates_dt)` → (s_idx, e_idx)
- `mpm_1mode(z, L_pencil=None)` → omega_rad_per_sample

### Dependencies (existing)
```python
from src.filters import design_hurst_comb_bank, create_filter_kernels, apply_filter_bank
from src.time_frequency.cmw import ormsby_spec_to_cmw_params, apply_cmw, apply_cmw_bank
from src.spectral.frequency_measurement import (
    measure_freq_at_peaks, measure_freq_at_troughs,
    measure_freq_at_zero_crossings
)
```

---

## 11. Key Parameter Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Ormsby method | modulate + analytic | Best symmetry, clean envelope |
| Daily nw | `int(nw_weekly * fs_daily / 52)` | Same span in time |
| Daily fs | ~258.8 (empirical from CSV dates) | Pre-1952 NYSE had 6-day weeks |
| CMW FWHM | Matched to Ormsby skirt midpoints | Via `ormsby_spec_to_cmw_params` |
| **Data range for filtering** | **ALL CSV rows (no date filter)** | **Avoids edge effects at display window boundaries** |
| AI-3 spacing | SPACING = 4.0 (normalized units) | Tighter than original 5.5 |
| AI-3 wrapped phase | overlay on each track, scaled ±π → ±0.6 | Visual check of phase continuity |
| AI-4 primary method | PT+TP interleaved + 5pt MA | Best visual match to Hurst; shows drift and filter crossings |
| AI-4 peak detection | distance-based (T*0.55 samples min) | Robust across varying-amplitude full record |
| AI-4 frequency clip | ±30% of filter centre + YMIN/YMAX | Removes edge noise; filters 8,12,16 fall entirely outside this |
| AI-4 parabolic interp | 3-point parabola at every peak/trough | Sub-sample precision; matches Hurst's appendix description |
| AI-4 smoothing | 5-pt centred MA on PT_TP measurements | Reduces 16 pts/filter to ~12, preserving variation |
| AI-6 window | 1.5 periods per filter | Balance resolution vs variance |
| AI-6 step | 50% overlap | Dense enough for visual display |
| AI-5 shading | Raw f(t) band | Simpler, direct from AI-4 |
| AI-8 | Reproduced as styled matplotlib table | Hurst's original IS a table, not a graph |

---

## 12. Acceptance Criteria

### AI-2
- [ ] 23 Ormsby filter responses visible in 7-13 rad/yr range
- [ ] CMW responses overlaid with different linestyle
- [ ] Both weekly and daily subplots present
- [ ] Idealized trapezoidal shapes visible as background
- [ ] Filter numbers 1-23 labeled

### AI-3
- [x] FC-1 through FC-10 stacked from top to bottom
- [x] Waveforms and envelopes visible
- [x] Date range 1934-12-07 to 1940-01-26
- [x] Both Ormsby and CMW panels present
- [x] Wrapped phase overlay on each track (thin gray)
- [x] Tighter spacing (SPACING=4.0)

### AI-4
- [x] Y-axis 7.4-12.6 rad/yr
- [x] Frequency measurements form zig-zag pattern per filter
- [x] Horizontal reference lines at 8, 9, 10, 11, 12 rad/yr
- [x] 4 methods compared (zero-crossing, peak+trough, MPM, wrapped phase)
- [x] Hurst-style single-panel figure with filter labels at both ends
- [x] Distance-based peak detection (not prominence-based)
- [x] ±28% frequency clipping around filter centre

### AI-5
- [ ] 6 frequency bands at 7.8, 8.6, 9.4, 10.2, 11.0, 11.8 rad/yr
- [ ] Each band has center line and shaded modulation region
- [ ] Diagonal hatch pattern on shading

### AI-6
- [ ] Horizontal line segments at multiple frequencies
- [ ] Dense clusters visible around spectral lines
- [ ] X-axis matches Hurst's range (weeks)

### AI-7
- [x] Straight line w = 0.3676*N
- [x] Both Fourier and digital filter analysis points plotted
- [x] Points scatter close to the line

### AI-8
- [x] All 34 harmonic rows (N=1..34) with ω_n = n*0.3676
- [x] T_Y column (N=1-14 only, matching Hurst's precision)
- [x] T_M column (N=3-34, 1 dp for ≥10, 2 dp for <10)
- [x] T_W column (N=30-34 only)
- [x] T_NOM labels at correct rows (18Y, 9Y, 4.3Y, 3Y, 18M, 12M, 9M, 6M)
- [x] Group borders: thick after N=2, 7, 14, 19, 26; thin dashed after N=12
- [x] Title: "FIGURE A I-8"
- [x] Caption: "Low-Frequency Portion: Spectral Model"

---

## 13. Notes

1. **Daily data filter length**: For daily data with fs~251, the filter length
   must be much longer to achieve the same frequency resolution. A rule of thumb
   is `nw_daily = nw_weekly * (fs_daily / fs_weekly)`. With nw_weekly=3501 and
   fs_daily~251, this gives nw_daily~16900. This is computationally expensive
   but necessary for correct daily-rate filter design.

2. **CMW response calculation**: The CMW frequency response is Gaussian and can
   be evaluated analytically. No need to compute a full FFT of a kernel.
   `H_cmw(f) = exp(-(f - f0)^2 / (2 * sigma_f^2))` evaluated at each frequency.

3. **MPM stability**: For very short segments (< 6 samples), MPM is unreliable.
   Skip segments shorter than min_length = 6 samples.

4. **AI-5 derivation from AI-4**: The shaded regions in AI-5 represent the
   envelope of frequency variation, not the raw traces. Use a Savitzky-Golay
   or moving average to smooth the upper and lower bounds of the frequency
   time series.

5. **AI-6 window alignment**: Hurst's figure starts at week ~20, suggesting
   the first few windows are skipped due to edge effects from filter startup.
   Skip the first and last `nw/2` samples from LSE analysis.

6. **AI-4 Hurst date range**: From looking at the AI-4 reference (x-axis 0-275 weeks),
   this matches roughly 1934 to 1940 at weekly sampling. The zig-zag structure
   is very clear in the reference figure, with some filters showing nearly constant
   frequency (those centered on spectral lines) and others showing large variation.

7. **AI-4 measurement density**: Weekly data gives ~13 measurements per filter in
   the 269-week display window (vs Hurst's estimated 10-12). The wrapped phase and
   zero-crossing methods give similar results for narrow-band filters. The key
   difference from Hurst's original is smoothness: Hurst's lines appear smoother
   because he may have hand-drawn or lightly smoothed the connecting lines.

8. **AI-8 is a table**: Hurst's AI-8 is a printed table with columns N, ω_n, T_Y,
   T_M, T_W, T_NOM. It is NOT a graphical figure. Minor numerical differences from
   Hurst's values (e.g., T_Y=2.8 vs Hurst's 2.9 for N=6) arise from his use of
   slide-rule era arithmetic vs. modern floating-point computation.

9. **All-CSV filtering decision**: Loading ALL available data (not just 1921-1965)
   avoids filter edge effects at the boundary of the display window. The Ormsby
   filters have 3501-tap kernels, so the first/last 1750 samples are contaminated
   by edge effects. With 6747 weekly samples available, this is well beyond the
   display window region.

10. **Wrapped phase method equivalence**: For a pure narrow-band signal, wrapped
    phase peaks (φ→0 crossings) and zero crossings of the real part occur at the
    same times. The phase method gives sub-sample interpolation but is conceptually
    identical to zero-crossing for clean Ormsby outputs.

11. **AI-4 brute-force scheme comparison** (2026-02-18): Tested 6 schemes (PP, TT,
    PT, TP, PT+TP interleaved, PP+TT merged). Best Hurst match is PT+TP interleaved
    with 5-pt MA (12 pts/filter). PP gives correct density (7 pts) but lines are too
    flat (near-constant frequency). The variation in Hurst's figure is captured better
    by the half-period interleaved scheme, likely because Hurst's wider filters allowed
    more beating from adjacent harmonics to produce real frequency modulation.

12. **Filters 8, 12, 16 "meaningless"** (Hurst's quote): These fall in frequency GAPS
    between spectral line clusters. In a gap, the filter output is low-amplitude beating
    between two adjacent harmonics. The instantaneous frequency swings across the full
    beat range, typically outside the ±30% passband window. This is physically meaningful
    (it confirms the discrete line structure) but Hurst discards these as they don't
    identify a single spectral line. In our reproduction, these filters still produce
    data because our clipping is wide; implementing amplitude gating would reproduce
    Hurst's behavior.

13. **Next measurement experiments** (TO DO):
    - Instantaneous phase derivative: `d(unwrap(angle(z)))/dt * fs` — continuous, no
      discretization. Sample at peaks or smooth for desired density.
    - Phase derivative sampled at peaks: frequency value at peak time = local rate of
      change of phase = true instantaneous frequency without inter-peak interval rounding.
    - Amplitude gating: suppress output for filters where RMS < threshold (reproduces
      Hurst's missing filters 8, 12, 16).
