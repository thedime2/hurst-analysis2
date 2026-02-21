# AI-4 Frequency vs Time — Measurement Investigation Notes

**Date:** 2026-02-18
**Author:** Investigation via Claude Code

---

## 1. What Hurst Actually Said

From Appendix A (direct quote):

> "Each sine curve in time was frequency analyzed to permit study of frequency variations
> as a function of time. Results for the 23 filters of this example are shown in Figure AI-4.
> Here, it is noted that the output of filters 1, 2, and 3 are clustered in a narrow frequency
> band, then there is a 'leap' in frequency to the output of filters 4, 5, 6, and 7. Outputs
> from several filters in the comb (such as 8, 12, and 16), whose response curves straddled the
> frequency gaps shown, fell completely outside the possible response pass-bands of the respective
> filter and were discarded as meaningless."

**Key interpretation:**
- "Frequency analyzed" = some period measurement method applied to each filter's sine wave output
- The filter OUTPUT (a narrow-band sine wave in the display window) is what gets analyzed
- The missing filters 8, 12, 16 confirm DISCRETE spectral lines — if energy were continuous, no filter would fall "outside its passband"

---

## 2. Filters 8, 12, 16 — The "Meaningless" Filters

### Their center frequencies (our 23-filter comb bank)

| Hurst FC | Center (rad/yr) | Location |
|----------|-----------------|----------|
| FC-8  | 9.0 | Between clusters A and B |
| FC-12 | 9.8 | Between clusters B and C |
| FC-16 | 10.6 | Between clusters C and D |

### Why they're "meaningless"

These filters fall in **anti-resonance zones** between spectral line clusters. Their
filter passband (±0.1 rad/yr) straddles a region with no strong harmonic. Instead, two
adjacent harmonics at distance ~0.3676 rad/yr on either side both partially enter through
the filter skirts.

The output is a **beat signal**: A(t)·cos(fc·t) where A(t) oscillates at the beat frequency
Δω = separation between the two captured lines. The instantaneous frequency swings between
the two line frequencies, crossing far outside the filter's passband on every beat cycle.

**Result:** Measured frequency is meaningless (swings outside passband) AND amplitude
is low (near-minimum of the beat envelope). Hurst correctly discards these.

### In our reproduction

Our ±30% clip window keeps all filters' data. To reproduce Hurst's behavior:
1. **Amplitude gate**: discard measurements where RMS(output in display window) < threshold × median RMS
2. **Tighter clip**: use ±15% instead of ±30% to exclude beat-driven excursions
3. **Both**: amplitude gate first, then frequency clip

Expected result: FC-8, FC-12, FC-16 would disappear from the AI-4 plot, leaving visible gaps
that confirm the discrete line structure — exactly as Hurst's figure shows.

---

## 3. Brute-Force Scheme Comparison (all 6 schemes)

Script: `fig_AI4_bruteforce.py`
Output: `fig_AI4_brute_6schemes.png`

### Point counts (23 filters, 269-week display window)

| Scheme | Total pts | Avg/filter | Hurst match |
|--------|-----------|------------|-------------|
| A: PP  | 181 | 7 | Density ✓, too flat |
| B: TT  | 187 | 8 | Same as A |
| C: PT  | 187 | 8 | Moderate variation |
| D: TP  | 182 | 7 | Moderate variation |
| E: PT+TP interleaved | 369 | 16 | Zig-zag ✓ but dense |
| F: PP+TT interleaved | 368 | 16 | Similar to E |

### Best scheme: E (PT+TP) with 5-pt smoothing

After 5-pt moving average: **277 pts, 12 avg/filter**

Reproduces:
- Filter 15/17 crossing around week 110 ✓
- FC-17 drifting from 10.7 → 10.0 over the 269-week window ✓
- Bottom group (FC-1..7) smooth and flat ✓
- Middle group (FC-9..12) showing moderate variation ✓

Not reproduced:
- Top cluster (FC-18..23) oscillation amplitude ±0.15 rad/yr vs Hurst's ±0.5 rad/yr
  → likely due to our narrower filter bandwidth (0.2 rad/yr passband vs Hurst's wider filters)

---

## 4. Parabolic Sub-Sample Interpolation — Clarification

**What Hurst actually said**: Parabolic interpolation is mentioned in the Appendix as a
technique to **smooth the filter output** when filters are spaced apart and the filter bank
output is decimated. The parabola is fitted between decimated output samples from adjacent
(spaced) filters to fill in the gaps between them — it is a **frequency-axis interpolation**
method, not a time-axis peak-sharpening technique.

**Not what he said**: Hurst did NOT describe fitting a parabola through consecutive sinusoid
peaks to sub-sample the peak timing. That is our own numerical improvement, useful for our
digital implementation but not something Hurst explicitly advocated for peak measurement.

**Context**: With analog computers, the comb filters were physically separated in frequency.
Parabolic interpolation between the decimated outputs of adjacent filters produces a smoother
FVT curve than simple linear interpolation, especially in regions where filter outputs vary
rapidly. This is why the technique appeared in the context of the filter spacing and decimated
output discussion.

Our implementation for sub-sample peak timing:

```python
def parabolic_peak(y, idx):
    """3-point parabolic sub-sample peak position."""
    y0, y1, y2 = y[idx-1], y[idx], y[idx+1]
    denom = y0 - 2*y1 + y2
    if abs(denom) < 1e-14: return float(idx)
    delta = 0.5 * (y0 - y2) / denom
    return idx + clip(delta, -1, 1)
```

For a 27-week period filter, integer rounding gives ±3–4% frequency error.
Sub-sample interpolation reduces this to <0.1%. Already implemented in all final scripts,
but should be understood as our addition rather than a direct Hurst prescription.

---

## 5. Candidate Methods for Further Investigation

### Method A: Instantaneous Phase Derivative

For analytic signal z(t) = A(t)·exp(jφ(t)):

```
phi = np.angle(z)                    # wrapped phase
phi_uw = np.unwrap(phi)              # unwrapped
omega_inst = np.diff(phi_uw) * fs   # rad/yr at every sample
```

Advantages:
- Continuous (every sample), not just at peaks
- No inter-peak interval discretization
- Captures genuine instantaneous frequency (local rate of phase change)

Issue:
- Noisy at every sample — needs heavy smoothing or subsampling
- At peaks of A(t), the phase derivative is momentarily undefined (AM/FM coupling)

Sampling strategy:
1. **At peaks**: measure `omega_inst[peak_idx]` — gets instantaneous frequency right at
   the amplitude peak, where the filter is most "locked onto" the spectral line
2. **Window average**: average `omega_inst` over one cycle centered on each peak
3. **Heavy smooth + subsample**: Gaussian smooth then pick one value per cycle

### Method B: Complex Ormsby Phase Derivative at Peaks

Same as Method A but the frequency value placed at each peak is taken from the
smoothed derivative rather than the inter-peak interval. This decouples the
TIMING of the measurement (at the peak) from the FREQUENCY ESTIMATE (local derivative).

Expected advantage: captures the actual instantaneous frequency even if
consecutive peaks are unevenly spaced due to AM envelopes.

### Method C: Amplitude-Gated Reproduction of Hurst's Missing Filters

Gate the AI-4 plot: only show filter k's measurements if its RMS energy in the
display window exceeds threshold × median RMS across all filters.

Expected: FC-8 (9.0), FC-12 (9.8), FC-16 (10.6) drop out, leaving visible
frequency gaps exactly matching Hurst's "missing filters 8, 12, 16" pattern.

---

## 6. What Hurst's Filter Design Might Have Been

**Important clarification**: Hurst did NOT use analog filters. His analysis was performed
on a digital computer (an IBM 7094 or similar mainframe of the era), as stated in the book.
All filtering was done numerically. References to "electronic frequency discriminator circuits"
in earlier notes were incorrect speculation and should be disregarded.

Hurst says "special purpose filters for frequency analysis." Key possibilities:

1. **Wider passband**: If his ±0.3 rad/yr passband (vs our ±0.1) admitted adjacent
   harmonics, the filter output would be a two-tone signal with significant FM.
   This would explain the larger oscillation amplitude in Hurst's top cluster.

2. **Different nw (filter length)**: Shorter filter kernels → wider transition bands
   → more adjacent harmonic leakage → more beating → more frequency variation.

3. **Exact comb filter frequencies**: Hurst says filters were arranged with
   "identically equal frequency spacings" — same as our design (0.2 rad/yr step).
   The center frequencies 7.6, 7.8, ..., 12.0 rad/yr are confirmed.

4. **Measurement by digital phase derivative**: The smoothed instantaneous frequency
   trace in Hurst's hand-drawn AI-4 figure is consistent with computing dφ/dt digitally
   (unwrapped analytic signal phase, differenced and smoothed). This is the numerical
   equivalent of what a frequency discriminator would do, but implemented in software.

---

## 6b. Phase Derivative Experiments (2026-02-18)

Script: `fig_AI4_phase_deriv.py`

### Method PDC: Cycle-Averaged Phase Derivative

```python
omega_inst = np.diff(np.unwrap(np.angle(z))) * fs   # rad/yr, at every sample
# For each peak at t_pk:
seg = omega_inst[pk - T/2 : pk + T/2]              # one full cycle
robust_mean = mean(seg[|seg - median| < 5*MAD])    # outlier-rejected
```

**Result: 7 avg/filter** — exactly Hurst's density!

But: gap filters (FC-17, FC-20) cause large excursions because their cycle average
is corrupted by low-amplitude beating (cycle mean ≠ fc for a two-tone signal in a gap).

**With amplitude gating (30% threshold)**: FC-17 and FC-20 suppressed.
- Remaining 21 filters show clean, smooth lines at ~7 pts/filter
- Top cluster (18-23) still flatter than Hurst (same issue as PT_TP: narrower filters)
- This is the most PHYSICALLY CORRECT method

### Method PD-S: Gaussian-Smoothed Derivative at Peaks+Troughs

sigma = T_cycle/4 per filter (adaptive), samples at peaks+troughs.
Result: 15 avg/filter, noisy, not recommended.

### Amplitude Gating Results

With threshold = 30% of median RMS (display window):
- **FC-17 (10.8 r/y): 29% of median → GATED**
- **FC-20 (11.4 r/y): 25% of median → GATED**

These are the "meaningless" filters in our data. Hurst found FC-8, FC-12, FC-16 instead
because his filter bank or display window differed. The mechanism is the same: anti-resonance
zones between spectral clusters.

### Recommendation

| Method | Density | Variation | Gap Handling | Best For |
|--------|---------|-----------|--------------|----------|
| PT+TP + 5pt MA | 12 pts/filter | Good | No (all 23 shown) | Visual Hurst match |
| PDC + amplitude gate | 7 pts/filter | Moderate | Yes (21 filters) | Physical correctness |
| PP | 7 pts/filter | Too flat | No | Not recommended |

**For AI-4 reproduction**: use PT+TP + 5pt MA (fig_AI4_final.png)
**For AI-4 gap-filter demonstration**: use PDC + amplitude gate (fig_AI4_amplitude_gated.png)

---

## 7. CMW vs Ormsby Comparison (2026-02-18)

Script: `fig_AI4_cmw_phase.py`

### CMW filter parameters

Our Ormsby spec (uniform step comb bank):
- Passband: 0.2 rad/yr (flat-top)
- Skirt: 0.3 rad/yr each side
- Total span: 0.8 rad/yr per filter

Matched CMW (Gaussian, skirt-midpoint FWHM):
- FWHM = `(f3+f4)/2 - (f1+f2)/2` = `(fc+0.25) - (fc-0.25)` = **0.500 rad/yr**
- sigma_f = 0.212 rad/yr
- This is **2.5x wider** than the Ormsby passband

### Measurement results

| Method | pts/filter | filters shown | notes |
|--------|-----------|---------------|-------|
| Ormsby PT+TP+5pt MA | 12 | 23/23 | current best |
| CMW PT+TP+5pt MA | 12 | 23/23 | same density |
| CMW PDC (all) | 7 | 23/23 | Hurst density |
| CMW PDC + amp gate | 7 | 22/23 | FC-17 suppressed |

### Key finding: density is frequency-driven, not bandwidth-driven

Both Ormsby and CMW give **identical point density** (12 and 7 pts/filter respectively)
because the number of peaks/troughs in the display window depends only on the filter's
**center frequency**, not its bandwidth. A filter at 10 rad/yr produces ~10/(2π)×52 ≈ 83
half-cycles per year regardless of whether it's Ormsby or Gaussian.

### Amplitude gating differences

- Ormsby: gates FC-17 (10.8, 29% of median) and FC-20 (11.4, 25%)
- CMW: gates only FC-17 (10.8, 15%)

CMW's wider FWHM rescues FC-20 by admitting more energy from adjacent spectral lines.
The CMW filter centered at 11.4 rad/yr admits energy from the 11.0 and 11.8 lines,
elevating its RMS above the gate threshold.

### Visual difference expectation

CMW wider FWHM → more adjacent harmonic energy → potentially more FM variation
in the top cluster (FC-18..23). Whether this actually shows in the FVT plot depends
on whether the phase derivative measurement can resolve the additional FM, or whether
the smoother Gaussian response produces a cleaner (flatter) instantaneous frequency.

---

## 9. New Reproduction Plan: AI-3 + AI-4 Reference Alignment (2026-02-18)

This section documents the planned next-generation reproduction attempt, driven by direct
alignment with the hi-res reference scans (`figure_AI3_v2.png`, `figure_AI4_v2.png`).

---

### 9.1 Display Window — Confirmed Correct

**Current setup (correct)**: `DATE_DISPLAY_START = '1934-12-07'`, `DATE_DISPLAY_END = '1940-01-26'` (~267 weeks)

The reference image AI-3 shows date labels "5-24-40" and "3-29-46" — these are a known
**editorial/transcription error** in the printed book (documented in the PRD). The actual
display window 1934-12-07 to 1940-01-26 is correct. The x-axis in AI-3 ticks to 250 WKS
with unlabeled space to ~267 weeks; AI-4 ticks to 275 weeks.

**Expected cycle counts** for our FC-1..FC-10 (7.6–9.6 rad/yr) in a 267-week window:
- FC-1 (7.6 r/y, period ~43 wk): ~6.2 visible cycles
- FC-5 (8.4 r/y, period ~39 wk): ~6.8 cycles
- FC-10 (9.6 r/y, period ~34 wk): ~7.9 cycles

These numbers should match what is visible in the reference image if our setup is correct.

---

### 9.2 AI-3: Align FC-1..FC-10 Outputs to Hi-Res Reference

**Task**: Generate AI-3 matching the visual style of `figure_AI3_v2.png`, single panel
(one side = Ormsby), then compare visually.

**Reference observations from the image**:
- 10 filter panels stacked vertically, FC-1 at top, FC-10 at bottom
- Each panel shows ±2 amplitude scale (FC-10 shows ±4), labeled at left of each track
- Waveforms are rendered as **dotted lines** on a solid zero-reference line
- Two vertical dashed reference lines cross all panels
- FC-10 has visibly larger amplitude than FC-1..FC-9 (hence ±4 scale)
- Panel heights appear equal; no envelope overlay shown (waveform only)

**Current AI-3 script issues** (from `fig_AI3_comb_outputs.py`):
- Shows 2-panel side-by-side (Ormsby + CMW): not Hurst's layout
- Includes envelope overlay (red dashed) and wrapped phase (gray): not in reference
- Normalization uses a global RMS across all filters: individual ±2 scaling needed
- Uses solid waveform line, not dotted

**Plan**:
1. Create `fig_AI3_v3.py` — single-panel reproduction matching reference style:
   - Waveform as dotted line (`linestyle=':'`)
   - Each track individually normalized to ±2 (scale each filter's own RMS separately)
   - FC-10 automatically gets ±4 if its RMS is ~2× the median
   - No envelope overlay, no phase overlay
   - Label format: `FC-N` at left of each track
   - Vertical reference lines at the two canonical week positions
   - Horizontal zero-line as thin solid gray

2. Produce overlay comparison: reference AI-3 image underlay (using `imshow`) with our
   waveform overlaid to check alignment of peaks, troughs, and frequency

---

### 9.3 Spaced vs Adjacent Filters

**What "spaced" means**: Instead of adjacent/overlapping filters (0.2 rad/yr step with
fully overlapping skirts), try filters with wider spacing so there are gaps between
passbands. Options:
- Step = 0.3676 rad/yr (Hurst harmonic spacing — centered ON the spectral lines)
- Step = 0.4 rad/yr
- Step = 0.6 rad/yr

**Analytic mode compatibility**: `create_filter_kernels()` with `method='modulate',
analytic=True` works regardless of filter spacing. Analytic mode is confirmed correct for
bandpass filters (as noted in MEMORY.md). Wider spacing simply means frequency gaps between
filter passbands — the kernel design is spacing-independent.

**The "spaced dots" hypothesis**: Filters centered on spectral lines:
- Each filter captures one line cleanly (no beat between adjacent lines)
- No "gap filters" exist that would be discarded
- FVT shows fewer filter tracks but each one is clean and stable
- Point density lower but frequency values more accurate

**Plan**:
1. Try spacings: 0.2 (current), 0.3676 (harmonic-aligned), 0.4, 0.6 rad/yr
2. For each spacing generate both:
   - **Dots**: scatter plot (no connecting lines) — to see raw measurement cloud
   - **Line**: consecutive measurements joined by straight segments
3. Compare density and scatter with reference AI-4
4. Best spacing goes into the final AI-4 reproduction

---

### 9.4 AI-4: Reference Image Overlay for Alignment

**Task**: Pixel-align `figure_AI4_v2.png` with our matplotlib axes so we can see exactly
where Hurst placed each measurement point relative to our computed values.

**Reference observations from AI-4 image**:
- Y-axis: 8 to 12 rad/yr, horizontal gridlines at integer values
- X-axis: 0 to 275 (last labeled tick), actual display ~267 weeks, gridlines every 25 wks
- Filter labels at **both left and right sides** of each track
- Lines are **connected straight segments** — no smooth curves, confirming no smoothing
- Point density: ~6-10 visible line segments per filter
- Several filter tracks visibly absent (Hurst's discarded gap filters)

**Alignment method**:
1. Load reference PNG into matplotlib via `imshow`, set `extent=[0, 275, 7.5, 12.5]`
   (mapping image pixels to data coordinates using axis corners)
2. Overlay our FVT measurements as colored scatter dots on the same axes
3. Visually identify:
   - Whether point positions (in weeks) align with Hurst's
   - Whether frequency values (rad/yr) align
   - Which measurement method (PT, PP, ZC) best matches Hurst's dot positions
   - Whether Hurst places the point at the SECOND EVENT or the MIDPOINT

**Script**: `fig_AI4_v3_overlay.py`

---

### 9.5 FVT Methodology Clarification (Corrected Understanding)

**What Hurst's AI-4 actually shows** (confirmed from reference image):

The lines in AI-4 are **connected straight line segments** between individual frequency
measurements. There is **no smoothing** applied to the FVT plot. The zig-zag look IS the
raw data.

**Correct measurement procedure**:
1. Find consecutive events in the filter output (real part):
   - **P→T**: peak to next trough — half-period → frequency = π / Δt_half
   - **T→P**: trough to next peak — same formula
   - **P→P**: peak to next peak — full period → frequency = 2π / Δt_full
   - **T→T**: trough to next trough — same as PP
   - Can interleave P→T and T→P to get one measurement per half-cycle
2. Each measurement is a single (time, frequency) point placed at the **second event**
3. Consecutive points joined by **straight line segments** — that is all
4. **No** moving average, Gaussian smoothing, or any other post-processing
5. Natural result: one point per half-cycle → ~7-10 points per filter in the window

**Previous experiments were over-smoothed**: The 5-pt MA applied in earlier scripts was
our addition, not Hurst's. The overlay (9.4) will confirm this by showing that the
reference zig-zag matches raw PT/PP measurements, not smoothed ones.

---

### 9.6 Additional Experiments: Reduced Weights / Tapered Windows

When measuring from peak-to-trough or peak-to-peak, the amplitude of the waveform at the
moment of measurement affects how "clean" the peak detection is. Two approaches to test:

**Reduced/tapered filter weights** (Nuttall, Hann, etc.): Apply a window function to the
Ormsby kernel before convolution. This reduces sidelobes but widens the main lobe. For
frequency measurement purposes, smoother envelopes may mean cleaner peak locations.
- Test: kernel × Hann window vs. rectangular (current)
- Expected: slightly fewer false-peak detections at the envelope roll-off points

**Amplitude-weighted frequency placement**: Instead of placing the measurement point
at the time of the second event, weight it by the signal amplitude at that moment.
High-amplitude peaks → frequency measurement carries more weight in the plot.
Low-amplitude events (near envelope troughs, i.e., in beat cycles) → discarded or
de-weighted. This is equivalent to the amplitude gating but applied continuously.

---

### 9.7 Implementation Steps (Ordered)

- **[ ] 9.7.1** Write `fig_AI3_v3.py`: corrected style reproduction
  - Dotted waveform, individual ±2/±4 normalization, FC-1..FC-10, no envelope
- **[ ] 9.7.2** Write `fig_AI4_v3_overlay.py`: reference image underlay + our measurements
  - Load reference PNG, map to data coordinates, overlay scatter dots
  - Try ZC, PT, PP methods side by side on same reference underlay
- **[ ] 9.7.3** Write `fig_AI4_v3_spaced.py`: spaced filter comparison
  - Spacings: 0.2, 0.3676, 0.4, 0.6 rad/yr
  - Dots only vs. connected lines for each spacing
- **[ ] 9.7.4** Write `fig_AI4_v3_final.py`: best reproduction
  - Corrected display window (already correct)
  - Best filter spacing (from 9.7.3)
  - Raw PT measurements, no smoothing, straight line segments
  - Labels at left AND right sides
  - Standard Hurst-style gridlines

---

### 9.8 Open Questions To Resolve

| Question | How to resolve |
|----------|----------------|
| Do our AI-3 waveforms match reference shape with current window? | Visual overlay (9.7.1) |
| Which FVT measurement method matches Hurst's dot positions? | Overlay alignment (9.7.2) |
| Do spaced filters reproduce AI-4 better than adjacent? | Spaced filter comparison (9.7.3) |
| Where exactly does Hurst place each frequency point (2nd event or midpoint)? | Overlay alignment reveals timing |
| Should we use amplitude gating to remove gap filters in the final figure? | Yes if overlay shows those tracks missing from Hurst's AI-4 |

---

## 10. Heatmap / Spectrogram Experiments (2026-02-18)

Script: `fig_AI4_spectrogram.py`

### Motivation

The FVT dot-plot (discrete frequency measurements at peaks/troughs) loses information
about the amplitude of each filter at each moment in time. A spectrogram view shows the
**full time-frequency-amplitude** structure simultaneously — closer to a continuous picture
of which spectral lines are active and when.

### Construction

For each of the 23 Ormsby (and CMW) comb filters:
- Extract the analytic signal envelope |z_k(t)| over the 269-week display window
- Place it as one row of a matrix at the filter's center frequency

The 23 discrete rows are then **linearly interpolated** to a 300-row fine frequency grid
(7.4 to 12.6 rad/yr), producing a continuous-looking heatmap. This is equivalent to
assuming linear amplitude variation between adjacent filter centers — valid because the
filter skirts overlap fully with their neighbors (0.2 rad/yr spacing, 0.3 rad/yr skirts).

### Amplitude Scaling

Three scaling modes were compared (see `fig_AI4_spec_scaling.png`):

| Scaling | Effect |
|---------|--------|
| Linear | Strong ridges dominate; weak filters invisible |
| **Sqrt (default)** | **Stretches low-amplitude regions; gap filters visible** |
| Log | Heavy noise floor amplification; gap filters over-represented |

Sqrt scaling was selected as the best balance for visual interpretation.

### Key Findings

**Amplitude ridges confirm spectral line positions**: The heatmap shows bright horizontal
ridges at the frequencies predicted by Hurst's nominal model (multiples of ~0.3676 rad/yr).
Gaps between ridges (anti-resonance zones) are dark — exactly where Hurst's "meaningless"
filters fell.

**Ormsby vs CMW heatmap comparison** (`fig_AI4_spec_heatmap.png`): CMW ridges are slightly
wider (FWHM 0.50 vs Ormsby 0.20 passband) and smoother — fewer sharp transitions between
bright and dark regions. This is consistent with the Gaussian spectral window spreading
energy more gradually.

**CMW minus Ormsby difference map** (`fig_AI4_spec_difference.png`): Shows where each
filter type captures more energy. CMW (wider FWHM) captures more energy in the transition
zones between spectral lines; Ormsby captures more energy right at the line centers due to
its flat passband.

**FVT overlay** (`fig_AI4_spec_overlay.png`): Overlaying the PT+TP+5pt MA frequency
measurements as cyan dots on the amplitude heatmap confirms that:
- Measurements cluster at bright ridges (filters locked onto spectral lines)
- Measurements at gap filters (FC-17, FC-20) fall in dark amplitude regions — confirming
  they are spurious beat-driven readings

**3D surface visualization** (`fig_AI4_spec_3d.png`): The amplitude ridges appear as
mountain ranges in the time-frequency plane. Ridge heights vary over time, consistent with
amplitude modulation of the underlying spectral lines. The Ormsby surface shows sharper,
more distinct peaks; the CMW surface is smoother.

### Filter Overlap Properties

- **Ormsby**: 0.2 rad/yr spacing, 0.3 rad/yr skirts — adjacent filters share their
  entire skirt region. The heatmap rows overlap significantly between filters.
- **CMW**: FWHM 0.50 rad/yr, sigma 0.212 rad/yr — adjacent filters overlap at ±sigma,
  creating a naturally smooth amplitude gradient between centers.

### Output Files (spectrogram series)

| File | Description |
|------|-------------|
| `fig_AI4_spectrogram.py` | Full spectrogram/heatmap generation script |
| `fig_AI4_spec_heatmap.png` | 2D heatmaps: Ormsby vs CMW (sqrt-scaled, interpolated) |
| `fig_AI4_spec_3d.png` | 3D surface: Ormsby vs CMW (amplitude vs time vs frequency) |
| `fig_AI4_spec_overlay.png` | Heatmap + FVT dot overlay (Ormsby, best method) |
| `fig_AI4_spec_difference.png` | CMW minus Ormsby normalised amplitude difference |
| `fig_AI4_spec_scaling.png` | 4-panel: linear vs sqrt vs log vs discrete-row scaling |

---

## 10. Output Files Summary

| File | Description |
|------|-------------|
| `fig_AI4_bruteforce.py` | Generates all 6-scheme comparison figures |
| `fig_AI4_brute_6schemes.png` | 2x3 grid: all 6 measurement schemes |
| `fig_AI4_brute_aligned.png` | AI-3 waveforms (top) + AI-4 FVT (bottom), aligned |
| `fig_AI4_brute_best.png` | Reference vs PP raw vs PP+TT |
| `fig_AI4_final.py` | Best reproduction (PT+TP + 5pt MA) -- PRIMARY |
| `fig_AI4_final.png` | Final Hurst-style single panel |
| `fig_AI4_final_4panel.png` | Reference vs PP raw vs PP smoothed vs PT+TP smoothed |
| `fig_AI4_phase_deriv.py` | Phase derivative experiments (PD, PD-S, PDC) |
| `fig_AI4_phase_deriv_4panel.png` | Reference, PDC all, PDC gated, PT+TP-5 comparison |
| `fig_AI4_phase_deriv_filter.png` | 5-panel diagnostic for FC-20 |
| `fig_AI4_amplitude_gated.png` | Hurst-style: amplitude gating to remove gap filters |
| `fig_AI4_cmw_phase.py` | CMW vs Ormsby FVT comparison |
| `fig_AI4_cmw_4panel.png` | Reference, Ormsby PT+TP-5, CMW PT+TP-5, CMW PDC+gate |
| `fig_AI4_cmw_vs_ormsby.png` | Side-by-side: Ormsby vs CMW (both PT+TP-5) |
| `fig_AI4_cmw_best.png` | Single-panel best CMW (PDC + gate) |
| `fig_AI4_cmw_waveforms.png` | Ormsby vs CMW waveforms for 4 selected filters |
| `fig_AI4_spectrogram.py` | Heatmap / 3D surface spectrogram (all 4 figures) |
| `fig_AI4_spec_heatmap.png` | 2D heatmaps: Ormsby vs CMW |
| `fig_AI4_spec_3d.png` | 3D amplitude surface: Ormsby vs CMW |
| `fig_AI4_spec_overlay.png` | Heatmap + FVT dot overlay |
| `fig_AI4_spec_difference.png` | CMW minus Ormsby amplitude difference map |
| `fig_AI4_spec_scaling.png` | Amplitude scaling comparison (linear/sqrt/log/discrete) |
| **v3 series (reference-aligned, 2026-02-18)** | |
| `fig_AI3_v3.py` | AI-3 reference-aligned style (dotted waveform, ±2/±4 per track) |
| `fig_AI3_v3_hurst_style.png` | Single-panel AI-3: FC-1..FC-10, dotted, no envelope |
| `fig_AI3_v3_comparison.png` | Side-by-side: our AI-3 vs reference scan |
| `fig_AI4_v3_overlay.py` | AI-4 reference image underlay + 3 method overlays |
| `fig_AI4_v3_overlay_pt.png` | PT half-period (2nd event) overlaid on reference |
| `fig_AI4_v3_overlay_pp.png` | PP full-period (2nd peak) overlaid on reference |
| `fig_AI4_v3_overlay_zc.png` | ZC half-period (midpoint) overlaid on reference |
| `fig_AI4_v3_overlay_all.png` | All 3 methods on reference underlay (different colors) |
| `fig_AI4_v3_spaced.py` | Spaced filter comparison (0.2 / 0.3676 / 0.4 / 0.6 r/y) |
| `fig_AI4_v3_spaced_grid.png` | 4×2 grid: 4 spacings × (dots \| lines) |
| `fig_AI4_v3_spaced_harmonic.png` | Single panel: harmonic-aligned (0.3676 r/y), lines |
| `fig_AI4_v3_spaced_compare.png` | All 4 spacings overlaid on one axis |
| `fig_AI4_v3_final.py` | Final AI-4 reproduction (PT raw, amplitude-gated, no smoothing) |
| `fig_AI4_v3_final.png` | Hurst-style single panel: gated PT, no smoothing |
| `fig_AI4_v3_4panel.png` | Reference + PT all + PT gated + harmonic-gated comparison |
| `fig_AI4_v3_final_overlay.png` | Reference underlay + gated PT overlay |
| **Spacing (decimation) experiment (2026-02-18)** | |
| `fig_AI4_v3_spacing_output.py` | Tests `spacing` parameter (decimation) on comb bank |
| `fig_AI4_v3_spacing_grid.png` | 4-panel: spacing=1/4/7/10 PT measurements |
| `fig_AI4_v3_spacing_s7_overlay.png` | spacing=7 PP overlay on reference image |
| `fig_AI4_v3_spacing_density.png` | Point count per filter vs spacing (PP solid / PT dashed) |
| `fig_AI4_v3_spacing_waveforms.png` | spacing=1 vs spacing=7 waveforms for 4 filters |

---

## 11. Decimation Spacing Experiment (2026-02-18)

Script: `fig_AI4_v3_spacing_output.py`

### Question
Does the `spacing` parameter (decimation factor) in `apply_filter_bank()` reproduce Hurst's
FVT point density and visual pattern? Does analytic (complex) Ormsby support it?

### Answer: Yes — infrastructure is fully compatible

`apply_filter_bank(signal, filters, spacing=N)`:
- Decimates input to every N-th sample
- Auto-redesigns kernels: `nw_dec = nw // N` (same time span, fewer taps)
- Places output back in full-length array with **NaN gaps** between computed positions
- Nyquist constraint: raises ValueError if `f4 > π × fs/N` — spacing=14 violated this for
  FC-20 (`f4=12.4 > π×52/14=11.65`), so spacings [1, 4, 7, 10] were tested instead

All 10 remaining spacings are safe: Nyquist at spacing=10 is π×52/10=16.34 > 12.4 rad/yr.

### Results

| spacing | nw_dec | PP avg/filter | PT avg/filter | PP vs Hurst (7) |
|---------|--------|---------------|---------------|-----------------|
| 1       | 3501   | **6.8**       | 14.9          | 0.98x ✅        |
| 4       | 875    | 6.7           | 14.4          | 0.96x           |
| 7       | 500    | 6.5           | 14.0          | 0.93x           |
| 10      | 350    | 6.4           | 12.3          | 0.91x           |

### Key Finding: spacing does NOT change measurement density

The PP count stays ~6.5–6.8/filter regardless of spacing. This is because:
- Each filter's effective bandwidth and center frequency are unchanged by decimation
- The number of complete cycles in the 269-week display window depends only on `fc`
- PP = 1 measurement per full cycle → `n_cycles ≈ n_weeks / T_wk` is spacing-independent

**What spacing does change:**
- The *time resolution* of the waveform dots (NaN gap = spacing weeks between points)
- The kernel tap count (fewer taps = faster, same time-span)
- The visual appearance: spacing=7 gives dotted waveform with ~7-week steps between dots

### PT vs PP Density

- **PT (peak→trough interleaved)**: ~2 measurements per full cycle → ~14 pts/filter
  - This is because PT gives both P→T AND T→P = 2 half-periods per full period
- **PP (peak→peak)**: ~1 measurement per full cycle → ~7 pts/filter = Hurst's density
- **Conclusion**: Hurst's ~7 pts/filter matches PP (full-period) measurements, not PT

### Recommendation for AI-4 reproduction

Use PP measurement (not PT), with **spacing=1** (baseline, most accurate peak positions).
The `spacing` parameter is useful for:
- Quickly prototyping (fewer taps = faster convolution)
- Visualizing which weeks are "active" in the filter output
- Exactly mimicking what Hurst would have computed if he used a decimated filter design

**spacing=7 is NOT needed** to reproduce Hurst's density — the density is set by the
measurement method (PP vs PT), not the spacing.

### Analytic mode confirmation

`create_filter_kernels(..., filter_type='modulate', analytic=True)` is fully compatible
with `spacing`. The bandpass centers (7.6–12.0 rad/yr) are well below the decimated
Nyquist (16.34 rad/yr at spacing=10). No approximation errors observed.
