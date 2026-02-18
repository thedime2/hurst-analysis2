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

## 4. Parabolic Sub-Sample Interpolation

Hurst's Appendix describes fitting a parabola through 3 measurement points for interpolation.
For peak timing, we use:

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
Parabolic interpolation reduces this to <0.1%. Already implemented in all final scripts.

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

Hurst says "special purpose filters for frequency analysis." Key possibilities:

1. **Wider passband**: If his ±0.3 rad/yr passband (vs our ±0.1) admitted adjacent
   harmonics, the filter output would be a two-tone signal with significant FM.
   This would explain the larger oscillation amplitude in Hurst's top cluster.

2. **Different nw (filter length)**: Shorter filter kernels → wider transition bands
   → more adjacent harmonic leakage → more beating → more frequency variation.

3. **Exact comb filter frequencies**: Hurst says filters were arranged with
   "identically equal frequency spacings" — same as our design (0.2 rad/yr step).
   The center frequencies 7.6, 7.8, ..., 12.0 rad/yr are confirmed.

4. **Measurement by "phase derivative"**: Hurst may have used an electronic frequency
   discriminator circuit, which is the analog equivalent of dφ/dt. This gives the
   smoothed instantaneous frequency trace we see in his hand-drawn AI-4 figure.

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

## 8. Output Files Summary

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
| `fig_AI4_cmw_phase.py` | CMW vs Ormsby FVT comparison (NEW) |
| `fig_AI4_cmw_4panel.png` | Reference, Ormsby PT+TP-5, CMW PT+TP-5, CMW PDC+gate |
| `fig_AI4_cmw_vs_ormsby.png` | Side-by-side: Ormsby vs CMW (both PT+TP-5) |
| `fig_AI4_cmw_best.png` | Single-panel best CMW (PDC + gate) |
| `fig_AI4_cmw_waveforms.png` | Ormsby vs CMW waveforms for 4 selected filters |
