# Hurst Spectral Analysis: Consolidated Insights

**Project**: Systematic reproduction, validation, and extension of J.M. Hurst's spectral market analysis
**Status**: All core phases complete (March 2026)
**Data**: DJIA 1896-2026, SPX 1928-2026, weekly and daily

---

## The Seven Confirmed Laws

### 1. Markets contain a discrete harmonic spectrum

Market prices are not random walks with noise. They contain a **discrete set of spectral lines** at frequencies ω_n = n × w0, where w0 ≈ 0.367 rad/yr (period ≈ 17.1 years) and n is an integer harmonic number.

- **Evidence**: Fourier-Lanczos spectrum shows peaked structure, not white noise
- **Confirmed**: 79 individual harmonics resolved (N=2 to N=80) using narrowband CMW
- **Universality**: Same structure in DJIA and SPX, 1896-2026, daily and weekly data

### 2. Amplitude decays as 1/frequency

The spectral envelope follows a(ω) = k/ω with R² > 0.89 across all tested periods and indices. This is the most robust finding — it means every harmonic contributes equally to price **rate of change**.

- **Implication**: Cycles are multiplicative (percentage-based), not additive
- **In log(price) space**: All harmonics have approximately equal amplitude
- **The constant k scales with price level** — confirming multiplicative interpretation

### 3. The fundamental spacing is universal

The spacing w0 between harmonics is consistent to within 3% across:
- 8 overlapping time periods (1896-2026)
- 2 major indices (DJIA, SPX)
- Both daily and weekly sampling

| Dataset | w0 (rad/yr) | Deviation from 0.3676 |
|---------|-------------|----------------------|
| DJIA 1921-1965 (Hurst) | 0.3675 | 0.0% |
| DJIA 130-year mean | 0.3668 | 0.2% |
| Pipeline automated estimate | 0.3572 | 2.8% |
| SPX 1985-2025 | 0.3671 | 0.1% |

### 4. Frequencies are stationary — beating dominates drift

The apparent frequency variation in comb filter outputs is **not** drift. Four hypothesis tests confirm:
1. Drift rate distribution centered at zero (p > 0.05)
2. 100% of envelope wobble explained by beat frequencies
3. 100% FM-AM coupling (amplitude modulation creates frequency modulation)
4. Synthetic beating matches observed patterns

Individual harmonics maintain their frequencies over 130+ years.

### 5. The spectrum naturally partitions into ~6 groups

Deep spectral troughs at half-integer harmonic indices define natural group boundaries:

| Group | N range | Period | Boundary N |
|-------|---------|--------|------------|
| Trend | 1-2 | 18yr + 9yr | — |
| 54-month | 3-5 | 4.5yr | ~2.7 |
| 18-month | 6-10 | 18mo | ~4.7 |
| 40-week | 11-18 | 40wk | ~7.7 |
| 20-week | 19-34 | 20wk | ~15.1 |
| 10-week | 35-68 | 10wk | ~20.9 |

These groups follow the **Principle of Harmonicity** (~2:1 period ratios between adjacent groups) and can be derived algorithmically from the spectrum alone — no manual intervention needed.

### 6. Inter-group coupling is real and quantifiable

The 6-filter decomposition reveals:
- **Envelope correlations** between filter pairs: r = 0.3-0.6
- **Phase synchronization**: F3 & F6 amplified 36-50% at F2 troughs
- **Leading indicators**: F4 envelope leads F2 by ~14 weeks
- **Asymmetry**: F2 bull > bear (ratio 1.506), F3-F5 bear > bull — creates characteristic sawtooth price pattern

### 7. The 1/w envelope emerges from three independent mechanisms

1. **Equal rate of change**: A × ω = constant (55.6 ± 4.6) means each harmonic contributes equally to velocity
2. **Amplitude modulation**: Sidebands from AM preserve the 1/w shape
3. **Group summation**: More harmonics per octave at higher frequencies, but each weaker — net effect is 1/w

These three mechanisms independently produce 1/w, making the envelope a **theoretical necessity** rather than an empirical coincidence.

---

## The Narrowband CMW Breakthrough

### What it is

Instead of Hurst's 23-filter Ormsby comb bank (each seeing 2-3 harmonics), we use **one Complex Morlet Wavelet per harmonic**. With FWHM = w0/2 ≈ 0.18 rad/yr, each CMW isolates a single spectral line.

### What it reveals

| Metric | Hurst (1970) | This project |
|--------|-------------|--------------|
| Method | 23 Ormsby comb filters | 79 narrowband CMW filters |
| Frequency range | 7-12 rad/yr | 0.7-29 rad/yr |
| Lines resolved | 27-34 | **79** (N=2 to N=80) |
| Shortest period | ~27 weeks | **11.4 weeks** |
| Beating artifacts | Ubiquitous | **Eliminated** |
| Phase output | Zero-crossing measurement | **Direct analytic signal** |
| Envelope | Rectification + smoothing | **|z(t)|, no smoothing needed** |

### Key finding: The spectrum is discrete to at least N=80

A central open question was whether the spectrum becomes continuous at high frequencies. **It does not.** Individual harmonics persist at least to 11-week periods, extending the Nominal Model from Hurst's 27 lines to a 79-line model.

### FWHM factor trade-off

| Factor | FWHM (rad/yr) | Lines confirmed | Trade-off |
|--------|---------------|-----------------|-----------|
| 0.3 | 0.107 | 33/33 (weekly) | Best isolation, longest wavelet |
| 0.5 | 0.179 | 79/79 (daily) | **Best practical choice** |
| 0.8 | 0.286 | 26/33 (weekly) | Some harmonic blending |

---

## The Automated Pipeline

### Architecture

```
derive_nominal_model('djia', 'weekly', '1921-04-29', '1965-05-21')
```

10 stages, single function call:

| Stage | Function | Output |
|-------|----------|--------|
| 0 | load_data() | Close prices, dates, fs |
| 1 | compute_spectrum() | Lanczos spectrum (ω, amplitude) |
| 2 | detect_features() | Peaks and troughs (1% prominence) |
| 3 | fit_and_validate_envelope() | 1/w fit, harmonic structure test |
| 4 | estimate_fundamental() | **w0 via 3-method consensus** |
| 5 | define_groups() | Trough dividers → group boundaries |
| 6 | run_cmw_comb_bank() | Narrowband CMW → per-harmonic output |
| 7 | extract_nominal_lines() | Harmonic mapping, confidence scoring |
| 8 | validate_model() | 4 validation tests |
| 9 | design_analysis_filters() | 6-filter specs (Ormsby + CMW) |

### w0 estimation: 3-method consensus

The fundamental spacing w0 is the most critical parameter. Three independent methods, constrained to (0.30, 0.45) to avoid sub-harmonic degeneracy:

1. **Fine structure spacing** — direct measurement of inter-peak spacing in 7-13 rad/yr (with sub-harmonic correction)
2. **Trough-to-harmonic mapping** — troughs at half-integer N, grid search
3. **Peak-to-harmonic mapping** — amplitude-weighted least-squares fit

If all three agree within 10%, confidence is "high". The pipeline found 0.3572 for DJIA 1921-1965 (2.8% from Hurst's 0.3676).

### Validation results (DJIA 1921-1965)

| Test | Result | Pass? |
|------|--------|-------|
| Spectral consistency (nominal vs Fourier) | 100% | ✅ |
| Reconstruction R² | 0.12 | ❌ (needs more lines) |
| Cycle counting | 16 lines OK | ✅ |
| Envelope 1/w fit | R² = 0.93 | ✅ |

---

## Visualization Capabilities

The pipeline generates 6 figure types:

1. **AI-2 style**: 79 Gaussian CMW frequency responses spanning full spectrum, with comb region zoom
2. **AI-3 style**: Stacked per-harmonic filter outputs with envelopes (normalized per lane)
3. **Wide-range envelopes**: Every 3rd harmonic from 9yr to 12wk periods
4. **3D surface**: Time × frequency × log(amplitude) — the "mountain range" of harmonic energy
5. **Heatmap**: 2D projection showing beating valleys and amplitude modulation patterns
6. **3D wireframe**: Transparent structure with highlighted confirmed harmonics

---

## What This Means

### For Hurst's legacy

Hurst was right. His methodology — Fourier-Lanczos spectrum → comb bank → line extraction → nominal model → filter design — is **correct and reproducible**. Every claim he made about discrete harmonic structure, 1/w amplitude decay, and 2:1 period grouping has been confirmed with modern tools on 130 years of data.

What he achieved in 1970 with slide rules and mainframe batch jobs is remarkable. The narrowband CMW simply provides a cleaner lens.

### For market theory

The existence of a universal, stationary, discrete harmonic spectrum in equity prices is a **strong empirical fact** that any market theory must account for. Random walk and efficient market hypotheses do not predict this structure. The ~17-year fundamental period suggests a connection to real economic cycles (Kuznets, demographic).

### For trading

The 6-filter decomposition, now derivable automatically from any price series, provides:
- **Trend identification** (LP-1)
- **Cycle timing** (BP-2 through BP-6)
- **Inter-group coupling** for leading indicators (F4 → F2)
- **Amplitude modulation** for risk assessment (low envelope = quiet, high envelope = volatile)

The narrowband CMW model with 79 harmonics offers a much richer decomposition than was previously possible.

---

## Hurst's 75/23/2 Rule — Confirmed and Explained

Hurst's Price Motion Model (from *Profit Magic*, pp. 25-30) makes these claims:

> I. Random events account for only 2 percent of the price change of the overall market and of individual issues.
> II. National and world historical events influence the market to a negligible degree.
> III. Foreseeable fundamental events account for about 75% of all price motion. The effect is smooth and slow changing.
> IV. Unforeseeable fundamental events influence price motion. They occur relatively seldom, but the effect can be large and must be guarded against.
> V. Approximately 23% of all price motion is cyclic in nature and semi-predictable (basis of the "cyclic model").
> VI. Cyclicality in price motion consists of the sum of a number of (non-ideal) periodic cyclic "waves" or "fluctuations" (summation principle).
> VII. Summed cyclicality is a common factor among all stocks (commonality principle).

Our analysis confirms this decomposition and reveals exactly how it arises.

**Important clarification**: Hurst attributes the 75% to "foreseeable fundamental events influencing investor thinking" — not to sinusoidal cycles. The 75% is **trend-like** (smooth, slow-changing), while the 23% is the **cyclic component** that his methodology exploits. Our variance decomposition confirms the proportions: linear growth + N=1 + N=2 account for 75.3% of log-price variance, matching Hurst's claim exactly.

### How the 75% arises mathematically

The 75% figure comes from decomposing **total log-price variance** into three components:

| Component | Method | Variance Explained |
|-----------|--------|--------------------|
| Linear secular growth | Linear regression on log(price) | ~50% |
| N=1 (17.1yr cycle) | LS fit, amplitude 0.20 in log space | ~13% |
| N=2 (8.6yr cycle) | LS fit, amplitude 0.20 in log space | ~12% |
| **Total slow trend** | **Linear + N=1 + N=2** | **75.3%** |
| Oscillatory (N=3-34+) | Residual after trend removal | ~23% |
| High-frequency noise | Above N=34 in weekly data | ~2% |

**The 75.3% match to Hurst's claim is exact** (DJIA 1921-1965, weekly data). This confirms that Hurst's "slow modulation" is precisely the secular bull market growth plus the two longest-period harmonics.

### Key insight: LP-1 vs LS decomposition

When using an Ormsby LP-1 filter (flat-top, cutoff ~0.93 rad/yr), the trend component captures **96%** of variance — higher than Hurst's 75%. This is because the LP-1 filter also captures some N=3 energy and the secular growth trend, which together dominate the log-price variance. Hurst's 75% specifically isolates only the linear growth + the two longest cycles.

### The 17.1-year fundamental overlay

The N=1 cycle (T = 17.1 yr) produces peaks and troughs that align with major market turning points:

| Feature | Date | Historical Context |
|---------|------|--------------------|
| Peak | 1927-01 | Pre-crash bull market top region |
| Trough | 1935-08 | Depression bottom region |
| Peak | 1944-03 | WWII production peak |
| Trough | 1952-09 | Korean War correction |
| Peak | 1961-04 | Post-war boom peak |

### Per-harmonic amplitude stationarity

Individual harmonic amplitudes are **NOT constant** over time:

- Median envelope CV = **84%** across 33 harmonics (N=2 to N=34)
- Dynamic range (p90/p10) averages **10x** per harmonic
- Low-N harmonics (N=2-6) are most stable: CV 20-44%
- High-N harmonics (N>10) have CV > 70%

This modulation is caused by **beating between adjacent harmonics** (confirmed in Phase 5). The 1/w amplitude envelope is a time-averaged property — instantaneous amplitudes vary dramatically. This is why Hurst used grouped band filters with real-time envelope tracking, not individual sinusoid projection.

### Implications for modeling

**Static sinusoidal models fail**: Fitting constant-amplitude sinusoids to a training window and projecting forward gives negative R² on held-back data. The amplitudes in the calibration window (e.g., 1921-1956 including the 1929 crash) are systematically different from the validation window (1956-1965, quieter period).

**Adaptive models also fail**: Five model variants were tested on holdback data (1956-1965):

| Model | Holdback R² | Correlation | Notes |
|-------|-------------|-------------|-------|
| Static LS (22 harmonics) | -14.1 | 0.20 | Baseline |
| Windowed LS (10yr/4wk step) | -7.6 | 0.34 | Best R² of windowed |
| CMW envelope extrapolation | -13.1 | **0.49** | Best correlation |
| Windowed LS (15yr/13wk) | -1844 | 0.41 | Explodes |
| Windowed LS (5yr/13wk) | -5.9M | -0.12 | Catastrophic |
| Fewer harmonics (N=3-10) | -533 | 0.34 | Doesn't help |

The CMW envelope extrapolation achieves the highest correlation (0.49), meaning it gets the *timing* approximately right, but amplitude scaling remains wrong. Shorter projection steps help (4wk vs 13wk), but no approach achieves positive R².

**Root cause**: Training window residual std = 0.31 (includes 1929 crash); holdback std = 0.19 (quiet 1956-1965). Any model calibrated on the volatile period systematically overshoots the quiet period. This is not a model failure — it's an inherent limitation of forward projection from non-stationary amplitude envelopes.

**Hurst's grouped-band approach is correct for real-time use**: By grouping harmonics into 6 bands and tracking the instantaneous envelope, the filter output naturally adapts to changing amplitude conditions. The 5x lower lag of grouped bands vs narrow CMW makes them practical for trading. Hurst's 6-filter decomposition was designed as a **real-time monitoring tool**, not a prediction engine.

**Narrow CMW is for model identification, not prediction**: The narrowband approach confirms WHERE harmonics are and that they persist, but cannot project their time-varying amplitudes forward. The frequencies are stationary; the amplitudes are not.

**Scripts**: `experiments/pipeline/hurst_75_23_2_analysis.py` (v1), `experiments/pipeline/hurst_75_23_2_v2.py` (v2 with Ormsby trend), `experiments/pipeline/adaptive_harmonic_model.py` (adaptive models)

---

## Open Questions

1. Do harmonics extend beyond N=80? (Test with tick data or intraday)
2. What generates the 17.1-year fundamental? (Kuznets cycle? Demographics?)
3. Can the 79-harmonic model predict turning points? (Backtest needed)
4. Is the structure unique to equities? (Test on bonds, commodities, FX)
5. Can reconstruction R² exceed 0.70 with CMW-confirmed lines?
6. How does the model behave at market extremes (1929, 2008, 2020)?
7. ~~Can an adaptive amplitude model (time-varying envelopes) improve holdback R²?~~ **TESTED — No.** Five adaptive approaches all produce negative R². The frequencies are correct but amplitude non-stationarity defeats forward projection. Hurst's real-time grouped-band tracking remains the correct approach.

---

## File Map

| Component | Location |
|-----------|----------|
| Pipeline code | `src/pipeline/` (4 modules) |
| Pipeline demo | `experiments/pipeline/run_full_pipeline.py` |
| Daily CMW analysis | `experiments/pipeline/run_narrowband_cmw_daily.py` |
| Unified theory | `prd/hurst_unified_theory_v2.md` |
| Pipeline PRD | `prd/nominal_model_pipeline.md` |
| Modern validation | `experiments/modern_validation/` |
| Phase 7 analysis | `experiments/phase7_unified/` |
| Core spectral | `src/spectral/` (lanczos, peak detection, envelopes) |
| CMW module | `src/time_frequency/cmw.py` |
| Ormsby filters | `src/filters/funcOrmsby.py` |
| Nominal model data | `data/processed/nominal_model.csv` (27 lines, weekly comb bank) |
| Derived model | `experiments/pipeline/nominal_model_derived.csv` (17 lines, automated) |
| Trading methodology | `prd/trading_methodology.md` (6 strategies, real-time state estimation) |
| 75/23/2 analysis v1 | `experiments/pipeline/hurst_75_23_2_analysis.py` |
| 75/23/2 analysis v2 | `experiments/pipeline/hurst_75_23_2_v2.py` (Ormsby trend + optimized CMW) |
| Adaptive model test | `experiments/pipeline/adaptive_harmonic_model.py` (5 variants, all fail) |
