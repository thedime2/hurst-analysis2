# Trading Methodology: Real-Time Cyclic State Estimation

## Premise

Forward projection of individual harmonic amplitudes **fails** (5 adaptive models tested, all negative R-squared on holdback). The frequencies are stationary and known to high precision; the amplitudes are non-stationary (median CV=84%, 10x dynamic range). This rules out any "predict and trade" approach based on sinusoidal extrapolation.

Hurst understood this. His methodology was never about prediction — it was about **real-time state estimation**. The 6-filter decomposition is a monitoring tool that tells you WHERE you are in each cycle RIGHT NOW, and inter-cycle relationships tell you what's LIKELY to happen next.

This document synthesizes all confirmed findings into a practical methodology.

---

> **WARNING — ERA-SPECIFIC ARTIFACTS (March 2026 Coupling Validation)**
>
> **Strategies 2 (F4 Leading Indicator) and 4 (Phase Synchronization)** were validated only on Hurst-era data (1935-1954). Cross-era coupling validation (`experiments/pipeline/validate_coupling_modern.py`) found these effects **REVERSE on modern data**:
>
> - **F4→F2 leading**: r=+0.38 (Hurst era) → r=−0.17 (1985-2025) — **REVERSED**
> - **F3/F6 trough amplification**: +36-50% (Hurst era) → ~0.8× suppressed (modern) — **REVERSED**
>
> The walk-forward backtest handles this via per-window adaptive weighting, but **do not use Strategies 2 or 4 with fixed parameters on post-1965 data**. The stable, universal strategies are: **1 (Synchronicity), 3 (Amplitude Regime), 5 (Asymmetry), and 6 (Band Calibration)**.

---

## What We Can and Cannot Do

### CAN do (confirmed, quantified):
1. **Track instantaneous phase** of 6 cycle bands in real time (lag: 5-64 weeks)
2. **Detect multi-band alignment** (synchronicity at troughs/peaks)
3. **Monitor envelope magnitude** to classify amplitude regime (strong/weak cycles)
4. **Use F4 as leading indicator** for F2 (14-week lead, r=0.38)
5. **Exploit asymmetry**: F2 bull phases last 50% longer than bear; F3-F5 opposite
6. **Anticipate beating patterns** from known harmonic spacing (predictable amplitude modulation)
7. **Measure amplitude range** from recent N cycles to calibrate expected move size

### CANNOT do:
1. Predict absolute amplitude at any future time
2. Foresee "unforeseeable fundamental events" (Hurst's Point IV — 1929, 2008, 2020 shocks)
3. Project individual harmonic envelopes more than ~4 weeks forward
4. Determine whether a quiet period will stay quiet or turn volatile

---

## The Six Inputs

All derived from the standard 6-filter decomposition applied to log(price):

| Filter | Band | Period | Lag | Role |
|--------|------|--------|-----|------|
| LP-1 | Trend | >18yr | ~27yr half-life | Direction bias |
| BP-2 | 54-month | 3.8yr | ~64wk | Primary cycle |
| BP-3 | 18-month | 1.3yr | ~27wk | Intermediate |
| BP-4 | 40-week | 0.7yr | ~15wk | Leading indicator |
| BP-5 | 20-week | 0.4yr | ~9wk | Short-term timing |
| BP-6 | 10-week | 9wk | ~5wk | Entry precision |

For each bandpass filter, the analytic signal (via native analytic Ormsby filter with `analytic=True`) provides:
- **Instantaneous phase** theta(t): where in the cycle we are (0=trough, pi=peak)
- **Envelope** |z(t)|: instantaneous amplitude (how strong this cycle is NOW)
- **Instantaneous frequency** d(theta)/dt: cycle speed (confirms nominal or drifted)

---

## Strategy 1: Synchronicity Scoring

### Principle
When multiple cycle bands reach troughs simultaneously, the combined effect produces a strong rally. This is Hurst's "Principle of Synchronicity" — confirmed by our backtesting showing every major bottom (1932, 1974, 2003, 2009, 2020) has alignment score = -1.0.

### Implementation

For each bandpass filter i (i=2..6), compute a phase score:

    s_i(t) = -cos(theta_i(t))

where theta_i is the instantaneous phase (0 at trough, pi at peak). This gives:
- s_i = -1 at trough (maximum bullish)
- s_i = +1 at peak (maximum bearish)
- s_i = 0 at zero crossings

The composite alignment score:

    S(t) = (1/5) * sum(s_i(t), i=2..6)

ranges from -1 (all troughs aligned) to +1 (all peaks aligned).

### Trading rules

| S(t) | Regime | Action |
|------|--------|--------|
| S < -0.6 | Strong trough alignment | Maximum long exposure |
| -0.6 < S < -0.2 | Mild trough alignment | Moderate long |
| -0.2 < S < +0.2 | Mixed/neutral | Reduced position or flat |
| +0.2 < S < +0.6 | Mild peak alignment | Moderate short or flat |
| S > +0.6 | Strong peak alignment | Maximum short or defensive |

### Enhancement: Amplitude-weighted scoring

Not all alignments are equal. Weight each band's score by its current envelope relative to its historical range:

    S_w(t) = sum(w_i * s_i(t)) / sum(w_i)

where w_i = |z_i(t)| / median(|z_i|). This gives more weight to bands that are currently "active" (high envelope) and less to bands in beating troughs (low envelope).

---

## Strategy 2: F4 Leading Indicator

### Principle
F4 (40-week) envelope growth leads F2 (54-month) envelope growth by ~14 weeks (r=0.38, p<0.05). Rising F4 volatility signals incoming F2 volatility.

### Implementation

Track the 13-week rate of change of F4 envelope:

    dE4(t) = (|z4(t)| - |z4(t-13)|) / |z4(t-13)|

When dE4 crosses above a threshold (e.g., +50%), this is an **early warning** that F2 amplitude is about to increase. Combined with F2 phase:

| F2 Phase | dE4 Rising | Interpretation |
|----------|-----------|----------------|
| Approaching trough | Yes | Major rally forming — prepare for strong long entry |
| Approaching peak | Yes | Major decline forming — prepare to exit/hedge |
| Mid-cycle rising | Yes | Acceleration of existing uptrend |
| Mid-cycle falling | Yes | Deepening of existing correction |

### Lookback
The 14-week lead means you get approximately one quarter of advance notice before F2 makes a major amplitude move. This is enough time to adjust portfolio positioning.

---

## Strategy 3: Amplitude Regime Classification

### Principle
Per-harmonic amplitudes are non-stationary with median CV=84% and 10x dynamic range. But the amplitude at time t provides information about the CURRENT regime. A cycle with high current envelope is "active" and tradeable; a cycle with low envelope is in a beating trough and should be ignored.

### Implementation

For each filter i, classify the current envelope into terciles based on the last N cycles:

| Regime | Envelope Percentile | Interpretation |
|--------|-------------------|----------------|
| STRONG | > 67th | Active cycle, large moves expected |
| NORMAL | 33rd - 67th | Typical behavior |
| WEAK | < 33rd | Beating trough, cycle suppressed |

The lookback window should be 3-5 complete cycles of that band:
- F2 (3.8yr): 11-19 years lookback
- F3 (1.3yr): 4-7 years
- F4 (0.7yr): 2-4 years
- F5 (0.4yr): 1-2 years
- F6 (9wk): 0.5-1 year

### Trading rules

1. **Only trade cycles that are in STRONG or NORMAL regime.** A cycle in WEAK regime (beating trough) is unreliable — its zero crossings are noise, not signal.

2. **Position size proportional to active cycles.** If 4 of 5 bands are STRONG, take large positions. If only 1 band is STRONG, trade small.

3. **Expect the regime to change.** Beating is periodic — a WEAK regime will eventually become STRONG. The beating period between adjacent harmonics is approximately:

        T_beat = 2*pi / (w_n+1 - w_n) = 2*pi / w0 = 17.1 years

    But in practice, multi-harmonic beating creates modulation on shorter timescales (2-5 years for mid-frequency bands).

---

## Strategy 4: Phase Synchronization Exploitation

### Principle
F3 and F6 amplitudes increase 36-50% when F2 is at a trough. This creates explosive V-shaped reversals at major bottoms.

### Implementation

When F2 approaches a trough (theta_2 in [5*pi/3, 2*pi] or [0, pi/3]):
1. **Expect** F3 amplitude to be ~36% higher than average
2. **Expect** F6 amplitude to be ~50% higher than average
3. The combined effect produces sharp reversals — price drops accelerate then reverse explosively

**Conversely**, when F2 approaches a peak:
- F3 and F6 are relatively suppressed
- Tops are rounded, gradual
- This creates the asymmetric sawtooth: sharp bottoms, slow tops

### Trading rules at F2 troughs

1. When F2 enters trough zone AND F3/F6 envelopes are rising: prepare for sharp reversal
2. Enter long when F5 or F6 crosses zero (rising) while F2 is in trough zone
3. Set tight stops — the reversal should be rapid if the synchronization is real
4. If the reversal doesn't materialize within 2-3 F6 cycles, exit (false alarm)

### Trading rules at F2 peaks

1. F2 peaks are drawn out (bull phase 50% longer than bear)
2. Don't try to time the exact top — the rollover is gradual
3. Instead, wait for F4 to confirm: when F4 phase turns from rising to falling after F2 peak zone, the decline is starting
4. This is a "confirm then act" approach, sacrificing some of the move for higher probability

---

## Strategy 5: Asymmetry Exploitation

### Confirmed asymmetry ratios

| Filter | Up/Down Duration Ratio | Implication |
|--------|----------------------|-------------|
| F2 | 1.506 (bull longer) | Hold longs longer, cut shorts sooner |
| F3 | 0.565 (bear longer) | Intermediate corrections drag on |
| F4 | 0.674 (bear longer) | Same pattern at shorter timescale |
| F5 | 0.301 (bear longer) | Short-cycle corrections prolonged |

### Trading rules

1. **In F2 rising phase**: Expect intermediate corrections (F3-F5) to be drawn out. Don't panic — the F2 bull phase has more room to run (average 121 weeks up vs 81 weeks down).

2. **In F2 falling phase**: F2 bear phases are shorter (81 weeks) but sharper. Exit quickly when F2 turns — don't wait for confirmation.

3. **Corrections within uptrends**: F3-F5 bear > bull means that corrections look scarier than they are. Use the prolonged correction (F3 avg 47wk down) as an accumulation opportunity if F2 is still rising.

4. **Rallies within downtrends**: F3-F5 bear > bull also means that counter-trend rallies are brief and sharp. Don't chase them.

---

## Strategy 6: Nominal Model Band Calibration

### Principle
The nominal model gives us the EXPECTED periods for each band. Real-time instantaneous frequency tells us whether the current cycle is running fast, slow, or nominal. This provides information about what to expect next.

### Implementation

For each filter, compute the ratio:

    R_i(t) = f_nominal_i / f_instantaneous_i(t)

| R_i | Status | Interpretation |
|-----|--------|---------------|
| 0.8-1.2 | Nominal | Cycle behaving as expected |
| > 1.2 | Slow | Cycle stretched — amplitude likely higher |
| < 0.8 | Fast | Cycle compressed — amplitude likely lower |

### Why this matters

The Principle of Variation (Hurst) states that actual cycle periods fluctuate +/-30% around nominal. But this variation is not random — it correlates with amplitude. Stretched cycles tend to have higher amplitude (energy conservation in the frequency domain), and compressed cycles have lower amplitude.

---

## Integrated Decision Framework

### Step 1: Compute state vector

At each time step, compute:
- Phase of each band: theta_i(t)
- Envelope of each band: |z_i(t)|
- Synchronicity score: S(t) and S_w(t)
- Amplitude regime per band: STRONG / NORMAL / WEAK
- F4 envelope rate of change: dE4(t)
- F2 phase zone: trough / rising / peak / falling

### Step 2: Classify market regime

| F2 Zone | S_w(t) | dE4 | Regime |
|---------|--------|-----|--------|
| Trough | < -0.4 | Rising | **MAJOR BOTTOM** — maximum long |
| Rising | < -0.2 | Any | **BULL TREND** — stay long, buy dips |
| Peak | > +0.4 | Rising | **MAJOR TOP** — exit or short |
| Falling | > +0.2 | Any | **BEAR TREND** — flat or short |
| Any | Near 0 | Flat | **NEUTRAL** — reduced position |

### Step 3: Size the position

    Position = Base_Size * Regime_Factor * Amplitude_Factor * Confidence_Factor

Where:
- Regime_Factor: 1.0 (major bottom/top), 0.7 (trend), 0.3 (neutral)
- Amplitude_Factor: proportion of bands in STRONG/NORMAL regime (0.2 to 1.0)
- Confidence_Factor: 1.0 if multiple strategies agree, 0.5 if conflicting signals

### Step 4: Set expectations

Use the amplitude regime and asymmetry ratios to set realistic targets:
- Expected move size: current envelope * asymmetry ratio for the dominant band
- Expected duration: nominal half-period * asymmetry ratio
- Stop loss: based on recent cycle amplitude range (use p90 of last 5 cycles)

---

## What This Is Not

This methodology does NOT:
1. **Predict turning points with precision** — it estimates WHERE in each cycle we are and what's likely next
2. **Replace fundamental analysis** — Hurst's Point IV (unforeseeable events) can override any cyclic signal
3. **Guarantee profits** — the 23% cyclic component is "semi-predictable" (Hurst's word), not deterministic
4. **Work as a black box** — the signals require judgment about macro context and risk tolerance

The edge comes from three sources:
1. **Structure** that random walk theory denies but 130 years of data confirms
2. **Quantified inter-band coupling** that provides leading indicators and amplification signals
3. **Amplitude context** that adapts position sizing to current cycle strength

---

## Implementation Status

All four phases are **COMPLETE** as of March 2026.

### Phase A: Real-time dashboard (COMPLETE)
- 6-filter decomposition using **analytic Ormsby filters** (not Hilbert transform)
- Phase, envelope, instantaneous frequency extracted directly from complex filter output
- Amplitude regime classification per band (STRONG/NORMAL/WEAK terciles)
- F4 leading indicator and synchronicity scoring
- Script: `experiments/pipeline/backtest_trading_methodology.py`

### Phase B: Synchronicity backtest (COMPLETE)
- Historical S_w(t) computed across DJIA 1927-2025 and SPX 1985-2025
- Alignment score confirms every major bottom (1932, 1974, 2003, 2009, 2020) at S=-1.0
- Sharpe 0.442 vs 0.319 B&H, max drawdown -43% vs -221%
- Script: `experiments/pipeline/backtest_trading_methodology.py`

### Phase C: Integrated framework + walk-forward (COMPLETE)
- All 6 strategies combined as multiplicative modifiers
- Walk-forward validation: 15yr train / 5yr test sliding windows
- **DJIA**: avg Sharpe 2.10 across 11 windows (vs B&H 0.55)
- **SPX**: avg Sharpe 1.69 across 10 windows (vs B&H 0.59)
- Adaptive strategy measures coupling per window and downweights unstable strategies
- Script: `experiments/pipeline/walkforward_backtest.py`

### Phase D: Live monitoring (COMPLETE)
- Weekly state vector update for DJIA and SPX
- Alert system: F2 zone transitions, sync threshold crossings, amplitude regime changes
- Position sizing recommendations based on current state
- 6-panel dashboard figure with 260-week history
- Current state (March 2026): both DJIA and SPX in NEUTRAL regime, 4/5 bands WEAK
- Script: `experiments/pipeline/live_dashboard.py`

---

## Quantitative Foundation

All strategies rest on empirically tested relationships. Cross-era coupling validation (`experiments/pipeline/validate_coupling_modern.py`) tested 4 eras: DJIA 1921-1965, 1965-2005, 1985-2025, and SPX 1985-2025.

| Finding | Confirmation | Stability |
|---------|-------------|-----------|
| 79 harmonic frequencies | Narrowband CMW, R2=0.9999 | **UNIVERSAL** — 130 years, 2 markets |
| 1/w amplitude envelope | R2>0.89 in all periods | **UNIVERSAL** |
| 6-band decomposition | 98.9% variance captured | **UNIVERSAL** |
| Synchronicity at major bottoms | Score=-1.0 at all 5 | **UNIVERSAL** — 1932-2020 |
| Beating dominates drift | 4/4 hypothesis tests | **UNIVERSAL** |
| F2 bull/bear asymmetry | 1.506 ratio (Hurst) → 2.2-3.6 modern | **STABLE** — stronger on modern data |
| F2-F3 / F3-F6 envelope correlation | r=0.3-0.6 | **STABLE** across all eras |
| F3/F6 amplified at F2 troughs | 36-50% increase (Hurst era) | **REVERSED** on modern data (ratio ~0.8) |
| F4 leads F2 by 14 weeks | r=+0.38 (Hurst era) | **REVERSED** on modern data (r=-0.17) |

### Critical Findings from Coupling Validation

**Strategies 1, 3, 5, 6 are STABLE** — synchronicity scoring, amplitude regime classification, asymmetry exploitation, and band calibration work consistently across all eras.

**Strategies 2 and 4 are HURST-ERA ARTIFACTS** — F4→F2 leading and F3/F6 trough amplification are specific to the 1935-1954 display window and reverse on modern data. The walk-forward backtest handles this by measuring coupling per window and downweighting these strategies when coupling is weak or reversed.

The adaptive walk-forward approach (Sharpe 2.10 DJIA, 1.69 SPX) only marginally outperforms fixed-parameter sync strategy (2.07, 1.73), confirming that the stable strategies (particularly synchronicity) carry most of the edge.

---

## References

- Hurst, J.M. *The Profit Magic of Stock Transaction Timing* (1970), pp. 25-30 (Price Motion Model)
- Hurst, J.M. *Cycles Course* (Cyclitec Services, 1973-1975)
- Project findings: `prd/hurst_unified_theory_v2.md` (Parts 10-11, 20)
- Coupling analysis: `experiments/phase7_unified/fig_hidden_relationships.py`
- Coupling validation (cross-era): `experiments/pipeline/validate_coupling_modern.py`
- Backtest implementation: `experiments/pipeline/backtest_trading_methodology.py`
- Walk-forward optimization: `experiments/pipeline/walkforward_backtest.py`
- Live dashboard: `experiments/pipeline/live_dashboard.py`
- Adaptive model failure: `experiments/pipeline/adaptive_harmonic_model.py`
- 75/23/2 confirmation: `experiments/pipeline/hurst_75_23_2_analysis.py`
