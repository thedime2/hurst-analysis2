# How Hurst Derived the 6 Page-152 Band-Pass Filters

## The Question

Hurst's page 152 decomposes the DJIA into 6 frequency bands using Ormsby band-pass filters. The book does not explicitly publish the filter specifications or explain why these particular 6 frequencies were chosen. This document reconstructs the derivation logic from primary sources and our own spectral analysis.

---

## The Short Answer

Each filter targets **one dominant nominal cycle** from Hurst's empirically-derived hierarchy. The 6 filters follow the **Principle of Harmonicity** (~2:1 period ratios between adjacent cycles), and the bandwidths accommodate the **Principle of Variation** (~20-30% period fluctuation). The gaps between filters are intentional -- they correspond to frequency ranges where the DJIA has minimal spectral energy.

---

## The Derivation Chain

### Step 1: Fourier-Lanczos Spectrum (Appendix A, Figure AI-1)

Hurst computed a Fourier-Lanczos spectrum of 44 years of weekly DJIA data (1921-1965). This revealed:

- Discrete spectral peaks (not continuous noise)
- Regular fine structure with spacing ~0.37 rad/yr
- Amplitude-frequency envelope: a(w) = k/w

### Step 2: Overlapping Comb Filters (Appendix A, Figures AI-2 to AI-4)

23 narrow Ormsby band-pass filters swept across the 7.6-12.0 rad/yr range confirmed that the spectrum consists of **discrete lines**, not a continuous distribution. This is "The Incredible Frequency-Separation Effect."

### Step 3: Nominal Model (Appendix A, Figures AI-5, AI-6 and Table II-1)

Hurst used "Fourier analysis and a digital discovery algorithm" which he stated "perfectly agreed with each other" to identify the dominant cycles. He published two versions:

**Profit Magic (1970) -- 7 primary cycles:**
- 18.1 yr, 9 yr, 4.7 yr, 2.4 yr, 1.3 yr, 6 mo, 3 mo

**Cyclitec Course (1973-75) -- 11 cycles with strict harmonicity:**
- 18 yr, 9 yr, 54 mo, 18 mo, 40 wk, 20 wk, 80 day, 40 day, 20 day, 10 day, 5 day

The key organizing principle is the **Principle of Harmonicity**: adjacent cycles in the hierarchy are related by simple integer ratios, usually 2:1 (occasionally 3:1).

### Step 4: Filter Design (Page 152)

Each filter is centered on one dominant nominal cycle, with bandwidth wide enough to capture the expected variation but narrow enough to separate adjacent cycles.

---

## The 6 Filters and Their Nominal Cycle Targets

| Filter | Center (rad/yr) | Center Period | Target Cycle | Ratio to Next |
|--------|-----------------|---------------|-------------|---------------|
| LP-1   | < 1.25          | > 5 yr        | 18-yr + 9-yr (trend) | -- |
| BP-2   | 1.65            | 3.81 yr       | 54-month (4.5 yr) | -- |
| BP-3   | 4.95            | 1.27 yr       | 18-month | 3.0:1 |
| BP-4   | 8.55            | 0.73 yr       | 40-week (9 mo) | 1.7:1 |
| BP-5   | 16.65           | 0.38 yr       | 20-week | 1.9:1 |
| BP-6   | 32.35           | 0.19 yr       | 80-day (10 wk) | 1.9:1 |

The period ratios between adjacent filters (3.0, 1.7, 1.9, 1.9) cluster around the 2:1 ratio predicted by the Principle of Harmonicity. The 3:1 jump between BP-2 and BP-3 occurs because the 3-year cycle (which would fill the 2:1 gap) was eliminated in the Cyclitec model revision.

---

## Optimal vs Actual Filter Specs

Comparing the filter center frequencies to the nominal cycle frequencies:

| Filter | Nominal omega (rad/yr) | Actual Center (rad/yr) | Delta |
|--------|----------------------|----------------------|-------|
| BP-2 (54-mo) | 1.40 | 1.65 | +0.25 |
| BP-3 (18-mo) | 4.22 | 4.95 | +0.73 |
| BP-4 (40-wk) | 8.38 | 8.55 | +0.17 |
| BP-5 (20-wk) | 16.76 | 16.65 | -0.11 |
| BP-6 (80-day) | 33.60 | 32.35 | -1.25 |

**Important caveat**: The "Actual Center" values are our user-estimated values from visual inspection of the book's graphics, not Hurst's published specifications. The book does not publish exact filter parameters. The discrepancies could reflect either:
1. Estimation error in our visual reading
2. Hurst deliberately shifted filter centers to capture asymmetric spectral energy
3. Hurst used slightly different nominal periods than the Cyclitec course values

---

## Spectral Energy Distribution

From our Lanczos spectrum analysis:

| Region | Energy |
|--------|--------|
| LP-1 (trend) | 93.2% |
| BP-2 (54-mo) | 3.1% |
| BP-3 (18-mo) | 1.0% |
| BP-4 (40-wk) | 0.2% |
| BP-5 (20-wk) | 0.1% |
| BP-6 (80-day) | <0.1% |
| Gaps between filters | 2.3% |
| **Total captured** | **97.7%** |

The trend dominates by energy, but Hurst's key insight was that **all cycles have equal maximum rate of change** (because a(w) = k/w). This means the smaller, faster cycles contribute equally to price *direction*, even though they contribute less to price *level*.

---

## Why There Are Gaps Between Filters

The gaps are not a flaw -- they are a feature. Because the DJIA has a **line spectrum** (discrete frequencies, not continuous noise), there are frequency ranges with genuinely minimal spectral energy. The filters are designed to capture the lines while rejecting the gaps.

Only 2.3% of spectral energy falls in the gaps. This validates the filter design: the gaps correspond to regions of low spectral density.

---

## Phase 3 Nominal Lines in Filter Passbands

Our Phase 3 analysis identified 27 nominal lines across 2.28-11.95 rad/yr. Of these:
- 1 line falls in BP-2 (54-month region)
- 9 lines fall in BP-3 (18-month region)
- 8 lines fall in BP-4 (40-week region)
- 0 lines fall in BP-5 or BP-6 (these are above the Phase 3 comb bank range)
- 9 lines fall outside all filter passbands

The 9 lines outside the passbands are concentrated at 6.75-7.06 rad/yr (between BP-3 and BP-4) and 10.16-11.95 rad/yr (above BP-4). These may represent:
- Sidebands of the dominant lines (modulation artifacts)
- Additional nominal cycles not captured by the 6-filter decomposition
- Frequency regions Hurst classified as "meaningless"

---

## The Principle of Harmonicity: Why ~2:1?

Hurst discovered that market cycles tend to have periods related by ratios of approximately 2:1 (sometimes 3:1). This is analogous to musical harmonics and suggests a nested oscillatory structure where each cycle contains roughly two cycles of the next shorter period.

This organizing principle determined the filter spacing: if cycles come in 2:1 ratios, you need one filter per octave of frequency to capture the dominant cycle at each level of the hierarchy. Six filters spanning from >5 years down to ~10 weeks cover 5 octaves of the frequency range accessible with weekly data.

---

## Remaining Questions

1. **Were the filter specs published anywhere?** The Cyclitec training course (1177-page scanned PDF) may contain explicit filter specifications, but the document is not OCR-searchable. Manual review of specific chapters would be needed.

2. **Which came first -- the filters or the model?** In the book's presentation, the page 152 decomposition (Chapter IX) appears before Appendix A. But chronologically, Hurst's spectral analysis (1960s) preceded the book's publication (1970). The filters were likely designed iteratively alongside the model development.

3. **Are the Profit Magic filter specs the same as the Cyclitec specs?** Hurst published "slightly different average lengths" in the two works. The Cyclitec course imposed stricter harmonicity, eliminating the 3-year, 1-year, 6-month, and 3-month cycles.

4. **How should we handle the BP-3 to BP-4 gap?** Nine Phase 3 nominal lines fall between filters. Future work could design additional filters to capture these, or use the parametric methods (MPM, Prony) to resolve individual lines within wider filter passbands.

---

## Diagnostic Figure

See `experiments/page_152/filter_derivation_analysis.png` -- shows the Lanczos spectrum overlaid with the 6 filter passbands and Hurst's Table II-1 nominal cycle frequencies.

---

## Sources

- Hurst, J.M., *The Profit Magic of Stock Transaction Timing* (1970), Chapters II, IX, Appendix A
- Hurst, J.M., *Cycles Course* (Cyclitec Services, 1973-1975)
- Sigma-L, "The Nominal Model" -- https://www.sigma-l.net/p/hurst-nominal-model
- Hurst Cycles, "Revisiting the Nominal Model" -- https://hurstcycles.com/revisiting-the-nominal-model/
