# Methods, and what each one is for

A unified testing framework for external forcing hypotheses of earthquake
triggering. Every component below exists because of one specific problem, not
because of convention. The problem numbers are cross-referenced throughout.

---

## Summary: eleven problems, eleven remedies

| # | problem | remedy | where |
|---|---|---|---|
| 1 | Aftershocks are not independent samples | Gardner-Knopoff declustering | every module |
| 2 | The catalogue has grown fourfold since 1970 | Target M 6.0 and above, where the rate is flat | `unified_test.py` |
| 3 | Astronomical variables encode the date (R squared 0.995) | Flat target plus a shift null | `forcing_bank.py` |
| 4 | High base rate (29% of days) | Report the baseline, use AUC and ratios | everywhere |
| 5 | Enormous hypothesis space | Max-statistic null | `unified_test.py` |
| 6 | Free parameters, such as window width | Parameter-free tests, or sweep and correct | `analysis.py` |
| 7 | Temporal leakage in machine learning | Blocked time validation | `solar_real.py` |
| 8 | No comparison model | Always report the model without the variable under test | `combined_model.py` |
| 9 | Julian against Gregorian calendar | Explicit conversion | `lunar_ephem.py` |
| 10 | Co-seismic contamination in GRACE | Causal sampling | `grace_hydro2.py` |
| 11 | A global variable cannot name a location | Per-event spatial sampling | `grace_hydro2.py` |

---

# A. Data

Every source is public and can be downloaded again. Links are in
[`README.md`](../README.md).

| data | source | span | used for |
|---|---|---|---|
| Earthquake catalogue | USGS ComCat | 1900 to 2026 | the target |
| Focal mechanisms | Global CMT | 1976 to 2026 | resolving stress onto the fault |
| Ocean tides | GOT4.10c (NASA GSFC) | 8 constituents | sea-floor loading |
| Geomagnetic indices | Kp and ap (GFZ Potsdam) | 1932 to 2026 | the SOLAR family |
| Sunspot number | SILSO (Belgium) | 1818 to 2026 | the SOLAR family |
| F10.7 radio flux | NRCan (Canada) | 2004 to 2026 | the SOLAR family |
| Solar wind | OMNI | 1996 to 2024 | proton density |
| Terrestrial water storage | GRACE TWS (GFZ GravIS) | 2002 to 2026 | the HYDRO family |
| Ocean bottom pressure | GRACE OBP (GFZ GravIS) | 2002 to 2026 | HYDRO offshore |
| Ephemerides | Meeus plus JPL elements | any epoch | positions of the Moon and planets |

---

# B. Preprocessing

## B.1 Declustering, for problem 1

Aftershocks are not independent events. One M9 drops hundreds of aftershocks at
almost the same lunar phase, and a standard statistical test treats them as
hundreds of free samples. That inflates any apparent periodicity.

**Method:** the space-time windows of Gardner and Knopoff (1974). When
comparing against a specific paper, their definition is used instead (150 km
and 6 months, for Marchitelli).

**Measured effect:** 13,148 events become 9,024 mainshocks, discarding 31.4%.
In the tidal test, harmonic 2 falls from **p = 0.0152 to p = 0.596** after
declustering. That "significant" signal was entirely aftershocks.

## B.2 Choice of target, for problems 2 and 3

The global M 4.5 catalogue rises from 7.08 to 20.61 earthquakes per day between
the 1970s and the 2010s. That is the growth of the seismograph network, not
tectonics. Any variable that is smooth in time will "predict" that trend.

This is not a hypothesis. A random forest predicts the **date** from nine
Earth-to-planet distances at **R squared = 0.995**, with a median error of 23
days out of a 55-year range.

**Method:** the target is restricted to **M 6.0 and above**, where the
catalogue is effectively flat (0.30 to 0.43 events per day in every decade, a
linear trend of +0.0019 per year). With nothing changing in time, leaking the
date buys nothing.

---

# C. Test statistic

## C.1 The main statistic

For every variable: divide the days into 10 quantile bins, compute the
earthquake rate per bin, and take the **largest deviation from the overall
rate**.

This reads directly as "modulation of plus or minus X%", so it is comparable
across families. Angles, stresses, solar indices and water loads all produce
one number with the same meaning.

## C.2 Parameter-free tests, for problem 6

The SSGEOS claim uses a window of plus or minus 2.95 days. Sweeping the window
from 0.5 days to a maximum of 3.69 days in 64 steps shows that the published
window is the **global minimum of p**. Only 5 of 64 reach p below 0.05, and the
one chosen is the best of them.

**Method:** the Schuster test (harmonics 1, 2 and 4) and the V-test have no
parameter to tune. Where a window-based statistic is still used, the entire
sweep enters the multiplicity correction (section E).

## C.3 Metrics that do not mislead, for problem 4

An M6 occurs somewhere on 29% of days. An accuracy of 84% can mean "always
guess no".

**Method:** always report side by side the base rate, a trivial baseline, AUC,
and information gain against climatology. Accuracy is never reported alone.

---

# D. The null model

## D.1 Circular time shift, the core of the framework

A null must destroy **only** the relationship under test and preserve
everything else.

**Method:** shift the whole earthquake series in time by a random offset, with
wraparound. This preserves:

- whatever aftershock clustering remains
- the secular trend of the catalogue
- reporting seasonality
- the marginal distribution of every variable
- the smoothness and periodicity of every variable

and destroys only the temporal alignment between earthquakes and forcing.

**Why not a random permutation:** shuffling blocks destroys the **smoothness**
of the features. A tree model overfits smooth features more easily than jagged
ones, so the real data and the shuffled data are not judged on equal terms. We
made this mistake once, and the result fell 5 sigma below the noise mean, which
is the signature of an unequal comparison rather than of a negative effect.

## D.2 Evidence that a naive null fails, for problem 3

A standard Schuster test on the raw catalogue reports:

| pair | naive p | calibrated p |
|---|---|---|
| Mercury and Mars | 0.0000 | **0.739** |
| Venus and Mercury | 0.0000 | **0.725** |
| Jupiter and Saturn | 0.0000 | **0.966** |

Every one of those "detections" is the time structure of the catalogue read as
planetary influence. Mercury and Venus are never far from the Sun as seen from
Earth, so their angles follow the year; Jupiter and Saturn move slowly enough
that a 45-year trend aliases into them.

---

# E. Multiplicity correction, for problem 5

With 340 variables, about 17 will reach p below 0.05 by chance alone.

**Method: a max-statistic null.**

1. Compute the statistic for every variable on the real data, and record the **best**
2. Shift the earthquake series, recompute **every** statistic, and record the best again
3. Repeat hundreds of times
4. p is the fraction of shifted datasets whose best beats the real best

Because step 2 searches as hard as step 1, the p-value is already corrected for
however many variables are added. **Adding variables cannot inflate it**, it
raises the bar the winner has to clear.

Verified directly: pure noise reaches p at or below 0.008 in **9.9%** of trials
when given the same freedom of search that SSGEOS used.

---

# F. For machine learning models

## F.1 Blocked time validation, for problem 7

A label of "an earthquake occurs in the next 48 hours" makes neighbouring rows
share 47 of 48 hours of history and carry almost always the same label. A
random split puts near-duplicates in the training set **and** the test set.

**Measured effect, on real data with no signal:**

| target | chronological AUC | random-split AUC |
|---|---|---|
| Global M 6 and above | 0.478 | 0.876 |
| Indonesia M 6 and above | 0.482 | 0.973 |
| Indonesia M 7 and above | 0.448 | **0.990** |

The random-split metrics for Indonesia M 7 (accuracy 0.987, precision 0.527,
recall 0.943, F1 0.676) are **comparable to those the LSTM paper reports**
(0.845, 0.681, 0.837, 0.751), from data whose chronological split shows
nothing.

**Method:** expanding-window forward validation. Train on the past, test on the
future, never the reverse and never interleaved.

## F.2 A mandatory comparison, for problem 8

**Method:** every model is reported alongside an identical model with the
variable under test removed.

The strongest example, on El Moudden et al.'s own GitHub data:

| model | R squared | accuracy |
|---|---|---|
| 9 planetary distances (their setup) | 0.687 | 66.54% |
| **date column only** | 0.645 | **69.19%** |

A model containing no planetary variable at all wins on their headline metric.

The strictest form is to measure the **incremental contribution** above a
strong baseline. Sky geometry above ETAS gives **-0.00305 nats per day**, which
is negative.

---

# G. Specific remedies

## G.1 Calendars, for problem 9

Every date before 15 October 1582 is recorded in the **Julian** calendar.
Feeding it to a Gregorian ephemeris shifts the lunar phase by the difference
between the calendars, 1 to 10 days over this range, while the Moon covers a
quarter of its cycle in 7.4 days.

**Effect:** 3 of 8 events before 1582 **change phase classification**
depending on which calendar is assumed.

**Method:** `lunar_ephem.py` implements both calendars explicitly and reports
the difference per event.

## G.2 Co-seismic contamination, for problem 10

**This is the most dangerous trap found**, because its result is positive.

GRACE measures co-seismic gravity change: a large earthquake moves crustal
mass, and the satellite sees it. A rate statistic that uses the month **after**
an earthquake therefore partly contains the signature of that earthquake.

The null does not catch it. When the times are shifted, an event no longer
samples its own co-seismic signal, so the real data is contaminated, the null
is clean, and the difference appears as a "finding".

**Measured effect:**

| sampling version | statistic | p |
|---|---|---|
| Lookahead (crossing the earthquake) | absolute rate | **0.0000** |
| **Causal (only months before)** | absolute rate | **0.3600** |

**Method:** every forcing variable is sampled **only from times before** the
earthquake. For the monthly GRACE rate: the difference between month a and
month a minus 1, both preceding the event.

The literature confirms this contamination is real and needs correcting
(RECOG RL01, *ESSD* 2021).

## G.3 Spatial specificity, for problem 11

Lunar phase, planetary aspects and solar activity are **global scalars**: their
value is identical for Aceh and Chile at the same second. A global quantity
structurally cannot name a location, and that is the fundamental defect that
disqualifies all three as a basis for prediction.

**Method:** where a variable has spatial structure, score every earthquake with
the value **at its own location**:

- **Tidal stress:** the Coulomb stress change is computed at the earthquake's
  own latitude and longitude, with the hour angles of the bodies, then resolved
  onto the fault plane from the GCMT focal mechanism
- **Water load:** GRACE TWS or OBP in the earthquake's own grid cell in the
  month of the earthquake

For the null, **time is shifted but location is held**, so the spatial pattern
of seismicity and the spatial pattern of water loading both stay intact, and
only the temporal link is destroyed.

## G.4 Offshore coverage, enlarging the sample without lowering the magnitude

85% of shallow M 6 and above earthquakes are at sea, where the land GRACE
product is empty. Discarding them leaves 289 events and a weak null.

**Method:** use **ocean bottom pressure** (`resobp`, GFZ GravIS) for offshore
events and TWS for land events. Physically they are the same thing: a mass of
water pressing on the crust.

Units are unified to pascals: TWS in cm of water times 98.07; OBP in hPa times
100.

**Effect:** 289 becomes **1,685 events (90.2% coverage)**, a factor of 5.8,
with the magnitude threshold **unchanged**.

---

# H. Validation: proving the pipeline can detect

A null result means nothing if the code is broken, because both produce the
same output.

**Method: positive controls.** Plant a signal in the real data, then run the
**whole** chain again from scratch: ephemerides, potential, strain, fault
resolution, phase extraction, Schuster.

| control | expected | recovered | R/n | verdict |
|---|---|---|---|---|
| Every event moved to its own **maximum** Coulomb stress | 0 degrees | **0.00** | 1.0000 | pass |
| Every event moved to its own **minimum** | 180 degrees | **178.8** | 0.927 | pass |
| Ocean-loaded version, moved to the maximum | 0 degrees | **0.00** | 1.0000 | pass |

If the coordinate frame, the sign of the shear traction, or the phase
construction were wrong, none of these would return to where it was planted.

**Power analysis** completes it: injecting a sinusoidal modulation of amplitude
A into n = 3,965 mainshocks, detection at p below 0.0167 is reached 80% of the
time for A = 8% and 99.5% for A = 12%. The null is therefore a **measurement**,
not a shortage of power.

**Null calibration** is checked separately: 5,000 synthetic catalogues with no
effect trigger p below 0.05 in 4.4%, 5.1% and 4.8% of cases for harmonics 4, 2
and 1, which is as it should be.

---

# I. Output: upper bounds, not verdicts

The product of this framework is not "significant or not significant" but **a
number bounding each hypothesis**.

| family | modulation bound | relative equivalent |
|---|---|---|
| HYDRO | 2.88% | 10.0% |
| TIDAL | 2.97% | 10.3% |
| LUNAR | 3.41% | 11.8% |
| SOLAR | 5.66% | 19.6% |
| PLANETARY | 9.03% | 31.3% |
| **ETAS (reference)** | **9.19% measured** | **31.8%** |

The ETAS row is not a bound but a measured effect. It supplies the scale that
makes the other bounds meaningful.

---

# J. Physical context: the stress ladder

Every result above is consistent with one ordering of magnitudes:

| source | stress at the fault |
|---|---|
| Stress transfer from a neighbouring earthquake | 10^5 to 10^7 Pa |
| Ocean tidal loading | about 10^4 Pa |
| Water load, measured at earthquake locations | about 1.1 x 10^3 Pa |
| Lunar body tide | 723 Pa |
| Venus (with Earth rotation included) | 3.9 x 10^-3 Pa |
| Jupiter (with Earth rotation included) | 1.1 x 10^-3 Pa |
| Neptune | 5.9 x 10^-7 Pa |

The result is set by the size of the forcing, not by the model. A neighbouring
earthquake pushes roughly a billion times harder than Venus.

---

# K. A note on honesty

During the development of this framework, **three false positives** appeared
and were killed by the components above:

1. **p = 0.05** from a V-test on tidal amplitude quartiles, which failed the quintile scaling check (slope -0.024, permutation p = 0.808)
2. **p = 0.0000** for half the planetary pairs, which failed against a calibrated null (all became p above 0.4)
3. **p below 0.0001** for the GRACE water-load rate, which failed under causal sampling (becoming p = 0.36)

All three looked convincing when they first appeared. All three were artefacts.

One habit catches them all: **ask where that number could have come from if not
from physics**, then test that guess directly.
