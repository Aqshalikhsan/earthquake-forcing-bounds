# Conclusions

Closed 17 August 2026. Every number below is produced by code in
[`src/`](../src/) and can be re-run.

---

## The original question

Is SSGEOS, the claim that the geometry of the planets, the Sun and the Moon can
be used to anticipate earthquakes, real and possible?

It grew into a test of **three papers** (`papers/`) and finally into
the most thorough examination of this hypothesis that has been carried out.

---

## Scale of the work

| | |
|---|---|
| Earthquakes analysed | 70,671 focal mechanisms (Global CMT) plus the USGS catalogue, 1900 to 2026 |
| Sky variables tested | **340** across three reference frames, aspects, tidal stress and barycentre |
| Third-party data | GOT4.10c (NASA), Kp/ap (GFZ), sunspot number (SILSO), F10.7 (NRCan), OMNI |
| Statistical tests | Schuster, V-test, Kolmogorov-Smirnov, binomial, permutation, max-statistic, Monte Carlo |
| Positive controls | passed; a planted signal returns at 0.00 degrees, R/n = 1.0000 |

The only synthetic data is in `lstm_leakage.py`, which exists to demonstrate a
mechanism.

---

## Results: three claims

### 🌙 Lunar phase (SSGEOS)

Their arithmetic is **correct** and their ephemeris is **accurate to 0.6
minutes**. What collapses is the statistics:

- a window of plus or minus 2.95 days is the **best of 64** possibilities, and it covers **79.9%** of the lunar cycle
- the window-free test (Schuster h4) gives **p = 0.81**
- 37 of 55 events predate 1900, and 10 sit on whole hours because their times were never measured
- 3 of 8 events before 1582 **change classification** depending on the Julian or Gregorian calendar
- pure noise reaches p at or below 0.008 in **9.9%** of trials given the same freedom of search

### 🪐 Planetary geometry

- **340 variables** against a max-statistic null that corrects for the whole search: **p = 0.88**
- the real data carries **less** structure than shuffled data (5.60% against 7.58%)
- an independent test in a different era (1900 to 1969) **reverses** the sign of the effect (ratio 0.95)
- at a tolerance of plus or minus 3 degrees, **97.6% of days** qualify as "critical"
- contribution above ETAS: **-0.00305 nats per day**, which is negative

### ☀️ Solar activity

- chronological AUC: **0.45 to 0.48** on all three targets, below a coin toss
- random-split AUC on the same data: as high as **0.990**
- the random-split metric is **comparable to the one the paper reports**, from data with no signal in it

---

## What does matter: the earthquakes themselves

| | spatio-temporal ETAS | lunar model |
|---|---|---|
| Information gain | **+1.81 nats per earthquake** | +0.002 |
| Skill ratio | | **863 : 1** |
| Molchan, 1% of volume | **45.6%** caught | 23.0% (the background is 23.2%) |

Aftershock forecast verification, 62 mainshocks of M 6.5 or above:

| | aftershocks |
|---|---|
| ETAS forecast | 106 |
| Time-independent forecast | 1.8 |
| **Observed** | **288** |

The static model is wrong by a factor of **161**.

**There is one reason: the stress ladder.**

| source | stress at the fault |
|---|---|
| transfer from a neighbouring earthquake | 100,000 to 10,000,000 Pa |
| ocean tidal loading | 10,000 Pa |
| lunar body tide | 723 Pa |
| Venus (with Earth rotation included) | **0.004 Pa** |

A neighbouring earthquake pushes roughly a billion times harder than Venus. The
model is not the limitation. The variables carry no information.

---

## Three concrete errors in El Moudden (2024)

Replicated from their own GitHub data, giving R squared 0.687 against the 0.67
they report, and then:

1. **A date-only model wins** on their headline metric: 69.19% against 66.54%
2. **The dataset is described incorrectly.** 98.8% of dates are exactly 2 days apart, which is the theskylive grid, not "only days with earthquakes"
3. **The 2011 spike is attributed to comets**, although the top row of their own data is the M9.1 Tohoku earthquake, and the two named comets were discovered in 2021 and 2023

---

## What is honestly still open

1. **Slow planetary pairs.** Uranus and Neptune need 171 years per cycle and the instrumental catalogue is 55 years old. Nobody can test this.
2. **Seasonal water loading in Indonesia.** The bound below 3.3% is loose because the USGS catalogue is too shallow there. The stress is about 10 kPa, which makes it **the most promising direction left**. It needs the BMKG catalogue.
3. **Multi-year and decadal modulation.** A trend-preserving null cannot see it, so the claims of Bendick and Bilham (2017) and Dumont et al. (2025) are neither tested nor excluded here.
4. **The split method used by Altaibek et al.** One sentence from the authors would close the question.

---

## Methodological lessons

Five traps explain every failure here, and two of them caught us during this
work:

1. Earthquakes cluster, and aftershocks are not independent samples
2. The catalogue has grown fourfold since 1970 because of instrumentation, not tectonics
3. Astronomical variables encode the date (R squared = 0.995)
4. The base rate is high: an M6 occurs on 29% of days
5. The hypothesis space is enormous: 97.6% of days can be called "critical"

**One rule catches all of them:** always report what the model achieves
**without** the variable under test. If the answer is "about the same", there is
no finding yet.

None of the three papers does this.

---

## What can be done with this

**Closest to publishable:** a replication and comment on El Moudden (2024). The
data is public, the errors reproduce exactly, and a 340-variable search with a
multiplicity correction has not been done before. A careful negative paper
becomes the reference every time this claim resurfaces after a large
earthquake, which it will.

**Most useful locally:** calibrating ETAS with the BMKG catalogue. ETAS here is
wrong by a factor of 2.7 because it was trained on USGS data. With a BMKG
catalogue the result is directly usable for aftershock warnings.

**Most interesting scientifically:** seasonal water loading in Indonesia. The
physics is strong, the question is open, and the pipeline already exists.

---

## Closing note

Frank Hoogerbeets appears to be sincere. His ephemeris is accurate, he refutes
the eclipse and supermoon myths correctly, and he corrects media that distort
him. What fails is the methodology, not the intent. These traps catch real
researchers in real journals.

And the most important point: **nothing found here changes what protects
people.** Yogyakarta 2006, Palu 2018 and Cianjur 2022 all arrived on ordinary
days, with no warning sign, inside the blind zone of early warning. All three
were survivable in a building constructed to code.

That remains the answer.
