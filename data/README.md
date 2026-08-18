# Folder map

Eight families of external forcing are tested in this repository, at **two
levels of resolution**. The split matters, and here is why:

| | family | why it is separated |
|---|---|---|
| 🌍 **global** | 🌙 Moon · 🪐 Planets · 🌀 Rotation · ☀️ Sun | the forcing really is the same everywhere on Earth on a given day, so one value per day **discards nothing** |
| 📍 **field** | 🌊 Tides · 💧 Hydrology · 🌬️ Atmosphere · 📡 Ionosphere | the forcing is **local**, so it has to be read at the earthquake's own location |

Global averaging discards 98% of the local variation
([`resolution_test.py`](../src/unified/resolution_test.py)). The four field
families are therefore tested per event rather than per day.

## Status

| topic | what is tested | folder | status |
|---|---|---|---|
| 🌙 **Moon** | cardinal lunar phase (the SSGEOS claim) | [`src/lunar/`](../src/lunar/) | collapses |
| 🪐 **Planets** | planetary geometry and aspects | [`src/planetary/`](../src/planetary/) | collapses |
| ☀️ **Sun** | solar activity and solar wind | [`src/solar/`](../src/solar/) | split artefact |
| 🌀 **Rotation** | length of day, polar motion | [`src/rotation/`](../src/rotation/) | null |
| 🌊 **Tides** | tidal stress plus ocean loading | [`src/tidal/`](../src/tidal/) | null (p = 0.79) |
| 💧 **Hydrology** | GRACE water load plus ocean bottom pressure | [`src/hydro/`](../src/hydro/) | null (p = 0.06) |
| 🌬️ **Atmosphere** | surface pressure loading | [`src/atmosphere/`](../src/atmosphere/) | null (p = 0.11) |
| 📡 **Ionosphere** | TEC precursor anomalies | [`src/ionosphere/`](../src/ionosphere/) | null (p = 0.90) |
| 🔗 **Unified** | all eight at once, one correction | [`src/unified/`](../src/unified/) | joint p = 0.19 |
| 📊 **Forecasting** | what actually works | [`src/forecasting/`](../src/forecasting/) | **works** |

```
lunar-earthquake/
├── README.md             overview, data sources, how to reproduce
├── docs/
│   ├── INDEX.md          ← you are here
│   ├── METHODS.md        eleven problems and eleven remedies
│   └── CONCLUSIONS.md    final conclusions
├── data/                 see the data section below
└── src/
    ├── _bootstrap.py     path setup, the only file that knows the layout
    ├── common/           shared modules
    ├── lunar/ tidal/ planetary/ solar/          ← the first four topics
    ├── hydro/ rotation/ atmosphere/ ionosphere/ ← four further families
    ├── unified/          all eight combined
    └── forecasting/      the comparison that works
```

To run: `python src/<topic>/<script>.py` from anywhere. Paths are handled by
[`src/_bootstrap.py`](../src/_bootstrap.py).

---

## 🔗 `src/unified/`, all eight families at once

The core of this repository. One catalogue, one statistic, one null, one
correction, so that the eight hypotheses can be compared directly.

| script | what it does |
|---|---|
| `forcing_bank.py` | builds **130 variables** from 8 families on one daily grid |
| `unified_test.py` | upper bound per family, circular-shift null, max-statistic correction |
| `resolution_test.py` | **measures the cost of global averaging**: r squared = 1.8%; the per-event design catches a 5% modulation where the global daily design needs 40 to 80% |
| `perevent_test.py` | the four field families read **at their own epicentres**; the null preserves the season |
| `power_by_family.py` | a real null or too little data? detection threshold per family |
| `lag_sweep.py` | 10 lags by 130 variables, whether any forcing **precedes** an earthquake |
| `ml_perevent.py` | matched case-control learning at full resolution, with a positive control and a leakage audit |
| `deep_test.py` | LSTM over 30-day sequences of all 130 variables |
| `combined_predictor.py` | everything as one predictor; contribution above ETAS |

**Result:** joint p = 0.19. Contribution of the sky above ETAS: **-0.0220 AUC**.
In the matched per-event design the real day is ranked first in 6.1% of pools
against a chance of 10.3%, while the seismicity reference reaches 31.0%.

---

## 🪐 `src/planetary/`, planetary geometry

Tests the claims of Awadh (2021) and El Moudden et al. (2024).

| script | what it does |
|---|---|
| `planetary.py` | JPL elements; aspect coverage (97.6% of days are "critical" at plus or minus 3 degrees); calibrated angle tests |
| `planetary_tide.py` | planetary tidal stress with Earth rotation included: Venus 0.0039 Pa against the Moon at 723 Pa |
| `planetary_ml.py` | 68 geometry features, blocked time validation, AUC 0.4933 |
| `combined_model.py` | contribution of geometry **above** ETAS, -0.00305 nats per day |
| `deep_planetary.py` | **340 variables**, max-statistic null, p = 0.88 |
| `step_by_step.py` | no model at all, just a rate table per bin |
| `independent_test.py` | pre-registered test across a different era (1900 to 1969), the sign of the effect reverses |

**Result:** nothing survives. Bound: modulation below 9.0%.

---

## ☀️ `src/solar/`, solar activity

Tests Altaibek et al. (2024), *Atmosphere* 15, 1290, an LSTM on proton density,
and behind it Marchitelli et al. (2020).

| script | what it does |
|---|---|
| `solar_real.py` | **the main one**, real Kp/ap, sunspot number and F10.7; chronological against random split |
| `solar_only.py` | the Kp/ap version, three targets (global, Indonesia M6, M7) |
| `lstm_leakage.py` | demonstrates the leakage mechanism on synthetic data |
| `proton_test.py` | hourly OMNI proton density, full version |
| `proton_light.py` | light version: 7 summary statistics per variable, 40 seconds |

**Result:** chronological AUC 0.45 to 0.50, a coin toss. Random-split AUC as
high as **0.99**, reproducing the published metric with no signal present.

---

## 🌙 `src/lunar/`, lunar phase (SSGEOS)

| script | what it does |
|---|---|
| `audit_data.py` | catalogue integrity; 37 of 55 events predate 1900, 10 sit on whole hours |
| `analysis.py` | replication plus the tests that were omitted; a window of plus or minus 2.95 days is the best of 64 |
| `honest_test.py` | pre-registered test on the USGS catalogue |

---

## 🌊 `src/tidal/`, real tidal stress

| script | what it does |
|---|---|
| `tidal_stress.py` | degree-2 potential, Love strain, fault-resolved Coulomb stress |
| `tidal_analysis.py` | lunar phase against tidal stress, on the same events |
| `ocean_load.py` | GOT4.10c to sea-floor load (strengthens megathrusts by a factor of 8) |
| `ocean_analysis.py` | megathrusts tested with the ocean term |
| `validate.py` | **positive control**, plant a signal and recover it at 0.00 degrees |
| `directional_test.py` | V-test plus an amplitude scaling check |
| `seasonal.py` | seasonal water loading; Indonesia below 3.3% |
| `ide_bvalue_test.py` | the size-frequency form of the claim (Ide et al. 2016): does b shift with tidal amplitude? |

**Note:** in the global bank, tidal stress is evaluated at a single reference
site and returns p = 0.048. Evaluated per event on each earthquake's own fault
plane it returns **p = 0.79**. Raising the resolution killed that false
positive.

---

## 💧 `src/hydro/` · 🌀 `src/rotation/` · 🌬️ `src/atmosphere/` · 📡 `src/ionosphere/`

The four families added after the first four topics.

| script | what it does |
|---|---|
| `hydro/grace_hydro.py` | GRACE water load, first version (land only, 15% of earthquakes) |
| `hydro/grace_hydro2.py` | plus ocean bottom pressure, reaching **90% of earthquakes**, units unified to Pa |
| `rotation/lod_test.py` | length of day and polar motion from IERS EOP |
| `atmosphere/fetch_pressure.py` | NCEP/NCAR, only the 939 cells containing earthquakes (75 MB rather than 850 MB) |
| `atmosphere/pressure_test.py` | per-cell anomaly (30-day running median), causal sampling |
| `ionosphere/fetch_tec.py` | UPC-IonSAT IONEX **1998 to 2024**, parallel and resumable |
| `ionosphere/tec_test.py` | TEC anomaly above the epicentre, 1 to 5 days **before** the earthquake |

**The stress ladder**, which is why this ordering matters:

| source | stress |
|---|---|
| earthquake stress transfer | 10^5 to 10^7 Pa |
| ocean tidal loading | about 10^4 Pa |
| GRACE water load at epicentres | about 1,130 Pa |
| lunar body tide | 723 Pa |
| atmospheric pressure anomaly | 563 Pa |
| Venus (with rotation included) | 0.0039 Pa |
| Neptune | 5.9 x 10^-7 Pa |

---

## 📊 `src/forecasting/`, what actually works

| script | what it does |
|---|---|
| `forecast.py` | spatio-temporal ETAS, CSEP scored, **863 times better than the lunar model** |
| `early_warning.py` | warning times and the blind zone (about 33 km) |
| `daily_index.py` | a working daily index, runnable at any time |

---

## 🔧 `src/common/`, used by every topic

| script | what it does |
|---|---|
| `ephem_vec.py` | vectorised ephemerides |
| `lunar_ephem.py` | Meeus, plus Julian and Gregorian calendars |
| `gcmt.py` | Global CMT parser (NDK format) |
| `fetch_gcmt.py` | extends the catalogue from the monthly archive; verifies the parse before overwriting |
| `parse_ssgeos.py` | reader for the SSGEOS datasets, present because the original repository has none |

---

## 📁 `data/`

| folder | contents |
|---|---|
| `catalogs/` | `gcmt.ndk` (**70,671** focal mechanisms, 1976 to 2026) · `gcmt_to2020.ndk` (backup) · `global_m6.csv` · regional catalogues · `m6_days.npy` · `m6_cnt.npy` |
| `solar/` | `kp_ap.txt` (GFZ) · `sunspot.txt` (SILSO) · `f107.txt` (NRCan) · `omni_hourly.csv` |
| `hydro/` | `grace_tws.nc` · `grace_obp.nc` (GFZ GravIS) |
| `ocean_tide/` | `got410/` (GOT4.10c, NASA GSFC) |
| `rotation/` | `iers_eop.txt` (IERS) |
| `atmosphere/` | `pressure_cells.npy` · `pressure_days.npy` · `pressure_cellid.npy` (NOAA PSL) |
| `thermal/` | outgoing longwave radiation, air temperature, precipitable water (NOAA PSL) |
| `ionosphere/` | `tec_daily.npy` (**9,250 days, 1998 to 2024**) · `tec_days.npy` · `_parts/` (per year, for resuming a download) |
| `external/` | `FinalDataEARTH.csv` (from the El Moudden GitHub repository) |
| `results/` | saved output of every test (`*_results.txt`) |

Every source is public and none requires a login. Download links for all of
them are in [`README.md`](../README.md).

New files are always written through `datafile()` in
[`src/_bootstrap.py`](../src/_bootstrap.py), which searches recursively by name,
so moving a file between subfolders does not break any script.
