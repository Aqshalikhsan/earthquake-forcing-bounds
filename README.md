# Calibrated bounds on external forcing of earthquakes

Thirteen candidate external forcings are tested against one global earthquake
catalogue under a single statistic, a single null hypothesis and a single
multiplicity correction. Because the conditions are held fixed and only the
forcing changes, the results are comparable across hypotheses, and every null
result carries an upper bound stating how large an effect would have been
detected.

The forcings tested are lunar phase and geometry, tidal stress, planetary
geometry, solar activity, hydrological loading, Earth rotation, atmospheric
loading, ionospheric total electron content, and thermal anomalies in three
forms. ETAS and the b-value are included on the reference side rather than as
forcings.

## Headline results

| Result | Value |
| --- | --- |
| Families reaching significance | none of eight |
| Joint corrected p, foreshocks / mainshocks / aftershocks | 0.100 / 0.187 / 0.273 |
| Variance retained when a spatial field is averaged to one value per day | 1.8% |
| Detectable local rate modulation, per-event design vs global design | 5% vs 40 to 80% |
| Positive control, planted 10 to 20% modulation recovered | 85 to 100% of trials |
| False-alarm rate at zero planted signal | 0 to 5% |
| Machine learning, real day ranked first within its matched pool | 6.1% against 10.3% chance |
| Seismicity reference, same folds and model | 31.0%, p < 0.001 |
| AUC swing caused by the validation split alone | 0.208 forcings, 0.008 seismicity |
| Date recoverable from the forcing bank by ridge regression | R squared 0.99 |
| ETAS reference | 9.19 points of modulation, p = 0.005 |

Two findings drive the rest. Reducing a spatial field to a daily global mean
discards about 98% of the local variation, so a null obtained that way bounds
the design rather than the hypothesis. And the metrics reported in the
literature are reproducible by adopting random rather than chronological
validation, on identical models and identical data.

## Where the data comes from

Every source is public and none requires registration. The table gives the
landing page for each product. Scripts under `src/` download most of them
automatically; the rest are noted as manual.

| Family | Product | Download |
| --- | --- | --- |
| Catalogue | Global CMT moment tensors | [globalcmt.org](https://www.globalcmt.org/CMTfiles.html), monthly files at [LDEO](https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY) |
| Catalogue | USGS ComCat | [FDSN event service](https://earthquake.usgs.gov/fdsnws/event/1/) |
| Lunar, planetary | Computed from Meeus and JPL elements | no download, see `src/common/` |
| Tidal | GOT4.10c ocean tide, NASA GSFC | distributed with [pyTMD](https://github.com/pyTMD/pyTMD) |
| Solar | Kp and ap indices, GFZ Potsdam | [kp.gfz.de](https://kp.gfz.de/en/data) |
| Solar | Sunspot number, SILSO | [sidc.be/SILSO](https://www.sidc.be/SILSO/datafiles) |
| Solar | F10.7 solar radio flux | [NRCan space weather](https://www.spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-5-en.php) |
| Solar | OMNI solar wind, hourly | [NASA SPDF](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/) |
| Hydrology | GRACE and GRACE-FO water storage | [GravIS TWS](https://gravis.gfz.de/tws) |
| Hydrology | GRACE ocean bottom pressure | [GravIS OBP](https://gravis.gfz.de/obp) |
| Rotation | IERS Earth orientation parameters | [IERS data products](https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html) |
| Atmosphere | NCEP/NCAR Reanalysis 1 surface pressure | [NOAA PSL](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html) |
| Ionosphere | UPC-IonSAT UQRG global ionosphere maps | [chapman.upc.es](https://chapman.upc.es/tomion/rapid/) |
| Thermal | NOAA interpolated outgoing longwave radiation | [NOAA PSL](https://psl.noaa.gov/data/gridded/data.interp_OLR.html) |
| Thermal | NCEP air temperature and precipitable water | [NOAA PSL](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html) |

Radon is absent from the study and this is a gap in the available data rather
than a result. No global archive of soil-gas radon monitoring exists.

## Further documentation

| document | contents |
| --- | --- |
| [docs/INDEX.md](docs/INDEX.md) | folder map, one table per family, what each script does |
| [docs/METHODS.md](docs/METHODS.md) | eleven problems in this kind of study and the remedy used for each |
| [docs/CONCLUSIONS.md](docs/CONCLUSIONS.md) | what was found, what is still open, and the methodological lessons |

## What is in this repository

`data/` holds the aligned inputs and every results file the manuscript quotes.
`src/` holds the analysis, organised by hypothesis family. `template/` holds
the manuscript and its figures.

```
src/
  common/        catalogue parsing, ephemerides, shared geometry
  lunar/         lunar phase, declination, distance
  tidal/         body tide plus ocean loading, Coulomb stress on each fault
                 plane, and the size-frequency form of the tidal claim
  planetary/     heliocentric and geocentric angles, aspects, planetary tides
  solar/         Kp, ap, sunspot number, F10.7, solar wind
  hydro/         GRACE water storage and ocean bottom pressure
  rotation/      length of day and polar motion
  atmosphere/    NCEP surface pressure
  ionosphere/    UPC-IonSAT total electron content
  thermal/       outgoing longwave radiation, air temperature, water vapour
  seismicity/    b-value
  forecasting/   ETAS reference and operational skill
  unified/       the shared framework, described below
  paper/         figure generation
```

The framework lives in `src/unified/`:

| File | What it does |
| --- | --- |
| `forcing_bank.py` | builds 130 variables on one daily grid, 1970 to 2024 |
| `validity.py` | masks days a source does not actually cover |
| `unified_test.py` | max-statistic test against a circular-shift null |
| `perevent_test.py` | field families read at each earthquake's own location |
| `resolution_test.py` | measures what global averaging discards |
| `positive_control.py` | plants signals of known size and recovers them |
| `ml_perevent.py` | matched case-control machine learning at full resolution |
| `deep_test.py` | LSTM over 30-day sequences of all 130 variables |
| `lag_sweep.py`, `lag_illusion.py` | lead time, and why a best lag is an artefact |
| `foreshock_perevent.py`, `aftershock_perevent.py` | the other two event classes |
| `signature_test.py`, `per_variable_extreme.py` | joint and per-variable extremeness |

## Reproducing the results

Python 3.13 with numpy, scipy, scikit-learn, netCDF4 and torch. Torch is needed
only for `deep_test.py`.

Run the downloads first, then the analyses. Each script writes a text report
into `data/results/`.

```
python src/common/fetch_gcmt.py          # catalogue
python src/atmosphere/fetch_pressure.py  # surface pressure
python src/thermal/fetch_olr.py          # outgoing longwave radiation
python src/thermal/fetch_surface.py      # air temperature, precipitable water
python src/ionosphere/fetch_tec.py       # global ionosphere maps, several hours

python src/unified/unified_test.py       # the eight families, global resolution
python src/unified/perevent_test.py      # the four field families, per event
python src/unified/resolution_test.py    # what averaging costs
python src/unified/positive_control.py   # what the framework can detect
python src/unified/ml_perevent.py        # matched case-control learning
python src/tidal/ide_bvalue_test.py      # size-frequency form of the tidal claim
python src/paper/make_figures.py         # figures for the manuscript
```

Seven raw arrays exceed the GitHub file-size limit of 100 MB and are therefore
not committed. They are listed with their sizes in `.gitignore`. None of them
is a result: each is a raw download or an array rebuilt from one, and the
script that produces it is in this repository, so running the download commands
above regenerates the complete tree.

## Design choices that matter

**The null is a circular time shift, not a permutation.** Shifting preserves
clustering, secular trend and seasonality, and destroys only the alignment with
the forcing. Block permutation was tried first and produced systematic negative
skill, which is diagnostic of a broken null rather than of a negative effect.

**Every replicate is searched as hard as the data.** The correction is a
max-statistic, so a reported p already accounts for however many variables a
family contains.

**Sampling is causal.** Only days strictly before the origin time are read. The
GRACE window starts 30 days before an event because a GRACE solution is a
monthly average, and a shorter window admits the earthquake's own co-seismic
gravity step into its own predictor. That error produced an apparent p below
0.0001 before it was caught.

**Offsets preserve the season.** Per-event nulls use whole-year offsets with a
jitter of plus or minus 15 days, so a seasonal coincidence cannot register as
signal.

**Machine learning controls are matched.** Each earthquake is paired with ten
control days at its own grid cell, offset by whole years and balanced before
and after, so location, season and catalogue growth are held fixed by
construction. Over 3,533 mainshocks the two classes differ in mean date by 143
days in a 55-year record.

## Detections that did not survive

Six apparent detections were withdrawn during this work, four of them our own.
They are recorded because a framework that has never rejected its own candidate
has not shown that it can.

| Apparent | After | Cause |
| --- | --- | --- |
| p = 0.0515 | p = 0.808 | quartile V-test, failed quintile scaling |
| p < 0.0001 | p = 0.417 to 0.966 | planetary Schuster test, uncalibrated null |
| p < 0.0001 | p = 0.36 | GRACE, co-seismic gravity inside the window |
| p = 0.010 | p = 0.347 | GRACE, monthly average overlapping the event |
| p = 0.025 | p = 0.565 | atmospheric level, failed split-half |
| p = 0.048 | p = 0.788 | tidal phase at one site vs each event's own fault |

## Scope

These bounds constrain a global moment-tensor catalogue at moment magnitude 5
to 6 over decades, at temporal scales of days to weeks. They do not address
multi-year or decadal modulation, which a trend-preserving null cannot see.
They do not address effects confined to particular fault mechanisms or regions,
which a globally pooled test dilutes. They do not exclude effects below the
resolution of a global catalogue: a dense local network reaching magnitude 2 to
3 measures b-values far better than is possible here, and crustal strain from
GNSS, which measures the quantity that actually loads a fault, was not
available.

## Manuscript

The manuscript is in `template/els-cas-templates/els-cas-templates/`, in
Elsevier `cas-sc` format. Build it with:

```
pdflatex forcing_bounds_cas_sc
bibtex   forcing_bounds_cas_sc
pdflatex forcing_bounds_cas_sc
pdflatex forcing_bounds_cas_sc
```
