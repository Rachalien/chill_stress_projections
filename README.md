# Projected changes in livestock cold stress across three Australian pastoral regions

Analysis code accompanying the manuscript on mid-century projections of the
Livestock Chill Index (LCI) in the Cobar-Lachlan (NSW), Yilgarn-Coolgardie
(WA), and Winton (QLD) pastoral regions.

The code computes daily gridded LCI from bias-corrected regional climate model
output, identifies the most severe 5% of chill days in each region, composites
the synoptic conditions on those days, attributes projected changes in LCI to
its temperature, wind, and rainfall drivers, and produces the manuscript
figures.

---

## The Livestock Chill Index

LCI combines wind speed, air temperature, and rainfall into a single measure of
the rate at which a wet, wind-exposed animal loses heat:

```
LCI = (11.7 + 3.1 * sqrt(ws)) * (40 - T) + 481 + 418 * (1 - exp(-0.04 * R))
```

with `ws` in m sâ»Â¹ at 2 m, `T` in Â°C, and `R` in mm dayâ»Â¹. Output is in
kJ mâ»Â² hrâ»Â¹. The formulation follows Nixon-Smith (1972) and Donnelly (1984).

LCI is nonlinear in its inputs. Computing it from composited driver fields is
therefore not equivalent to compositing daily LCI, and the two differ by a
non-negligible margin. All event-day LCI in this analysis is computed from raw
daily fields and composited afterwards, never reconstructed from composited
drivers.

---

## Regions, seasons, and thresholds

| Region | State | Season | Months | Threshold (kJ mâ»Â² hrâ»Â¹) |
|---|---|---|---|---|
| Cobar-Lachlan | NSW | dry | Mayâ€“Oct | 1100 |
| Yilgarn-Coolgardie | WA | dry | Mayâ€“Oct | 1200 |
| Winton | QLD | wet | Novâ€“Mar | 1000 |

Region boundaries are dissolved from the ABS LGA 2025 boundaries
(`LGA_2025_AUST_GDA2020`, EPSG:7844, column `LGA_NAME25`). Yilgarn-Coolgardie
and Cobar-Lachlan each combine two LGAs; Winton is a single LGA.

**April belongs to neither season.** The wet season is Novâ€“Mar and the dry
season is Mayâ€“Oct, so April is excluded from event identification, seasonal
climatologies, and composites. This matters: a `wet if ... else dry` fallback
will silently reclassify April as dry rather than dropping it, which changes
both the dry-season climatology and the event set. Season assignment returns
`(None, None)` for April, and callers must handle that rather than assume every
date carries a season.

---

## Data

| Dataset | Role | Location |
|---|---|---|
| NARCliM2 bias-corrected CORDEX (CMIP6) | projections | `/g/data/ia39/` |
| BARRA-R2 reanalysis | observational validation | `/g/data/ob53/` |
| AGCD v1-0-3 + AWRA-L wind | independent observational check | `/g/data/zv2/`, `/g/data/fj8/` |
| ACCESS-G surface roughness | 10 m to 2 m wind correction | `/g/data/dx2/` |
| ABS LGA 2025 boundaries | region definitions | local |

**Ensemble.** Eight NARCliM2 members: ACCESS-ESM1-5, EC-Earth3-Veg,
MPI-ESM1-2-HR, and NorESM2-MM, each downscaled by WRF412R3 and WRF412R5.
UKESM1-0-LL is excluded because its calendar is incompatible with the daily
alignment used here.

**Periods.** Historical 1990â€“2014; mid-century 2040â€“2060 under SSP3-7.0.

**Wind correction.** LCI is defined on 2 m wind, while the model output
provides 10 m wind. Wind is converted using the logarithmic profile with a
spatially varying roughness length, following Allen et al. (1998), FAO
Irrigation and Drainage Paper 56, Eq. 47.

---

## Two conventions that carry the results

**Contemporaneous referencing.** Each period's event composite is referenced to
that same period's seasonal climatology, not to a shared historical baseline.
Under this convention the uniform background warming cancels out of the
difference between the two anomaly patterns, and what remains is whether the
synoptic pattern producing severe chill has itself changed. Referencing both
periods to a common climatology conflates the two and the thermodynamic and
dynamic contributions cannot be separated afterwards. This follows Hansen et
al. (2023), *Climate Dynamics*.

**Shapley attribution.** Projected changes in event-day LCI are decomposed into
temperature, wind, and rainfall contributions using an analytic Shapley
decomposition, which is symmetric in the drivers and exactly additive. The
decomposition carries a nonlinear residual term reconciling the exact Shapley
values, evaluated at mean event-day conditions, with the mean of daily LCI
values. That term is a real property of the index rather than an error, and is
reported rather than absorbed into the driver shares.

An earlier OLS regression decomposition was abandoned: it produced inflated
slopes at Winton and was not identifiable for the Cobar-Lachlan dry season.

---

## Pipeline

Scripts are grouped by stage. Each stage depends on the outputs of the ones
above it.

### 1. Observational baseline

| Script | Output |
|---|---|
| `barra_daily_lci_yearly_land.py` | Daily gridded LCI from BARRA-R2, one file per year |
| `barra_lci_monthly_climatology.py` | Monthly LCI climatology from BARRA-R2 |
| `agcd_lci_monthly_gridded.py` | Monthly LCI climatology from AGCD + AWRA-L |
| `barra_regional_lci_metrics.py` | Regional LCI metrics from BARRA-R2 |
| `barra_lci_event_timing_metrics.py` | Gridded chill-season start, end, and span from BARRA-R2 |

### 2. Projections

| Script | Output |
|---|---|
| `synoptic_composite_lci_top5pct.py` | Per-member synoptic composites on top-5% LCI days, plus per-member daily regional LCI series |
| `lci_grid_composite_top5pct.py` | Gridded event-day LCI composites (imports shared functions from the script above) |
| `compute_synoptic_composite_significance.py` | Welch t-test with Benjamini-Hochberg FDR correction on the composite change |
| `narclim2_lci_event_timing_maps.py` | Gridded chill-season timing for NARCliM2, historical and future |
| `component_lci_attribution.py` | Shapley decomposition into temperature, wind, and rainfall terms |

`synoptic_composite_lci_top5pct.py` is the root of the projection side. It
defines the members, region specifications, season boundaries, LCI function,
and event selection, and several downstream scripts import these directly
rather than redefining them. Changing a season boundary or threshold there
propagates; changing one in a downstream copy does not.

### 3. Region extraction

`extract_lci_timing_by_region.py` area-averages the gridded timing fields from
stage 1 and stage 2 onto the three named regions and writes a single CSV.

### 4. Figures

| Script | Figure |
|---|---|
| `plot_lci_composite_focal_3x3_mid_century.py` | Gridded LCI composites: historical, mid-century, difference |
| `plot_synoptic_composites_compact_focal_with_barra_vectors.py` | Historical synoptic composites on top-5% LCI days |
| `plot_synoptic_composites_change_focal_mid_century.py` | Projected change in the composite anomaly pattern |
| `plot_chill_window_horizon_bars.py` | Chill-season timing, historical against mid-century |
| `plot_supp_lci_trends_midcentury.py` | Supplementary: regional LCI trends |
| `plot_lci_distribution_change.py` | Supplementary: daily LCI distributions with the p95 event cutoff |
| `plot_barra_agcd_monthly_bias_maps.py` | Supplementary: BARRA-R2 against AGCD monthly bias maps |
| `plot_barra_vs_agcd_lci_scatter.py` | Supplementary: BARRA-R2 against AGCD regional scatter |

`plot_chill_window_horizon_bars.py` holds its dates as literals at the top of
the file, transcribed from the CSV written in stage 3. Regenerating that CSV
does not update the figure; the values must be re-transcribed.

---

## Running

The pipeline was developed and run on the NCI Gadi HPC system (PBS Pro). Stage
1 and stage 2 are batch jobs; stage 3 and stage 4 run interactively in a few
minutes each.

```bash
module use /g/data/xp65/public/modules
module load conda/analysis3
```

Dependencies: `xarray`, `dask`, `numpy`, `pandas`, `scipy`, `matplotlib`,
`cartopy`, `geopandas`, `shapely`, `rioxarray`, `netCDF4`/`h5netcdf`. All are
present in the `conda/analysis3` environment.

Input paths are set as module-level constants near the top of each script and
point at NCI project storage. Running outside NCI requires changing them.

---

## Notes for anyone rerunning this

**Stale outputs do not announce themselves.** Several scripts glob a directory
for per-member files. If a rerun writes seven of eight members, the eighth is
silently taken from the previous run and the mixture produces no error. Clear
the output directory before a full regeneration rather than writing over it.

**`xr.concat` drops attributes.** The default `combine_attrs="override"`
retains only the first member's attributes, including `n_days`. That count is
used as the sample size in the significance test, so a silently inherited value
propagates into the p-values.

**Season boundaries live in more than one place.** The Novâ€“Mar definition must
match across event identification, climatology, composites, and timing
extraction. A change applied to one script and not the others produces output
that looks plausible and is internally inconsistent.

---

## Contact
rachel.taylor@anu.edu.au
<name, affiliation, ORCID, contact address>

