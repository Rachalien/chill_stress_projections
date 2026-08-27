#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synoptic_composite_lci_top5pct.py

For one NARCliM2 ensemble member, compute synoptic field composites for the
top-5% daily LCI events per region × season, comparing:
  - historical:  1990–2014 (historical experiment)
  - mid-century  2040–2060 (ssp370 experiment)

Winton's wet season is Nov-Mar (April excluded, because April
belongs to neither WET_MONTHS nor DRY_MONTHS and is dropped from event
identification, climatology, and composites for Winton). Cobar-Lachlan and
Yilgarn-Coolgardie are unaffected, since they only ever use the dry season
(May-Oct), which never included April to begin with.

Four fields are composited on the full AUST-05i domain:
  psl (hPa), sfcWind (m s-1), pr (mm d-1), tas (°C)

These per-member composites are later averaged across members in the
plotting step (plot_synoptic_composites.py).

Strategy
--------
Pass 1: regional LCI scalars
    For each year in historical and ssp370, load the four LCI driver
    variables, compute daily LCI, and reduce to an area-weighted
    regional-mean scalar for each of the three regions. Results are
    stored in a tidy DataFrame (tiny memory footprint).

Threshold
    For each (region, season, period), compute the p95 of the regional
    daily LCI distribution. All days at or above p95 are event days.

Pass 2: composite accumulation
    Open each composite variable lazily for the full experiment period.
    Use Dask-backed .isel(time=event_indices).mean("time").compute()
    to read and average only the event-day slices across the full domain.
    This is efficient because Dask only loads the HDF5 chunks that
    contain the selected time steps.

Output
------
One NetCDF per region per model:
    {OUTDIR}/{model_label}_composite_{region}_mid_century.nc

Also, one CSV per model with the full daily regional LCI series (used
internally for p95 thresholding, saved because it's free and useful for
plot_lci_distributions.py):
    {OUTDIR}/daily_lci/{model_label}_regional_daily_lci.csv
Columns: date, Winton, Yilgarn_Coolgardie, Cobar_Lachlan, season,
         season_year, period (hist/future)

Variables (float32, dims lat × lon):
    {field}_{season}_{period}       mean, field in (psl, sfcWind, pr, tas),
                                     season in (wet, dry), period in (hist, future)
    {field}_{season}_{period}_std   sample std (ddof=1) across event days,
                                     for a downstream Welch's t-test
Attributes on each variable: n_days (int)
Global attributes: model, region, top_pct, hist_period, future_period, scenario

NOTE: computing std requires materializing the full event-day stack per
field (previously only a running Dask mean was needed), which increases
peak memory and runtime versus the mean-only version. If jobs run close to
the 32gb/8hr limits below, bump mem and/or walltime in the PBS script.

PBS
---
Submit one job per member (8 jobs, UKESM excluded).
Recommended resources: ncpus=4, mem=32gb, walltime=08:00:00 (see NOTE above
-- may need increasing now that std is also computed).
See submit_synoptic_composites_mid_century.sh.

Usage
-----
  python synoptic_composite_lci_top5pct.py --model ACCESS-ESM1-5_WRF412R3
"""

from __future__ import annotations

import os
import re
import glob
import argparse
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
xr.set_options(keep_attrs=True)


# ── CONFIG ────────────────────────────────────────────────────────────────────

NARCLIM_ROOT = (
    "/g/data/ia39/australian-climate-service/release/CORDEX/output-CMIP6/"
    "bias-adjusted-output/AUST-05i/NSW-Government"
)
BC_VERSION = "v1-r1-ACS-QME-BARRAR2-1980-2022"

OUTDIR = "/scratch/dx2/rt9243/chill_projections/synoptic_composites"

MEMBERS = {
    "ACCESS-ESM1-5_WRF412R3": ("ACCESS-ESM1-5",  "r6i1p1f1", "NARCliM2-0-WRF412R3"),
    "ACCESS-ESM1-5_WRF412R5": ("ACCESS-ESM1-5",  "r6i1p1f1", "NARCliM2-0-WRF412R5"),
    "EC-Earth3-Veg_WRF412R3": ("EC-Earth3-Veg",  "r1i1p1f1", "NARCliM2-0-WRF412R3"),
    "EC-Earth3-Veg_WRF412R5": ("EC-Earth3-Veg",  "r1i1p1f1", "NARCliM2-0-WRF412R5"),
    "MPI-ESM1-2-HR_WRF412R3": ("MPI-ESM1-2-HR",  "r1i1p1f1", "NARCliM2-0-WRF412R3"),
    "MPI-ESM1-2-HR_WRF412R5": ("MPI-ESM1-2-HR",  "r1i1p1f1", "NARCliM2-0-WRF412R5"),
    "NorESM2-MM_WRF412R3":    ("NorESM2-MM",     "r1i1p1f1", "NARCliM2-0-WRF412R3"),
    "NorESM2-MM_WRF412R5":    ("NorESM2-MM",     "r1i1p1f1", "NARCliM2-0-WRF412R5"),
}

# Historical: load from 1989 so that Nov–Dec 1989 is available for wet season 1990
HIST_LOAD_YEARS   = list(range(1989, 2015))
FUTURE_LOAD_YEARS = list(range(2039, 2061))  # Nov 2039 needed for wet season 2040

# Season ranges to include in each composite
HIST_SEASON_RANGE   = (1990, 2014)
FUTURE_SEASON_RANGE = (2040, 2060)  # matches narclim2_lci_event_timing_ssp370_2040-2060.nc

TOP_PCT = 5  # composite top 5% = days at/above p95

WET_MONTHS = {11, 12, 1, 2, 3}    # April removed. Winton only,
                                   # since Winton is the only region using "wet" season
DRY_MONTHS = {5, 6, 7, 8, 9, 10}  # unchanged, already excludes April -- no leakage risk
                                   # for Cobar-Lachlan/Yilgarn-Coolgardie's dry season

# Variables loaded for the full-domain composite pass
COMPOSITE_VARS = ["pslAdjust", "sfcWindAdjust", "prAdjust", "tasmaxAdjust", "tasminAdjust"]

# Variables needed for computing LCI in pass 1
LCI_VARS = ["tasmaxAdjust", "tasminAdjust", "sfcWindAdjust", "prAdjust"]
_LGA_SHP = (
    "/g/data/dx2/rt9243/Datasets/"
    "LGA_2025_AUST_GDA2020/LGA_2025_AUST_GDA2020.shp"
)

REGION_SPECS = {
    "Winton": {
        "shp":         _LGA_SHP,
        "dissolve":    False,
        "region_col":  "LGA_NAME25",
        "region_vals": ["Winton"],
    },
    "Yilgarn_Coolgardie": {
        "shp":         _LGA_SHP,
        "dissolve":    True,
        "region_col":  "LGA_NAME25",
        "region_vals": ["Yilgarn", "Coolgardie"],
    },
    "Cobar_Lachlan": {
        "shp":         _LGA_SHP,
        "dissolve":    True,
        "region_col":  "LGA_NAME25",
        "region_vals": ["Cobar", "Lachlan"],
    },
}

SPAN_RE = re.compile(r"_(\d{8})-(\d{8})\.nc$")


# ── FILE DISCOVERY ────────────────────────────────────────────────────────────
# Mirrors the pattern in lci_metrics_narclim2_aust05i.py exactly.

def _span_years(path):
    m = SPAN_RE.search(path)
    return (int(m.group(1)[:4]), int(m.group(2)[:4])) if m else (None, None)


def _dedupe_same_span(files):
    by_span = defaultdict(list)
    for f in files:
        m = SPAN_RE.search(f)
        key = (m.group(1), m.group(2)) if m else (None, None)
        by_span[key].append(f)
    kept = []
    for _, group in by_span.items():
        non_latest = [g for g in group if "latest" not in g.split(os.sep)]
        kept.append(sorted(non_latest or group)[0])
    return sorted(kept, key=lambda p: (SPAN_RE.search(p).group(1)
                                       if SPAN_RE.search(p) else ""))


def find_var_files(model, variant, rcm, experiment, var):
    pattern = os.path.join(
        NARCLIM_ROOT, model, experiment, variant, rcm, BC_VERSION,
        "day", var, "*",
        f"{var}_AUST-05i_{model}_{experiment}_{variant}_NSW-Government_"
        f"{rcm}_{BC_VERSION}_day_*.nc",
    )
    return _dedupe_same_span(sorted(set(glob.glob(pattern))))


def open_var_lazy(model, variant, rcm, experiment, var, years,
                  chunks=None):
    """
    Open a DataArray for `var` lazily (Dask-backed), covering all of `years`.
    Feb 29 is dropped. Returned DataArray has a clean noleap-style time axis
    of date strings (via strftime, so calendar differences are transparent).
    """
    if chunks is None:
        chunks = {"time": 30, "lat": 200, "lon": 200}
    all_files = find_var_files(model, variant, rcm, experiment, var)
    y0, y1 = min(years), max(years)
    keep = [f for f in all_files
            if not (_span_years(f)[1] < y0 or _span_years(f)[0] > y1)]
    if not keep:
        raise FileNotFoundError(
            f"No files for {var} | {model}/{rcm} | {experiment} | {y0}–{y1}"
        )
    try:
        ds = xr.open_mfdataset(keep, combine="by_coords", engine="h5netcdf",
                               chunks=chunks, parallel=False)
    except Exception:
        ds = xr.open_mfdataset(keep, combine="by_coords",
                               chunks=chunks, parallel=False)
    da = ds[var].sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))
    # Drop Feb 29 (works for any calendar via boolean mask)
    mask_feb29 = (da.time.dt.month == 2) & (da.time.dt.day == 29)
    da = da.isel(time=~mask_feb29)
    return da


def time_strings(da):
    """Return numpy array of 'YYYY-MM-DD' strings for a DataArray's time axis."""
    return da.time.dt.strftime("%Y-%m-%d").values


# ── REGION MASKS ──────────────────────────────────────────────────────────────

def build_region_mask(spec, lat, lon):
    """
    Return a boolean numpy array (nlat, nlon) = True inside the region.
    Matches the masking approach used in the existing LCI metrics pipeline.
    """
    gdf = gpd.read_file(spec["shp"])
    gdf = gdf[gdf[spec["region_col"]].isin(spec["region_vals"])].copy()
    gdf = gdf.reset_index(drop=True)
    if spec.get("dissolve") and len(spec["region_vals"]) > 1:
        gdf = gdf.dissolve().explode(index_parts=False).dissolve().reset_index(drop=True)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    regs = regionmask.from_geopandas(gdf, overlap=False)
    mask_da = regs.mask(lon.values, lat.values)  # NaN outside, integer inside
    return ~np.isnan(mask_da.values)  # (nlat, nlon) bool


def area_weights_2d(lat, mask):
    """
    Cosine-of-latitude area weights restricted to region mask.
    Returns (nlat, nlon) float array, zero outside mask.
    """
    cos_lat = np.cos(np.deg2rad(lat.values))  # (nlat,)
    w2d = cos_lat[:, np.newaxis] * mask        # (nlat, nlon)
    return w2d


# ── LCI FORMULA ───────────────────────────────────────────────────────────────

def calculate_lci(ws, T_C, R_mm_day):
    """
    Livestock Chill Index (kJ m-2 hr-1).
    ws        : wind speed (m s-1)
    T_C       : daily mean temperature (°C)
    R_mm_day  : precipitation (mm d-1)
    Matches calculate_lci_xr() in lci_metrics_narclim2_aust05i.py exactly.
    """
    return (
        (11.7 + 3.1 * np.sqrt(np.maximum(ws, 0.0))) * (40.0 - T_C)
        + 481.0
        + 418.0 * (1.0 - np.exp(-0.04 * np.maximum(R_mm_day, 0.0)))
    )


# ── SEASON ASSIGNMENT ─────────────────────────────────────────────────────────

def season_of_date(date_str):
    """
    Return (season, season_year) for a 'YYYY-MM-DD' string.
    Wet season (Nov-Mar) is labelled by the Jan-Mar year. Dry season is
    May-Oct. April is in neither WET_MONTHS nor DRY_MONTHS (excluded from
    the Winton wet-season analysis) and returns
    (None, None) -- callers must handle this, not assume every date gets
    a season.
    """
    year, month = int(date_str[:4]), int(date_str[5:7])
    if month in WET_MONTHS:
        return "wet", year + 1 if month >= 11 else year
    if month in DRY_MONTHS:
        return "dry", year
    return None, None


def annotate_seasons(date_series):
    """
    Given a pandas Series of 'YYYY-MM-DD' strings, return a DataFrame
    with columns [season, season_year].

    Uses explicit WET_MONTHS/DRY_MONTHS membership rather than a wet/else
    fallback, so months in neither set (currently just April) come back
    with season=None and are naturally dropped by every downstream filter
    that matches on season == "wet"/"dry" -- they don't silently end up in
    "dry" the way an else-branch would.
    """
    parsed = pd.to_datetime(date_series)
    months = parsed.dt.month.values
    years  = parsed.dt.year.values

    season = np.full(months.shape, None, dtype=object)
    season[np.isin(months, list(WET_MONTHS))] = "wet"
    season[np.isin(months, list(DRY_MONTHS))] = "dry"
    season_year = np.where(months >= 11, years + 1, years)

    return pd.DataFrame({"season": season, "season_year": season_year},
                        index=date_series.index)


# ── PASS 1: REGIONAL DAILY LCI SCALAR TIME SERIES ────────────────────────────

def compute_lci_regional_series(model, variant, rcm, experiment, years,
                                 region_masks, lat, lon):
    """
    Open all four LCI driver variables lazily for the full period in one
    open_mfdataset call each, compute LCI as a lazy xarray graph, then
    reduce to area-weighted regional-mean scalars via a single Dask
    .compute() call per region.

    Replaces the original year-by-year loop which made ~235 open_mfdataset
    calls per experiment and loaded ~170 GB of full-domain arrays
    sequentially. This version makes 4 calls and reads only what Dask
    needs for the spatial reduction.
    """
    log.info(f"  Opening LCI vars lazily | {experiment} | {min(years)}-{max(years)}")

    # Build xarray DataArray weights (must share lat/lon coords with lci
    # so that broadcasting in lci * w_da works correctly)
    coords = {"lat": lat, "lon": lon}
    region_weight_da = {}
    region_total_w   = {}
    for rname, mask in region_masks.items():
        w2d = area_weights_2d(lat, mask)          # (nlat, nlon) numpy
        region_weight_da[rname] = xr.DataArray(
            w2d, dims=["lat", "lon"], coords=coords
        )
        region_total_w[rname] = float(w2d.sum())

    # One open_mfdataset call per variable for the full period.
    # Chunk by time so Dask can parallelise the spatial reduction
    # across time slices with multiple workers.
    ckw = {"time": 30, "lat": -1, "lon": -1}
    da_tasmax = open_var_lazy(model, variant, rcm, experiment, "tasmaxAdjust", years, ckw)
    da_tasmin = open_var_lazy(model, variant, rcm, experiment, "tasminAdjust", years, ckw)
    da_ws     = open_var_lazy(model, variant, rcm, experiment, "sfcWindAdjust", years, ckw)
    da_pr     = open_var_lazy(model, variant, rcm, experiment, "prAdjust",      years, ckw)

    # Date strings are metadata; no data loaded here
    dates = time_strings(da_tasmax)

    # Build LCI as a lazy xarray graph (no compute yet)
    ws_safe = da_ws.clip(min=0.0)
    pr_safe = da_pr.clip(min=0.0)
    T_C = (da_tasmax + da_tasmin) * 0.5
    lci = (
        (11.7 + 3.1 * np.sqrt(ws_safe)) * (40.0 - T_C)
        + 481.0
        + 418.0 * (1.0 - np.exp(-0.04 * pr_safe))
    )   # lazy DataArray (time, lat, lon)

    # Reduce to regional scalar time series; one .compute() per region
    result_cols = {"date": dates}
    for rname, w_da in region_weight_da.items():
        tw = region_total_w[rname]
        if tw <= 0:
            log.warning(f"  Zero total weight for {rname}; filling NaN")
            result_cols[rname] = np.full(len(dates), np.nan)
            continue
        log.info(f"  Computing regional LCI series | {rname}")
        regional = (lci * w_da).sum(["lat", "lon"]) / tw   # lazy scalar (time,)
        result_cols[rname] = regional.compute().values      # triggers Dask read

    return pd.DataFrame(result_cols)


# ── EVENT IDENTIFICATION ──────────────────────────────────────────────────────

def find_top_pct_events(lci_df, region_names):
    """
    For each (region, season, period), compute the (100-TOP_PCT)th percentile
    of regional daily LCI values and return a list of
    (date_str, region, season, period) tuples for all days at or above it.

    lci_df must have columns: date, season, season_year, period, *region_names
    """
    events = []

    for region in region_names:
        for season in ("wet", "dry"):
            for period, (yr0, yr1) in [("hist",   HIST_SEASON_RANGE),
                                        ("future", FUTURE_SEASON_RANGE)]:
                sub = lci_df[
                    (lci_df["season"] == season) &
                    (lci_df["season_year"] >= yr0) &
                    (lci_df["season_year"] <= yr1)
                ][["date", region]].dropna()

                if sub.empty:
                    log.warning(f"  No LCI data for {region}/{season}/{period}")
                    continue

                threshold = np.nanpercentile(sub[region].values, 100 - TOP_PCT)
                top = sub[sub[region] >= threshold]["date"].tolist()

                log.info(
                    f"  {region}/{season}/{period}: p{100-TOP_PCT}="
                    f"{threshold:.1f} kJ m-2 hr-1, n={len(top)}"
                )
                events.extend((d, region, season, period) for d in top)

    return events  # list of (date_str, region, season, period)


# ── PASS 2: COMPOSITE ACCUMULATION ───────────────────────────────────────────

def accumulate_composites(model, variant, rcm, events, lat, lon):
    """
    For each (region, season, period) key, open the relevant experiment's
    composite variables lazily and use Dask to compute the mean over event-day
    indices only.  Dask reads only the HDF5 chunks containing those time steps.

    Returns a dict:
        (region, season, period) -> {
            'psl':     (nlat, nlon) float32 array in hPa,
            'sfcWind': (nlat, nlon) float32 array in m s-1,
            'pr':      (nlat, nlon) float32 array in mm d-1,
            'tas':     (nlat, nlon) float32 array in °C,
            'n':       int (number of event days composited)
        }
    """
    event_df = pd.DataFrame(events, columns=["date", "region", "season", "period"])
    event_df["experiment"] = event_df["period"].map(
        {"hist": "historical", "future": "ssp370"}
    )

    result = {}
    clim   = {}   # (season, period) -> {field: array, "n": int}; region-agnostic

    for experiment, exp_group in event_df.groupby("experiment"):
        years_to_load = HIST_LOAD_YEARS if experiment == "historical" else FUTURE_LOAD_YEARS
        period = "hist" if experiment == "historical" else "future"
        log.info(f"  Opening composite vars lazily | {experiment}")

        # Open all composite vars for this experiment, lazily
        lazy = {}
        for v in COMPOSITE_VARS:
            lazy[v] = open_var_lazy(model, variant, rcm, experiment, v, years_to_load)

        # Build date→index mapping from one reference variable
        ref_dates = time_strings(lazy["tasmaxAdjust"])
        date_to_idx = {d: i for i, d in enumerate(ref_dates)}

        # Composite per (region, season, period) key within this experiment
        keys = exp_group[["region", "season", "period"]].drop_duplicates()
        for _, row in keys.iterrows():
            k = (row["region"], row["season"], row["period"])
            key_events = exp_group[
                (exp_group["region"]  == row["region"]) &
                (exp_group["season"]  == row["season"]) &
                (exp_group["period"]  == row["period"])
            ]["date"].tolist()

            indices = [date_to_idx[d] for d in key_events if d in date_to_idx]
            missing = len(key_events) - len(indices)
            if missing:
                log.warning(f"  {k}: {missing} event date(s) not found in time axis")
            if not indices:
                log.warning(f"  {k}: no valid event indices, skipping")
                continue

            log.info(f"  Compositing {k}: {len(indices)} days")

            # Materialize the full event-day stack once per field so mean and
            # std come from exactly the same data (avoids a second Dask pass).
            psl_vals     = lazy["pslAdjust"].isel(time=indices).compute().values / 100.0  # Pa -> hPa
            sfcWind_vals = lazy["sfcWindAdjust"].isel(time=indices).compute().values
            pr_vals      = lazy["prAdjust"].isel(time=indices).compute().values
            tasmax_vals  = lazy["tasmaxAdjust"].isel(time=indices).compute().values
            tasmin_vals  = lazy["tasminAdjust"].isel(time=indices).compute().values
            tas_vals     = (tasmax_vals + tasmin_vals) * 0.5

            # std uses ddof=1 (sample std) for consistency with a Welch's
            # t-test downstream. Only computed for hist/future event
            # composites, not the seasonal climatology below -- the
            # climatology's ~4500-day sample size makes its own sampling
            # uncertainty negligible next to the ~200-day event composites,
            # so it's treated as a fixed constant for significance testing.
            result[k] = {
                "psl":         psl_vals.mean(axis=0).astype(np.float32),
                "psl_std":     psl_vals.std(axis=0, ddof=1).astype(np.float32),
                "sfcWind":     sfcWind_vals.mean(axis=0).astype(np.float32),
                "sfcWind_std": sfcWind_vals.std(axis=0, ddof=1).astype(np.float32),
                "pr":          pr_vals.mean(axis=0).astype(np.float32),
                "pr_std":      pr_vals.std(axis=0, ddof=1).astype(np.float32),
                "tas":         tas_vals.mean(axis=0).astype(np.float32),
                "tas_std":     tas_vals.std(axis=0, ddof=1).astype(np.float32),
                "n":           len(indices),
            }

        # ── Seasonal climatology: mean over ALL season days (not just events) ──
        # Region-agnostic; used to form anomalies (event composite − climatology).
        seas = annotate_seasons(pd.Series(ref_dates))
        yr0, yr1 = HIST_SEASON_RANGE if period == "hist" else FUTURE_SEASON_RANGE
        for season in ("wet", "dry"):
            sel = (
                (seas["season"].values == season) &
                (seas["season_year"].values >= yr0) &
                (seas["season_year"].values <= yr1)
            )
            clim_idx = np.where(sel)[0]
            if len(clim_idx) == 0:
                log.warning(f"  Climatology {season}/{period}: no days, skipping")
                continue
            log.info(f"  Climatology {season}/{period}: {len(clim_idx)} days")

            psl_c    = lazy["pslAdjust"].isel(time=clim_idx).mean("time").compute().values / 100.0
            wind_c   = lazy["sfcWindAdjust"].isel(time=clim_idx).mean("time").compute().values
            pr_c     = lazy["prAdjust"].isel(time=clim_idx).mean("time").compute().values
            tasmax_c = lazy["tasmaxAdjust"].isel(time=clim_idx).mean("time").compute().values
            tasmin_c = lazy["tasminAdjust"].isel(time=clim_idx).mean("time").compute().values
            tas_c    = (tasmax_c + tasmin_c) * 0.5

            clim[(season, period)] = {
                "psl":     psl_c.astype(np.float32),
                "sfcWind": wind_c.astype(np.float32),
                "pr":      pr_c.astype(np.float32),
                "tas":     tas_c.astype(np.float32),
                "n":       int(len(clim_idx)),
            }

        # Close datasets to release file handles
        for da in lazy.values():
            try:
                da.close()
            except Exception:
                pass

    return result, clim


# ── OUTPUT ────────────────────────────────────────────────────────────────────

def write_composite_netcdf(result, clim, lat, lon, region_name, model_label):
    """Write one composite NetCDF per (model, region), incl. seasonal climatology."""
    os.makedirs(OUTDIR, exist_ok=True)
    outpath = os.path.join(OUTDIR, f"{model_label}_composite_{region_name}_mid_century.nc")

    units = {"psl": "hPa", "sfcWind": "m s-1", "pr": "mm d-1", "tas": "degC"}
    data_vars = {}

    # Event composites for this region
    for (region, season, period), d in result.items():
        if region != region_name:
            continue
        tag = f"{season}_{period}"  # e.g. wet_hist, dry_future
        for field in ("psl", "sfcWind", "pr", "tas"):
            data_vars[f"{field}_{tag}"] = xr.DataArray(
                d[field],
                dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
                attrs={"units": units[field], "n_days": d["n"]},
            )
            data_vars[f"{field}_{tag}_std"] = xr.DataArray(
                d[f"{field}_std"],
                dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
                attrs={
                    "units": units[field],
                    "n_days": d["n"],
                    "description": "sample std (ddof=1) across event days, for Welch's t-test",
                },
            )

    # Seasonal climatology (region-agnostic; written into every region file)
    for (season, period), d in clim.items():
        tag = f"{season}_{period}"
        for field in ("psl", "sfcWind", "pr", "tas"):
            data_vars[f"{field}_{tag}_clim"] = xr.DataArray(
                d[field],
                dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
                attrs={"units": units[field], "n_days": d["n"],
                       "description": "seasonal-mean climatology (all season days)"},
            )

    if not data_vars:
        log.warning(f"No composite data for region {region_name}; skipping write")
        return

    ds = xr.Dataset(
        data_vars,
        attrs={
            "model":         model_label,
            "region":        region_name,
            "top_pct":       TOP_PCT,
            "hist_period":   f"{HIST_SEASON_RANGE[0]}-{HIST_SEASON_RANGE[1]}",
            "future_period": f"{FUTURE_SEASON_RANGE[0]}-{FUTURE_SEASON_RANGE[1]}",
            "scenario":      "ssp370",
            "fields":        "psl (hPa), sfcWind (m s-1), pr (mm d-1), tas (degC)",
            "note":          (
                f"Top-{TOP_PCT}% daily LCI event composites on AUST-05i domain. "
                "Per-member output; average across members in plot script."
            ),
        },
    )

    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in data_vars}
    tmp = outpath + ".tmp"
    ds.to_netcdf(tmp, engine="h5netcdf", encoding=enc)
    os.replace(tmp, outpath)
    log.info(f"Wrote: {outpath}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compute top-5% LCI event synoptic composites for one NARCliM2 member."
    )
    ap.add_argument(
        "--model", required=True, choices=list(MEMBERS),
        help="Member label, e.g. ACCESS-ESM1-5_WRF412R3"
    )
    args = ap.parse_args()

    model_label = args.model
    model, variant, rcm = MEMBERS[model_label]
    log.info(f"=== {model_label} ===")

    # ── Grid coordinates (from first historical file) ─────────────────────
    log.info("Reading grid coordinates...")
    ref = open_var_lazy(model, variant, rcm, "historical", "tasmaxAdjust", [1990],
                        chunks={"time": 1, "lat": 200, "lon": 200})
    lat = ref.lat.load()
    lon = ref.lon.load()

    # ── Region masks ──────────────────────────────────────────────────────
    log.info("Building region masks...")
    region_masks = {}
    for rname, spec in REGION_SPECS.items():
        log.info(f"  {rname}")
        region_masks[rname] = build_region_mask(spec, lat, lon)

    # ── Pass 1: Regional daily LCI scalar time series ─────────────────────
    log.info("Pass 1: historical LCI time series")
    hist_df = compute_lci_regional_series(
        model, variant, rcm, "historical", HIST_LOAD_YEARS,
        region_masks, lat, lon,
    )

    log.info("Pass 1: ssp370 LCI time series")
    future_df = compute_lci_regional_series(
        model, variant, rcm, "ssp370", FUTURE_LOAD_YEARS,
        region_masks, lat, lon,
    )

    # ── Assign season labels and period tag ──────────────────────────────
    for df, period_tag in [(hist_df, "hist"), (future_df, "future")]:
        seas_info = annotate_seasons(df["date"])
        df["season"]      = seas_info["season"].values
        df["season_year"] = seas_info["season_year"].values
        df["period"]      = period_tag

    lci_df = pd.concat([hist_df, future_df], ignore_index=True)

    # ── Save daily regional LCI series ─────────────────────────────────────
    # Already computed above purely for thresholding; saving it costs nothing
    # extra and lets plot_lci_distribution_change.py show the full daily
    # distribution behind the p95 event cutoff.
    daily_dir = os.path.join(OUTDIR, "daily_lci")
    os.makedirs(daily_dir, exist_ok=True)
    daily_path = os.path.join(daily_dir, f"{model_label}_regional_daily_lci.csv")
    lci_df.to_csv(daily_path, index=False)
    log.info(f"Wrote daily regional LCI series: {daily_path}")

    # ── Event identification ──────────────────────────────────────────────
    log.info("Identifying top-5% event days...")
    events = find_top_pct_events(lci_df, list(REGION_SPECS.keys()))
    log.info(f"Total event-days (all regions/seasons/periods): {len(events)}")

    # ── Pass 2: Composite accumulation ────────────────────────────────────
    log.info("Pass 2: accumulating composites...")
    result, clim = accumulate_composites(model, variant, rcm, events, lat, lon)

    # ── Write output ──────────────────────────────────────────────────────
    for rname in REGION_SPECS:
        write_composite_netcdf(result, clim, lat, lon, rname, model_label)

    log.info("=== Done ===")


if __name__ == "__main__":
    main()