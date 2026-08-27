#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
narclim2_lci_event_timing_maps.py

Compute continental NARCliM2 LCI event timing maps for historical and future
periods, including 10 m -> 2 m wind-height correction.

For each scenario, period, season and threshold, this script computes:

    first_day      : first day-of-season with LCI >= threshold
    last_day       : last day-of-season with LCI >= threshold
    span_days      : last_day - first_day + 1
    event_days     : number of exceedance days in season
    any_event      : whether threshold is exceeded at least once

Then it averages:
    1. over years within each period for each member
    2. over ensemble members

Outputs
-------
NetCDFs:
    narclim2_lci_event_timing_historical_1990-2014.nc
    narclim2_lci_event_timing_ssp126_2040-2060.nc
    narclim2_lci_event_timing_ssp126_2080-2100.nc
    narclim2_lci_event_timing_ssp370_2040-2060.nc
    narclim2_lci_event_timing_ssp370_2080-2100.nc

Figures:
    figures/timing_maps_<scenario>_<period>_<season>_thr<threshold>.png

Each figure has:
    columns = first date, last date, span length
    rows    = historical, future, future - historical

Usage
-----
module use /g/data/xp65/public/modules
module load conda/analysis3

python /g/data/dx2/rt9243/Python_Code/narclim2_lci_event_timing_maps.py
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
xr.set_options(keep_attrs=True)


# ============================================================
# CONFIG
# ============================================================

NARCLIM_ROOT = (
    "/g/data/ia39/australian-climate-service/release/CORDEX/output-CMIP6/"
    "bias-adjusted-output/AUST-05i/NSW-Government"
)
BC_VERSION = "v1-r1-ACS-QME-BARRAR2-1980-2022"

OUTDIR = Path(
    "/scratch/dx2/rt9243/chill_projections/"
    "narclim2_lci_event_timing_maps"
)

Z0_PATH = "/g/data/dx2/rt9243/Datasets/sfc_rough_len_Aust_fc.nc"
Z0_MIN, Z0_MAX = 1e-4, 1.9

NARCLIM_MEMBERS = [
    ("ACCESS-ESM1-5",  "r6i1p1f1", "NARCliM2-0-WRF412R3"),
    ("ACCESS-ESM1-5",  "r6i1p1f1", "NARCliM2-0-WRF412R5"),
    ("EC-Earth3-Veg",  "r1i1p1f1", "NARCliM2-0-WRF412R3"),
    ("EC-Earth3-Veg",  "r1i1p1f1", "NARCliM2-0-WRF412R5"),
    ("MPI-ESM1-2-HR",  "r1i1p1f1", "NARCliM2-0-WRF412R3"),
    ("MPI-ESM1-2-HR",  "r1i1p1f1", "NARCliM2-0-WRF412R5"),
    ("NorESM2-MM",     "r1i1p1f1", "NARCliM2-0-WRF412R3"),
    ("NorESM2-MM",     "r1i1p1f1", "NARCliM2-0-WRF412R5"),
]

EXPERIMENT_YEAR_RANGE = {
    "historical": (1989, 2014),  # includes Nov-Dec 1989 for wet-1990
    "ssp126": (2015, 2100),
    "ssp370": (2015, 2100),
}

HIST_PERIOD = (1990, 2014)

FUTURE_PERIODS = {
    "2040-2060": (2040, 2060),
    "2080-2100": (2080, 2100),
}

SCENARIOS = ["ssp126", "ssp370"]
SEASONS = ["wet", "dry"]
THRESHOLDS = [1000, 1100, 1200]

WET_MONTHS = {11, 12, 1, 2, 3}    # April excluded; wet season is Nov-Mar. NOTE: this
                                   # constant isn't actually consumed by
                                   # compute_daily_lci_for_season(), which
                                   # hardcodes its own month list directly --
                                   # kept in sync here for readability only.
DRY_MONTHS = {5, 6, 7, 8, 9, 10}

SPAN_RE = re.compile(r"_(\d{8})-(\d{8})\.nc$")

EXTENT = [110, 155.5, -45, -9]

SEASON_LABELS = {
    "wet": "November-March",
    "dry": "May-October",
}

DRY_DAY_TICKS = [1, 32, 62, 93, 123, 154, 184]
DRY_DAY_LABELS = ["1 May", "1 Jun", "1 Jul", "1 Aug", "1 Sep", "1 Oct", "31 Oct"]

WET_DAY_TICKS = [1, 31, 62, 93, 124, 154]
WET_DAY_LABELS = ["1 Nov", "1 Dec", "1 Jan", "1 Feb", "1 Mar", "31 Mar"]


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def member_label(model: str, rcm: str) -> str:
    return f"{model}_{rcm.replace('NARCliM2-0-', '')}"


def _file_span_key(path: str) -> tuple:
    m = SPAN_RE.search(path)
    return (m.group(1), m.group(2)) if m else (None, None)


def _span_years(path: str) -> tuple[Optional[int], Optional[int]]:
    m = SPAN_RE.search(path)
    if not m:
        return None, None
    return int(m.group(1)[:4]), int(m.group(2)[:4])


def dedupe_same_span(files: Iterable[str]) -> list[str]:
    by_span = defaultdict(list)

    for f in files:
        by_span[_file_span_key(f)].append(f)

    kept = []
    for _, group in by_span.items():
        if len(group) == 1:
            kept.append(group[0])
            continue

        non_latest = [g for g in group if "latest" not in g.split(os.sep)]
        kept.append(sorted(non_latest or group)[0])

    return sorted(kept, key=lambda p: _file_span_key(p)[0] or "")


def find_var_files(
    model: str,
    variant: str,
    rcm: str,
    experiment: str,
    var: str,
) -> list[str]:
    pattern = os.path.join(
        NARCLIM_ROOT,
        model,
        experiment,
        variant,
        rcm,
        BC_VERSION,
        "day",
        var,
        "*",
        f"{var}_AUST-05i_{model}_{experiment}_{variant}_NSW-Government_"
        f"{rcm}_{BC_VERSION}_day_*.nc",
    )

    hits = glob.glob(pattern)
    return dedupe_same_span(sorted(set(hits)))


def files_for_year(all_files: list[str], year: int) -> list[str]:
    keep = []

    for f in all_files:
        y0, y1 = _span_years(f)
        if y0 is not None and y0 <= year <= y1:
            keep.append(f)

    return keep


def open_year_files(year_files: list[str], var: str, year: int) -> xr.DataArray:
    if not year_files:
        raise FileNotFoundError(f"No files for {var} {year}")

    if len(year_files) == 1:
        ds = xr.open_dataset(year_files[0], engine="h5netcdf", mask_and_scale=True)
    else:
        ds = xr.open_mfdataset(
            year_files,
            combine="by_coords",
            engine="h5netcdf",
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )

    if var not in ds:
        raise KeyError(f"{var!r} not found. Variables: {list(ds.data_vars)}")

    return ds[var].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))


def drop_feb29(da: xr.DataArray) -> xr.DataArray:
    if "time" not in da.dims:
        return da

    return da.sel(
        time=~((da.time.dt.month == 2) & (da.time.dt.day == 29))
    )


# ============================================================
# WIND HEIGHT CORRECTION
# ============================================================

def load_z0_static(z0_path: str = Z0_PATH) -> xr.DataArray:
    ds = xr.open_dataset(z0_path)
    z0 = ds["sfc_rough_len"].mean("time", skipna=True)
    z0 = z0.clip(min=Z0_MIN, max=Z0_MAX)
    z0.attrs.update({
        "units": "m",
        "long_name": "Surface roughness length for momentum",
        "source": z0_path,
    })
    return z0


def regrid_z0(
    z0_raw: xr.DataArray,
    target_lat: xr.DataArray,
    target_lon: xr.DataArray,
) -> xr.DataArray:
    z0_interp = z0_raw.interp(
        lat=target_lat,
        lon=target_lon,
        method="nearest",
    )
    return z0_interp.clip(min=Z0_MIN, max=Z0_MAX)


def wind_10m_to_2m(ws10: xr.DataArray, z0: xr.DataArray) -> xr.DataArray:
    """
    Convert 10 m wind speed to 2 m using the logarithmic wind profile:

        v2 = v10 * ln(2 / z0) / ln(10 / z0)

    ws10 : wind speed at 10 m, m s-1
    z0   : roughness length, m, on the same lat/lon grid
    """
    correction = np.log(2.0 / z0) / np.log(10.0 / z0)
    ws2 = ws10.clip(min=0.0) * correction

    ws2.attrs.update(ws10.attrs)
    ws2.attrs["height"] = "2 m"
    ws2.attrs["wind_height_correction"] = (
        "log wind profile: v2 = v10 * ln(2/z0) / ln(10/z0)"
    )

    return ws2


# ============================================================
# LCI CALCULATION
# ============================================================

def calculate_lci_xr(
    ws2m: xr.DataArray,
    tas_c: xr.DataArray,
    pr_mmday: xr.DataArray,
) -> xr.DataArray:
    """
    LCI = (11.7 + 3.1*sqrt(ws)) * (40 - T)
          + 481 + 418*(1 - exp(-0.04*R))

    NARCliM2 bias-adjusted inputs used here:
        tas_c    : mean of tasmaxAdjust and tasminAdjust, degC
        ws2m     : 2 m wind, m s-1
        pr_mmday : prAdjust, mm day-1
    """
    lci = (
        (11.7 + 3.1 * np.sqrt(ws2m)) * (40.0 - tas_c)
        + 481.0
        + 418.0 * (1.0 - np.exp(-0.04 * pr_mmday))
    )

    lci.name = "lci"
    lci.attrs.update({
        "long_name": "Livestock Chill Index",
        "units": "kJ m-2 hr-1",
    })

    return lci


def build_var_file_lists(
    model: str,
    variant: str,
    rcm: str,
    experiment: str,
) -> dict[str, list[str]]:
    vars_needed = ["tasmaxAdjust", "tasminAdjust", "sfcWindAdjust", "prAdjust"]

    out = {}
    for var in vars_needed:
        files = find_var_files(model, variant, rcm, experiment, var)
        if not files:
            raise FileNotFoundError(
                f"No files found for {member_label(model, rcm)} "
                f"{experiment} {var}"
            )
        out[var] = files

    return out


def get_reference_grid(var_files: dict[str, list[str]]) -> tuple[xr.DataArray, xr.DataArray]:
    ref_path = var_files["tasmaxAdjust"][0]
    ds_ref = xr.open_dataset(ref_path, engine="h5netcdf")
    return ds_ref["lat"], ds_ref["lon"]


def compute_daily_lci_for_year(
    var_files: dict[str, list[str]],
    year: int,
    z0_on_grid: xr.DataArray,
) -> xr.DataArray:
    """
    Compute one calendar year of daily LCI for one NARCliM2 member.
    """
    tasmax = open_year_files(files_for_year(var_files["tasmaxAdjust"], year), "tasmaxAdjust", year)
    tasmin = open_year_files(files_for_year(var_files["tasminAdjust"], year), "tasminAdjust", year)
    wind10 = open_year_files(files_for_year(var_files["sfcWindAdjust"], year), "sfcWindAdjust", year)
    pr = open_year_files(files_for_year(var_files["prAdjust"], year), "prAdjust", year)

    tasmax = drop_feb29(tasmax)
    tasmin = drop_feb29(tasmin)
    wind10 = drop_feb29(wind10)
    pr = drop_feb29(pr)

    tasmax, tasmin, wind10, pr = xr.align(
        tasmax,
        tasmin,
        wind10,
        pr,
        join="inner",
    )

    tas_c = ((tasmax + tasmin) / 2.0).astype("float32")
    ws2m = wind_10m_to_2m(wind10.astype("float32"), z0_on_grid)
    pr_mmday = pr.astype("float32")

    lci = calculate_lci_xr(ws2m, tas_c, pr_mmday)
    return lci.transpose("time", "lat", "lon")


def compute_daily_lci_for_season(
    var_files: dict[str, list[str]],
    year: int,
    season: str,
    z0_on_grid: xr.DataArray,
) -> xr.DataArray:
    """
    Return daily LCI for one labelled season.

    dry Y = May-Oct Y
    wet Y = Nov-Dec Y-1 + Jan-Mar Y (April excluded, matching
            the same fix applied in synoptic_composite_lci_top5pct.py)
    """
    if season == "dry":
        lci_y = compute_daily_lci_for_year(var_files, year, z0_on_grid)
        return lci_y.sel(time=lci_y.time.dt.month.isin(sorted(DRY_MONTHS)))

    if season == "wet":
        lci_prev = compute_daily_lci_for_year(var_files, year - 1, z0_on_grid)
        lci_curr = compute_daily_lci_for_year(var_files, year, z0_on_grid)

        nov_dec = lci_prev.sel(time=lci_prev.time.dt.month.isin([11, 12]))
        jan_mar = lci_curr.sel(time=lci_curr.time.dt.month.isin([1, 2, 3]))

        return xr.concat([nov_dec, jan_mar], dim="time").sortby("time")

    raise ValueError(f"Unknown season: {season}")


# ============================================================
# TIMING METRICS
# ============================================================

def timing_metrics_one_season(
    lci: xr.DataArray,
    thresholds: list[int],
) -> xr.Dataset:
    """
    Compute first/last/span/frequency metrics for one season.

    Returns dims:
        threshold, lat, lon
    """
    thr = xr.DataArray(
        np.array(thresholds, dtype=np.float32),
        dims=("threshold",),
        coords={"threshold": np.array(thresholds, dtype=np.float32)},
        attrs={"units": "kJ m-2 hr-1"},
    )

    exceed = lci.expand_dims({"threshold": thr}) >= thr

    season_day = xr.DataArray(
        np.arange(1, lci.sizes["time"] + 1, dtype=np.float32),
        dims=("time",),
        coords={"time": lci.time},
        name="season_day",
    )

    event_days = exceed.sum("time", skipna=True).astype("int16")
    any_event = event_days > 0

    first_day = season_day.where(exceed).min("time", skipna=True)
    last_day = season_day.where(exceed).max("time", skipna=True)

    first_day = first_day.where(any_event).astype("float32")
    last_day = last_day.where(any_event).astype("float32")
    span_days = (last_day - first_day + 1).where(any_event).astype("float32")

    return xr.Dataset({
        "first_day": first_day,
        "last_day": last_day,
        "span_days": span_days,
        "event_days": event_days.astype("float32"),
        "event_probability": any_event.astype("float32"),
    })


def compute_member_period_climatology(
    model: str,
    variant: str,
    rcm: str,
    experiment: str,
    period_label: str,
    period: tuple[int, int],
    thresholds: list[int],
    seasons: list[str],
    z0_raw: xr.DataArray,
) -> xr.Dataset:
    """
    Compute period climatology for one member and one experiment/period.
    """
    label = member_label(model, rcm)
    year0, year1 = period

    print("\n" + "=" * 80)
    print(f"[MEMBER] {label} | {experiment} | {period_label} ({year0}-{year1})")
    print("=" * 80)

    var_files = build_var_file_lists(model, variant, rcm, experiment)
    ref_lat, ref_lon = get_reference_grid(var_files)

    print(f"[{label}] regridding z0 to NARCliM2 grid")
    z0_grid = regrid_z0(z0_raw, ref_lat, ref_lon)

    season_clims = []

    for season in seasons:
        yearly = []

        for year in range(year0, year1 + 1):
            print(f"[{label}] {experiment} {period_label} {season} {year}")

            try:
                lci = compute_daily_lci_for_season(
                    var_files=var_files,
                    year=year,
                    season=season,
                    z0_on_grid=z0_grid,
                )
            except Exception as exc:
                print(f"  [WARN] skipping {label} {season} {year}: {exc}")
                continue

            if lci.sizes.get("time", 0) == 0:
                print(f"  [WARN] empty season for {label} {season} {year}")
                continue

            ds_y = timing_metrics_one_season(lci, thresholds)
            ds_y = ds_y.expand_dims(year=[year])
            yearly.append(ds_y)

        if not yearly:
            print(f"  [WARN] no yearly outputs for {label} {season}")
            continue

        ds_season = xr.concat(yearly, dim="year")

        clim = xr.Dataset({
            "first_day": ds_season["first_day"].mean("year", skipna=True),
            "last_day": ds_season["last_day"].mean("year", skipna=True),
            "span_days": ds_season["span_days"].mean("year", skipna=True),
            "event_days": ds_season["event_days"].mean("year", skipna=True),
            "event_probability": ds_season["event_probability"].mean("year", skipna=True),
        })

        clim = clim.expand_dims(season=[season])
        season_clims.append(clim)

    if not season_clims:
        raise RuntimeError(f"No seasonal climatologies produced for {label} {experiment}")

    ds_member = xr.concat(season_clims, dim="season")
    ds_member = ds_member.expand_dims(member=[label])
    ds_member = ds_member.transpose("member", "season", "threshold", "lat", "lon")

    ds_member.attrs.update({
        "member": label,
        "experiment": experiment,
        "period_label": period_label,
        "period_years": f"{year0}-{year1}",
        "wind_height_correction": (
            "sfcWindAdjust converted from 10 m to 2 m using "
            "v2 = v10 * ln(2/z0) / ln(10/z0)"
        ),
        "z0_source": Z0_PATH,
        "lci_formula": (
            "(11.7 + 3.1*sqrt(ws2m)) * (40 - T) + "
            "481 + 418*(1 - exp(-0.04*R))"
        ),
    })

    return ds_member


def ensemble_from_members(member_datasets: list[xr.Dataset]) -> xr.Dataset:
    """
    Combine member period climatologies into ensemble mean and standard deviation.
    """
    all_members = xr.concat(member_datasets, dim="member")

    out = xr.Dataset(coords={
        "season": all_members["season"],
        "threshold": all_members["threshold"],
        "lat": all_members["lat"],
        "lon": all_members["lon"],
    })

    for var in [
        "first_day",
        "last_day",
        "span_days",
        "event_days",
        "event_probability",
    ]:
        out[f"{var}_ensmean"] = all_members[var].mean("member", skipna=True).astype("float32")
        out[f"{var}_ensstd"] = all_members[var].std("member", skipna=True).astype("float32")
        out[f"{var}_nmodels"] = all_members[var].notnull().sum("member").astype("uint8")

    out = out.assign_coords(member=all_members["member"])
    out.attrs.update(all_members.attrs)
    out.attrs["members_used"] = ", ".join(map(str, all_members["member"].values))
    out.attrs["n_members"] = int(all_members.sizes["member"])

    return out


def compute_period_ensemble(
    experiment: str,
    period_label: str,
    period: tuple[int, int],
    thresholds: list[int],
    seasons: list[str],
    z0_raw: xr.DataArray,
    outdir: Path,
    write_per_member: bool = False,
) -> xr.Dataset:
    member_datasets = []

    per_member_dir = outdir / "per_member" / experiment / period_label
    ensure_dir(per_member_dir)

    for model, variant, rcm in NARCLIM_MEMBERS:
        label = member_label(model, rcm)

        try:
            ds_m = compute_member_period_climatology(
                model=model,
                variant=variant,
                rcm=rcm,
                experiment=experiment,
                period_label=period_label,
                period=period,
                thresholds=thresholds,
                seasons=seasons,
                z0_raw=z0_raw,
            )
        except Exception as exc:
            print(f"[WARN] member failed: {label} {experiment} {period_label}: {exc}")
            continue

        if write_per_member:
            mp = per_member_dir / f"timing_{label}_{experiment}_{period_label}.nc"
            write_netcdf(ds_m, mp)

        member_datasets.append(ds_m)

    if len(member_datasets) < 2:
        raise RuntimeError(
            f"Only {len(member_datasets)} members completed for "
            f"{experiment} {period_label}; need at least 2."
        )

    ens = ensemble_from_members(member_datasets)
    ens.attrs.update({
        "experiment": experiment,
        "period_label": period_label,
        "period_years": f"{period[0]}-{period[1]}",
        "note": (
            "Timing metrics computed per member from daily LCI, then averaged "
            "over members. First/last/span ignore years with no event at a grid cell."
        ),
    })

    return ens


def add_change_from_historical(
    future: xr.Dataset,
    historical: xr.Dataset,
) -> xr.Dataset:
    """
    Add future-minus-historical change fields to a future ensemble dataset.
    """
    out = future.copy()

    for var in [
        "first_day",
        "last_day",
        "span_days",
        "event_days",
        "event_probability",
    ]:
        out[f"{var}_change"] = (
            future[f"{var}_ensmean"] - historical[f"{var}_ensmean"]
        ).astype("float32")

        out[f"{var}_change"].attrs.update({
            "long_name": f"{var} future minus historical",
            "units": future[f"{var}_ensmean"].attrs.get("units", ""),
        })

    out.attrs["change_baseline"] = "NARCliM2 historical 1990-2014"
    return out


# ============================================================
# NETCDF WRITING
# ============================================================

def write_netcdf(ds: xr.Dataset, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    enc = {}
    for v in ds.data_vars:
        if ds[v].dtype.kind in "iu":
            enc[v] = {"zlib": True, "complevel": 4, "shuffle": True}
        else:
            enc[v] = {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
            }

    tmp = str(path) + ".tmp"
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)

    print(f"[OK] wrote {path}")


# ============================================================
# PLOTTING
# ============================================================

def setup_map(ax) -> None:
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=10)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="0.15", zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.35", zorder=4)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="0.55", zorder=4)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.25,
        color="grey",
        alpha=0.4,
        linestyle="--",
    )
    gl.xlocator = mticker.FixedLocator([115, 125, 135, 145, 155])
    gl.ylocator = mticker.FixedLocator([-40, -30, -20, -10])


def pcolormesh_map(ax, da, cmap, norm):
    return ax.pcolormesh(
        da["lon"].values,
        da["lat"].values,
        da.values,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )


def date_norm_and_ticks(season: str):
    if season == "dry":
        levels = np.arange(1, 185 + 1, 15)
        ticks = DRY_DAY_TICKS
        labels = DRY_DAY_LABELS
    elif season == "wet":
        levels = np.arange(1, 151 + 1, 15)
        ticks = WET_DAY_TICKS
        labels = WET_DAY_LABELS
    else:
        raise ValueError(season)

    cmap = plt.get_cmap("viridis", len(levels) - 1).copy()
    cmap.set_bad("white")
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    return cmap, norm, ticks, labels


def span_norm():
    levels = np.arange(0, 185 + 1, 15)
    cmap = plt.get_cmap("YlOrRd", len(levels) - 1).copy()
    cmap.set_bad("white")
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    ticks = np.arange(0, 181, 30)
    labels = [str(int(t)) for t in ticks]
    return cmap, norm, ticks, labels


def change_norm(limit: float = 60.0):
    levels = np.arange(-limit, limit + 1, 10)
    cmap = plt.get_cmap("RdBu_r", len(levels) - 1).copy()
    cmap.set_bad("white")
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    ticks = np.arange(-limit, limit + 1, 20)
    labels = [str(int(t)) for t in ticks]
    return cmap, norm, ticks, labels


def plot_combo(
    historical: xr.Dataset,
    future: xr.Dataset,
    scenario: str,
    period_label: str,
    season: str,
    threshold: int,
    outpath: str | Path,
    min_event_probability: float = 0.10,
    dpi: int = 220,
) -> None:
    """
    Plot 3 x 3 timing figure:
        rows: historical, future, change
        cols: first day, last day, span days
    """
    hist = historical.sel(season=season, threshold=float(threshold))
    fut = future.sel(season=season, threshold=float(threshold))

    hist_mask = hist["event_probability_ensmean"] >= min_event_probability
    fut_mask = fut["event_probability_ensmean"] >= min_event_probability
    change_mask = hist_mask & fut_mask

    fields = [
        (
            hist["first_day_ensmean"].where(hist_mask),
            fut["first_day_ensmean"].where(fut_mask),
            fut["first_day_change"].where(change_mask),
            "Mean first chill date",
            "date",
        ),
        (
            hist["last_day_ensmean"].where(hist_mask),
            fut["last_day_ensmean"].where(fut_mask),
            fut["last_day_change"].where(change_mask),
            "Mean last chill date",
            "date",
        ),
        (
            hist["span_days_ensmean"].where(hist_mask),
            fut["span_days_ensmean"].where(fut_mask),
            fut["span_days_change"].where(change_mask),
            "Mean span length",
            "span",
        ),
    ]

    date_cmap, date_norm, date_ticks, date_labels = date_norm_and_ticks(season)
    span_cmap, span_norm_obj, span_ticks, span_labels = span_norm()
    chg_cmap, chg_norm, chg_ticks, chg_labels = change_norm(60)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(15.5, 12))
    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[1, 1, 1, 0.08],
        hspace=0.08,
        wspace=0.04,
        left=0.04,
        right=0.98,
        top=0.91,
        bottom=0.10,
    )

    axes = [
        [fig.add_subplot(gs[row, col], projection=proj) for col in range(3)]
        for row in range(3)
    ]

    row_labels = [
        "Historical\n1990-2014",
        f"{scenario}\n{period_label}",
        "Future - historical",
    ]

    col_titles = [f[3] for f in fields]

    pc_date = None
    pc_span = None
    pc_chg = None

    for col, (hist_da, fut_da, chg_da, title, kind) in enumerate(fields):
        for row in range(3):
            ax = axes[row][col]
            setup_map(ax)

            if row == 0:
                da = hist_da
                cmap = date_cmap if kind == "date" else span_cmap
                norm = date_norm if kind == "date" else span_norm_obj
            elif row == 1:
                da = fut_da
                cmap = date_cmap if kind == "date" else span_cmap
                norm = date_norm if kind == "date" else span_norm_obj
            else:
                da = chg_da
                cmap = chg_cmap
                norm = chg_norm

            pc = pcolormesh_map(ax, da, cmap, norm)

            if row in (0, 1) and kind == "date":
                pc_date = pc
            elif row in (0, 1) and kind == "span":
                pc_span = pc
            else:
                pc_chg = pc

            if row == 0:
                ax.set_title(title, fontsize=13, fontweight="bold")

            if col == 0:
                ax.text(
                    -0.04,
                    0.5,
                    row_labels[row],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="0.25",
                )

    # Colourbars: date, span, change
    cax_date = fig.add_subplot(gs[3, 0])
    cb_date = fig.colorbar(pc_date, cax=cax_date, orientation="horizontal", ticks=date_ticks)
    cb_date.ax.set_xticklabels(date_labels, rotation=35, ha="right", fontsize=8)
    cb_date.set_label("Day of season", fontsize=10)

    cax_span = fig.add_subplot(gs[3, 1])
    cb_span = fig.colorbar(pc_span, cax=cax_span, orientation="horizontal", ticks=span_ticks)
    cb_span.ax.set_xticklabels(span_labels, fontsize=8)
    cb_span.set_label("Days", fontsize=10)

    cax_chg = fig.add_subplot(gs[3, 2])
    cb_chg = fig.colorbar(pc_chg, cax=cax_chg, orientation="horizontal", ticks=chg_ticks)
    cb_chg.ax.set_xticklabels(chg_labels, fontsize=8)
    cb_chg.set_label("Change, days", fontsize=10)

    fig.suptitle(
        (
            f"NARCliM2 LCI event timing: {SEASON_LABELS[season]}, "
            f"threshold {threshold} kJ m$^{{-2}}$ hr$^{{-1}}$"
        ),
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )

    fig.text(
        0.04,
        0.035,
        (
            f"Masked where event probability is < {min_event_probability:.0%} "
            "in either historical or future period. "
            "Positive date change means later in the season."
        ),
        fontsize=9,
        color="0.35",
    )

    outpath = Path(outpath)
    ensure_dir(outpath.parent)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[OK] wrote {outpath}")


def plot_all(
    historical: xr.Dataset,
    future_by_scenario_period: dict[tuple[str, str], xr.Dataset],
    thresholds: list[int],
    seasons: list[str],
    outdir: Path,
    min_event_probability: float,
    dpi: int,
) -> None:
    figdir = outdir / "figures"
    ensure_dir(figdir)

    for (scenario, period_label), future in future_by_scenario_period.items():
        for season in seasons:
            for threshold in thresholds:
                outpath = (
                    figdir
                    / f"timing_maps_{scenario}_{period_label}_{season}_thr{threshold}.png"
                )

                plot_combo(
                    historical=historical,
                    future=future,
                    scenario=scenario,
                    period_label=period_label,
                    season=season,
                    threshold=threshold,
                    outpath=outpath,
                    min_event_probability=min_event_probability,
                    dpi=dpi,
                )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute NARCliM2 LCI event timing maps with 10 m -> 2 m "
            "wind correction, for historical and future periods."
        )
    )

    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--scenarios", default="ssp126,ssp370")
    ap.add_argument("--periods", default=",".join(FUTURE_PERIODS),
                    help="comma-separated future period labels, e.g. 2040-2060")
    ap.add_argument("--reuse-historical", action="store_true",
                    help="load the existing historical NetCDF instead of recomputing")
    ap.add_argument("--seasons", default="wet,dry")
    ap.add_argument("--thresholds", default="1000,1100,1200")
    ap.add_argument("--write-per-member", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--min-event-probability", type=float, default=0.10)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    periods = [p.strip() for p in args.periods.split(",") if p.strip()]
    bad = [p for p in periods if p not in FUTURE_PERIODS]
    if bad:
        ap.error(f"unknown periods: {bad}\nvalid: {list(FUTURE_PERIODS)}")
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    thresholds = [int(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    hist_path = outdir / "narclim2_lci_event_timing_historical_1990-2014.nc"

    future_paths = {
        (scenario, period_label): (
            outdir / f"narclim2_lci_event_timing_{scenario}_{period_label}.nc"
        )
        for scenario in scenarios
        for period_label in periods
    }

    if args.plot_only:
        print(f"[INFO] opening historical: {hist_path}")
        historical = xr.open_dataset(hist_path, decode_timedelta=False)

        future_by_scenario_period = {}
        for key, path in future_paths.items():
            print(f"[INFO] opening future: {path}")
            future_by_scenario_period[key] = xr.open_dataset(path, decode_timedelta=False)

    else:
        print("[INFO] loading z0")
        z0_raw = load_z0_static()

        if args.reuse_historical and hist_path.exists():
            print(f"[INFO] reusing existing historical baseline: {hist_path}")
            historical = xr.open_dataset(hist_path, decode_timedelta=False).load()
        else:
            print("[INFO] computing historical baseline")
            historical = compute_period_ensemble(
                experiment="historical",
                period_label="1990-2014",
                period=HIST_PERIOD,
                thresholds=thresholds,
                seasons=seasons,
                z0_raw=z0_raw,
                outdir=outdir,
                write_per_member=args.write_per_member,
            )
            write_netcdf(historical, hist_path)

        future_by_scenario_period = {}

        for scenario in scenarios:
            for period_label, period in [(p, FUTURE_PERIODS[p]) for p in periods]:
                print("\n" + "#" * 80)
                print(f"[FUTURE] {scenario} {period_label}")
                print("#" * 80)

                future = compute_period_ensemble(
                    experiment=scenario,
                    period_label=period_label,
                    period=period,
                    thresholds=thresholds,
                    seasons=seasons,
                    z0_raw=z0_raw,
                    outdir=outdir,
                    write_per_member=args.write_per_member,
                )

                future = add_change_from_historical(future, historical)

                outpath = future_paths[(scenario, period_label)]
                write_netcdf(future, outpath)

                future_by_scenario_period[(scenario, period_label)] = future

    if not args.no_plots:
        plot_all(
            historical=historical,
            future_by_scenario_period=future_by_scenario_period,
            thresholds=thresholds,
            seasons=seasons,
            outdir=outdir,
            min_event_probability=args.min_event_probability,
            dpi=args.dpi,
        )

    print("[DONE]")


if __name__ == "__main__":
    main()