#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_synoptic_composites_compact_focal_with_barra_vectors.py

Compact historical synoptic-composite figure for the three focal region-season
combinations used in the LCI paper.

Figure layout
-------------
Rows:    Cobar-Lachlan May-Oct | Yilgarn-Coolgardie May-Oct | Winton Nov-Mar
Columns: MSLP | temperature | precipitation
Period:  Historical only, 1990-2014 composites from the existing synoptic composite NetCDFs

Winton's wet season is Nov-Mar (April excluded, matching the fix applied in
synoptic_composite_lci_top5pct.py -- both the filled NARCliM2 fields and the
BARRA-R2 wind-vector overlay now use the same April-excluded event set, so
they stay consistent with each other).

By default, all fields are plotted as event anomalies relative to the matching
seasonal climatology:
    {field}_{season}_hist - {field}_{season}_hist_clim

Inputs
------
Reads per-member files produced by synoptic_composite_lci_top5pct.py:
    {COMPOSITE_DIR}/{member}_composite_{region}_mid_century.nc

Expected variables include:
    psl_dry_hist, tas_dry_hist, pr_dry_hist, and *_clim versions
    psl_wet_hist, tas_wet_hist, pr_wet_hist, and *_clim versions

Usage
-----
    module use /g/data/xp65/public/modules
    module load conda/analysis3

    python plot_synoptic_composites_compact_focal.py

Optional:
    python plot_synoptic_composites_compact_focal.py --mode absolute
    python plot_synoptic_composites_compact_focal.py --dpi 300
    python plot_synoptic_composites_compact_focal.py --barra-wind-vectors
    python plot_synoptic_composites_compact_focal.py \
        --composite-dir /scratch/dx2/rt9243/chill_projections/synoptic_composites \
        --figdir /scratch/dx2/rt9243/chill_projections/synoptic_composites/figures
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_COMPOSITE_DIR = "/scratch/dx2/rt9243/chill_projections/synoptic_composites"
DEFAULT_FIGDIR = "/scratch/dx2/rt9243/chill_projections/synoptic_composites/figures"


def truncate_cmap(name: str, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    """Return a colormap using only the [minval, maxval] slice of a named one."""
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_trunc",
        base(np.linspace(minval, maxval, n)),
    )


# Temperature event anomalies are negative almost everywhere (these are cold
# composites), so the colourbar is cut at zero rather than spanning
# +/-6. Using the lower (blue) half of RdBu_r keeps the colours identical to
# the previous symmetric version -- a full RdBu_r squeezed into [-6, 0] would
# put red at zero.
#
# The tas column uses extend="min" rather than "both" (see FIELDS), so grid
# cells warmer than the seasonal mean are left unfilled and read as blank
# white, matching zero. Filling them with an over-colour instead made large
# parts of the Winton and Yilgarn panels solid red for what is a
# de-emphasised, mostly weak positive tail.
TAS_ANOM_CMAP = truncate_cmap("RdBu_r", 0.0, 0.5)

# User-requested focal season only.  Dry season is labelled "cold-season" here.
REGIONS = [
    {
        "key": "Cobar_Lachlan",
        "row_label": "Cobar-Lachlan\nMay-Oct",
        "season": "dry",
        "marker_lon": 146.0,
        "marker_lat": -32.5,
    },
    {
        "key": "Yilgarn_Coolgardie",
        "row_label": "Yilgarn-Coolgardie\nMay-Oct",
        "season": "dry",
        "marker_lon": 120.0,
        "marker_lat": -30.5,
    },
    {
        "key": "Winton",
        "row_label": "Winton\nNov-Mar",
        "season": "wet",
        "marker_lon": 143.0,
        "marker_lat": -22.5,
    },
]

FIELDS = [
    {
        "key": "mslp",
        "stem": "psl",
        "title": "MSLP",
        "units_absolute": "hPa",
        "units_anomaly": "hPa",
        "levels_absolute": np.arange(1000, 1030, 2),
        "levels_anomaly": np.arange(-6, 6.5, 0.5),
        "cmap_absolute": "Blues_r",
        "cmap_anomaly": "RdBu_r",
        "extend": "both",
    },
    {
        "key": "tas",
        "stem": "tas",
        "title": "Temperature",
        "units_absolute": "°C",
        "units_anomaly": "°C",
        "levels_absolute": np.arange(-5, 40, 2),
        "levels_anomaly": np.arange(-6, 0.5, 0.5),  # cut at zero: cold composites, signal is negative
        "cmap_absolute": "RdYlBu_r",
        "cmap_anomaly": TAS_ANOM_CMAP,
        # Anomaly levels stop at zero, so anything warmer than the seasonal
        # mean is left unfilled (blank white) instead of getting an over-colour.
        "extend_anomaly": "min",
        "extend": "both",
    },
    {
        "key": "pr",
        "stem": "pr",
        "title": "Precipitation",
        "units_absolute": "mm day$^{-1}$",
        "units_anomaly": "mm day$^{-1}$",
        "levels_absolute": np.arange(0, 12.5, 0.5),
        "levels_anomaly": np.arange(-1.5, 6.5, 0.5),  # floored at -1.5: signal is mostly positive
        "cmap_absolute": "YlGnBu",
        "cmap_anomaly": "BrBG",
        "extend": "both",
    },
]

SEASON_TEXT = {"dry": "May-October", "wet": "November-March"}

LON_MIN, LON_MAX = 112.0, 154.0
LAT_MIN, LAT_MAX = -44.5, -9.5
WIND_ABS_LEVELS = np.arange(3, 12, 2)
WIND_ANOM_LEVELS = np.arange(-3, 3.5, 0.5)

# BARRA-R2 observed wind-vector overlay.
# These vectors are optional context only: the filled fields remain NARCliM2
# synoptic composites; arrows are independently composited from BARRA-R2
# historical top-5% LCI event days for the same region/month window.
HIST_START, HIST_END = 1990, 2014
DEFAULT_BARRA_LCI_DIR = "/scratch/dx2/rt9243/barra_daily_lci"
DEFAULT_BARRA_WIND_CACHE_DIR = (
    "/scratch/dx2/rt9243/chill_projections/synoptic_composites/barra_wind_vector_cache"
)
BARRA_WIND_ROOT = (
    "/g/data/ob53/BARRA2/output/reanalysis/AUS-11/BOM/ERA5/"
    "historical/hres/BARRA-R2/v1/day"
)
BARRA_WIND_VERSION = "v20250528"

Z0_PATH = "/g/data/dx2/rt9243/Datasets/sfc_rough_len_Aust_fc.nc"
Z0_MIN, Z0_MAX = 1e-4, 1.9

_LGA_SHP = (
    "/g/data/dx2/rt9243/Datasets/"
    "LGA_2025_AUST_GDA2020/LGA_2025_AUST_GDA2020.shp"
)

BARRA_REGION_SPECS = {
    "Cobar_Lachlan": {
        "shp": _LGA_SHP,
        "region_col": "LGA_NAME25",
        "region_values": ["Cobar", "Lachlan"],
        "dissolve": True,
    },
    "Yilgarn_Coolgardie": {
        "shp": _LGA_SHP,
        "region_col": "LGA_NAME25",
        "region_values": ["Yilgarn", "Coolgardie"],
        "dissolve": True,
    },
    "Winton": {
        "shp": _LGA_SHP,
        "region_col": "LGA_NAME25",
        "region_values": ["Winton"],
        "dissolve": False,
    },
}

# ── Data helpers ──────────────────────────────────────────────────────────────

def _open_dataset_loaded(path: str) -> xr.Dataset:
    """Open a small composite file and load it so file handles can close."""
    try:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            return ds.load()
    except Exception:
        with xr.open_dataset(path) as ds:
            return ds.load()


def load_ensemble_mean(region_key: str, composite_dir: str) -> Tuple[xr.Dataset, int, List[str]]:
    """Load all member composite files for one region and return ensemble mean."""
    pattern = os.path.join(composite_dir, f"*_composite_{region_key}_mid_century.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No composite files matched: {pattern}")

    datasets: List[xr.Dataset] = []
    members: List[str] = []
    for path in files:
        datasets.append(_open_dataset_loaded(path))
        members.append(os.path.basename(path).split("_composite_")[0])

    # Capture each member's n_days per variable BEFORE the mean collapses
    # the member dimension. xr.concat's default combine_attrs="override"
    # keeps only the first dataset's attrs, and .mean(keep_attrs=True) just
    # carries that through -- so without this step, every panel's n= was
    # silently reporting one member's day count (whichever file sorted
    # first alphabetically), not a total across the ensemble.
    n_days_per_member: Dict[str, List[int]] = {}
    for var in datasets[0].data_vars:
        vals = [ds[var].attrs.get("n_days") for ds in datasets]
        if any(v is not None for v in vals):
            n_days_per_member[var] = [int(v) if v is not None else None for v in vals]

    stacked = xr.concat(datasets, dim="member").assign_coords(member=members)
    ens = stacked.mean("member", skipna=True, keep_attrs=True)
    ens.attrs.update(datasets[0].attrs)
    ens.attrs["n_members"] = len(datasets)
    ens.attrs["n_days_per_member"] = n_days_per_member

    for ds in datasets:
        ds.close()
    stacked.close()

    return ens, len(files), files


def field_to_plot(ens: xr.Dataset, stem: str, season: str, mode: str) -> xr.DataArray:
    """Return one historical field: absolute composite or event anomaly."""
    var = f"{stem}_{season}_hist"
    if var not in ens:
        raise KeyError(f"Missing variable {var!r}. Available variables include: {list(ens.data_vars)[:12]}")

    da = ens[var]
    if mode == "absolute":
        return da

    clim_var = f"{var}_clim"
    if clim_var not in ens:
        raise KeyError(
            f"Missing climatology variable {clim_var!r}; use --mode absolute, "
            "or regenerate composites with *_clim fields."
        )
    out = da - ens[clim_var]
    out.attrs.update(da.attrs)
    return out


def n_days_text(ens: xr.Dataset, stem: str, season: str) -> str:
    """
    Build the panel's n= annotation from the per-member day counts captured
    in load_ensemble_mean (ens.attrs["n_days_per_member"]) -- NOT from the
    collapsed ensemble-mean variable's own attrs, which only ever reflect
    one member (see the comment in load_ensemble_mean).

    Reports the total number of composited event-days summed across all
    members, i.e. the true ensemble sample size.
    """
    var = f"{stem}_{season}_hist"
    per_member = ens.attrs.get("n_days_per_member", {})
    vals = [v for v in per_member.get(var, []) if v is not None]
    if not vals:
        return ""
    return f"n={sum(vals)}"


def print_n_days_diagnostics(region_key: str, season: str, ens: xr.Dataset) -> None:
    """Print the full per-member day-count breakdown, so an unusually uneven
    split between members (which n_days_text's total would otherwise hide)
    is visible when the script runs."""
    per_member = ens.attrs.get("n_days_per_member", {})
    if not per_member:
        return
    print(f"\n[n_days per member] {region_key} ({season})")
    for var, vals in per_member.items():
        vals_clean = [v for v in vals if v is not None]
        if vals_clean:
            print(f"  {var:20s} {vals_clean}  total={sum(vals_clean)}")


# ── BARRA-R2 observed wind-vector helpers ─────────────────────────────────────

def barra_lci_path(year: int, barra_lci_dir: str) -> str:
    return os.path.join(barra_lci_dir, f"barra_lci_daily_aus_land_{year}.nc")


def barra_wind_path(var: str, year: int, month: int) -> str:
    return os.path.join(
        BARRA_WIND_ROOT,
        var,
        BARRA_WIND_VERSION,
        f"{var}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_day_"
        f"{year}{month:02d}-{year}{month:02d}.nc",
    )


def load_z0_static(z0_path: str = Z0_PATH) -> xr.DataArray:
    ds = xr.open_dataset(z0_path)
    z0 = ds["sfc_rough_len"].mean("time", skipna=True)
    return z0.clip(min=Z0_MIN, max=Z0_MAX).load()


def regrid_z0(z0_raw: xr.DataArray, target_lat, target_lon) -> xr.DataArray:
    if float(z0_raw.lat[0]) > float(z0_raw.lat[-1]):
        z0_raw = z0_raw.sortby("lat")
    if float(z0_raw.lon[0]) > float(z0_raw.lon[-1]):
        z0_raw = z0_raw.sortby("lon")

    # nearest-neighbour with extrapolation prevents a thin NaN rim where the
    # plotting extent slightly exceeds the ACCESS-G z0 domain.
    z0_interp = z0_raw.interp(
        lat=target_lat,
        lon=target_lon,
        method="nearest",
        kwargs={"fill_value": "extrapolate"},
    )
    return z0_interp.clip(min=Z0_MIN, max=Z0_MAX).load()


def wind_component_10m_to_2m(component: xr.DataArray, z0: xr.DataArray) -> xr.DataArray:
    """
    Log-law height correction for signed wind components.

    Do NOT clip signed components: negative uas/vas values are physically valid
    westward/southward winds.  The same positive scale factor is applied to both
    components, preserving direction while reducing vector magnitude from 10 m
    to 2 m.
    """
    c = np.log(2.0 / z0) / np.log(10.0 / z0)
    out = component * c
    out.attrs.update(component.attrs)
    out.attrs["height"] = "2 m"
    out.attrs["wind_height_correction"] = (
        "v2 = v10 * ln(2/z0) / ln(10/z0), applied to signed wind component; "
        f"z0 from {Z0_PATH}"
    )
    return out


def _open_barra_dataset(path: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="h5netcdf", mask_and_scale=True)
    except Exception:
        return xr.open_dataset(path, mask_and_scale=True)


def _time_strings(da_or_ds) -> np.ndarray:
    return da_or_ds.time.dt.strftime("%Y-%m-%d").values.astype(str)


def _season_and_year(date_index: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wet season is Nov-Mar (April excluded, matching the
    fix in synoptic_composite_lci_top5pct.py). Dry season is May-Oct.
    April falls in neither and comes back as None, so it's naturally
    dropped by the season_arr == season filter in
    _find_barra_event_dates -- it does NOT fall through to "dry".
    """
    months = date_index.month.values
    years = date_index.year.values
    wet_months = {11, 12, 1, 2, 3}
    dry_months = {5, 6, 7, 8, 9, 10}

    season = np.full(months.shape, None, dtype=object)
    season[np.isin(months, list(wet_months))] = "wet"
    season[np.isin(months, list(dry_months))] = "dry"

    is_wet = season == "wet"
    season_year = np.where(is_wet & (months >= 11), years + 1, years)
    return season, season_year


def _load_region_gdf_for_barra(region_key: str):
    import geopandas as gpd

    spec = BARRA_REGION_SPECS[region_key]
    gdf = gpd.read_file(spec["shp"])
    if gdf.crs is None:
        raise ValueError(f"No CRS found in shapefile: {spec['shp']}")

    gdf = gdf[gdf[spec["region_col"]].isin(spec["region_values"])].copy()
    if len(gdf) == 0:
        raise ValueError(
            f"No rows matched {spec['region_col']} in {spec['region_values']}"
        )

    if spec.get("dissolve", False):
        gdf = gdf.dissolve().explode(index_parts=False).dissolve()

    return gdf.to_crs("EPSG:4326")


def _region_mask_on_grid(region_key: str, lat: np.ndarray, lon: np.ndarray, buffer_deg: float = 1.0):
    """
    Build a small bbox mask and cosine-latitude weights for one region on a
    given lat/lon grid. Returns slices and a normalised 2-D weight DataArray.
    """
    import regionmask

    gdf = _load_region_gdf_for_barra(region_key)
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy

    lon_idx = np.where((lon >= bounds[0] - buffer_deg) & (lon <= bounds[2] + buffer_deg))[0]
    lat_idx = np.where((lat >= bounds[1] - buffer_deg) & (lat <= bounds[3] + buffer_deg))[0]
    if lon_idx.size == 0 or lat_idx.size == 0:
        raise ValueError(f"{region_key}: no BARRA grid cells inside region bbox")

    lon_slice = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
    lat_slice = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
    lon_sub = lon[lon_slice]
    lat_sub = lat[lat_slice]

    regs = regionmask.from_geopandas(gdf, overlap=False)
    mask = regs.mask(lon_sub, lat_sub)
    inside = np.isfinite(mask.values)
    if not inside.any():
        raise ValueError(f"{region_key}: region mask is empty on BARRA grid")

    raw_w = inside * np.cos(np.deg2rad(lat_sub))[:, np.newaxis]
    raw_w = raw_w.astype("float64")
    raw_w = raw_w / raw_w.sum()

    weights = xr.DataArray(
        raw_w,
        dims=("lat", "lon"),
        coords={"lat": lat_sub, "lon": lon_sub},
        name="region_weights",
    )
    return lat_slice, lon_slice, weights


def _find_barra_event_dates(region: dict, barra_lci_dir: str) -> Tuple[List[str], float]:
    """
    Identify observed BARRA-R2 top-5% regional LCI event dates for the region's
    focal month window over 1990-2014.

    Dry/May-Oct regions use calendar-year seasons.  Winton/Nov-Mar uses wet
    season-year labelling, so Nov-Dec 1989 are included for season-year 1990.
    """
    season = region["season"]
    first_year = HIST_START - 1 if season == "wet" else HIST_START
    years = range(first_year, HIST_END + 1)

    # Reference grid from the first available BARRA LCI file.
    ref_path = None
    for y in years:
        p = barra_lci_path(y, barra_lci_dir)
        if os.path.exists(p):
            ref_path = p
            break
    if ref_path is None:
        raise FileNotFoundError(f"No BARRA daily LCI files found in {barra_lci_dir}")

    with _open_barra_dataset(ref_path) as ref:
        lat = ref["lat"].values
        lon = ref["lon"].values

    lat_slice, lon_slice, weights = _region_mask_on_grid(region["key"], lat, lon)

    all_dates: List[str] = []
    all_values: List[float] = []

    for year in years:
        path = barra_lci_path(year, barra_lci_dir)
        if not os.path.exists(path):
            print(f"[WARN] Missing BARRA LCI file, skipping: {path}")
            continue

        with _open_barra_dataset(path) as ds:
            if "lci_daily" not in ds:
                raise KeyError(f"'lci_daily' not found in {path}")

            lci = ds["lci_daily"].isel(lat=lat_slice, lon=lon_slice)
            regional = (lci * weights).sum(("lat", "lon"), skipna=True).load()

            dates = pd.DatetimeIndex(lci.time.values.astype("datetime64[ns]"))
            season_arr, season_year = _season_and_year(dates)
            keep = (
                (season_arr == season)
                & (season_year >= HIST_START)
                & (season_year <= HIST_END)
            )

            all_dates.extend(dates[keep].strftime("%Y-%m-%d").tolist())
            all_values.extend(regional.values[keep].astype(float).tolist())

    if len(all_values) == 0:
        raise RuntimeError(f"No BARRA regional LCI values found for {region['key']} {season}")

    values = np.asarray(all_values, dtype=float)
    threshold = float(np.nanpercentile(values, 95.0))
    event_dates = [
        d for d, v in zip(all_dates, values)
        if np.isfinite(v) and v >= threshold
    ]

    print(
        f"[BARRA wind] {region['key']} {SEASON_TEXT[season]}: "
        f"{len(event_dates)} top-5% event days, p95={threshold:.1f}"
    )
    return event_dates, threshold


def _compute_barra_wind_vector_composite(region: dict, barra_lci_dir: str) -> xr.Dataset:
    """
    Composite BARRA-R2 uas/vas over observed top-5% regional LCI event days.
    Output vectors are corrected from 10 m to 2 m using the log wind profile.
    """
    event_dates, threshold = _find_barra_event_dates(region, barra_lci_dir)
    event_dates_by_month: Dict[Tuple[int, int], List[str]] = {}
    for d in event_dates:
        y, m = int(d[:4]), int(d[5:7])
        event_dates_by_month.setdefault((y, m), []).append(d)

    z0_raw = load_z0_static()

    usum = None
    vsum = None
    count = 0
    height_checked = False

    for (year, month), dates in sorted(event_dates_by_month.items()):
        upath = barra_wind_path("uas", year, month)
        vpath = barra_wind_path("vas", year, month)
        if not os.path.exists(upath) or not os.path.exists(vpath):
            print(f"[WARN] Missing BARRA wind file(s), skipping {year}-{month:02d}")
            continue

        with _open_barra_dataset(upath) as uds, _open_barra_dataset(vpath) as vds:
            uas = uds["uas"].sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))
            vas = vds["vas"].sel(lon=slice(LON_MIN, LON_MAX), lat=slice(LAT_MIN, LAT_MAX))

            # Defensive sorting, in case a future BARRA file stores coordinates differently.
            if float(uas.lat[0]) > float(uas.lat[-1]):
                uas = uas.sortby("lat")
                vas = vas.sortby("lat")
            if float(uas.lon[0]) > float(uas.lon[-1]):
                uas = uas.sortby("lon")
                vas = vas.sortby("lon")

            if not height_checked:
                try:
                    h = float(uds["height"].values)
                    if not np.isclose(h, 10.0):
                        print(f"[WARN] BARRA uas height is {h:g} m, not 10 m. Check before using.")
                    else:
                        print("[BARRA wind] uas/vas height coordinate = 10 m; applying 10 m -> 2 m correction.")
                except Exception:
                    print("[WARN] Could not read BARRA wind height coordinate; assuming 10 m.")
                height_checked = True

            time_str = _time_strings(uas)
            idx = np.where(np.isin(time_str, np.asarray(dates, dtype=str)))[0]
            if idx.size == 0:
                print(f"[WARN] No matching BARRA wind times for {year}-{month:02d}")
                continue

            uas = uas.isel(time=idx)
            vas = vas.isel(time=idx)

            z0 = regrid_z0(z0_raw, uas.lat, uas.lon)
            uas2 = wind_component_10m_to_2m(uas, z0)
            vas2 = wind_component_10m_to_2m(vas, z0)

            u_month = uas2.sum("time", skipna=True).load()
            v_month = vas2.sum("time", skipna=True).load()

            if usum is None:
                usum = u_month.copy(deep=True)
                vsum = v_month.copy(deep=True)
            else:
                usum = usum + u_month
                vsum = vsum + v_month
            count += int(idx.size)

    if count == 0 or usum is None or vsum is None:
        raise RuntimeError(f"No BARRA wind vectors accumulated for {region['key']}")

    uas_mean = (usum / count).astype("float32")
    vas_mean = (vsum / count).astype("float32")
    speed = np.sqrt(uas_mean**2 + vas_mean**2).astype("float32")

    out = xr.Dataset(
        {
            "uas_2m": uas_mean,
            "vas_2m": vas_mean,
            "sfcWind_2m": speed,
        }
    )
    out.attrs.update({
        "source": "BARRA-R2 historical uas/vas",
        "purpose": (
            "Observed historical wind-vector context for NARCliM2 historical "
            "synoptic composite figure; not a NARCliM2 vector field."
        ),
        "event_selection": (
            "BARRA-R2 regional daily LCI top 5% over the matching focal season "
            f"and {HIST_START}-{HIST_END} period."
        ),
        "region_key": region["key"],
        "season": region["season"],
        "season_text": SEASON_TEXT[region["season"]],
        "hist_start": HIST_START,
        "hist_end": HIST_END,
        "n_event_days": count,
        "regional_lci_p95": threshold,
        "wind_height": "2 m",
        "height_correction": (
            "10 m uas/vas corrected to 2 m using log wind profile with "
            f"z0 from {Z0_PATH}; signed components scaled without clipping."
        ),
    })
    return out


def load_or_build_barra_wind_vectors(
    region: dict,
    barra_lci_dir: str,
    cache_dir: str,
    overwrite: bool = False,
) -> xr.Dataset:
    """Load cached BARRA wind-vector composite, or build/cache it if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    # NOTE: this cache key does NOT encode the wet/dry month definition. If
    # WET_MONTHS/DRY_MONTHS in _season_and_year() ever change again (as they
    # did to exclude April for Winton), any existing cache file under this
    # same name is now stale and must be deleted or rebuilt with
    # --overwrite-barra-wind-cache -- it will NOT be invalidated automatically.
    cache_path = os.path.join(
        cache_dir,
        f"barra_r2_wind_vectors_{region['key']}_{region['season']}_{HIST_START}-{HIST_END}.nc",
    )

    if os.path.exists(cache_path) and not overwrite:
        print(f"[BARRA wind] Loading cached vectors: {cache_path}")
        with xr.open_dataset(cache_path) as ds:
            return ds.load()

    print(f"[BARRA wind] Building vectors for {region['key']} ({SEASON_TEXT[region['season']]})")
    ds = _compute_barra_wind_vector_composite(region, barra_lci_dir)
    ds.to_netcdf(cache_path)
    print(f"[BARRA wind] Wrote cache: {cache_path}")
    return ds


def add_barra_wind_vectors(
    ax,
    ds: xr.Dataset,
    *,
    vector_step: int,
    vector_scale: float,
    add_key: bool = False,
) -> None:
    """Draw BARRA-R2 observed 2 m wind vectors on a map axis."""
    lon = ds["lon"].values
    lat = ds["lat"].values
    u = ds["uas_2m"].values
    v = ds["vas_2m"].values

    step = max(1, int(vector_step))
    sl = (slice(None, None, step), slice(None, None, step))
    xq, yq = np.meshgrid(lon[::step], lat[::step])
    uq = u[sl]
    vq = v[sl]

    finite = np.isfinite(uq) & np.isfinite(vq)
    uq = np.where(finite, uq, np.nan)
    vq = np.where(finite, vq, np.nan)

    q = ax.quiver(
        xq,
        yq,
        uq,
        vq,
        transform=ccrs.PlateCarree(),
        color="0.10",
        scale=vector_scale,
        scale_units="width",
        width=0.0023,
        headwidth=3.4,
        headlength=4.3,
        headaxislength=3.8,
        pivot="middle",
        zorder=5,
    )

    if add_key:
        # Previously this sat right on the bottom axis edge (y=0.075), where
        # it was both getting clipped by the frame and blending into the
        # dense arrow field behind it. Moving it up into the panel interior
        # and giving it an opaque background fixes both -- no genuinely
        # empty patch of map is needed if the label brings its own backdrop.
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.02), 0.36, 0.085,
            boxstyle="round,pad=0.012",
            transform=ax.transAxes,
            facecolor="white", edgecolor="0.4", linewidth=0.6,
            alpha=0.88, zorder=8.5,
        ))
        # X is the arrow's CENTER, not its left end -- the quiver plot above
        # uses pivot="middle", and quiverkey always inherits the same pivot
        # from its parent Quiver (no way to override it independently here).
        # At U=5, scale=vector_scale, scale_units="width", the arrow is
        # ~5/vector_scale of the axes width, so roughly half of that extends
        # LEFT of X. X=0.055 put the arrow's left end right at (and slightly
        # past) the axes edge -- 0.115 leaves clear room on both sides.
        qk = ax.quiverkey(
            q,
            0.115,
            0.062,
            5,
            "5 m s$^{-1}$",
            labelpos="E",
            labelsep=0.13,  # inches, gap between arrow glyph and text (default
                             # 0.1); nudge this rather than moving the text
                             # directly -- QuiverKey recomputes the label's
                             # position from labelsep on every draw, so a
                             # manual qk.text.set_position() gets silently
                             # overwritten when the figure is saved.
            coordinates="axes",
            fontproperties={"size": 7.5},
            zorder=9,
        )



# ── Plot helpers ──────────────────────────────────────────────────────────────

def setup_map(ax, row_i: int, col_i: int) -> None:
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.85, edgecolor="0.12", zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="0.35", zorder=4)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="0.45", zorder=4)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.25,
        color="grey",
        alpha=0.45,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = col_i == 0
    gl.bottom_labels = row_i == 2
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}
    gl.xlocator = mticker.FixedLocator([115, 125, 135, 145])
    gl.ylocator = mticker.FixedLocator([-40, -30, -20])


def add_region_marker(ax, lon: float, lat: float) -> None:
    ax.plot(
        lon,
        lat,
        marker="o",
        markersize=11,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=1.8,
        transform=ccrs.PlateCarree(),
        zorder=7,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
    )


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.025,
        0.965,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=8,
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", boxstyle="round,pad=0.20"),
    )


def plot_compact(
        composite_dir: str,
        figdir: str,
        mode: str,
        dpi: int,
        outfile: str,
        wind_contours: bool = False,
        barra_wind_vectors: bool = False,
        barra_lci_dir: str = DEFAULT_BARRA_LCI_DIR,
        barra_wind_cache_dir: str = DEFAULT_BARRA_WIND_CACHE_DIR,
        overwrite_barra_wind_cache: bool = False,
        vector_step: int = 18,
        vector_scale: float = 75.0,
    ) -> Path:
    os.makedirs(figdir, exist_ok=True)
    proj = ccrs.PlateCarree()

    # Slightly wide, not tall.  Designed to survive shrinking in a manuscript draft.
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12.2, 9.2),
        subplot_kw={"projection": proj},
        gridspec_kw={"hspace": 0.06, "wspace": 0.035},
    )

    # Load each region once, because every field in a row comes from the same file set.
    region_data: Dict[str, Tuple[xr.Dataset, int]] = {}
    for spec in REGIONS:
        ens, n_members, _ = load_ensemble_mean(spec["key"], composite_dir)
        region_data[spec["key"]] = (ens, n_members)
        print_n_days_diagnostics(spec["key"], spec["season"], ens)

    # Optional observed historical BARRA-R2 wind vectors. These are built from
    # BARRA event days and cached, so the first run may take a little while.
    barra_vector_data: Dict[str, xr.Dataset] = {}
    if barra_wind_vectors:
        for spec in REGIONS:
            barra_vector_data[spec["key"]] = load_or_build_barra_wind_vectors(
                spec,
                barra_lci_dir=barra_lci_dir,
                cache_dir=barra_wind_cache_dir,
                overwrite=overwrite_barra_wind_cache,
            )

    contour_handles = [None, None, None]
    # Actual wind-speed values behind the MSLP-column contours, accumulated
    # across regions. contour() silently omits any requested level that falls
    # outside the data range, so the nominal WIND_*_LEVELS array is not what
    # ends up drawn -- the caption has to report the measured range instead.
    wind_observed: List[np.ndarray] = []
    letters = "abcdefghi"

    for row_i, region in enumerate(REGIONS):
        ens, n_members = region_data[region["key"]]
        lat = ens["lat"].values
        lon = ens["lon"].values
        x, y = np.meshgrid(lon, lat)

        for col_i, field in enumerate(FIELDS):
            ax = axes[row_i, col_i]
            setup_map(ax, row_i, col_i)

            da = field_to_plot(ens, field["stem"], region["season"], mode)
            data = np.ma.masked_invalid(da.values)

            levels = field[f"levels_{mode}"]
            cmap = field[f"cmap_{mode}"]
            # pr's anomaly range is asymmetric around zero (floored at -1.5
            # to de-emphasize the less-relevant negative tail), so a plain
            # linear level mapping would push white off
            # zero on the BrBG colormap. TwoSlopeNorm keeps zero centered.
            # tas is now cut at zero (see TAS_ANOM_CMAP) rather than being
            # symmetric, but it needs no norm: its levels end at zero, so a
            # plain linear mapping across a blue-to-white ramp already puts
            # white at zero. TwoSlopeNorm would in fact fail here, since it
            # requires vcenter < vmax and both are 0.
            norm = None
            if mode == "anomaly" and field["key"] == "pr":
                norm = TwoSlopeNorm(vmin=float(levels[0]), vcenter=0.0, vmax=float(levels[-1]))
            cf = ax.contourf(
                x,
                y,
                data,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend=field.get(f"extend_{mode}", field["extend"]),
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            contour_handles[col_i] = cf
            
            # Optional wind contours on the MSLP column only.
            # In anomaly mode, these are event wind-speed anomalies relative to seasonal climatology.
            # In absolute mode, these are absolute event wind speeds.
            if wind_contours and field["key"] == "mslp":
                wdata = field_to_plot(ens, "sfcWind", region["season"], mode)
                wlev = WIND_ANOM_LEVELS if mode == "anomaly" else WIND_ABS_LEVELS

                wvals = np.asarray(wdata.values)
                wfinite = wvals[np.isfinite(wvals)]
                if wfinite.size:
                    wind_observed.append(wfinite)

                cs = ax.contour(
                    x,
                    y,
                    wdata.values,
                    levels=wlev,
                    colors="0.15",
                    linewidths=0.75,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )
                ax.clabel(cs, inline=True, fontsize=7, fmt="%g")
            
            if barra_wind_vectors and field["key"] == "mslp":
                add_barra_wind_vectors(
                    ax,
                    barra_vector_data[region["key"]],
                    vector_step=vector_step,
                    vector_scale=vector_scale,
                    add_key=(row_i == 0),
                )

            add_region_marker(ax, region["marker_lon"], region["marker_lat"])
            add_panel_label(ax, f"({letters[row_i * 3 + col_i]})")

            if row_i == 0:
                ax.set_title(field["title"], fontsize=18, fontweight="bold", pad=7)

            if col_i == 0:
                # rotation_mode="anchor" aligns the text before rotating it,
                # rather than aligning the bbox of the already-rotated text.
                # Without it, va="center" on multi-line rotated text sits high.
                # With it, ha acts along the text, i.e. vertically once rotated,
                # so ha="center" is what centres the label on the row and
                # va="bottom" sets how far it stands off from the panel edge.
                ax.text(
                    -0.20,
                    0.5,
                    region["row_label"],
                    transform=ax.transAxes,
                    fontsize=15,
                    fontweight="bold",
                    rotation=90,
                    rotation_mode="anchor",
                    ha="center",
                    va="bottom",
                    multialignment="center",
                )

            ndays = n_days_text(ens, field["stem"], region["season"])
            if ndays:
                ax.text(
                    0.98,
                    0.965,
                    ndays,
                    transform=ax.transAxes,
                    fontsize=10,
                    ha="right",
                    va="top",
                    zorder=8,
                    bbox=dict(facecolor="white", alpha=0.62, edgecolor="none", pad=1.6),
                )

    # One clear shared colourbar per column.  This saves space and keeps labels readable.
    for col_i, field in enumerate(FIELDS):
        cb = fig.colorbar(
            contour_handles[col_i],
            ax=axes[:, col_i].ravel().tolist(),
            orientation="horizontal",
            fraction=0.052,
            pad=0.045,
            aspect=28,
        )
        cb.set_label(field[f"units_{mode}"], fontsize=13, labelpad=4)
        cb.ax.tick_params(labelsize=11, length=3)

    n_members_set = sorted({n for _, n in region_data.values()})
    n_members_text = str(n_members_set[0]) if len(n_members_set) == 1 else "/".join(map(str, n_members_set))
    fig.suptitle(
        "Historical 95th percentile LCI synoptic composites",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )

    mode_text = (
        "event anomalies relative to seasonal climatology"
        if mode == "anomaly" else "absolute event composites"
    )
    caption = (
        f"Historical (1990-2014) 95th percentile LCI synoptic composites, {n_members_text}-member "
        f"NARCliM2 ensemble mean, shown as {mode_text}. Columns show MSLP, temperature "
        f"and precipitation; rows show Cobar-Lachlan and Yilgarn-Coolgardie (May-Oct) "
        f"and Winton (Nov-Mar, April excluded). Open circles mark the region centroid; "
        f"n values give the number of composited event-days."
    )
    if wind_contours:
        wlev = WIND_ANOM_LEVELS if mode == "anomaly" else WIND_ABS_LEVELS
        wind_kind = "wind speed anomaly" if mode == "anomaly" else "wind speed"
        if wind_observed:
            wvals = np.concatenate(wind_observed)
            wmin, wmax = float(wvals.min()), float(wvals.max())
            # Only the levels that actually fall inside the data range get
            # drawn, so report those rather than the full requested array.
            drawn = [lv for lv in wlev if wmin <= lv <= wmax]
            if drawn:
                caption += (
                    f" Grey contours on the MSLP column show {wind_kind} at "
                    f"{wlev[1] - wlev[0]:g} m s$^{{-1}}$ intervals, spanning "
                    f"{min(drawn):g} to {max(drawn):g} m s$^{{-1}}$ "
                    f"(data range {wmin:+.2f} to {wmax:+.2f} m s$^{{-1}}$)."
                )
        else:
            caption += (
                f" Grey contours on the MSLP column show {wind_kind} at "
                f"{wlev[1] - wlev[0]:g} m s$^{{-1}}$ intervals."
            )
    if barra_wind_vectors:
        caption += (
            " Black arrows show observed BARRA-R2 2 m wind vectors composited over the "
            "same regional 95th percentile LCI event days (reference vector: 5 m s$^{-1}$, top row)."
        )
    print("\n[Suggested figure caption]")
    print(caption)

    if wind_contours and wind_observed:
        wvals = np.concatenate(wind_observed)
        wlev = WIND_ANOM_LEVELS if mode == "anomaly" else WIND_ABS_LEVELS
        wmin, wmax = float(wvals.min()), float(wvals.max())
        drawn = [lv for lv in wlev if wmin <= lv <= wmax]
        unused = [lv for lv in wlev if not (wmin <= lv <= wmax)]
        print("\n[Wind contour range check]")
        print(f"  data min/max = {wmin:+.2f} / {wmax:+.2f} m s-1   "
              f"p1/p99 = {np.percentile(wvals, 1):+.2f} / {np.percentile(wvals, 99):+.2f}")
        print(f"  requested levels : {', '.join(f'{lv:g}' for lv in wlev)}")
        print(f"  actually drawn   : {', '.join(f'{lv:g}' for lv in drawn) if drawn else '(none)'}")
        if unused:
            print(f"  never drawn      : {', '.join(f'{lv:g}' for lv in unused)}")

    outpath = Path(figdir) / outfile
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for ens, _ in region_data.values():
        ens.close()
    for ds in barra_vector_data.values():
        ds.close()

    return outpath


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make one compact 3x3 historical focal-season synoptic composite figure."
    )
    parser.add_argument("--composite-dir", default=DEFAULT_COMPOSITE_DIR)
    parser.add_argument("--figdir", default=DEFAULT_FIGDIR)
    parser.add_argument("--mode", choices=["anomaly", "absolute"], default="anomaly")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--outfile",
        default="synoptic_composites_historical_focal_3x3.png",
        help="Output filename, written inside --figdir.",
    )
    parser.add_argument(
        "--wind-contours",
        action="store_true",
        help="Overlay NARCliM2 sfcWind contours on the MSLP column.",
    )
    parser.add_argument(
        "--barra-wind-vectors",
        action="store_true",
        help=(
            "Overlay BARRA-R2 observed historical 2 m wind vectors on the MSLP "
            "column as circulation context. These are not NARCliM2 vectors."
        ),
    )
    parser.add_argument("--barra-lci-dir", default=DEFAULT_BARRA_LCI_DIR)
    parser.add_argument("--barra-wind-cache-dir", default=DEFAULT_BARRA_WIND_CACHE_DIR)
    parser.add_argument(
        "--overwrite-barra-wind-cache",
        action="store_true",
        help="Rebuild cached BARRA-R2 wind-vector composites.",
    )
    parser.add_argument(
        "--vector-step",
        type=int,
        default=18,
        help="Subsample step for BARRA wind arrows. Larger = fewer arrows.",
    )
    parser.add_argument(
        "--vector-scale",
        type=float,
        default=75.0,
        help="Matplotlib quiver scale for BARRA wind arrows. Larger = shorter arrows.",
    )
    args = parser.parse_args()

    outpath = plot_compact(
        composite_dir=args.composite_dir,
        figdir=args.figdir,
        mode=args.mode,
        dpi=args.dpi,
        outfile=args.outfile,
        wind_contours=args.wind_contours,
        barra_wind_vectors=args.barra_wind_vectors,
        barra_lci_dir=args.barra_lci_dir,
        barra_wind_cache_dir=args.barra_wind_cache_dir,
        overwrite_barra_wind_cache=args.overwrite_barra_wind_cache,
        vector_step=args.vector_step,
        vector_scale=args.vector_scale,
    )
    print(f"[OK] wrote {outpath}")


if __name__ == "__main__":
    main()