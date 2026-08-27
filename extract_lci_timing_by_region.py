#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_lci_timing_by_region.py

Pulls chill-season start/end date tables for three named LGA-based regions
(Winton, Yilgarn-Coolgardie, Cobar-Lachlan) out of the gridded outputs
produced by:

    barra_lci_event_timing_metrics.py   -> barra_lci_event_timing_climatology_<start>_<end>.nc
    narclim2_lci_event_timing_maps.py   -> narclim2_lci_event_timing_historical_1990-2014.nc
                                            narclim2_lci_event_timing_<scenario>_<period>.nc

Neither of those scripts saves point/region values anywhere -- they only write
continental grids and maps. This script re-opens the grids they already wrote
and area-averages each region's grid cells to get one first/last/span value
per region, season, threshold (and per scenario/period for NARCliM2).

Region boundaries use the LGA 2025 shapefile, with Yilgarn-Coolgardie and
Cobar-Lachlan each dissolved from two LGAs and Winton taken as a single
LGA. BARRA_CLIM_PATH and NARCLIM_OUTDIR below must point at the output
filenames written by the two upstream scripts named above.

Output
------
CSV: lci_chill_season_timing_by_region_mid_century.csv
Columns:
    dataset, scenario, period, region, season, threshold,
    first_day, first_date, first_day_std, last_day, last_date, last_day_std,
    span_days, event_probability

first_day_std/last_day_std are the region-averaged ensemble std (spread
across the 8 NARCliM2 members, in days) for the corresponding day metric.
NaN for BARRA-R2 (single observational product, no ensemble spread).

Usage
-----
module use /g/data/xp65/public/modules
module load conda/analysis3

python extract_lci_timing_by_region.py
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

try:
    from shapely.vectorized import contains as shp_contains
    HAVE_SHAPELY_VECTORIZED = True
except ImportError:
    HAVE_SHAPELY_VECTORIZED = False


# ============================================================
# REGION SPECS
# ============================================================
_LGA_SHP = (
    "/g/data/dx2/rt9243/Datasets/"
    "LGA_2025_AUST_GDA2020/LGA_2025_AUST_GDA2020.shp"
)
_REGION_SPECS = [
    {
        "name":          "Winton",
        "shp_path":      _LGA_SHP,
        "region_col":    "LGA_NAME25",
        "region_values": ["Winton"],
        "dissolve_all":  False,
    },
    {
        "name":          "Yilgarn-Coolgardie",
        "shp_path":      _LGA_SHP,
        "region_col":    "LGA_NAME25",
        "region_values": ["Yilgarn", "Coolgardie"],
        "dissolve_all":  True,
    },
    {
        "name":          "Cobar-Lachlan",
        "shp_path":      _LGA_SHP,
        "region_col":    "LGA_NAME25",
        "region_values": ["Cobar", "Lachlan"],
        "dissolve_all":  True,
    },
]

BARRA_CLIM_PATH = (
    "/scratch/dx2/rt9243/chill_projections/lci_event_timing_maps/"
    "barra_lci_event_timing_climatology_1980_2024.nc"
)

NARCLIM_OUTDIR = Path(
    "/scratch/dx2/rt9243/chill_projections/narclim2_lci_event_timing_maps"
)
NARCLIM_HIST_PATH = NARCLIM_OUTDIR / "narclim2_lci_event_timing_historical_1990-2014.nc"
NARCLIM_FUTURE_PATHS = {
    ("ssp126", "2040-2060"): NARCLIM_OUTDIR / "narclim2_lci_event_timing_ssp126_2040-2060.nc",
    ("ssp126", "2080-2100"): NARCLIM_OUTDIR / "narclim2_lci_event_timing_ssp126_2080-2100.nc",
    ("ssp370", "2040-2060"): NARCLIM_OUTDIR / "narclim2_lci_event_timing_ssp370_2040-2060.nc",
    ("ssp370", "2080-2100"): NARCLIM_OUTDIR / "narclim2_lci_event_timing_ssp370_2080-2100.nc",
}

SEASONS = ["wet", "dry"]
THRESHOLDS = [1000, 1100, 1200]

OUT_CSV = "lci_chill_season_timing_by_region_mid_century.csv"


# ============================================================
# HELPERS
# ============================================================

def day_of_season_to_date(day: float, season: str) -> str:
    """Convert a 1-based day-of-season number back to a calendar date label."""
    if pd.isna(day):
        return ""
    if season == "dry":
        start = datetime.date(2001, 5, 1)   # 1 May, non-leap reference year
    elif season == "wet":
        start = datetime.date(2000, 11, 1)  # 1 Nov, spans into next year
    else:
        raise ValueError(season)
    d = start + datetime.timedelta(days=int(round(day)) - 1)
    return d.strftime("%d %b")


def region_grid_mask(lat: np.ndarray, lon: np.ndarray, geometry) -> np.ndarray:
    """
    Boolean mask, shape (len(lat), len(lon)), True where the grid-cell
    centre falls inside `geometry`. Uses shapely.vectorized if available
    (fast); falls back to a plain loop otherwise.
    """
    lon2d, lat2d = np.meshgrid(lon, lat)

    if HAVE_SHAPELY_VECTORIZED:
        return shp_contains(geometry, lon2d, lat2d)

    from shapely.geometry import Point
    mask = np.zeros(lon2d.shape, dtype=bool)
    for i in range(lon2d.shape[0]):
        for j in range(lon2d.shape[1]):
            mask[i, j] = geometry.contains(Point(lon2d[i, j], lat2d[i, j]))
    return mask


def region_average(da: xr.DataArray, mask: np.ndarray) -> float:
    """Simple (unweighted) mean over grid cells inside the region mask."""
    mask_da = xr.DataArray(mask, dims=("lat", "lon"), coords={"lat": da["lat"], "lon": da["lon"]})
    vals = da.where(mask_da)
    out = vals.mean(dim=("lat", "lon"), skipna=True).item()
    return out


def load_region_geometries() -> dict[str, "object"]:
    """Resolve each region spec to a single shapely geometry (union'd if needed).
    This only depends on the shapefile, not on any particular model grid, so
    it's computed once and reused across BARRA and NARCliM2.
    """
    _shp_cache: dict[str, gpd.GeoDataFrame] = {}
    geometries = {}

    for spec in _REGION_SPECS:
        shp_path = spec["shp_path"]
        if shp_path not in _shp_cache:
            gdf = gpd.read_file(shp_path)
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            _shp_cache[shp_path] = gdf
        gdf = _shp_cache[shp_path]

        region_col = spec["region_col"]
        match = gdf[gdf[region_col].isin(spec["region_values"])]

        if match.empty:
            available = sorted(gdf[region_col].unique().tolist())
            close = [a for a in available for v in spec["region_values"] if v.lower() in a.lower()]
            raise ValueError(
                f"None of {spec['region_values']} found in {region_col} "
                f"({shp_path}). Closest matches: {close}"
            )

        if spec["dissolve_all"]:
            geometry = match.geometry.union_all()
        else:
            geometry = match.geometry.iloc[0]
            if len(match) > 1:
                print(f"[WARN] {spec['name']}: {spec['region_values']} matched "
                      f"{len(match)} features but dissolve_all=False; using the first.")

        geometries[spec["name"]] = geometry

    return geometries


def build_masks_for_grid(
    geometries: dict[str, "object"], sample_ds: xr.Dataset, grid_label: str
) -> dict[str, np.ndarray]:
    """Rasterise each region geometry onto a specific dataset's lat/lon grid.
    Must be called separately per dataset -- BARRA and NARCliM2 are on
    different grids (different resolution/extent), so a mask built for one
    cannot be reused on the other.
    """
    lat = sample_ds["lat"].values
    lon = sample_ds["lon"].values

    masks = {}
    for name, geometry in geometries.items():
        mask = region_grid_mask(lat, lon, geometry)
        if not mask.any():
            print(f"[WARN] no grid cells fell inside {name} on the {grid_label} grid; "
                  f"the polygon may be smaller than the grid spacing there. Consider "
                  f"using the nearest single grid point instead for this region.")
        masks[name] = mask

    return masks


# ============================================================
# EXTRACTION
# ============================================================

def extract_from_dataset(
    ds: xr.Dataset,
    masks: dict[str, np.ndarray],
    dataset_label: str,
    scenario: str,
    period: str,
    first_var: str,
    last_var: str,
    span_var: str,
    prob_var: str,
    first_std_var: str | None = None,
    last_std_var: str | None = None,
) -> list[dict]:
    rows = []
    for region, mask in masks.items():
        for season in SEASONS:
            if season not in ds["season"].values:
                continue
            for threshold in THRESHOLDS:
                if float(threshold) not in ds["threshold"].values:
                    continue

                sub = ds.sel(season=season, threshold=float(threshold))

                first_day = region_average(sub[first_var], mask)
                last_day = region_average(sub[last_var], mask)
                span_days = region_average(sub[span_var], mask)
                prob = region_average(sub[prob_var], mask)

                # Region-averaged ensemble std (spread across the 8 members,
                # in days) for the start/end day metrics, if available.
                # Only present for NARCliM2 (ensemble); BARRA is a single
                # observational product with no ensemble spread, so these
                # stay NaN there.
                first_day_std = (
                    region_average(sub[first_std_var], mask) if first_std_var and first_std_var in sub else np.nan
                )
                last_day_std = (
                    region_average(sub[last_std_var], mask) if last_std_var and last_std_var in sub else np.nan
                )

                rows.append({
                    "dataset": dataset_label,
                    "scenario": scenario,
                    "period": period,
                    "region": region,
                    "season": season,
                    "threshold": threshold,
                    "first_day": round(first_day, 1) if not np.isnan(first_day) else np.nan,
                    "first_date": day_of_season_to_date(first_day, season),
                    "first_day_std": round(first_day_std, 1) if not np.isnan(first_day_std) else np.nan,
                    "last_day": round(last_day, 1) if not np.isnan(last_day) else np.nan,
                    "last_date": day_of_season_to_date(last_day, season),
                    "last_day_std": round(last_day_std, 1) if not np.isnan(last_day_std) else np.nan,
                    "span_days": round(span_days, 1) if not np.isnan(span_days) else np.nan,
                    "event_probability": round(prob, 2) if not np.isnan(prob) else np.nan,
                })
    return rows


def main() -> None:
    all_rows = []
    geometries = load_region_geometries()

    # ---------------- BARRA-R2 ----------------
    print(f"[INFO] opening BARRA climatology: {BARRA_CLIM_PATH}")
    barra = xr.open_dataset(BARRA_CLIM_PATH, decode_timedelta=False)
    masks_barra = build_masks_for_grid(geometries, barra, grid_label="BARRA-R2")

    all_rows += extract_from_dataset(
        barra, masks_barra,
        dataset_label="BARRA-R2",
        scenario="observed",
        period="1980-2024",
        first_var="first_day_mean",
        last_var="last_day_mean",
        span_var="span_days_mean",
        prob_var="event_probability",
    )

    # ---------------- NARCliM2 historical ----------------
    print(f"[INFO] opening NARCliM2 historical: {NARCLIM_HIST_PATH}")
    narclim_hist = xr.open_dataset(NARCLIM_HIST_PATH, decode_timedelta=False)
    masks_narclim = build_masks_for_grid(geometries, narclim_hist, grid_label="NARCliM2")

    all_rows += extract_from_dataset(
        narclim_hist, masks_narclim,
        dataset_label="NARCliM2",
        scenario="historical",
        period="1990-2014",
        first_var="first_day_ensmean",
        last_var="last_day_ensmean",
        span_var="span_days_ensmean",
        prob_var="event_probability_ensmean",
        first_std_var="first_day_ensstd",
        last_std_var="last_day_ensstd",
    )

    # ---------------- NARCliM2 future ----------------
    # Reuses masks_narclim -- all NARCliM2 outputs (historical + future) share
    # the same model grid. If that's ever not true, build a fresh mask here too.
    for (scenario, period), path in NARCLIM_FUTURE_PATHS.items():
        print(f"[INFO] opening NARCliM2 future: {path}")
        narclim_fut = xr.open_dataset(path, decode_timedelta=False)

        all_rows += extract_from_dataset(
            narclim_fut, masks_narclim,
            dataset_label="NARCliM2",
            scenario=scenario,
            period=period,
            first_var="first_day_ensmean",
            last_var="last_day_ensmean",
            span_var="span_days_ensmean",
            prob_var="event_probability_ensmean",
            first_std_var="first_day_ensstd",
            last_std_var="last_day_ensstd",
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()