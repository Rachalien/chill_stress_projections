#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
barra_regional_lci_metrics.py

Compute regional annual and seasonal LCI metrics from BARRA-R2 daily LCI files.
Outputs are designed for direct overlay on CMIP6 ensemble time-series plots
produced by toe_lci_metrics_aus10i.py.

Season definitions
------------------
Annual  : full calendar year Y
Dry     : May–October, calendar year Y          (labelled by year Y)
Wet     : November (year Y-1) through April (year Y)  (labelled by ending year Y)
          e.g. "wet season 1990" = Nov 1989 + Dec 1989 + Jan–Apr 1990

Metrics
-------
lci_max  : area-weighted regional mean of the per-grid-cell seasonal maximum
           of daily LCI  [kJ m-2 hr-1]
freq     : area-weighted regional mean of the per-grid-cell count of days with
           LCI >= threshold  [days / season]
           thresholds: 1000, 1100, 1200 kJ m-2 hr-1

These definitions match how freq and lci_max are constructed in the CMIP6
per-model pipeline (lci_metrics_per_model_then_ensemble_aus10i.py), so values
are directly comparable on the time-series plots.

Years processed: 1979–2024
  2025 omitted: file only runs to 2025-11-30 (incomplete annual and dry seasons).
  First complete wet season available is wet-1980 (Nov 1979 + Jan–Apr 1980).

Input files
-----------
/scratch/dx2/rt9243/barra_daily_lci/barra_lci_daily_aus_land_{year}.nc
Variables used:
  lci_daily        (time, lat, lon)  float32  kJ m-2 hr-1
  lci_ge_1000      (time, lat, lon)  byte     1 if daily LCI >= 1000, else 0
  lci_ge_1100      (time, lat, lon)  byte     1 if daily LCI >= 1100, else 0
  lci_ge_1200      (time, lat, lon)  byte     1 if daily LCI >= 1200, else 0

Output files
------------
One CSV per region at:
  {OUTDIR}/barra_lci_metrics_{safe_region_name}.csv

Columns:
  year        int     calendar label (see season definitions above)
  season      str     "annual" | "wet" | "dry"
  metric      str     "lci_max" | "freq"
  threshold   float   threshold value, or NaN for lci_max rows
  value       float   metric value
  n_days      int     number of days used (data quality check)

Usage (interactive on Gadi ARE or login node)
---------------------------------------------
  module load conda/analysis3
  python /g/data/dx2/rt9243/Python_Code/barra_regional_lci_metrics.py

PBS (recommended for the full 46-year run)
------------------------------------------
  #PBS -l walltime=02:00:00,mem=32GB,ncpus=1
  #PBS -l storage=gdata/dx2+gdata/ob53+scratch/dx2
  #PBS -j oe
  module load conda/analysis3
  python /g/data/dx2/rt9243/Python_Code/barra_regional_lci_metrics.py
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  – registers .rio accessor on xarray objects
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)

SCRIPT_NAME = "barra_regional_lci_metrics.py"

# ── Paths ─────────────────────────────────────────────────────────────────────

BARRA_DIR = "/scratch/dx2/rt9243/barra_daily_lci"
OUTDIR    = "/scratch/dx2/rt9243/barra_regional_metrics"

# ── Year range ────────────────────────────────────────────────────────────────

YEARS_ALL = list(range(1979, 2025))   # 1979–2024 inclusive; 2025 excluded (partial year)

# ── Metrics / thresholds ──────────────────────────────────────────────────────

THRESHOLDS = [1000, 1100, 1200]

# ── CRS ───────────────────────────────────────────────────────────────────────

DATA_CRS = "EPSG:4326"   # WGS84, matching BARRA lat/lon and reprojected shapefiles

# ── Season month sets ─────────────────────────────────────────────────────────

WET_MONTHS = [11, 12, 1, 2, 3, 4]    # Nov–Apr (cross-year)
DRY_MONTHS = [5, 6, 7, 8, 9, 10]     # May–Oct

# ── Region definitions ────────────────────────────────────────────────────────
# Region definitions are shared with the projection-side scripts so that
# spatial clipping is consistent between observations and models.
_LGA_SHP = (
    "/g/data/dx2/rt9243/Datasets/"
    "LGA_2025_AUST_GDA2020/LGA_2025_AUST_GDA2020.shp"
)
REGIONS = [
    {
        "name": "Gulf Rivers",
        "shp_path": (
            "/g/data/dx2/rt9243/Datasets/"
            "QSC_Extracted_Data_20260319_083732973385-11900/"
            "QSC_Extracted_Data_20260319_083732973385-11900/"
            "Regional_planning_interests___Strategic_environmental_area.shp"
        ),
        "region_col": "region",
        "region_value": "Gulf Rivers",
        "dissolve_all": False,
    },
    {
        "name": "Lower Murray / Riverina",
        "shp_path": (
            "/g/data/dx2/rt9243/Datasets/"
            "basin-plan-data-sets-april-2020/"
            "Basin Plan Data Sets as at April 2020/"
            "Surface Water Water Resource Plan Areas/"
            "Surface Water Water Resource Plan Areas.shp"
        ),
        "region_col": "SWWRPACODE",
        "region_value": "SW8",
        "dissolve_all": False,
    },
    {
        "name": "Limestone Coast",
        "shp_path": (
            "/g/data/dx2/rt9243/Datasets/"
            "SAGovtRegions_shp/SAGovtRegions_GDA2020.shp"
        ),
        "region_col": "region",
        "region_value": "Limestone Coast",
        "dissolve_all": False,
    },
    {
        "name": "Barkly",
        "shp_path": (
            "/g/data/dx2/rt9243/Datasets/"
            "LandSystems_barky_1M/LandSystems_barky_1M/Data/ESRI/barky_1m.shp"
        ),
        "region_col": None,
        "region_value": None,
        "dissolve_all": True,
    },
    {
        "name": "Kimberley",
        "shp_path": (
            "/g/data/dx2/rt9243/Datasets/"
            "Regional_Development_Commissions_Regions_DPIRD_020_WA_GDA2020_Public_Shapefile/"
            "Regional_Development_Commissions_Regions_DPIRD_020.shp"
        ),
        "region_col": "region",
        "region_value": "Kimberley",
        "dissolve_all": False,
    },
    {
        "name": "Barcaldine LGA",
        "shp_path": (
            "/g/data/dx2/rt9243/Datasets/"
            "QSC_Extracted_Data_20260415_104816853083-2088/"
            "Local_Government_Areas.shp"
        ),
        "region_col": "lga",
        "region_value": "Barcaldine Regional",
        "dissolve_all": False,
    },
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegionSpec:
    name: str
    shp_path: str
    region_col: Optional[str] = None
    region_values: Optional[List[str]] = None
    dissolve_all: bool = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_name(region_name: str) -> str:
    """Convert region name to a filesystem-safe string."""
    return (
        region_name
        .lower()
        .replace(" / ", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Region loading
# ─────────────────────────────────────────────────────────────────────────────

def load_region_geometry(spec: RegionSpec, target_crs: str = DATA_CRS) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(spec.shp_path)
    if gdf.crs is None:
        raise ValueError(f"Shapefile has no CRS: {spec.shp_path}")

    if spec.region_col is None or spec.region_values is None:
        # dissolve_all with no filter: select entire shapefile
        region_gdf = gdf.copy()
    else:
        region_gdf = gdf[gdf[spec.region_col].isin(spec.region_values)].copy()
    if len(region_gdf) == 0:
        raise ValueError(
            f"No rows found for {spec.name} "
            f"({spec.region_col} in {spec.region_values})"
        )
    if spec.dissolve_all:
        region_gdf = region_gdf.dissolve().explode(index_parts=False).dissolve()

    return region_gdf.to_crs(target_crs)


def load_all_regions(region_dicts) -> Dict[str, gpd.GeoDataFrame]:
    out: Dict[str, gpd.GeoDataFrame] = {}
    for rd in region_dicts:
        # Normalise legacy dicts that use singular region_value
        rd = dict(rd)
        if "region_value" in rd and "region_values" not in rd:
            val = rd.pop("region_value")
            rd["region_values"] = [val] if val is not None else None
        elif "region_value" in rd:
            rd.pop("region_value")
        spec = RegionSpec(**rd)
        try:
            out[spec.name] = load_region_geometry(spec)
            print(f"  [OK]   {spec.name}")
        except Exception as exc:
            print(f"  [WARN] {spec.name}: {exc}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BARRA file I/O
# ─────────────────────────────────────────────────────────────────────────────

BARRA_VARS = ["lci_daily", "lci_ge_1000", "lci_ge_1100", "lci_ge_1200"]


def barra_path(year: int) -> str:
    return os.path.join(BARRA_DIR, f"barra_lci_daily_aus_land_{year}.nc")


def open_barra_year(year: int) -> Optional[xr.Dataset]:
    """
    Open one year's BARRA file, returning only the LCI variables we need.
    Returns None if the file is missing.
    """
    path = barra_path(year)
    if not os.path.exists(path):
        print(f"    [WARN] missing: {path}")
        return None
    ds = xr.open_dataset(path, decode_times=True)
    keep = [v for v in BARRA_VARS if v in ds]
    return ds[keep]


# ─────────────────────────────────────────────────────────────────────────────
# Spatial clipping and area-weighting
# ─────────────────────────────────────────────────────────────────────────────

def clip_to_region(
    ds: xr.Dataset,
    region_gdf: gpd.GeoDataFrame,
    region_name: str,
) -> Optional[xr.Dataset]:
    """
    Clip the full dataset to region_gdf in one operation (more efficient than
    clipping each variable separately).  Returns None on failure or empty clip.

    all_touched=True ensures small regions (e.g. Barcaldine LGA) capture at
    least the grid cells whose centres may fall just outside the polygon edge.
    """
    try:
        ds_rio = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        ds_rio = ds_rio.rio.write_crs(DATA_CRS, inplace=False)
        clipped = ds_rio.rio.clip(
            region_gdf.geometry,
            region_gdf.crs,
            drop=True,
            all_touched=True,
        )
        # Sanity check: confirm at least one non-null cell survives
        if clipped["lci_daily"].isnull().all():
            print(f"    [WARN] clip produced all-NaN for {region_name}; check shapefile CRS")
            return None
        return clipped
    except Exception as exc:
        print(f"    [WARN] clip failed for {region_name}: {exc}")
        return None


def area_weighted_mean_timeseries(
    da: xr.DataArray,
) -> xr.DataArray:
    """
    Compute the latitude-weighted spatial mean of a (time, lat, lon) DataArray,
    returning a (time,) DataArray.  Assumes da has already been clipped.
    """
    lat_weights = xr.DataArray(
        np.cos(np.deg2rad(da["lat"])),
        coords={"lat": da["lat"]},
        dims=("lat",),
    )
    return da.weighted(lat_weights).mean(dim=("lat", "lon"), skipna=True)


# ─────────────────────────────────────────────────────────────────────────────
# Seasonal metric computation
# ─────────────────────────────────────────────────────────────────────────────

def assign_wet_season_year(time_index: pd.DatetimeIndex) -> pd.Series:
    """
    Assign a wet-season year label to each date.

    Convention (matches Australian climate practice):
      Nov and Dec of calendar year Y → wet season Y+1
      Jan through Apr of calendar year Y → wet season Y
      May through Oct (dry season months) → NaN (excluded)
    """
    months = time_index.month
    years  = time_index.year
    wet_year = pd.Series(np.nan, index=time_index, dtype="float64")
    wet_year[np.isin(months, [1, 2, 3, 4])] = years[np.isin(months, [1, 2, 3, 4])]
    wet_year[np.isin(months, [11, 12])]      = years[np.isin(months, [11, 12])] + 1
    return wet_year


def compute_metrics_from_daily(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a daily DataFrame into annual and seasonal LCI metrics.

    Parameters
    ----------
    daily_df : pd.DataFrame
        DatetimeIndex.  Columns: lci_daily, lci_ge_1000, lci_ge_1100, lci_ge_1200.
        All values are area-weighted regional means (float).

        For lci_ge_* columns: values represent the area-weighted fraction of grid
        cells exceeding the threshold on that day (0–1).  Summing over a period
        gives the area-weighted mean exceedance day-count, directly comparable to
        CMIP6 freq (which averages per-grid-cell annual counts spatially).
        This equivalence follows from linearity of expectation:
            sum_t( E_x[I_{x,t}] ) = E_x[ sum_t(I_{x,t}) ]

    Returns
    -------
    pd.DataFrame with columns: year, season, metric, threshold, value, n_days
    """
    rows: List[dict] = []
    idx = daily_df.index  # DatetimeIndex

    # ── Annual ───────────────────────────────────────────────────────────────
    for year, grp in daily_df.groupby(idx.year):
        n = len(grp)
        rows.append({
            "year": int(year), "season": "annual",
            "metric": "lci_max", "threshold": np.nan,
            "value": float(grp["lci_daily"].max()), "n_days": n,
        })
        for thr in THRESHOLDS:
            col = f"lci_ge_{thr}"
            if col in grp.columns:
                rows.append({
                    "year": int(year), "season": "annual",
                    "metric": "freq", "threshold": float(thr),
                    "value": float(grp[col].sum()), "n_days": n,
                })

    # ── Dry season (May–Oct) ─────────────────────────────────────────────────
    dry_mask = np.isin(idx.month, DRY_MONTHS)
    dry_df   = daily_df.loc[dry_mask]
    for year, grp in dry_df.groupby(dry_df.index.year):
        n = len(grp)
        rows.append({
            "year": int(year), "season": "dry",
            "metric": "lci_max", "threshold": np.nan,
            "value": float(grp["lci_daily"].max()), "n_days": n,
        })
        for thr in THRESHOLDS:
            col = f"lci_ge_{thr}"
            if col in grp.columns:
                rows.append({
                    "year": int(year), "season": "dry",
                    "metric": "freq", "threshold": float(thr),
                    "value": float(grp[col].sum()), "n_days": n,
                })

    # ── Wet season (Nov–Apr, cross-year) ─────────────────────────────────────
    wet_mask = np.isin(idx.month, WET_MONTHS)
    wet_df   = daily_df.loc[wet_mask].copy()
    wet_df["wet_year"] = assign_wet_season_year(wet_df.index)
    wet_df = wet_df.dropna(subset=["wet_year"])
    wet_df["wet_year"] = wet_df["wet_year"].astype(int)

    for wet_year, grp in wet_df.groupby("wet_year"):
        n = len(grp)
        # Require at least 5 months of data (150 days) for a valid wet season.
        # The first wet season (wet-1980) has Nov+Dec 1979 + Jan–Apr 1980 ≈ 181 days.
        # The last wet season (wet-2024) has Nov+Dec 2023 + Jan–Apr 2024 ≈ 181 days.
        if n < 150:
            print(f"      [WARN] wet-{wet_year}: only {n} days; skipping (incomplete season)")
            continue
        rows.append({
            "year": int(wet_year), "season": "wet",
            "metric": "lci_max", "threshold": np.nan,
            "value": float(grp["lci_daily"].max()), "n_days": n,
        })
        for thr in THRESHOLDS:
            col = f"lci_ge_{thr}"
            if col in grp.columns:
                rows.append({
                    "year": int(wet_year), "season": "wet",
                    "metric": "freq", "threshold": float(thr),
                    "value": float(grp[col].sum()), "n_days": n,
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-region processing
# ─────────────────────────────────────────────────────────────────────────────

def process_region(region_name: str, region_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    For one region: load all BARRA years, clip to region, area-weight,
    collect as a daily time series, then aggregate to annual/seasonal metrics.

    Approach
    --------
    We clip + area-average each year's data immediately after loading so we
    never hold the full spatial grid for all years in memory simultaneously.
    The resulting daily regional series is a 1-D float array, and very small.
    """
    print(f"\n  Processing: {region_name}")
    daily_chunks: List[pd.DataFrame] = []

    for year in YEARS_ALL:
        ds = open_barra_year(year)
        if ds is None:
            continue

        try:
            # ── Clip whole dataset to region in one operation ──────────────
            clipped = clip_to_region(ds, region_gdf, region_name)
            if clipped is None:
                ds.close()
                continue

            # ── Area-weighted daily mean for each variable ─────────────────
            chunk_cols: dict = {}

            # lci_daily: float
            lci_mean = area_weighted_mean_timeseries(
                clipped["lci_daily"].astype("float32")
            )
            chunk_cols["lci_daily"] = lci_mean.values.astype(float)
            time_vals = pd.to_datetime(lci_mean["time"].values)

            # lci_ge_*: byte (0/1) → float mean (fraction of cells exceeding)
            for thr in THRESHOLDS:
                col = f"lci_ge_{thr}"
                if col in clipped:
                    ge_mean = area_weighted_mean_timeseries(
                        clipped[col].astype("float32")
                    )
                    chunk_cols[col] = ge_mean.values.astype(float)

        except Exception as exc:
            print(f"    [WARN] year {year} failed: {exc}")
            ds.close()
            continue
        finally:
            ds.close()

        chunk_df = pd.DataFrame(chunk_cols, index=time_vals)
        daily_chunks.append(chunk_df)
        print(f"    {year}: {len(chunk_df)} days, "
              f"lci_daily mean={chunk_cols['lci_daily'].mean():.1f}")

    if not daily_chunks:
        print(f"  [WARN] No data collected for {region_name}; check shapefile path and region value")
        return pd.DataFrame()

    daily_df = pd.concat(daily_chunks).sort_index()
    print(f"  Total daily records: {len(daily_df)} "
          f"({daily_df.index[0].date()} to {daily_df.index[-1].date()})")

    metrics_df = compute_metrics_from_daily(daily_df)
    print(f"  Metric rows generated: {len(metrics_df)}")
    return metrics_df


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dir(OUTDIR)
    print("=" * 60)
    print(f"  {SCRIPT_NAME}")
    print(f"  BARRA-R2 regional LCI metrics  |  years {YEARS_ALL[0]}–{YEARS_ALL[-1]}")
    print(f"  Output: {OUTDIR}")
    print("=" * 60)

    print("\nLoading region geometries ...")
    regions = load_all_regions(REGIONS)
    if not regions:
        raise RuntimeError("No regions loaded; cannot proceed.")

    summary_rows: List[dict] = []

    for region_name, region_gdf in regions.items():
        result_df = process_region(region_name, region_gdf)

        if result_df.empty:
            print(f"  [SKIP] {region_name}: no output produced")
            continue

        # Sort for readability
        result_df = (
            result_df
            .sort_values(["season", "metric", "threshold", "year"])
            .reset_index(drop=True)
        )

        # Add provenance column
        result_df.insert(0, "region", region_name)

        out_path = os.path.join(
            OUTDIR,
            f"barra_lci_metrics_{safe_name(region_name)}.csv",
        )
        result_df.to_csv(out_path, index=False)
        print(f"\n  [OK] wrote {out_path}  ({len(result_df)} rows)")

        # Accumulate for combined file
        summary_rows.append(result_df)

    # Also write one combined CSV, convenient for quick inspection
    if summary_rows:
        combined = pd.concat(summary_rows, ignore_index=True)
        combined_path = os.path.join(OUTDIR, "barra_lci_metrics_all_regions.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n  [OK] combined CSV: {combined_path}  ({len(combined)} rows total)")

    print("\nDone.")


if __name__ == "__main__":
    main()