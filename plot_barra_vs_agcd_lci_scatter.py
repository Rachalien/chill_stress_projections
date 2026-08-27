#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_barra_vs_agcd_lci_scatter.py

Scatter plots comparing BARRA-R2 and AGCD-derived LCI seasonal metrics
for the three study regions over the common period 1979-2018.

Produces two figures:
    barra_vs_agcd_lci_max_scatter.png  : lci_max (wet and dry seasons)
    barra_vs_agcd_freq_scatter.png     : freq at region-specific threshold

Layout: 2 rows (wet / dry) x 3 columns (Barkly / Barcaldine LGA / Limestone Coast).
Points are coloured by year to reveal any temporal drift in the relationship.
Each panel annotates Pearson r, RMSE, and mean bias (BARRA minus AGCD).

Inputs
------
BARRA_METRICS : barra_lci_metrics_1979_2025.nc
    Gridded (year, lat, lon); 2m-wind-corrected metrics.
AGCD_METRICS  : agcd_lci_metrics_regional_1979_2018.nc
    Pre-computed regional means from agcd_lci_regional.py.

Notes
-----
- BARRA regional means are extracted on-the-fly via rioxarray clip
  and cosine-latitude weighting.
- Wet years 1979 and 2019 are excluded: both are incomplete
  (1979 lacks Nov-Dec 1978; 2019 lacks Jan-Apr 2019 in AGCD).
- Region-specific thresholds for freq panels:
    Barkly: 1000 kJ m-2 hr-1
    Barcaldine LGA: 1100 kJ m-2 hr-1
    Limestone Coast: 1200 kJ m-2 hr-1
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray  # noqa: F401
import xarray as xr
from scipy import stats

# ============================================================
# CONFIG
# ============================================================

BARRA_METRICS_PATH = "/scratch/dx2/rt9243/barra_lci_metrics_1979_2025.nc"
AGCD_METRICS_PATH  = (
    "/scratch/dx2/rt9243/chill_projections/agcd_lci/"
    "agcd_lci_metrics_regional_1979_2018.nc"
)
OUTDIR = "/scratch/dx2/rt9243/chill_projections/agcd_lci/scatter_plots"

BARRA_CRS = "EPSG:4283"   # GDA94; consistent with compare_barra_vs_cmip6 script

COMPARE_START = 1979
COMPARE_END   = 2018

# Wet years excluded from scatter: both endpoints are incomplete seasons.
# 1979: only Jan-Apr 1979 available (no Nov-Dec 1978 in either dataset).
# 2019: only Nov-Dec 2018 available in AGCD (AWRA wind stops end of 2018).
WET_YEAR_EXCLUDE = {1979, 2019}

_LGA_SHP = (
    "/g/data/dx2/rt9243/Datasets/"
    "LGA_2025_AUST_GDA2020/LGA_2025_AUST_GDA2020.shp"
)

# Region-specific exceedance thresholds for freq panels (kJ m-2 hr-1)
REGION_THRESHOLDS = {
    "Winton":             1000,
    "Yilgarn-Coolgardie": 1200,
    "Cobar-Lachlan":      1100,
}

REGION_DISPLAY = {
    "Winton":             "Winton",
    "Yilgarn-Coolgardie": "Yilgarn-Coolgardie",
    "Cobar-Lachlan":      "Cobar-Lachlan",
}

REGIONS = [
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
REGION_NAMES = [r["name"] for r in REGIONS]

SEASON_LABEL = {
    "wet": "November\u2013April",
    "dry": "May\u2013October",
}

METRIC_AXIS_LABEL = {
    "lci_max": r"LCI$_\mathrm{max}$ (kJ m$^{-2}$ hr$^{-1}$)",
    "freq":    r"Frequency (days season$^{-1}$)",
}


# ============================================================
# REGION LOADING
# ============================================================

def load_region_geometry(spec: dict, target_crs: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(spec["shp_path"])
    if gdf.crs is None:
        raise ValueError(f"Shapefile has no CRS: {spec['shp_path']}")
    col, vals = spec["region_col"], spec["region_values"]
    gdf = gdf[gdf[col].isin(vals)].copy()
    if len(gdf) == 0:
        raise ValueError(f"No rows found: {col!r} in {vals!r}")
    if spec.get("dissolve_all"):
        gdf = gdf.dissolve().explode(index_parts=False).dissolve()
    return gdf.to_crs(target_crs)

def load_all_regions(region_list: list, target_crs: str) -> Dict[str, gpd.GeoDataFrame]:
    return {spec["name"]: load_region_geometry(spec, target_crs) for spec in region_list}


# ============================================================
# BARRA REGIONAL EXTRACTION
# ============================================================

def barra_clip_area_weighted_mean(
    da: xr.DataArray,
    region_gdf: gpd.GeoDataFrame,
    region_name: str,
) -> xr.DataArray:
    """
    Clip da (year, lat, lon) to region_gdf and return the cosine-latitude-
    weighted spatial mean, preserving the year dimension.
    """
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da = da.rio.write_crs(BARRA_CRS, inplace=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clipped = da.rio.clip(
            [region_gdf.geometry.union_all()],
            region_gdf.crs,
            drop=True,
        )

    if clipped.size == 0 or bool(clipped.isnull().all()):
        raise RuntimeError(f"Empty clip result for region '{region_name}'")

    lat_weights = xr.DataArray(
        np.cos(np.deg2rad(clipped["lat"])),
        coords={"lat": clipped["lat"]},
        dims=("lat",),
    )
    return clipped.weighted(lat_weights).mean(dim=("lat", "lon"), skipna=True)


def extract_barra_regional_series(
    ds: xr.Dataset,
    metric: str,
    season: str,
    region_name: str,
    region_gdf: gpd.GeoDataFrame,
    threshold: Optional[float] = None,
) -> pd.Series:
    """
    Return a pd.Series(float, index=year) for one BARRA metric/season/region.
    """
    var = f"{metric}_{season}"
    if var not in ds:
        raise KeyError(f"{var!r} not found in BARRA dataset")

    da = ds[var].astype("float32")

    if "threshold" in da.dims:
        if threshold is None:
            raise ValueError(f"threshold required for variable {var!r}")
        da = da.sel(threshold=float(threshold))

    regional = barra_clip_area_weighted_mean(da, region_gdf, region_name)
    return pd.Series(regional.values.astype(float), index=ds["year"].values)


# ============================================================
# AGCD REGIONAL LOADING
# ============================================================

def load_agcd_metrics(path: str) -> xr.Dataset:
    ds = xr.open_dataset(path, decode_timedelta=False)
    # Drop rioxarray CRS artefacts written by agcd_lci_regional.py
    drop = [v for v in ("crs", "spatial_ref") if v in ds]
    if drop:
        ds = ds.drop_vars(drop)
    return ds


def agcd_regional_series(
    ds: xr.Dataset,
    var: str,
    region_name: str,
    threshold: Optional[float] = None,
    exclude_years: Optional[set] = None,
) -> pd.Series:
    """
    Return a pd.Series(float, index=year) from the pre-computed AGCD regional NetCDF.
    """
    da = ds[var].sel(region=region_name)

    if "threshold" in da.dims:
        if threshold is None:
            raise ValueError(f"threshold required for variable {var!r}")
        da = da.sel(threshold=float(threshold))

    s = pd.Series(da.values.astype(float), index=ds["year"].values)

    if exclude_years:
        s = s.drop(labels=[y for y in exclude_years if y in s.index], errors="ignore")

    return s


# ============================================================
# STATISTICS
# ============================================================

def scatter_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Pearson r, RMSE, and mean bias (y - x) after dropping NaN pairs.
    Returns (nan, nan, nan) if fewer than 3 valid pairs.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    r, _ = stats.pearsonr(x, y)
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    bias = float(np.mean(y - x))
    return r, rmse, bias


# ============================================================
# FIGURE
# ============================================================

def make_scatter_figure(
    barra_data: Dict[Tuple[str, str], pd.Series],
    agcd_data:  Dict[Tuple[str, str], pd.Series],
    metric: str,
    all_years: np.ndarray,
    outpath: str,
    suptitle: str,
) -> None:
    """
    2-row x 3-column scatter figure.
    Rows: wet (top), dry (bottom).
    Columns: Barkly, Barcaldine LGA, Limestone Coast.
    Points coloured by year.
    """
    seasons = ["wet", "dry"]
    nrows, ncols = len(seasons), len(REGION_NAMES)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 4.2 * nrows),
        squeeze=False,
    )

    year_cmap = cm.get_cmap("plasma_r")
    year_norm = plt.Normalize(vmin=int(all_years.min()), vmax=int(all_years.max()))

    for row, season in enumerate(seasons):
        for col, rname in enumerate(REGION_NAMES):
            ax = axes[row, col]
            key = (rname, season)

            barra_s = barra_data.get(key)
            agcd_s  = agcd_data.get(key)

            if barra_s is None or agcd_s is None:
                ax.set_visible(False)
                continue

            # Align on common years (handles wet year exclusions automatically)
            common_idx = barra_s.index.intersection(agcd_s.index)
            common_idx = common_idx[~np.isnan(barra_s.loc[common_idx].values.astype(float))
                                    & ~np.isnan(agcd_s.loc[common_idx].values.astype(float))]
            barra_vals = barra_s.loc[common_idx].values.astype(float)
            agcd_vals  = agcd_s.loc[common_idx].values.astype(float)
            yr_vals    = common_idx.values.astype(float)

            # Scatter coloured by year
            sc = ax.scatter(
                agcd_vals,
                barra_vals,
                c=yr_vals,
                cmap=year_cmap,
                norm=year_norm,
                s=30,
                linewidths=0.3,
                edgecolors="white",
                zorder=3,
            )

            # 1:1 line spanning data range
            finite = np.isfinite(agcd_vals) & np.isfinite(barra_vals)
            if finite.any():
                vmin = min(agcd_vals[finite].min(), barra_vals[finite].min())
                vmax = max(agcd_vals[finite].max(), barra_vals[finite].max())
                pad  = (vmax - vmin) * 0.06
                ref  = np.array([vmin - pad, vmax + pad])
                ax.plot(ref, ref, color="0.3", linewidth=0.9, zorder=2)
                ax.set_xlim(ref)
                ax.set_ylim(ref)

            # Stats annotation
            r, rmse, bias = scatter_stats(agcd_vals, barra_vals)
            n_pts = int(finite.sum())
            stat_lines = [
                f"r = {r:.2f}",
                f"RMSE = {rmse:.1f}",
                f"bias = {bias:+.1f}",
                f"n = {n_pts}",
            ]
            ax.text(
                0.04, 0.97,
                "\n".join(stat_lines),
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=8.5,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2.5),
            )

            # Season label (bottom right, subdued)
            ax.text(
                0.97, 0.03,
                SEASON_LABEL[season],
                transform=ax.transAxes,
                va="bottom", ha="right",
                fontsize=8.5, color="0.45",
            )

            # Column title (region name, top row only)
            if row == 0:
                thr = REGION_THRESHOLDS[rname]
                col_title = REGION_DISPLAY[rname]
                if metric == "freq":
                    col_title += f"\n(threshold: {thr} kJ m\u207b\u00b2 hr\u207b\u00b9)"
                ax.set_title(col_title, fontsize=10, fontweight="bold", pad=6)

            # Axis labels
            if col == 0:
                ax.set_ylabel(
                    f"BARRA-R2\n{METRIC_AXIS_LABEL[metric]}",
                    fontsize=9.5,
                )
            if row == nrows - 1:
                ax.set_xlabel(
                    f"AGCD + AWRA\n{METRIC_AXIS_LABEL[metric]}",
                    fontsize=9.5,
                )

            ax.grid(alpha=0.2, linewidth=0.5)
            ax.tick_params(labelsize=8.5)

    fig.suptitle(suptitle, fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 0.88, 0.97])   # reserves right margin for colourbar

    # Shared colourbar - added after tight_layout so it sits in the reserved margin
    sm = cm.ScalarMappable(cmap=year_cmap, norm=year_norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        fraction=0.015,
        pad=0.03,
        shrink=0.6,
    )
    cbar.set_label("Year", fontsize=9)
    cbar.ax.tick_params(labelsize=8.5)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {outpath}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    print("[INFO] Loading region geometries ...")
    regions_gdf = load_all_regions(REGIONS, target_crs=BARRA_CRS)

    print("[INFO] Opening BARRA metrics ...")
    ds_barra = xr.open_dataset(BARRA_METRICS_PATH, decode_timedelta=False)

    print("[INFO] Opening AGCD metrics ...")
    ds_agcd = load_agcd_metrics(AGCD_METRICS_PATH)

    # Common year range: intersection of both datasets, capped at COMPARE_END
    barra_years = set(int(y) for y in ds_barra["year"].values)
    agcd_years  = set(int(y) for y in ds_agcd["year"].values)
    compare_years = np.array(sorted(
        barra_years & agcd_years & set(range(COMPARE_START, COMPARE_END + 1))
    ))
    print(f"[INFO] Comparison period: {compare_years[0]}-{compare_years[-1]} "
          f"({len(compare_years)} years)")
    print(f"[INFO] Wet years excluded: {sorted(WET_YEAR_EXCLUDE)}")

    # Subset BARRA to comparison period before spatial extraction (memory)
    ds_barra_sel = ds_barra.sel(year=compare_years)

    # ------------------------------------------------------------------
    # Extract BARRA regional series
    # ------------------------------------------------------------------
    print("[INFO] Extracting BARRA regional means (this may take a minute) ...")
    barra_lcimax: Dict = {}
    barra_freq:   Dict = {}

    for rname, rgdf in regions_gdf.items():
        thr = float(REGION_THRESHOLDS[rname])
        print(f"  {rname} ...")
        for season in ("wet", "dry"):
            try:
                barra_lcimax[(rname, season)] = extract_barra_regional_series(
                    ds_barra_sel, "lci_max", season, rname, rgdf,
                )
            except Exception as e:
                print(f"    [WARN] lci_max_{season}: {e}")

            try:
                barra_freq[(rname, season)] = extract_barra_regional_series(
                    ds_barra_sel, "freq", season, rname, rgdf, threshold=thr,
                )
            except Exception as e:
                print(f"    [WARN] freq_{season}: {e}")

    # ------------------------------------------------------------------
    # Extract AGCD regional series
    # ------------------------------------------------------------------
    print("[INFO] Extracting AGCD regional series ...")
    agcd_lcimax: Dict = {}
    agcd_freq:   Dict = {}

    for rname in REGION_NAMES:
        thr = float(REGION_THRESHOLDS[rname])
        for season in ("wet", "dry"):
            excl = WET_YEAR_EXCLUDE if season == "wet" else None
            try:
                s = agcd_regional_series(
                    ds_agcd, f"lci_max_{season}", rname, exclude_years=excl,
                )
                agcd_lcimax[(rname, season)] = s.loc[s.index.isin(compare_years)]
            except Exception as e:
                print(f"  [WARN] agcd lci_max_{season} for {rname}: {e}")

            try:
                s = agcd_regional_series(
                    ds_agcd, f"freq_{season}", rname, threshold=thr, exclude_years=excl,
                )
                agcd_freq[(rname, season)] = s.loc[s.index.isin(compare_years)]
            except Exception as e:
                print(f"  [WARN] agcd freq_{season} for {rname}: {e}")

    # ------------------------------------------------------------------
    # Figure 1: lci_max scatter
    # ------------------------------------------------------------------
    print("[INFO] Plotting lci_max scatter ...")
    make_scatter_figure(
        barra_data=barra_lcimax,
        agcd_data=agcd_lcimax,
        metric="lci_max",
        all_years=compare_years,
        outpath=os.path.join(OUTDIR, "barra_vs_agcd_lci_max_scatter.png"),
        suptitle=(
            r"BARRA-R2 vs AGCD+AWRA: maximum daily LCI (kJ m$^{-2}$ hr$^{-1}$), 1979-2018"
        ),
    )

    # ------------------------------------------------------------------
    # Figure 2: freq scatter
    # ------------------------------------------------------------------
    print("[INFO] Plotting freq scatter ...")
    make_scatter_figure(
        barra_data=barra_freq,
        agcd_data=agcd_freq,
        metric="freq",
        all_years=compare_years,
        outpath=os.path.join(OUTDIR, "barra_vs_agcd_freq_scatter.png"),
        suptitle=(
            "BARRA-R2 vs AGCD+AWRA: frequency of days exceeding LCI threshold "
            "(days season\u207b\u00b9), 1979-2018"
        ),
    )

    print(f"\n[DONE] Figures written to {OUTDIR}")


if __name__ == "__main__":
    main()