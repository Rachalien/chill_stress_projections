#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_supp_lci_trends_midcentury.py
==================================
Supplementary figure: trends in livestock cold stress at the three focal
regions, historical through to mid-century, under SSP3-7.0.

Replaces the Time of Emergence bar figure (plot_toe_emergence_timeline*.py),
which is no longer part of the analysis.

Layout
------
Two rows x three columns, one column per focal region:

                Winton            Yilgarn-Coolgardie   Cobar-Lachlan
                (Nov-Mar)         (May-Oct)            (May-Oct)
  row a)  Seasonal maximum daily LCI            (kJ m-2 hr-1)
  row b)  Days per season with LCI >= threshold (days)

Each region uses its own focal season and focal threshold:

  Winton              wet  Nov-Mar   1000 kJ m-2 hr-1
  Yilgarn-Coolgardie  dry  May-Oct   1200 kJ m-2 hr-1
  Cobar-Lachlan       dry  May-Oct   1100 kJ m-2 hr-1

Per panel
---------
  * annual ensemble-mean value (thin line, faint markers)
  * inter-member spread: p5-p95 across the 8 members (ribbon), or +/-1
    ensemble SD from the CSV if per-member NetCDFs are unavailable
  * fitted trend line over the full displayed window, with the slope per
    decade and its Mann-Kendall p-value annotated in the panel
  * optional 20-yr centred rolling mean (--rolling)
  * dotted vertical rule at the historical -> SSP3-7.0 join

Data sources
------------
Primary (ensemble mean, +/-1 SD, historical reference statistics):

  <TOE>/<scenario>/<metric>_<season>[_thr<N>]/regional/
      regional_timeseries_<scenario>_<metric>_<season>[_thr<N>].csv

  Columns: region, period, year, value, ensstd, hist_mean, hist_std,
           hist_p_low, hist_p_high, toe_sn, toe_pct
  (produced by narclim2_toe_pipeline.py -- Step E)

Optional (true p5-p95 ribbon and per-member traces):

  <NETCDF_BASE>/<scenario>/per_member/
      lci_metrics_<member>_<scenario>_<y0>-<y1>.nc
  (produced by lci_metrics_narclim2_aust05i.py -- Step B)

These are the same sources as the main-text timeseries figures, so the
Supplementary trend numbers reconcile exactly with the main text. Note in
particular that `freq` is a per-cell exceedance count that is area-averaged
afterwards; it is NOT the exceedance count of the regional-mean daily LCI
series in synoptic_composites/daily_lci/*.csv, and the two are not
interchangeable.

Partial wet-season years
------------------------
Wet seasons are labelled by their Jan-Mar year, so wet season Y draws on
Nov-Dec of year Y-1. Two wet-season years are therefore incomplete in the
gridded output and are dropped by default (--keep-partial-wet-years to
retain them):

  * 1990  -- the historical run starts 1990-01-01, so Nov-Dec 1989 is absent
  * 2015  -- Nov-Dec 2014 sits in the historical file, Jan-Mar 2015 in the
             ssp370 file, and neither contains the whole season

Dry-season panels are unaffected (May-Oct never spans a calendar year).

Usage
-----
  python plot_supp_lci_trends_midcentury.py --check-only
  python plot_supp_lci_trends_midcentury.py
  python plot_supp_lci_trends_midcentury.py --end 2059 --rolling
  python plot_supp_lci_trends_midcentury.py --no-members --trend ols

Outputs
-------
  <OUTDIR>/supp_lci_trends_midcentury_<scenario>.png   (300 dpi)
  <OUTDIR>/supp_lci_trends_midcentury_<scenario>.pdf   (vector)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# Optional heavy dependencies -- only needed for the per-member ribbon
try:
    import geopandas as gpd
    import xarray as xr
    import rioxarray  # noqa: F401  -- activates the .rio accessor
    _HAS_GEO = True
    _GEO_ERR = ""
except ImportError as _exc:          # pragma: no cover
    _HAS_GEO = False
    _GEO_ERR = str(_exc)

SCRIPT_NAME = "plot_supp_lci_trends_midcentury.py"


# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

TOE = Path("/scratch/dx2/rt9243/chill_projections/LCI_time_of_emergence_narclim2")
NETCDF_BASE = Path(
    "/scratch/dx2/rt9243/chill_projections"
    "/LCI_metrics_per_model_then_ensemble/AUST-05i-NARCliM2"
)
OUTDIR = Path("/scratch/dx2/rt9243/chill_projections/publication_figures_narclim")

_LGA_SHP = (
    "/g/data/dx2/rt9243/Datasets/"
    "LGA_2025_AUST_GDA2020/LGA_2025_AUST_GDA2020.shp"
)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis configuration
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO = "ssp370"
HIST_END = 2014          # last year of the historical experiment
DEFAULT_START = 1990
DEFAULT_END = 2060       # "mid-century"
ROLLING_YEARS = 20
ROLLING_MIN = 15
DATA_CRS = "EPSG:4326"

MEMBERS = [
    "ACCESS-ESM1-5_WRF412R3",
    "ACCESS-ESM1-5_WRF412R5",
    "EC-Earth3-Veg_WRF412R3",
    "EC-Earth3-Veg_WRF412R5",
    "MPI-ESM1-2-HR_WRF412R3",
    "MPI-ESM1-2-HR_WRF412R5",
    "NorESM2-MM_WRF412R3",
    "NorESM2-MM_WRF412R5",
]

# Per-member NetCDF year spans, keyed by experiment
MEMBER_SPAN = {"historical": (1990, 2014), "ssp370": (2015, 2100)}

# One column per focal region, in west-to-east / north-to-south reading order.
# `season` must match the suffix used in the CSV filenames ("wet" / "dry").
# One column per focal region. Order is Cobar-Lachlan, Yilgarn-Coolgardie,
# Winton, matching the region order used throughout the manuscript and the
# other figures. Panel letters follow this order automatically.
# `season` must match the suffix used in the CSV filenames ("wet" / "dry").
FOCAL_REGIONS = [
    dict(
        name="Cobar-Lachlan",
        season="dry",
        season_label="May\u2013Oct",
        threshold=1100,
        region_values=["Cobar", "Lachlan"],
        dissolve=True,
    ),
    dict(
        name="Yilgarn-Coolgardie",
        season="dry",
        season_label="May\u2013Oct",
        threshold=1200,
        region_values=["Yilgarn", "Coolgardie"],
        dissolve=True,
    ),
    dict(
        name="Winton",
        season="wet",
        season_label="Nov\u2013Mar",
        threshold=1000,
        region_values=["Winton"],
        dissolve=False,
    ),
]

# Row definitions
ROWS = [
    dict(
        metric="lci_max",
        use_threshold=False,
        row_title="LCI$_{\\mathrm{MAX}}$",
        ylabel="LCI (kJ m$^{-2}$ hr$^{-1}$)",
        slope_unit="kJ m$^{-2}$ hr$^{-1}$ decade$^{-1}$",
        slope_fmt="{:+.1f}",
    ),
    dict(
        metric="freq",
        use_threshold=True,
        row_title="Frequency",
        ylabel="Days per season",
        slope_unit="d decade$^{-1}$",
        slope_fmt="{:+.2f}",
    ),
]

# Wet-season years that are structurally incomplete in the gridded output
PARTIAL_WET_YEARS = {1990, 2015}


# ══════════════════════════════════════════════════════════════════════════════
# Style
# ══════════════════════════════════════════════════════════════════════════════

C_ANNUAL = "#5B8DB8"     # annual ensemble-mean line
C_RIBBON = "#9DBFD8"     # inter-member spread
C_TREND = "#1F3F6E"      # fitted trend line
TREND_LW = 1.4           # line weight
HEADROOM_FRAC = 0.42     # extra y-axis space reserved above data for annotations
C_ROLL = "#B3541E"       # rolling mean
C_JOIN = "0.45"          # historical / scenario join rule
C_HISTMEAN = "0.35"      # 1990-2014 reference mean

FS_ROWTITLE = 16
FS_COLTITLE = 16
FS_AXIS = 15
FS_TICK = 14
FS_ANNOT = 12
FS_LEGEND = 13

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})


# ══════════════════════════════════════════════════════════════════════════════
# Path builders
# ══════════════════════════════════════════════════════════════════════════════

def suffix(metric: str, season: str, threshold: Optional[int]) -> str:
    """Directory / filename stem, matching narclim2_toe_pipeline.suffix()."""
    s = f"{metric}_{season}"
    if threshold is not None:
        s += f"_thr{int(threshold)}"
    return s


def regional_csv_path(scenario: str, metric: str, season: str,
                      threshold: Optional[int]) -> Path:
    sfx = suffix(metric, season, threshold)
    return TOE / scenario / sfx / "regional" / f"regional_timeseries_{scenario}_{sfx}.csv"


def member_nc_path(member: str, experiment: str) -> Path:
    y0, y1 = MEMBER_SPAN[experiment]
    return NETCDF_BASE / experiment / "per_member" / \
        f"lci_metrics_{member}_{experiment}_{y0}-{y1}.nc"


# ══════════════════════════════════════════════════════════════════════════════
# Region geometry
# ══════════════════════════════════════════════════════════════════════════════

def load_region_gdf(spec: dict) -> "gpd.GeoDataFrame":
    """Load and (optionally) dissolve the LGA polygons for one focal region."""
    gdf = gpd.read_file(_LGA_SHP)
    gdf = gdf[gdf["LGA_NAME25"].isin(spec["region_values"])].copy()
    if gdf.empty:
        raise ValueError(
            f"No LGA_NAME25 rows matched {spec['region_values']} for "
            f"region {spec['name']!r}. Check for en-dashes vs hyphens."
        )
    gdf = gdf.reset_index(drop=True)
    if spec["dissolve"] and len(spec["region_values"]) > 1:
        gdf = gdf.dissolve().explode(index_parts=False).dissolve().reset_index(drop=True)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(DATA_CRS)


def area_weighted_mean(da: "xr.DataArray",
                       gdf: "gpd.GeoDataFrame") -> "xr.DataArray":
    """Clip to the region polygon and take a cos(lat)-weighted spatial mean."""
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    da = da.rio.write_crs(DATA_CRS, inplace=False)
    clipped = da.rio.clip([gdf.geometry.union_all()], gdf.crs, drop=True)
    weights = xr.DataArray(
        np.cos(np.deg2rad(clipped["lat"])),
        coords={"lat": clipped["lat"]}, dims=("lat",),
    )
    return clipped.weighted(weights).mean(dim=("lat", "lon"), skipna=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_ensemble_series(scenario: str, region: str, metric: str, season: str,
                         threshold: Optional[int],
                         year_start: int, year_end: int,
                         drop_partial_wet: bool) -> pd.DataFrame:
    """
    Return a tidy DataFrame [year, value, ensstd, hist_mean, hist_std] spanning
    historical and scenario years continuously, restricted to the display window.
    """
    path = regional_csv_path(scenario, metric, season, threshold)
    if not path.exists():
        raise FileNotFoundError(f"Regional timeseries CSV not found:\n  {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    sub = df[df["region"] == region].copy()
    if sub.empty:
        raise ValueError(
            f"Region {region!r} not present in {path.name}. "
            f"Available: {sorted(df['region'].unique())}"
        )

    sub["year"] = sub["year"].astype(int)
    sub = sub[(sub["year"] >= year_start) & (sub["year"] <= year_end)]
    sub = sub.sort_values("year").reset_index(drop=True)

    if season == "wet" and drop_partial_wet:
        n_before = len(sub)
        sub = sub[~sub["year"].isin(PARTIAL_WET_YEARS)].reset_index(drop=True)
        dropped = n_before - len(sub)
        if dropped:
            print(f"    dropped {dropped} partial wet-season year(s): "
                  f"{sorted(PARTIAL_WET_YEARS)}")

    dupes = sub["year"].duplicated(keep=False)
    if dupes.any():
        print(f"    [WARN] duplicate years in {path.name} for {region}: "
              f"{sorted(sub.loc[dupes, 'year'].unique())}; keeping scenario rows.")
        sub = sub.sort_values(["year", "period"]).drop_duplicates("year", keep="first")

    keep = ["year", "value", "ensstd", "hist_mean", "hist_std"]
    for col in keep:
        if col not in sub.columns:
            sub[col] = np.nan
    return sub[keep].reset_index(drop=True)


def load_member_envelope(scenario: str, spec: dict, metric: str, season: str,
                         threshold: Optional[int],
                         year_start: int, year_end: int,
                         drop_partial_wet: bool) -> Optional[dict]:
    """
    Build the true p5-p95 inter-member envelope by loading every per-member
    NetCDF (historical + scenario) and area-averaging over the region.
    """
    if not _HAS_GEO:
        print(f"    [WARN] geo stack unavailable ({_GEO_ERR}); "
              "falling back to +/-1 ensemble SD.")
        return None

    var = f"{metric}_{season}"
    gdf = load_region_gdf(spec)
    series: dict[str, pd.Series] = {}

    for member in MEMBERS:
        pieces = []
        for experiment in ("historical", scenario):
            path = member_nc_path(member, experiment)
            if not path.exists():
                print(f"    [WARN] missing {path.name}")
                continue
            try:
                ds = xr.open_dataset(path, engine="h5netcdf",
                                     decode_timedelta=False,
                                     drop_variables=["member"])
            except Exception as exc:
                print(f"    [WARN] could not open {path.name}: {exc}")
                continue
            if var not in ds.data_vars:
                print(f"    [WARN] {var!r} absent from {path.name}")
                ds.close()
                continue
            da = ds[var].astype("float32")
            if threshold is not None and "threshold" in da.dims:
                da = da.sel(threshold=float(threshold))
            try:
                regional = area_weighted_mean(da, gdf)
                pieces.append(pd.Series(regional.values.astype(float),
                                        index=regional["year"].values.astype(int)))
            except Exception as exc:
                print(f"    [WARN] clip/weight failed ({member}/{experiment}): {exc}")
            finally:
                ds.close()

        if pieces:
            s = pd.concat(pieces)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            series[member] = s

    if len(series) < 2:
        print("    [WARN] fewer than 2 members loaded; "
              "falling back to +/-1 ensemble SD.")
        return None

    frame = pd.DataFrame(series)
    frame = frame[(frame.index >= year_start) & (frame.index <= year_end)]
    if season == "wet" and drop_partial_wet:
        frame = frame[~frame.index.isin(PARTIAL_WET_YEARS)]
    if frame.empty:
        return None

    print(f"    envelope from {len(series)} members, "
          f"{int(frame.index.min())}\u2013{int(frame.index.max())}")
    return {
        "years": frame.index.values.astype(int),
        "p5": frame.quantile(0.05, axis=1).values.astype(float),
        "p95": frame.quantile(0.95, axis=1).values.astype(float),
        "n_members": len(series),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Trend statistics
# ══════════════════════════════════════════════════════════════════════════════

def fit_trend(years: np.ndarray, values: np.ndarray,
              method: str = "theilsen") -> dict:
    """
    Fit a linear trend and test it with the Mann-Kendall (Kendall tau) test.
    Returns slopes in units per decade.
    """
    mask = np.isfinite(values)
    x = years[mask].astype(float)
    y = values[mask].astype(float)
    if len(x) < 10:
        return dict(slope=np.nan, lo=np.nan, hi=np.nan, p=np.nan,
                    intercept=np.nan, n=len(x), method=method)

    if method == "ols":
        res = stats.linregress(x, y)
        slope, intercept = res.slope, res.intercept
        half = 1.96 * res.stderr
        lo, hi = slope - half, slope + half
    else:
        slope, intercept, lo, hi = stats.theilslopes(y, x, alpha=0.95)

    _, p = stats.kendalltau(x, y)

    return dict(slope=slope * 10.0, lo=lo * 10.0, hi=hi * 10.0, p=float(p),
                intercept=intercept, raw_slope=slope, n=len(x), method=method)


def rolling_mean(years: np.ndarray, values: np.ndarray) -> np.ndarray:
    s = pd.Series(values.astype(float), index=years)
    return s.rolling(ROLLING_YEARS, center=True,
                     min_periods=ROLLING_MIN).mean().values


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_panel(ax, ts: pd.DataFrame, envelope: Optional[dict], row: dict,
               spec: dict, args, panel_letter: str) -> None:
    years = ts["year"].values.astype(int)
    values = ts["value"].values.astype(float)

    # ── inter-member spread ────────────────────────────────────────────────
    if envelope is not None:
        ax.fill_between(envelope["years"], envelope["p5"], envelope["p95"],
                        color=C_RIBBON, alpha=0.45, linewidth=0, zorder=2)
    else:
        sd = ts["ensstd"].values.astype(float)
        if np.isfinite(sd).any():
            ax.fill_between(years, values - sd, values + sd,
                            color=C_RIBBON, alpha=0.45, linewidth=0, zorder=2)

    # ── 1990-2014 reference mean ───────────────────────────────────────────
    hist_mean = ts["hist_mean"].dropna()
    if not hist_mean.empty:
        ax.axhline(float(hist_mean.iloc[0]), color=C_HISTMEAN, linewidth=0.9,
                   linestyle=(0, (6, 4)), alpha=0.8, zorder=3)

    # ── annual ensemble mean ───────────────────────────────────────────────
    ax.plot(years, values, color=C_ANNUAL, linewidth=1.0, alpha=0.9,
            marker="o", markersize=2.4, markerfacecolor=C_ANNUAL,
            markeredgecolor="none", zorder=4)

    # ── optional rolling mean ──────────────────────────────────────────────
    if args.rolling:
        ax.plot(years, rolling_mean(years, values), color=C_ROLL,
                linewidth=1.8, alpha=0.95, zorder=5)

    # ── fitted trend (SSP3-7.0) ────────────────────────────────────────────
    trend = fit_trend(years, values, method=args.trend)
    if np.isfinite(trend["slope"]):
        xs = np.array([years.min(), years.max()], dtype=float)
        ax.plot(xs, trend["intercept"] + trend["raw_slope"] * xs,
                color=C_TREND, linewidth=TREND_LW, zorder=6)

    # ── headroom reservation above plotted series ──────────────────────────
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    ymin, ymax = ax.get_ylim()
    headroom = HEADROOM_FRAC * (ymax - ymin)
    ax.set_ylim(ymin, ymax + headroom)

    if np.isfinite(trend["slope"]):
        # No scenario prefix here: the trend is fitted across the whole
        # displayed record, which spans the historical period as well as
        # SSP3-7.0, so labelling it "SSP3-7.0" would misattribute it.
        label = (
            row["slope_fmt"].format(trend["slope"])
            + " " + row["slope_unit"]
        )
        ax.text(0.035, 0.85, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=FS_ANNOT, color=C_TREND,
                zorder=7)

    # ── historical / scenario join ─────────────────────────────────────────
    # Thickened: at 0.9 pt with alpha 0.8 this rule was hard to
    # pick out against the annual series.
    ax.axvline(HIST_END + 0.5, color=C_JOIN, linewidth=1.8, linestyle=":",
               alpha=1.0, zorder=3)

    # ── axes cosmetics ─────────────────────────────────────────────────────
    ax.set_xlim(args.start - 1, args.end + 1)
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS_TICK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.text(0.02, 0.98, f"({panel_letter})", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS_AXIS, fontweight="bold")

    if row["use_threshold"]:
        ax.text(0.98, 0.98,
                f"\u2265{spec['threshold']} kJ m$^{{-2}}$ hr$^{{-1}}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=FS_ANNOT, color="0.30")

    print(f"    SSP3-7.0 trend {trend['slope']:+.3f} per decade "
          f"[{trend['lo']:+.3f}, {trend['hi']:+.3f}]  "
          f"p={trend['p']:.2e}  n={trend['n']}  ({trend['method']})")


def build_legend(has_envelope: bool, args) -> list:
    handles = [
        Line2D([0], [0], color=C_ANNUAL, linewidth=1.2, marker="o",
               markersize=3.5, label="Ensemble mean (annual)"),
        Patch(facecolor=C_RIBBON, alpha=0.45,
              label="5th\u201395th percentile across members" if has_envelope
                    else "\u00b11 SD across members"),
        Line2D([0], [0], color=C_TREND, linewidth=TREND_LW,
               label=f"Linear trend ({args.start}\u2013{args.end})"),
        Line2D([0], [0], color=C_HISTMEAN, linewidth=0.9,
               linestyle=(0, (6, 4)), label="1990\u20132014 mean"),
        Line2D([0], [0], color=C_JOIN, linewidth=1.8, linestyle=":",
               label="Historical \u2192 future"),
    ]
    if args.rolling:
        handles.insert(3, Line2D([0], [0], color=C_ROLL, linewidth=1.8,
                                 label=f"{ROLLING_YEARS}-yr running mean"))
    return handles


def make_figure(args) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    nrows, ncols = len(ROWS), len(FOCAL_REGIONS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14.0, 7.4),
        sharex=True, sharey="row" if args.share_y else False,
    )
    axes = np.atleast_2d(axes)

    letters = "abcdefghijkl"
    any_envelope = False

    for j, spec in enumerate(FOCAL_REGIONS):
        for i, row in enumerate(ROWS):
            ax = axes[i, j]
            thr = spec["threshold"] if row["use_threshold"] else None

            print(f"  {spec['name']} | {row['metric']}_{spec['season']}"
                  + (f"_thr{thr}" if thr else ""))

            ts = load_ensemble_series(
                SCENARIO, spec["name"], row["metric"], spec["season"], thr,
                args.start, args.end, not args.keep_partial_wet_years,
            )

            envelope = None
            if not args.no_members:
                envelope = load_member_envelope(
                    SCENARIO, spec, row["metric"], spec["season"], thr,
                    args.start, args.end, not args.keep_partial_wet_years,
                )
            any_envelope = any_envelope or envelope is not None

            plot_panel(ax, ts, envelope, row, spec, args,
                       letters[i * ncols + j])

            if j == 0:
                ax.set_ylabel(row["ylabel"], fontsize=FS_AXIS)
            if i == nrows - 1:
                ax.set_xlabel("Year", fontsize=FS_AXIS)
            if i == 0:
                ax.set_title(
                    f"{spec['name']}\n{spec['season_label']}",
                    fontsize=FS_COLTITLE, fontweight="bold", pad=10,
                )

    # Row titles down the right-hand side
    for i, row in enumerate(ROWS):
        axes[i, -1].text(
            1.035, 0.5, row["row_title"], transform=axes[i, -1].transAxes,
            rotation=270, ha="left", va="center", fontsize=FS_ROWTITLE,
        )

    fig.legend(
        handles=build_legend(any_envelope, args),
        loc="lower center", ncols=3,
        bbox_to_anchor=(0.5, -0.045),
        frameon=False, fontsize=FS_LEGEND, handlelength=2.4,
        columnspacing=1.6,
    )

    fig.suptitle(
        "Modelled trends in livestock cold stress",
        fontsize=19, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.955, f"{args.start}\u2013{args.end}",
        ha="center", va="top", fontsize=14, color="0.40",
    )

    fig.tight_layout(rect=[0, 0.02, 0.985, 0.93])

    metadata = {
        "Title": (
            f"Supplementary: modelled trends in livestock cold stress, "
            f"three focal regions ({args.start}-{args.end})"
        ),
        "Description": (
            f"Created by {SCRIPT_NAME}; "
            f"scenario={SCENARIO} (full series, ribbon, trend); "
            f"input={TOE}/<scenario>/<metric>_<season>[_thr<N>]/regional/; "
            f"per-member NetCDFs={NETCDF_BASE}; "
            f"NARCliM2-0 AUST-05i bias-corrected (v1-r1-ACS-QME-BARRAR2-1980-2022), "
            f"{len(MEMBERS)} members; "
            f"regions=Winton (wet Nov-Mar, thr 1000), "
            f"Yilgarn-Coolgardie (dry May-Oct, thr 1200), "
            f"Cobar-Lachlan (dry May-Oct, thr 1100); "
            f"trend={args.trend}, p from Mann-Kendall on the ensemble mean; "
            f"partial wet-season years "
            f"{'retained' if args.keep_partial_wet_years else 'dropped (1990, 2015)'}"
        ),
        "Software": SCRIPT_NAME,
    }

    stem = f"supp_lci_trends_midcentury_{SCENARIO}"
    png = OUTDIR / f"{stem}.png"
    pdf = OUTDIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", metadata=metadata)
    fig.savefig(pdf, bbox_inches="tight",
                metadata={"Title": metadata["Title"], "Creator": SCRIPT_NAME})
    plt.close(fig)

    print(f"\n[OK] PNG: {png}")
    print(f"[OK] PDF: {pdf}")


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

def check_only(args) -> int:
    """Verify every input exists."""
    print("=" * 74)
    print("INPUT VALIDATION")
    print("=" * 74)
    missing = 0

    print("\nRegional timeseries CSVs (required):")
    for spec in FOCAL_REGIONS:
        for row in ROWS:
            thr = spec["threshold"] if row["use_threshold"] else None
            path = regional_csv_path(SCENARIO, row["metric"], spec["season"], thr)
            ok = path.exists()
            print(f"  [{'OK ' if ok else 'MISSING'}] {path}")
            if not ok:
                missing += 1
                continue
            df = pd.read_csv(path)
            if spec["name"] not in set(df["region"].unique()):
                print(f"          [!] region {spec['name']!r} absent; "
                      f"present: {sorted(df['region'].unique())}")
                missing += 1
            else:
                sub = df[df["region"] == spec["name"]]
                print(f"          years {int(sub['year'].min())}"
                      f"\u2013{int(sub['year'].max())}, {len(sub)} rows")

    print("\nPer-member NetCDFs (optional; needed for the p5\u2013p95 ribbon):")
    for experiment in ("historical", SCENARIO):
        for member in MEMBERS:
            path = member_nc_path(member, experiment)
            print(f"  [{'OK ' if path.exists() else 'MISSING'}] {path.name}")

    print("\nOutput directory:")
    print(f"  {OUTDIR}  ({'exists' if OUTDIR.exists() else 'will be created'})")

    print("\n" + "=" * 74)
    if missing:
        print(f"{missing} problem(s) found.")
        return 1
    print("All required inputs present.")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Supplementary trend figure: LCI max and threshold "
                     "exceedance frequency to mid-century, three focal regions."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", type=int, default=DEFAULT_START,
                   help="First year plotted.")
    p.add_argument("--end", type=int, default=DEFAULT_END,
                   help="Last year plotted (mid-century).")
    p.add_argument("--trend", choices=["theilsen", "ols"], default="theilsen",
                   help="Trend estimator; p-value is Mann-Kendall either way.")
    p.add_argument("--rolling", action="store_true",
                   help=f"Overlay a {ROLLING_YEARS}-yr centred running mean.")
    p.add_argument("--no-members", action="store_true",
                   help="Skip per-member NetCDFs; use +/-1 ensemble SD instead.")
    p.add_argument("--share-y", action="store_true",
                   help="Share the y-axis across regions within each row.")
    p.add_argument("--keep-partial-wet-years", action="store_true",
                   help="Retain wet-season years 1990 and 2015 (incomplete).")
    p.add_argument("--check-only", action="store_true",
                   help="Validate inputs and exit without plotting.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.check_only:
        sys.exit(check_only(args))

    print(f"=== {SCRIPT_NAME} ===")
    print(f"Scenario: {SCENARIO} | window {args.start}\u2013{args.end} | trend={args.trend}")
    make_figure(args)


if __name__ == "__main__":
    main()