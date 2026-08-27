#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
barra_lci_event_timing_metrics.py

Compute and plot continental BARRA-R2 maps of the timing of seasonal
LCI threshold exceedances.

For each threshold and season, this script computes:

    first_day_mean  : mean first day-of-season with LCI >= threshold
    last_day_mean   : mean last day-of-season with LCI >= threshold
    span_days_mean  : mean length of the seasonal exceedance window
                      = last_day - first_day + 1

Seasons
-------
dry : May-Oct, labelled by calendar year Y
wet : Nov-Apr, labelled by ending year Y
      e.g. wet 1980 = Nov-Dec 1979 + Jan-Apr 1980

Inputs
------
/scratch/dx2/rt9243/barra_daily_lci/barra_lci_daily_aus_land_{year}.nc

Expected variable:
    lci_daily(time, lat, lon)

Outputs
-------
NetCDF:
    barra_lci_event_timing_climatology_<start>_<end>.nc

Figures:
    timing_maps_<season>_thr<threshold>.png

Usage
-----
module use /g/data/xp65/public/modules
module load conda/analysis3

python barra_lci_event_timing_maps.py \
    --start-year 1980 \
    --end-year 2024 \
    --seasons wet,dry \
    --thresholds 1000,1100,1200
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ============================================================
# CONFIG
# ============================================================

BARRA_DAILY_DIR = "/scratch/dx2/rt9243/barra_daily_lci"

DEFAULT_OUTDIR = (
    "/scratch/dx2/rt9243/chill_projections/lci_event_timing_maps"
)

THRESHOLDS = [1000, 1100, 1200]

DRY_MONTHS = [5, 6, 7, 8, 9, 10]
WET_MONTHS = [11, 12, 1, 2, 3, 4]

EXTENT = [110, 155.5, -45, -9]

SEASON_LABELS = {
    "wet": "November-April",
    "dry": "May-October",
}

# 1-based day-of-season labels.
# Dry season is May-Oct: day 1 approx 1 May.
# Wet season is Nov-Apr: day 1 approx 1 Nov.
DRY_DAY_TICKS = [1, 32, 62, 93, 123, 154, 184]
DRY_DAY_LABELS = ["1 May", "1 Jun", "1 Jul", "1 Aug", "1 Sep", "1 Oct", "31 Oct"]

WET_DAY_TICKS = [1, 31, 62, 93, 124, 153, 181]
WET_DAY_LABELS = ["1 Nov", "1 Dec", "1 Jan", "1 Feb", "1 Mar", "1 Apr", "30 Apr"]


# ============================================================
# BASIC HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def barra_daily_path(year: int) -> str:
    return os.path.join(BARRA_DAILY_DIR, f"barra_lci_daily_aus_land_{year}.nc")


def open_barra_lci_year(year: int) -> xr.DataArray:
    path = barra_daily_path(year)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ds = xr.open_dataset(path, decode_times=True)

    if "lci_daily" not in ds:
        raise KeyError(f"'lci_daily' not found in {path}. Found: {list(ds.data_vars)}")

    da = ds["lci_daily"].astype("float32")
    return da


def open_barra_lci_season(year: int, season: str) -> xr.DataArray:
    """
    Open one labelled season.

    dry Y = May-Oct Y
    wet Y = Nov-Dec Y-1 + Jan-Apr Y
    """
    if season == "dry":
        da = open_barra_lci_year(year)
        return da.sel(time=da.time.dt.month.isin(DRY_MONTHS))

    if season == "wet":
        da_prev = open_barra_lci_year(year - 1)
        da_curr = open_barra_lci_year(year)

        nov_dec = da_prev.sel(time=da_prev.time.dt.month.isin([11, 12]))
        jan_apr = da_curr.sel(time=da_curr.time.dt.month.isin([1, 2, 3, 4]))

        return xr.concat([nov_dec, jan_apr], dim="time").sortby("time")

    raise ValueError(f"Unknown season: {season}")


# ============================================================
# EVENT TIMING CALCULATION
# ============================================================

def timing_metrics_one_season(
    lci: xr.DataArray,
    thresholds: List[int],
) -> xr.Dataset:
    """
    Compute first/last/span for one season and all thresholds.

    Returns
    -------
    xr.Dataset with dims:
        threshold, lat, lon
    """
    threshold_da = xr.DataArray(
        np.array(thresholds, dtype=np.float32),
        dims=("threshold",),
        coords={"threshold": np.array(thresholds, dtype=np.float32)},
        attrs={"units": "kJ m-2 hr-1"},
    )

    # dims: threshold, time, lat, lon
    exceed = lci.expand_dims(threshold=threshold_da) >= threshold_da

    # 1-based day-of-season.
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

    ds = xr.Dataset(
        {
            "first_day": first_day,
            "last_day": last_day,
            "span_days": span_days,
            "event_days": event_days,
            "any_event": any_event.astype("int8"),
        }
    )

    ds["first_day"].attrs.update({
        "long_name": "First day of season with LCI >= threshold",
        "units": "day of season",
        "note": "1-based. NaN where threshold was never exceeded.",
    })
    ds["last_day"].attrs.update({
        "long_name": "Last day of season with LCI >= threshold",
        "units": "day of season",
        "note": "1-based. NaN where threshold was never exceeded.",
    })
    ds["span_days"].attrs.update({
        "long_name": "Span between first and last exceedance",
        "units": "days",
        "note": "last_day - first_day + 1. This includes gaps between exceedance days.",
    })
    ds["event_days"].attrs.update({
        "long_name": "Number of exceedance days in season",
        "units": "days season-1",
    })
    ds["any_event"].attrs.update({
        "long_name": "At least one exceedance in season",
        "units": "1",
    })

    return ds


def build_yearly_timing_dataset(
    start_year: int,
    end_year: int,
    seasons: List[str],
    thresholds: List[int],
) -> xr.Dataset:
    """
    Build yearly timing metrics with dims:
        season, year, threshold, lat, lon
    """
    season_dsets = []

    for season in seasons:
        yearly = []

        for year in range(start_year, end_year + 1):
            try:
                lci = open_barra_lci_season(year, season)
            except FileNotFoundError as exc:
                print(f"[WARN] Missing input for {season} {year}: {exc}")
                continue

            if lci.sizes["time"] == 0:
                print(f"[WARN] Empty season: {season} {year}")
                continue

            print(
                f"[INFO] {season} {year}: "
                f"{str(lci.time.values[0])[:10]} to {str(lci.time.values[-1])[:10]} "
                f"({lci.sizes['time']} days)"
            )

            ds_y = timing_metrics_one_season(lci, thresholds)
            ds_y = ds_y.expand_dims(year=[year])
            yearly.append(ds_y)

        if not yearly:
            print(f"[WARN] No output for season={season}")
            continue

        ds_s = xr.concat(yearly, dim="year")
        ds_s = ds_s.expand_dims(season=[season])
        season_dsets.append(ds_s)

    if not season_dsets:
        raise RuntimeError("No yearly timing datasets were produced.")

    ds = xr.concat(season_dsets, dim="season")
    ds = ds.transpose("season", "year", "threshold", "lat", "lon")

    ds.attrs.update({
        "title": "BARRA-R2 yearly LCI event timing metrics",
        "description": (
            "First day, last day and span length of seasonal LCI threshold "
            "exceedances for each grid cell."
        ),
        "season_definition": (
            "dry=May-Oct labelled by calendar year; "
            "wet=Nov-Apr labelled by ending year"
        ),
    })

    return ds


def climatology_from_yearly(ds_yearly: xr.Dataset) -> xr.Dataset:
    """
    Average yearly timing metrics over years.

    Important:
    first_day_mean and last_day_mean are averaged only over years where
    an exceedance occurred at that grid cell.
    """
    clim = xr.Dataset(
        {
            "first_day_mean": ds_yearly["first_day"].mean("year", skipna=True),
            "last_day_mean": ds_yearly["last_day"].mean("year", skipna=True),
            "span_days_mean": ds_yearly["span_days"].mean("year", skipna=True),
            "event_days_mean": ds_yearly["event_days"].mean("year", skipna=True),
            "event_probability": ds_yearly["any_event"].mean("year", skipna=True),
        }
    )

    clim["first_day_mean"].attrs.update({
        "long_name": "Mean first day of season with LCI >= threshold",
        "units": "day of season",
    })
    clim["last_day_mean"].attrs.update({
        "long_name": "Mean last day of season with LCI >= threshold",
        "units": "day of season",
    })
    clim["span_days_mean"].attrs.update({
        "long_name": "Mean seasonal span between first and last exceedance",
        "units": "days",
    })
    clim["event_days_mean"].attrs.update({
        "long_name": "Mean number of exceedance days per season",
        "units": "days season-1",
    })
    clim["event_probability"].attrs.update({
        "long_name": "Fraction of years with at least one threshold exceedance",
        "units": "1",
    })

    clim.attrs.update(ds_yearly.attrs)
    clim.attrs["climatology_note"] = (
        "Means are calculated over the year dimension. "
        "first_day_mean, last_day_mean and span_days_mean ignore years/grid cells "
        "with no exceedance because those years are NaN."
    )

    return clim


# ============================================================
# NETCDF WRITING
# ============================================================

def write_netcdf(ds: xr.Dataset, path: str) -> None:
    ensure_dir(os.path.dirname(path))

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

    tmp = path + ".tmp"
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)
    print(f"[OK] wrote {path}")


# ============================================================
# PLOTTING
# ============================================================

def setup_map(ax) -> None:
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
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
    lon = da["lon"].values
    lat = da["lat"].values

    pcm = ax.pcolormesh(
        lon,
        lat,
        da.values,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
        zorder=1,
    )
    return pcm


def date_norm_and_ticks(season: str):
    if season == "dry":
        levels = np.arange(1, 185 + 1, 15)
        ticks = DRY_DAY_TICKS
        ticklabels = DRY_DAY_LABELS
    elif season == "wet":
        levels = np.arange(1, 181 + 1, 15)
        ticks = WET_DAY_TICKS
        ticklabels = WET_DAY_LABELS
    else:
        raise ValueError(season)

    cmap = plt.get_cmap("viridis", len(levels) - 1).copy()
    cmap.set_bad("white")
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    return cmap, norm, ticks, ticklabels


def span_norm():
    levels = np.arange(0, 185 + 1, 15)
    cmap = plt.get_cmap("YlOrRd", len(levels) - 1).copy()
    cmap.set_bad("white")
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    ticks = np.arange(0, 181, 30)
    ticklabels = [str(int(t)) for t in ticks]
    return cmap, norm, ticks, ticklabels


def plot_timing_panel(
    clim: xr.Dataset,
    season: str,
    threshold: int,
    outpath: str,
    min_event_probability: float = 0.1,
    dpi: int = 250,
) -> None:
    """
    Plot first date, last date and span length for one season/threshold.
    """
    sub = clim.sel(season=season, threshold=float(threshold))

    # Optional mask: avoid showing noisy dates in places where exceedances are rare.
    mask = sub["event_probability"] >= min_event_probability

    first = sub["first_day_mean"].where(mask)
    last = sub["last_day_mean"].where(mask)
    span = sub["span_days_mean"].where(mask)

    date_cmap, date_norm, date_ticks, date_labels = date_norm_and_ticks(season)
    span_cmap, span_norm_obj, span_ticks, span_labels = span_norm()

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1, 0.08],
        hspace=0.08,
        wspace=0.04,
        left=0.04,
        right=0.98,
        top=0.87,
        bottom=0.12,
    )

    axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]

    fields = [
        (first, "Mean first chill date", date_cmap, date_norm),
        (last, "Mean last chill date", date_cmap, date_norm),
        (span, "Mean span length", span_cmap, span_norm_obj),
    ]

    pcms = []
    for ax, (da, title, cmap, norm) in zip(axes, fields):
        setup_map(ax)
        pcm = pcolormesh_map(ax, da, cmap, norm)
        ax.set_title(title, fontsize=14, fontweight="bold")
        pcms.append(pcm)

    # Date colourbar for first/last maps.
    cax1 = fig.add_subplot(gs[1, 0:2])
    cb1 = fig.colorbar(
        pcms[0],
        cax=cax1,
        orientation="horizontal",
        ticks=date_ticks,
    )
    cb1.ax.set_xticklabels(date_labels, rotation=35, ha="right")
    cb1.set_label("Day of season", fontsize=11)

    # Span colourbar.
    cax2 = fig.add_subplot(gs[1, 2])
    cb2 = fig.colorbar(
        pcms[2],
        cax=cax2,
        orientation="horizontal",
        ticks=span_ticks,
    )
    cb2.ax.set_xticklabels(span_labels)
    cb2.set_label("Days", fontsize=11)

    title = (
        f"BARRA-R2 LCI event timing climatology: "
        f"{SEASON_LABELS[season]}, threshold {threshold} kJ m$^{{-2}}$ hr$^{{-1}}$"
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.965)

    fig.text(
        0.04,
        0.04,
        (
            f"Masked where threshold is exceeded in < "
            f"{min_event_probability:.0%} of years."
        ),
        fontsize=9,
        color="0.35",
    )

    ensure_dir(os.path.dirname(outpath))
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[OK] wrote {outpath}")


def plot_all_maps(
    clim: xr.Dataset,
    outdir: str,
    seasons: List[str],
    thresholds: List[int],
    min_event_probability: float,
    dpi: int,
) -> None:
    figdir = os.path.join(outdir, "figures")
    ensure_dir(figdir)

    for season in seasons:
        for threshold in thresholds:
            outpath = os.path.join(
                figdir,
                f"timing_maps_{season}_thr{int(threshold)}.png",
            )
            plot_timing_panel(
                clim=clim,
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
        description="Compute and plot BARRA-R2 continental LCI event timing maps."
    )
    ap.add_argument("--start-year", type=int, default=1980)
    ap.add_argument("--end-year", type=int, default=2024)
    ap.add_argument("--seasons", default="wet,dry")
    ap.add_argument("--thresholds", default="1000,1100,1200")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--write-yearly", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--dpi", type=int, default=250)
    ap.add_argument(
        "--min-event-probability",
        type=float,
        default=0.10,
        help=(
            "Mask grid cells where the threshold is exceeded in less than this "
            "fraction of years. Default 0.10."
        ),
    )

    args = ap.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    thresholds = [int(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    ensure_dir(args.outdir)

    clim_path = os.path.join(
        args.outdir,
        f"barra_lci_event_timing_climatology_{args.start_year}_{args.end_year}.nc",
    )
    yearly_path = os.path.join(
        args.outdir,
        f"barra_lci_event_timing_yearly_{args.start_year}_{args.end_year}.nc",
    )

    if args.plot_only:
        print(f"[INFO] opening existing climatology: {clim_path}")
        clim = xr.open_dataset(clim_path, decode_timedelta=False)
    else:
        yearly = build_yearly_timing_dataset(
            start_year=args.start_year,
            end_year=args.end_year,
            seasons=seasons,
            thresholds=thresholds,
        )

        if args.write_yearly:
            write_netcdf(yearly, yearly_path)

        clim = climatology_from_yearly(yearly)
        write_netcdf(clim, clim_path)

    plot_all_maps(
        clim=clim,
        outdir=args.outdir,
        seasons=seasons,
        thresholds=thresholds,
        min_event_probability=args.min_event_probability,
        dpi=args.dpi,
    )

    print("[DONE]")


if __name__ == "__main__":
    main()