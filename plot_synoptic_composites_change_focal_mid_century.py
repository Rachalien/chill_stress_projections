#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_synoptic_composites_change_focal_mid_century.py

Compact synoptic-composite CHANGE figure for the three focal region-season
combinations used in the LCI paper.

This is the change-of-anomaly companion to
plot_synoptic_composites_compact_focal_with_barra_vectors.py. Where that
script shows the
historical event anomaly only, this script shows how that anomaly pattern
itself changes between the historical and mid-century periods:

    change = (future_composite - future_clim) - (historical_composite - historical_clim)
            = future event anomaly - historical event anomaly

Because both terms are already anomalies relative to their own period's
seasonal climatology, the mean mid-century warming signal cancels out of
this difference. What is left is whether the top-5% LCI event pattern itself
strengthens, weakens, or shifts under SSP3-7.0, independent of the background
warming trend. This is the panel used to distinguish a change in event
dynamics from simple thermodynamic shift.

Figure layout
-------------
Rows:    Cobar-Lachlan May-Oct | Yilgarn-Coolgardie May-Oct | Winton Nov-Mar
Columns: MSLP | Temperature | Precipitation
Periods: Historical 1990-2014 vs mid-century 2040-2060, SSP3-7.0

Inputs
------
Reads per-member files produced by synoptic_composite_lci_top5pct.py:
    {COMPOSITE_DIR}/{member}_composite_{region}_mid_century.nc

Expected variables include:
    psl_dry_hist, psl_dry_hist_clim, psl_dry_future, psl_dry_future_clim
    tas_dry_hist, tas_dry_hist_clim, tas_dry_future, tas_dry_future_clim
    pr_dry_hist,  pr_dry_hist_clim,  pr_dry_future,  pr_dry_future_clim
    (and the _wet_ equivalents for Winton)

NOTE: synoptic_composite_lci_top5pct.py now writes future composites for
2040-2060 (not 2080-2099) into files tagged "_mid_century.nc", so the old
{member}_composite_{region}.nc files (2080-2099) are left untouched on disk.
This script only reads the _mid_century.nc files, so it can't accidentally
mix the two periods -- but it also won't find any data until
synoptic_composite_lci_top5pct.py has actually been rerun on Gadi for all
8 members.

Usage
-----
    module use /g/data/xp65/public/modules
    module load conda/analysis3

    python plot_synoptic_composites_change_focal.py

Optional:
    python plot_synoptic_composites_change_focal.py --dpi 300
    python plot_synoptic_composites_change_focal.py --wind-contours
    python plot_synoptic_composites_change_focal.py \
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
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def clean_ticks(values, dp: int = 2) -> np.ndarray:
    """Round tick positions and squash float dust to a clean zero.

    np.arange(-1.6, 0.1, 0.4) ends at -4.44e-16 rather than 0.0, which a
    "%g" formatter renders as "-4.44089e-16" on the colourbar. Rounding alone
    turns that into "-0", so near-zero values are snapped to exactly 0.0.
    """
    arr = np.round(np.asarray(values, dtype=float), dp)
    return np.where(np.abs(arr) < 10.0 ** (-dp), 0.0, arr)


def truncate_cmap(name: str, minval: float = 0.0, maxval: float = 1.0,
                  n: int = 256) -> mcolors.LinearSegmentedColormap:
    """Return a colormap using only the [minval, maxval] slice of a named one."""
    base = plt.get_cmap(name)
    return mcolors.LinearSegmentedColormap.from_list(
        f"{name}_trunc", base(np.linspace(minval, maxval, n)),
    )


# Matches the temperature treatment in
# plot_synoptic_composites_compact_focal_with_barra_vectors.py: the blue half
# of RdBu_r, with levels ending at zero and extend="min", so anything above
# zero is left unfilled rather than being given a colour.
TAS_CHANGE_CMAP = truncate_cmap("RdBu_r", 0.0, 0.5)


def brown_white_blue_cmap(n: int = 256) -> mcolors.LinearSegmentedColormap:
    """
    Diverging colormap: brown (dry/negative) -> white (zero) -> blue (wet/positive).

    Built from BrBG's brown half (keeps the dry side consistent with the rest
    of the paper's rainfall figures) stitched to Blues (white -> dark blue)
    for the wet side, so that positives read as blue rather than BrBG's
    green.
    """
    browns = plt.get_cmap("BrBG")(np.linspace(0.0, 0.48, n // 2))
    blues = plt.get_cmap("Blues")(np.linspace(0.04, 1.0, n // 2))
    white = np.array([[1.0, 1.0, 1.0, 1.0]])
    colors = np.vstack([browns, white, blues])
    # N=len(colors): the default N=256 resamples/blends the LUT and can
    # shift the exact-white anchor off-centre; matching N to the control
    # points keeps white pinned precisely at the midpoint.
    return mcolors.LinearSegmentedColormap.from_list("BrWhBu", colors, N=len(colors))


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_COMPOSITE_DIR = "/scratch/dx2/rt9243/chill_projections/synoptic_composites"
DEFAULT_FIGDIR = "/scratch/dx2/rt9243/chill_projections/synoptic_composites/figures"
DEFAULT_SIG_DIR = "/scratch/dx2/rt9243/chill_projections/synoptic_composites/significance"

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

# Change-panel (future anomaly - historical anomaly) levels and colours.
#
# Ranges below are set from the "[Colourbar range check]" this script prints,
# using the observed data spans:
#     MSLP           -2.28 / +2.28   (p1/p99 -1.03 / +1.89)
#     Temperature    -1.55 / +0.25   (p1/p99 -1.08 / +0.15)
#     Precipitation  -3.26 / +3.37   (p1/p99 -1.41 / +1.40)
# MSLP and precipitation turn out to be near-symmetric about zero, so a
# symmetric range wastes nothing and puts white exactly at zero under a plain
# linear mapping -- no TwoSlopeNorm needed. Temperature is genuinely one-sided,
# so it is cut at zero instead (see below).
FIELDS = [
    {
        "key": "mslp",
        "stem": "psl",
        "title": "MSLP",
        "units_change": "hPa",
        "levels_change": np.arange(-2.5, 2.75, 0.25),
        "cmap_change": "RdBu_r",
        "extend": "both",
        # Explicit ticks: left to itself matplotlib labels every third level
        # (0.75 steps, 2 dp), which crowds the bar.
        "cbar_ticks": np.arange(-2, 2.5, 1),
    },
    {
        "key": "tas",
        "stem": "tas",
        "title": "Temperature",
        "units_change": "\u00b0C",
        # Data runs -1.55 to +0.25, so a symmetric range would leave most of
        # the positive half of the bar unused -- exactly the "numbers on the
        # scale that never appear in the panels" problem. Cut at zero instead,
        # matching the historical composite figure: the blue half of RdBu_r,
        # with extend="min" leaving the small positive tail unfilled (white)
        # rather than giving it a colour of its own.
        "levels_change": np.arange(-1.6, 0.1, 0.1),
        "cmap_change": TAS_CHANGE_CMAP,
        "extend": "min",
        "cbar_ticks": np.arange(-1.6, 0.1, 0.4),
    },
    {
        "key": "pr",
        "stem": "pr",
        "title": "Precipitation",
        "units_change": "mm day$^{-1}$",
        # Previously -1.5 to +6, carried over from the historical-anomaly
        # figure where the signal really is mostly positive. For the *change*
        # field that was wrong in both directions: it clipped real drying down
        # to -1.5 when the data reaches -3.26, and reserved +3.4 to +6 for
        # values that never occur. The change field is near-symmetric, so a
        # symmetric range fixes both and removes the need for TwoSlopeNorm.
        "levels_change": np.arange(-3.5, 3.75, 0.25),
        "cmap_change": brown_white_blue_cmap(),
        "extend": "both",
        # Whole numbers only: this bar's labels are the widest of the three
        # ("-3.00" etc.) and at 0.75 spacing they ran together.
        "cbar_ticks": np.arange(-3, 4, 1),
    },
]

SEASON_TEXT = {"dry": "May-October", "wet": "November-March"}

LON_MIN, LON_MAX = 112.0, 154.0
LAT_MIN, LAT_MAX = -44.5, -9.5
WIND_CHG_LEVELS = np.arange(-3, 3.5, 0.5)


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
    # first alphabetically), not a total or average across the ensemble.
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


def load_significance(region_key: str, sig_dir: str) -> xr.Dataset:
    """
    Load the per-region significance file produced by
    compute_synoptic_composite_significance.py. Returns None (with a
    warning) if the file doesn't exist yet, so the figure can still be made
    without stippling rather than hard failing.
    """
    path = Path(sig_dir) / f"{region_key}_significance_mid_century.nc"
    if not path.exists():
        print(f"[WARN] no significance file found at {path}; plotting without stippling. "
              f"Run compute_synoptic_composite_significance.py first to add it.")
        return None
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        return ds.load()


def _anomaly(ens: xr.Dataset, stem: str, season: str, period: str) -> xr.DataArray:
    """Return one period's event anomaly: composite - matching climatology."""
    var = f"{stem}_{season}_{period}"
    clim_var = f"{var}_clim"
    if var not in ens:
        raise KeyError(f"Missing variable {var!r}. Available variables include: {list(ens.data_vars)[:12]}")
    if clim_var not in ens:
        raise KeyError(f"Missing climatology variable {clim_var!r}; regenerate composites with *_clim fields.")
    out = ens[var] - ens[clim_var]
    out.attrs.update(ens[var].attrs)
    return out


def anomaly_change(ens: xr.Dataset, stem: str, season: str) -> xr.DataArray:
    """
    Return future event anomaly minus historical event anomaly:
        (future_composite - future_clim) - (hist_composite - hist_clim)

    Both terms are anomalies relative to their own period's climatology, so
    this isolates the change in the event pattern itself from the background
    mid-century warming shared by both periods.
    """
    anom_hist = _anomaly(ens, stem, season, "hist")
    anom_future = _anomaly(ens, stem, season, "future")
    out = anom_future - anom_hist
    out.attrs.update(anom_hist.attrs)
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
    per_member = ens.attrs.get("n_days_per_member", {})
    parts = []
    for period, tag in (("hist", "hist"), ("future", "fut")):
        var = f"{stem}_{season}_{period}"
        vals = [v for v in per_member.get(var, []) if v is not None]
        if vals:
            parts.append(f"n_{tag}={sum(vals)}")
    return ", ".join(parts)


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


# ── Plot helpers ──────────────────────────────────────────────────────────────

def setup_map(ax, row_i: int, col_i: int, n_rows: int) -> None:
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
    gl.bottom_labels = row_i == n_rows - 1
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
        markersize=10,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=1.6,
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


def plot_change_compact(
        composite_dir: str,
        figdir: str,
        dpi: int,
        outfile: str,
        wind_contours: bool = False,
        sig_dir: str = DEFAULT_SIG_DIR,
        stipple: bool = True,
        stipple_var: str = "sig_fdr",
        level_overrides: Dict[str, np.ndarray] | None = None,
    ) -> Path:
    os.makedirs(figdir, exist_ok=True)
    level_overrides = level_overrides or {}
    # Actual data ranges per field, accumulated during drawing and reported at
    # the end. The unused ends of the MSLP and temperature colourbars are
    # trimmed; these numbers are what that trimming is based on, rather
    # than a guess.
    observed: Dict[str, list] = {f["key"]: [] for f in FIELDS}
    proj = ccrs.PlateCarree()
    n_rows = len(REGIONS)
    n_cols = len(FIELDS)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12.2, 9.2),
        subplot_kw={"projection": proj},
        gridspec_kw={"hspace": 0.06, "wspace": 0.035},
    )

    # Load each region once; every column (field) in a row draws from the same file set.
    region_data: Dict[str, Tuple[xr.Dataset, int]] = {}
    region_sig: Dict[str, xr.Dataset] = {}
    for spec in REGIONS:
        ens, n_members, _ = load_ensemble_mean(spec["key"], composite_dir)
        region_data[spec["key"]] = (ens, n_members)
        region_sig[spec["key"]] = load_significance(spec["key"], sig_dir) if stipple else None
        print_n_days_diagnostics(spec["key"], spec["season"], ens)

    contour_handles = [None] * n_cols
    letters = "abcdefghi"

    for row_i, region in enumerate(REGIONS):
        ens, n_members = region_data[region["key"]]
        lat = ens["lat"].values
        lon = ens["lon"].values
        x, y = np.meshgrid(lon, lat)

        for col_i, field in enumerate(FIELDS):
            ax = axes[row_i, col_i]
            setup_map(ax, row_i, col_i, n_rows)

            da = anomaly_change(ens, field["stem"], region["season"])
            data = np.ma.masked_invalid(da.values)

            finite = np.asarray(da.values)[np.isfinite(da.values)]
            if finite.size:
                observed[field["key"]].append(finite)

            levels = level_overrides.get(field["key"], field["levels_change"])
            norm = field.get("norm_change")
            # An override that is asymmetric about zero hits the same trap the
            # pr column already documents: a plain linear mapping onto a
            # diverging colormap puts white somewhere other than zero. Pin it.
            if field["key"] in level_overrides and norm is None:
                lo, hi = float(levels[0]), float(levels[-1])
                if lo < 0.0 < hi and abs(lo + hi) > 1e-9:
                    norm = mcolors.TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)

            cf = ax.contourf(
                x,
                y,
                data,
                levels=levels,
                cmap=field["cmap_change"],
                norm=norm,
                extend=field.get("extend", "both"),
                transform=ccrs.PlateCarree(),
                zorder=1,
            )
            contour_handles[col_i] = cf

            # Stipple grid cells where the change is significant (BH-FDR
            # corrected by default; pass stipple_var="sig_raw" for the
            # uncorrected p<0.05 mask instead).
            sig_ds = region_sig[region["key"]]
            if sig_ds is not None:
                sig_var = f"{field['stem']}_{stipple_var}"
                if sig_var in sig_ds:
                    sig_mask = sig_ds[sig_var].values
                    ax.contourf(
                        x, y, sig_mask,
                        levels=[0.5, 1.5],
                        colors="none",
                        hatches=["..."],
                        transform=ccrs.PlateCarree(),
                        zorder=2,
                    )
                else:
                    print(f"[WARN] {sig_var!r} not found in significance file for "
                          f"{region['key']}; skipping stippling for this panel.")

            # Optional wind-speed change contours on the MSLP column only.
            if wind_contours and field["key"] == "mslp":
                wdata = anomaly_change(ens, "sfcWind", region["season"])
                cs = ax.contour(
                    x,
                    y,
                    wdata.values,
                    levels=WIND_CHG_LEVELS,
                    colors="0.15",
                    linewidths=0.75,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                )
                ax.clabel(cs, inline=True, fontsize=7, fmt="%g")

            add_region_marker(ax, region["marker_lon"], region["marker_lat"])
            add_panel_label(ax, f"({letters[row_i * n_cols + col_i]})")

            if row_i == 0:
                ax.set_title(field["title"], fontsize=18, fontweight="bold", pad=7)

            if col_i == 0:
                # rotation_mode="anchor" aligns the text before rotating it,
                # rather than aligning the bbox of the already-rotated text.
                # Without it, va="center" on multi-line rotated text sits high.
                # With it, ha acts along the text, i.e. vertically once rotated,
                # so ha="center" is what centres the label on the row and
                # va="bottom" sets how far it stands off from the panel edge.
                # multialignment centres the region name against the month
                # range, which otherwise sit flush along one edge and ragged
                # along the other.
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
                    0.02,
                    0.035,
                    ndays,
                    transform=ax.transAxes,
                    fontsize=9,
                    ha="left",
                    va="bottom",
                    zorder=8,
                    bbox=dict(facecolor="white", alpha=0.62, edgecolor="none", pad=1.6),
                )

    # One shared horizontal colourbar per column (field).
    for col_i, field in enumerate(FIELDS):
        # Hardcoded ticks suit the default levels only, so a CLI level
        # override falls back to matplotlib's automatic choice.
        ticks = None if field["key"] in level_overrides else field.get("cbar_ticks")
        cb = fig.colorbar(
            contour_handles[col_i],
            ax=axes[:, col_i].ravel().tolist(),
            orientation="horizontal",
            fraction=0.052,
            pad=0.045,
            aspect=28,
            ticks=None if ticks is None else clean_ticks(ticks),
        )
        if ticks is not None:
            cb.ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _pos: f"{v:g}")
            )
        cb.set_label(f"\u0394 {field['title']} anomaly ({field['units_change']})", fontsize=12, labelpad=4)
        cb.ax.tick_params(labelsize=11, length=3)

    n_members_set = sorted({n for _, n in region_data.values()})
    n_members_text = str(n_members_set[0]) if len(n_members_set) == 1 else "/".join(map(str, n_members_set))
    fig.suptitle(
        "Change in synoptic event anomaly, future minus historical",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    outpath = Path(figdir) / outfile
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for ens, _ in region_data.values():
        ens.close()
    for sig_ds in region_sig.values():
        if sig_ds is not None:
            sig_ds.close()

    stipple_label = "FDR-corrected p<0.05" if stipple_var == "sig_fdr" else "uncorrected p<0.05"
    hatching_note = (
        f" Hatching marks grid cells where the change is statistically significant "
        f"({stipple_label}, Welch's t-test pooled across the 8 members)."
        if stipple and any(v is not None for v in region_sig.values()) else ""
    )

    caption = (
        f"Change in the 95th percentile LCI synoptic event anomaly, future minus historical "
        f"(2040-2060 minus 1990-2014, SSP3-7.0, {n_members_text}-member NARCliM2 "
        f"ensemble mean). Columns show MSLP, temperature and precipitation; rows "
        f"show Cobar-Lachlan, Yilgarn-Coolgardie and Winton. Open circles mark the "
        f"region centroid; n values give the number of composited event-days in "
        f"the historical and future periods.{hatching_note}"
    )
    print("\n[Suggested figure caption]")
    print(caption)

    # Colourbar range diagnostic (trim the unused ends of the MSLP and
    # temperature bars). p1/p99 are the useful guide -- min/max are usually
    # set by a handful of grid cells and would keep the bar wider than it
    # needs to be.
    print("\n[Colourbar range check -- actual data vs plotted levels]")
    for field in FIELDS:
        chunks = observed[field["key"]]
        if not chunks:
            continue
        vals = np.concatenate(chunks)
        levels = level_overrides.get(field["key"], field["levels_change"])
        print(
            f"  {field['title']:14s} data min/max = {vals.min():+7.2f} / {vals.max():+7.2f}   "
            f"p1/p99 = {np.percentile(vals, 1):+7.2f} / {np.percentile(vals, 99):+7.2f}   "
            f"plotted = {float(levels[0]):+.2f} to {float(levels[-1]):+.2f}"
        )

    return outpath


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Make one compact 3x3 focal-region figure of future-minus-historical "
            "change in the top-5% LCI event anomaly, one row per variable."
        )
    )
    parser.add_argument("--composite-dir", default=DEFAULT_COMPOSITE_DIR)
    parser.add_argument("--figdir", default=DEFAULT_FIGDIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--outfile",
        default="synoptic_composites_change_focal_3x3_mid_century.png",
        help="Output filename, written inside --figdir.",
    )
    parser.add_argument(
        "--wind-contours",
        action="store_true",
        help="Overlay sfcWind anomaly-change contours on the MSLP row.",
    )
    parser.add_argument(
        "--sig-dir", default=DEFAULT_SIG_DIR,
        help="Directory with {region}_significance_mid_century.nc files "
             "from compute_synoptic_composite_significance.py.",
    )
    parser.add_argument(
        "--no-stipple", action="store_true",
        help="Skip significance stippling entirely (e.g. before the significance "
             "script has been run).",
    )
    parser.add_argument(
        "--stipple-var", choices=["sig_fdr", "sig_raw"], default="sig_fdr",
        help="Which significance mask to stipple with: BH-FDR-corrected (default) "
             "or raw uncorrected p<0.05.",
    )
    parser.add_argument(
        "--mslp-levels", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"),
        help="Override the MSLP colourbar range, e.g. --mslp-levels -3 3 0.25. "
             "Run once without it and read the printed colourbar range check "
             "to see what the data actually spans.",
    )
    parser.add_argument(
        "--tas-levels", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"),
        help="Override the temperature colourbar range, e.g. --tas-levels -1 1.5 0.125.",
    )
    args = parser.parse_args()

    level_overrides = {}
    for key, spec in (("mslp", args.mslp_levels), ("tas", args.tas_levels)):
        if spec is not None:
            lo, hi, step = spec
            if step <= 0 or hi <= lo:
                parser.error(f"--{key}-levels needs MIN < MAX and STEP > 0")
            # +step/2 so MAX itself is included despite float accumulation.
            level_overrides[key] = np.arange(lo, hi + step / 2, step)

    outpath = plot_change_compact(
        composite_dir=args.composite_dir,
        figdir=args.figdir,
        dpi=args.dpi,
        outfile=args.outfile,
        wind_contours=args.wind_contours,
        sig_dir=args.sig_dir,
        stipple=not args.no_stipple,
        stipple_var=args.stipple_var,
        level_overrides=level_overrides,
    )
    print(f"[OK] wrote {outpath}")


if __name__ == "__main__":
    main()