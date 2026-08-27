#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_lci_composite_focal_3x3_mid_century.py

Three-by-three figure of the top-5% LCI event composite, in Livestock Chill
Index units, for the three focal regions.

Figure layout
-------------
Rows:    Cobar-Lachlan May-Oct | Yilgarn-Coolgardie May-Oct | Winton Nov-Mar
Columns: Historical (1990-2014) | Mid-century (2040-2060, SSP3-7.0) | Change

Columns 1 and 2 show absolute event-day mean LCI and share a single colour
scale, so the two periods are directly comparable by eye. Each region's chill
threshold (Cobar-Lachlan 1100, Yilgarn-Coolgardie 1200, Winton 1000
kJ m-2 hr-1) is drawn as a labelled contour on those two panels. Column 3 is
the difference, mid-century minus historical, on its own diverging scale.

Sign convention in column 3: negative means less severe cold stress in the
future. The default colormap is RdBu (not RdBu_r), so declining chill reads
warm/red and intensifying chill reads blue, which matches physical intuition
for a warming signal. Pass --diff-cmap RdBu_r to flip it.

What the difference panel does and does not show
------------------------------------------------
Event days are defined by the p95 of the regional-mean LCI computed
independently within each period, so column 3 is the change in the SEVERITY of
each period's own worst 5% of days. It is not the change in how often a fixed
chill threshold is crossed; the frequency signal lives in the other figures.
The composite is also conditioned on that row's region: event days are selected
on the regional-mean LCI of the marked region, so values far from the marker
describe the wider field on those days, not that location's own worst days.

Inputs
------
Per-member files from lci_grid_composite_top5pct.py:
    {COMPOSITE_DIR}/{member}_lci_composite_{region}_mid_century.nc
Expected variables per region:
    lci_dry_hist, lci_dry_future   (Cobar-Lachlan, Yilgarn-Coolgardie)
    lci_wet_hist, lci_wet_future   (Winton)

Usage
-----
    module use /g/data/xp65/public/modules
    module load conda/analysis3

    python plot_lci_composite_focal_3x3_mid_century.py
    python plot_lci_composite_focal_3x3_mid_century.py --print-range
    python plot_lci_composite_focal_3x3_mid_century.py \
        --vmin 700 --vmax 1200 --vstep 20 --dmax 40 --dstep 5
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
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader, natural_earth
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import shapely
import shapely.vectorized
from shapely.ops import unary_union

SCRIPT_NAME = os.path.basename(__file__)

# -- Defaults ------------------------------------------------------------------

DEFAULT_COMPOSITE_DIR = "/scratch/dx2/rt9243/chill_projections/lci_grid_composites"
DEFAULT_FIGDIR = "/scratch/dx2/rt9243/chill_projections/lci_grid_composites/figures"

REGIONS = [
    {
        "key": "Cobar_Lachlan",
        "row_label": "Cobar-Lachlan\nMay-Oct",
        "season": "dry",
        "threshold": 1100.0,
        "marker_lon": 146.0,
        "marker_lat": -32.5,
    },
    {
        "key": "Yilgarn_Coolgardie",
        "row_label": "Yilgarn-Coolgardie\nMay-Oct",
        "season": "dry",
        "threshold": 1200.0,
        "marker_lon": 120.0,
        "marker_lat": -30.5,
    },
    {
        "key": "Winton",
        "row_label": "Winton\nNov-Mar",
        "season": "wet",
        "threshold": 1000.0,
        "marker_lon": 143.0,
        "marker_lat": -22.5,
    },
]

COLUMN_TITLES = ["Historical", "Mid-century", "Change"]

SEASON_TEXT = {"dry": "May-October", "wet": "November-March"}

LON_MIN, LON_MAX = 112.0, 154.0
LAT_MIN, LAT_MAX = -44.5, -9.5


# -- Data helpers --------------------------------------------------------------

def _open_dataset_loaded(path: str) -> xr.Dataset:
    """Open a small composite file and load it so file handles can close."""
    try:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            return ds.load()
    except Exception:
        with xr.open_dataset(path) as ds:
            return ds.load()


def load_ensemble_mean(region_key: str, composite_dir: str) -> Tuple[xr.Dataset, int, List[str]]:
    """
    Load all member files for one region and return the ensemble mean.

    Members are weighted equally regardless of their individual event-day
    counts, matching the convention used by the synoptic composite figures.
    """
    pattern = os.path.join(composite_dir, f"*_lci_composite_{region_key}_mid_century.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No LCI composite files matched: {pattern}")

    datasets: List[xr.Dataset] = []
    members: List[str] = []
    for path in files:
        datasets.append(_open_dataset_loaded(path))
        members.append(os.path.basename(path).split("_lci_composite_")[0])

    # Capture per-member n_days before the mean collapses the member dimension.
    # xr.concat keeps only the first dataset's attrs, so reading n_days off the
    # ensemble mean would silently report one member's count as if it were the
    # ensemble total (the bug fixed in the synoptic composite figures).
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
    ens.attrs["member_labels"] = members

    for ds in datasets:
        ds.close()
    stacked.close()

    return ens, len(files), files


def get_field(ens: xr.Dataset, season: str, period: str) -> xr.DataArray:
    var = f"lci_{season}_{period}"
    if var not in ens:
        raise KeyError(
            f"Missing variable {var!r}. Available: {sorted(ens.data_vars)[:12]}. "
            f"Run lci_grid_composite_top5pct.py for this region first."
        )
    return ens[var]


def n_days_text(ens: xr.Dataset, season: str) -> str:
    """Total composited event-days summed across members, per period."""
    per_member = ens.attrs.get("n_days_per_member", {})
    parts = []
    for period, tag in (("hist", "hist"), ("future", "fut")):
        vals = [v for v in per_member.get(f"lci_{season}_{period}", []) if v is not None]
        if vals:
            parts.append(f"n_{tag}={sum(vals)}")
    return ", ".join(parts)


def print_diagnostics(region_key: str, season: str, ens: xr.Dataset) -> None:
    """Per-member day counts and p95 thresholds, so an uneven split is visible."""
    print(f"\n[diagnostics] {region_key} ({season})")
    per_member = ens.attrs.get("n_days_per_member", {})
    for var, vals in per_member.items():
        clean = [v for v in vals if v is not None]
        if clean:
            print(f"  n_days {var:22s} {clean}  total={sum(clean)}")
    for period in ("hist", "future"):
        pvar = f"p95_{season}_{period}"
        if pvar in ens:
            print(f"  ensemble-mean {pvar} = {float(ens[pvar].values):.1f} kJ m-2 hr-1")


# -- Plot helpers --------------------------------------------------------------

def setup_map(ax, row_i: int, col_i: int, n_rows: int, mask_ocean: bool) -> None:
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    if mask_ocean:
        # LCI is a livestock exposure index; over-ocean values are not
        # meaningful, so the ocean is painted over the filled contours rather
        # than left to dominate the colour scale visually.
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.85, edgecolor="0.12", zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.45, edgecolor="0.35", zorder=4)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="0.45", zorder=4)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.25,
        color="grey", alpha=0.45, linestyle="--",
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
        lon, lat, marker="o", markersize=10,
        markerfacecolor="none", markeredgecolor="red", markeredgewidth=1.6,
        transform=ccrs.PlateCarree(), zorder=7,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
    )


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.025, 0.965, label, transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="top", ha="left", zorder=8,
        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none",
                  boxstyle="round,pad=0.20"),
    )


_LAND_GEOM_CACHE: Dict[str, object] = {}
_LAND_MASK_CACHE: Dict[Tuple[bytes, bytes], np.ndarray] = {}


def _land_geometry(resolution: str = "110m"):
    """Union of Natural Earth land polygons, cached.

    Resolution matches cfeature.OCEAN/COASTLINE's default (110m) so the land
    mask lines up with the coastline already drawn by setup_map.
    """
    if resolution not in _LAND_GEOM_CACHE:
        path = natural_earth(resolution=resolution, category="physical", name="land")
        geoms = list(Reader(path).geometries())
        _LAND_GEOM_CACHE[resolution] = unary_union(geoms)
    return _LAND_GEOM_CACHE[resolution]


def land_mask(x: np.ndarray, y: np.ndarray, resolution: str = "110m") -> np.ndarray:
    """Boolean array, True where the (x, y) grid point sits over land.

    Cached by grid contents, since all three regions share the same
    domain-wide lat/lon grid and this is only worth computing once.
    """
    key = (x.tobytes(), y.tobytes())
    if key not in _LAND_MASK_CACHE:
        land = _land_geometry(resolution)
        if hasattr(shapely, "contains_xy"):
            mask = shapely.contains_xy(land, x, y)          # shapely >= 2.0
        else:
            mask = shapely.vectorized.contains(land, x, y)  # shapely < 2.0
        _LAND_MASK_CACHE[key] = mask
    return _LAND_MASK_CACHE[key]


def add_threshold_contour(ax, x, y, data, threshold: float, label_contours: bool,
                           land: np.ndarray):
    """Draw this region's chill threshold as a white contour, land only.

    The field is masked to land before contouring, rather than just setting
    the line colour to white, so the contour -- and any inline label -- can
    never fall over the ocean regardless of the fill colormap in use. A thin
    dark halo keeps the white line legible against light fill colours.
    """
    land_data = np.ma.masked_where(~land, data)
    halo = [pe.withStroke(linewidth=1.6, foreground="0.15")]
    cs = ax.contour(
        x, y, land_data, levels=[threshold], colors="white", linewidths=0.9,
        transform=ccrs.PlateCarree(), zorder=5,
    )
    cs.set_path_effects(halo)
    if label_contours:
        labels = ax.clabel(cs, inline=True, fontsize=7, fmt=lambda v: f"{v:.0f}",
                           colors="white")
        for lab in labels:
            lab.set_path_effects(halo)
    return cs


# -- Main figure ---------------------------------------------------------------

def plot_lci_composite_3x3(
        composite_dir: str,
        figdir: str,
        dpi: int,
        outfile: str,
        vmin: float,
        vmax: float,
        vstep: float,
        dmax: float,
        dstep: float,
        abs_cmap: str,
        diff_cmap: str,
        mask_ocean: bool,
        label_contours: bool,
        print_range: bool,
    ) -> Path:
    os.makedirs(figdir, exist_ok=True)
    proj = ccrs.PlateCarree()
    n_rows, n_cols = len(REGIONS), 3

    abs_levels = np.arange(vmin, vmax + vstep, vstep)
    diff_levels = np.arange(-dmax, dmax + dstep, dstep)

    region_data: Dict[str, Tuple[xr.Dataset, int]] = {}
    for spec in REGIONS:
        ens, n_members, _ = load_ensemble_mean(spec["key"], composite_dir)
        region_data[spec["key"]] = (ens, n_members)
        print_diagnostics(spec["key"], spec["season"], ens)

    if print_range:
        print("\n[data ranges over the plotted domain]")
        for spec in REGIONS:
            ens, _ = region_data[spec["key"]]
            hist = get_field(ens, spec["season"], "hist")
            fut = get_field(ens, spec["season"], "future")
            diff = fut - hist
            for name, da in (("hist", hist), ("future", fut), ("change", diff)):
                v = da.values
                print(f"  {spec['key']:20s} {name:7s} "
                      f"min={np.nanmin(v):8.1f}  max={np.nanmax(v):8.1f}  "
                      f"p2={np.nanpercentile(v, 2):8.1f}  p98={np.nanpercentile(v, 98):8.1f}")
        print("  (use these to set --vmin/--vmax/--dmax if the defaults clip badly)\n")

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(12.2, 9.2),
        subplot_kw={"projection": proj},
        gridspec_kw={"hspace": 0.06, "wspace": 0.035},
    )

    letters = "abcdefghi"
    abs_handle = None
    diff_handle = None

    for row_i, region in enumerate(REGIONS):
        ens, _ = region_data[region["key"]]
        lat = ens["lat"].values
        lon = ens["lon"].values
        x, y = np.meshgrid(lon, lat)
        land = land_mask(x, y)

        hist = get_field(ens, region["season"], "hist")
        fut = get_field(ens, region["season"], "future")
        panels = [hist, fut, fut - hist]

        for col_i, da in enumerate(panels):
            ax = axes[row_i, col_i]
            setup_map(ax, row_i, col_i, n_rows, mask_ocean)
            data = np.ma.masked_invalid(da.values)

            if col_i < 2:
                cf = ax.contourf(
                    x, y, data, levels=abs_levels, cmap=abs_cmap,
                    extend="both", transform=ccrs.PlateCarree(), zorder=1,
                )
                abs_handle = cf
                add_threshold_contour(ax, x, y, data, region["threshold"],
                                      label_contours, land)
            else:
                cf = ax.contourf(
                    x, y, data, levels=diff_levels, cmap=diff_cmap,
                    extend="both", transform=ccrs.PlateCarree(), zorder=1,
                )
                diff_handle = cf

            add_region_marker(ax, region["marker_lon"], region["marker_lat"])
            add_panel_label(ax, f"({letters[row_i * n_cols + col_i]})")

            if row_i == 0:
                ax.set_title(COLUMN_TITLES[col_i], fontsize=18,
                             fontweight="bold", pad=7)

            if col_i == 0:
                ax.text(
                    -0.20, 0.5, region["row_label"], transform=ax.transAxes,
                    fontsize=15, fontweight="bold", rotation=90,
                    rotation_mode="anchor", ha="center", va="bottom",
                    multialignment="center",
                )

            ndays = n_days_text(ens, region["season"])
            if ndays and col_i == 0:
                ax.text(
                    0.02, 0.035, ndays, transform=ax.transAxes, fontsize=9,
                    ha="left", va="bottom", zorder=8,
                    bbox=dict(facecolor="white", alpha=0.62, edgecolor="none",
                              pad=1.6),
                )

    # One wide colourbar spanning the two absolute columns, one for the change.
    cb_abs = fig.colorbar(
        abs_handle, ax=axes[:, :2].ravel().tolist(), orientation="horizontal",
        fraction=0.052, pad=0.045, aspect=42,
    )
    cb_abs.set_label("LCI (kJ m$^{-2}$ hr$^{-1}$)",
                     fontsize=12, labelpad=4)
    cb_abs.ax.tick_params(labelsize=11, length=3)

    cb_diff = fig.colorbar(
        diff_handle, ax=axes[:, 2].ravel().tolist(), orientation="horizontal",
        fraction=0.052, pad=0.045, aspect=20,
    )
    cb_diff.set_label("Change in LCI (kJ m$^{-2}$ hr$^{-1}$)",
                      fontsize=12, labelpad=4)
    cb_diff.ax.tick_params(labelsize=11, length=3)

    n_members_set = sorted({n for _, n in region_data.values()})
    n_members_text = (str(n_members_set[0]) if len(n_members_set) == 1
                      else "/".join(map(str, n_members_set)))

    fig.suptitle(
        "Average historical and future 95th percentile LCI and change",
        fontsize=17, fontweight="bold", y=1.01,
    )

    outpath = Path(figdir) / outfile
    metadata = {
        "Title": ("Top-5% LCI event composites in LCI units for three focal "
                  "regions: historical, mid-century and their difference"),
        "Description": (
            f"Created by {SCRIPT_NAME}; "
            f"input={composite_dir}/*_lci_composite_*_mid_century.nc; "
            f"years=1990-2014 vs 2040-2060 SSP3-7.0; "
            f"members={n_members_text}; "
            f"figure=lci_composite_focal_3x3_mid_century"
        ),
        "Software": SCRIPT_NAME,
    }
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white",
                metadata=metadata)
    plt.close(fig)

    for ens, _ in region_data.values():
        ens.close()

    caption = (
        f"Composite Livestock Chill Index on the top 5% of chill event days for "
        f"the three focal regions ({n_members_text}-member NARCliM2 ensemble "
        f"mean). Rows show Cobar-Lachlan (May-October), Yilgarn-Coolgardie "
        f"(May-October) and Winton (November-March). Columns show the historical "
        f"period (1990-2014), the mid-century period (2040-2060, SSP3-7.0), and "
        f"the difference between them. Event days are the days on which the "
        f"area-averaged Livestock Chill Index for the marked region exceeds its "
        f"95th percentile, determined separately within each period, so the "
        f"difference panel describes the change in severity of each period's own "
        f"most severe 5% of days rather than a change in how often a fixed "
        f"threshold is crossed. Values are the average of daily gridded index "
        f"values across those days. The white contour, shown over land only, "
        f"marks each region's chill threshold (Cobar-Lachlan 1100, "
        f"Yilgarn-Coolgardie 1200, Winton 1000 "
        f"kJ m-2 hr-1). Open circles mark the region centroid; n values give the "
        f"total number of composited event days across all ensemble members."
    )
    print("\n[Suggested figure caption -- detail belongs here, not on the figure]")
    print(caption)

    return outpath


# -- CLI -----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="3x3 figure of top-5% LCI event composites in LCI units: "
                    "historical, mid-century and change, for three focal regions."
    )
    p.add_argument("--composite-dir", default=DEFAULT_COMPOSITE_DIR)
    p.add_argument("--figdir", default=DEFAULT_FIGDIR)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--outfile", default="lci_composite_focal_3x3_mid_century.png")
    p.add_argument("--vmin", type=float, default=700.0,
                   help="Lower bound of the absolute LCI colour scale.")
    p.add_argument("--vmax", type=float, default=1200.0,
                   help="Upper bound of the absolute LCI colour scale.")
    p.add_argument("--vstep", type=float, default=50.0)
    p.add_argument("--dmax", type=float, default=40.0,
                   help="Symmetric bound of the change colour scale.")
    p.add_argument("--dstep", type=float, default=5.0)
    p.add_argument("--abs-cmap", default="RdYlGn_r",
                   help="Colormap for the two absolute columns.")
    p.add_argument("--diff-cmap", default="RdBu",
                   help="Colormap for the change column. RdBu makes a decline "
                        "in chill read red; RdBu_r flips it.")
    p.add_argument("--no-mask-ocean", action="store_true",
                   help="Keep over-ocean LCI values visible.")
    p.add_argument("--no-contour-labels", action="store_true",
                   help="Draw the threshold contour without inline labels.")
    p.add_argument("--print-range", action="store_true",
                   help="Print min/max/p2/p98 of every panel before plotting, to "
                        "help set the colour scales.")
    args = p.parse_args()

    outpath = plot_lci_composite_3x3(
        composite_dir=args.composite_dir,
        figdir=args.figdir,
        dpi=args.dpi,
        outfile=args.outfile,
        vmin=args.vmin,
        vmax=args.vmax,
        vstep=args.vstep,
        dmax=args.dmax,
        dstep=args.dstep,
        abs_cmap=args.abs_cmap,
        diff_cmap=args.diff_cmap,
        mask_ocean=not args.no_mask_ocean,
        label_contours=not args.no_contour_labels,
        print_range=args.print_range,
    )
    print(f"[OK] wrote {outpath}")


if __name__ == "__main__":
    main()