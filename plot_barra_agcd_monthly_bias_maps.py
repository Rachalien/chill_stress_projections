#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_barra_agcd_monthly_bias_maps.py

Load BARRA-R2 and AGCD+McVicar et al. (2008)-L monthly LCI climatologies, regrid AGCD onto
the BARRA grid, and produce monthly bias maps (BARRA minus AGCD) for three
LCI metrics.

Outputs (in FIGDIR)
-------------------
monthly_bias_lci_mean.png       -- 4x3 panels (Jan-Dec), daily mean LCI
monthly_bias_lci_max.png        -- 4x3 panels (Jan-Dec), monthly max daily LCI
monthly_bias_lci_freq_1000.png  -- 4x3 panels, exceedance frequency at 1000 kJ m-2 hr-1
monthly_bias_lci_freq_1100.png  -- 4x3 panels, exceedance frequency at 1100 kJ m-2 hr-1
monthly_bias_lci_freq_1200.png  -- 4x3 panels, exceedance frequency at 1200 kJ m-2 hr-1

Usage
-----
    python plot_barra_agcd_monthly_bias_maps.py [--var mean|max|freq] [--threshold 1000]

    If --var and --threshold are omitted, all five figures are produced.
"""

from __future__ import annotations

import os
import argparse
import glob
from typing import List, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# ============================================================
# CONFIG
# ============================================================

BIAS_DIR = "/scratch/dx2/rt9243/chill_projections/monthly_bias"

AGCD_CLIM_PATH  = os.path.join(BIAS_DIR, "agcd_lci_monthly_climatology_1979_2018.nc")
BARRA_CLIM_PATH = os.path.join(BIAS_DIR, "barra_lci_monthly_climatology_1979_2018.nc")
FIGDIR          = os.path.join(BIAS_DIR, "figures")

THRESHOLDS = [1000, 1100, 1200]
FREQ_VMAX = {1000: 5.0, 1100: 2.0, 1200: 1.0}
# Map extent (BARRA/McVicar et al. (2008) common domain)
LON_MIN, LON_MAX = 112.0, 154.0
LAT_MIN, LAT_MAX = -44.5, -9.5

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── Rain-gauge precipitation mask ────────────────────────────────────────────
# AGCD precipitation is interpolated from the rain-gauge network, and across
# central Australia the network is too sparse to constrain it. The ACCESS-S2
# calibration weights flag those cells. They coincide with the persistent
# positive-bias blob, so masking here removes an observational artefact rather
# than suppressing a genuine BARRA-R2 bias -- this must be stated in the
# caption, not applied silently.
PR_MASK_DIR = "/g/data/dx2/access-s2/calibration/OBS/grid_05/v4/pr_mask"
PR_MASK_THRESHOLD = 0.5   # keep cells whose interpolated gauge weight is >= this
GRID_TOLERANCE = 0.03     # degrees; nearest-neighbour match radius on a 0.05 deg grid


def load_pr_mask(lat: np.ndarray, lon: np.ndarray,
                 threshold: float = PR_MASK_THRESHOLD,
                 mask_dir: str = PR_MASK_DIR) -> xr.DataArray:
    """
    Return a boolean DataArray on the BARRA grid: True = retain the cell.

    Nearest-neighbour matching with an explicit tolerance is used rather than
    interpolation. Both grids are nominally 0.05 deg but their cell-centre
    offsets are not guaranteed to coincide, and interpolating a 0/1 weight
    field across an offset grid would produce fractional weights at every
    cell, silently softening the mask edges.
    """
    files = sorted(glob.glob(os.path.join(mask_dir, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {mask_dir}")
    if len(files) > 1:
        print(f"[WARN] {len(files)} files in mask dir; using "
              f"{os.path.basename(files[0])}. Check this is the intended one.")
    print(f"[INFO] Loading rain-gauge mask: {files[0]}")

    ds = xr.open_dataset(files[0], decode_timedelta=False)

    candidates = [v for v in ds.data_vars if {"lat", "lon"}.issubset(set(ds[v].dims))]
    if not candidates:
        raise KeyError(f"No lat/lon variable in {files[0]}; found {list(ds.data_vars)}")
    if len(candidates) > 1:
        print(f"[WARN] multiple mask candidates {candidates}; using {candidates[0]!r}")
    da = ds[candidates[0]]

    for extra in [d for d in da.dims if d not in ("lat", "lon")]:
        print(f"[INFO] mask has extra dim {extra!r} (size {da.sizes[extra]}); taking index 0")
        da = da.isel({extra: 0})

    if float(da.lat[0]) > float(da.lat[-1]):
        da = da.sortby("lat")
    if float(da.lon[0]) > float(da.lon[-1]):
        da = da.sortby("lon")

    vals = np.unique(da.values[np.isfinite(da.values)])
    print(f"[INFO] mask weights: {len(vals)} distinct value(s), "
          f"range {vals.min():.3f} to {vals.max():.3f}")
    if len(vals) <= 12:
        print(f"[INFO] weight values: {np.round(vals, 3)}")
    else:
        n_mid = int(((da.values > 0) & (da.values < 1)).sum())
        print(f"[INFO] {n_mid} cells hold intermediate weights strictly between 0 and 1; "
              f"threshold {threshold} decides these")

    print(f"[INFO] mask grid : lat {float(da.lat[0]):.4f} to {float(da.lat[-1]):.4f} "
          f"step {float(da.lat[1] - da.lat[0]):.4f} | "
          f"lon {float(da.lon[0]):.4f} to {float(da.lon[-1]):.4f} "
          f"step {float(da.lon[1] - da.lon[0]):.4f}")
    print(f"[INFO] BARRA grid: lat {lat[0]:.4f} to {lat[-1]:.4f} "
          f"step {lat[1] - lat[0]:.4f} | "
          f"lon {lon[0]:.4f} to {lon[-1]:.4f} step {lon[1] - lon[0]:.4f}")

    on_barra = da.reindex(lat=lat, lon=lon, method="nearest", tolerance=GRID_TOLERANCE)

    n_unmatched = int(np.isnan(on_barra.values).sum())
    total = on_barra.size
    print(f"[INFO] {n_unmatched}/{total} BARRA cells ({100 * n_unmatched / total:.1f}%) "
          f"found no mask cell within {GRID_TOLERANCE} deg")
    if n_unmatched > 0.5 * total:
        raise RuntimeError(
            "More than half the BARRA cells failed to match a mask cell. The two "
            "grids are probably offset by more than the tolerance -- inspect the "
            "printed grid descriptions above before raising GRID_TOLERANCE, since "
            "a loose tolerance will shift the mask by up to a cell."
        )

    keep = on_barra >= threshold   # unmatched cells are NaN, so compare False and are dropped
    print(f"[INFO] mask retains {int(keep.values.sum())}/{total} cells "
          f"({100 * int(keep.values.sum()) / total:.1f}%)")
    return keep
    
# ============================================================
# DATA LOADING AND REGRIDDING
# ============================================================

def load_climatologies() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Load BARRA and AGCD climatologies, regrid AGCD onto the BARRA grid,
    and return (ds_barra, ds_agcd_regridded, ds_bias).

    Bias = BARRA - AGCD_regridded.
    """
    print("[INFO] Loading BARRA climatology ...")
    ds_barra = xr.open_dataset(BARRA_CLIM_PATH, decode_timedelta=False)

    print("[INFO] Loading AGCD climatology ...")
    ds_agcd  = xr.open_dataset(AGCD_CLIM_PATH,  decode_timedelta=False)

    # Ensure AGCD lat/lon are ascending for interp
    if float(ds_agcd.lat[0]) > float(ds_agcd.lat[-1]):
        ds_agcd = ds_agcd.sortby("lat")
    if float(ds_agcd.lon[0]) > float(ds_agcd.lon[-1]):
        ds_agcd = ds_agcd.sortby("lon")

    # Ensure BARRA lat/lon are ascending
    if float(ds_barra.lat[0]) > float(ds_barra.lat[-1]):
        ds_barra = ds_barra.sortby("lat")
    if float(ds_barra.lon[0]) > float(ds_barra.lon[-1]):
        ds_barra = ds_barra.sortby("lon")

    print("[INFO] Regridding AGCD to BARRA grid ...")
    # interp_like does bilinear interpolation; fill_value=np.nan outside AGCD domain
    ds_agcd_regrid = ds_agcd.interp(
        lat=ds_barra.lat,
        lon=ds_barra.lon,
        method="linear",
        kwargs={"fill_value": np.nan},
    )

    # Compute bias dataset
    print("[INFO] Computing bias (BARRA - AGCD) ...")
    bias_vars = {}
    for var in ("lci_mean", "lci_max"):
        if var in ds_barra and var in ds_agcd_regrid:
            bias_vars[var] = ds_barra[var] - ds_agcd_regrid[var]

    if "lci_freq" in ds_barra and "lci_freq" in ds_agcd_regrid:
        bias_vars["lci_freq"] = ds_barra["lci_freq"] - ds_agcd_regrid["lci_freq"]

    ds_bias = xr.Dataset(bias_vars, coords=ds_barra.coords)

    return ds_barra, ds_agcd_regrid, ds_bias


# ============================================================
# PLOTTING HELPERS
# ============================================================

def _symmetric_vlim(data: np.ndarray, pct: float = 97.0) -> float:
    """Return a symmetric colorbar limit based on the given percentile of abs values."""
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return 1.0
    return float(np.percentile(np.abs(finite), pct))


def _setup_ax(ax) -> None:
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color="0.2")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3,  edgecolor="0.4",  facecolor="none")
    ax.add_feature(cfeature.STATES,    linewidth=0.25, edgecolor="0.55", facecolor="none")
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=0.25, color="grey", alpha=0.4, linestyle="--")
    gl.xlocator = mticker.FixedLocator([120, 130, 140, 150])
    gl.ylocator = mticker.FixedLocator([-40, -30, -20])


def plot_monthly_bias(
    bias_field: xr.DataArray,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str,
    cbar_label: str,
    outpath: str,
    cmap: str = "RdBu",
    vmax: Optional[float] = None,
    cbar_extend: str = "both",
) -> None:
    """
    Plot 12 monthly bias panels (4 rows × 3 columns, Jan-Dec).

    bias_field : DataArray with dims (month, lat, lon), month 1-indexed.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    proj = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        4, 3,
        figsize=(13, 14),
        subplot_kw={"projection": proj},
        gridspec_kw={"hspace": 0.06, "wspace": 0.04},
    )

    bias_np = bias_field.values   # (12, nlat, nlon)

    if vmax is None:
        vmax = _symmetric_vlim(bias_np, pct=97.0)
    vmin = -vmax

    X, Y = np.meshgrid(lon, lat)
    transform = ccrs.PlateCarree()
    cf_last = None

    for idx in range(12):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        _setup_ax(ax)

        data = bias_np[idx]   # month is 1-indexed, idx is 0-indexed

        data_masked = np.ma.masked_invalid(data)
        cf = ax.contourf(
            X, Y, data_masked,
            levels=np.linspace(vmin, vmax, 21),
            cmap=cmap,
            extend="both",
            transform=ccrs.PlateCarree(),
        )
        ax.contour(
            X, Y, data_masked,
            levels=[0], colors="k", linewidths=0.5,
            transform=ccrs.PlateCarree(),
        )

        ax.set_title(MONTH_NAMES[idx], fontsize=12, pad=3, fontweight="bold")

        # Row and column labels
        if col == 0:
            ax.set_ylabel("")

        cf_last = cf

    # Shared colourbar
    fig.suptitle(title, fontsize=15, y=1.00)
    fig.tight_layout(rect=[0, 0, 0.90, 0.98])

    # Enlarged: the bar itself was thin (fraction 0.018) and short
    # (shrink 0.5) as well as small-labelled, so widening and lengthening it
    # matters as much as the font sizes.
    cbar = fig.colorbar(
        cf_last,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        fraction=0.028,
        pad=0.02,
        shrink=0.75,
    )
    cbar.set_label(cbar_label, fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] wrote {outpath}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot BARRA-R2 minus AGCD+McVicar et al. (2008) monthly LCI bias maps."
    )
    ap.add_argument("--var",       choices=["mean", "max", "freq"], default=None,
                    help="Which LCI variable to plot (default: all)")
    ap.add_argument("--threshold", type=int, choices=THRESHOLDS, default=None,
                    help="Threshold for freq plots (default: all three)")
    ap.add_argument("--no-mask", action="store_true",
                    help="Skip the rain-gauge precipitation mask (diagnostic use).")
    ap.add_argument("--mask-threshold", type=float, default=PR_MASK_THRESHOLD,
                    help="Retain cells whose interpolated gauge weight is >= this.")
    args = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)

    ds_barra, ds_agcd, ds_bias = load_climatologies()
    lat = ds_barra.lat.values
    lon = ds_barra.lon.values

    if not args.no_mask:
        def _n_valid(ds: xr.Dataset) -> int:
            ref = ds[list(ds.data_vars)[0]]
            while ref.ndim > 2:
                ref = ref.isel({ref.dims[0]: 0})
            return int(np.isfinite(ref.values).sum())

        n_before = _n_valid(ds_bias)
        keep = load_pr_mask(lat, lon, threshold=args.mask_threshold)
        ds_bias = ds_bias.where(keep)
        n_after = _n_valid(ds_bias)
        print(f"[INFO] valid bias cells: {n_before} -> {n_after} "
              f"({n_before - n_after} removed, {100 * (n_before - n_after) / n_before:.1f}%)")
    else:
        print("[INFO] --no-mask given; rain-gauge mask NOT applied")
        
    do_vars = [args.var] if args.var else ["mean", "max", "freq"]
    do_thrs = [args.threshold] if args.threshold else THRESHOLDS

    # ── lci_mean bias ─────────────────────────────────────────────────────────
    if "mean" in do_vars and "lci_mean" in ds_bias:
        print("[INFO] Plotting lci_mean bias ...")
        plot_monthly_bias(
            bias_field = ds_bias["lci_mean"],
            lat        = lat,
            lon        = lon,
            vmax       = 60.0,
            title      = (
                "BARRA-R2 minus AGCD+McVicar et al. (2008): monthly mean daily LCI bias\n"
                "(kJ m$^{-2}$ hr$^{-1}$), climatology 1979\u20132018"
            ),
            cbar_label = r"$\Delta$LCI (kJ m$^{-2}$ hr$^{-1}$)",
            outpath    = os.path.join(FIGDIR, "monthly_bias_lci_mean.png"),
            cmap       = "RdBu",
        )

    # ── lci_max bias ──────────────────────────────────────────────────────────
    if "max" in do_vars and "lci_max" in ds_bias:
        print("[INFO] Plotting lci_max bias ...")
        plot_monthly_bias(
            bias_field = ds_bias["lci_max"],
            lat        = lat,
            lon        = lon,
            vmax       = 100.0,
            title      = (
                "BARRA-R2 minus AGCD+McVicar et al. (2008): monthly maximum daily LCI bias\n"
                "(kJ m$^{-2}$ hr$^{-1}$), climatology 1979\u20132018"
            ),
            cbar_label = r"$\Delta$LCI$_\mathrm{max}$ (kJ m$^{-2}$ hr$^{-1}$)",
            outpath    = os.path.join(FIGDIR, "monthly_bias_lci_max.png"),
            cmap       = "RdBu",
        )

    # ── lci_freq bias (one figure per threshold) ──────────────────────────────
    if "freq" in do_vars and "lci_freq" in ds_bias:
        for thr in do_thrs:
            print(f"[INFO] Plotting lci_freq bias at threshold {thr} ...")
            freq_bias = ds_bias["lci_freq"].sel(threshold=float(thr))
            plot_monthly_bias(
                bias_field = freq_bias,
                lat        = lat,
                lon        = lon,
                vmax       = FREQ_VMAX[thr],
                title      = (
                    f"BARRA-R2 minus AGCD+McVicar et al. (2008): monthly exceedance frequency bias\n"
                    f"threshold {thr} kJ m\u207b\u00b2 hr\u207b\u00b9"
                    r", climatology 1979$-$2018"
                ),
                cbar_label = r"$\Delta$Frequency (days month$^{-1}$)",
                outpath    = os.path.join(FIGDIR, f"monthly_bias_lci_freq_{thr}.png"),
                cmap       = "PuOr_r",
            )

    print(f"\n[DONE] Figures written to {FIGDIR}")


if __name__ == "__main__":
    main()