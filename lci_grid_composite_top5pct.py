#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lci_grid_composite_top5pct.py

Compute GRIDDED composites of the Livestock Chill Index itself on top-5% LCI
event days, for one NARCliM2 ensemble member.

Why this script exists
----------------------
synoptic_composite_lci_top5pct.py composites the four synoptic driver fields
(psl, sfcWind, pr, tas) on event days, but never writes LCI itself as a grid.
LCI cannot be recovered from those composited fields, because it is nonlinear
in wind (sqrt), rain (exponential) and the temperature-wind cross term:

    mean_over_days[ LCI(T, V, R) ]  !=  LCI( mean T, mean V, mean R )

The size of that gap is exactly the "nonlinear" term already quantified in the
Shapley attribution (23% of the total dLCI at Winton). So the event-day mean of
LCI has to be formed from daily gridded LCI, which is what this script does.

Key difference from synoptic_composite_lci_top5pct.py
-----------------------------------------------------
This script does NOT repeat Pass 1 (the full-period regional daily LCI series).
That series was already computed and written to disk by the mid-century run:

    {COMPOSITE_DIR}/daily_lci/{model}_regional_daily_lci.csv

Event days are re-derived from that CSV using the pipeline's own
find_top_pct_events(), so the event dates reproduce the existing composites
exactly, member for member, without re-reading the full ~170 GB of driver data.
Only the event-day slices are then read from disk. Use --recompute-series if the
CSV is missing (falls back to the full Pass 1, much slower).

Event definition
----------------
Unchanged from the pipeline: p95 of the regional-mean daily LCI distribution,
computed independently within each (region, season, period). The mid-century
composite therefore samples the worst 5% of mid-century days, not the days
exceeding a fixed historical threshold. The future-minus-historical difference
is a change in the SEVERITY of each period's own worst 5%, not a change in the
frequency of exceeding a fixed level.

Only each region's focal season is computed by default:
    Cobar-Lachlan       dry (May-Oct)
    Yilgarn-Coolgardie  dry (May-Oct)
    Winton              wet (Nov-Mar, April excluded)
Pass --all-seasons to compute both seasons for every region.

Output
------
One NetCDF per region per model:
    {OUTDIR}/{model_label}_lci_composite_{region}_mid_century.nc

Variables (float32, lat x lon), for each season/period actually computed:
    lci_{season}_{period}        event-day mean LCI (kJ m-2 hr-1)
    lci_{season}_{period}_std    sample std (ddof=1) across event days
    lci_{season}_{period}_clim   seasonal-mean LCI over ALL season days
                                 (only with --with-climatology)
Scalar variables:
    p95_{season}_{period}        this member's regional-mean p95 threshold
Attributes on each grid variable: n_days
Global attributes: model, region, top_pct, hist_period, future_period, scenario

Usage
-----
    module use /g/data/xp65/public/modules
    module load conda/analysis3

    python lci_grid_composite_top5pct.py --check-only
    python lci_grid_composite_top5pct.py --model ACCESS-ESM1-5_WRF412R3

PBS
---
See submit_lci_grid_composites.sh (array job, one task per member).
Much lighter than the parent pipeline: ncpus=4, mem=32gb, walltime=02:00:00 is
ample without --with-climatology.

Must sit in the same directory as synoptic_composite_lci_top5pct.py (or have it
on PYTHONPATH), from which all loaders, masks, season logic and the LCI formula
are imported rather than duplicated.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from synoptic_composite_lci_top5pct import (
    MEMBERS,
    REGION_SPECS,
    OUTDIR as COMPOSITE_DIR,
    HIST_LOAD_YEARS,
    FUTURE_LOAD_YEARS,
    HIST_SEASON_RANGE,
    FUTURE_SEASON_RANGE,
    TOP_PCT,
    calculate_lci,
    open_var_lazy,
    time_strings,
    build_region_mask,
    annotate_seasons,
    find_top_pct_events,
    compute_lci_regional_series,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
xr.set_options(keep_attrs=True)


# ── CONFIG ────────────────────────────────────────────────────────────────────

OUTDIR = "/scratch/dx2/rt9243/chill_projections/lci_grid_composites"

DAILY_LCI_DIR = os.path.join(COMPOSITE_DIR, "daily_lci")

# Variables needed to build daily gridded LCI. Note pslAdjust is deliberately
# NOT loaded here -- it plays no part in the LCI formula, and skipping it is
# most of the I/O saving relative to the parent pipeline.
LCI_VARS = ["tasmaxAdjust", "tasminAdjust", "sfcWindAdjust", "prAdjust"]

# Each region's focal season, matching the three rows of the paper's figures.
FOCAL_SEASON = {
    "Cobar_Lachlan": "dry",
    "Yilgarn_Coolgardie": "dry",
    "Winton": "wet",
}

PERIODS = [("hist", "historical", HIST_LOAD_YEARS, HIST_SEASON_RANGE),
           ("future", "ssp370", FUTURE_LOAD_YEARS, FUTURE_SEASON_RANGE)]


# ── REGIONAL DAILY LCI SERIES ────────────────────────────────────────────────

def load_daily_lci_series(model_label, model, variant, rcm, region_masks,
                          lat, lon, recompute=False):
    """
    Return the tidy daily regional LCI DataFrame used for p95 thresholding.

    Preferred path: read the CSV already written by the mid-century run of
    synoptic_composite_lci_top5pct.py, so the event dates derived here are
    identical to those behind the existing synoptic composites. Fall back to
    recomputing Pass 1 only if asked, or if the CSV is absent.
    """
    csv_path = os.path.join(DAILY_LCI_DIR, f"{model_label}_regional_daily_lci.csv")

    if not recompute and os.path.exists(csv_path):
        log.info(f"Reading cached daily regional LCI series: {csv_path}")
        df = pd.read_csv(csv_path, dtype={"date": str})
        _validate_series(df, csv_path)
        return df

    if not recompute:
        log.warning(f"No cached series at {csv_path}; falling back to full Pass 1.")

    log.info("Pass 1: recomputing regional daily LCI series (slow)")
    frames = []
    for period, experiment, years, _ in PERIODS:
        sub = compute_lci_regional_series(
            model, variant, rcm, experiment, years, region_masks, lat, lon,
        )
        seas_info = annotate_seasons(sub["date"])
        sub["season"] = seas_info["season"].values
        sub["season_year"] = seas_info["season_year"].values
        sub["period"] = period
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def _validate_series(df, csv_path):
    """
    Guard against reading a stale CSV from a different analysis period. A
    silently mismatched year range would produce composites labelled
    mid-century that were in fact thresholded on end-of-century days.
    """
    required = {"date", "season", "season_year", "period"} | set(REGION_SPECS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns {sorted(missing)}")

    for period, _, _, (yr0, yr1) in PERIODS:
        sub = df[df["period"] == period]
        if sub.empty:
            raise ValueError(f"{csv_path} contains no rows for period {period!r}")
        have = (int(sub["season_year"].min()), int(sub["season_year"].max()))
        if have[0] > yr0 or have[1] < yr1:
            raise ValueError(
                f"{csv_path} covers season_year {have} for period {period!r}, "
                f"which does not span the configured range ({yr0}, {yr1}). "
                f"Rerun with --recompute-series, or regenerate the CSV."
            )
    log.info("  cached series validated against configured season ranges")


# ── EVENT-DAY GRIDDED LCI COMPOSITES ─────────────────────────────────────────

def composite_gridded_lci(model, variant, rcm, events, thresholds, wanted_keys,
                          with_climatology=False):
    """
    For each (region, season, period) in wanted_keys, compute the event-day
    mean and sample std of gridded daily LCI.

    Returns (result, clim) where
        result[(region, season, period)] = {"lci": arr, "lci_std": arr, "n": int}
        clim[(season, period)]           = {"lci": arr, "n": int}   (may be empty)
    """
    event_df = pd.DataFrame(events, columns=["date", "region", "season", "period"])

    result, clim = {}, {}

    for period, experiment, years, (yr0, yr1) in PERIODS:
        keys_here = [k for k in wanted_keys if k[2] == period]
        if not keys_here:
            continue

        log.info(f"Opening LCI variables lazily | {experiment} | {min(years)}-{max(years)}")
        lazy = {v: open_var_lazy(model, variant, rcm, experiment, v, years)
                for v in LCI_VARS}

        ref_dates = time_strings(lazy["tasmaxAdjust"])
        date_to_idx = {d: i for i, d in enumerate(ref_dates)}

        for key in keys_here:
            region, season, _ = key
            dates = event_df[
                (event_df["region"] == region) &
                (event_df["season"] == season) &
                (event_df["period"] == period)
            ]["date"].tolist()

            indices = [date_to_idx[d] for d in dates if d in date_to_idx]
            missing = len(dates) - len(indices)
            if missing:
                log.warning(f"  {key}: {missing} event date(s) absent from time axis")
            if not indices:
                log.warning(f"  {key}: no valid event indices, skipping")
                continue

            log.info(f"  Compositing LCI {key}: {len(indices)} days "
                     f"(p95 = {thresholds.get(key, float('nan')):.1f} kJ m-2 hr-1)")

            # Materialise the event-day stack once, then form daily gridded LCI
            # and reduce. This is the whole point of the script: the mean is
            # taken AFTER applying the nonlinear formula, never before.
            tasmax = lazy["tasmaxAdjust"].isel(time=indices).compute().values
            tasmin = lazy["tasminAdjust"].isel(time=indices).compute().values
            ws = lazy["sfcWindAdjust"].isel(time=indices).compute().values
            pr = lazy["prAdjust"].isel(time=indices).compute().values

            lci_stack = calculate_lci(ws, (tasmax + tasmin) * 0.5, pr)
            del tasmax, tasmin, ws, pr

            result[key] = {
                "lci": np.nanmean(lci_stack, axis=0).astype(np.float32),
                "lci_std": np.nanstd(lci_stack, axis=0, ddof=1).astype(np.float32),
                "n": len(indices),
            }
            del lci_stack

        if with_climatology:
            seasons_here = sorted({k[1] for k in keys_here})
            seas = annotate_seasons(pd.Series(ref_dates))
            for season in seasons_here:
                sel = (
                    (seas["season"].values == season) &
                    (seas["season_year"].values >= yr0) &
                    (seas["season_year"].values <= yr1)
                )
                clim_idx = np.where(sel)[0]
                if len(clim_idx) == 0:
                    log.warning(f"  Climatology {season}/{period}: no days, skipping")
                    continue
                log.info(f"  Climatology LCI {season}/{period}: {len(clim_idx)} days")

                # Lazy: the full-season stack is far too large to materialise,
                # so LCI is built as a Dask graph and reduced chunk by chunk.
                # Still the nonlinear formula applied per day, then averaged.
                T_C = (lazy["tasmaxAdjust"].isel(time=clim_idx)
                       + lazy["tasminAdjust"].isel(time=clim_idx)) * 0.5
                ws_safe = lazy["sfcWindAdjust"].isel(time=clim_idx).clip(min=0.0)
                pr_safe = lazy["prAdjust"].isel(time=clim_idx).clip(min=0.0)
                lci_lazy = (
                    (11.7 + 3.1 * np.sqrt(ws_safe)) * (40.0 - T_C)
                    + 481.0
                    + 418.0 * (1.0 - np.exp(-0.04 * pr_safe))
                )
                clim[(season, period)] = {
                    "lci": lci_lazy.mean("time").compute().values.astype(np.float32),
                    "n": int(len(clim_idx)),
                }

        for da in lazy.values():
            try:
                da.close()
            except Exception:
                pass

    return result, clim


# ── OUTPUT ────────────────────────────────────────────────────────────────────

def write_lci_composite_netcdf(result, clim, thresholds, lat, lon,
                               region_name, model_label, outdir):
    """Write one gridded-LCI composite NetCDF per (model, region)."""
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir,
                           f"{model_label}_lci_composite_{region_name}_mid_century.nc")

    coords = {"lat": lat, "lon": lon}
    data_vars = {}

    for (region, season, period), d in result.items():
        if region != region_name:
            continue
        tag = f"{season}_{period}"
        data_vars[f"lci_{tag}"] = xr.DataArray(
            d["lci"], dims=("lat", "lon"), coords=coords,
            attrs={"units": "kJ m-2 hr-1", "n_days": d["n"],
                   "description": "event-day mean of daily gridded LCI"},
        )
        data_vars[f"lci_{tag}_std"] = xr.DataArray(
            d["lci_std"], dims=("lat", "lon"), coords=coords,
            attrs={"units": "kJ m-2 hr-1", "n_days": d["n"],
                   "description": "sample std (ddof=1) across event days"},
        )
        thr = thresholds.get((region, season, period))
        if thr is not None:
            data_vars[f"p95_{tag}"] = xr.DataArray(
                np.float32(thr),
                attrs={"units": "kJ m-2 hr-1",
                       "description": "this member's regional-mean p95 event threshold"},
            )

    for (season, period), d in clim.items():
        data_vars[f"lci_{season}_{period}_clim"] = xr.DataArray(
            d["lci"], dims=("lat", "lon"), coords=coords,
            attrs={"units": "kJ m-2 hr-1", "n_days": d["n"],
                   "description": "seasonal-mean of daily gridded LCI (all season days)"},
        )

    if not data_vars:
        log.warning(f"No LCI composite data for {region_name}; skipping write")
        return None

    ds = xr.Dataset(
        data_vars,
        attrs={
            "model": model_label,
            "region": region_name,
            "top_pct": TOP_PCT,
            "hist_period": f"{HIST_SEASON_RANGE[0]}-{HIST_SEASON_RANGE[1]}",
            "future_period": f"{FUTURE_SEASON_RANGE[0]}-{FUTURE_SEASON_RANGE[1]}",
            "scenario": "ssp370",
            "fields": "lci (kJ m-2 hr-1)",
            "note": (
                f"Top-{TOP_PCT}% daily LCI event composites of gridded LCI itself, "
                "AUST-05i domain. LCI is computed per day then averaged over event "
                "days, never reconstructed from composited T/V/R. Event days are "
                "defined by the p95 of the regional-mean LCI within each period "
                "separately. Per-member output; average across members in the plot "
                "script."
            ),
        },
    )

    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in data_vars}
    tmp = outpath + ".tmp"
    ds.to_netcdf(tmp, engine="h5netcdf", encoding=enc)
    os.replace(tmp, outpath)
    log.info(f"Wrote: {outpath}")
    return outpath


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Gridded top-5% LCI event composites for one NARCliM2 member."
    )
    ap.add_argument("--model", choices=list(MEMBERS),
                    help="Member label, e.g. ACCESS-ESM1-5_WRF412R3")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--all-seasons", action="store_true",
                    help="Composite both seasons for every region, not just each "
                         "region's focal season.")
    ap.add_argument("--with-climatology", action="store_true",
                    help="Also compute seasonal-mean gridded LCI over all season "
                         "days, enabling anomaly plots later. Roughly doubles "
                         "runtime; not needed for the absolute-LCI figure.")
    ap.add_argument("--recompute-series", action="store_true",
                    help="Force a full Pass 1 instead of reading the cached daily "
                         "regional LCI CSV.")
    ap.add_argument("--check-only", action="store_true",
                    help="Validate paths, masks and the cached series, then exit "
                         "without reading any field data.")
    args = ap.parse_args()

    if not args.check_only and not args.model:
        ap.error("--model is required unless --check-only is given")

    model_label = args.model or sorted(MEMBERS)[0]
    model, variant, rcm = MEMBERS[model_label]
    log.info(f"=== {model_label} ===")

    log.info("Reading grid coordinates...")
    ref = open_var_lazy(model, variant, rcm, "historical", "tasmaxAdjust", [1990],
                        chunks={"time": 1, "lat": 200, "lon": 200})
    lat = ref.lat.load()
    lon = ref.lon.load()

    log.info("Building region masks...")
    region_masks = {rname: build_region_mask(spec, lat, lon)
                    for rname, spec in REGION_SPECS.items()}
    for rname, m in region_masks.items():
        log.info(f"  {rname}: {int(m.sum())} cells")

    lci_df = load_daily_lci_series(model_label, model, variant, rcm,
                                   region_masks, lat, lon,
                                   recompute=args.recompute_series)

    log.info("Identifying top-5% event days...")
    events = find_top_pct_events(lci_df, list(REGION_SPECS.keys()))

    # find_top_pct_events logs the thresholds but does not return them; recompute
    # the same percentiles here so they can be written into the output files and
    # quoted in the figure caption.
    thresholds = {}
    for region in REGION_SPECS:
        for season in ("wet", "dry"):
            for period, _, _, (yr0, yr1) in PERIODS:
                sub = lci_df[
                    (lci_df["season"] == season) &
                    (lci_df["season_year"] >= yr0) &
                    (lci_df["season_year"] <= yr1)
                ][["date", region]].dropna()
                if not sub.empty:
                    thresholds[(region, season, period)] = float(
                        np.nanpercentile(sub[region].values, 100 - TOP_PCT)
                    )

    if args.all_seasons:
        wanted_keys = [(r, s, p) for r in REGION_SPECS
                       for s in ("wet", "dry")
                       for p, *_ in PERIODS]
    else:
        wanted_keys = [(r, FOCAL_SEASON[r], p) for r in REGION_SPECS
                       for p, *_ in PERIODS]
    log.info(f"Compositing keys: {wanted_keys}")

    if args.check_only:
        log.info("--check-only: paths, masks and cached series all validated. "
                 "Exiting before reading field data.")
        return

    result, clim = composite_gridded_lci(
        model, variant, rcm, events, thresholds, wanted_keys,
        with_climatology=args.with_climatology,
    )

    for rname in REGION_SPECS:
        if any(k[0] == rname for k in result):
            write_lci_composite_netcdf(result, clim, thresholds, lat, lon,
                                       rname, model_label, args.outdir)

    log.info("=== Done ===")


if __name__ == "__main__":
    main()