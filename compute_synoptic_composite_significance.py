#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_synoptic_composite_significance.py

Grid-cell significance test for the synoptic-composite CHANGE figure
(plot_synoptic_composites_change_focal_mid_century.py): is the change in
top-5% LCI
event anomaly (future minus historical) statistically distinguishable
from zero, once ensemble member-to-member spread is accounted for?

Method
------
For each (region, season, field), the 8 per-member composite files each
contribute a mean, a std (ddof=1), and an n_days count for the historical
event composite and for the future event composite (added to
synoptic_composite_lci_top5pct.py alongside the existing means).

1.  POOL across the 8 members into one "historical" sample and one
    "future" sample per grid cell, using the exact combined-sample
    mean/variance formula (this correctly folds in both the within-member
    day-to-day spread AND the between-member spread -- it does not assume
    the 8 members agree with each other):

        N = sum(n_i)
        M = sum(n_i * mean_i) / N
        S^2 = [ sum_i( (n_i-1)*var_i + n_i*(mean_i - M)^2 ) ] / (N - 1)

2.  WELCH'S T-TEST between the pooled historical and pooled future
    samples. The numerator uses the *anomaly* difference (event mean minus
    each period's seasonal climatology mean), i.e. exactly what the change
    figure plots:

        delta = (M_future - clim_future) - (M_hist - clim_hist)
        SE    = sqrt(S_hist^2 / N_hist + S_future^2 / N_future)
        t     = delta / SE

    Climatology means use the simple ensemble-mean across the 8 members'
    climatology fields. Climatology sampling uncertainty (~4500-4600 days)
    is treated as negligible next to the ~150-300 day event composites and
    is not included in SE -- if that assumption is ever challenged, this
    is the place to revisit it.

    Degrees of freedom use the Welch-Satterthwaite approximation.

3.  BENJAMINI-HOCHBERG FDR correction is applied within each
    (region, season, field) map separately, across all valid (non-NaN)
    grid cells in that map. This controls the expected false-discovery
    proportion given ~600k simultaneous tests per map; the raw p<0.05
    mask is also retained, allowing the two to be compared.

Inputs
------
Reads per-member files produced by synoptic_composite_lci_top5pct.py
(post std-variable update):
    {COMPOSITE_DIR}/{member}_composite_{region}_mid_century.nc

Requires the *_std variables (added alongside the means). If a file is
missing them, this will raise. synoptic_composite_lci_top5pct.py must be
run for all 8 members first.

Output
------
One NetCDF per region:
    {OUTDIR}/{region}_significance_mid_century.nc
Variables per (field, season): t_stat, p_raw, p_fdr, sig_raw (p<0.05),
sig_fdr (BH-corrected), dims (lat, lon).

Usage
-----
    module use /g/data/xp65/public/modules
    module load conda/analysis3

    python compute_synoptic_composite_significance.py
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from scipy import stats

DEFAULT_COMPOSITE_DIR = "/scratch/dx2/rt9243/chill_projections/synoptic_composites"
DEFAULT_OUTDIR = "/scratch/dx2/rt9243/chill_projections/synoptic_composites/significance"

REGIONS = [
    {"key": "Cobar_Lachlan", "season": "dry"},
    {"key": "Yilgarn_Coolgardie", "season": "dry"},
    {"key": "Winton", "season": "wet"},
]

FIELDS = ["psl", "sfcWind", "pr", "tas"]

ALPHA = 0.05


# ── Data loading ──────────────────────────────────────────────────────────────

def _open_dataset_loaded(path: str) -> xr.Dataset:
    try:
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            return ds.load()
    except Exception:
        with xr.open_dataset(path) as ds:
            return ds.load()


def load_member_files(region_key: str, composite_dir: str) -> List[xr.Dataset]:
    pattern = os.path.join(composite_dir, f"*_composite_{region_key}_mid_century.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No composite files matched: {pattern}")
    if len(files) != 8:
        print(f"[WARN] expected 8 member files for {region_key}, found {len(files)}: {files}")
    return [_open_dataset_loaded(f) for f in files]


# ── Pooling ───────────────────────────────────────────────────────────────────

def pool_members(datasets: List[xr.Dataset], field: str, season: str, period: str
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine the 8 members' (mean, std, n) for one field/season/period into a
    single pooled (mean, variance, n) per grid cell, using the exact
    combined-sample formula (accounts for both within- and between-member
    spread; does not assume member agreement).
    """
    var = f"{field}_{season}_{period}"
    std_var = f"{var}_std"

    means, stds, ns = [], [], []
    for ds in datasets:
        if var not in ds or std_var not in ds:
            raise KeyError(
                f"Missing {var!r} or {std_var!r}. "
                "synoptic_composite_lci_top5pct.py must be run with the "
                "std-variable update for all 8 members before this script."
            )
        means.append(ds[var].values)
        stds.append(ds[std_var].values)
        ns.append(ds[var].attrs["n_days"])

    means = np.stack(means, axis=0)   # (member, lat, lon)
    stds = np.stack(stds, axis=0)
    ns = np.array(ns, dtype=np.float64).reshape(-1, 1, 1)  # (member, 1, 1)

    variances = stds ** 2

    N = ns.sum(axis=0)  # (lat, lon)
    M = (ns * means).sum(axis=0) / N

    within = ((ns - 1) * variances).sum(axis=0)
    between = (ns * (means - M[None, :, :]) ** 2).sum(axis=0)
    S2 = (within + between) / (N - 1)

    return M, S2, N.squeeze(axis=0) if N.ndim == 3 else N


def climatology_ensemble_mean(datasets: List[xr.Dataset], field: str, season: str, period: str
                               ) -> np.ndarray:
    """Simple ensemble-mean of the seasonal climatology across members."""
    var = f"{field}_{season}_{period}_clim"
    vals = np.stack([ds[var].values for ds in datasets], axis=0)
    return vals.mean(axis=0)


# ── Significance test ─────────────────────────────────────────────────────────

def welch_t_test(
    M_hist: np.ndarray, S2_hist: np.ndarray, N_hist: np.ndarray,
    M_future: np.ndarray, S2_future: np.ndarray, N_future: np.ndarray,
    clim_hist: np.ndarray, clim_future: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welch's t-test on the anomaly difference. Returns (t_stat, df, p_raw)."""
    anom_hist = M_hist - clim_hist
    anom_future = M_future - clim_future
    delta = anom_future - anom_hist

    se2_hist = S2_hist / N_hist
    se2_future = S2_future / N_future
    se = np.sqrt(se2_hist + se2_future)

    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = delta / se

    df = (se2_hist + se2_future) ** 2 / (
        (se2_hist ** 2) / (N_hist - 1) + (se2_future ** 2) / (N_future - 1)
    )

    valid = np.isfinite(t_stat) & np.isfinite(df)
    p_raw = np.full(t_stat.shape, np.nan, dtype=np.float64)
    p_raw[valid] = 2 * stats.t.sf(np.abs(t_stat[valid]), df[valid])

    return t_stat, df, p_raw


def fdr_bh(pvals: np.ndarray, alpha: float = ALPHA) -> Tuple[np.ndarray, float]:
    """Benjamini-Hochberg FDR. Returns (sig_mask, critical_p)."""
    flat = pvals.ravel()
    valid = np.isfinite(flat)
    pv = flat[valid]
    n = pv.size
    if n == 0:
        return np.zeros(pvals.shape, dtype=bool), np.nan

    order = np.argsort(pv)
    ranked = pv[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresh

    if below.any():
        crit_p = ranked[np.max(np.where(below))]
    else:
        crit_p = -1.0  # nothing survives

    sig_flat = np.zeros(flat.shape, dtype=bool)
    sig_flat[valid] = flat[valid] <= crit_p
    return sig_flat.reshape(pvals.shape), crit_p


# ── Main per-region processing ─────────────────────────────────────────────────

def process_region(region_key: str, season: str, composite_dir: str) -> xr.Dataset:
    datasets = load_member_files(region_key, composite_dir)
    lat = datasets[0]["lat"]
    lon = datasets[0]["lon"]

    data_vars = {}
    for field in FIELDS:
        M_hist, S2_hist, N_hist = pool_members(datasets, field, season, "hist")
        M_future, S2_future, N_future = pool_members(datasets, field, season, "future")
        clim_hist = climatology_ensemble_mean(datasets, field, season, "hist")
        clim_future = climatology_ensemble_mean(datasets, field, season, "future")

        t_stat, df, p_raw = welch_t_test(
            M_hist, S2_hist, N_hist, M_future, S2_future, N_future,
            clim_hist, clim_future,
        )
        sig_raw = p_raw < ALPHA
        sig_fdr, crit_p = fdr_bh(p_raw, ALPHA)

        n_valid = int(np.isfinite(p_raw).sum())
        n_sig_raw = int(np.nansum(sig_raw))
        n_sig_fdr = int(sig_fdr.sum())
        print(
            f"  {region_key}/{season}/{field}: {n_valid} valid cells, "
            f"{n_sig_raw} sig (raw p<{ALPHA}), {n_sig_fdr} sig (FDR, "
            f"critical p={crit_p:.4g})"
        )

        for name, arr in (
            ("t_stat", t_stat), ("p_raw", p_raw),
            ("sig_raw", sig_raw.astype(np.int8)), ("sig_fdr", sig_fdr.astype(np.int8)),
        ):
            data_vars[f"{field}_{name}"] = xr.DataArray(
                arr.astype(np.float32) if arr.dtype != np.int8 else arr,
                dims=("lat", "lon"),
                coords={"lat": lat, "lon": lon},
            )

    for ds in datasets:
        ds.close()

    return xr.Dataset(
        data_vars,
        attrs={
            "region": region_key,
            "season": season,
            "test": "Welch's t-test on event anomaly change, pooled across 8 members",
            "alpha": ALPHA,
            "fdr_method": "Benjamini-Hochberg, applied per field within this region/season",
            "note": (
                "sig_raw = 1 where p_raw < alpha (uncorrected). sig_fdr = 1 where "
                "BH-FDR-corrected significance holds. Climatology sampling "
                "uncertainty treated as negligible; see script docstring."
            ),
        },
    )


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--composite-dir", default=DEFAULT_COMPOSITE_DIR)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for spec in REGIONS:
        print(f"=== {spec['key']} ({spec['season']}) ===")
        ds = process_region(spec["key"], spec["season"], args.composite_dir)
        outpath = Path(args.outdir) / f"{spec['key']}_significance_mid_century.nc"
        ds.to_netcdf(outpath, engine="h5netcdf")
        print(f"[OK] wrote {outpath}")


if __name__ == "__main__":
    main()