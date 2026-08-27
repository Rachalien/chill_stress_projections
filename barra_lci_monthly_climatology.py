#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
barra_lci_monthly_climatology.py

Compute climatological monthly LCI statistics from the existing per-year
BARRA-R2 daily LCI gridded files.

Outputs
-------
{OUTDIR}/barra_lci_monthly_climatology_1979_2018.nc
    lci_mean (month, lat, lon)            -- climatological monthly mean daily LCI
    lci_max  (month, lat, lon)            -- climatological mean of per-month maximum
    lci_freq (month, threshold, lat, lon) -- climatological mean exceedance days/month

PBS guidance (Gadi normal queue)
---------------------------------
  ncpus=1, mem=16GB, walltime=02:00:00
  storage: scratch/dx2+gdata/dx2
  module: conda/analysis3 (via xp65)
"""

from __future__ import annotations

import os
import argparse
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr

# ============================================================
# CONFIG
# ============================================================

BARRA_DAILY_DIR  = "/scratch/dx2/rt9243/barra_daily_lci"
BARRA_FILE_TMPL  = "barra_lci_daily_aus_land_{year}.nc"
BARRA_LCI_VAR    = "lci_daily"

OUTDIR      = "/scratch/dx2/rt9243/chill_projections/monthly_bias"
START_YEAR  = 1979
END_YEAR    = 2018
THRESHOLDS  = [1000, 1100, 1200]


# ============================================================
# MONTHLY ACCUMULATOR  (identical logic to agcd_lci_monthly_gridded.py)
# ============================================================

class MonthlyAccumulator:
    """
    Running accumulators for three climatological monthly LCI fields.
    See agcd_lci_monthly_gridded.py for full documentation.
    """

    def __init__(self, n_lat: int, n_lon: int, thresholds: List[int]) -> None:
        self.thresholds = thresholds
        sh  = (12, n_lat, n_lon)
        shf = (12, len(thresholds), n_lat, n_lon)

        self.mean_sum = np.zeros(sh,  dtype=np.float64)
        self.mean_n   = np.zeros(sh,  dtype=np.float64)
        self.max_sum  = np.zeros(sh,  dtype=np.float64)
        self.max_n    = np.zeros(sh,  dtype=np.float64)
        self.freq_sum = np.zeros(shf, dtype=np.float64)
        self.freq_n   = np.zeros(shf, dtype=np.float64)

    def accumulate_year(self, lci_np: np.ndarray, months: np.ndarray) -> None:
        for m in range(1, 13):
            mi    = m - 1
            mask  = months == m
            if not mask.any():
                continue

            lci_m = lci_np[mask]
            valid = ~np.isnan(lci_m)

            self.mean_sum[mi] += np.nansum(lci_m, axis=0)
            self.mean_n[mi]   += valid.sum(axis=0).astype(np.float64)

            all_nan   = (~valid).all(axis=0)
            month_max = np.where(all_nan, np.nan, np.nanmax(lci_m, axis=0))
            self.max_sum[mi] += np.where(np.isnan(month_max), 0.0, month_max)
            self.max_n[mi]   += (~np.isnan(month_max)).astype(np.float64)

            for ti, thr in enumerate(self.thresholds):
                exceed = (lci_m >= float(thr)).astype(np.float64)
                exceed[~valid] = 0.0
                self.freq_sum[mi, ti] += exceed.sum(axis=0)
                self.freq_n[mi, ti]   += (valid.any(axis=0)).astype(np.float64)

    def climatology(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lci_mean = np.where(self.mean_n > 0,
                            (self.mean_sum / self.mean_n).astype(np.float32),
                            np.float32(np.nan))
        lci_max  = np.where(self.max_n > 0,
                            (self.max_sum  / self.max_n).astype(np.float32),
                            np.float32(np.nan))
        lci_freq = np.where(self.freq_n > 0,
                            (self.freq_sum / self.freq_n).astype(np.float32),
                            np.float32(np.nan))
        return lci_mean, lci_max, lci_freq


# ============================================================
# OUTPUT
# ============================================================

def build_dataset(
    lci_mean: np.ndarray,
    lci_max:  np.ndarray,
    lci_freq: np.ndarray,
    lat:  np.ndarray,
    lon:  np.ndarray,
    thresholds: List[int],
    start_year: int,
    end_year:   int,
) -> xr.Dataset:
    ds = xr.Dataset(
        {
            "lci_mean": (["month", "lat", "lon"], lci_mean,
                         {"long_name": "Climatological monthly mean daily LCI",
                          "units": "kJ m-2 hr-1"}),
            "lci_max":  (["month", "lat", "lon"], lci_max,
                         {"long_name": "Climatological mean of per-month maximum daily LCI",
                          "units": "kJ m-2 hr-1"}),
            "lci_freq": (["month", "threshold", "lat", "lon"], lci_freq,
                         {"long_name": "Climatological mean exceedance days per month",
                          "units": "days month-1"}),
        },
        coords={
            "month":     np.arange(1, 13, dtype=np.int32),
            "lat":       (["lat"], lat,  {"units": "degrees_north"}),
            "lon":       (["lon"], lon,  {"units": "degrees_east"}),
            "threshold": (["threshold"], np.array(thresholds, dtype=np.float32),
                          {"units": "kJ m-2 hr-1", "long_name": "LCI exceedance threshold"}),
        },
    )
    ds.attrs = {
        "title":      "BARRA-R2 LCI monthly climatology",
        "description": (
            "Monthly LCI statistics computed from BARRA-R2 daily LCI "
            f"(2m wind correction applied). Climatological averages over "
            f"{start_year}\u2013{end_year}."
        ),
        "source":  BARRA_DAILY_DIR,
        "history": f"Created by barra_lci_monthly_climatology.py; period {start_year}-{end_year}",
    }
    return ds


def write_netcdf_atomic(ds: xr.Dataset, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    enc = {}
    for v in ds.data_vars:
        enc[v] = {"zlib": True, "complevel": 4, "shuffle": True, "dtype": "float32"}
    for c in ("lat", "lon"):
        enc[c] = {"zlib": True, "complevel": 1, "dtype": "float64"}
    tmp = path + ".tmp"
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)
    print(f"[OK] wrote {path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute BARRA-R2 gridded monthly LCI climatology."
    )
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--end-year",   type=int, default=END_YEAR)
    ap.add_argument("--outdir",     default=OUTDIR)
    args = ap.parse_args()

    years      = list(range(args.start_year, args.end_year + 1))
    thresholds = THRESHOLDS
    os.makedirs(args.outdir, exist_ok=True)

    accum: Optional[MonthlyAccumulator] = None
    grid_lat = grid_lon = None
    years_ok: List[int] = []

    for year in years:
        path = os.path.join(BARRA_DAILY_DIR, BARRA_FILE_TMPL.format(year=year))
        if not os.path.exists(path):
            print(f"[WARN] Skipping {year}: file not found ({path})")
            continue

        print(f"[{year}] Loading {BARRA_LCI_VAR} ...")
        ds  = xr.open_dataset(path)

        if BARRA_LCI_VAR not in ds:
            print(f"[WARN] {BARRA_LCI_VAR} not in {path}; skipping.")
            ds.close()
            continue

        lci = ds[BARRA_LCI_VAR].load()
        ds.close()

        n_days = len(lci.time)
        if n_days < 180:
            print(f"[WARN] Only {n_days} days for {year}; skipping.")
            del lci
            continue

        # Ensure lat is ascending
        if float(lci.lat[0]) > float(lci.lat[-1]):
            lci = lci.sortby("lat")

        # Initialise accumulator
        if accum is None:
            nlat, nlon = len(lci.lat), len(lci.lon)
            accum    = MonthlyAccumulator(nlat, nlon, thresholds)
            grid_lat = lci.lat.values.copy()
            grid_lon = lci.lon.values.copy()

        months = lci.time.dt.month.values
        accum.accumulate_year(lci.values, months)

        del lci
        years_ok.append(year)
        print(f"[{year}] Done.")

    if accum is None:
        raise RuntimeError("No years processed successfully.")

    print(f"\n[INFO] Processed {len(years_ok)} years: {years_ok[0]}-{years_ok[-1]}")
    print("[INFO] Computing climatological means ...")
    lci_mean, lci_max, lci_freq = accum.climatology()

    ds = build_dataset(lci_mean, lci_max, lci_freq,
                       grid_lat, grid_lon, thresholds,
                       years_ok[0], years_ok[-1])

    out = os.path.join(
        args.outdir,
        f"barra_lci_monthly_climatology_{years_ok[0]}_{years_ok[-1]}.nc",
    )
    write_netcdf_atomic(ds, out)
    print(f"[DONE] Output: {out}")


if __name__ == "__main__":
    main()
