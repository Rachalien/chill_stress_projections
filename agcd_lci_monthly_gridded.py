#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agcd_lci_monthly_gridded.py

Compute climatological monthly LCI statistics from AGCD+AWRA-L inputs
over the common period 1979-2018, retaining the full spatial grid.

Outputs
-------
{OUTDIR}/agcd_lci_monthly_climatology_1979_2018.nc
    lci_mean (month, lat, lon)            -- climatological monthly mean daily LCI
    lci_max  (month, lat, lon)            -- climatological mean of per-month maximum
    lci_freq (month, threshold, lat, lon) -- climatological mean exceedance days/month

Processing
----------
Year-by-year. For each year the full gridded daily LCI (time, lat, lon) is
computed, grouped by calendar month, and monthly statistics accumulated.
Annual LCI arrays are discarded after monthly aggregation.

PBS guidance (Gadi normal queue)
---------------------------------
  ncpus=1, mem=32GB, walltime=06:00:00
  storage: gdata/zv2+gdata/fj8+scratch/dx2+gdata/dx2
  module: conda/analysis3 (via xp65)
"""

from __future__ import annotations

import os
import argparse
from typing import List, Optional, Tuple, Dict

import numpy as np
import xarray as xr

# ============================================================
# CONFIG
# ============================================================

AGCD_ROOT        = "/g/data/zv2/agcd/v1-0-3"
AGCD_TMAX_TMPL   = "{root}/tmax/mean/r005/01day/agcd_v1_tmax_mean_r005_daily_{year}.nc"
AGCD_TMIN_TMPL   = "{root}/tmin/mean/r005/01day/agcd_v1_tmin_mean_r005_daily_{year}.nc"
AGCD_PRECIP_TMPL = "{root}/precip/calib/r005/01day/agcd_v1_precip_calib_r005_daily_{year}.nc"
AWRA_WIND_TMPL   = "/g/data/fj8/BoM/AWRA/DATA/wind_investigation_data/both/wind_{year}.nc"

OUTDIR      = "/scratch/dx2/rt9243/chill_projections/monthly_bias"
START_YEAR  = 1979
END_YEAR    = 2018
THRESHOLDS  = [1000, 1100, 1200]


# ============================================================
# SHARED LCI FUNCTIONS
# ============================================================

def open_agcd_var(template: str, year: int, var_name: str) -> xr.DataArray:
    path = template.format(root=AGCD_ROOT, year=year)
    if not os.path.exists(path):
        raise FileNotFoundError(f"AGCD file not found: {path}")
    ds = xr.open_dataset(path)
    return ds[var_name].load()


def open_awra_wind(year: int) -> xr.DataArray:
    path = AWRA_WIND_TMPL.format(year=year)
    if not os.path.exists(path):
        raise FileNotFoundError(f"AWRA wind file not found: {path}")
    ds = xr.open_dataset(path)
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    ws = ds["wind"].where(ds["wind"] > -900.0).load()
    if float(ws.lat[0]) > float(ws.lat[-1]):
        ws = ws.sortby("lat")
    if float(ws.lon[0]) > float(ws.lon[-1]):
        ws = ws.sortby("lon")
    return ws

def align_agcd_to_wind_grid(da: xr.DataArray, wind: xr.DataArray, name: str) -> xr.DataArray:
    if float(da.lat[0]) > float(da.lat[-1]):
        da = da.sortby("lat")
    if float(da.lon[0]) > float(da.lon[-1]):
        da = da.sortby("lon")
    # Interpolate lat/lon only; interp_like would also attempt time interpolation
    # across incompatible coordinate types (cftime vs datetime64).
    aligned = da.interp(lat=wind.lat, lon=wind.lon, method="linear")
    if float(aligned.isnull().mean()) > 0.5:
        raise ValueError(f"{name}: >50% NaN after interp")
    return aligned

def align_to_common_dates(*arrays: xr.DataArray) -> Tuple[xr.DataArray, ...]:
    import pandas as pd

    # Extract YYYY-MM-DD strings directly from raw time values; avoids
    # type-dependent behaviour of da.time.dt.strftime across cftime / datetime64.
    date_strs = []
    for da in arrays:
        strs = np.array([str(t)[:10] for t in da.time.values])
        date_strs.append(strs)

    common = sorted(
        set(date_strs[0]).intersection(*(set(s) for s in date_strs[1:]))
    )
    if not common:
        raise ValueError(
            f"No common dates. First array dates: {date_strs[0][:3]}, "
            f"last array dates: {date_strs[-1][:3]}"
        )

    common_set   = set(common)
    common_times = pd.to_datetime(common)

    result = []
    for da, strs in zip(arrays, date_strs):
        idx = [i for i, s in enumerate(strs) if s in common_set]
        if not idx:
            raise ValueError(
                f"align_to_common_dates: no matching dates for array "
                f"(first dates: {strs[:3]})"
            )
        sel   = da.isel(time=idx)
        order = np.argsort([common.index(s) for s in strs[idx]])
        sel   = sel.isel(time=order)
        sel   = sel.assign_coords(time=common_times.values)
        result.append(sel)

    return tuple(result)


def calculate_lci(ws: xr.DataArray, T_C: xr.DataArray, R_mm: xr.DataArray) -> xr.DataArray:
    lci = (11.7 + 3.1 * np.sqrt(ws)) * (40.0 - T_C) + 481.0 + 418.0 * (1.0 - np.exp(-0.04 * R_mm))
    lci.name = "lci"
    lci.attrs = {"long_name": "Livestock Chill Index", "units": "kJ m-2 hr-1"}
    return lci


# ============================================================
# MONTHLY ACCUMULATOR HELPERS
# ============================================================

class MonthlyAccumulator:
    """
    Running accumulators for three LCI climatological monthly fields.

    For each calendar month (1-12) and grid cell, tracks:
      mean  : sum of daily LCI values + count of valid days
      max   : sum of per-month maxima  + count of valid years
      freq  : sum of exceedance days   + count of valid years  (per threshold)
    """

    def __init__(self, n_lat: int, n_lon: int, thresholds: List[int]) -> None:
        self.thresholds = thresholds
        self.n_thr      = len(thresholds)
        sh  = (12, n_lat, n_lon)
        shf = (12, self.n_thr, n_lat, n_lon)

        self.mean_sum   = np.zeros(sh,  dtype=np.float64)
        self.mean_n     = np.zeros(sh,  dtype=np.float64)   # valid day count per cell
        self.max_sum    = np.zeros(sh,  dtype=np.float64)
        self.max_n      = np.zeros(sh,  dtype=np.float64)   # valid year count per cell
        self.freq_sum   = np.zeros(shf, dtype=np.float64)
        self.freq_n     = np.zeros(shf, dtype=np.float64)

    def accumulate_year(self, lci_np: np.ndarray, months: np.ndarray) -> None:
        """
        Add one year's worth of gridded LCI (n_days, n_lat, n_lon) to accumulators.
        months: array of integer month values (1-12), same length as lci_np axis 0.
        """
        for m in range(1, 13):
            mi   = m - 1
            mask = months == m
            if not mask.any():
                continue

            lci_m  = lci_np[mask]                    # (n_days_in_month, nlat, nlon)
            valid  = ~np.isnan(lci_m)                # (n_days_in_month, nlat, nlon)

            # ── Mean ──────────────────────────────────────────────────────
            self.mean_sum[mi] += np.nansum(lci_m, axis=0)
            self.mean_n[mi]   += valid.sum(axis=0).astype(np.float64)

            # ── Max ───────────────────────────────────────────────────────
            all_nan = (~valid).all(axis=0)
            month_max = np.where(all_nan, np.nan, np.nanmax(lci_m, axis=0))
            self.max_sum[mi] += np.where(np.isnan(month_max), 0.0, month_max)
            self.max_n[mi]   += (~np.isnan(month_max)).astype(np.float64)

            # ── Freq ──────────────────────────────────────────────────────
            for ti, thr in enumerate(self.thresholds):
                exceed = (lci_m >= float(thr)).astype(np.float64)
                exceed[~valid] = 0.0    # NaN days don't count
                self.freq_sum[mi, ti] += exceed.sum(axis=0)
                # year contributes to freq count if ≥1 valid day in month
                self.freq_n[mi, ti]   += (valid.any(axis=0)).astype(np.float64)

    def climatology(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lci_mean, lci_max, lci_freq) as float32 arrays."""
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
    month_coord  = np.arange(1, 13, dtype=np.int32)
    thr_coord    = np.array(thresholds, dtype=np.float32)

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
                          "units": "days month-1",
                          "note": "Exceedance counted per grid cell; see threshold coordinate."}),
        },
        coords={
            "month":     month_coord,
            "lat":       (["lat"], lat,  {"units": "degrees_north", "long_name": "latitude"}),
            "lon":       (["lon"], lon,  {"units": "degrees_east",  "long_name": "longitude"}),
            "threshold": (["threshold"], thr_coord,
                         {"units": "kJ m-2 hr-1", "long_name": "LCI exceedance threshold"}),
        },
    )
    ds.attrs = {
        "title":       "AGCD+AWRA-L LCI monthly climatology",
        "description": (
            "Monthly LCI statistics computed from AGCD tmax/tmin/precip and "
            "AWRA-L 2m wind, on the AWRA-L 0.05 deg grid. "
            "Climatological averages over the period "
            f"{start_year}\u2013{end_year}."
        ),
        "lci_formula":  "(11.7 + 3.1*sqrt(ws)) * (40 - T) + 481 + 418*(1 - exp(-0.04*R))",
        "history":      f"Created by agcd_lci_monthly_gridded.py; period {start_year}-{end_year}",
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
        description="Compute AGCD+AWRA-L gridded monthly LCI climatology."
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
        print(f"[{year}] Loading inputs ...")
        try:
            tmax   = open_agcd_var(AGCD_TMAX_TMPL,   year, "tmax")
            tmin   = open_agcd_var(AGCD_TMIN_TMPL,   year, "tmin")
            precip = open_agcd_var(AGCD_PRECIP_TMPL, year, "precip")
            wind   = open_awra_wind(year)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {year}: {e}")
            continue

        print(f"[{year}] Aligning grids and time axes ...")
        try:
            tmax_a   = align_agcd_to_wind_grid(tmax,   wind, "tmax")
            tmin_a   = align_agcd_to_wind_grid(tmin,   wind, "tmin")
            precip_a = align_agcd_to_wind_grid(precip, wind, "precip")
            tmax_a, tmin_a, precip_a, wind = align_to_common_dates(
                tmax_a, tmin_a, precip_a, wind
            )
        except (ValueError, Exception) as e:
            print(f"[WARN] Skipping {year}: {e}")
            continue

        if len(tmax_a.time) < 180:
            print(f"[WARN] Only {len(tmax_a.time)} days for {year}; skipping.")
            continue

        print(f"[{year}] Computing LCI ({len(tmax_a.time)} days) ...")
        lci = calculate_lci(wind, (tmax_a + tmin_a) * 0.5, precip_a)

        # Initialise accumulator from first successful year
        if accum is None:
            nlat, nlon = len(lci.lat), len(lci.lon)
            accum    = MonthlyAccumulator(nlat, nlon, thresholds)
            grid_lat = lci.lat.values.copy()
            grid_lon = lci.lon.values.copy()

        print(f"[{year}] Accumulating monthly statistics ...")
        lci_np = lci.values                         # (ndays, nlat, nlon)
        months = lci.time.dt.month.values
        accum.accumulate_year(lci_np, months)

        del lci, lci_np, tmax_a, tmin_a, precip_a, wind
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
        f"agcd_lci_monthly_climatology_{years_ok[0]}_{years_ok[-1]}.nc",
    )
    write_netcdf_atomic(ds, out)
    print(f"[DONE] Output: {out}")


if __name__ == "__main__":
    main()