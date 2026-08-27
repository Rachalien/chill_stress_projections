#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
barra_daily_lci_yearly_land.py

Compute daily Livestock Chill Index (LCI) from BARRA-R2 daily reanalysis
over Australian land, writing one NetCDF per year.

Inputs:
  tas      [K]
  sfcWind  [m s-1]
  pr       [kg m-2 s-1]

LCI:
  (11.7 + 3.1*sqrt(v))*(40 - T) + 481 + 418*(1 - exp(-0.04*R))

where:
  v = wind speed [m s-1]
  T = temperature [degC]
  R = daily precipitation [mm day-1]

Main design choices:
- process one year at a time
- use only the `latest/` directory to avoid duplicate monthly files
- subset to Australian bbox before rechunking
- apply Australian land mask before computing LCI
- handle partial years (e.g. 2025) automatically
"""

import os
import glob
import argparse
import warnings

import numpy as np
import xarray as xr

xr.set_options(keep_attrs=True)

BARRA_ROOT = (
    "/g/data/ob53/BARRA2/output/reanalysis/"
    "AUS-11/BOM/ERA5/historical/hres/BARRA-R2/v1"
)

DEFAULT_NE_SHP = (
    "/g/data/dx2/rt9243/reference_data/natural_earth/"
    "ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
)

AUS_LAT_MIN = -45.0
AUS_LAT_MAX = -9.0
AUS_LON_MIN = 110.0
AUS_LON_MAX = 155.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--barra-root", default=BARRA_ROOT)
    p.add_argument("--outdir", required=True)
    p.add_argument("--start-year", type=int, default=1979)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--chunks-time", type=int, default=31)
    p.add_argument("--chunks-lat", type=int, default=120)
    p.add_argument("--chunks-lon", type=int, default=120)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-land-mask", action="store_true")
    p.add_argument("--ne-shapefile", default=DEFAULT_NE_SHP)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_lon_bounds(ds):
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360)).sortby("lon")
    return ds


def subset_aus_bbox(da):
    da = normalize_lon_bounds(da)
    da = da.sel(
        lat=slice(AUS_LAT_MIN, AUS_LAT_MAX),
        lon=slice(AUS_LON_MIN, AUS_LON_MAX),
    )
    return da


def find_daily_files_for_year(barra_root, var, year):
    """
    Restrict to latest/ only, to avoid double-counting latest + versioned dirs.
    """
    pattern = os.path.join(
        barra_root,
        "day",
        var,
        "latest",
        f"{var}_AUS-11_ERA5_historical_hres_BOM_BARRA-R2_v1_day_{year}??-{year}??.nc"
    )
    files = sorted(glob.glob(pattern))
    return files


def open_year_variable(barra_root, var, year, debug=False):
    files = find_daily_files_for_year(barra_root, var, year)

    if debug:
        print(f"[DEBUG] {var} {year}: found {len(files)} files")
        for f in files[:3]:
            print(f"         {f}")
        if len(files) > 3:
            print("         ...")

    if not files:
        raise FileNotFoundError(f"No files found for {var} in {year}")

    # Open without forcing awkward spatial chunks up front
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        coords="minimal",
        data_vars="minimal",
        compat="override",
        decode_times=True,
    )

    drop_names = [name for name in ds.variables if name.endswith("_bnds") or name == "bnds"]
    ds = ds.drop_vars(drop_names, errors="ignore")

    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in opened dataset for {year}")

    da = ds[var]

    for coord_name in ["height", "forecast_reference_time", "crs"]:
        if coord_name in da.coords:
            da = da.drop_vars(coord_name, errors="ignore")

    return da


def align_three(a, b, c):
    a, b, c = xr.align(a, b, c, join="inner")
    return a, b, c


def pr_flux_to_mm_day(pr_flux):
    out = pr_flux * 86400.0
    out.name = "pr_daily_total_mm"
    out.attrs.update({
        "long_name": "Daily total precipitation",
        "units": "mm day-1",
        "description": "Converted from precipitation flux by multiplying by 86400.",
    })
    return out


def tas_k_to_c(tas):
    out = tas - 273.15
    out.name = "tas_c"
    out.attrs.update({
        "long_name": "Near-surface air temperature",
        "units": "degC",
        "description": "Daily mean temperature converted from Kelvin.",
    })
    return out

# ── Surface roughness length (z0) for 10 m → 2 m wind correction ─────────────
Z0_PATH = "/g/data/dx2/rt9243/Datasets/sfc_rough_len_Aust_fc.nc"
# z0 must be > 0 and < 2 m so that ln(2/z0) > 0; clip defensively.
Z0_MIN, Z0_MAX = 1e-4, 1.9


def load_z0_static(z0_path=Z0_PATH):
    """
    Load ACCESS-G surface roughness length, collapse time to a static 2-D
    (lat, lon) field via time-mean, and clip to physical bounds.
    """
    ds = xr.open_dataset(z0_path)
    z0 = ds["sfc_rough_len"].mean("time", skipna=True)
    z0 = z0.clip(min=Z0_MIN, max=Z0_MAX)
    z0.attrs.update({
        "units": "m",
        "long_name": "Surface roughness length for momentum (ACCESS-G time-mean)",
        "source": z0_path,
    })
    return z0


def regrid_z0(z0_raw, target_lat, target_lon):
    """
    Interpolate z0 from the ACCESS-G grid onto (target_lat, target_lon).
    Uses nearest-neighbour so cells outside the z0 domain get a valid value
    rather than NaN (relevant for southern/western BARRA-R2 extents).
    """
    z0_interp = z0_raw.interp(
        lat=target_lat,
        lon=target_lon,
        method="nearest",
    )
    return z0_interp.clip(min=Z0_MIN, max=Z0_MAX)


def wind_10m_to_2m(ws, z0):
    """
    Convert 10-m wind speed to 2-m using the log wind profile.

    ws  : xarray.DataArray (m s-1), any shape with (lat, lon) dims.
    z0  : xarray.DataArray (lat, lon), roughness length (m), on the same grid.

    Returns a DataArray of the same shape as ws.
    """
    c = np.log(2.0 / z0) / np.log(10.0 / z0)   # (lat, lon) correction factor
    ws2m = ws.clip(min=0.0) * c
    ws2m.attrs.update(ws.attrs)
    ws2m.attrs["height"] = "2 m"
    ws2m.attrs["wind_height_correction"] = (
        "log wind profile, v2 = v10 * ln(2/z0) / ln(10/z0); "
        f"z0 from ACCESS-G {Z0_PATH}"
    )
    return ws2m


def wind_10m_to_2m_np(wind_np, z0_np):
    """
    Numpy version of the 10→2 m log-law correction.

    wind_np : (time, lat, lon) float array.
    z0_np   : (lat, lon) float array, roughness length (m).

    Returns corrected (time, lat, lon) array.
    """
    c = np.log(2.0 / z0_np) / np.log(10.0 / z0_np)   # (lat, lon)
    return np.clip(wind_np, 0, None) * c[np.newaxis, :, :]
# ─────────────────────────────────────────────────────────────────────────────

def calculate_lci_daily(ws, t_c, r_mm_day):
    ws_safe = ws.clip(min=0.0)
    r_safe = r_mm_day.clip(min=0.0)

    out = (
        (11.7 + 3.1 * np.sqrt(ws_safe)) * (40.0 - t_c)
        + 481.0
        + 418.0 * (1.0 - np.exp(-0.04 * r_safe))
    )
    out.name = "lci_daily"
    out.attrs.update({
        "long_name": "Livestock Chill Index",
        "units": "kJ m-2 hr-1",
        "description": "Daily LCI computed from daily tas, sfcWind, and daily total pr.",
    })
    return out


def build_australia_land_mask_from_local_shapefile(lat, lon, shapefile_path, debug=False):
    """
    Build Australia land mask from a local Natural Earth shapefile.
    Requires geopandas + regionmask.
    """
    import geopandas as gpd
    import regionmask

    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    gdf = gpd.read_file(shapefile_path)

    # Natural Earth usually uses ADMIN and/or NAME_LONG
    if "ADMIN" in gdf.columns:
        aus = gdf[gdf["ADMIN"] == "Australia"].copy()
    elif "NAME_LONG" in gdf.columns:
        aus = gdf[gdf["NAME_LONG"] == "Australia"].copy()
    else:
        raise ValueError(
            f"Could not find Australia field in shapefile columns: {list(gdf.columns)}"
        )

    if len(aus) == 0:
        raise ValueError("Australia polygon not found in shapefile.")

    # regionmask can build directly from geopandas
    regs = regionmask.from_geopandas(aus, names="ADMIN")

    lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    regmask = regs.mask(lon2d, lat2d)

    mask = xr.DataArray(
        ~np.isnan(regmask),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="aus_land_mask",
    )
    mask.attrs.update({
        "long_name": "Australian land mask",
        "description": f"Mask derived from local shapefile: {shapefile_path}",
        "units": "1",
    })

    if debug:
        print(f"[DEBUG] Australia land cells: {int(mask.sum().values)}")

    return mask


def build_encoding(ds):
    enc = {}
    for name, da in ds.data_vars.items():
        if np.issubdtype(da.dtype, np.floating):
            enc[name] = {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(np.nan),
            }
        elif np.issubdtype(da.dtype, np.integer):
            enc[name] = {
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
            }
        else:
            enc[name] = {"zlib": True, "complevel": 4}
    return enc


def process_year(year, args, z0):
    print(f"\n=== Processing {year} ===")

    tas = open_year_variable(args.barra_root, "tas", year, debug=args.debug)
    wind = open_year_variable(args.barra_root, "sfcWind", year, debug=args.debug)
    pr = open_year_variable(args.barra_root, "pr", year, debug=args.debug)

    tas, wind, pr = align_three(tas, wind, pr)

    tas = subset_aus_bbox(tas)
    wind = subset_aus_bbox(wind)
    pr = subset_aus_bbox(pr)

    tas, wind, pr = align_three(tas, wind, pr)

    if tas.sizes.get("time", 0) == 0:
        raise ValueError(f"No overlapping data after alignment/subset for {year}")

    # Rechunk only after subsetting
    chunk_map = {
        "time": args.chunks_time,
        "lat": args.chunks_lat,
        "lon": args.chunks_lon,
    }
    tas = tas.chunk(chunk_map)
    wind = wind.chunk(chunk_map)
    pr = pr.chunk(chunk_map)

    if args.no_land_mask:
        land_mask = xr.DataArray(
            np.ones((tas.sizes["lat"], tas.sizes["lon"]), dtype=bool),
            coords={"lat": tas.lat, "lon": tas.lon},
            dims=("lat", "lon"),
            name="aus_land_mask",
        )
        land_mask.attrs.update({
            "long_name": "Australian land mask",
            "description": "All bbox cells retained because --no-land-mask was used.",
            "units": "1",
        })
    else:
        try:
            land_mask = build_australia_land_mask_from_local_shapefile(
                tas.lat, tas.lon, args.ne_shapefile, debug=args.debug
            )
        except Exception as e:
            print(f"[WARN] Failed to build Australia land mask: {e}")
            print("[WARN] Continuing with Australia bounding box only.")
            land_mask = xr.DataArray(
                np.ones((tas.sizes["lat"], tas.sizes["lon"]), dtype=bool),
                coords={"lat": tas.lat, "lon": tas.lon},
                dims=("lat", "lon"),
                name="aus_land_mask",
            )
            land_mask.attrs.update({
                "long_name": "Australian land mask",
                "description": "All bbox cells retained because local land-mask construction failed.",
                "units": "1",
            })

    tas = tas.where(land_mask)
    wind = wind.where(land_mask)
    pr = pr.where(land_mask)

    tas_c = tas_k_to_c(tas)
    pr_mm_day = pr_flux_to_mm_day(pr)

    # Convert sfcWind from 10 m to 2 m via log wind profile before LCI calculation.
    wind2m = wind_10m_to_2m(wind, z0)

    lci = calculate_lci_daily(wind2m, tas_c, pr_mm_day)

    lci_ge_1000 = (lci >= 1000.0).astype("int8").rename("lci_ge_1000")
    lci_ge_1100 = (lci >= 1100.0).astype("int8").rename("lci_ge_1100")
    lci_ge_1200 = (lci >= 1200.0).astype("int8").rename("lci_ge_1200")

    lci_ge_1000.attrs.update({"description": "1 if daily LCI >= 1000", "units": "1"})
    lci_ge_1100.attrs.update({"description": "1 if daily LCI >= 1100", "units": "1"})
    lci_ge_1200.attrs.update({"description": "1 if daily LCI >= 1200", "units": "1"})

    ds_out = xr.Dataset(
        data_vars={
            "tas": tas.astype("float32"),
            "tas_c": tas_c.astype("float32"),
            "sfcWind_10m": wind.astype("float32"),
            "sfcWind_2m": wind2m.astype("float32"),
            "pr": pr.astype("float32"),
            "pr_daily_total_mm": pr_mm_day.astype("float32"),
            "lci_daily": lci.astype("float32"),
            "aus_land_mask": land_mask.astype("int8"),
            "lci_ge_1000": lci_ge_1000,
            "lci_ge_1100": lci_ge_1100,
            "lci_ge_1200": lci_ge_1200,
        },
        coords={
            "time": tas.time,
            "lat": tas.lat,
            "lon": tas.lon,
        },
        attrs={
            "title": f"BARRA-R2 daily LCI over Australian land for {year}",
            "source_dataset": "BARRA-R2 AUS-11 daily reanalysis",
            "barra_root": args.barra_root,
            "year_requested": year,
            "time_coverage_start_actual": str(tas.time.values[0]),
            "time_coverage_end_actual": str(tas.time.values[-1]),
            "note_partial_years": (
                "This file contains whatever dates were available for tas, sfcWind, and pr "
                "after inner alignment. Partial years are handled automatically."
            ),
            "bbox": (
                f"lat {AUS_LAT_MIN} to {AUS_LAT_MAX}, "
                f"lon {AUS_LON_MIN} to {AUS_LON_MAX}"
            ),
            "lci_formula": (
                "(11.7 + 3.1*sqrt(v))*(40 - T) + 481 + 418*(1 - exp(-0.04*R))"
            ),
            "lci_variable_definitions": (
                "v = sfcWind_2m [m s-1] (corrected from 10 m to 2 m via log wind profile "
                "using ACCESS-G z0), T = tas_c [degC], R = daily precipitation [mm day-1]"
            ),
            "wind_height_correction": (
                f"log wind profile v2 = v10 * ln(2/z0) / ln(10/z0); z0 from {Z0_PATH}"
            ),
            "precip_conversion": "pr [kg m-2 s-1] multiplied by 86400 to obtain mm day-1",
            "land_mask_shapefile": args.ne_shapefile,
            "history": "Created by barra_daily_lci_yearly_land.py",
        }
    )

    return ds_out


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    ensure_dir(args.outdir)

    # Load and regrid z0 once using a reference BARRA file for the target grid.
    print("Loading z0 and regridding to BARRA-R2 grid...")
    z0_raw = load_z0_static()
    _ref_files = find_daily_files_for_year(args.barra_root, "tas", args.start_year)
    if not _ref_files:
        raise FileNotFoundError(
            f"Cannot find a reference BARRA tas file for {args.start_year} to build z0 grid."
        )
    _ds_ref = xr.open_dataset(_ref_files[0])
    _tas_ref = subset_aus_bbox(_ds_ref["tas"])
    z0_barra = regrid_z0(z0_raw, _tas_ref.lat, _tas_ref.lon)
    _ds_ref.close()
    print(f"  z0 stats on BARRA grid: min={float(z0_barra.min()):.4f}  "
          f"mean={float(z0_barra.mean()):.4f}  max={float(z0_barra.max()):.4f} m")

    for year in range(args.start_year, args.end_year + 1):
        outfile = os.path.join(args.outdir, f"barra_lci_daily_aus_land_{year}.nc")

        if os.path.exists(outfile) and not args.overwrite:
            print(f"Skipping existing file: {outfile}")
            continue

        try:
            ds_out = process_year(year, args, z0_barra)
            tmpfile = outfile + ".tmp"
            print(f"Writing {outfile}")
            ds_out.to_netcdf(tmpfile, encoding=build_encoding(ds_out))
            os.replace(tmpfile, outfile)
            print(f"Done: {outfile}")
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            print(f"[WARN] Skipping year {year}")
        except Exception as e:
            print(f"[ERROR] Failed for year {year}: {repr(e)}")

    print("\nAll done.")


if __name__ == "__main__":
    main()