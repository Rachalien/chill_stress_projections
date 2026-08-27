#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
component_lci_attribution.py

Attribute the projected change in severe (top-5% LCI) livestock cold-stress
conditions to its three drivers -- temperature, wind speed, and rainfall --
using the *analytic* LCI formula rather than an empirical regression.

Why analytic, not regression
----------------------------
An OLS regression of seasonal-max LCI on its drivers learns each sensitivity
from historical interannual variability. That fails in two opposite ways:
  * where a driver co-varies with circulation (e.g. warm dry-season years are
    quiet-circulation years), its learned slope is inflated -> the forced
    change is over-predicted;
  * where a driver barely varies in the events that set the maximum (e.g.
    temperature in Cobar-Lachlan's reliably-cold dry-season outbreaks), its
    slope is unidentifiable (~0) -> the forced change is under-predicted and
    dumped into the residual.
Because LCI is a known deterministic function, we do not need to *learn* the
sensitivities -- we read them straight off the formula's partial derivatives,
evaluated at the actual severe-day conditions. Same physics everywhere, no
variance-dependence, and a residual that is genuine formula curvature rather
than a natural-variability-vs-forced mismatch.

LCI (Nixon-Smith / Donnelly), matching calculate_lci() in the pipeline:
    LCI = (11.7 + 3.1*sqrt(V)) * (40 - T) + 481 + 418*(1 - exp(-0.04*R))
with T in degC, V in m/s (2 m), R in mm/day. The +481 constant is irrelevant
to any *change* and to all derivatives (it cancels / differentiates away).

Two tiers
---------
Tier 1  Decompose f(mu_F) - f(mu_H), where mu are the event-day-mean
        regional-mean drivers, via:
          - exact Shapley over {T, V, R}  (sums exactly to f(mu_F)-f(mu_H));
          - first-order partial derivatives (for intuition; small residual).
Tier 2  Reconcile to the *true* event-day-mean reported regional LCI. Because
        LCI is nonlinear, E[f] = f(mu) + 1/2 * tr(H . Cov) + higher order,
        and for this formula the only non-zero Hessian entries are V-V, R-R
        and the T-V cross term. We add that second-order (Jensen) correction
        and report the higher-order remainder. The full budget closes:
          Shapley(T,V,R) + Jensen(VV,RR,TV) + higher_resid = true Delta LCI.

        NOTE: the variances/covariances used here are *temporal* (across event
        days) on the regional-mean series. Within-region spatial nonlinearity
        is not modelled and will appear in the higher-order remainder. If that
        remainder is large for any region, we move to the grid-level variant
        (Tier 2b) -- but the remainder tells us whether that is necessary.

Outputs
-------
  {OUTDIR}/component_lci_attribution.csv   tidy results, all terms in kJ and %
  {OUTDIR}/component_lci_curves_wet.png    driver-response curves (wet season)
  {OUTDIR}/component_lci_curves_dry.png    driver-response curves (dry season)

Usage
-----
  python component_lci_attribution.py --self-test     # math only, no data
  python component_lci_attribution.py --check-only    # validate paths/masks
  python component_lci_attribution.py --members ACCESS-ESM1-5_WRF412R3
  python component_lci_attribution.py                 # all 8 members

PBS (suggested)
---------------
  #PBS -q normal
  #PBS -l walltime=04:00:00
  #PBS -l mem=32GB
  #PBS -l ncpus=4
  #PBS -l storage=gdata/ia39+gdata/dx2+scratch/dx2
  module use /g/data/xp65/public/modules
  module load conda/analysis3
  python component_lci_attribution.py

This script imports loaders/masks/config from synoptic_composite_lci_top5pct.py,
which must be importable (same directory or on PYTHONPATH).
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import logging

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reuse the tested pipeline ────────────────────────────────────────────────
from synoptic_composite_lci_top5pct import (
    MEMBERS,
    REGION_SPECS,
    HIST_LOAD_YEARS,
    FUTURE_LOAD_YEARS,
    HIST_SEASON_RANGE,
    FUTURE_SEASON_RANGE,
    TOP_PCT,
    open_var_lazy,
    time_strings,
    build_region_mask,
    area_weights_2d,
    annotate_seasons,
    find_top_pct_events,
    find_var_files,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("attribution")

OUTDIR = "/scratch/dx2/rt9243/chill_projections/component_attribution"

REGION_NAMES = list(REGION_SPECS.keys())            # Winton, Yilgarn_Coolgardie, Cobar_Lachlan
SEASONS      = ("wet", "dry")
VARS_LOAD    = ["tasmaxAdjust", "tasminAdjust", "sfcWindAdjust", "prAdjust"]

# Pretty labels for printing/plotting
PRETTY = {"Winton": "Winton",
          "Yilgarn_Coolgardie": "Yilgarn-Coolgardie",
          "Cobar_Lachlan": "Cobar-Lachlan"}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LCI FORMULA, DERIVATIVES, AND DECOMPOSITIONS  (the cheap, exact part)     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def lci(T, V, R):
    """LCI (kJ m-2 hr-1). Vectorised; matches pipeline calculate_lci()."""
    V = np.maximum(V, 0.0)
    R = np.maximum(R, 0.0)
    return (11.7 + 3.1 * np.sqrt(V)) * (40.0 - T) + 481.0 + 418.0 * (1.0 - np.exp(-0.04 * R))


def lci_gradient(T, V, R):
    """First partial derivatives (dLCI/dT, dLCI/dV, dLCI/dR)."""
    Vs = np.sqrt(max(V, 1e-9))
    fT = -(11.7 + 3.1 * Vs)
    fV = (1.55 / Vs) * (40.0 - T)
    fR = 16.72 * np.exp(-0.04 * max(R, 0.0))
    return np.array([fT, fV, fR])


def lci_hessian_nonzero(T, V, R):
    """Non-zero second derivatives: (f_VV, f_RR, f_TV). All others are 0."""
    Vs = np.sqrt(max(V, 1e-9))
    f_VV = -0.775 * (40.0 - T) / (max(V, 1e-9) ** 1.5)
    f_RR = -0.6688 * np.exp(-0.04 * max(R, 0.0))
    f_TV = -1.55 / Vs
    return f_VV, f_RR, f_TV


def shapley_decomposition(muH, muF):
    """
    Exact Shapley attribution of f(muF) - f(muH) to {T, V, R}.
    muH, muF are dicts with keys 'T','V','R'. Returns dict of contributions
    (kJ) that sum exactly to lci(muF) - lci(muH).
    """
    keys = ["T", "V", "R"]
    phi = {k: 0.0 for k in keys}
    perms = list(itertools.permutations(keys))
    for perm in perms:
        cur = dict(muH)
        base = lci(cur["T"], cur["V"], cur["R"])
        for k in perm:
            cur[k] = muF[k]
            new = lci(cur["T"], cur["V"], cur["R"])
            phi[k] += (new - base)
            base = new
    n = len(perms)
    return {k: phi[k] / n for k in keys}


def analytic_first_order(muH, muF):
    """
    First-order (midpoint) partial-derivative attribution, for intuition.
    Returns (contribs dict, residual) where residual = f(muF)-f(muH) - sum.
    Midpoint gradient is used for second-order accuracy.
    """
    gH = lci_gradient(muH["T"], muH["V"], muH["R"])
    gF = lci_gradient(muF["T"], muF["V"], muF["R"])
    g = 0.5 * (gH + gF)
    d = np.array([muF["T"] - muH["T"], muF["V"] - muH["V"], muF["R"] - muH["R"]])
    contrib = g * d
    total = lci(muF["T"], muF["V"], muF["R"]) - lci(muH["T"], muH["V"], muH["R"])
    residual = total - contrib.sum()
    return {"T": contrib[0], "V": contrib[1], "R": contrib[2]}, residual


def jensen_term(mu, cov):
    """
    Second-order (Jensen) correction to E[f] at conditions mu with 3x3
    covariance cov (order [T,V,R]). Returns (total, dict of sub-terms).
        E[f] ~= f(mu) + 1/2 ( f_VV Var(V) + f_RR Var(R) + 2 f_TV Cov(T,V) )
    """
    f_VV, f_RR, f_TV = lci_hessian_nonzero(mu["T"], mu["V"], mu["R"])
    varV, varR, covTV = cov[1, 1], cov[2, 2], cov[0, 1]
    term_VV = 0.5 * f_VV * varV
    term_RR = 0.5 * f_RR * varR
    term_TV = f_TV * covTV          # the 1/2 * 2 cancels for the cross term
    return term_VV + term_RR + term_TV, {"VV": term_VV, "RR": term_RR, "TV": term_TV}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA: regional-mean daily driver series (reuses pipeline loaders)         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_region_weight_da(lat, lon):
    """One area-weight DataArray + total weight per region (built once)."""
    coords = {"lat": lat, "lon": lon}
    wda, tot = {}, {}
    for rname, spec in REGION_SPECS.items():
        mask = build_region_mask(spec, lat, lon)
        w2d = area_weights_2d(lat, mask)
        wda[rname] = xr.DataArray(w2d, dims=["lat", "lon"], coords=coords)
        tot[rname] = float(w2d.sum())
        log.info(f"  region {rname}: {int(mask.sum())} grid cells, total weight {tot[rname]:.2f}")
        if tot[rname] <= 0:
            raise RuntimeError(f"Region {rname} has zero weight -- check shapefile/values")
    return wda, tot


def compute_regional_driver_series(model, variant, rcm, experiment, years,
                                   region_weight_da, region_total_w):
    """
    Area-weighted regional-mean daily T (degC), V (m/s), R (mm/day), LCI for
    every region, computed in a single Dask reduction. Mirrors the lazy graph
    of compute_lci_regional_series() but keeps the driver means too.
    Returns a DataFrame indexed 0..N with columns:
        date, <region>__T, <region>__V, <region>__R, <region>__LCI  (all regions)
    """
    ckw = {"time": 30, "lat": -1, "lon": -1}
    da_tasmax = open_var_lazy(model, variant, rcm, experiment, "tasmaxAdjust", years, ckw)
    da_tasmin = open_var_lazy(model, variant, rcm, experiment, "tasminAdjust", years, ckw)
    da_ws     = open_var_lazy(model, variant, rcm, experiment, "sfcWindAdjust", years, ckw)
    da_pr     = open_var_lazy(model, variant, rcm, experiment, "prAdjust",      years, ckw)

    dates = time_strings(da_tasmax)

    ws_safe = da_ws.clip(min=0.0)
    pr_safe = da_pr.clip(min=0.0)
    T_C = (da_tasmax + da_tasmin) * 0.5
    lci_da = ((11.7 + 3.1 * np.sqrt(ws_safe)) * (40.0 - T_C)
              + 481.0 + 418.0 * (1.0 - np.exp(-0.04 * pr_safe)))

    lazy = {}
    for rname, w_da in region_weight_da.items():
        tw = region_total_w[rname]
        lazy[f"{rname}__T"]   = (T_C    * w_da).sum(["lat", "lon"]) / tw
        lazy[f"{rname}__V"]   = (ws_safe * w_da).sum(["lat", "lon"]) / tw
        lazy[f"{rname}__R"]   = (pr_safe * w_da).sum(["lat", "lon"]) / tw
        lazy[f"{rname}__LCI"] = (lci_da  * w_da).sum(["lat", "lon"]) / tw

    log.info(f"  reducing regional driver series | {experiment} | {min(years)}-{max(years)}")
    ds = xr.Dataset(lazy).compute()   # one read of each variable

    out = {"date": dates}
    for k in lazy:
        out[k] = ds[k].values
    return pd.DataFrame(out)


def collect_member_events(driver_df):
    """
    Given one member's concatenated driver_df (hist + future), find the top-5%
    event days per region/season/period and return event-day driver rows.

    Returns dict keyed by (region, season, period) -> ndarray (n_days, 4)
    with columns [T, V, R, LCI].
    """
    # Build the wide LCI frame that find_top_pct_events expects.
    ann = annotate_seasons(driver_df["date"])
    lci_wide = pd.DataFrame({
        "date":        driver_df["date"].values,
        "season":      ann["season"].values,
        "season_year": ann["season_year"].values,
    })
    for r in REGION_NAMES:
        lci_wide[r] = driver_df[f"{r}__LCI"].values

    events = find_top_pct_events(lci_wide, REGION_NAMES)  # (date, region, season, period)

    # Fast per-date lookup of driver values.
    idx = driver_df.set_index("date")
    pools = {}
    for date, region, season, period in events:
        row = idx.loc[date]
        vals = np.array([row[f"{region}__T"], row[f"{region}__V"],
                         row[f"{region}__R"], row[f"{region}__LCI"]], dtype=float)
        pools.setdefault((region, season, period), []).append(vals)
    return {k: np.vstack(v) for k, v in pools.items()}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DRIVER                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run(members):
    os.makedirs(OUTDIR, exist_ok=True)

    # Grid + masks once (AUST-05i grid is shared across members).
    m0 = members[0]
    model, variant, rcm = MEMBERS[m0]
    sample = open_var_lazy(model, variant, rcm, "historical", "tasmaxAdjust",
                           HIST_LOAD_YEARS, {"time": 1, "lat": -1, "lon": -1})
    lat, lon = sample.lat, sample.lon
    region_weight_da, region_total_w = build_region_weight_da(lat, lon)

    # Pool event-day driver rows across members.
    pooled = {}   # (region, season, period) -> list of ndarrays
    for mlabel in members:
        model, variant, rcm = MEMBERS[mlabel]
        log.info(f"[{mlabel}] computing regional driver series")
        hist = compute_regional_driver_series(model, variant, rcm, "historical",
                                              HIST_LOAD_YEARS, region_weight_da, region_total_w)
        fut  = compute_regional_driver_series(model, variant, rcm, "ssp370",
                                              FUTURE_LOAD_YEARS, region_weight_da, region_total_w)
        driver_df = pd.concat([hist, fut], ignore_index=True)
        member_pools = collect_member_events(driver_df)
        for key, arr in member_pools.items():
            pooled.setdefault(key, []).append(arr)

    pooled = {k: np.vstack(v) for k, v in pooled.items()}

    # Decompose per region/season.
    rows = []
    stats = {}   # (region, season) -> dict for plotting
    for region in REGION_NAMES:
        for season in SEASONS:
            kH = (region, season, "hist")
            kF = (region, season, "future")
            if kH not in pooled or kF not in pooled:
                log.warning(f"missing pool for {region}/{season}; skipping")
                continue

            H, F = pooled[kH], pooled[kF]                  # (n,4): T,V,R,LCI
            muH = {"T": H[:, 0].mean(), "V": H[:, 1].mean(), "R": H[:, 2].mean()}
            muF = {"T": F[:, 0].mean(), "V": F[:, 1].mean(), "R": F[:, 2].mean()}
            covH = np.cov(H[:, :3], rowvar=False)
            covF = np.cov(F[:, :3], rowvar=False)
            lciH_true = H[:, 3].mean()                     # reported event-day mean LCI
            lciF_true = F[:, 3].mean()
            true_delta = lciF_true - lciH_true

            # Tier 1
            shap = shapley_decomposition(muH, muF)         # sums to f(muF)-f(muH)
            ana, ana_res = analytic_first_order(muH, muF)
            of_means_delta = lci(muF["T"], muF["V"], muF["R"]) - lci(muH["T"], muH["V"], muH["R"])

            # Tier 2: Jensen reconciliation to the true reported metric
            jH, jH_parts = jensen_term(muH, covH)
            jF, jF_parts = jensen_term(muF, covF)
            dJ = jF - jH
            dJ_parts = {k: jF_parts[k] - jH_parts[k] for k in jF_parts}
            higher_resid = true_delta - (of_means_delta + dJ)

            denom = true_delta if abs(true_delta) > 1e-9 else np.nan
            rows.append({
                "region": PRETTY[region], "season": season,
                "n_hist": H.shape[0], "n_future": F.shape[0],
                "T_hist": muH["T"], "T_future": muF["T"], "dT": muF["T"] - muH["T"],
                "V_hist": muH["V"], "V_future": muF["V"], "dV": muF["V"] - muH["V"],
                "R_hist": muH["R"], "R_future": muF["R"], "dR": muF["R"] - muH["R"],
                "LCI_hist": lciH_true, "LCI_future": lciF_true, "true_dLCI": true_delta,
                "shapley_T": shap["T"], "shapley_V": shap["V"], "shapley_R": shap["R"],
                "analytic_T": ana["T"], "analytic_V": ana["V"], "analytic_R": ana["R"],
                "analytic_resid": ana_res,
                "jensen_VV": dJ_parts["VV"], "jensen_RR": dJ_parts["RR"],
                "jensen_TV": dJ_parts["TV"], "jensen_total": dJ,
                "higher_resid": higher_resid,
                "pct_T": 100 * shap["T"] / denom,
                "pct_V": 100 * shap["V"] / denom,
                "pct_R": 100 * shap["R"] / denom,
                "pct_jensen": 100 * dJ / denom,
                "pct_higher_resid": 100 * higher_resid / denom,
            })
            stats[(region, season)] = dict(muH=muH, muF=muF)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTDIR, "component_lci_attribution.csv")
    df.to_csv(csv_path, index=False)
    log.info(f"wrote {csv_path}")
    _print_summary(df)

    for season in SEASONS:
        _make_curve_plot(season, stats)
    _make_decomposition_bar_plot(df)

    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  REPORTING + DIAGNOSTIC PLOT                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _print_summary(df):
    print("\n" + "=" * 78)
    print("ANALYTIC LCI ATTRIBUTION  (Shapley, kJ m-2 hr-1; % of true Delta LCI)")
    print("=" * 78)
    for _, r in df.iterrows():
        print(f"\n{r['region']:>20s} | {r['season']} | true dLCI = {r['true_dLCI']:7.1f} kJ"
              f"  (n_hist={int(r['n_hist'])}, n_fut={int(r['n_future'])})")
        print(f"    Shapley   T {r['shapley_T']:7.1f} ({r['pct_T']:5.0f}%)"
              f"   V {r['shapley_V']:6.1f} ({r['pct_V']:4.0f}%)"
              f"   R {r['shapley_R']:6.1f} ({r['pct_R']:4.0f}%)")
        print(f"    Jensen    VV {r['jensen_VV']:6.1f}   RR {r['jensen_RR']:6.1f}"
              f"   TV(interaction) {r['jensen_TV']:6.1f}   [total {r['pct_jensen']:4.0f}%]")
        print(f"    higher-order remainder {r['higher_resid']:6.1f} ({r['pct_higher_resid']:4.0f}%)")
    print("\nBudget check: Shapley(T+V+R) + Jensen(total) + higher_resid = true dLCI")
    print("=" * 78 + "\n")


def _make_curve_plot(season, stats):
    drivers = [("T", "Temperature (degC)", 0),
               ("V", "Wind speed (m/s)", 1),
               ("R", "Rainfall (mm/day)", 2)]
    fig, axes = plt.subplots(len(REGION_NAMES), 3, figsize=(13, 11))
    if len(REGION_NAMES) == 1:
        axes = axes[np.newaxis, :]

    for i, region in enumerate(REGION_NAMES):
        key = (region, season)
        if key not in stats:
            for j in range(3):
                axes[i, j].axis("off")
            continue
        muH, muF = stats[key]["muH"], stats[key]["muF"]
        for j, (dk, dlabel, _) in enumerate(drivers):
            ax = axes[i, j]
            # Sweep the focal driver; hold the other two at the historical mean
            # (solid) and the future mean (dashed) to show the curve and its shift.
            lo = min(muH[dk], muF[dk])
            hi = max(muH[dk], muF[dk])
            pad = 0.35 * (hi - lo + 1e-6) + (1.0 if dk != "R" else 2.0)
            xs = np.linspace(lo - pad, hi + pad, 200)

            def curve(at):
                vals = {"T": at["T"], "V": at["V"], "R": at["R"]}
                yy = []
                for x in xs:
                    vals[dk] = x
                    yy.append(lci(vals["T"], vals["V"], vals["R"]))
                return np.array(yy)

            ax.plot(xs, curve(muH), "-", lw=1.8, color="#444", label="others @ hist")
            ax.plot(xs, curve(muF), "--", lw=1.2, color="#999", label="others @ future")
            yH = lci(muH["T"], muH["V"], muH["R"])
            yF = lci(muF["T"], muF["V"], muF["R"])
            ax.plot(muH[dk], yH, "o", ms=9, color="#1f6fb2", label="historical")
            ax.plot(muF[dk], yF, "o", ms=9, color="#c0392b", label="end-of-century")
            ax.annotate("", xy=(muF[dk], yF), xytext=(muH[dk], yH),
                        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2))
            if j == 0:
                ax.set_ylabel(f"{PRETTY[region]}\nLCI (kJ m$^{{-2}}$ hr$^{{-1}}$)", fontsize=9)
            if i == len(REGION_NAMES) - 1:
                ax.set_xlabel(dlabel, fontsize=9)
            if i == 0 and j == 2:
                ax.legend(fontsize=7, loc="best")
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.25)

    fig.suptitle(f"LCI sensitivity to each driver, with event-day operating points "
                 f"sliding hist -> end-of-century  ({'Nov-Mar' if season=='wet' else 'May-Oct'})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTDIR, f"component_lci_curves_{season}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log.info(f"wrote {out}")

def _make_decomposition_bar_plot(df):
    """Waterfall bar chart: driver contributions stacking to the true dLCI."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    order = ["Winton", "Cobar-Lachlan", "Yilgarn-Coolgardie"]
    focal = {"Winton": "wet", "Cobar-Lachlan": "dry", "Yilgarn-Coolgardie": "dry"}
    col = {"T": "#c0392b", "R": "#2471a3", "V": "#27ae60", "N": "#95a5a6"}
    order = [r for r in order if r in set(df["region"])]

    fig, axes = plt.subplots(1, len(order), figsize=(4.4 * len(order), 5.2), sharey=True)
    if len(order) == 1:
        axes = [axes]
    for ax, region in zip(axes, order):
        sub = {r["season"]: r for _, r in df[df["region"] == region].iterrows()}
        for xi, season in enumerate(["wet", "dry"]):
            if season not in sub:
                continue
            r = sub[season]
            rem = r["jensen_total"] + r["higher_resid"]
            run = 0.0
            for key, val in [("T", r["shapley_T"]), ("R", r["shapley_R"]),
                             ("V", r["shapley_V"]), ("N", rem)]:
                ax.bar(xi, val, bottom=run, width=0.62, color=col[key],
                       edgecolor="white", linewidth=0.7, zorder=2)
                run += val
            ax.plot(xi, r["true_dLCI"], "o", ms=11, color="black", zorder=4)
            if abs(r["true_dLCI"]) > 1e-9:
                pct = 100 * r["shapley_T"] / r["true_dLCI"]
                ax.text(xi, r["shapley_T"] / 2, f"{pct:.0f}%", ha="center",
                        va="center", color="white", fontsize=10, fontweight="bold", zorder=3)
        ax.axhline(0, color="#555", lw=1)
        labels = []
        for s in ["wet", "dry"]:
            tag = "Nov-Mar" if s == "wet" else "May-Oct"
            if focal.get(region) == s:
                tag += "\n(focal)"
            labels.append(tag)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(region, fontsize=12, fontweight="bold")
        ax.set_xlim(-0.6, 1.6); ax.grid(axis="y", alpha=0.25, zorder=0)
    axes[0].set_ylabel("Delta LCI (kJ m$^{-2}$ hr$^{-1}$)", fontsize=11)
    handles = [Patch(fc=col["T"], label="Temperature"),
               Patch(fc=col["R"], label="Rainfall"),
               Patch(fc=col["V"], label="Wind"),
               Patch(fc=col["N"], label="Nonlinear remainder"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
                      markersize=10, label="True dLCI")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Analytic (Shapley) decomposition of projected LCI change  -  "
                 "end-of-century, SSP3-7.0", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out = os.path.join(OUTDIR, "component_lci_decomposition_bars.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"wrote {out}")
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SELF-TEST / CHECK-ONLY / CLI                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def self_test():
    """Verify the decomposition algebra on synthetic conditions (no data)."""
    print("Self-test: Shapley exactness and budget closure")
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(5):
        muH = {"T": rng.uniform(0, 12), "V": rng.uniform(2, 9), "R": rng.uniform(0, 15)}
        muF = {"T": muH["T"] + rng.uniform(2, 5), "V": muH["V"] + rng.uniform(-0.3, 0.3),
               "R": muH["R"] + rng.uniform(-3, 1)}
        shap = shapley_decomposition(muH, muF)
        total = lci(muF["T"], muF["V"], muF["R"]) - lci(muH["T"], muH["V"], muH["R"])
        err = abs(sum(shap.values()) - total)
        print(f"  Shapley sum {sum(shap.values()):8.3f} vs f(muF)-f(muH) {total:8.3f}  err {err:.2e}")
        ok = ok and err < 1e-9
    # Finite-difference check of the analytic gradient at one point.
    T, V, R, h = 5.0, 6.0, 8.0, 1e-4
    g = lci_gradient(T, V, R)
    gnum = np.array([
        (lci(T + h, V, R) - lci(T - h, V, R)) / (2 * h),
        (lci(T, V + h, R) - lci(T, V - h, R)) / (2 * h),
        (lci(T, V, R + h) - lci(T, V, R - h)) / (2 * h),
    ])
    print(f"  gradient analytic {g}  vs numeric {gnum}  max err {np.abs(g-gnum).max():.2e}")
    ok = ok and np.abs(g - gnum).max() < 1e-4
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


def check_only(members):
    print("Check-only: validating paths, files, and region masks")
    shp = REGION_SPECS[REGION_NAMES[0]]["shp"]
    print(f"  shapefile exists: {os.path.exists(shp)}  ({shp})")
    m0 = members[0]
    model, variant, rcm = MEMBERS[m0]
    files = find_var_files(model, variant, rcm, "historical", "tasmaxAdjust")
    print(f"  found {len(files)} tasmaxAdjust files for {m0} historical")
    if not files:
        print("  ERROR: no input files found -- check NARCLIM_ROOT / member spec")
        return False
    sample = open_var_lazy(model, variant, rcm, "historical", "tasmaxAdjust",
                           HIST_LOAD_YEARS, {"time": 1, "lat": -1, "lon": -1})
    lat, lon = sample.lat, sample.lon
    print(f"  grid: {lat.size} lat x {lon.size} lon")
    build_region_weight_da(lat, lon)   # logs cell counts, raises on zero weight
    print(f"  output dir: {OUTDIR}")
    print("CHECK-ONLY PASSED")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", default=list(MEMBERS.keys()),
                    help="subset of member labels (default: all 8)")
    ap.add_argument("--check-only", action="store_true",
                    help="validate paths/files/masks and exit")
    ap.add_argument("--self-test", action="store_true",
                    help="run decomposition math self-test (no data) and exit")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    bad = [m for m in args.members if m not in MEMBERS]
    if bad:
        ap.error(f"unknown members: {bad}\nvalid: {list(MEMBERS.keys())}")

    if args.check_only:
        sys.exit(0 if check_only(args.members) else 1)

    run(args.members)


if __name__ == "__main__":
    main()