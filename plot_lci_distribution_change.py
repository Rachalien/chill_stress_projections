#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_lci_distribution_change.py

Diagnostic figure showing the full distribution of daily regional
LCI values for each focal region (restricted to that region's relevant
season), historical vs mid-century, with the p95 ("top-5% event") cutoff
marked as a vertical line -- so the threshold used to define composite
events can be seen against the distribution it's drawn from.

Also prints a translation-vs-deformation spread check to console
((p95-p50) for each period) -- doesn't appear on the figure itself.

Regions and their relevant season (matches synoptic_composite_lci_top5pct.py
and the rest of the paper's focal-region convention):
    Cobar-Lachlan        May-Oct
    Yilgarn-Coolgardie   May-Oct
    Winton               Nov-Mar (April excluded)

IMPORTANT CAVEAT
----------------------------------------------------
The p95 lines shown here are computed from all 8 members' daily values
POOLED together, purely for a descriptive/diagnostic picture of "where does
the cutoff sit relative to the distribution". This is NOT the same
threshold actually used to define events in the paper's composites --
synoptic_composite_lci_top5pct.py computes p95 separately PER MEMBER (each
model has its own bias/distribution), so the pooled line here will be close
to, but not numerically identical to, the average of the 8 per-member
thresholds. The pooled value is descriptive only and should not be quoted
as the operational threshold.

Inputs
------
Reads the per-member daily LCI CSVs written by
synoptic_composite_lci_top5pct.py:
    {OUTDIR}/daily_lci/{member}_regional_daily_lci.csv
Columns: date, Winton, Yilgarn_Coolgardie, Cobar_Lachlan, season,
         season_year, period (hist/future)

Usage
-----
    module use /g/data/xp65/public/modules
    module load conda/analysis3

    python plot_lci_distribution_change.py
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DAILY_DIR = "/scratch/dx2/rt9243/chill_projections/synoptic_composites/daily_lci"
DEFAULT_FIGDIR = "/scratch/dx2/rt9243/chill_projections/synoptic_composites/figures"

HIST_SEASON_RANGE = (1990, 2014)
FUTURE_SEASON_RANGE = (2040, 2060)
TOP_PCT = 5  # matches synoptic_composite_lci_top5pct.py

REGIONS = [
    {"col": "Cobar_Lachlan", "label": "Cobar-Lachlan", "season": "dry",
     "month_range": "May-Oct", "xlim_group": "dry"},
    {"col": "Yilgarn_Coolgardie", "label": "Yilgarn-Coolgardie", "season": "dry",
     "month_range": "May-Oct", "xlim_group": "dry"},
    {"col": "Winton", "label": "Winton", "season": "wet",
     "month_range": "Nov-Mar", "xlim_group": "wet"},
]

# Panels (a) and (b) share a single x-axis range so the two May-Oct regions are
# directly comparable. Groups without an explicit override below get
# their range from the data, spanning every region in the group.
#
# Winton is cropped instead, to 500-1050. Change the second value to 1000
# for a tighter crop of the upper tail.
XLIM_OVERRIDE = {"wet": (500.0, 1050.0)}
XLIM_PAD = 25.0

# Matches the light-blue-past / dark-blue-future convention used elsewhere
# in the paper's figures.
HIST_COLOR = "#8FB8DE"
FUT_COLOR = "#1B4F72"
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"

def load_pooled_daily(daily_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(daily_dir, "*_regional_daily_lci.csv")))
    if not files:
        raise FileNotFoundError(f"No daily LCI CSVs found in {daily_dir}")
    if len(files) != 8:
        print(f"[WARN] expected 8 member CSVs, found {len(files)}: {files}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["member"] = os.path.basename(f).replace("_regional_daily_lci.csv", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def region_period_values(df: pd.DataFrame, region_col: str, season: str, period: str) -> np.ndarray:
    yr0, yr1 = HIST_SEASON_RANGE if period == "hist" else FUTURE_SEASON_RANGE
    sub = df[
        (df["season"] == season) &
        (df["season_year"] >= yr0) &
        (df["season_year"] <= yr1) &
        (df["period"] == period)
    ]
    return sub[region_col].dropna().values


def plot_distributions(daily_dir: str, figdir: str, dpi: int, outfile: str) -> Path:
    os.makedirs(figdir, exist_ok=True)
    df = load_pooled_daily(daily_dir)

    # Pre-pass: pull every region's values once, so panels sharing an
    # xlim_group can be placed on a common axis range before anything is
    # drawn. The arrays are cached here and reused in the draw loop rather
    # than being extracted from the dataframe a second time.
    values = {
        reg["col"]: (
            region_period_values(df, reg["col"], reg["season"], "hist"),
            region_period_values(df, reg["col"], reg["season"], "future"),
        )
        for reg in REGIONS
    }

    group_xlim: dict[str, tuple[float, float]] = {}
    for reg in REGIONS:
        g = reg["xlim_group"]
        if g in XLIM_OVERRIDE:
            continue
        h, f = values[reg["col"]]
        lo, hi = float(min(h.min(), f.min())), float(max(h.max(), f.max()))
        if g in group_xlim:
            lo, hi = min(lo, group_xlim[g][0]), max(hi, group_xlim[g][1])
        group_xlim[g] = (lo, hi)
    group_xlim = {g: (lo - XLIM_PAD, hi + XLIM_PAD)
                  for g, (lo, hi) in group_xlim.items()}
    group_xlim.update(XLIM_OVERRIDE)

    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(8, 9.5))

    print("\n[Translation vs deformation check: p95-p50 spread, hist vs future]")
    for panel_i, (ax, reg) in enumerate(zip(axes, REGIONS)):
        hist_vals, fut_vals = values[reg["col"]]

        p95_hist = np.percentile(hist_vals, 100 - TOP_PCT)
        p95_fut = np.percentile(fut_vals, 100 - TOP_PCT)

        bins = np.linspace(
            min(hist_vals.min(), fut_vals.min()),
            max(hist_vals.max(), fut_vals.max()),
            45,
        )

        ax.hist(hist_vals, bins=bins, density=True, alpha=0.55,
                color=HIST_COLOR, edgecolor="none", label="Historical (1990-2014)")
        ax.hist(fut_vals, bins=bins, density=True, alpha=0.55,
                color=FUT_COLOR, edgecolor="none", label="Mid-century (2040-2060)")

        ax.axvline(p95_hist, color=HIST_COLOR, linestyle="--", linewidth=1.8)
        ax.axvline(p95_fut, color=FUT_COLOR, linestyle="--", linewidth=1.8)

        ymax = ax.get_ylim()[1] * 1.15
        ax.set_ylim(top=ymax)

        cutoffs = sorted(
            [(p95_hist, HIST_COLOR), (p95_fut, FUT_COLOR)],
            key=lambda t: t[0],
        )
        for (value, colour), side, offset in zip(cutoffs, ("right", "left"), (-8, 8)):
            ax.annotate(
                f"p95={value:.0f}",
                xy=(value, ymax * 0.99),
                xytext=(offset, 0),
                textcoords="offset points",
                color=colour,
                ha=side,
                va="top",
                fontsize=10,
                fontweight="bold",
                rotation=90,
            )

        # Panel letter sits on the title line rather than inside the axes:
        # the upper right holds the p95 labels and the upper left is where
        # Winton's peak sits. annotate with va="baseline" and a 6-point
        # offset matches how set_title places its text, so letter and title
        # share a baseline.
        ax.annotate(
            f"({PANEL_LETTERS[panel_i]})",
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(0, 6),
            textcoords="offset points",
            ha="left",
            va="baseline",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_title(reg["label"], fontsize=12, loc="left", x=0.055,
                     fontweight="bold")
        ax.set_xlim(*group_xlim[reg["xlim_group"]])
        ax.set_xlabel("Daily regional-mean LCI (kJ m$^{-2}$ hr$^{-1}$)", fontsize=10.5)
        ax.set_ylabel("Density", fontsize=10.5)
        # Legend entries are identical across panels, so they are collected
        # once here and drawn as a single figure-level legend below. Keeping
        # a per-axes legend puts it in the upper right, directly over the p95
        # labels, and the upper left is occupied by Winton's peak.
        if ax is axes[0]:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # Season window and sample sizes are no longer on the figure, so they
        # go to console for the caption instead of being lost.
        print(f"  {reg['label']:20s} {reg['month_range']}: "
              f"n={len(hist_vals)} hist, {len(fut_vals)} future daily values "
              f"(pooled across 8 members)")

        spread_hist = p95_hist - np.percentile(hist_vals, 50)
        spread_fut = p95_fut - np.percentile(fut_vals, 50)
        print(f"  {reg['label']:20s} hist p95-p50={spread_hist:6.1f}   "
              f"future p95-p50={spread_fut:6.1f}   "
              f"diff={spread_fut - spread_hist:+.1f}")

    fig.suptitle("Daily LCI distributions by region", fontsize=13.5, y=0.995, fontweight="bold")
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center", ncol=2, frameon=False, fontsize=10.5,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.965])

    outpath = Path(figdir) / outfile
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outpath


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--daily-dir", default=DEFAULT_DAILY_DIR)
    ap.add_argument("--figdir", default=DEFAULT_FIGDIR)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--outfile", default="lci_distributions_p95_by_region.png")
    args = ap.parse_args()

    outpath = plot_distributions(args.daily_dir, args.figdir, args.dpi, args.outfile)
    print(f"[OK] wrote {outpath}")


if __name__ == "__main__":
    main()