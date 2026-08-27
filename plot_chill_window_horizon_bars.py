#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_chill_window_horizon_bars.py

"Horizon bar" style figure: for each region, a light-blue bar shows the
historical chill-season window and a dark-blue bar shows the mid-century
window, both placed on a real calendar axis so that any shift or
narrowing is read directly off the dates.

Regions/season/threshold combos:
    Cobar-Lachlan       May-Oct   1100 kJ m-2 hr-1
    Yilgarn-Coolgardie  May-Oct   1200 kJ m-2 hr-1
    Winton              Nov-Mar   1000 kJ m-2 hr-1

Dates below are pasted directly from the region-averaged extraction
(lci_chill_season_timing_by_region_mid_century.csv), filtered to
NARCliM2/ssp370/2040-2060.

Future period and scenario: SSP3-7.0, 2040-2060 (NARCliM2 mid-century),
matching the period label in the underlying event-timing file.

Presentation conventions:
  - Panel titles show the month range rather than "dry"/"wet season".
  - Cobar-Lachlan and Yilgarn-Coolgardie (both May-Oct) share one x-axis
    range, so their bars are directly comparable at a glance rather than
    each panel auto-scaling independently. Winton (Nov-Mar) keeps its own
    range since it's not on the same calendar window.
  - +/-1 SD whiskers show ensemble spread (across the 8 NARCliM2 members)
    around each bar's start/end date. Values pasted from
    lci_chill_season_timing_by_region_mid_century.csv on 2026-07-26.

Output
------
PNG (300 dpi) saved to OUT_PATH.
"""

from __future__ import annotations

import os
import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================
# DATA
# ============================================================
# fut_first/fut_last are pasted from lci_chill_season_timing_by_region_mid_century.csv
# (dataset=NARCliM2, scenario=ssp370, period=2040-2060). past_first/past_last
# are NARCliM2 historical (1990-2014).
#
# *_std fields are +/-1 SD (ensemble spread across the 8 members, in days)
# for the corresponding first/last day, pasted from
# lci_chill_season_timing_by_region_mid_century.csv on 2026-07-26.
REGIONS = [
    {
        "name": "Cobar-Lachlan",
        "season": "dry",
        "threshold": 1100,
        "past_first": "17 Jun", "past_last": "16 Aug",
        "past_first_std": 6.1, "past_last_std": 11.2,
        "fut_first": "25 Jun", "fut_last": "29 Jul",
        "fut_first_std": 8.6, "fut_last_std": 11.7,
    },
    {
        "name": "Yilgarn-Coolgardie",
        "season": "dry",
        "threshold": 1200,
        "past_first": "6 Jul", "past_last": "20 Jul",
        "past_first_std": 6.8, "past_last_std": 7.6,
        "fut_first": "6 Jul", "fut_last": "15 Jul",
        "fut_first_std": 7.5, "fut_last_std": 8.6,
    },
    {
        "name": "Winton",
        "season": "wet",
        "threshold": 1000,
        # NOTE: these dates changed from an earlier pasting -- the old
        # 26 Jan/28 Feb (past) and 24 Jan/22 Feb (future) were from before
        # the April-exclusion fix reached this CSV. Re-verified against
        # lci_chill_season_timing_by_region_mid_century.csv on 2026-07-26.
        "past_first": "17 Jan", "past_last": "10 Feb",
        "past_first_std": 11.9, "past_last_std": 7.0,
        "fut_first": "18 Jan", "fut_last": "31 Jan",
        "fut_first_std": 12.2, "fut_last_std": 9.6,
    },
]

# Month range shown in each panel title, in place of "dry"/"wet season".
SEASON_MONTH_RANGE = {
    "dry": "May\u2013Oct",
    "wet": "Nov\u2013Mar",  # April excluded, matches the fix applied elsewhere in the pipeline
}

HIST_LABEL = "Historical (1990-2014)"
# The mid-century period is 2040-2060, matching the period_label carried by
# the NARCliM2 event-timing file this figure's dates are drawn from. Changing
# this label requires regenerating that file on the same period.
FUT_LABEL = "Future (SSP3-7.0, 2040-2060)"

REF_YEAR = 2001    # non-leap dummy year, dates are calendar-day-of-season only
PAD_DAYS = 10       # axis padding either side of the widest window per season group
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
HIST_COLOR = "#8FB8DE"   # light blue - past
FUT_COLOR = "#1B4F72"    # dark blue - future

OUT_PATH = "/scratch/dx2/rt9243/chill_projections/lci_event_timing/chill_window_horizon_bars.png"


# ============================================================
# HELPERS
# ============================================================

def parse_date(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(f"{s} {REF_YEAR}", "%d %b %Y")


# ============================================================
# PLOT
# ============================================================

def main() -> None:
    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(9, 6.6))

    # ---- Pass 1: parse all dates up front, so we can compute a shared
    # x-axis range per season group before drawing anything. ----
    parsed = []
    for reg in REGIONS:
        p0, p1 = parse_date(reg["past_first"]), parse_date(reg["past_last"])
        f0, f1 = parse_date(reg["fut_first"]), parse_date(reg["fut_last"])
        parsed.append((reg, p0, p1, f0, f1))

    # Group by season so Cobar-Lachlan and Yilgarn-Coolgardie (both May-Oct)
    # share one x-axis range and are directly comparable.
    # Winton is alone in "wet" so it keeps its own range either way.
    #
    # Padding must cover the largest +/-1 SD whisker in the group, not just
    # a flat PAD_DAYS -- otherwise a big std (member disagreement can be
    # 10+ days for some regions) pushes a whisker cap past the axis edge
    # and it gets silently clipped rather than drawn.
    group_xlim: dict[str, tuple[datetime.datetime, datetime.datetime]] = {}
    by_season = defaultdict(list)
    for reg, p0, p1, f0, f1 in parsed:
        by_season[reg["season"]].append((reg, p0, p1, f0, f1))
    for season, entries in by_season.items():
        max_std = max(
            std
            for reg, p0, p1, f0, f1 in entries
            for std in (
                reg["past_first_std"], reg["past_last_std"],
                reg["fut_first_std"], reg["fut_last_std"],
            )
        )
        pad = PAD_DAYS + max_std
        xmin = min(min(p0, f0) for reg, p0, p1, f0, f1 in entries) - datetime.timedelta(days=pad)
        xmax = max(max(p1, f1) for reg, p0, p1, f0, f1 in entries) + datetime.timedelta(days=pad)
        group_xlim[season] = (xmin, xmax)

    # ---- Pass 2: draw ----
    for panel_i, (ax, (reg, p0, p1, f0, f1)) in enumerate(zip(axes, parsed)):
        xmin, xmax = group_xlim[reg["season"]]

        # Past: light-blue fill. Future: dark-blue fill.
        ax.barh(1, (p1 - p0).days, left=p0, height=0.55,
                facecolor=HIST_COLOR, edgecolor="black", linewidth=1.0, zorder=3)
        ax.barh(0, (f1 - f0).days, left=f0, height=0.55,
                facecolor=FUT_COLOR, edgecolor="black", linewidth=1.0, zorder=3)

        # +/-1 SD whiskers at each bar's start/end date. std values are in
        # days (region-averaged ensemble spread across the 8 members).
        for d, std in ((p0, reg["past_first_std"]), (p1, reg["past_last_std"])):
            ax.errorbar([d], [1], xerr=[datetime.timedelta(days=std)],
                        fmt="none", ecolor="black", elinewidth=1.1, capsize=3.5, zorder=4)
        for d, std in ((f0, reg["fut_first_std"]), (f1, reg["fut_last_std"])):
            ax.errorbar([d], [0], xerr=[datetime.timedelta(days=std)],
                        fmt="none", ecolor="black", elinewidth=1.1, capsize=3.5, zorder=4)

        ax.axhline(-0.65, color="black", linewidth=1.0, xmin=0.02, xmax=0.98)

        for d in (p0, p1):
            ax.text(d, 1.40, d.strftime("%-d %b"), ha="center", va="bottom", fontsize=8.5)
        for d in (f0, f1):
            ax.text(d, -0.42, d.strftime("%-d %b"), ha="center", va="top", fontsize=8.5)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-1.0, 2.15)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

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
        ax.set_title(
            f"{reg['name']}  \u2014  {SEASON_MONTH_RANGE[reg['season']]}, "
            f"{reg['threshold']} kJ m\u207b\u00b2 hr\u207b\u00b9",
            loc="left", x=0.055, fontsize=11, pad=6,
        )

    legend_elems = [
        Patch(facecolor=HIST_COLOR, edgecolor="black", label=HIST_LABEL),
        Patch(facecolor=FUT_COLOR, edgecolor="black", label=FUT_LABEL),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Chill-season timing: historical vs future", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0.035, 1, 0.97])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"[OK] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()