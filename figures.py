# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
figures.py  -  the publication figure suite (one shared design system)
======================================================================
Every figure is rendered through the shared style in config.apply_style,
with a single palette, consistent type, neutral titles, panel tags, and
300-dpi PDF+PNG output (config.savefig). figures.py owns Figs 1-5 and 7-8;
Fig 6, the solver-verification figure, is produced by solver_study.py from
its own run data, in the same style.

  fig_framework        Fig 1  framework architecture and data flow
  fig_study_area       Fig 2  RI inventory map (a), condition-state (b) and
                               scour-severity (c) distributions, merged
  fig_screen           Fig 3  objective screen: cost-carbon aligned | cost-adaptation unaligned
  fig_carbon_floor     Fig 4  emissions: do-nothing, cost-optimal, carbon floor
  fig_knee             Fig 5  marginal cost of scour protection (a), knee;
                               (b) swept across the countermeasure-cost range, merged
  fig_robustness       Fig 7  priority-alignment Monte Carlo (screen stability)
  fig_embodied_threshold  Fig 8  two-sided validation: embodied-carbon threshold sweep
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches, matplotlib.colors

import config as C
import problem as P
import optimizer as OPT
import adaptation as A

C.apply_style()
PAL = C.PALETTE
CMAP = C._iam_cmap()


def _scour_critical(prob, b):
    return prob.severity.get(str(b.bid).strip(), 0.0) >= 0.6


# ----------------------------------------------------------------- Fig 1
def fig_framework():
    """Framework as a left-to-right dataflow for resilient infrastructure asset
    management: asset data -> deterioration -> option economics -> objective
    screen -> constrained optimization -> outputs, with a sourced-parameters
    rail feeding the economics stage. The diagram is method-general; the bridge
    case study instantiates each input."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(C.COL2_IN, C.COL2_IN * 0.58))
    ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.set_aspect("equal"); ax.axis("off")
    ink, pr, cb, ac, ne = PAL["ink"], PAL["primary"], PAL["carbon"], PAL["accent"], PAL["neutral"]

    def box(x0, x1, y0, y1, title, subs, edge, fill, lw=1.3, tsize=6.6, ssize=5.0):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                     boxstyle="round,pad=0.02,rounding_size=1.0",
                     facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3))
        cx = (x0 + x1) / 2
        ax.text(cx, y1 - 1.9, title, ha="center", va="top", fontsize=tsize,
                weight="bold", color=ink, zorder=4)
        for k, sx in enumerate(subs):
            ax.text(cx, y1 - 4.0 - k * 2.0, sx, ha="center", va="top",
                    fontsize=ssize, color="#454545", zorder=4)

    def harrow(x0, x1, y, color=ne, lw=1.2):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                     mutation_scale=8, color=color, lw=lw, zorder=2, shrinkA=0, shrinkB=0))
    def varrow(x, y0, y1, color=ne, lw=1.2):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                     mutation_scale=8, color=color, lw=lw, zorder=2, shrinkA=0, shrinkB=0))
    def seg(x0, y0, x1, y1, color=ne, lw=1.0):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=2, solid_capstyle="round")

    # ---- input column: the asset data the method requires (method-general) ----
    ax.text(10.2, 56.2, "ASSET DATA  -  per asset", ha="center", va="center",
            fontsize=6.2, weight="bold", color=ne)
    ax.text(10.2, 54.0, "inventory + condition history", ha="center", va="center",
            fontsize=4.9, color="#6b6b6b")
    inb = [("Condition", "state + deterioration history"),
           ("Hazard exposure", "per-asset severity"),
           ("Usage / demand", "throughput, user delay"),
           ("Geometry & age", "size for unit-cost scaling")]
    yc = [45, 35, 25, 15]
    for (t, sline), y in zip(inb, yc):
        box(1.5, 20.5, y - 4.0, y + 4.0, t, [sline], pr, "#EAF1F6", tsize=6.2, ssize=4.6)
    for y in yc:
        seg(20.5, y, 22.2, y, color=pr, lw=1.0)
    seg(22.2, yc[-1], 22.2, yc[0], color=pr, lw=1.0)
    harrow(22.2, 24.0, 30, color=pr)

    # ---- spine (y center 30) ----
    box(24.0, 39.5, 23.5, 36.5, "Deterioration",
        ["Markov chain over", "the condition state"], cb, "#EAF3EE")
    box(42.0, 64.0, 23.5, 36.5, "Option economics",
        ["four options scored on", "cost, carbon, capital,", "hazard removed"], cb, "#EAF3EE")
    box(66.5, 82.0, 23.5, 36.5, "Objective screen",
        ["value-per-capital rank", "price of redundancy (optimized programs)"], ac, "#F7E9E4", lw=2.1)
    box(84.5, 99.5, 23.5, 36.5, "Optimization",
        ["knapsack, min cost", r"s.t. $B$, $\tau$  ($\varepsilon$-constr.)"], ac, "white")
    harrow(39.5, 42.0, 30); harrow(64.0, 66.5, 30); harrow(82.0, 84.5, 30)

    # ---- parameters rail feeding the economics stage ----
    box(42.0, 70.0, 41, 53, "Parameters",
        ["sourced: unit costs . value of time . emission factors",
         "discount rate . hazard-countermeasure cost",
         "specified inputs explicit and sensitivity-tested"],
        ne, "#F4F4F2", tsize=6.0, ssize=4.7)
    varrow(55.0, 41, 36.5, color=ne)

    # ---- outputs ----
    box(66.5, 99.5, 6, 19, "Outputs",
        ["cost-adaptation frontier",
         r"knee . protection target $\tau$ . emissions floor",
         "selected program"],
        ink, "#F1F1F1", tsize=6.4, ssize=5.0)
    varrow(92.0, 23.5, 19, color=ne)
    return C.savefig(fig, "fig_framework")


# ----------------------------------------------------------------- Fig 2
def fig_study_area(prob):
    """Inventory map plus condition-state and scour-severity distributions,
    combined into one 3-panel figure. Marker color = condition state; ring =
    scour-critical. A satellite basemap is drawn when map tiles are reachable
    (contextily + Esri World Imagery); the pipeline falls back to a plain
    coordinate plot where no network tile access is available (e.g. an
    offline sandbox). Adds a latitude-corrected scale bar and a north arrow
    in the satellite mode, since Web Mercator distorts distance with latitude."""
    full = prob.full
    lon = np.array([getattr(b, "x", np.nan) for b in full])
    lat = np.array([getattr(b, "y", np.nan) for b in full])
    states = np.array([b.state0 for b in full])
    ok = np.isfinite(lon) & np.isfinite(lat)
    crit = ok & np.array([_scour_critical(prob, b) for b in full])
    fig, (ax, a1, a2) = plt.subplots(1, 3, figsize=(C.COL2_IN * 1.55, C.COL1_IN * 1.18))
    sat = False
    try:
        import contextily as cx
        from pyproj import Transformer
        tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        X, Y = tf.transform(lon, lat)
        pad_x = 0.08 * (np.nanmax(X[ok]) - np.nanmin(X[ok]) + 1.0)
        pad_y = 0.08 * (np.nanmax(Y[ok]) - np.nanmin(Y[ok]) + 1.0)
        ax.set_xlim(np.nanmin(X[ok]) - pad_x, np.nanmax(X[ok]) + pad_x)
        ax.set_ylim(np.nanmin(Y[ok]) - pad_y, np.nanmax(Y[ok]) + pad_y)
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery,
                       crs="EPSG:3857", attribution_size=3)

        for st in sorted(C.STATE_COLORS, reverse=True):
            m = ok & (states == st)
            ax.scatter(X[m], Y[m], s=13, c=C.STATE_COLORS[st], linewidths=0.3,
                       edgecolors="white", label=C.STATE_LABELS[st], alpha=0.95, zorder=3)
        ax.scatter(X[crit], Y[crit], s=48, facecolors="none", edgecolors=PAL["accent"],
                   linewidths=1.1, label="scour-critical", zorder=4)
        ax.set_xticks([]); ax.set_yticks([]); sat = True

        lat0 = float(np.nanmean(lat[ok]))
        mercator_scale = 1.0 / np.cos(np.radians(lat0))
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        span_m = xlim[1] - xlim[0]; vspan_m = ylim[1] - ylim[0]
        ground_km_visible = (span_m / mercator_scale) / 1000.0
        bar_km = max(1, round(ground_km_visible / 4, -1) or 5)
        bar_map_len = bar_km * 1000.0 * mercator_scale
        x1 = xlim[1] - 0.07 * span_m
        x0 = x1 - bar_map_len
        y0 = ylim[0] + 0.07 * vspan_m
        ax.plot([x0, x1], [y0, y0], color="white", lw=2.2, solid_capstyle="butt",
                path_effects=[pe.withStroke(linewidth=4, foreground="black")], zorder=6)
        ax.text((x0 + x1) / 2, y0 + 0.018 * vspan_m,
                f"{bar_km:.0f} km", color="white", fontsize=6.5, ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="black")], zorder=6)

        nx = (x0 + x1) / 2
        ny0 = y0 + 0.10 * vspan_m
        ny1 = ny0 + 0.09 * vspan_m
        ax.annotate("", xy=(nx, ny1), xytext=(nx, ny0),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6,
                                    path_effects=[pe.withStroke(linewidth=3, foreground="black")]),
                    zorder=6)
        ax.text(nx, ny1 + 0.012 * vspan_m, "N", color="white",
                fontsize=7.5, ha="center", va="bottom", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="black")], zorder=6)
    except Exception:
        ax.clear()
        for st in sorted(C.STATE_COLORS, reverse=True):
            m = ok & (states == st)
            ax.scatter(lon[m], lat[m], s=11, c=C.STATE_COLORS[st], linewidths=0,
                       label=C.STATE_LABELS[st], alpha=0.9)
        ax.scatter(lon[crit], lat[crit], s=42, facecolors="none",
                   edgecolors=PAL["accent"], linewidths=1.0, label="scour-critical")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="lower left", fontsize=6.0, ncol=2, handletextpad=0.3,
              columnspacing=0.8, borderaxespad=0.25,
              facecolor="white", framealpha=0.85 if sat else 0)
    ax.set_title("(a) Bridge inventory", loc="left")

    order = [4, 3, 2, 1, 0]
    counts = [int((states == s).sum()) for s in order]
    a1.bar([C.STATE_LABELS[s] for s in order], counts,
           color=[C.STATE_COLORS[s] for s in order], width=0.74)
    a1.set_ylabel("Number of bridges"); a1.set_title("(b) Condition state", loc="left")
    a1.set_ylim(0, max(counts) * 1.15)
    for i, v in enumerate(counts):
        a1.text(i, v + max(counts) * 0.015, str(v), ha="center", fontsize=7.5, color=PAL["ink"])
    C.ygrid(a1)

    sev = np.array([prob.severity.get(str(b.bid).strip(), 0.0) for b in prob.full])
    bins = [0.0, 0.001, 0.3, 0.6, 0.9, 1.001]
    cnt = [int(((sev >= bins[i]) & (sev < bins[i + 1])).sum()) for i in range(5)]
    enames = ["low\n(<0.3)", "mod\n(0.3-0.6)", "high\n(0.6-0.9)", "critical\n(>=0.9)"]
    ecnt = cnt[1:]
    a2.bar(enames, ecnt, color=PAL["accent"], width=0.66)
    a2.set_ylabel("Number of bridges")
    a2.set_title("(c) Scour severity, exposed bridges", loc="left")
    a2.set_ylim(0, max(ecnt) * 1.25)
    for i, v in enumerate(ecnt):
        a2.text(i, v + max(ecnt) * 0.02, str(v), ha="center", fontsize=7.5, color=PAL["ink"])
    ntot = len(prob.full)
    a2.text(0.04, 0.92, f"{cnt[0]} of {ntot} bridges ({100*cnt[0]/ntot:.0f}%)\nhave no scour exposure",
            transform=a2.transAxes, ha="left", va="top", fontsize=6.6, color=PAL["neutral"])
    C.ygrid(a2)

    fig.tight_layout()
    return C.savefig(fig, "fig_study_area")


# ----------------------------------------------------------------- Fig 4
def fig_screen(prob):
    """The objective screen, visualized with the statistic it actually uses: percentile
    ranks of the value densities (objective change per unit of rehabilitation capital),
    paired by asset exactly as in screening.py. Tied values share their average rank."""
    from scipy.stats import rankdata
    from screening import rank_corr as _rank_corr
    V = prob.V; n = len(V)
    cap1 = np.array([V[i][1][OPT.CAPITAL] for i in range(n)]); m = cap1 > 0
    dcost = np.array([V[i][0][OPT.COST] - V[i][1][OPT.COST] for i in range(n)])[m] / cap1[m]
    dcarb = np.array([V[i][0][OPT.CARBON] - V[i][1][OPT.CARBON] for i in range(n)])[m] / cap1[m]
    scrm = np.array([V[i][2][OPT.SCOUR] - V[i][0][OPT.SCOUR] for i in range(n)])[m] / cap1[m]
    aadt = np.array([b.aadt for b in prob.candidates])[m]
    r_cc = _rank_corr(dcost, dcarb); r_cs = _rank_corr(dcost, scrm)
    pr = lambda v: (rankdata(v) - 0.5) / len(v) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(C.COL2_IN, 3.1))
    sc = a1.scatter(pr(dcost), pr(dcarb), c=aadt, cmap=CMAP, s=12, linewidths=0, alpha=0.85)
    cb = fig.colorbar(sc, ax=a1, fraction=0.046, pad=0.03)
    cb.set_label("AADT (veh/day)", fontsize=8); cb.ax.tick_params(labelsize=7)
    a1.set_xlabel("Cost value density, percentile rank"); a1.set_ylabel("Emissions value density, percentile rank")
    a1.set_title("(a) Cost and emissions", loc="left")
    a1.text(0.04, 0.90, f"$\\rho_{{vpc}}$ = {r_cc:.3f}", transform=a1.transAxes, fontsize=8.5,
            color=PAL["primary"], fontweight="bold")
    rng = np.random.default_rng(1)
    a2.scatter(pr(dcost), pr(scrm) + rng.uniform(-1.2, 1.2, m.sum()), s=12, color=PAL["accent"], linewidths=0, alpha=0.5)
    a2.set_xlabel("Cost value density, percentile rank"); a2.set_ylabel("Adaptation value density, percentile rank")
    a2.set_title("(b) Cost and adaptation", loc="left")
    a2.text(0.04, 0.10, f"$\\rho_{{vpc}}$ = {r_cs:.3f}", transform=a2.transAxes, fontsize=8.5,
            color=PAL["accent"], fontweight="bold")
    for ax in (a1, a2): ax.set_xlim(0, 100); ax.set_ylim(-3, 103); C.ygrid(ax)
    fig.tight_layout()
    return C.savefig(fig, "fig_screen")


# ----------------------------------------------------------------- Fig 5
def fig_carbon_floor():
    """Program emissions against the carbon floor. The cost-optimal program
    already sits on the floor (a separate carbon objective gains nothing) and
    is far below inaction."""
    df = pd.read_csv(C.OUT_DIR / "carbon_cap_results.csv")
    key = {str(r["program"]): r["carbon_k_tCO2e"] for _, r in df.iterrows()}
    def pick(*names):
        for k in key:
            if any(t in k.lower() for t in names):
                return key[k]
        return None
    do_nothing = pick("nothing", "defer", "inaction")
    cost_opt = pick("cost-optimal", "cost optimal", "cost_opt")
    floor = pick("floor", "carbon-optimal", "carbon optimal", "minimum")
    fig, ax = plt.subplots(figsize=(C.COL1_IN * 1.4, C.COL1_IN))
    ax.bar([0, 1], [do_nothing, cost_opt], width=0.5,
           color=[PAL["inaction"], PAL["primary"]], zorder=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Do nothing", "Cost-optimal\nprogram"])
    ax.set_ylabel("Program emissions (thousand tCO$_2$e)")
    ax.set_title("Emissions vs the carbon floor")
    ax.set_xlim(-0.7, 2.6); ax.set_ylim(0, do_nothing * 1.16)
    ax.axhline(floor, ls="--", lw=1.3, color=PAL["carbon"], zorder=1)
    ax.text(1.5, floor, "carbon floor\n(minimum achievable)", color=PAL["carbon"],
            fontsize=6.8, va="center", ha="left")
    ax.text(0, do_nothing * 1.02, f"{do_nothing:,.0f}", ha="center", fontsize=8.5, color=PAL["ink"])
    ax.text(1, cost_opt + do_nothing * 0.03, f"{cost_opt:,.0f}", ha="center", fontsize=8.5, color=PAL["ink"])
    red = (do_nothing - cost_opt) / do_nothing * 100
    ax.annotate("", xy=(0.5, do_nothing), xytext=(0.5, floor),
                arrowprops=dict(arrowstyle="<->", color=PAL["neutral"], lw=1.3))
    ax.text(0.57, (do_nothing + floor) / 2, f"-{red:.0f}%", color=PAL["neutral"],
            fontsize=9.5, fontweight="bold", va="center", ha="left")
    C.ygrid(ax)
    fig.tight_layout()
    return C.savefig(fig, "fig_carbon_floor")


# ----------------------------------------------------------------- Fig 6
def fig_knee(prob=None):
    """Marginal cost of scour protection: (a) on the refined grid, with the
    knee marked at its actually-computed location (max-curvature definition;
    see outputs/knee_estimate.csv), not assumed; (b) the same curve swept
    across the HEC-23 countermeasure-cost range, showing the knee persists."""
    kc = pd.read_csv(C.OUT_DIR / "knee_curve.csv")
    ke = pd.read_csv(C.OUT_DIR / "knee_estimate.csv")
    x = kc["protected_pct"].values; y = kc["premium_pct"].values
    knee_pct = float(ke.loc[ke["definition"] == "max_chord_distance", "protected_pct"].iloc[0])
    knee_prem = float(ke.loc[ke["definition"] == "max_chord_distance", "premium_pct"].iloc[0])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(C.COL2_IN * 1.05, C.COL1_IN * 0.95))
    ax.axvspan(0, knee_pct, color=PAL["band"], alpha=0.06)
    ax.plot(x, y, "-", color=PAL["primary"], lw=1.6)
    ax.plot(knee_pct, knee_prem, "o", color=PAL["primary"], ms=7, zorder=4)
    ax.annotate(f"knee ~ {knee_pct:.0f}%:\n~{knee_prem:.1f}% premium",
                xy=(knee_pct, knee_prem), xytext=(max(knee_pct - 55, 2), y.max() * 0.55),
                fontsize=8, color=PAL["neutral"], ha="left",
                arrowprops=dict(arrowstyle="->", color=PAL["neutral"], lw=1))
    ax.annotate(f"full protection\n+{y[-1]:.1f}%", xy=(100, y[-1]),
                xytext=(max(65, knee_pct - 25), y[-1] * 0.68),
                fontsize=8, color=PAL["neutral"], ha="center",
                arrowprops=dict(arrowstyle="->", color=PAL["neutral"], lw=1))
    ax.set_xlabel("Share of scour exposure protected (%)")
    ax.set_ylabel("Increase in program cost (%)")
    ax.set_title("(a) Cumulative life-cycle cost premium", loc="left")
    ax.set_xlim(-3, 105)
    C.ygrid(ax)

    fracs = [0.0, 0.50, 0.75, 0.90, 0.95, 1.00]
    cm_fracs = (0.15, 0.20, 0.25)
    colors = [PAL["primary"], PAL["accent"], PAL["carbon"]]
    for cm, col in zip(cm_fracs, colors):
        A.CM_FRACTION = cm
        p = P.build()
        rows = OPT.protection_curve(p.V, p.total_scour, fracs, p.budget_M)
        xs = [r["frac"] * 100 for r in rows if r["feasible"]]
        ys = [r["premium_pct"] for r in rows if r["feasible"]]
        ax2.plot(xs, ys, "-o", color=col, label=f"{cm:.0%} of rehabilitation")
    A.CM_FRACTION = 0.20
    ax2.set_xlabel("Share of scour exposure protected (%)")
    ax2.set_ylabel("Increase in program cost (%)")
    ax2.set_title("(b) Sensitivity to countermeasure cost", loc="left")
    ax2.legend(loc="upper left", title="Countermeasure cost", title_fontsize=8, fontsize=7.5)
    ax2.set_xlim(-3, 105)
    C.ygrid(ax2)

    fig.tight_layout()
    return C.savefig(fig, "fig_knee")


def fig_robustness():
    """Priority-alignment Monte Carlo (512 Latin-hypercube draws). (a) Per-draw
    screening statistics: cost-carbon value-per-capital correlation against
    cost-adaptation value-per-capital correlation, with the 0.95 redundancy
    cutoff marked on each axis. (b) Distribution of the ninety-percent
    adaptation premium across draws, with the median and the
    fifth-to-ninety-fifth-percentile band marked."""
    mc = pd.read_csv(C.OUT_DIR / "montecarlo_results.csv")
    rho_c = mc["rho_vpc_carbon"].values
    rho_a = mc["rho_vpc_adapt"].values
    prem = mc["prem90_%"].values

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(C.COL2_IN, 3.1))
    a1.scatter(rho_c, rho_a, s=14, color=PAL["primary"], linewidths=0, alpha=0.55)
    a1.axvline(0.95, color=PAL["neutral"], lw=0.9, ls="--")
    a1.axhline(0.95, color=PAL["neutral"], lw=0.9, ls="--")
    a1.set_xlabel("Cost-carbon priority alignment, $\\rho_{vpc}$")
    a1.set_ylabel("Cost-adaptation priority alignment, $\\rho_{vpc}$")
    a1.set_title("(a) Screening statistics across draws", loc="left")
    C.ygrid(a1)

    med = float(np.median(prem)); p5 = float(np.percentile(prem, 5)); p95 = float(np.percentile(prem, 95))
    a2.hist(prem, bins=24, color=PAL["accent"], alpha=0.85, edgecolor="white", linewidth=0.4)
    a2.axvline(med, color=PAL["primary"], lw=1.6)
    a2.axvspan(p5, p95, color=PAL["band"], alpha=0.12)
    a2.text(0.04, 0.92, f"median {med:.1f}%\n[{p5:.1f}, {p95:.1f}]", transform=a2.transAxes,
            fontsize=8, color=PAL["ink"], va="top")
    a2.set_xlabel("Ninety-percent adaptation premium (%)")
    a2.set_ylabel("Draws")
    a2.set_title("(b) Distribution across draws", loc="left")
    C.ygrid(a2)

    fig.tight_layout()
    return C.savefig(fig, "fig_robustness")


def fig_embodied_threshold():
    """Embodied-carbon boundary test. Alignment (rho_vpc, left axis) and carbon price of
    redundancy (delta, right axis) as rehabilitation embodied intensity is scaled.
    The two crossings (rho = 0.95 and delta = 5 %) are interpolated in log-intensity from
    the sampled multipliers and marked separately; the regions show what the combined
    rule does: remove, retain on alignment grounds, or retain with material loss."""
    df = pd.read_csv(C.OUT_DIR / "embodied_threshold.csv")
    x, rv, por = df["emb_kgCO2e_m2"].values, df["rho_vpc"].values, df["por_carbon_pct"].values
    lx = np.log(x)

    def crossing(y, level, falling):
        idx = np.where((y < level) if falling else (y > level))[0]
        if len(idx) == 0 or idx[0] == 0:
            return None
        k = idx[0]
        return float(np.exp(np.interp(level, [y[k], y[k - 1]] if falling else [y[k - 1], y[k]],
                                      [lx[k], lx[k - 1]] if falling else [lx[k - 1], lx[k]])))
    x_rho, x_del = crossing(rv, 0.95, True), crossing(por, 5.0, False)
    fig, ax = plt.subplots(figsize=(C.COL2_IN, 3.0))
    if x_rho: ax.axvspan(x.min(), x_rho, color=PAL["carbon"], alpha=0.07)
    if x_rho and x_del: ax.axvspan(x_rho, x_del, color=PAL["neutral"], alpha=0.10)
    if x_del: ax.axvspan(x_del, x.max(), color=PAL["accent"], alpha=0.07)
    ax.plot(x, rv, "-o", color=PAL["primary"], ms=4)
    ax.axhline(0.95, ls="--", lw=0.9, color=PAL["primary"], alpha=0.6)
    for xv in (x_rho, x_del):
        if xv: ax.axvline(xv, ls="--", lw=1.0, color=PAL["ink"])
    ax.plot([340], [np.interp(np.log(340), lx, rv)], marker="*", ms=11, color=PAL["ink"], zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel(r"rehabilitation embodied intensity (kgCO$_2$e/m$^2$, log scale)")
    ax.set_ylabel(r"cost-carbon alignment  $\rho_{vpc}$", color=PAL["primary"])
    ax.tick_params(axis="y", labelcolor=PAL["primary"])
    ylo = min(0.60, rv.min() - 0.05); ax.set_ylim(ylo, 1.01)
    ax2 = ax.twinx()
    ax2.plot(x, por, "-s", color=PAL["accent"], ms=3.5)
    ax2.axhline(5.0, ls="--", lw=0.9, color=PAL["accent"], alpha=0.6)
    ax2.set_ylabel(r"carbon price of redundancy  $\delta$ (%)", color=PAL["accent"])
    ax2.tick_params(axis="y", labelcolor=PAL["accent"])
    ax2.set_ylim(0, max(por) * 1.05); ax2.spines["top"].set_visible(False)
    ytxt = ylo + 0.04
    ax.text(x.min() * 1.1, ytxt, "removed", fontsize=7, color=PAL["carbon"], va="bottom")
    if x_rho and x_del:
        ax.text(np.sqrt(x_rho * x_del), ytxt, "retained,\nalignment\ngrounds", fontsize=6.3,
                color=PAL["ink"], ha="center", va="bottom")
    if x_del:
        ax.text(np.sqrt(x_del * x.max()), 0.995, "retained, material loss", fontsize=7, color=PAL["accent"],
                ha="center", va="top")
    ax.text(340 * 1.15, np.interp(np.log(340), lx, rv) - 0.012, "Rhode Island\nbaseline", fontsize=6.3,
            color=PAL["ink"], va="top")
    print(f"  embodied crossings: rho=0.95 at {x_rho:.0f} kg/m2 ({x_rho/340:.1f}x); "
          f"delta=5% at {x_del:.0f} kg/m2 ({x_del/340:.1f}x)")
    C.ygrid(ax)
    fig.tight_layout()
    return C.savefig(fig, "fig_embodied_threshold")


def fig_program_map(prob):
    """Map of the selected program at the adaptation knee: each candidate bridge
    colored by its optimal intervention (defer, rehabilitate, countermeasure, or
    both), with out-of-scope bridges shown faint for context. A satellite
    basemap (Esri World Imagery via contextily) is drawn when map tiles are
    reachable; the pipeline falls back to a plain coordinate plot otherwise."""
    import optimizer as OPT
    import pandas as pd_
    ke = pd_.read_csv(C.OUT_DIR / "knee_estimate.csv")
    knee_frac = float(ke.loc[ke["definition"]=="max_chord_distance","protected_pct"].iloc[0]) / 100.0
    ch, _ = OPT.min_cost_with_protection(prob.V, knee_frac * prob.total_scour, prob.budget_M)
    ch = np.array(ch)
    clon = np.array([getattr(b, "x", np.nan) for b in prob.candidates])
    clat = np.array([getattr(b, "y", np.nan) for b in prob.candidates])
    cand_ids = set(id(b) for b in prob.candidates)
    nlon = np.array([getattr(b, "x", np.nan) for b in prob.full if id(b) not in cand_ids])
    nlat = np.array([getattr(b, "y", np.nan) for b in prob.full if id(b) not in cand_ids])
    ok = np.isfinite(clon) & np.isfinite(clat)
    nz = np.isfinite(nlon) & np.isfinite(nlat)
    labels = {0: "Defer", 1: "Rehabilitate", 2: "Countermeasure", 3: "Rehab + countermeasure"}
    cols = {0: PAL["neutral"], 1: PAL["primary"], 2: PAL["accent"], 3: PAL["carbon"]}
    sizes = {0: 11, 1: 20, 2: 26, 3: 34}
    fig, ax = plt.subplots(figsize=(C.COL1_IN * 1.3, C.COL1_IN * 1.3))
    sat = False
    try:
        import contextily as cx
        from pyproj import Transformer
        tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        X, Y = tf.transform(clon, clat)
        nX, nY = tf.transform(nlon, nlat)
        ax.scatter(nX[nz], nY[nz], s=6, c="white", linewidths=0, alpha=0.5,
                   label=f"Not in scope ({int(nz.sum())})", zorder=2)
        for g in (0, 1, 2, 3):
            m = ok & (ch == g)
            ax.scatter(X[m], Y[m], s=sizes[g], c=cols[g], edgecolors="white",
                       linewidths=0.3, label=f"{labels[g]} ({int(m.sum())})", zorder=3)
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery,
                       crs="EPSG:3857", attribution_size=3)
        ax.set_xticks([]); ax.set_yticks([]); sat = True
    except Exception:
        ax.scatter(nlon[nz], nlat[nz], s=6, c="#D7D7D7", linewidths=0,
                   label=f"Not in scope ({int(nz.sum())})", zorder=1)
        for g in (0, 1, 2, 3):
            m = ok & (ch == g)
            ax.scatter(clon[m], clat[m], s=sizes[g], c=cols[g], linewidths=0,
                       alpha=0.9, label=f"{labels[g]} ({int(m.sum())})", zorder=3)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="lower left", fontsize=6.0, framealpha=0.85 if sat else 0,
              facecolor="white")
    ax.set_title("Selected program at the adaptation knee")
    fig.tight_layout()
    return C.savefig(fig, "fig_program_map")


def _loss_heatmap(ax, Z, ylabels, xlabels, ref_cell, title):
    """One panel: carbon price of redundancy (%) on a shared log color scale; cells above 5 %
    (emissions need independent treatment) hatched; reference cell outlined."""
    norm = matplotlib.colors.LogNorm(vmin=1e-3, vmax=100)
    im = ax.imshow(np.clip(Z, 1e-3, 100), cmap="OrRd", norm=norm, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z[i, j]
            ax.text(j, i, f"{v:.1f}" if v >= 0.1 else ("0" if v < 0.005 else f"{v:.2f}"), ha="center", va="center",
                    fontsize=7, color="white" if v > 8 else PAL["ink"])
            if v > 5:
                ax.add_patch(matplotlib.patches.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///", lw=0, alpha=.35))
    if ref_cell is not None:
        ax.add_patch(matplotlib.patches.Rectangle((ref_cell[1] - .5, ref_cell[0] - .5), 1, 1, fill=False, lw=2.2, ec=PAL["ink"]))
    ax.set_xticks(range(len(xlabels)), xlabels); ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    ax.set_xlabel("benefit-accounting period (years)"); ax.set_title(title, loc="left", fontsize=9)
    return im


def fig_diversion_horizon():
    """Central result: emissions lost by planning for cost alone (% of minimum achievable) at the
    program budget, (a) over diverted-traffic schedule and analysis period (2 % discount rate),
    (b) over discount rate and analysis period (evidence-based diversion)."""
    g = pd.read_csv(C.OUT_DIR / "diversion_horizon_grid.csv"); g = g[g.feasible]
    order = list(dict.fromkeys(g.schedule)); yrs = sorted(g.analysis_years.unique())
    Za = np.array([[g[(g.schedule == s) & (g.analysis_years == h)].delta_pct.iloc[0] for h in yrs] for s in order])
    lab = {"lambda 0.00": "evidence-based\n(reference)", "lambda 1.00": "original\nassumption", "evidence 2024 only": "evidence,\n2024 only"}
    ya = [lab.get(s, f"{float(s.split()[1]):.0%} toward\noriginal") for s in order]
    ref_a = (order.index("lambda 0.00"), yrs.index(int(C.ANALYSIS_YEARS)))
    dpath = C.OUT_DIR / "discount_horizon_grid.csv"
    two = dpath.exists()
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(C.COL2_IN * (1.55 if two else 1), 3.5), squeeze=False)
    im = _loss_heatmap(axes[0, 0], Za, ya, [str(h) for h in yrs], ref_a, "(a) Diverted-traffic schedule (2% discount rate)")
    if two:
        dgr = pd.read_csv(dpath); dgr = dgr[dgr.feasible]
        rates = sorted(dgr.discount_rate.unique()); yrs_b = sorted(dgr.analysis_years.unique())
        Zb = np.array([[dgr[(dgr.discount_rate == r) & (dgr.analysis_years == h)].delta_pct.iloc[0] for h in yrs_b] for r in rates])
        ref_b = (rates.index(0.02), yrs_b.index(int(C.ANALYSIS_YEARS))) if 0.02 in rates else None
        _loss_heatmap(axes[0, 1], Zb, [f"{r:.0%}" for r in rates], [str(h) for h in yrs_b], ref_b,
                      "(b) Discount rate (evidence-based diversion)")
        axes[0, 1].set_ylabel("real discount rate", fontsize=8)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cb.set_label("emissions lost by planning for cost alone (%)", fontsize=8)
    return C.savefig(fig, "fig_diversion_horizon")


def run_all():
    prob = P.build()
    out = [fig_framework(), fig_study_area(prob), fig_screen(prob),
           fig_carbon_floor(), fig_knee(prob), fig_robustness(),
           fig_embodied_threshold(), fig_diversion_horizon(), fig_program_map(prob)]
    for pdf, png in out:
        print("saved", png)
    return out


if __name__ == "__main__":
    run_all()
