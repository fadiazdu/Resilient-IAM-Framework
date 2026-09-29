# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
paper_figures.py  -  figures of the paper and of the supplemental materials
=============================================================================
Reads the outputs of run_all.py and writes to figures/ (600-dpi PNG and vector PDF with embedded fonts):
  Paper:       fig1_framework, fig2_study_area, fig3_screen, fig4_budget, fig5_conventions,
               fig6_knee, fig7_embodied
  Supplement:  figS1_program_map, figS2_diversion_sweep, figS3_montecarlo, figS4_solver
Style: drawn at the printed width (7.0 in, two columns), text 7-8 pt at that size, thin lines, markers
only where needed, and every series distinguishable in black and white by line style or marker shape
as well as by color (ASCE requires figures that print legibly in black and white).
Run alone to redraw the figures from existing outputs: python paper_figures.py
"""
from __future__ import annotations
import os, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.lines import Line2D
import config as C

OUT, FIG = C.OUT_DIR, C.FIG_DIR
FIG.mkdir(parents=True, exist_ok=True)
DPI = int(os.environ.get("IAM_FIG_DPI") or 600)
W2 = 7.0                                              # two-column printed width (in)

# ------------------------------------------------------------------ style
# Okabe-Ito palette: vivid on screen and distinguishable by color-blind readers. Series differ by color;
# markers or dashes are added only where they aid reading. Gray is used only for reference lines.
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#E6E6E6"
BLUE, VERMIL, GREEN, ORANGE, SKY, PURPLE, YELLOW = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#F0E442"
MAGENTA = "#A33A7A"
REFLINE = "#7A7A7A"
COND_ORDER = (4, 3, 2, 1, 0)                          # NBI 8-9 ... NBI 0-3
COND_LABEL = {4: "8-9", 3: "7", 2: "5-6", 1: "4", 0: "0-3"}
COND_COLOR = {4: "#2C7BB6", 3: "#ABD9E9", 2: "#FFFFBF", 1: "#FDAE61", 0: "#D7191C"}   # diverging, bright on imagery (approved)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "legend.handlelength": 2.2, "legend.borderaxespad": 0.4,
    "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left", "axes.titlepad": 5,
    "lines.markersize": 3, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular", "lines.linewidth": 1.1})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png")
    if DPI >= 300:
        fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig); print(f"  saved figures/{name}")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5); ax.set_axisbelow(True)


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold")


def tolerance(ax, label=True, where="right"):
    ax.axhline(5.0, color=REFLINE, lw=0.8, ls=(0, (4, 2)), zorder=1)
    if label:
        x, ha = (0.99, "right") if where == "right" else (0.01, "left")
        ax.text(x, 5.0, "5% tolerance", transform=ax.get_yaxis_transform(), ha=ha, va="bottom", fontsize=6.5, color=MUTED)


# ------------------------------------------------------------------ map helpers
def _basemap(ax):
    try:
        import contextily as cx
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.Esri.WorldImagery, attribution=False)
        return True
    except Exception as e:
        print(f"  (satellite basemap unavailable: {type(e).__name__}; plain background)")
        ax.set_facecolor("#E9EEF1"); return False


def _map_frame(ax, lon, lat, pad=0.025):
    ax.set_xlim(lon.min() - pad, lon.max() + pad); ax.set_ylim(lat.min() - pad, lat.max() + pad)
    ax.set_aspect(1 / np.cos(np.radians(lat.mean())), adjustable="box")
    ax.set_xlabel("Longitude (°W)"); ax.set_ylabel("Latitude (°N)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):.1f}"))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    for s in ("top", "right"): ax.spines[s].set_visible(True)


def _scalebar_north(ax, km=20):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); w, h = x1 - x0, y1 - y0
    dlon = km / (111.32 * np.cos(np.radians((y0 + y1) / 2)))
    xs, ys = x1 - 0.06 * w - dlon, y0 + 0.05 * h
    ax.add_patch(mpatches.Rectangle((xs - 0.015 * w, ys - 0.02 * h), dlon + 0.03 * w, 0.07 * h, fc="white", ec=INK, lw=0.4, alpha=0.9, zorder=6))
    ax.plot([xs, xs + dlon], [ys, ys], color=INK, lw=1.6, solid_capstyle="butt", zorder=7)
    ax.text(xs + dlon / 2, ys + 0.012 * h, f"{km} km", ha="center", va="bottom", fontsize=6.5, zorder=7)
    ann = ax.annotate("N", xy=(0.93, 0.965), xytext=(0.93, 0.87), xycoords="axes fraction", textcoords="axes fraction",
                      ha="center", va="center", fontsize=6.5, fontweight="bold", color=INK, zorder=8,
                      bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=INK, lw=0.4, alpha=0.9),
                      arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=8, shrinkA=4, shrinkB=0))
    ann.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])


# ------------------------------------------------------------------ Fig. 1  framework (monochrome)
def fig1_framework():
    W, H = W2, 3.35
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    m, gap = 0.06, 0.2
    bw = (W - 2 * m - 4 * gap) / 5
    xs = [m + k * (bw + gap) for k in range(5)]

    def box(x, y, w, h, title, body, chars):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc="white", ec=INK, lw=0.8))
        body_txt = textwrap.fill(body, chars); n = body_txt.count("\n") + 1
        block = 0.13 + 0.09 + n * 0.118                     # title, gap, body lines (in)
        top = y + h / 2 + block / 2
        ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=7.2, fontweight="bold", color=INK)
        ax.text(x + w / 2, top - 0.22, body_txt, ha="center", va="top", fontsize=6.8, color=INK, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=8, shrinkA=0, shrinkB=0))

    top_y, top_h = 1.95, 1.3
    bot_y, bot_h = 0.08, 0.95
    # public data (left column)
    box(xs[0], 2.3, bw, 0.95, "Inventory", "Condition, traffic, truck share, detour, deck area, scour code", 22)
    box(xs[0], 1.19, bw, 0.95, "Inspections", "Dated condition ratings, 2019-2024", 22)
    box(xs[0], bot_y, bw, bot_h, "Operating status", "Closed and load-posted bridges, 2019-2024", 22)
    # model chain (top row)
    chain = [("Deterioration", "Continuous-time Markov model estimated from inspection intervals"),
             ("Life-cycle evaluation", "Cost, emissions, capital, and scour exposure of four options per bridge"),
             ("Objective screen", "Loss from omitting an objective across the decision domain; alignment explains the result"),
             ("Optimization", "Minimum cost subject to budget, protection requirement, and NHS condition")]
    for k, (t, b) in enumerate(chain, start=1):
        box(xs[k], top_y, bw, top_h, t, b, 22)
    for k in range(1, 4):
        arrow(xs[k] + bw, top_y + top_h / 2, xs[k + 1], top_y + top_h / 2)
    # bottom row
    box(xs[1], bot_y, xs[2] + bw - xs[1], bot_h, "Parameters and conventions",
        "Unit costs, value of time, emission factors, diverted traffic, discount rate, analysis period", 48)
    box(xs[3], bot_y, xs[4] + bw - xs[3], bot_h, "Outputs",
        "Objectives needing independent treatment; cost-adaptation frontier and knee; selected program", 48)
    # data flows
    arrow(xs[0] + bw, 2.3 + 0.475, xs[1], 2.3 + 0.475)
    arrow(xs[0] + bw, 1.19 + 0.475, xs[1], top_y + 0.25)
    arrow(xs[0] + bw, bot_y + bot_h / 2, xs[1], bot_y + bot_h / 2)
    arrow(xs[2] + bw / 2, bot_y + bot_h, xs[2] + bw / 2, top_y)
    arrow(xs[3] + bw / 2, top_y, xs[3] + bw / 2, bot_y + bot_h)
    arrow(xs[4] + bw / 2, top_y, xs[4] + bw / 2, bot_y + bot_h)
    save(fig, "fig1_framework")


# ------------------------------------------------------------------ Fig. 2  study area (approved original design)
# Drawn with its original style settings, which apply only while this figure is created.
FIG2_RC = {"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
           "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2, "axes.linewidth": 0.6,
           "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
           "axes.edgecolor": "#222222", "axes.labelcolor": "#222222", "xtick.color": "#222222", "ytick.color": "#222222",
           "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
           "axes.titlelocation": "left", "axes.titlepad": 6, "legend.frameon": False}
FIG2_COND = {4: ("8-9", "#2C7BB6"), 3: ("7", "#ABD9E9"), 2: ("5-6", "#FFFFBF"), 1: ("4", "#FDAE61"), 0: ("0-3", "#D7191C")}


def _fig2_scalebar_north(ax, km=20):
    ink = "#222222"
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); w, h = x1 - x0, y1 - y0
    dlon = km / (111.32 * np.cos(np.radians((y0 + y1) / 2)))
    xs, ys = x1 - 0.06 * w - dlon, y0 + 0.05 * h
    ax.add_patch(mpatches.Rectangle((xs - 0.015 * w, ys - 0.02 * h), dlon + 0.03 * w, 0.075 * h, fc="white", ec=ink, lw=0.4, alpha=0.9, zorder=6))
    ax.plot([xs, xs + dlon], [ys, ys], color=ink, lw=2.2, solid_capstyle="butt", zorder=7)
    ax.text(xs + dlon / 2, ys + 0.012 * h, f"{km} km", ha="center", va="bottom", fontsize=7, zorder=7)
    ann = ax.annotate("N", xy=(0.93, 0.965), xytext=(0.93, 0.865), xycoords="axes fraction", textcoords="axes fraction",
                      ha="center", va="center", fontsize=7, fontweight="bold", color=ink, zorder=8,
                      bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ink, lw=0.4, alpha=0.9),
                      arrowprops=dict(arrowstyle="-|>", color=ink, lw=1.1, mutation_scale=9, shrinkA=4, shrinkB=0))
    ann.arrow_patch.set_path_effects([pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])


def fig2_study_area():
    import problem as P
    ink, grid_c, muted = "#222222", "#E3E3E3", "#5F5F5F"
    with plt.rc_context(FIG2_RC):
        prob = P.build(); full = prob.full
        lon = np.array([b.x for b in full]); lat = np.array([b.y for b in full]); st = np.array([b.state0 for b in full])
        sev = np.array([prob.severity.get(str(b.bid).strip(), 0.0) for b in full])
        port = pd.read_csv(C.PORTFOLIO_CSV, dtype={"bid": str}); port["bid"] = port.bid.str.strip()
        code = dict(zip(port.bid, port.scour_113.astype(str).str.strip()))
        crit = np.array([code.get(str(b.bid).strip(), "") in ("0", "1", "2", "3") for b in full])
        fig = plt.figure(figsize=(W2, 6.2))
        gs = fig.add_gridspec(5, 2, width_ratios=[2.05, 1], height_ratios=[0.3, 1, 0.75, 1, 0.55], wspace=0.32, hspace=0.0)
        ax = fig.add_subplot(gs[:, 0]); b1 = fig.add_subplot(gs[1, 1]); b2 = fig.add_subplot(gs[3, 1])
        _map_frame(ax, lon, lat); _basemap(ax)
        for s in (4, 3, 2, 1, 0):
            m = st == s
            ax.scatter(lon[m], lat[m], s=9, c=FIG2_COND[s][1], edgecolors=ink, linewidths=0.3, zorder=4)
        ax.scatter(lon[crit], lat[crit], s=34, facecolors="none", edgecolors=PURPLE, linewidths=0.9, zorder=5)
        _fig2_scalebar_north(ax)
        ax.set_title("(a) Inventory")
        hs = [Line2D([], [], ls="", marker="o", ms=5, mfc=FIG2_COND[s][1], mec=ink, mew=0.3, label=f"NBI {FIG2_COND[s][0]}") for s in (4, 3, 2, 1, 0)]
        hs.append(Line2D([], [], ls="", marker="o", ms=7.5, mfc="none", mec=PURPLE, mew=1.0, label="Scour-critical"))
        ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=7, handletextpad=0.2, columnspacing=1.0)
        counts = [int((st == s).sum()) for s in (4, 3, 2, 1, 0)]
        bars = b1.bar(range(5), counts, width=0.7, color=[FIG2_COND[s][1] for s in (4, 3, 2, 1, 0)], edgecolor=ink, linewidth=0.5)
        b1.bar_label(bars, fontsize=7, padding=1.5)
        b1.set_xticks(range(5), [FIG2_COND[s][0] for s in (4, 3, 2, 1, 0)]); b1.set_xlabel("Governing NBI rating")
        b1.set_ylabel("Bridges"); b1.set_title("(b) Condition"); b1.grid(axis="y", color=grid_c, lw=0.5); b1.set_axisbelow(True)
        b1.set_ylim(0, max(counts) * 1.15)
        w = np.round(sev[sev > 0], 1); vals = sorted(set(w)); cnt = [int((w == v).sum()) for v in vals]
        bars = b2.bar(range(len(vals)), cnt, width=0.7, color=VERMIL, edgecolor=ink, linewidth=0.5)
        b2.bar_label(bars, fontsize=7, padding=1.5); b2.set_xticks(range(len(vals)), [f"{v:.1f}" for v in vals])
        b2.set_xlabel("Scour weight"); b2.set_ylabel("Bridges"); b2.grid(axis="y", color=grid_c, lw=0.5); b2.set_axisbelow(True)
        b2.set_ylim(0, max(cnt) * 1.42); b2.set_title("(c) Scour exposure")
        b2.text(0.03, 0.95, f"{int((sev > 0).sum())} of {len(full)} bridges exposed", transform=b2.transAxes, va="top", fontsize=6.8, color=muted)
        save(fig, "fig2_study_area")


# ------------------------------------------------------------------ Fig. 3  priority alignment
def fig3_screen():
    import problem as P, optimizer as OPT
    from scipy.stats import rankdata
    from screening import rank_corr
    prob = P.build(); V = prob.V; n = len(V)
    cap = np.array([V[i][1][OPT.CAPITAL] for i in range(n)]); m = cap > 0
    dc = np.array([V[i][0][OPT.COST] - V[i][1][OPT.COST] for i in range(n)])[m] / cap[m]
    de = np.array([V[i][0][OPT.CARBON] - V[i][1][OPT.CARBON] for i in range(n)])[m] / cap[m]
    da = np.array([V[i][2][OPT.SCOUR] - V[i][0][OPT.SCOUR] for i in range(n)])[m] / cap[m]
    aadt = np.array([b.aadt for b in prob.candidates])[m] / 1000
    pr = lambda v: (rankdata(np.round(v, 10)) - 0.5) / len(v) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.8), gridspec_kw={"wspace": 0.28})
    sc = a1.scatter(pr(dc), pr(de), c=aadt, cmap="viridis", s=7, linewidths=0, vmin=0, vmax=np.percentile(aadt, 98))
    cax = a1.inset_axes([0.55, 0.13, 0.4, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="max"); cb.outline.set_linewidth(0.4)
    cb.ax.xaxis.set_label_position("top"); cb.set_label("Traffic (10³ vehicles/day)", fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=6, length=1.5, pad=1, width=0.4)
    a1.text(0.04, 0.95, f"ρ = {rank_corr(dc, de):.3f}", transform=a1.transAxes, va="top", fontsize=8, color=BLUE, fontweight="bold")
    panel(a1, "a", "Cost and emissions")
    n0 = int((da == 0).sum()); tie = float(np.median(pr(da)[da == 0]))
    jit = np.random.default_rng(1).uniform(-0.8, 0.8, m.sum())
    a2.scatter(pr(dc), pr(da) + jit, s=7, color=VERMIL, alpha=0.55, linewidths=0)
    a2.annotate(f"{n0} bridges without scour exposure\nshare one tied rank", xy=(55, tie + 2), xytext=(55, tie + 18),
                ha="center", va="bottom", fontsize=6.8, color=MUTED, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, mutation_scale=7))
    a2.text(0.04, 0.1, f"ρ = {rank_corr(dc, da):.3f}", transform=a2.transAxes, va="top", fontsize=8, color=VERMIL, fontweight="bold")
    panel(a2, "b", "Cost and adaptation")
    for ax, yl in ((a1, "Emissions value density (percentile)"), (a2, "Adaptation value density (percentile)")):
        ax.set_xlim(0, 100); ax.set_ylim(-3, 103); ax.set_xlabel("Cost value density (percentile)"); ax.set_ylabel(yl); grid(ax)
    save(fig, "fig3_screen")


# ------------------------------------------------------------------ Fig. 4  budget and budget use
def fig4_budget():
    b = pd.read_csv(OUT / "budget_use_sweep.csv"); b = b[b.feasible.astype(str) == "True"].astype({"excess_pct": float, "cost_capital_M": float, "carbon_capital_M": float})
    u = pd.read_csv(OUT / "budget_unconstrained.csv").iloc[0]
    xs = float(u.cost_stopping_multiple); ks = float(u.cost_capital_unconstrained_M); B0 = float(u.program_budget_M)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"wspace": 0.3})
    a1.plot(b.budget_mult, b.excess_pct, color=VERMIL, lw=1.3)
    tolerance(a1, where="left")
    for ax in (a1, a2):
        ax.axvline(xs, color=REFLINE, lw=0.8, ls=":", zorder=1)
    a1.text(xs - 0.03, 0.96, f"Cost program stops\nat {xs:.2f} times\nthe program budget", transform=a1.get_xaxis_transform(),
            ha="right", va="top", fontsize=6.8, color=MUTED)
    a1.set_ylabel("Excess emissions over the minimum (%)"); a1.set_ylim(bottom=0); panel(a1, "a", "Excess emissions")
    a2.plot(b.budget_mult, b.budget_mult * B0, color=REFLINE, lw=0.9, ls=":", label="Budget")
    a2.plot(b.budget_mult, b.carbon_capital_M, color=GREEN, lw=1.2, label="Emissions-optimal program")
    a2.plot(b.budget_mult, b.cost_capital_M, color=BLUE, lw=1.2, ls=(0, (5, 2)), label="Cost-optimal program", zorder=4)
    a2.annotate(f"{ks:,.0f} million USD", xy=(xs, ks), xytext=(xs + 0.1, ks - 190), fontsize=6.8, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a2.set_ylabel("Capital spent (million USD)"); a2.legend(loc="upper left"); panel(a2, "b", "Budget use")
    for ax in (a1, a2):
        ax.set_xlabel("Budget ratio (budget / program budget)"); ax.set_xlim(b.budget_mult.min(), b.budget_mult.max()); grid(ax)
    save(fig, "fig4_budget")


# ------------------------------------------------------------------ Fig. 5  accounting conventions
def fig5_conventions():
    g = pd.read_csv(OUT / "diversion_horizon_grid.csv"); r = pd.read_csv(OUT / "discount_horizon_grid.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.8), sharey=True, gridspec_kw={"wspace": 0.08})
    for key, lab, col, lw in (("lambda 0.00", "NBI records 2019-2024 (reference)", BLUE, 1.4), ("evidence 2024 only", "NBI records 2024 only", ORANGE, 1.0),
                              ("lambda 1.00", "High-diversion scenario", VERMIL, 1.0)):
        d = g[g.schedule == key].sort_values("analysis_years")
        a1.plot(d.analysis_years, d.delta_pct, color=col, lw=lw, label=lab, zorder=4 if key == "lambda 0.00" else 3)
    rcol = [BLUE, SKY, GREEN, ORANGE, VERMIL]
    for k, rt in enumerate(sorted(r.discount_rate.unique())):
        d = r[r.discount_rate == rt].sort_values("analysis_years"); ref = abs(rt - C.DR) < 1e-9
        a2.plot(d.analysis_years, d.delta_pct, color=rcol[k % 5], lw=1.4 if ref else 1.0, label=f"{100 * rt:g}%" + (" (reference)" if ref else ""), zorder=4 if ref else 3)
    for ax, letter, title in ((a1, "a", "Diverted traffic"), (a2, "b", "Discount rate")):
        tolerance(ax, label=False); ax.axvline(35, color=REFLINE, lw=0.8, ls=":", zorder=1)
        ax.set_xlabel("Benefit-accounting period (years)"); ax.set_xlim(10, 40); ax.set_xticks(range(10, 41, 5)); grid(ax); panel(ax, letter, title)
    a1.text(35.5, 21, "Reference period", rotation=90, ha="left", va="center", fontsize=6.5, color=MUTED)
    a1.text(27.5, 5.0, "5% tolerance", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    a1.set_ylabel("Excess emissions over the minimum (%)"); a1.set_ylim(-1, 46)
    lk = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)      # reference line passes behind the legend
    a1.legend(loc="upper right", **lk); a2.legend(title="Real discount rate", title_fontsize=7, loc="upper right", alignment="left", **lk)
    save(fig, "fig5_conventions")


# ------------------------------------------------------------------ Fig. 6  cost-adaptation frontier and knee
def fig6_knee():
    import problem as P, optimizer as OPT, adaptation as A
    k = pd.read_csv(OUT / "knee_curve.csv").sort_values("protected_pct")
    ke = pd.read_csv(OUT / "knee_estimate.csv").set_index("definition").loc["max_chord_distance"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"wspace": 0.28})
    a1.plot(k.protected_pct, k.premium_pct, color=BLUE, lw=1.3)
    a1.plot(ke.protected_pct, ke.premium_pct, marker="o", ms=4.5, mfc="white", mec=VERMIL, mew=1.1, ls="", zorder=5)
    a1.annotate(f"Knee: {ke.protected_pct:.1f}% addressed,\n+{ke.premium_pct:.2f}% life-cycle cost", xy=(ke.protected_pct, ke.premium_pct),
                xytext=(ke.protected_pct - 40, ke.premium_pct + 0.6), fontsize=6.8, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a1.set_ylim(0, k.premium_pct.max() * 1.08); panel(a1, "a", "Frontier and knee")
    base = A.CM_FRACTION
    spec = [(0.15, SKY, "-"), (0.20, BLUE, "-"), (0.25, GREEN, "-"), (0.40, VERMIL, (0, (5, 2)))]
    try:
        for frac, col, ls in spec:
            A.CM_FRACTION = frac; prob = P.build(); tot, B = prob.total_scour, prob.budget_M
            c0 = OPT.cost_optimal(prob.V, B)[1]["cost"]; xv, yv = [], []
            for f in np.linspace(0, 1, 41):
                rr = OPT.min_cost_with_protection(prob.V, f * tot, B, lex=False)
                xv.append(rr[1]["protected"] / tot * 100); yv.append((rr[1]["cost"] - c0) / c0 * 100)
            ref = abs(frac - base) < 1e-9
            a2.plot(xv, yv, color=col, ls=ls, lw=1.4 if ref else 1.0, label=f"{frac:.2f}" + (" (reference)" if ref else " (stress)" if frac == 0.40 else ""))
    finally:
        A.CM_FRACTION = base
    a2.set_ylim(bottom=0); panel(a2, "b", "Countermeasure cost")
    a2.legend(title="Countermeasure cost as a share\nof rehabilitation cost", title_fontsize=7, loc="upper left", alignment="left")
    for ax in (a1, a2):
        ax.set_xlabel("Weighted scour exposure addressed (%)"); ax.set_ylabel("Life-cycle cost premium (%)"); ax.set_xlim(0, 100); grid(ax)
    save(fig, "fig6_knee")


# ------------------------------------------------------------------ Fig. 7  embodied-carbon boundary
def fig7_embodied():
    e = pd.read_csv(OUT / "embodied_threshold.csv").sort_values("emb_kgCO2e_m2")
    x, rho, dl = e.emb_kgCO2e_m2.values, e.rho_vpc.values, e.por_carbon_pct.values

    def cross(y, level):
        for i in range(1, len(y)):
            if (y[i - 1] - level) * (y[i] - level) <= 0 and y[i] != y[i - 1]:
                lx = np.log10(x[i - 1]) + (level - y[i - 1]) * (np.log10(x[i]) - np.log10(x[i - 1])) / (y[i] - y[i - 1])
                return 10 ** lx
    xd, xr = cross(dl, 5.0), cross(rho, 0.95)
    fig, a1 = plt.subplots(figsize=(W2, 2.9)); a2 = a1.twinx(); a2.spines["right"].set_visible(True)
    lo, hi = x.min() / 1.3, x.max() * 1.3
    a1.axvspan(lo, xd, color=GREEN, alpha=0.07, lw=0, zorder=0); a1.axvspan(xd, hi, color=VERMIL, alpha=0.07, lw=0, zorder=0)
    v1 = a1.axvline(xr, color=MAGENTA, lw=0.8, ls=":", label=f"Alignment falls below 0.95 ({xr:,.0f} kgCO$_2$e/m$^2$)")
    v2 = a1.axvline(xd, color=INK, lw=0.9, ls=(0, (4, 2)), label=f"Excess exceeds 5% ({xd:,.0f} kgCO$_2$e/m$^2$)")
    l1, = a1.plot(x, rho, color=MAGENTA, lw=1.3, label="Cost-emissions alignment (left axis)")
    l2, = a2.plot(x, dl, color=VERMIL, lw=1.3, label="Excess emissions over the minimum (right axis)")
    a2.axhline(5, color=REFLINE, lw=0.7, ls=(0, (4, 2)))
    a1.text(np.sqrt(lo * xr), 0.74, "Emissions can be omitted", ha="center", va="center", fontsize=7, color=GREEN, fontweight="bold")
    a1.text(3.6e4, 0.985, "Emissions must be retained", ha="center", va="top", fontsize=7, color=VERMIL, fontweight="bold")
    y340 = float(np.interp(np.log10(340), np.log10(x), rho))
    a1.annotate("Reference, 340 kgCO$_2$e/m$^2$", xy=(340, y340), xytext=(340, 0.88), ha="center", fontsize=6.5, color=MUTED,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6, mutation_scale=6))
    a1.set_xscale("log"); a1.set_xlim(lo, hi); a1.set_ylim(0.5, 1.0); a2.set_ylim(0, max(dl) * 1.05)
    a1.set_xlabel("Rehabilitation embodied intensity (kgCO$_2$e/m$^2$)")
    a1.set_ylabel("Cost-emissions alignment", color=MAGENTA); a2.set_ylabel("Excess emissions over the minimum (%)", color=VERMIL)
    a1.tick_params(axis="y", colors=MAGENTA); a2.tick_params(axis="y", colors=VERMIL)
    a1.legend(handles=[l1, l2, v1, v2], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    save(fig, "fig7_embodied")


# ------------------------------------------------------------------ Fig. S1  program at the knee
def figS1_program_map():
    import problem as P, optimizer as OPT
    prob = P.build(); ke = pd.read_csv(OUT / "knee_estimate.csv").set_index("definition").loc["max_chord_distance"]
    ch, _ = OPT.min_cost_with_protection(prob.V, ke.protected_pct / 100 * prob.total_scour, prob.budget_M)
    cand = {str(b.bid).strip(): g for b, g in zip(prob.candidates, ch)}
    lon = np.array([b.x for b in prob.full]); lat = np.array([b.y for b in prob.full])
    cls = np.array([cand.get(str(b.bid).strip(), -1) for b in prob.full])
    fig, ax = plt.subplots(figsize=(4.4, 5.6))
    _map_frame(ax, lon, lat); _basemap(ax)
    spec = [(-1, "Outside the program", "white", "o", 5), (0, "Deferred", "#A8A8A8", "o", 8), (1, "Rehabilitated", BLUE, "o", 18),
            (2, "Countermeasure", PURPLE, "o", 18), (3, "Rehabilitated and countermeasure", VERMIL, "o", 18)]
    hs = []
    for g, lab, col, mk, sz in spec:
        sel = cls == g
        ax.scatter(lon[sel], lat[sel], s=sz, c=col, marker=mk, edgecolors=INK, linewidths=0.3, zorder=4 + max(g, 0))
        hs.append(Line2D([], [], ls="", marker=mk, ms=np.sqrt(sz) + 0.8, mfc=col, mec=INK, mew=0.3, label=f"{lab} ({int(sel.sum())})"))
    _scalebar_north(ax)
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, columnspacing=1.0, handletextpad=0.2, fontsize=6.5)
    save(fig, "figS1_program_map")


# ------------------------------------------------------------------ Fig. S2  continuous diversion sweep
def figS2_diversion_sweep():
    d = pd.read_csv(OUT / "diversion_sweep.csv")
    fig, a1 = plt.subplots(figsize=(W2, 2.7)); a2 = a1.twinx(); a2.spines["right"].set_visible(True)
    for bm, col in ((1.0, BLUE), (1.5, VERMIL)):
        r = d[d.budget_mult == bm].sort_values("diversion_scale")
        a1.plot(r.diversion_scale, r.delta_pct, color=col, lw=1.2, label=f"Excess emissions, budget ratio {bm:g}")
    r = d[d.budget_mult == 1.0].sort_values("diversion_scale"); r = r[r.diversion_scale >= 0.25]
    a2.plot(r.diversion_scale, r.rho, color=GREEN, lw=1.2, ls=(0, (5, 2)), label="Cost-emissions alignment (right axis)")
    tolerance(a1); a1.axvline(1.0, color=REFLINE, lw=0.8, ls=":")
    a1.set_xlabel("Diverted traffic relative to the evidence-based schedule"); a1.set_ylabel("Excess emissions over the minimum (%)")
    a2.set_ylabel("Cost-emissions alignment", color=GREEN); a2.tick_params(axis="y", colors=GREEN); a2.set_ylim(0.9, 1.0)
    a1.set_ylim(bottom=0); a1.set_xlim(0, d.diversion_scale.max()); grid(a1)
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    leg = a2.legend(h1 + h2, l1 + l2, loc="center right", frameon=True, facecolor="white", edgecolor="none", framealpha=1); leg.set_zorder(10)
    save(fig, "figS2_diversion_sweep")


# ------------------------------------------------------------------ Fig. S3  Monte Carlo distributions
def figS3_montecarlo():
    mc = pd.read_csv(OUT / "montecarlo_results.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"wspace": 0.28})
    for ax, col, color, xl, letter, title in ((a1, "por_carbon_%", BLUE, "Excess emissions over the minimum (%)", "a", "Emissions"),
                                              (a2, "prem90_%", ORANGE, "Cost premium at 90% coverage (%)", "b", "Protection")):
        v = mc[col].dropna()
        ax.hist(v, bins=30, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(v.median(), color=INK, lw=0.9, ls=(0, (4, 2)), label=f"Median {v.median():.2f}%")
        ax.set_xlabel(xl); ax.set_ylabel("Draws"); grid(ax); panel(ax, letter, title)
    a1.axvline(5, color=VERMIL, lw=1.0, ls=":", label="5% tolerance")
    a1.legend(loc="upper center"); a2.legend(loc="upper right")
    save(fig, "figS3_montecarlo")


# ------------------------------------------------------------------ Fig. S4  solver benchmark
def figS4_solver():
    c = pd.read_csv(OUT / "solver_convergence.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw={"wspace": 0.3})
    a1.fill_between(c.generation, c.igd_min, c.igd_max, color=SKY, alpha=0.3, lw=0, label="Range over eight seeds")
    a1.plot(c.generation, c.igd_mean, color=BLUE, lw=1.2, label="Mean")
    a1.set_xlabel("Generation"); a1.set_ylabel("IGD to the integer-programming front"); grid(a1); a1.legend(loc="lower left")
    panel(a1, "a", "Genetic-algorithm convergence")
    sp = OUT / "solver_scalability.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        a2.plot(s.n_assets, s.exact_front_s, color=BLUE, lw=1.1, marker="o", ms=2.8, label="Integer programming (nine-point front)")
        a2.plot(s.n_assets, s.ga_front_s, color=VERMIL, lw=1.1, ls=(0, (5, 2)), marker="s", ms=2.8, label="Genetic algorithm (one front)")
        a2.legend(loc="upper left")
    a2.set_xlabel("Number of assets"); a2.set_ylabel("Solve time (s)"); grid(a2); panel(a2, "b", "Solve time")
    save(fig, "figS4_solver")


MAIN = ("fig1_framework", "fig2_study_area", "fig3_screen", "fig4_budget", "fig5_conventions", "fig6_knee", "fig7_embodied")
SUPPLEMENT = ("figS1_program_map", "figS2_diversion_sweep", "figS3_montecarlo", "figS4_solver")


def main():
    """Figures 1-7 of the paper and S1-S4 of the supplemental materials."""
    for name in MAIN + SUPPLEMENT:
        print(name)
        try:
            globals()[name]()
        except FileNotFoundError as e:
            print(f"  skipped: input not found ({e.filename}); run run_all.py first")


if __name__ == "__main__":
    main()
