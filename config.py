# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
config.py  -  single source of truth for the bridge IAM pipeline
=================================================================
Every cost, carbon, and modelling constant lives here with its value,
rationale, and source. Flagged parameters (swept in sensitivity, not
measured) are grouped at the end. No other module hard-codes a number.

Pipeline run order (see run_all.py):
  1 fetch_nbi    2 build_matrix    3 build_portfolio    4 engine
  5 economics    6 optimizer       7 conflict           8 front
  9 sensitivity 10 figures
"""
from __future__ import annotations
import os
from pathlib import Path

# -----------------------------------------------------------------
# Paths (everything resolves next to this file unless overridden)
# -----------------------------------------------------------------
ROOT       = Path(os.environ.get("IAM_ROOT", Path(__file__).resolve().parent))
DATA_DIR   = ROOT   # the two data files live in the package root, not a subfolder:
                     # engine.py's _find_matrix_csv() and adaptation.py's CSV both look
                     # here directly (next to the scripts), so this must match, or a
                     # rebuild via build_matrix.py/build_portfolio.py would silently
                     # write to a location the rest of the pipeline never reads.
OUT_DIR    = Path(os.environ.get("IAM_OUT",  ROOT / "outputs"))
FIG_DIR    = Path(os.environ.get("IAM_FIG",  ROOT / "figures"))
for _d in (DATA_DIR, OUT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_NBI_TMPL    = str(DATA_DIR / "nbi_raw_{state}_{year}.csv")  # per-year raw NBI
MATRIX_CSV      = str(DATA_DIR / "transition_matrix.csv")
PORTFOLIO_CSV   = str(DATA_DIR / "nbi_real_portfolio.csv")      # matches adaptation.py's CSV exactly
RESULTS_DIR     = OUT_DIR                                       # CSV/JSON artifacts

# -----------------------------------------------------------------
# Study area and data vintage
# -----------------------------------------------------------------
STATE_FIPS   = "44"                       # Rhode Island
NBI_YEARS    = [2019, 2020, 2021, 2022, 2023, 2024]  # pooled consecutive transitions
SEED         = 12345                       # reproducible Monte Carlo

# -----------------------------------------------------------------
# Condition model: NBI governing component rating (0-9, the minimum of
# deck/superstructure/substructure) -> 5-state model, state 4 = best
# (rating 8-9) ... 0 = worst (rating 0-3). This is an author-defined
# collapse for the Markov model's granularity; it is coarser than FHWA's
# per-rating descriptors and does not align with FHWA's official 3-tier
# Good/Fair/Poor classification (23 CFR 490.409), whose Fair band (ratings
# 5-6) and Poor band (ratings 0-4) split differently than the boundaries
# used here. See STATE_LABELS below for the exact rating band per state;
# this crosswalk is the single source used both to assign each bridge's
# starting state (build_portfolio.py) and to estimate the transition
# matrix (build_matrix.py), so the two stay consistent with each other.
# -----------------------------------------------------------------
NBI_TO_STATE = {9:4, 8:4, 7:3, 6:2, 5:2, 4:1, 3:0, 2:0, 1:0, 0:0}
N_STATES     = 5
CANDIDATE_MAX_STATE = 2                    # Poor+Fair candidates; robustness set = 1
HORIZON      = 10                          # analysis horizon (years)

# -----------------------------------------------------------------
# Discount rate.  Source: OMB Circular A-94 App. C (2025), ~10-yr real
# rate near 2%; USDOT BCA Guidance (2024) upper bound 7%.
# -----------------------------------------------------------------
DR        = 0.023     # real 30-year Treasury rate, OMB Circular A-94 Appendix C (2025), for constant-dollar cost-effectiveness analysis
DR_SENS   = (0.023, 0.04, 0.07)

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# Capital unit costs (Rhode Island) live in engine.py, not here: they were
# duplicated in both files, and engine.py's copies are the ones actually
# used (economics.py imports engine, not config, for these). A duplicate
# set of constants here that nothing reads is a latent inconsistency risk,
# so it has been removed rather than kept in sync by hand. See engine.py's
# REHAB_FRACTION, cost_major_nhs_m2(), cost_major_nonhs_m2(), and
# PREV_FRACTION for the FHWA-sourced unit costs and their derivation.
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# Routine O&M by state ($/m2/yr). Source: NCDOT IMPP preservation unit
# costs annualized, scaled by condition (NCHRP 668). State 0 = closed.
# -----------------------------------------------------------------
MAINT        = {4:11.0, 3:16.0, 2:30.0, 1:54.0, 0:0.0}
INSPECTION   = 5_000.0    # USER ASSUMPTION: biennial NBIS inspection; negligible LCC sensitivity

# -----------------------------------------------------------------
# Road-user cost.  VOT 21.80 $/person-hr (USDOT BCA VTTS, auto, 2024$).
# Detour speed/VOC are engineering defaults; per-bridge detour length
# now comes from NBI Item 19 (column detour_km_019), 8 km fallback.
# -----------------------------------------------------------------
VOT          = 21.80
DETOUR_SPD   = 50.0       # km/h
DETOUR_KM    = 8.0        # FLAGGED fallback when Item 19 == 0/blank
TRUCK_VOC    = 0.85       # $/veh-km truck operating cost (engineering default)
# Truck share: each bridge's NBI Item 109 value (engine.truck_share); no network default

# -----------------------------------------------------------------
# Failure consequence.  Per-bridge = full replacement + capitalized
# detour disruption over a 2-yr closure (built in engine). Fallback only.
# -----------------------------------------------------------------
FAIL_COST     = 12_500_000.0
CLOSURE_YEARS = 2.0

# -----------------------------------------------------------------
# Carbon factors
#   EF_CAR : EPA 0.404 kgCO2/mi / 1.609 = 0.251 kgCO2e/veh-km
#   EF_TRUCK, EMB_*, operational tiers, low-carbon deltas are FLAGGED.
# -----------------------------------------------------------------
EF_CAR        = 0.404 / 1.60934    # kgCO2e/veh-km (EPA Green Vehicle Guide)
EF_TRUCK      = 1.00               # FLAGGED kgCO2e/veh-km (HDV proxy)
EMB_REHAB_M2  = 340.0              # FLAGGED kgCO2e/m2 embodied, rehab (bridge LCA lit.)
OPER_TIER     = {4:0.0, 3:0.0, 2:0.05, 1:0.25, 0:1.0}  # FLAGGED use-phase multiplier by state
LOWC_EMB      = 0.65               # FLAGGED low-carbon embodied multiplier (-35%)
LOWC_COST     = 1.18               # FLAGGED low-carbon cost multiplier (+18%)

# -----------------------------------------------------------------
# Program budget.  Source: RI Bridge Formula Program (IIJA) $47.1M/yr.
# Mega-projects (single cost > annual budget) excluded as dedicated grants.
# -----------------------------------------------------------------
BUDGET_ANNUAL_M = 50.99   # RI Bridge Formula Program, FY2022 apportionment (FHWA Notice N 4510.861/
                          # .867); $255.0M over FY2022-2026 (ARTBA 2025 RI bridge report) confirms
                          # ~$51.0M/yr. Corrects an earlier $47M figure that had no traceable source.
PROGRAM_YEARS   = 10        # years of Bridge Formula Program funding pooled into the capital budget
BUDGET_M        = BUDGET_ANNUAL_M * PROGRAM_YEARS   # program capital (independent of the analysis period)
# ---- reference formulation (see README) ------------------------------------------------
ANALYSIS_YEARS  = 35        # benefit-accounting period: FHWA LCCA policy (1996) minimum of 35 years
NHS_SD_LIMIT    = 0.10      # 23 CFR 490.411: structurally deficient NHS deck area <= 10 percent
NHS_WINDOW_YEARS = 3        # enforced in years 1-3 (the 3-year period of 23 U.S.C. 119(f)(2))
REFERENCE_DIVERSION = "pooled"   # diverted-traffic schedule: 'pooled' or 'y2024' (from NBI Items 41, 109), or 'original'
ORIGINAL_PHI    = [1.0, 0.25, 0.05, 0.0, 0.0]   # the previously assumed schedule, kept for comparison

# =================================================================
# FLAGGED ADAPTATION PARAMETERS  (swept in sensitivity, not measured)
# =================================================================
# Countermeasure capital as a fraction of rehab capital. HEC-23 is the
# design reference; cost comes from state DOT bid tabs (range 0.15-0.25).
CM_FRACTION   = 0.20
CM_FRAC_SENS  = (0.15, 0.20, 0.25, 0.40)
# Countermeasure embodied carbon as a fraction of rehab embodied (riprap = low).
EMB_CM_FRAC   = 0.10
EMB_CM_SENS   = (0.05, 0.10, 0.20)

# NBI Item 113 (Scour Critical) code -> hazard severity weight in [0,1].
# FHWA: codes 0-3 are scour critical; 4 marginal; 5-9 stable; N not over
# water; U unknown foundation (flagged); T tidal low-risk. Ordinal weights
# are a MODELLING CHOICE, swept below. Item 61 <= 4 adds +0.1 (capped 1.0).
SCOUR_SEVERITY = {0:1.0, 1:0.9, 2:0.8, 3:0.6, 4:0.3}   # numeric codes
SCOUR_LETTER   = {"N":0.0, "T":0.1, "U":0.5}
SCOUR_BLANK    = 0.5                                    # missing -> flagged uncertain
SCOUR_MAPS_SENS = {                                     # alternatives for robustness
    "baseline":     dict(num={0:1.0,1:0.9,2:0.8,3:0.6,4:0.3}, U=0.5, T=0.1),
    "strict(0-3)":  dict(num={0:1.0,1:0.9,2:0.8,3:0.6,4:0.0}, U=0.0, T=0.0),
    "binary":       dict(num={0:1.0,1:1.0,2:1.0,3:1.0,4:0.0}, U=1.0, T=0.0),
    "conservative": dict(num={0:1.0,1:0.95,2:0.9,3:0.8,4:0.5}, U=1.0, T=0.2),
}

# =================================================================
# PUBLICATION FIGURE STYLE  (one look for every figure)
# =================================================================
# ASCE column widths (inches)
COL1_IN = 3.50
COL2_IN = 7.16
# Colorblind-safe palette (Okabe-Ito derived) used consistently.
PALETTE = {
    "ink":      "#23272A",   # text / axes
    "primary":  "#21618C",   # cost / efficiency
    "accent":   "#C0563B",   # adaptation / resilience / scour
    "carbon":   "#3C7A52",   # carbon
    "inaction": "#922B21",   # do-nothing / inaction
    "neutral":  "#5A5A5A",   # annotations
    "good":     "#2E7D5B",
    "fair":     "#E1A100",
    "poor":     "#922B21",
    "band":     "#21618C",   # shaded regions (used at low alpha)
    "grid":     "#D7D7D7",
}
# Condition-state ramp (best -> failed): one harmonized green->amber->red sequence.
STATE_COLORS = {4: "#2E7D5B", 3: "#7FB069", 2: "#E1A100", 1: "#D9772B", 0: "#B5402F"}
# Display labels for the 5 model states, by governing NBI rating band (not
# FHWA's own Good/Fair/Poor vocabulary: that vocabulary is defined at a
# different granularity -- e.g. FHWA calls a rating of 5 "Fair", not "Poor" --
# so borrowing those words for a 5-way collapse would misstate them. Labeling
# by rating band is exact and requires no crosswalk to a name.
STATE_LABELS = {4: "NBI 8-9", 3: "NBI 7", 2: "NBI 5-6", 1: "NBI 4", 0: "NBI 0-3"}


def _iam_cmap():
    """Sequential colormap for continuous overlays (e.g. AADT), tuned to the palette."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "iam_seq", ["#E9EFF3", "#7FA8C0", "#21618C", "#11303F"])


def apply_style():
    """Apply the shared matplotlib style. Call once before plotting."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 9,
        "text.color": PALETTE["ink"], "axes.edgecolor": "#8A8A8A",
        "axes.labelcolor": PALETTE["ink"], "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"], "axes.linewidth": 0.8,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlepad": 7,
        "axes.labelsize": 9.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,                       # grids added explicitly per axis
        "grid.color": PALETTE["grid"], "grid.linewidth": 0.6,
        "grid.linestyle": "-", "grid.alpha": 0.9,
        "legend.frameon": False, "legend.fontsize": 8, "legend.handlelength": 1.6,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "lines.linewidth": 1.9, "lines.markersize": 5, "lines.markeredgewidth": 0,
    })


def ygrid(ax):
    """Light horizontal reference grid, consistent across the suite."""
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def panel_label(ax, letter, dx=-0.02, dy=1.04):
    """Bold panel tag, e.g. (a), placed at the top-left of an axis."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left", color=PALETTE["ink"])


def savefig(fig, name):
    """Save a figure as vector PDF (submission) plus PNG (preview)."""
    pdf = FIG_DIR / f"{name}.pdf"; png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf); fig.savefig(png)
    return str(pdf), str(png)


# ---- Value of travel time by vehicle class (sensitivity case only; the reference applies one value per vehicle)
# USDOT (2025), Benefit-Cost Analysis Guidance for Discretionary Grant Programs, Appendix A, Table A-2 (2024 dollars):
# all-purpose (blended personal and business) 21.80 $/person-h (the reference VOT in engine.py),
# personal 20.10 $/person-h, truck drivers 37.20 $/h. Truck operating costs in that guidance exclude driver wages.
VOT_PERSONAL      = 20.10
VOT_TRUCK_DRIVER  = 37.20
