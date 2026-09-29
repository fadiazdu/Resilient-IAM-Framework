# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
bridge_iam_real.py
==================================================================
Self-contained real-data loader for the resilient infrastructure
asset-management framework, grounded in the Rhode Island (FIPS 44)
National Bridge Inventory and standard federal cost authorities.

This single file needs only two data files beside it:
  - the full RI portfolio CSV (e.g. nbi_real_portfolio.csv, ~658 rows)
  - transition_matrix.csv (pooled RI MLE deterioration matrix)
and the packages numpy and pandas. No other module is imported and
nothing in the file needs editing; the data files are located
automatically from the folder the script sits in.

State convention: 4 = Good, 0 = failed. Deterioration moves a bridge
from 4 toward 0. The real matrix is lower-triangular with state 0
absorbing, which is the correct direction.
==================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path


# =================================================================
# 1. PARAMETER PROVENANCE  (value | why | source)
# =================================================================
# Every assignment is grounded; no value is invented. Items flagged
# "ANCHOR PENDING" use a defensible placeholder structure whose
# magnitude must be locked from the cited primary table.
# -----------------------------------------------------------------
SEED        = 2024                 # reproducibility of Monte-Carlo draws
import config as _C
HORIZON     = _C.ANALYSIS_YEARS      # benefit-accounting period (yr); budget period is config.PROGRAM_YEARS

# --- Discount rate -----------------------------------------------
# 0.02 real: OMB Circular A-94 cost-effectiveness uses the real
# Treasury rate matched to the horizon; CY2025 App. C (2026 Budget)
# places the ~10-yr real rate near 2%. Upper sensitivity bound 0.07
# reflects USDOT BCA Guidance (2024).
# Source: OMB Circular A-94 App. C (2025); USDOT BCA Guidance (2024).
DR          = _C.DR   # reference real discount rate (config.py)
DR_SENS     = (0.023, 0.04, 0.07)   # reported discount-rate sensitivity band

# --- Capital unit costs (Rhode Island specific) ------------------
# Source: FHWA Bridge Replacement Unit Costs 2024 (sd2024), RI rows.
# Replacement (cost used for 2024 estimates): NHS 738 $/ft2, non-NHS
# 847 $/ft2. Rehabilitation = 68% of replacement (FHWA-defined ratio).
FT2_PER_M2          = 10.7639
REPL_NHS_FT2        = 738.0
REPL_NONHS_FT2      = 847.0
REHAB_FRACTION      = 0.68     # module-level; perturbed directly by montecarlo.py
# Full replacement rate ($/m2), used for the structural-failure consequence
# (a failed bridge is replaced, not rehabilitated).
COST_REPL_NHS_M2    = REPL_NHS_FT2   * FT2_PER_M2                    # ~7,944 $/m2
COST_REPL_NONHS_M2  = REPL_NONHS_FT2 * FT2_PER_M2                    # ~9,117 $/m2


def cost_major_nhs_m2():
    """NHS major-rehab unit cost (M USD/m2). A function, not a frozen constant,
    so perturbing REHAB_FRACTION (e.g. in the parameter Monte Carlo) actually
    reaches every cost computed from it; a module-level constant computed once
    at import time would not (this was a real bug: the Monte Carlo perturbed
    REHAB_FRACTION but the capital costs derived from it were already baked
    in and never changed)."""
    return REPL_NHS_FT2 * REHAB_FRACTION * FT2_PER_M2   # ~5,402 $/m2 at the baseline 0.68


def cost_major_nonhs_m2():
    """Non-NHS major-rehab unit cost (M USD/m2); see cost_major_nhs_m2()."""
    return REPL_NONHS_FT2 * REHAB_FRACTION * FT2_PER_M2   # ~6,200 $/m2 at the baseline 0.68
# Minor intervention = preventive maintenance package. Grounded as a
# bundle of NCDOT preservation activities (LMC deck overlay $20/ft2 +
# joints $5 + steel painting $17 + concrete spall repair $8 ~ $50/ft2 =
# ~$538/m2), which is ~10% of the rehabilitation rate.
# Source: NCDOT IMPP Typical Bridge Preservation Unit Costs; FHWA rehab.
PREV_FRACTION       = 0.10

# --- Road-user value of time -------------------------------------
# 21.80 $/person-hr (auto, 2024$). Source: USDOT BCA Guidance VTTS
# (Table A-3); RI wages sit near the national median these rest on.
VOT          = 21.80
VOT_TRUCK    = None   # None: every diverted vehicle is valued at VOT (reference); a number values truck time separately (sensitivity case)

# --- Detour assumption (NBI Item 19 not carried into portfolio) --
# ANCHOR PENDING: per-bridge NBI Item 19 detour length.
DETOUR_KM    = 8.0
DETOUR_MULT  = 1.0   # scales every measured detour distance; varied only in the Monte Carlo
DETOUR_SPD   = 50.0
def _reference_diversion():
    """(closure shares, load-posting shares) by state 0..4, from diversion_evidence.py (NBI Item 41).
    'original' is the high-diversion scenario: closure-type diversion 1.0 / 0.25 / 0.05, no posting stream."""
    import json
    if _C.REFERENCE_DIVERSION == "original":
        return list(_C.ORIGINAL_PHI), [0.0] * 5
    ev = json.load(open(_C.ROOT / "diversion_evidence.json"))[_C.REFERENCE_DIVERSION]
    return list(ev["closed"]), list(ev["posted"])
PHI, PHI_POST = _reference_diversion()
# PHI[s]: share of bridges in state s that are closed; all traffic detours (car/truck mix of the bridge).
# PHI_POST[s]: share posted for load; only trucks detour (truck operating cost and truck emission factor).


def set_diversion(closed, posted=None):
    """Set both diversion streams in place (every scenario must set both)."""
    PHI[:] = list(closed); PHI_POST[:] = list(posted) if posted is not None else [0.0] * 5
TRUCK_VOC    = 0.85
TRUCK_MULT   = 1.0    # multiplier on each bridge's reported truck share (varied only in the Monte Carlo)
TRUCK_FALLBACK = 0.07 # statewide median of NBI Item 109, used only where a bridge's value is missing


def truck_share(b):
    """Truck share of the bridge's traffic: its own NBI Item 109 value (x TRUCK_MULT). One source
    for road-user cost and use-phase emissions, so the two can never use different values."""
    t = getattr(b, "truck_frac", float("nan"))
    return TRUCK_MULT * (TRUCK_FALLBACK if t != t else t)

# --- Routine O&M by condition state ($/m2/yr) --------------------
# Grounded by annualizing NCDOT preservation activities over their
# service cycles and scaling by condition (cost rises as condition
# worsens, per NCHRP 668): good ~$1/ft2/yr (washing + periodic sealing)
# rising to ~$5/ft2/yr in poor condition (frequent patching, steel
# repair, painting); converted at 10.764 ft2/m2. State 0 = failed
# (closed, no routine O&M).
# Source: NCDOT IMPP Typical Bridge Preservation Unit Costs; NCHRP 668.
MAINT        = {4: 11.0, 3: 16.0, 2: 30.0, 1: 54.0, 0: 0.0}

# --- Inspection ($/biennial cycle) -------------------------------
# USER ASSUMPTION (confirm against RIDOT inspection unit cost): a clean
# national per-bridge routine-inspection cost is not published; routine
# NBIS inspections for typical bridges run on the order of a few
# thousand dollars all-in (access, traffic control), more for
# underwater/fracture-critical. Set to $5,000; LCC sensitivity is
# negligible relative to capital. Source basis: NBIS biennial requirement.
INSPECTION   = 5_000.0

# --- Structural-failure consequence ($) --------------------------
# Derived per bridge in build_portfolio_real as full replacement cost
# plus capitalized user-detour disruption over the closure/rebuild
# period. The constant below is only a fallback if fail_cost is unset.
# Source: FHWA replacement (RI) + USDOT VTTS + detour exposure.
FAIL_COST    = 12_500_000.0
CLOSURE_YEARS = 2.0    # emergency closure + reconstruction duration for a failed bridge

# --- Candidate triage --------------------------------------------
# Primary candidate set: model state <= 2 (FHWA Poor + Fair). Robustness
# set: state <= 1 (Poor only). Source: FHWA 23 CFR 490 Good/Fair/Poor.
CANDIDATE_MAX_STATE = 2


# =================================================================
# 2. BRIDGE DATA STRUCTURE
# =================================================================
@dataclass
class Bridge:
    bid: str
    x: float; y: float
    deck_m2: float
    n_lanes: int
    aadt: float
    state0: int
    year_built: int
    span_m: float
    is_highway: bool
    region: str
    truck_frac: float = float("nan")   # NBI Item 109 (fraction of ADT)
    T: np.ndarray = field(default_factory=lambda: np.eye(5))
    int_type: str       = "none"
    int_year: int        = 1
    cap_cost_M: float    = 0.0
    bcr: float           = 0.0
    risk_score: float    = 0.0
    fail_cost: float     = 0.0     # bridge-specific failure consequence ($)
    strategic_imp: float = 0.0
    traj_base: List[int] = field(default_factory=list)
    traj_int:  List[int] = field(default_factory=list)


# =================================================================
# 3. SIMULATION + COST ENGINE
# =================================================================
MAJOR_RESET_DIST = {4: 1.0}   # post-major-rehab state distribution; default: resets fully to state 4.
                              # Perturbed by the rehab-reset-state sensitivity check (state 3, or a
                              # probabilistic mix) without changing default behavior elsewhere.


def _mc_exact(b: Bridge, int_year: int, int_type: str) -> Tuple[List[np.ndarray], float]:
    """Exact state-distribution propagation over the horizon.

    Replaces Monte Carlo path simulation with closed-form expectation:
    the model has only five states, so the year-by-year distribution is
    propagated exactly through the transition matrix rather than
    estimated from sampled paths and then rounded. Rounding the mean
    simulated state before evaluating the (nonlinear, discrete) cost and
    diversion functions understates their expectation in general, since
    E[f(S)] != f(round(E[S])) for a nonlinear f; exact propagation
    removes that gap entirely rather than approximating it more finely.

    An intervention in `int_year` resets condition by moving probability
    mass between states: 'major' moves all mass to the distribution in
    MAJOR_RESET_DIST (state 4 by default); 'minor' moves the mass at
    state s to min(s + 2, 3). State 0 is an absorbing failure. Returns
    the list of state-distributions (length 5 arrays, year 0..HORIZON)
    and the end-of-horizon failure probability.
    """
    dist = np.zeros(5)
    dist[b.state0] = 1.0
    dists = [dist.copy()]
    for t in range(1, HORIZON + 1):
        prev = dists[-1]
        if t == int_year and int_type != "none":
            reset = np.zeros(5)
            if int_type == "major":
                for r, frac in MAJOR_RESET_DIST.items():
                    reset[r] += prev.sum() * frac
            else:
                for s in range(5):
                    r = min(s + 2, 3)
                    reset[r] += prev[s]
            prev = reset
        nxt = np.zeros(5)
        nxt[0] += prev[0]                      # state 0 is absorbing
        for s in range(1, 5):
            if prev[s] > 0:
                nxt += prev[s] * b.T[s]
        dists.append(nxt)
    p_fail = float(dists[-1][0])
    return dists, p_fail


def _pv_user_cost(b: Bridge, dists: List[np.ndarray]) -> float:
    """Discounted present value of road-user (detour/posting) cost (M USD),
    as the exact expectation of the per-state cost over the state
    distribution at each year (dists[yr] is a length-5 probability vector)."""
    vpy   = b.aadt * 365
    dk    = getattr(b, "detour_km", DETOUR_KM)   # real NBI Item 19; falls back if unset
    dt_h  = dk / DETOUR_SPD
    vot_d = VOT * dt_h
    ts    = truck_share(b)
    # Closed bridge: all vehicles detour (time for all, truck operating cost for the truck share).
    # Load-posted bridge: only trucks detour (time and truck operating cost for each truck).
    if VOT_TRUCK is None:                      # reference: one time value for every vehicle
        closure = vot_d + dk * TRUCK_VOC * ts
        posting = ts * (vot_d + dk * TRUCK_VOC)
    else:                                      # sensitivity case: cars at VOT, trucks at VOT_TRUCK
        vot_t = VOT_TRUCK * dt_h
        closure = (1 - ts) * vot_d + ts * vot_t + dk * TRUCK_VOC * ts
        posting = ts * (vot_t + dk * TRUCK_VOC)
    per_state = vpy * (np.array(PHI, dtype=float) * closure + np.array(PHI_POST, dtype=float) * posting)
    # Audit correction: two undocumented terms were removed here, a state-1 truck-posting
    # charge that applied the truck share twice, and a state-2 slowdown based on average
    # span rather than bridge length. User cost is detour time plus truck operating cost only.
    total = 0.0
    for yr in range(1, HORIZON + 1):
        d = (1 + DR) ** (-yr)
        total += d * float(dists[yr] @ per_state)
    return total / 1e6


def _lcc_omr(b: Bridge, dists: List[np.ndarray]) -> float:
    """Discounted O&M + inspection + failure-risk cost (M USD), no capital,
    as the exact expectation over the state distribution at each year."""
    fc = b.fail_cost if b.fail_cost > 0 else FAIL_COST
    maint = np.array([MAINT[s] for s in range(5)])
    frisk = np.array([fc * 0.22, fc * 0.04, 0.0, 0.0, 0.0])   # states 0,1,2,3,4
    total = 0.0
    for yr in range(1, HORIZON + 1):
        d = (1 + DR) ** (-yr)
        insp = INSPECTION if yr % 2 == 0 else 0.0
        total += d * (b.deck_m2 * float(dists[yr] @ maint) + insp + float(dists[yr] @ frisk))
    return total / 1e6


def _cap_cost_M(b: Bridge, yr: int, int_type: str) -> float:
    """NHS-aware discounted capital cost (M USD).

    Selects the RI replacement unit cost by NHS membership (is_highway
    proxy), applies the 68% rehab fraction for a major intervention and
    the preventive fraction for a minor one.
    """
    if int_type == "major":
        rate = cost_major_nhs_m2() if b.is_highway else cost_major_nonhs_m2()
    else:
        base = cost_major_nhs_m2() if b.is_highway else cost_major_nonhs_m2()
        rate = base * PREV_FRACTION
    return b.deck_m2 * rate * (1 + DR) ** (-yr) / 1e6


def _bcr_simulated(b: Bridge, int_year: int, int_type: str
                   ) -> Tuple[float, float]:
    """BCR from exact state-distribution propagation (used for states 0-2)."""
    dists_b, _ = _mc_exact(b, -1, "none")
    dists_i, _ = _mc_exact(b, int_year, int_type)
    avoided_omr = _lcc_omr(b, dists_b) - _lcc_omr(b, dists_i)
    avoided_uc  = _pv_user_cost(b, dists_b) - _pv_user_cost(b, dists_i)
    frisk_ind = np.array([0.22, 0.04, 0.0, 0.0, 0.0])   # states 0,1,2,3,4
    risk_b = sum(float(dists_b[y] @ frisk_ind) for y in range(1, HORIZON + 1))
    risk_i = sum(float(dists_i[y] @ frisk_ind) for y in range(1, HORIZON + 1))
    safety = max(0.0, (risk_b - risk_i) * (b.fail_cost if b.fail_cost > 0 else FAIL_COST) / 1e6)
    cap    = max(0.001, _cap_cost_M(b, int_year, int_type))
    bcr    = max(0.0, avoided_omr + avoided_uc + safety) / cap
    return round(bcr, 3), round(cap, 3)


def _risk_score(b: Bridge, dists: List[np.ndarray]) -> float:
    """Composite 0-10 risk score from the exact expected condition and
    traffic exposure."""
    state_vals = np.array([0, 1, 2, 3, 4])
    avg_s = float(np.mean([dists[y] @ state_vals for y in range(1, HORIZON + 1)]))
    structural = (4 - avg_s) / 4 * 5
    traffic    = min(5.0, np.log10(max(b.aadt, 100)) / np.log10(100_000) * 5)
    base = 0.60 * structural + 0.40 * traffic
    if b.is_highway: base *= 1.30
    if b.n_lanes >= 4: base *= 1.10
    return round(min(10.0, base), 2)


# =================================================================
# 4. DATA FILE DISCOVERY  (no manual editing required)
# =================================================================
REQUIRED_COLS = {"bid", "x_lon", "y_lat", "deck_m2", "n_lanes", "aadt",
                 "truck_frac", "state0", "year_built", "span_m", "is_highway"}


def _search_dirs() -> List[Path]:
    dirs = []
    try:
        dirs.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    dirs.append(Path.cwd())
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d); out.append(d)
    return out


def _find_matrix_csv() -> Path:
    for d in _search_dirs():
        p = d / "transition_matrix.csv"
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find 'transition_matrix.csv' next to this script.\n"
        "Put it and the full RI portfolio CSV in the same folder, then rerun.")


def _find_portfolio_csv() -> Path:
    """Return the largest valid RI portfolio CSV found beside the script."""
    preferred = ["nbi_real_portfolio.csv", "real_ri.csv"]
    valid = []
    for d in _search_dirs():
        candidates = [d / n for n in preferred]
        candidates += [p for p in d.glob("*.csv")
                       if p.name not in preferred
                       and p.name not in ("transition_matrix.csv",
                                          "transition_counts.csv")
                       and "matrix" not in p.name.lower()
                       and "count" not in p.name.lower()]
        for p in candidates:
            if not p.exists():
                continue
            try:
                cols = set(pd.read_csv(p, nrows=1).columns)
                with open(p) as fh:
                    n_rows = sum(1 for _ in fh) - 1
            except Exception:
                continue
            if REQUIRED_COLS.issubset(cols) and n_rows >= 50:
                valid.append((p, p.stat().st_size))
    if not valid:
        raise FileNotFoundError(
            "Could not find a valid RI portfolio CSV next to this script.\n"
            "Put the full inventory file (e.g. 'nbi_real_portfolio.csv', the "
            "~658-row version) in the same folder. A truncated copy with too "
            "few rows is ignored.")
    return max(valid, key=lambda t: t[1])[0]   # largest valid file wins


MATRIX_CSV    = _find_matrix_csv()
PORTFOLIO_CSV = _find_portfolio_csv()


# =================================================================
# 5. REAL TRANSITION MATRIX
# =================================================================
def load_real_matrix(path: Path = MATRIX_CSV) -> np.ndarray:
    """Load and validate the pooled RI deterioration matrix (5x5).

    Lower-triangular with state 0 absorbing: probability moves from
    better states toward worse ones, the correct deterioration direction.
    """
    df = pd.read_csv(path, index_col=0)
    T = df.to_numpy(dtype=float)
    assert T.shape == (5, 5), f"matrix must be 5x5, got {T.shape}"
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-6), "rows must sum to 1"
    assert np.allclose(T[0], [1, 0, 0, 0, 0]), "state 0 must be absorbing"
    assert np.allclose(np.triu(T, k=1), 0.0, atol=1e-9), \
        "matrix must not move bridges to better states"
    return T


REAL_T = load_real_matrix()


# =================================================================
# 6. COORDINATE CONVERSION  (packed NBI DMS -> decimal degrees)
# =================================================================
def nbi_to_decimal(packed: float, is_lon: bool) -> float:
    """Convert an NBI packed coordinate to signed decimal degrees.

    Latitude is DDMMSSss, longitude DDDMMSSss (degrees, minutes, seconds,
    hundredths). As integers the longitude loses its leading zero, so it
    is left-padded to nine digits. U.S. longitudes are west (negative).
    """
    s = str(int(round(packed)))
    s = s.zfill(9) if is_lon else s.zfill(8)
    deg = int(s[:-6]); mn = int(s[-6:-4]); sec = int(s[-4:]) / 100.0
    dec = deg + mn / 60.0 + sec / 3600.0
    return -dec if is_lon else dec


# =================================================================
# 7. BUILD THE REAL PORTFOLIO
# =================================================================
def compute_failure_cost(b: Bridge) -> float:
    """Bridge-specific structural-failure consequence ($).

    A failed bridge must be replaced (full replacement, not rehab) and
    imposes user-detour costs over the emergency-closure and
    reconstruction period. Grounded components: FHWA RI replacement
    unit cost, USDOT VTTS, and detour exposure.
    """
    repl_rate = COST_REPL_NHS_M2 if b.is_highway else COST_REPL_NONHS_M2
    replacement = b.deck_m2 * repl_rate
    dk = getattr(b, "detour_km", DETOUR_KM)
    vot_d = VOT * (dk / DETOUR_SPD)                     # $/veh per detour (time)
    voc_d = dk * TRUCK_VOC * truck_share(b)             # $/veh per detour (truck op.)
    annual_detour = b.aadt * 365 * (vot_d + voc_d)      # full-closure user cost per yr
    return replacement + annual_detour * CLOSURE_YEARS


def build_portfolio_real(candidate_max_state: int = CANDIDATE_MAX_STATE):
    """Load the RI portfolio, triage to the candidate set, attach the
    real matrix and per-bridge failure cost. Returns (candidates, full, T)."""
    df = pd.read_csv(PORTFOLIO_CSV)
    missing = REQUIRED_COLS - set(df.columns)
    assert not missing, f"portfolio missing columns: {missing}"
    df = df.dropna(subset=list(REQUIRED_COLS)).copy()

    global TRUCK_FALLBACK
    if df["truck_frac"].notna().any():
        TRUCK_FALLBACK = float(df["truck_frac"].median())   # statewide median of reported Item 109 values
    full = []
    for _, r in df.iterrows():
        b = Bridge(
            bid        = str(r["bid"]),
            x          = nbi_to_decimal(r["x_lon"], is_lon=True),
            y          = nbi_to_decimal(r["y_lat"], is_lon=False),
            deck_m2    = float(r["deck_m2"]),
            n_lanes    = int(r["n_lanes"]),
            aadt       = float(r["aadt"]),
            state0     = int(r["state0"]),
            year_built = int(r["year_built"]),
            span_m     = float(r["span_m"]),
            is_highway = bool(r["is_highway"]),
            region     = "RI",
            truck_frac = float(r["truck_frac"]) if pd.notna(r["truck_frac"]) else float("nan"),
            T          = REAL_T.copy(),
        )
        raw = r.get("detour_km_019", None)
        try:
            dk = float(raw)
            if pd.isna(dk):
                dk = DETOUR_KM                  # truly missing -> flagged fallback
        except (TypeError, ValueError):
            dk = DETOUR_KM                       # unparsable -> flagged fallback
        # NBI Item 19 code 000 is a valid "ground-level bypass available" reading,
        # not a missing value (FHWA Coding Guide, Item 19): a zero detour is kept
        # as zero, not silently replaced by the fallback.
        b.detour_km = dk * DETOUR_MULT          # measured Item 19 distance (multiplier = 1 at baseline)
        b.fail_cost = compute_failure_cost(b)   # not an objective term; uses the measured detour
        full.append(b)

    candidates = [b for b in full if b.state0 <= candidate_max_state]
    return candidates, full, REAL_T


# =================================================================
# 8. SANITY CHECK  (run directly)
# =================================================================
if __name__ == "__main__":
    cands, full, T = build_portfolio_real()

    print("=" * 64)
    print("REAL RHODE ISLAND PORTFOLIO — SANITY CHECK")
    print("=" * 64)
    print(f"Portfolio file used        : {PORTFOLIO_CSV.name}")
    print(f"Bridges with complete data : {len(full)}")
    dist = {s: sum(1 for b in full if b.state0 == s) for s in range(5)}
    print(f"Condition distribution     : {dist}")
    print(f"Candidate set (state<= {CANDIDATE_MAX_STATE})    : {len(cands)}")
    print(f"  Poor-only (state<=1)     : {sum(1 for b in full if b.state0 <= 1)}")

    xs = [b.x for b in full]; ys = [b.y for b in full]
    print(f"Lon range  : {min(xs):.3f} .. {max(xs):.3f}  (RI ~ -71.9..-71.1)")
    print(f"Lat range  : {min(ys):.3f} .. {max(ys):.3f}  (RI ~ 41.1..42.0)")

    print("-" * 64)
    print("Deterioration test (no intervention) — expected state by year:")
    for s0 in (4, 3, 2):
        probe = Bridge(bid="probe", x=-71.4, y=41.8, deck_m2=500,
                       n_lanes=2, aadt=10000, state0=s0, year_built=1990,
                       span_m=20, is_highway=False, region="RI",
                       T=REAL_T.copy())
        dists, _ = _mc_exact(probe, -1, "none")
        state_vals = np.array([0, 1, 2, 3, 4])
        exp_traj = [round(float(d @ state_vals), 2) for d in dists]
        trend = "DECLINES (correct)" if exp_traj[-1] <= exp_traj[0] else "IMPROVES (BUG)"
        print(f"  start={s0}: {exp_traj}  ->  {trend}")

    total_rehab_M = sum(_cap_cost_M(b, 1, "major") for b in cands)
    print("-" * 64)
    print(f"Candidate rehabilitation capital need (yr-1, disc.): "
          f"${total_rehab_M:,.1f} M")
    print("  (FHWA RI poor-bridge rehab estimate ~ $600 M for 119 poor bridges)")

    print("-" * 64)
    print("Sample candidate economics (first 3):")
    for b in cands[:3]:
        bcr, cap = _bcr_simulated(b, 1, "major")
        print(f"  {b.bid:>16}  state0={b.state0}  deck={b.deck_m2:>7.0f} m2  "
              f"NHS={b.is_highway}  cap=${cap:5.2f}M  BCR={bcr:5.2f}")
