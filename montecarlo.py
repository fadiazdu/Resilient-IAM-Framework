#!/usr/bin/env python3
# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
montecarlo.py  -  parameter-uncertainty robustness of the objective screen
==========================================================================
Monte Carlo over input uncertainty (not sampling uncertainty; the inventory is
the whole population). Each draw perturbs the cost, carbon, hazard, and
deterioration inputs within sourced or flagged ranges, rebuilds the option
economics from the real model (no duplication), and records:
  - rho_vpc  : Spearman corr of cost-per-capital vs carbon-per-capital (screen statistic)
  - rho_cc   : Spearman corr of cost vs carbon deferral marginals
  - rho_cs   : Spearman corr of cost-saving vs scour-removed (adaptation axis)
  - por_carbon : price of redundancy for carbon, % by which the cost-optimal
                 program's carbon exceeds the carbon-optimal program's carbon
  - prem90   : adaptation premium at 90% protection (knee stability)
All ranges are declared in PARAM_RANGES with source/flag. Seed is fixed.
Writes outputs/montecarlo_results.csv.
"""
from __future__ import annotations
import sys, numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import engine as M, economics as E, adaptation as A, problem as P, optimizer as OPT
import screening as S

CO, CA, CAP, SC = OPT.COST, OPT.CARBON, OPT.CAPITAL, OPT.SCOUR
SEED = 12345
N = 512                                          # draws; run_all.py passes IAM_MC_DRAWS
# NOTE: the reduced path count for the sweep is applied inside main() and restored
# afterwards, so the baseline analysis always runs at the full simulation fidelity.

# parameter ranges: (attr-holder, name, low, high, source/flag)
PARAM_RANGES = [
    # The discount rate is a prescribed accounting convention (OMB Circular A-94), not an uncertain
    # input: it is examined as a scenario dimension in objective_diagnostics.discount_horizon_grid.
    ("engine", "VOT",           18.5, 25.1, "USDOT VTTS +/-15%"),
    ("engine", "TRUCK_MULT",    0.80, 1.20, "multiplier on each bridge's reported Item 109 truck share"),
    ("engine", "DETOUR_MULT",   0.75, 1.25, "scales measured Item 19 detours (replaces the inactive fallback)"),
    ("engine", "REHAB_FRACTION",0.60, 0.75, "FHWA practice spread"),
    ("econ",   "EF_CAR",        0.226,0.276,"EPA +/-10%"),
    ("econ",   "EF_TRUCK",      0.70, 1.30, "FLAGGED HDV proxy"),
    ("adapt",  "CM_FRACTION",   0.15, 0.30, "specified sensitivity range"),
]
BASE_T = np.array(M.REAL_T, dtype=float)       # baseline transition matrix

import pandas as _pd
# Bridge-cluster bootstrap matrices from build_matrix.py (audit correction: replaces the
# Dirichlet resampling of annual-record counts, which treated repeated records as observations).
BOOT = np.load(C.ROOT / "transition_bootstrap.npy")


def resample_matrix(d):
    """Transition matrix for draw d: the d-th bridge-cluster bootstrap fit."""
    return np.array(BOOT[d % len(BOOT)], dtype=float)


def screen_stats(prob):
    """Screening statistics for one draw, computed exactly as in screening.py (tie-broken optima)."""
    V = prob.V; B = prob.budget_M; tot = prob.total_scour; n = len(V)
    dcost = np.array([V[i][0][CO] - V[i][1][CO] for i in range(n)])
    dcarb = np.array([V[i][0][CA] - V[i][1][CA] for i in range(n)])
    cap1  = np.array([V[i][1][CAP] for i in range(n)])
    scrm  = np.array([V[i][2][SC] - V[i][0][SC] for i in range(n)])
    m = cap1 > 0
    vpc_cost, vpc_carb, vpc_scour = dcost[m] / cap1[m], dcarb[m] / cap1[m], scrm[m] / cap1[m]
    rho_vpc_carbon = S.rank_corr(vpc_cost, vpc_carb)
    rho_cc_raw     = S.rank_corr(dcost, dcarb)
    rho_vpc_adapt  = S.rank_corr(vpc_cost, vpc_scour)
    _, co = OPT.cost_optimal(V, B)
    _, cmin = OPT.carbon_optimal(V, B)
    # the maximum exposure addressed and the 90% premium depend only on objective values,
    # which are identical across ties, so single-stage solves suffice (halves runtime)
    ch_a = OPT.solve_choice(V, lambda i, g: -V[i][g][SC], B)
    a_max = sum(V[i][ch_a[i]][SC] for i in range(n))
    por = (co["carbon"] - cmin["carbon"]) / cmin["carbon"] * 100.0
    por_a = (a_max - co["protected"]) / a_max * 100.0 if a_max > 0 else np.nan
    ch90 = OPT.solve_choice(V, lambda i, g: V[i][g][CO], B, min_protection=0.90 * tot)
    prem90 = ((sum(V[i][ch90[i]][CO] for i in range(n)) - co["cost"]) / co["cost"] * 100.0
              if ch90 is not None else np.nan)
    return rho_vpc_carbon, rho_cc_raw, rho_vpc_adapt, por, prem90, por_a, co["carbon"] - cmin["carbon"]


def main(n_draws=None):
    global N
    if n_draws: N = n_draws
    rng = np.random.default_rng(SEED)
    # Latin-hypercube design over the continuous parameters (documented, seeded);
    # the transition matrix is the draw's bridge-cluster bootstrap fit.
    from scipy.stats import qmc
    _lo = np.array([p[2] for p in PARAM_RANGES]); _hi = np.array([p[3] for p in PARAM_RANGES])
    _lhs = qmc.LatinHypercube(d=len(PARAM_RANGES), seed=SEED).random(N)
    DESIGN = _lo + _lhs * (_hi - _lo)        # N x len(PARAM_RANGES), scaled to each range
    holders = {"engine": M, "econ": E, "adapt": A}
    # save every constant the sweep perturbs, so the process is left exactly as found
    saved = {(h, n): getattr(holders[h], n) for h, n, *_ in PARAM_RANGES}
    base = P.build()                                    # baseline population, fixed for all draws
    BASE_IDS = {str(b.bid).strip() for b in base.candidates}
    import pickle
    part = C.OUT_DIR / ".mc_partial.pkl"          # checkpoint of finished draws (removed when complete)
    ckpt = pickle.load(open(part, "rb")) if part.exists() else {}
    rows = ckpt.get("rows", []) if ckpt.get("N") == N else []
    done = {int(r[0]) for r in rows}
    if done:
        print(f"  resuming: {len(done)} of {N} draws already finished", flush=True)
    import time as _time
    _t0 = _time.time()
    print(f"  running {N} draws; progress is printed every 10 draws "
          f"(this step typically takes the longest)", flush=True)
    try:
        for d in range(N):
            if d in done:
                continue
            for j, (holder, name, lo, hi, _) in enumerate(PARAM_RANGES):
                setattr(holders[holder], name, float(DESIGN[d, j]))
            M.REAL_T = resample_matrix(d)
            prob = P.build(fixed_ids=BASE_IDS)
            assert len(prob.candidates) == len(BASE_IDS), "candidate population changed"
            rows.append((d,) + tuple(float(DESIGN[d, j]) for j in range(len(PARAM_RANGES)))
                        + tuple(screen_stats(prob)))       # a failed draw stops the run (none dropped)
            if len(rows) % 16 == 0:
                C.OUT_DIR.mkdir(parents=True, exist_ok=True)
                pickle.dump({"N": N, "rows": rows}, open(part, "wb"))
            if (d + 1) % 10 == 0:
                el = _time.time() - _t0
                eta = el / (d + 1) * (N - d - 1)
                print(f"  draw {d+1}/{N} done  |  elapsed {el/60:5.1f} min  |  "
                      f"est. remaining {eta/60:5.1f} min", flush=True)
    finally:                                          # restore baseline state unconditionally
        for (h, n), v in saved.items():
            setattr(holders[h], n, v)
        M.REAL_T = BASE_T
    pnames = [f"p_{name}" for _, name, *_ in PARAM_RANGES]
    stats = ["rho_vpc_carbon", "rho_cc_raw", "rho_vpc_adapt", "por_carbon_%", "prem90_%", "por_adapt_%", "carbon_abs_loss_t"]
    df = _pd.DataFrame(rows, columns=["draw"] + pnames + stats)
    df["carbon_class"] = np.where((df["rho_vpc_carbon"] >= 0.95) & (df["por_carbon_%"] <= 5.0), "redundant", "competing")
    df["adapt_class"] = np.where((df["rho_vpc_adapt"] >= 0.95) & (df["por_adapt_%"] <= 5.0), "redundant", "competing")
    print("\n=== Monte Carlo robustness of the screen (N=%d draws) ===" % len(df))
    for lab in stats:
        col = df[lab].dropna()
        print(f"  {lab:18s}: median {col.median():+.4f}  [P5 {col.quantile(.05):+.4f}, "
              f"P95 {col.quantile(.95):+.4f}]  (min {col.min():+.4f}, max {col.max():+.4f})")
    print(f"  carbon removed in {(df.carbon_class=='redundant').mean()*100:.1f}% of draws; "
          f"adaptation retained in {(df.adapt_class=='competing').mean()*100:.1f}%; solver status {OPT.SOLVE_LOG}")
    df = df.sort_values("draw").reset_index(drop=True)
    df.to_csv(C.OUT_DIR / "montecarlo_results.csv", index=False)
    part.unlink(missing_ok=True)
    print(f"  saved -> {C.OUT_DIR / 'montecarlo_results.csv'}")

    # Figure fig_robustness is rendered in figures.py from montecarlo_results.csv, in the shared style.

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else N
    main()
