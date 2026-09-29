# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
test_budgeted_nonconvex.py
==================================================================
The realistic case: a hard program budget turns the problem into a
multi-objective multiple-choice knapsack. We test whether THIS
constrained front is non-convex, i.e. whether unsupported Pareto
solutions appear that a weighted-sum sweep can never find. That is the
condition under which NSGA-II is genuinely necessary.

Also fixes the degenerate risk objective: risk is now traffic exposure
to substandard condition (condition-sensitive across all states, not
the near-zero failure-hazard term).

Objectives (minimise), constraint capital <= budget:
  COST   ($M)   : capital + discounted O&M + discounted user delay
  RISK   (Mveh) : sum_t AADT * deficiency_weight(state)   [NOT monetised]
  CARBON (tCO2e): embodied + operational detour emissions

Decision per bridge: defer / conventional rehab / low-carbon rehab.
Imports only bridge_iam_real + numpy + scipy.
==================================================================
"""
from __future__ import annotations
import numpy as np
import config as C
from scipy.optimize import linprog
import engine as M

EF_CAR  = (8887.0 / 22.2) / 1000 / 1.60934   # kgCO2/km; EPA-420-F-23-014 (Jun 2023),
                                              # "Tailpipe GHG Emissions from a Typical Passenger
                                              # Vehicle": 8,887 gCO2/gal gasoline / 22.2 mpg
                                              # fleet-average fuel economy = ~400 gCO2/mile
EF_TRUCK = 1.00
EMB_REHAB_M2 = 340.0
LOWC_EMB, LOWC_COST = 0.65, 1.18
TOTAL_BUDGET_M = C.BUDGET_M                      # single source of truth: config.py
# deficiency exposure weight by state (condition-sensitive, non-monetised)
DEFIC = {4: 0.0, 3: 0.1, 2: 0.3, 1: 0.7, 0: 1.0}


def operational_carbon_t(b, dists):
    """Use-phase carbon (tCO2e), exact expectation over the state
    distribution at each year (dists[y] is a length-5 probability vector)."""
    ts = M.truck_share(b)                              # the bridge's reported truck share
    vkm = b.aadt * 365 * getattr(b, "detour_km", M.DETOUR_KM)
    ef_mix = (1 - ts) * EF_CAR + ts * EF_TRUCK         # closure: all traffic
    ef_post = ts * EF_TRUCK                            # load posting: trucks only
    tier = (np.array(M.PHI, dtype=float) * ef_mix + np.array(M.PHI_POST, dtype=float) * ef_post) * vkm / 1000.0
    return sum(float(dists[y] @ tier) for y in range(1, M.HORIZON + 1))


def cost_no_failure_M(b, dists):
    maint = np.array([M.MAINT[s] for s in range(5)])
    omr = 0.0
    for y in range(1, M.HORIZON + 1):
        d = (1 + M.DR) ** (-y)
        omr += d * (b.deck_m2 * float(dists[y] @ maint) + (M.INSPECTION if y % 2 == 0 else 0))
    return omr / 1e6 + M._pv_user_cost(b, dists)


def risk_exposure(b, dists):
    defic = np.array([DEFIC[s] for s in range(5)])
    return sum(float(dists[y] @ defic) * b.aadt for y in range(1, M.HORIZON + 1)) / 1e6


def option_data(b):
    rate = M.cost_major_nhs_m2() if b.is_highway else M.cost_major_nonhs_m2()
    cap = b.deck_m2 * rate / 1e6
    emb = b.deck_m2 * EMB_REHAB_M2 / 1000.0
    dd, _ = M._mc_exact(b, -1, "none")
    di, _ = M._mc_exact(b, 1, "major")
    # (cost, risk, carbon, capital) per option
    return [
        (cost_no_failure_M(b, dd), risk_exposure(b, dd), operational_carbon_t(b, dd), 0.0),
        (cap + cost_no_failure_M(b, di), risk_exposure(b, di), emb + operational_carbon_t(b, di), cap),
        (cap * LOWC_COST + cost_no_failure_M(b, di), risk_exposure(b, di),
         emb * LOWC_EMB + operational_carbon_t(b, di), cap * LOWC_COST),
    ]


# ---- objective + constraint evaluation (fast, precomputed) ------
def evaluate(genes, V):
    f = np.zeros(3); cap = 0.0
    for i, g in enumerate(genes):
        f += V[i][g][:3]; cap += V[i][g][3]
    return f, cap


def repair(genes, V, budget):
    """Reduce capital to feasibility: step the highest-capital active option
    down one notch (low->conv->defer) until within budget."""
    g = genes.copy()
    cap = sum(V[i][g[i]][3] for i in range(len(g)))
    while cap > budget:
        active = [i for i in range(len(g)) if g[i] > 0]
        if not active:
            break
        i = max(active, key=lambda i: V[i][g[i]][3])
        saved = V[i][g[i]][3] - V[i][g[i] - 1][3]
        g[i] -= 1; cap -= saved
    return g


# ---- compact NSGA-II for genes in {0,1,2} -----------------------
def dom(a, b):
    return np.all(a <= b) and np.any(a < b)


def nondominated_idx(F):
    n = len(F); keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and keep[j] and dom(F[j], F[i]):
                keep[i] = False; break
    return np.where(keep)[0]


def crowding(F):
    n = len(F); d = np.zeros(n)
    for m in range(F.shape[1]):
        o = np.argsort(F[:, m]); d[o[0]] = d[o[-1]] = 1e9
        rng = F[o[-1], m] - F[o[0], m]
        if rng > 0:
            for k in range(1, n - 1):
                d[o[k]] += (F[o[k + 1], m] - F[o[k - 1], m]) / rng
    return d


def nsga2(V, budget, pop=160, gen=160, seed=42):
    rng = np.random.default_rng(seed); N = len(V)
    P = np.array([repair(rng.integers(0, 3, N), V, budget) for _ in range(pop)])

    def feval(P):
        return np.array([evaluate(ind, V)[0] for ind in P])

    F = feval(P)
    for _ in range(gen):
        # offspring: uniform crossover + mutation, then repair
        kids = []
        for _ in range(pop // 2):
            a, b = P[rng.integers(pop)], P[rng.integers(pop)]
            mask = rng.random(N) < 0.5
            for child in (np.where(mask, a, b), np.where(mask, b, a)):
                mut = rng.random(N) < 0.05
                child = child.copy(); child[mut] = rng.integers(0, 3, mut.sum())
                kids.append(repair(child, V, budget))
        K = np.array(kids); FK = feval(K)
        allP = np.vstack([P, K]); allF = np.vstack([F, FK])
        # elitist non-dominated + crowding truncation
        chosen = []
        remaining = np.arange(len(allP))
        while len(chosen) < pop and len(remaining) > 0:
            nd = remaining[nondominated_idx(allF[remaining])]
            if len(chosen) + len(nd) <= pop:
                chosen.extend(nd.tolist())
            else:
                cd = crowding(allF[nd]); order = np.argsort(-cd)
                chosen.extend(nd[order[:pop - len(chosen)]].tolist())
            remaining = np.array([r for r in remaining if r not in set(nd.tolist())])
        P = allP[chosen]; F = allF[chosen]
    nd = nondominated_idx(F)
    return P[nd], F[nd]


def is_supported(fp, F, tol=1e-9):
    """Exact: is there lambda>=0, sum=1, with fp minimising lambda.f over F?"""
    A_ub = [fp - fq for fq in F]
    res = linprog(c=[0, 0, 0], A_ub=A_ub, b_ub=[tol] * len(F),
                  A_eq=[[1, 1, 1]], b_eq=[1.0], bounds=[(1e-6, 1)] * 3,
                  method="highs")
    return res.success


if __name__ == "__main__":
    cands, _full, _T = M.build_portfolio_real()
    print(f"  Candidates: {len(cands)}  | budget ${TOTAL_BUDGET_M:.0f}M (capital)")
    V = [option_data(b) for b in cands]

    P, F = nsga2(V, TOTAL_BUDGET_M)
    print(f"  NSGA-II non-dominated, budget-feasible solutions: {len(F)}")

    # conflict snapshot: ranges across the front
    print("=" * 70)
    print("BUDGET-CONSTRAINED FRONT (ranges across Pareto solutions)")
    print(f"   cost $M     : {F[:,0].min():8.0f} .. {F[:,0].max():8.0f}")
    print(f"   risk Mveh   : {F[:,1].min():8.2f} .. {F[:,1].max():8.2f}")
    print(f"   carbon tCO2e: {F[:,2].min():8.0f} .. {F[:,2].max():8.0f}")

    # pairwise rank correlations (do they conflict?)
    def rho(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return np.corrcoef(ra, rb)[0, 1]
    print(f"   rank corr  cost~risk {rho(F[:,0],F[:,1]):+.2f} | "
          f"cost~carbon {rho(F[:,0],F[:,2]):+.2f} | risk~carbon {rho(F[:,1],F[:,2]):+.2f}")
    print("   (negative = conflict)")

    # ---- exact non-convexity test on the constrained front -----
    span = F.max(0) - F.min(0); span[span == 0] = 1.0
    Fn = (F - F.min(0)) / span
    unsup = sum(0 if is_supported(Fn[i], Fn) else 1 for i in range(len(Fn)))
    print("-" * 70)
    print("EXACT NON-CONVEXITY TEST ON THE BUDGETED FRONT")
    print(f"   Pareto solutions that are UNSUPPORTED (no weighting can reach,")
    print(f"   so a weighted-sum sweep would miss them): {unsup} / {len(Fn)} "
          f"({100*unsup/len(Fn):.0f}%)")
    if unsup > 0:
        print("   -> The constrained front is NON-CONVEX. A weighted-sum sweep and")
        print("      any single ranking provably miss part of the Pareto set, so a")
        print("      true Pareto method (NSGA-II) is genuinely necessary here.")
    else:
        print("   -> Still convex; weighted-sum would suffice even under budget.")
