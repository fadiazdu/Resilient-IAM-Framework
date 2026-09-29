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
solver_study.py  -  verification of the scalable solver (NSGA-II) against the exact method
==========================================================================================
Addresses solver verification: algorithm configuration, convergence behavior, run-to-run
variability, runtime, and scalability against the exact epsilon-constraint method.

The bi-objective problem is (minimize cost, minimize residual scour) over the four options
per asset under the capital budget. The exact Pareto front (cost vs protection) is computed
by epsilon-constraint and used as the reference for generational distance (GD) and inverted
generational distance (IGD). The genetic algorithm uses a cost-effective knapsack repair,
seeded initialization, fast non-dominated sorting, and crowding-distance selection.

Outputs: outputs/solver_study.csv and figures/fig_solver_verification.{png,pdf}
         (two panels: convergence and scalability). Fixed seeds; fully reproducible.
"""
from __future__ import annotations
import time, csv, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import problem as P, optimizer as OPT

CO, CA, CAP, SC = OPT.COST, OPT.CARBON, OPT.CAPITAL, OPT.SCOUR
POP, GENS, N_SEEDS = 120, 120, 8

# ---------- problem matrices ----------
def matrices(V, n):
    cost = np.array([[V[i][g][CO]  for g in range(4)] for i in range(n)])
    cap  = np.array([[V[i][g][CAP] for g in range(4)] for i in range(n)])
    sc   = np.array([[V[i][g][SC]  for g in range(4)] for i in range(n)])
    return cost, cap, sc

# ---------- exact reference front (cost vs ACHIEVED protection) ----------
def exact_front(V, B, tot, npts=21):
    """Each target fraction is solved to a real MIP and its solver status is
    checked; only proven-optimal solutions are kept (min_cost_with_protection
    already discards anything CBC does not report as Optimal). The point
    plotted is the ACHIEVED protected fraction the discrete solution reaches,
    not the requested target -- these can differ because scour severities are
    discrete per-asset quantities, so an arbitrary continuous target is not
    generally hit exactly."""
    pts = []
    for f in np.linspace(0, 1, npts):
        r = OPT.min_cost_with_protection(V, f * tot, B, lex=False)
        if r:
            achieved = r[1]["protected"] / tot
            pts.append((achieved, r[1]["cost"]))
    return np.array(pts)   # (achieved_protected_frac, cost) -- both from proven-optimal solves

# ---------- NSGA-II with cost-effective repair ----------
def make_ga(cost, cap, sc, B, tot):
    n = len(cost); AR = np.arange(n)
    def repair(ind):
        used = cap[AR, ind].sum()
        while used > B:
            act = np.where(ind > 0)[0]
            if len(act) == 0: break
            dcap = cap[act, ind[act]] - cap[act, ind[act] - 1]
            dcost = cost[act, ind[act] - 1] - cost[act, ind[act]]
            ratio = np.where(dcap > 0, dcost / np.maximum(dcap, 1e-9), 1e18)
            i = act[np.argmin(ratio)]
            used -= (cap[i, ind[i]] - cap[i, ind[i] - 1]); ind[i] -= 1
        return ind
    def objs(ind): return np.array([cost[AR, ind].sum(), tot - sc[AR, ind].sum()])
    def nd(F):
        k = np.ones(len(F), bool)
        for i in range(len(F)):
            if k[i]:
                dom = np.all(F <= F[i], 1) & np.any(F < F[i], 1); dom[i] = False
                if dom.any(): k[i] = False
        return np.where(k)[0]
    def crowd(F, idx):
        d = np.zeros(len(idx))
        for m in range(2):
            o = np.argsort(F[idx, m]); d[o[0]] = d[o[-1]] = 1e18
            rg = F[idx, m].max() - F[idx, m].min() or 1
            for a in range(1, len(idx) - 1):
                d[o[a]] += (F[idx[o[a + 1]], m] - F[idx[o[a - 1]], m]) / rg
        return d
    return repair, objs, nd, crowd

def normalize(ref, F):
    allp = np.vstack([ref, F]); span = allp.max(0) - allp.min(0); span[span == 0] = 1
    return (ref - allp.min(0)) / span, (F - allp.min(0)) / span
def gd_igd(ref_pf, ga_pf):
    E, G = normalize(ref_pf, ga_pf)
    dm = lambda A, Bb: np.sqrt(((A[:, None, :] - Bb[None, :, :]) ** 2).sum(2))
    return dm(G, E).min(1).mean(), dm(E, G).min(1).mean()

def run_ga(cost, cap, sc, B, tot, seed, ref=None, track=False):
    n = len(cost); AR = np.arange(n)
    repair, objs, nd, crowd = make_ga(cost, cap, sc, B, tot)
    rng = np.random.default_rng(seed)
    seeds = [repair(np.argmin(cost, 1).copy())]
    push = np.argmin(cost, 1).copy()
    for i in range(n): push[i] = max(range(4), key=lambda g: (sc[i, g], -cost[i, g]))
    seeds += [repair(push), np.zeros(n, int)]
    while len(seeds) < POP: seeds.append(repair(rng.integers(0, 4, n)))
    pop = np.array(seeds); F = np.array([objs(p) for p in pop])
    trace = []
    for gen in range(GENS):
        kids = []
        for _ in range(POP // 2):
            a, b = pop[rng.integers(len(pop))], pop[rng.integers(len(pop))]
            mk = rng.random(n) < 0.5
            for ch in (np.where(mk, a, b), np.where(mk, b, a)):
                ch = ch.copy(); mu = rng.random(n) < 0.04; ch[mu] = rng.integers(0, 4, mu.sum())
                kids.append(repair(ch))
        allP = np.vstack([pop, np.array(kids)]); allF = np.vstack([F, np.array([objs(k) for k in kids])])
        sel = []; rem = np.arange(len(allP))
        while len(sel) < POP and len(rem):
            fr = rem[nd(allF[rem])]
            if len(sel) + len(fr) <= POP: sel.extend(fr.tolist())
            else:
                c = crowd(allF, fr); sel.extend(fr[np.argsort(-c)][:POP - len(sel)].tolist())
            rem = np.setdiff1d(rem, fr)
        sel = np.array(sel); pop, F = allP[sel], allF[sel]
        if track and ref is not None:
            gaF = F[nd(F)]; pf = np.column_stack([(tot - gaF[:, 1]) / tot, gaF[:, 0]])
            trace.append(gd_igd(ref, pf)[1])   # IGD per generation
    gaF = F[nd(F)]; pf = np.column_stack([(tot - gaF[:, 1]) / tot, gaF[:, 0]])
    return pf, trace

# ---------- scaled instance (replicate with small noise; budget scales per asset) ----------
def scaled_instance(V0, B0, tot0, factor, rng):
    n0 = len(V0)
    if factor <= 1.0:
        idx = rng.choice(n0, int(n0 * factor), replace=False)
        V = [V0[i] for i in idx]
    else:
        V = list(V0)
        for _ in range(int(factor) - 1):
            for i in range(n0):
                j = (rng.uniform(0.9, 1.1, 4))
                V.append([[V0[i][g][k] * (j[g] if k in (CO, CA, CAP) else 1.0) for k in range(4)] for g in range(4)])
    cost, cap, sc = matrices(V, len(V))
    return V, cost, cap, sc, B0 * (len(V) / n0), sc[:, 2:].max(1).sum() if len(V) else tot0

def main(n_seeds=N_SEEDS, do_scal=True):
    prob = P.build(); V0, B0, tot0 = prob.V, prob.budget_M, prob.total_scour
    cost0, cap0, sc0 = matrices(V0, len(V0))
    ref = exact_front(V0, B0, tot0)
    exact_min = ref[:, 1].min()

    # ---- multi-seed runs: variability + convergence ----
    print(f"NSGA-II config: pop={POP}, generations={GENS}, seeds={n_seeds}, "
          f"crossover=uniform, mutation=0.04, repair=cost-effective knapsack")
    traces, finals = [], []
    for s in range(n_seeds):
        t0 = time.time()
        pf, tr = run_ga(cost0, cap0, sc0, B0, tot0, seed=s, ref=ref, track=True)
        rt = time.time() - t0
        gd, igd = gd_igd(ref, pf)
        gap = (pf[:, 1].min() - exact_min) / exact_min * 100
        finals.append((gd, igd, gap, rt)); traces.append(tr)
        print(f"  seed {s}: GD {gd:.4f}  IGD {igd:.4f}  min-cost gap {gap:+.2f}%  runtime {rt:.1f}s")
    F = np.array(finals)
    print(f"\n  ACROSS {n_seeds} SEEDS (mean +/- sd):")
    for j, lab in enumerate(["GD", "IGD", "min-cost gap %", "runtime s"]):
        print(f"    {lab:16s}: {F[:,j].mean():.4f} +/- {F[:,j].std():.4f}")

    # convergence trace (plotted in the combined verification figure below)
    T = np.array([t[:min(len(x) for x in traces)] for t in traces])
    g = np.arange(1, T.shape[1] + 1)

    # ---- scalability: exact vs GA runtime vs N, on the SAME task ----
    # Both methods are asked to produce a front of the same number of
    # trade-off points (SCAL_NPTS), not a single scalarized solve for the
    # exact method against a full-front search for the GA: that earlier
    # comparison timed two different-sized tasks against each other.
    SCAL_NPTS = 9
    scal = []; S = None
    if do_scal:
        rng = np.random.default_rng(0)
        for fac in [0.3, 0.6, 1.0, 2.0, 3.0, 4.0]:
            V, cst, cp, scc, B, tt = scaled_instance(V0, B0, tot0, fac, rng)
            n = len(V)
            t0 = time.time()
            for frac in np.linspace(0, 1, SCAL_NPTS):
                OPT.min_cost_with_protection(V, frac * tt, B, lex=False)
            t_exact = time.time() - t0
            t0 = time.time(); run_ga(cst, cp, scc, B, tt, seed=0, track=False); t_ga = time.time() - t0
            scal.append((n, t_exact, t_ga))
            print(f"  N={n:5d}: exact front ({SCAL_NPTS} pts) {t_exact:6.2f}s | GA (1 front) {t_ga:6.2f}s")
        S = np.array(scal)

    # ---- traces saved for the supplementary figure (outputs only; no effect on the computation) ----
    import pandas as _pd
    _pd.DataFrame(dict(generation=g, igd_mean=T.mean(0), igd_min=T.min(0), igd_max=T.max(0))).to_csv(C.OUT_DIR / "solver_convergence.csv", index=False)
    if S is not None:
        _pd.DataFrame(S, columns=["n_assets", "exact_front_s", "ga_front_s"]).to_csv(C.OUT_DIR / "solver_scalability.csv", index=False)

    # ---- combined solver-verification figure (convergence | scalability) ----
    C.apply_style()
    if S is not None:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(C.COL2_IN, 2.9))
    else:
        fig, a1 = plt.subplots(figsize=(C.COL1_IN * 1.45, C.COL1_IN)); a2 = None
    a1.plot(g, T.mean(0), color=C.PALETTE["primary"], lw=1.9, label="mean IGD")
    a1.fill_between(g, T.min(0), T.max(0), color=C.PALETTE["primary"], alpha=0.15,
                    label="min-max band", linewidth=0)
    a1.set_xlabel("Generation"); a1.set_ylabel("IGD to exact front (normalized)")
    a1.set_title("(a) Convergence to the exact front", loc="left"); a1.legend(loc="upper right")
    C.ygrid(a1)
    if a2 is not None:
        a2.plot(S[:, 0], S[:, 1], "o-", color=C.PALETTE["primary"], label=f"exact ({SCAL_NPTS}-pt front)")
        a2.plot(S[:, 0], S[:, 2], "s-", color=C.PALETTE["accent"], label="NSGA-II (full front)")
        a2.set_xlabel("Number of assets"); a2.set_ylabel("Solve time (s)")
        a2.set_title("(b) Scalability: exact vs genetic algorithm", loc="left"); a2.legend(loc="upper left")
        C.ygrid(a2)
    fig.tight_layout()
    C.savefig(fig, "fig_solver_verification")
    plt.close()

    with open(C.OUT_DIR / "solver_study.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for j, lab in enumerate(["GD_mean", "GD_sd", "IGD_mean", "IGD_sd", "gap_mean", "gap_sd", "rt_mean", "rt_sd"]):
            w.writerow([lab, [F[:,0].mean(),F[:,0].std(),F[:,1].mean(),F[:,1].std(),
                              F[:,2].mean(),F[:,2].std(),F[:,3].mean(),F[:,3].std()][j]])
        w.writerow([]); w.writerow(["N", "exact_s", "ga_s"])
        for r in scal: w.writerow(list(r))
    print(f"  saved -> {C.OUT_DIR}/solver_study.csv and {C.FIG_DIR}/fig_solver_*")

if __name__ == "__main__":
    main()
