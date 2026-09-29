# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
extended_analyses.py  -  extended analyses of the objective screen and the protection frontier
==============================================================================
Every function writes one CSV to outputs/ so each number reported in the
manuscript traces to a file.

  carbon_regret_grid    carbon price of redundancy over protection target x budget
                        (the domain D in which carbon is omitted while adaptation
                        is imposed)
  carbon_price_sweep    minimize cost + carbon price x emissions
  diversion_rules       alternative diverted-traffic schedules PHI
  cm_effectiveness      partial countermeasure effectiveness
  mapping_alignment     cost-adaptation alignment per scour mapping
  knee_composition      portfolio composition around the knee
  rule_comparison       alignment-only vs regret-only vs combined rules,
                        and alternative thresholds
"""
from __future__ import annotations
import csv
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C
import problem as P
import optimizer as OPT
import screening as S
import adaptation as A
import engine as M

CO, CA, CAP, SC = 0, 1, 2, 3


def _write(name, header, rows):
    path = C.OUT_DIR / name
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"  saved -> {path}")


def _rho_adapt(V):
    """Cost-adaptation priority alignment, built exactly as in screening.py."""
    n = len(V)
    cap1 = np.array([V[i][1][CAP] for i in range(n)]); m = cap1 > 0
    dcost = np.array([V[i][0][CO] - V[i][1][CO] for i in range(n)])[m] / cap1[m]
    scrm = np.array([V[i][2][SC] - V[i][0][SC] for i in range(n)])[m] / cap1[m]
    return float(S.rank_corr(dcost, scrm))


# ------------------------------------------------------------------ carbon_regret_grid
def carbon_regret_grid(prob=None, budget_mults=(0.5, 0.75, 1.0, 1.25, 1.5),
                       tau_fracs=(0.0, 0.25, 0.5, 0.75, 0.9, 1.0)):
    prob = prob or P.build()
    V, B0, tot, n = prob.V, prob.budget_M, prob.total_scour, len(prob.V)
    print("CARBON PRICE OF REDUNDANCY OVER PROTECTION TARGET x BUDGET")
    print(f"  {'B/B0':>5} {'tau%':>5} {'delta%':>8} {'abs_t':>9} {'same%':>6}")
    rows = []
    for bm in budget_mults:
        B = bm * B0
        for tf in tau_fracs:
            rc = OPT.min_cost_with_protection(V, tf * tot, B)
            re = OPT.carbon_optimal(V, B, min_protection=tf * tot)
            if rc is None or re is None:
                rows.append((bm, tf * 100, "infeasible", "", "")); print(f"  {bm:5.2f} {tf*100:5.0f}  infeasible"); continue
            (chc, sc), (che, se) = rc, re
            d = (sc["carbon"] - se["carbon"]) / se["carbon"] * 100
            same = np.mean([chc[i] == che[i] for i in range(n)]) * 100
            rows.append((bm, tf * 100, round(d, 6), round(sc["carbon"] - se["carbon"], 1), round(same, 1)))
            print(f"  {bm:5.2f} {tf*100:5.0f} {d:8.4f} {sc['carbon']-se['carbon']:9.1f} {same:6.1f}")
    _write("carbon_regret_grid.csv", ["budget_mult", "tau_pct", "delta_pct", "abs_loss_t", "same_option_pct"], rows)
    feas = [r[2] for r in rows if r[2] != "infeasible"]
    print(f"  max delta over feasible grid: {max(feas):.4f}%  ({len(feas)} feasible of {len(rows)} cells)")
    return rows


# ------------------------------------------------------------------ carbon_price_sweep
def carbon_price_sweep(prob=None, prices=(0, 50, 100, 200, 500, 1000)):
    prob = prob or P.build()
    V, B, n = prob.V, prob.budget_M, len(prob.V)
    ch0, s0 = OPT.cost_optimal(V, B)
    print("CARBON-PRICE SWEEP (minimize cost + price x emissions)")
    print(f"  {'$/t':>6} {'cost_M':>8} {'carbon_kt':>10} {'same%':>6}")
    rows = []
    for pr in prices:
        ch, s = OPT.priced_optimal(V, B, pr)
        same = np.mean([ch[i] == ch0[i] for i in range(n)]) * 100
        rows.append((pr, round(s["cost"], 2), round(s["carbon"] / 1e3, 3), round(same, 1)))
        print(f"  {pr:6d} {s['cost']:8.1f} {s['carbon']/1e3:10.3f} {same:6.1f}")
    _write("carbon_price_sweep.csv", ["price_usd_per_t", "cost_M", "carbon_kt", "same_option_pct"], rows)
    return rows


# ------------------------------------------------------------------ diversion_rules
import json as _json
_EV = _json.load(open(C.ROOT / "diversion_evidence.json"))
DIVERSION_RULES = {
    "evidence, pooled 2019-24": (_EV["pooled"]["closed"], _EV["pooled"]["posted"]),
    "evidence, 2024 only":      (_EV["y2024"]["closed"], _EV["y2024"]["posted"]),
    "high-diversion scenario":  (_EV["original"]["closed"], _EV["original"]["posted"]),
    "mild partial":             [1.0, 0.10, 0.02, 0.0, 0.0],
    "early diversion":          [1.0, 0.50, 0.15, 0.0, 0.0],
    "lowest-state-only":        [1.0, 0.0, 0.0, 0.0, 0.0],
}


def diversion_rules():
    print("DIVERTED-TRAFFIC RULES (PHI shared by user cost and use-phase emissions)")
    rows, phi0 = [], (list(M.PHI), list(M.PHI_POST))
    try:
        for name, phi in DIVERSION_RULES.items():
            M.set_diversion(*(phi if isinstance(phi, tuple) else (phi, None)))
            r = S.run(P.build(), save=False)
            rows.append((name, str(phi), round(r["rho_vpc"], 4), round(r["por_carbon"], 4),
                         r["carbon_class"], round(r["rho_cs"], 4), round(r["por_adapt"], 1), r["adapt_class"],
                         round(r["cost_opt"]["cost"], 1), round(r["prem90"], 2)))
    finally:
        M.set_diversion(*phi0)
    print(f"  {'rule':16s} {'rho_c':>7} {'delta_c%':>9} {'class_c':>10} {'rho_a':>7} {'delta_a%':>8} {'cost_M':>8} {'prem90':>7}")
    for r in rows:
        print(f"  {r[0]:16s} {r[2]:7.4f} {r[3]:9.4f} {r[4]:>10} {r[5]:7.4f} {r[6]:8.1f} {r[8]:8.1f} {r[9]:7.2f}")
    _write("diversion_rules.csv", ["rule", "phi_states0to4", "rho_cost_carbon", "delta_carbon_pct", "carbon_class",
                                    "rho_cost_adapt", "delta_adapt_pct", "adapt_class", "cost_opt_M", "prem90_pct"], rows)
    return rows


# ------------------------------------------------------------------ cm_effectiveness
def cm_effectiveness(effects=(1.0, 0.75, 0.5), target_share=0.45):
    """Partial effectiveness scales every a_(i,g) by the same factor, so the
    ranking-based alignment is unchanged; what changes is how much coverage a
    fixed exposure target requires. Reported at a target every case can reach."""
    print(f"COUNTERMEASURE EFFECTIVENESS (target: {target_share:.0%} of total weighted exposure)")
    rows, e0 = [], A.CM_EFFECT
    try:
        for e in effects:
            A.CM_EFFECT = e
            prob = P.build(); V, B, tot = prob.V, prob.budget_M, prob.total_scour
            _, co = OPT.cost_optimal(V, B)
            _, s = OPT.min_cost_with_protection(V, target_share * tot, B)
            prem = (s["cost"] - co["cost"]) / co["cost"] * 100
            rows.append((e, round(_rho_adapt(V), 4), round(target_share / e * 100, 1), round(prem, 2), s["n_cm"]))
            print(f"  effectiveness {e:.2f}: rho_adapt {rows[-1][1]:.4f}  coverage needed {rows[-1][2]:5.1f}%  "
                  f"premium {prem:5.2f}%  countermeasures {s['n_cm']}")
    finally:
        A.CM_EFFECT = e0
    _write("cm_effectiveness.csv", ["effectiveness", "rho_cost_adapt", "coverage_needed_pct", "premium_pct", "n_countermeasures"], rows)
    return rows


# ------------------------------------------------------------------ mapping_alignment
def mapping_alignment():
    import sensitivity as SE
    print("COST-ADAPTATION ALIGNMENT PER SCOUR MAPPING")
    rows = []
    for name, fn in SE.MAPS.items():
        V, tot, nc = SE.build(fn, 0.20, 0.10)
        B = P.build().budget_M
        _, co = OPT.cost_optimal(V, B); _, am = OPT.adaptation_optimal(V, B)
        da = (am["protected"] - co["protected"]) / am["protected"] * 100 if am["protected"] > 0 else float("nan")
        rows.append((name, nc, round(tot, 2), round(_rho_adapt(V), 4), round(da, 1)))
        print(f"  {name:13s} n={nc}  total={tot:6.2f}  rho_adapt={rows[-1][3]:+.4f}  delta_adapt={da:.1f}%")
    A.CM_FRACTION, A.EMB_CM_FRAC = 0.20, 0.10
    _write("mapping_alignment.csv", ["mapping", "n_candidates", "total_exposure", "rho_cost_adapt", "delta_adapt_pct"], rows)
    return rows


# ------------------------------------------------------------------ knee_composition
def knee_composition(prob=None, shares=(0.70, 0.75, 0.80)):
    """Which bridges receive countermeasures near the knee, relative to the cost-optimal program."""
    prob = prob or P.build()
    V, B, tot, n = prob.V, prob.budget_M, prob.total_scour, len(prob.V)
    ch0, _ = OPT.cost_optimal(V, B)
    print("PORTFOLIO COMPOSITION AROUND THE KNEE (options: 0 defer, 1 rehab, 2 cm, 3 rehab+cm)")
    rows = []
    for sh in shares:
        ch, s = OPT.min_cost_with_protection(V, sh * tot, B)
        cm_idx = [i for i in range(n) if ch[i] in (2, 3)]
        from_rehab = sum(1 for i in cm_idx if ch0[i] == 1)
        from_defer = sum(1 for i in cm_idx if ch0[i] == 0)
        rehab_dropped = sum(1 for i in range(n) if ch0[i] == 1 and ch[i] in (0, 2))
        rows.append((sh * 100, round(s["protected"] / tot * 100, 1), len(cm_idx), from_rehab, from_defer, rehab_dropped,
                     round(s["capital"], 1)))
        print(f"  target {sh:.0%}: achieved {rows[-1][1]}%  countermeasures {len(cm_idx)} "
              f"(on bridges rehabilitated in cost-optimal: {from_rehab}; deferred there: {from_defer})  "
              f"rehabilitations dropped: {rehab_dropped}  capital {s['capital']:.1f}M")
    _write("knee_composition.csv", ["target_pct", "achieved_pct", "n_countermeasures", "cm_on_costopt_rehab",
                                     "cm_on_costopt_defer", "rehabs_dropped", "capital_M"], rows)
    return rows


# ------------------------------------------------------------------ rule_comparison
def rule_comparison(rho_cut=0.95, eta=5.0):
    """Classify carbon under alignment-only, regret-only, and combined rules on the
    embodied sweep and the Monte Carlo draws; plus alternative thresholds at baseline."""
    print("DECISION-RULE COMPARISON")
    rows = []
    emb = pd.read_csv(C.OUT_DIR / "embodied_threshold.csv")
    for _, r in emb.iterrows():
        a, g = r["rho_vpc"] >= rho_cut, r["por_carbon_pct"] <= eta
        rows.append(("embodied", r["mult"], round(r["rho_vpc"], 4), round(r["por_carbon_pct"], 3),
                     "remove" if a else "retain", "remove" if g else "retain", "remove" if (a and g) else "retain"))
    mc = pd.read_csv(C.OUT_DIR / "montecarlo_results.csv")
    a = (mc["rho_vpc_carbon"] >= rho_cut); g = (mc["por_carbon_%"] <= eta)
    rows.append(("montecarlo", len(mc), "", "", f"remove in {a.mean()*100:.1f}%", f"remove in {g.mean()*100:.1f}%",
                 f"remove in {(a & g).mean()*100:.1f}%"))
    disagree = [r for r in rows[:-1] if r[4] != r[5]]
    print(f"  embodied sweep: alignment-only and regret-only rules disagree at multipliers "
          f"{[r[1] for r in disagree]}")
    print(f"  Monte Carlo: {rows[-1][4:]} (alignment-only, regret-only, combined)")
    _write("rule_comparison.csv", ["case", "multiplier_or_draws", "rho", "delta_pct", "alignment_only",
                                    "regret_only", "combined"], rows)
    base = pd.read_csv(C.OUT_DIR / "screening_results.csv").set_index("metric")["value"]
    rho0, d0 = float(base["rho_vpc_carbon"]), float(base["por_carbon_pct"])
    trows = []
    for rc in (0.90, 0.95, 0.975, 0.99):
        for e in (1.0, 2.0, 5.0, 10.0):
            trows.append((rc, e, "remove" if (rho0 >= rc and d0 <= e) else
                          ("retain (alignment grounds)" if d0 <= e else "retain (material loss)")))
    _write("threshold_grid.csv", ["rho_cutoff", "eta_pct", "carbon_decision_at_baseline"], trows)
    print(f"  baseline carbon (rho={rho0:.4f}, delta={d0:.4f}%): removed for rho cutoff <= 0.95; "
          f"retained on alignment grounds at 0.975 and 0.99 for every eta tested")
    return rows, trows


# ------------------------------------------------------------------ shared knee math
def knee_points(f_arr, p_arr):
    """Knee indices among solved frontier points (both axes min-max normalized):
    maximum distance from the chord joining the endpoints, and maximum curvature
    from finite-difference derivatives (no smoothing). Used by run_all.knee_curve
    and by knee_stability, so both apply identical math."""
    fn = (f_arr - f_arr.min()) / (f_arr.max() - f_arr.min())
    pn = (p_arr - p_arr.min()) / (p_arr.max() - p_arr.min() + 1e-12)
    x1, y1, x2, y2 = fn[0], pn[0], fn[-1], pn[-1]
    chord_d = np.abs((y2 - y1) * fn - (x2 - x1) * pn + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)
    d1 = np.gradient(pn, fn); d2 = np.gradient(d1, fn)
    curvature = np.abs(d2) / (1 + d1 ** 2) ** 1.5
    return int(np.argmax(chord_d)), int(np.argmax(curvature))


# ------------------------------------------------------------------ frontier_regret
def frontier_regret(prob=None, npts=41):
    """Carbon price of redundancy at every protection target of the reported frontier."""
    prob = prob or P.build()
    V, B, tot, n = prob.V, prob.budget_M, prob.total_scour, len(prob.V)
    st0 = dict(OPT.SOLVE_LOG)
    print(f"CARBON PRICE OF REDUNDANCY AT ALL {npts} FRONTIER TARGETS (budget {B:.1f} M$)")
    rows = []
    for f in np.linspace(0, 1, npts):
        rc = OPT.min_cost_with_protection(V, f * tot, B)
        re = OPT.carbon_optimal(V, B, min_protection=f * tot)
        (chc, sc), (che, se) = rc, re
        d = (sc["carbon"] - se["carbon"]) / se["carbon"] * 100
        rows.append((round(f * 100, 2), round(sc["protected"] / tot * 100, 2), round(sc["capital"], 2),
                     round(sc["cost"], 2), round(sc["carbon"], 1), round(se["carbon"], 1),
                     round(sc["carbon"] - se["carbon"], 1), round(d, 6)))
    new = {k: v - st0.get(k, 0) for k, v in OPT.SOLVE_LOG.items() if v - st0.get(k, 0)}
    d = [r[-1] for r in rows]
    print(f"  max {max(d):.4f}%  median {np.median(d):.4f}%  max abs {max(r[6] for r in rows):.1f} t  solver {new}")
    _write("frontier_regret.csv", ["required_pct", "achieved_pct", "capital_M", "cost_M", "carbon_costopt_t",
                                   "carbon_carbonopt_t", "abs_loss_t", "delta_pct"], rows)
    return rows


# ------------------------------------------------------------------ deterioration_rates
def deterioration_rates(factors=(1.0, 1.5, 2.0)):
    """Faster-deterioration scenarios: every fitted annual rate lambda_s is multiplied by the
    factor and the annual matrix recomputed as expm(Q). Scaling the generator keeps the
    multi-step transitions of the inspection-interval model; factor 1.0 reproduces the baseline."""
    import json
    from scipy.linalg import expm
    import build_matrix as BM
    rates = np.array(json.load(open(C.OUT_DIR / "deterioration_fit.json"))["rates"])
    print("DETERIORATION-RATE SCENARIOS (fitted annual rates scaled)")
    base = np.array(M.REAL_T, float); rows = []
    try:
        for fct in factors:
            M.REAL_T = expm(BM.generator(rates * fct))
            r = S.run(P.build(), save=False)
            rows.append((fct, round(r["rho_vpc"], 4), round(r["por_carbon"], 4), r["carbon_class"],
                         round(r["rho_cs"], 4), round(r["por_adapt"], 1), r["adapt_class"],
                         round(r["cost_opt"]["cost"], 1), round(r["prem90"], 2)))
    finally:
        M.REAL_T = base
    for r in rows:
        print(f"  x{r[0]:.1f}: rho_c {r[1]:.4f} delta_c {r[2]:.4f}% {r[3]} | rho_a {r[4]:.4f} {r[6]} | cost {r[7]:.0f}M prem90 {r[8]:.2f}%")
    _write("deterioration_rates.csv", ["factor", "rho_cost_carbon", "delta_carbon_pct", "carbon_class",
                                       "rho_cost_adapt", "delta_adapt_pct", "adapt_class", "cost_opt_M", "prem90_pct"], rows)
    return rows


# ------------------------------------------------------------------ knee_stability
def knee_stability(prob=None, grids=(21, 41, 81)):
    """Knee location on progressively finer grids (single-stage solves: frontier cost
    does not depend on tie-breaking)."""
    prob = prob or P.build()
    V, B, tot = prob.V, prob.budget_M, prob.total_scour
    _, co = OPT.cost_optimal(V, B); C0 = co["cost"]
    print("KNEE STABILITY ACROSS GRID RESOLUTIONS")
    rows = []
    for npts in grids:
        f_list, p_list = [], []
        for f in np.linspace(0, 1, npts):
            r = OPT.min_cost_with_protection(V, f * tot, B, lex=False)
            f_list.append(r[1]["protected"] / tot); p_list.append((r[1]["cost"] - C0) / C0 * 100)
        fa, pa = np.array(f_list), np.array(p_list)
        ic, iv = knee_points(fa, pa)
        mono = bool(np.all(np.diff(pa[np.argsort(fa)]) >= -1e-9))
        rows.append((npts, round(fa[iv] * 100, 2), round(pa[iv], 2), round(fa[ic] * 100, 2), round(pa[ic], 2), mono))
        print(f"  {npts:3d} targets: curvature knee {fa[iv]*100:.1f}% (+{pa[iv]:.2f}%), chord knee {fa[ic]*100:.1f}% "
              f"(+{pa[ic]:.2f}%), cost nondecreasing in achieved protection: {mono}")
    _write("knee_stability.csv", ["targets", "curv_knee_pct", "curv_premium_pct", "chord_knee_pct",
                                   "chord_premium_pct", "nondominated"], rows)
    return rows


def run_all(prob=None):
    prob = prob or P.build()
    out = {}
    for label, fn in [("grid", lambda: carbon_regret_grid(prob)), ("price", lambda: carbon_price_sweep(prob)),
                      ("diversion", diversion_rules), ("effect", cm_effectiveness),
                      ("mapping", mapping_alignment), ("knee", lambda: knee_composition(prob)),
                      ("rules", rule_comparison), ("frontier", lambda: frontier_regret(prob)),
                      ("deterioration", deterioration_rates), ("knee_stability", lambda: knee_stability(prob))]:
        print(); out[label] = fn()
    print(f"\nSolver status counts across all solves so far: {OPT.SOLVE_LOG}")
    return out


if __name__ == "__main__":
    run_all()
