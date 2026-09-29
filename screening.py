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
screening.py  -  objective-screening (orthogonality) analysis
=================================================================
Implements the framework's objective screen and writes its results.

For each candidate asset the marginal contribution of acting (rehabilitating)
rather than deferring is computed for cost and for carbon, and the marginal
hazard removed by a countermeasure for adaptation. Both pairs (cost-carbon,
cost-adaptation) are screened by the SAME rule, applied by the single
`classify()` function below, on the SAME kind of statistic: the Spearman
rank correlation of VALUE-PER-CAPITAL (VPC), the quantity that actually
drives a budget-constrained selection, not the raw (un-normalized) marginal
values. Using VPC for one pair and raw marginals for the other would make
the two screens answer different questions; classify() and the two rho_vpc_*
statistics below deliberately do not do that.

Reports:
  - rho_vpc_carbon : Spearman corr of cost-per-capital vs carbon-per-capital
  - rho_cc         : Spearman corr of the RAW (un-normalized) cost and carbon
                     marginals -- a descriptive statistic only (Fig. 4a);
                     not used for the redundancy classification
  - rho_vpc_adapt  : Spearman corr of cost-per-capital vs scour-removed-per-
                     capital (the adaptation-axis analogue of rho_vpc_carbon)
  - por_carbon     : price of redundancy (Definition 3) for carbon; well
                     posed because the carbon-optimal program's carbon is
                     strictly positive
  - adapt_prem90/100 : cost premium to secure 90%/100% protection. This is
                     an ANALOGOUS diagnostic for adaptation, not a literal
                     instance of Definition 3: the scour-optimal program
                     achieves 0 residual (protected == total_scour) under
                     the program budget, so Definition 3's own ratio is
                     undefined (0/0) for adaptation. classify() therefore
                     decides the adaptation case on rho_vpc_adapt alone,
                     which is well posed and, empirically, decisive.
Writes outputs/screening_results.csv.
"""
from __future__ import annotations
import csv
import numpy as np
from scipy.stats import spearmanr
import config as C
import problem as P
import optimizer as OPT

CO, CA, CAP, SC = OPT.COST, OPT.CARBON, OPT.CAPITAL, OPT.SCOUR

RHO_CUTOFF   = 0.95    # pre-specified: below this, ranking is not similar enough to imply redundancy
DELTA_CUTOFF = 5.0     # pre-specified: percent; see Section 7.3 for the joint justification



def rank_corr(a, b):
    """Spearman correlation with algebraic ties preserved (audit correction): densities are
    rounded to ten decimals so values that are equal in exact arithmetic (e.g. zero-detour
    bridges) share a tied rank instead of being ordered by floating-point noise."""
    return float(spearmanr(np.round(np.asarray(a, float), 10), np.round(np.asarray(b, float), 10)).correlation)

def classify(rho, delta, rho_cutoff=RHO_CUTOFF, delta_cutoff=DELTA_CUTOFF):
    """One shared redundancy rule for every objective pair. Redundant only if
    BOTH the ranking is similar enough (rho >= rho_cutoff) AND the price of
    substituting the proxy is small enough (delta <= delta_cutoff, when
    delta is defined); otherwise the objective is treated as competing.
    `delta=None` (Definition 3's ratio undefined, e.g. a zero denominator)
    means the rho leg alone decides -- the classification never divides by
    zero to reach a verdict."""
    if rho is None or rho < rho_cutoff:
        return "competing"
    if delta is not None and delta > delta_cutoff:
        return "competing"
    return "redundant"


def run(prob=None, save=True):
    prob = prob or P.build()
    V, B, tot, n = prob.V, prob.budget_M, prob.total_scour, len(prob.V)

    dcost = np.array([V[i][0][CO] - V[i][1][CO] for i in range(n)])   # cost saved, rehab vs defer
    dcarb = np.array([V[i][0][CA] - V[i][1][CA] for i in range(n)])   # carbon saved, rehab vs defer
    cap1  = np.array([V[i][1][CAP] for i in range(n)])
    scrm  = np.array([V[i][2][SC] - V[i][0][SC] for i in range(n)])   # scour removed by countermeasure
    m = cap1 > 0

    vpc_cost  = dcost[m] / cap1[m]
    vpc_carb  = dcarb[m] / cap1[m]
    vpc_scour = scrm[m] / cap1[m]     # same per-bridge denominator as cost/carbon, for a like-for-like ranking

    rho_vpc_carbon = rank_corr(vpc_cost, vpc_carb)
    rho_cc         = rank_corr(dcost, dcarb)             # descriptive only (Fig. 4a)
    rho_vpc_adapt  = rank_corr(vpc_cost, vpc_scour)

    ch_co, co = OPT.cost_optimal(V, B)                 # ties -> minimum emissions
    # Expected NHS structurally deficient deck share of the cost-optimal program, per year of the
    # condition window (constraint rows: coef . y <= limit * total - outside).
    nhs_total = sum(b.deck_m2 for b in prob.full if b.is_highway)
    nhs_share = [100.0 * (C.NHS_SD_LIMIT - (rhs - sum(coef[i][ch_co[i]] for i in range(len(coef)))) / nhs_total)
                 for coef, rhs in getattr(V, "extra_rows", [])] if nhs_total else []
    if nhs_share:
        print("  NHS expected deficient deck share, cost-optimal program, years 1..%d: %s (limit %.0f%%)"
              % (len(nhs_share), ", ".join(f"{x:.2f}%" for x in nhs_share), 100 * C.NHS_SD_LIMIT))
    ch_cmin, cmin = OPT.carbon_optimal(V, B)           # ties -> minimum cost
    ch_amax, amax = OPT.adaptation_optimal(V, B)       # ties -> minimum cost
    # price of redundancy, delta(Y|X): loss on candidate Y from implementing the X-optimal program
    por_carbon = (co["carbon"] - cmin["carbon"]) / cmin["carbon"] * 100.0          # Y minimized
    abs_carbon_t = co["carbon"] - cmin["carbon"]
    e_defer = sum(V[i][0][CA] for i in range(n))                                   # emissions of deferring all
    forgone_pct = abs_carbon_t / (e_defer - cmin["carbon"]) * 100.0                # share of achievable saving
    por_adapt = ((amax["protected"] - co["protected"]) / amax["protected"] * 100.0  # Y maximized
                 if amax["protected"] > 0 else None)
    agree_carbon = float(np.mean([ch_co[i] == ch_cmin[i] for i in range(n)])) * 100.0

    r90 = OPT.min_cost_with_protection(V, 0.90 * tot, B)
    r100 = OPT.min_cost_with_protection(V, tot, B)
    prem90 = (r90[1]["cost"] - co["cost"]) / co["cost"] * 100.0
    prem100 = (r100[1]["cost"] - co["cost"]) / co["cost"] * 100.0 if r100 else float("inf")

    carbon_class = classify(rho_vpc_carbon, por_carbon)
    adapt_class  = classify(rho_vpc_adapt, por_adapt)

    print("OBJECTIVE SCREEN  (one shared rule: redundant iff rho>=%.2f AND delta<=%.1f%%; "
          "delta undefined -> rho alone decides)" % (RHO_CUTOFF, DELTA_CUTOFF))
    print(f"  rho_vpc(cost, carbon)   = {rho_vpc_carbon:+.4f}   delta={por_carbon:.3f}%   -> {carbon_class.upper()}")
    print(f"  rho_S(cost, carbon) RAW = {rho_cc:+.4f}   (descriptive; Fig. 4a; not used for classification)")
    print(f"  rho_vpc(cost, adapt)    = {rho_vpc_adapt:+.4f}   delta={por_adapt:.1f}%   -> {adapt_class.upper()}")
    print(f"  carbon: absolute loss {abs_carbon_t:.1f} t; share of achievable saving forgone {forgone_pct:.4f}%; "
          f"same option on {agree_carbon:.1f}% of assets")
    print(f"  cost-optimal program  : ${co['cost']:.0f}M  {co['carbon']/1e3:.1f}k tCO2e  protect {co['protected']:.2f} of {tot:.2f}")
    print(f"  carbon-optimal program: ${cmin['cost']:.0f}M  {cmin['carbon']/1e3:.1f}k tCO2e")
    print(f"  adaptation premium (descriptive, analogous to Def. 3, not identical): "
          f"+{prem90:.2f}% at 90% protection, +{prem100:.2f}% at 100%")

    if save:
        out = C.OUT_DIR / "screening_results.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in [("rho_vpc_carbon", rho_vpc_carbon), ("rho_cc_raw", rho_cc),
                         ("rho_vpc_adapt", rho_vpc_adapt),
                         ("cost_opt_cost_M", co["cost"]), ("cost_opt_carbon_k", co["carbon"] / 1e3),
                         ("carbon_opt_carbon_k", cmin["carbon"] / 1e3),
                         ("por_carbon_pct", por_carbon), ("carbon_abs_loss_t", abs_carbon_t),
                         ("carbon_saving_forgone_pct", forgone_pct), ("defer_all_carbon_k", e_defer / 1e3),
                         ("carbon_same_option_pct", agree_carbon),
                         ("adapt_max_protected", amax["protected"]), ("por_adapt_pct", por_adapt),
                         ("adapt_prem90_pct", prem90), ("adapt_prem100_pct", prem100)] + [(f"nhs_sd_share_y{t + 1}_pct", x) for t, x in enumerate(nhs_share)]:
                w.writerow([k, round(float(v), 4)])
            w.writerow(["carbon_classification", carbon_class])
            w.writerow(["adaptation_classification", adapt_class])
        print(f"  saved -> {out}")
    return dict(rho_vpc=rho_vpc_carbon, rho_cc=rho_cc, rho_cs=rho_vpc_adapt, por_carbon=por_carbon,
               por_adapt=por_adapt, abs_carbon_t=abs_carbon_t, forgone_pct=forgone_pct,
               prem90=prem90, prem100=prem100, cost_opt=co, carbon_opt=cmin,
               carbon_class=carbon_class, adapt_class=adapt_class)


if __name__ == "__main__":
    run()
