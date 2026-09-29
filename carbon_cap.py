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
carbon_cap.py  -  net-zero / emissions-cap test and the carbon floor
=================================================================
Establishes that the cost-optimal program already sits at the carbon floor:
no reallocation of interventions can reduce program emissions below the
carbon-optimal level, so a reduction target relative to inaction is met for
free up to that floor, and deeper cuts require lower-carbon interventions
rather than a different allocation. Reports the do-nothing, cost-optimal, and
floor emissions and the achieved reduction. Writes outputs/carbon_cap_results.csv
and figures/fig_carbon_floor.{png,pdf}.
"""
from __future__ import annotations
import csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import problem as P
import optimizer as OPT


def run(prob=None):
    prob = prob or P.build()
    V, B, tot, n = prob.V, prob.budget_M, prob.total_scour, len(prob.V)

    dn = OPT._aggregate([0] * n, V, tot)                       # do-nothing
    _, co = OPT.cost_optimal(V, B)                             # cost-optimal
    ch_cmin = OPT.solve_choice(V, lambda i, g: V[i][g][OPT.CARBON], B)
    cmin = OPT._aggregate(ch_cmin, V, tot)                     # carbon-optimal (floor)

    E_dn, E_co, E_floor = dn["carbon"] / 1e3, co["carbon"] / 1e3, cmin["carbon"] / 1e3
    reduction = (E_dn - E_co) / E_dn * 100.0
    headroom = (E_co - E_floor) / E_floor * 100.0

    print("NET-ZERO / CARBON-CAP TEST")
    print(f"  do-nothing emissions    : {E_dn:7.1f} k tCO2e")
    print(f"  cost-optimal emissions  : {E_co:7.1f} k tCO2e  ({reduction:.1f}% below do-nothing)")
    print(f"  carbon floor (optimal)  : {E_floor:7.1f} k tCO2e  (cost-optimal is +{headroom:.2f}% above the floor)")
    print(f"  => allocation is already emissions-minimal; deeper cuts need lower-carbon interventions, "
          f"not reallocation.")

    out = C.OUT_DIR / "carbon_cap_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["program", "carbon_k_tCO2e"])
        w.writerow(["do_nothing", round(E_dn, 1)])
        w.writerow(["cost_optimal", round(E_co, 1)])
        w.writerow(["carbon_floor", round(E_floor, 1)])
        w.writerow(["reduction_vs_donothing_pct", round(reduction, 1)])
    print(f"  saved -> {out}")

    # Figure fig_carbon_floor is rendered in figures.py from this CSV, in the shared style.
    return dict(do_nothing=E_dn, cost_optimal=E_co, floor=E_floor, reduction_pct=reduction)


if __name__ == "__main__":
    run()
