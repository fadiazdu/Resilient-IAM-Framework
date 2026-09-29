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
budget_sweep.py  -  carbon redundancy across the budget range
===============================================================
Definition 1 (objective redundancy) requires the cost-optimal and
carbon-optimal integer selections to agree across the budget range
relevant to the program, not only at the single program budget. This
sweeps the capital budget from one quarter to twice the program level
and reports, at each level, the per-asset selection agreement between
the two integer-optimal programs and the price of redundancy.

Writes outputs/budget_sweep.csv.
"""
from __future__ import annotations
import csv
import numpy as np
import config as C
import problem as P
import optimizer as OPT

CO, CA = OPT.COST, OPT.CARBON
FRACS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]


def sweep(prob=None):
    prob = prob or P.build()
    V, B0 = prob.V, prob.budget_M
    rows = []
    for frac in FRACS:
        B = B0 * frac
        ch_cost = OPT.solve_choice(V, lambda i, g: V[i][g][CO], B)
        ch_carb = OPT.solve_choice(V, lambda i, g: V[i][g][CA], B)
        if ch_cost is None or ch_carb is None:
            rows.append((frac, B, None, None)); continue
        agree = float(np.mean([ch_cost[i] == ch_carb[i] for i in range(len(ch_cost))])) * 100.0
        agg_cost = OPT._aggregate(ch_cost, V)
        agg_carb = OPT._aggregate(ch_carb, V)
        delta = (agg_cost["carbon"] - agg_carb["carbon"]) / agg_carb["carbon"] * 100.0 if agg_carb["carbon"] else 0.0
        rows.append((frac, B, agree, delta))
    return rows


def main():
    rows = sweep()
    print("BUDGET-RANGE REDUNDANCY SWEEP")
    print(f"{'B/B0':>6} {'budget_M':>10} {'agreement%':>11} {'delta%':>8}")
    for frac, B, agree, delta in rows:
        if agree is None:
            print(f"{frac:>6.2f} {B:>10.0f}  infeasible")
        else:
            print(f"{frac:>6.2f} {B:>10.0f} {agree:>11.1f} {delta:>8.3f}")
    out = C.OUT_DIR / "budget_sweep.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["B_over_B0", "budget_M", "selection_agreement_pct", "delta_carbon_pct"])
        for frac, B, agree, delta in rows:
            w.writerow([f"{frac:.2f}", f"{B:.0f}",
                       "" if agree is None else f"{agree:.2f}",
                       "" if delta is None else f"{delta:.3f}"])
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
