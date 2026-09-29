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
baselines.py  -  single-criterion practice baselines vs the framework
=================================================================
Compares the framework's cost-optimal program against three rules held to the
same capital budget: do-nothing (defer all), worst-condition-first, and
highest-traffic-first. Reports cost, carbon, scour protected, and the cost
penalty relative to the optimized program. Writes outputs/baselines_results.csv.
"""
from __future__ import annotations
import csv
import config as C
import problem as P
import optimizer as OPT


def run(prob=None):
    prob = prob or P.build()
    V, B, tot, n = prob.V, prob.budget_M, prob.total_scour, len(prob.V)
    cand = prob.candidates

    def agg(ch): return OPT._aggregate(ch, V, tot)

    def greedy(order):
        ch = [0] * n; cap = 0.0
        for i in order:
            k = V[i][1][OPT.CAPITAL]
            if cap + k <= B:
                ch[i] = 1; cap += k
        return ch

    _, co = OPT.cost_optimal(V, B)
    rows = [("framework cost-optimal", co)]
    rows.append(("do-nothing (defer all)", agg([0] * n)))
    rows.append(("worst-condition-first",
                 agg(greedy(sorted(range(n), key=lambda i: (cand[i].state0, -cand[i].aadt))))))
    rows.append(("highest-traffic-first",
                 agg(greedy(sorted(range(n), key=lambda i: -cand[i].aadt)))))

    print("SINGLE-CRITERION BASELINES (capital budget held equal)")
    out = C.OUT_DIR / "baselines_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["program", "cost_M", "carbon_k", "protected", "cost_penalty_vs_optimal_pct"])
        for name, s in rows:
            pen = (s["cost"] - co["cost"]) / co["cost"] * 100.0
            print(f"  {name:24s}: ${s['cost']:5.0f}M  {s['carbon']/1e3:6.1f}k  prot {s['protected']:5.2f}  (+{pen:5.1f}%)")
            w.writerow([name, round(s["cost"], 1), round(s["carbon"] / 1e3, 1),
                        round(s["protected"], 2), round(pen, 1)])
    print(f"  saved -> {out}")
    return rows


if __name__ == "__main__":
    run()
