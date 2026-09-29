# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
climate_adaptation_real.py
==================================================================
Real efficiency-vs-adaptation conflict test on Rhode Island, using the
actual NBI Item 113 scour severity now in the portfolio and the cheap
HEC-23 countermeasure lever (not full replacement).

Per-bridge choice in {defer, rehabilitate, countermeasure, rehab+cm}:
  rehabilitate restores governing condition (no scour benefit)
  countermeasure removes scour exposure (no condition benefit), priced
    at a flagged fraction of rehab capital, low embodied carbon
  rehab+cm does both

Objectives (minimise) s.t. total capital <= budget:
  COST    ($M)    capital + discounted O&M + discounted user delay
  CARBON  (tCO2e) embodied + use-phase
  CLIMATE (-)     residual scour exposure = sum_i severity_i * [not protected]

We compare the cost-optimal and the climate-optimal programs under the
real $470M budget: portfolio overlap, the cost premium to minimise
residual scour, and how much scour the cost-optimal program ignores.
==================================================================
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pulp
import economics as TB
import config as C
M = TB.M

import os
# Portfolio CSV: defaults to nbi_real_portfolio.csv next to these scripts;
# override with the PORTFOLIO_CSV environment variable if it lives elsewhere.
CSV = Path(os.environ.get("PORTFOLIO_CSV",
           Path(__file__).resolve().parent / "nbi_real_portfolio.csv"))
M.PORTFOLIO_CSV = CSV                       # force engine to read the enriched file

CM_FRACTION  = 0.20                          # countermeasure capital / rehab capital (specified; tested 0.15-0.25)
CM_EFFECT    = 1.0                           # share of the scour weight a countermeasure removes (specified; tested)
EMB_CM_FRAC  = 0.10                          # countermeasure embodied / rehab embodied (flagged; riprap is low-carbon)
EMB_REHAB_M2 = TB.EMB_REHAB_M2
BUDGET_M     = C.BUDGET_M                      # single source of truth: config.py


def option_table(b, severity):
    """[defer, rehab, cm, rehab+cm] -> (cost_M, carbon_t, capital_M, scour_removed)."""
    base = TB.option_data(b)                 # [defer, conv-rehab, low-rehab]
    rehab_cap = base[1][3]
    cm_cap = CM_FRACTION * rehab_cap
    cm_q   = EMB_CM_FRAC * (b.deck_m2 * EMB_REHAB_M2 / 1000.0)
    defer = (base[0][0],          base[0][2],        0.0,             0.0)
    rehab = (base[1][0],          base[1][2],        rehab_cap,       0.0)
    cm    = (base[0][0] + cm_cap, base[0][2] + cm_q, cm_cap,          CM_EFFECT * severity)
    rcm   = (base[1][0] + cm_cap, base[1][2] + cm_q, rehab_cap+cm_cap, CM_EFFECT * severity)
    return [defer, rehab, cm, rcm]


def solve(V, scalar, budget):
    n = len(V); p = pulp.LpProblem("m", pulp.LpMinimize)
    y = [[pulp.LpVariable(f"y_{i}_{g}", cat="Binary") for g in range(4)] for i in range(n)]
    for i in range(n):
        p += pulp.lpSum(y[i]) == 1
    p += pulp.lpSum(scalar(i, g) * y[i][g] for i in range(n) for g in range(4))
    p += pulp.lpSum(V[i][g][2] * y[i][g] for i in range(n) for g in range(4)) <= budget
    p.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=45))
    return [max(range(4), key=lambda g: y[i][g].value()) for i in range(n)]


def stats(ch, V, total_scour):
    cost = sum(V[i][ch[i]][0] for i in range(len(V)))
    carb = sum(V[i][ch[i]][1] for i in range(len(V)))
    resid = total_scour - sum(V[i][ch[i]][3] for i in range(len(V)))
    n_protect = sum(1 for i in range(len(V)) if ch[i] in (2, 3))
    return cost, carb, resid, n_protect


if __name__ == "__main__":
    cands, full, T = M.build_portfolio_real()
    df = pd.read_csv(CSV, dtype={"bid": str})
    scour = dict(zip(df["bid"].astype(str).str.strip(), df["scour_severity"].astype(float)))
    sev = lambda b: scour.get(str(b.bid).strip(), 0.0)

    matched = sum(1 for b in full if str(b.bid).strip() in scour)
    climate_cands_raw = [b for b in full if b.state0 <= 2 or sev(b) > 0]
    V_raw = [option_table(b, sev(b)) for b in climate_cands_raw]
    climate_cands = [b for b, v in zip(climate_cands_raw, V_raw) if v[1][2] <= C.BUDGET_ANNUAL_M]
    n_crit = sum(1 for b in climate_cands if sev(b) >= 0.6)
    print(f"  bridges {len(full)} | scour matched {matched}/{len(full)} | "
          f"climate candidates {len(climate_cands)} | scour-critical among them {n_crit}")
    print(f"  budget ${BUDGET_M:.0f}M | countermeasure = {CM_FRACTION:.0%} of rehab capital (flagged)\n")

    V = [option_table(b, sev(b)) for b in climate_cands]
    total_scour = sum(sev(b) for b in climate_cands)

    cost_ch = solve(V, lambda i, g: V[i][g][0], BUDGET_M)                       # minimise cost
    clim_ch = solve(V, lambda i, g: (sev(climate_cands[i]) if g in (0, 1) else 0.0), BUDGET_M)  # minimise residual scour

    c0, q0, r0, p0 = stats(cost_ch, V, total_scour)
    c1, q1, r1, p1 = stats(clim_ch, V, total_scour)
    overlap = np.mean([cost_ch[i] == clim_ch[i] for i in range(len(V))]) * 100
    premium = (c1 - c0) / c0 * 100
    ignored = r0 / total_scour * 100 if total_scour else 0.0

    print("  RESULT (real RI scour, budget-constrained)")
    print(f"    total scour exposure in candidate set : {total_scour:6.2f}")
    print(f"    cost-optimal    : cost ${c0:6.0f}M  carbon {q0/1e3:6.1f}k  residual-scour {r0:6.2f}  protected {p0:3d}")
    print(f"    climate-optimal : cost ${c1:6.0f}M  carbon {q1/1e3:6.1f}k  residual-scour {r1:6.2f}  protected {p1:3d}")
    print(f"    portfolio overlap                      : {overlap:5.1f}%")
    print(f"    cost premium to minimise scour         : {premium:+6.1f}%")
    print(f"    scour the cost-optimal program ignores : {ignored:5.1f}% of total")
    verdict = "CONFLICT" if (premium > 2 and overlap < 95) else "ALIGNED / negligible"
    print(f"    -> {verdict}")
