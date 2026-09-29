# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
problem.py  -  assemble the efficiency-vs-adaptation problem once
=================================================================
Loads the engine portfolio, attaches NBI Item 113 scour severity from
the portfolio CSV, selects the climate-candidate set (Poor/Fair OR any
scour exposure), and builds the per-bridge option tables the optimizer
consumes. Every analysis (conflict, front, sensitivity, figures) imports
`build()` so they all see the identical problem.

Candidates whose rehabilitation capital cost exceeds one year's Bridge
Formula Program apportionment (config.BUDGET_ANNUAL_M) are treated as
mega-projects: a single such project would consume more than a full
year's entire formula-funded allocation, so it is not something this
routine preservation program could fund, and it is excluded from the
optimized candidate set. Excluded structures stay in `full` (they are
still part of the inventory) but are held out of `candidates`/`V` and
listed separately in `megaprojects` for reporting; the manuscript states
their number and combined cost rather than silently dropping them.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import config as C
import adaptation as A           # option_table, CSV, BUDGET_M, M (engine)

M = A.M
BUDGET_M = A.BUDGET_M
MEGA_THRESHOLD_M = C.BUDGET_ANNUAL_M     # one year's apportionment


class ClimateProblem:
    """Container for the assembled problem."""
    def __init__(self, full, candidates, V, severity, total_scour, megaprojects):
        self.full = full                 # all structures with complete required fields
        self.candidates = candidates     # climate-candidate Bridge objects, mega-projects excluded
        self.V = V                       # option tables aligned to candidates
        self.severity = severity         # bid -> severity
        self.total_scour = total_scour   # sum of candidate severities
        self.budget_M = BUDGET_M
        self.megaprojects = megaprojects  # (Bridge, rehab_capital_M) excluded as mega-projects

    def severity_of(self, i):
        return self.severity.get(str(self.candidates[i].bid).strip(), 0.0)


def _scour_map(csv_path):
    df = pd.read_csv(csv_path, dtype={"bid": str})
    return dict(zip(df["bid"].astype(str).str.strip(),
                    df["scour_severity"].astype(float)))



class ConstrainedV(list):
    """Option table with extra linear rows (sum_ig coef[i][g] y_ig <= rhs) read by optimizer.solve_choice."""
    extra_rows = []


def condition_rows(candidates, full, limit=None, years=None):
    """NHS minimum condition (23 CFR 490.411): in each year of the window, expected structurally
    deficient (state 0-1, i.e. an NBI component rating of 4 or less) NHS deck area must not exceed
    `limit` of total NHS deck area. NHS bridges outside the program are deferred and enter the
    right-hand side through their expected deterioration."""
    limit = C.NHS_SD_LIMIT if limit is None else limit
    years = range(1, C.NHS_WINDOW_YEARS + 1) if years is None else years
    in_prog = {str(b.bid).strip() for b in candidates}
    sd = lambda dists: np.array([float(np.sum(dists[t][:2])) for t in range(len(dists))])
    total = sum(b.deck_m2 for b in full if b.is_highway)
    outside = np.zeros(max(years) + 1)
    for b in full:
        if b.is_highway and str(b.bid).strip() not in in_prog:
            outside += sd(M._mc_exact(b, -1, "none")[0])[:max(years) + 1] * b.deck_m2
    prob_sd = []
    for b in candidates:
        if not b.is_highway:
            prob_sd.append(None); continue
        pd_, pi_ = sd(M._mc_exact(b, -1, "none")[0]), sd(M._mc_exact(b, 1, "major")[0])
        prob_sd.append((pd_, pi_, pd_, pi_))           # defer, rehab, countermeasure, rehab + countermeasure
    rows = []
    for t in years:
        coef = [[0.0] * 4 if p is None else [b.deck_m2 * p[g][t] for g in range(4)] for p, b in zip(prob_sd, candidates)]
        rows.append((coef, limit * total - outside[t]))
    return rows


def with_condition(V, candidates, full):
    cv = ConstrainedV(V)
    cv.extra_rows = condition_rows(candidates, full) if C.NHS_SD_LIMIT is not None else []
    return cv


def build(fixed_ids=None):
    """Return a ClimateProblem built from the real RI portfolio.

    fixed_ids: optional set of bridge identifiers. When given (Monte Carlo draws), the
    candidate population is exactly this set, so sampled costs change coefficients but not
    which bridges are in the study (audit correction: membership previously shifted with the
    sampled rehabilitation cost through the mega-project threshold)."""
    cands, full, _T = M.build_portfolio_real()
    severity = _scour_map(A.CSV)
    sev = lambda b: severity.get(str(b.bid).strip(), 0.0)
    raw_candidates = [b for b in full if b.state0 <= 2 or sev(b) > 0]
    raw_V = [A.option_table(b, sev(b)) for b in raw_candidates]

    candidates, V, megaprojects = [], [], []
    for b, v in zip(raw_candidates, raw_V):
        rehab_capital = v[1][2]                       # option 1 = rehab; index 2 = capital
        excluded = (str(b.bid).strip() not in fixed_ids) if fixed_ids is not None else rehab_capital > MEGA_THRESHOLD_M
        if excluded:
            megaprojects.append((b, rehab_capital))
        else:
            candidates.append(b); V.append(v)

    total = sum(sev(b) for b in candidates)
    V = with_condition(V, candidates, full)     # NHS minimum-condition requirement
    return ClimateProblem(full, candidates, V,
                          {str(b.bid).strip(): sev(b) for b in full},
                          total, megaprojects)
