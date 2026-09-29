# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
optimizer.py  -  the one place the optimization runs
=================================================================
Per-bridge decision in {defer, rehab, countermeasure, rehab+cm}. Each
option carries (cost_$M, carbon_tCO2e, capital_$M, scour_removed).

The program is a multi-objective multiple-choice knapsack. We solve it
exactly with the epsilon-constraint method (PuLP/CBC): one objective is
minimized, the others enter as constraints. This is exact for the
integer problem and is what every paper result is built on. A genetic
algorithm (economics.nsga2) is the scalable twin for larger inventories;
it is justified by integer multi-objective theory (Ehrgott 2005), not by
any non-convexity claim, because the cost-carbon trade-off is convex.

Functions
  solve_choice(V, cost_coeff, budget, min_protection)  -> choice indices
  min_cost_with_protection(V, required, budget)         -> full stats tuple
  cost_optimal(V, budget) / climate_optimal(V, sev, budget)
  protection_curve(V, total_scour, fracs, budget)       -> the knee data
"""
from __future__ import annotations
import pulp

N_OPT = 4   # defer, rehab, countermeasure, rehab+cm

# index of each component inside an option tuple
COST, CARBON, CAPITAL, SCOUR = 0, 1, 2, 3
SOLVE_LOG = {}
SOLVE_TIMES = []   # wall-clock seconds per solve (for sizing the limit)
import os, time
TIME_LIMIT = float(os.environ.get('IAM_SOLVER_TIME') or 600)
MIP_GAP = float(os.environ.get('IAM_MIP_GAP') or 1e-6)   # certified relative optimality gap (~$500 on a $500M objective)


def _new_choice_vars(n):
    return [[pulp.LpVariable(f"y_{i}_{g}", cat="Binary") for g in range(N_OPT)]
            for i in range(n)]


def solve_choice(V, cost_coeff, budget, min_protection=None, time_limit=None, bound=None, initial=None):
    """Exact ILP. Minimize sum cost_coeff(i,g), subject to:
       exactly one option per bridge; total capital <= budget;
       optional total scour_removed >= min_protection.
    `cost_coeff` is a callable (i, g) -> float so any scalarized objective
    (pure cost, residual scour, weighted sum) reuses the same solver.
    Returns the chosen option index per bridge, or None if infeasible."""
    n = len(V)
    p = pulp.LpProblem("portfolio", pulp.LpMinimize)
    y = _new_choice_vars(n)
    for i in range(n):
        p += pulp.lpSum(y[i]) == 1                                  # one option each
    p += pulp.lpSum(cost_coeff(i, g) * y[i][g]
                    for i in range(n) for g in range(N_OPT))        # objective
    p += pulp.lpSum(V[i][g][CAPITAL] * y[i][g]
                    for i in range(n) for g in range(N_OPT)) <= budget
    if min_protection is not None:
        p += pulp.lpSum(V[i][g][SCOUR] * y[i][g]
                        for i in range(n) for g in range(N_OPT)) >= min_protection
    for r_coef, r_rhs in getattr(V, "extra_rows", []):              # e.g. NHS condition requirement
        p += pulp.lpSum(r_coef[i][g] * y[i][g] for i in range(n) for g in range(N_OPT)
                        if r_coef[i][g] != 0.0) <= r_rhs
    if bound is not None:                                           # lexicographic stage 2:
        b_coeff, b_rhs = bound                                      # keep stage-1 objective at its optimum
        p += pulp.lpSum(b_coeff(i, g) * y[i][g]
                        for i in range(n) for g in range(N_OPT)) <= b_rhs
    if initial is not None:                                         # warm start from a feasible program
        for i in range(n):
            for g in range(N_OPT):
                y[i][g].setInitialValue(1 if initial[i] == g else 0)
    t_solve = time.time()
    p.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit or TIME_LIMIT, warmStart=initial is not None,
                              gapRel=MIP_GAP))   # optimality proven within this relative gap
    SOLVE_TIMES.append(time.time() - t_solve)
    status = pulp.LpStatus[p.status]
    SOLVE_LOG[status] = SOLVE_LOG.get(status, 0) + 1
    # Audit correction: only a proven optimum is accepted and only proven infeasibility returns
    # None; a time limit or an unproven incumbent stops the run instead of posing as infeasible.
    if status == "Infeasible":
        return None
    if status != "Optimal" or p.sol_status != pulp.LpSolutionOptimal:
        raise RuntimeError(f"solver did not prove optimality (status={status}, sol_status={p.sol_status})")
    return [max(range(N_OPT), key=lambda g: y[i][g].value()) for i in range(n)]


def solve_lex(V, primary, secondary, budget, min_protection=None, abs_tol=1e-7):
    """Near-lexicographic optimum: minimize `primary`; then, among programs whose primary
    value is within `abs_tol` (in the primary's units) of the proven optimum, minimize
    `secondary`. Ties are thereby resolved by a stated rule rather than solver path.
    Tolerance: max(abs_tol, 1e-7 x |optimum|), i.e. about $163 on a $1,630M cost and 74 kg on
    738 kt of emissions, far below reported precision and within what the solver can verify."""
    ch1 = solve_choice(V, primary, budget, min_protection=min_protection)
    if ch1 is None:
        return None
    z1 = sum(primary(i, ch1[i]) for i in range(len(V)))
    # Tie tolerance: the stated absolute tolerance, but never tighter than 1e-7 of the optimum, the
    # solver's own feasibility precision (a tighter bound cannot be verified and can stall the proof).
    tol = max(abs_tol, 1e-7 * abs(z1))
    ch2 = solve_choice(V, secondary, budget, min_protection=min_protection, bound=(primary, z1 + tol),
                       initial=ch1)   # the stage-1 program is feasible for stage 2: start from it
    if ch2 is None:
        raise RuntimeError("second lexicographic stage failed although the first-stage program satisfies it")
    return ch2


def _cost(V):   return lambda i, g: V[i][g][COST]
def _carb(V):   return lambda i, g: V[i][g][CARBON]
def _negsc(V):  return lambda i, g: -V[i][g][SCOUR]


def carbon_optimal(V, budget, min_protection=None):
    """Minimize emissions (ties -> minimum cost), optionally under a protection requirement."""
    ch = solve_lex(V, _carb(V), _cost(V), budget, min_protection, abs_tol=1e-3)
    return (ch, _aggregate(ch, V)) if ch is not None else None


def adaptation_optimal(V, budget):
    """Maximize weighted scour exposure addressed (ties -> minimum cost)."""
    ch = solve_lex(V, _negsc(V), _cost(V), budget)
    return ch, _aggregate(ch, V)


def priced_optimal(V, budget, price_per_t):
    """Minimize cost (M$) + carbon price ($/t) x emissions (t); ties -> minimum emissions."""
    lam = price_per_t / 1e6
    ch = solve_lex(V, lambda i, g: V[i][g][COST] + lam * V[i][g][CARBON], _carb(V), budget)
    return ch, _aggregate(ch, V)


def _aggregate(ch, V, total_scour=None):
    cost = sum(V[i][ch[i]][COST] for i in range(len(V)))
    carb = sum(V[i][ch[i]][CARBON] for i in range(len(V)))
    cap = sum(V[i][ch[i]][CAPITAL] for i in range(len(V)))
    prot = sum(V[i][ch[i]][SCOUR] for i in range(len(V)))
    ncm = sum(1 for i in range(len(V)) if ch[i] in (2, 3))
    out = dict(cost=cost, carbon=carb, capital=cap, protected=prot, n_cm=ncm)
    if total_scour is not None:
        out["residual"] = total_scour - prot
    return out


def cost_optimal(V, budget):
    """Minimize total program cost (ties -> minimum emissions). Returns (choices, stats)."""
    ch = solve_lex(V, _cost(V), _carb(V), budget)
    return ch, _aggregate(ch, V)


def climate_optimal(V, severity_of, budget):
    """Minimize residual scour exposure under the budget.
    `severity_of(i)` returns bridge i's scour severity (charged when not
    protected, i.e. options defer/rehab)."""
    ch = solve_choice(V, lambda i, g: (severity_of(i) if g in (0, 1) else 0.0), budget)
    return ch, _aggregate(ch, V)


def min_cost_with_protection(V, required, budget, lex=True):
    """Minimize cost while protecting at least `required` units of scour.
    lex=True breaks ties among cost optima by minimum emissions (used for reported
    programs); lex=False solves the single-stage problem, as in the solver benchmark,
    where only the frontier cost matters and both methods solve the same problem."""
    ch = (solve_lex(V, _cost(V), _carb(V), budget, min_protection=required) if lex
          else solve_choice(V, _cost(V), budget, min_protection=required))
    if ch is None:
        return None
    return ch, _aggregate(ch, V)


def protection_curve(V, total_scour, fracs, budget):
    """The knee: for each target share of scour protected, the min-cost
    program and its premium over the unconstrained (0%) cost.
    Returns list of dicts with frac, cost, premium_pct, carbon, capital, n_cm."""
    rows, base = [], None
    for f in fracs:
        res = min_cost_with_protection(V, f * total_scour, budget)
        if res is None:
            rows.append(dict(frac=f, feasible=False)); continue
        _ch, s = res
        if base is None:
            base = s["cost"]
        rows.append(dict(frac=f, feasible=True, cost=s["cost"],
                         premium_pct=(s["cost"] - base) / base * 100.0,
                         carbon=s["carbon"], capital=s["capital"], n_cm=s["n_cm"]))
    return rows
