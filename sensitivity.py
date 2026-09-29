# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
sensitivity.py  -  deterministic sensitivity cases
=================================================================
Each case changes one assumption and repeats the objective screen or the protection premiums:
  (1) countermeasure cost fraction CM_FRACTION (HEC-23 bid range 0.15-0.25, +0.40 stress)
  (2) NBI Item 113 -> severity mapping (baseline / strict / binary / conservative)
  (3) countermeasure embodied-carbon fraction (enters CARBON only, not the cost knee)
  (4) discount rate (reference 0.023, OMB A-94 Appendix C 30-year real Treasury rate; 0.07 A-94 benefit-cost base case)
  (5) analysis period (10, 20, 35, 40 years)
  (6) rehabilitation reset state (NBI 8-9, NBI 7, probabilistic mix)
  (7) mega-project inclusion
  (8) value of travel time by vehicle class (USDOT 2025 all-purpose, personal, and truck-driver values)
  (9) maintenance expenditure in the lowest condition state (reference 0; alternative: NBI 4 rate)
Writes outputs/sensitivity_results.csv (cases 1-7) and outputs/valuation_sensitivity.csv (cases 8-9).
"""
from __future__ import annotations
import csv
import numpy as np, pandas as pd
import adaptation as CR
import optimizer as OPT
import config as C
import problem as P
import screening as S
M = CR.M

_df = pd.read_csv(CR.CSV, dtype={"bid": str, "scour_113": str, "channel_61": str})
_raw = {str(r.bid).strip(): (str(r.scour_113).strip().upper(), str(r.channel_61).strip())
        for r in _df.itertuples()}
_full = M.build_portfolio_real()[1]


def make_map(num_map, U, T, four):
    """Build a severity function from an NBI Item 113 code map (channel<=4 adds +0.1)."""
    full_num = dict(num_map); full_num[4] = four
    def f(b):
        code, ch = _raw.get(str(b.bid).strip(), ("N", ""))
        if code in ("N", "", "NAN"): base = 0.0
        elif code == "T": base = T
        elif code == "U": base = U
        else:
            try: base = full_num.get(int(code), 0.0)
            except ValueError: base = 0.5
        try: base = min(1.0, base + 0.1) if int(ch) <= 4 else base
        except (ValueError, TypeError): pass
        return base
    return f

MAPS = {
    "baseline":     make_map({0:1.0,1:0.9,2:0.8,3:0.6}, U=0.5, T=0.1, four=0.3),
    "strict(0-3)":  make_map({0:1.0,1:0.9,2:0.8,3:0.6}, U=0.0, T=0.0, four=0.0),
    "binary":       make_map({0:1.0,1:1.0,2:1.0,3:1.0}, U=1.0, T=0.0, four=0.0),
    "conservative": make_map({0:1.0,1:0.95,2:0.9,3:0.8}, U=1.0, T=0.2, four=0.5),
}


def build(sev_fn, cm_frac, emb_frac):
    """Rebuild candidate option tables at a given countermeasure cost/carbon and
    mapping. Applies the SAME mega-project exclusion as problem.py (rehab
    capital > one year's apportionment) rather than a second, independent
    copy of the candidate filter, so this module cannot silently drift out
    of sync with the main candidate-building logic again."""
    import problem as P
    CR.CM_FRACTION, CR.EMB_CM_FRAC = cm_frac, emb_frac
    cc_raw = [b for b in _full if b.state0 <= 2 or sev_fn(b) > 0]
    V_raw = [CR.option_table(b, sev_fn(b)) for b in cc_raw]
    cc, V = [], []
    for b, v in zip(cc_raw, V_raw):
        if v[1][2] <= P.MEGA_THRESHOLD_M:      # option 1 = rehab; index 2 = capital
            cc.append(b); V.append(v)
    V = P.with_condition(V, cc, _full)            # same NHS condition requirement as problem.build
    return V, sum(sev_fn(b) for b in cc), len(cc)


def premiums(V, total, B):
    base = OPT.min_cost_with_protection(V, 0.0, B)[1]["cost"]
    out = {}
    for f in (0.75, 0.90, 1.00):
        out[int(f*100)] = (OPT.min_cost_with_protection(V, f*total, B)[1]["cost"] - base) / base * 100
    return out


def run(save=True, verbose=True):
    B = CR.BUDGET_M
    rows = []
    if verbose: print("  (1) COUNTERMEASURE COST FRACTION (mapping=baseline)\n      CM_frac   +75%   +90%  +100%")
    for cm in C.CM_FRAC_SENS:
        V, tot, _ = build(MAPS["baseline"], cm, 0.10); pr = premiums(V, tot, B)
        rows.append(dict(sweep="cm_cost_fraction", case=f"{cm:.2f}",
                         p75=round(pr[75],1), p90=round(pr[90],1), p100=round(pr[100],1)))
        if verbose: print(f"        {cm:.2f}   {pr[75]:+5.1f}  {pr[90]:+5.1f}  {pr[100]:+6.1f}")

    if verbose: print("\n  (2) SCOUR SEVERITY MAPPING (CM=0.20)\n      mapping        nCand  totScour   +75%   +90%  +100%")
    for name, fn in MAPS.items():
        V, tot, nc = build(fn, 0.20, 0.10); pr = premiums(V, tot, B)
        rows.append(dict(sweep="severity_mapping", case=name, n_cand=nc,
                         total_scour=round(tot,2), p75=round(pr[75],1),
                         p90=round(pr[90],1), p100=round(pr[100],1)))
        if verbose: print(f"      {name:<13} {nc:5d}  {tot:7.2f}  {pr[75]:+5.1f}  {pr[90]:+5.1f}  {pr[100]:+6.1f}")

    if verbose: print("\n  (3) COUNTERMEASURE EMBODIED-CARBON FRACTION (carbon at 100% protect, baseline, CM=0.20)")
    for emb in C.EMB_CM_SENS:
        V, tot, _ = build(MAPS["baseline"], 0.20, emb)
        carb = OPT.min_cost_with_protection(V, tot, B)[1]["carbon"]
        rows.append(dict(sweep="cm_carbon_fraction", case=f"{emb:.2f}", carbon_k=round(carb/1e3,1)))
        if verbose: print(f"      EMB_frac {emb:.2f} -> program carbon {carb/1e3:7.1f} k tCO2e")
    if verbose: print("      (cost knee is invariant to this parameter; it enters carbon only)")

    if verbose: print("\n  (4) DISCOUNT RATE (mapping=baseline, CM=0.20; 0.023 reference, 0.07 benefit-cost base case)\n      DR     cost-optimal($M)   +90%")
    _dr0 = M.DR
    for dr in (C.DR, 0.07):
        M.DR = dr
        V, tot, _ = build(MAPS["baseline"], 0.20, 0.10)
        base = OPT.min_cost_with_protection(V, 0.0, B)[1]["cost"]
        p90 = (OPT.min_cost_with_protection(V, 0.90 * tot, B)[1]["cost"] - base) / base * 100
        rows.append(dict(sweep="discount_rate", case=f"{dr:.2f}", cost_opt_M=round(base), p90=round(p90, 1)))
        if verbose: print(f"      {dr:.2f}    {base:9.0f}      {p90:+5.1f}")
    M.DR = _dr0
    CR.CM_FRACTION, CR.EMB_CM_FRAC = 0.20, 0.10   # restore defaults

    if verbose: print("\n  (5) ANALYSIS PERIOD (mapping=baseline, CM=0.20; 35 yr reference, 10/20/40 yr sensitivity)")
    _hz0 = M.HORIZON
    for hz in (10, 20, 35, 40):
        M.HORIZON = hz
        prob = P.build()
        r = S.run(prob, save=False)
        rows.append(dict(sweep="planning_horizon", case=f"{hz}yr",
                         cost_opt_M=round(r["cost_opt"]["cost"]),
                         p90=round(r["prem90"], 1)))
        if verbose: print(f"      {hz:2d} yr   rho_vpc={r['rho_vpc']:.4f}   cost=${r['cost_opt']['cost']:.0f}M   +90%={r['prem90']:.1f}%   -> {r['carbon_class']}")
    M.HORIZON = _hz0

    if verbose: print("\n  (6) REHABILITATION RESET STATE (major intervention: state 4 baseline, state 3, probabilistic mix)")
    _reset0 = dict(M.MAJOR_RESET_DIST)
    for name, dist in [("state4", {4: 1.0}), ("state3", {3: 1.0}), ("probabilistic", {4: 0.6, 3: 0.4})]:
        M.MAJOR_RESET_DIST = dist
        prob = P.build()
        r = S.run(prob, save=False)
        rows.append(dict(sweep="rehab_reset_state", case=name,
                         cost_opt_M=round(r["cost_opt"]["cost"]),
                         p90=round(r["prem90"], 1)))
        if verbose: print(f"      {name:14s}  rho_vpc={r['rho_vpc']:.4f}  rho_adapt={r['rho_cs']:+.4f}  cost=${r['cost_opt']['cost']:.0f}M  -> {r['carbon_class']}/{r['adapt_class']}")
    M.MAJOR_RESET_DIST = _reset0

    if verbose: print("\n  (7) MEGA-PROJECT INCLUSION (excluded by default vs included)")
    _mega0 = P.MEGA_THRESHOLD_M
    for name, thresh in [("excluded (baseline)", _mega0), ("included", float("inf"))]:
        P.MEGA_THRESHOLD_M = thresh
        prob = P.build()
        r = S.run(prob, save=False)
        rows.append(dict(sweep="mega_project_inclusion", case=name,
                         cost_opt_M=round(r["cost_opt"]["cost"]),
                         p90=round(r["prem90"], 1)))
        if verbose: print(f"      {name:20s}  n={len(prob.candidates):3d}  rho_vpc={r['rho_vpc']:.4f}  cost=${r['cost_opt']['cost']:.0f}M  -> {r['carbon_class']}")
    P.MEGA_THRESHOLD_M = _mega0

    # (8) and (9): valuation assumptions, reported as excess emissions and the cost program's capital by budget
    import objective_diagnostics as D
    def by_budget(sweep, case):
        pb = P.build(); out = []
        for mult in (1.0, 1.5, 2.0):
            r = D.screen_point(pb, mult)
            out.append(dict(sweep=sweep, case=case, budget_mult=mult,
                            excess_pct=round(r["delta"], 4) if r else "", cost_program_capital_M=round(r["cost_capital"], 1) if r else ""))
        if verbose: print(f"      {case:34s} " + "  ".join(f"{o['budget_mult']:g}xB: {o['excess_pct']}% (cost program {o['cost_program_capital_M']} M$)" for o in out))
        return out
    val = []
    if verbose: print("\n  (8) VALUE OF TRAVEL TIME BY VEHICLE CLASS (USDOT 2025 Table A-2; reference: one value per vehicle)")
    _vot0, _vt0 = M.VOT, M.VOT_TRUCK
    ref = by_budget("time_value", f"all vehicles {_vot0:.2f} (reference)"); val += ref
    for car, name in ((_vot0, "all-purpose"), (C.VOT_PERSONAL, "personal")):
        M.VOT, M.VOT_TRUCK = car, C.VOT_TRUCK_DRIVER
        val += by_budget("time_value", f"trucks {C.VOT_TRUCK_DRIVER:.2f}, cars {car:.2f} ({name})")
    M.VOT, M.VOT_TRUCK = _vot0, _vt0
    if verbose: print("\n  (9) MAINTENANCE IN THE LOWEST STATE (reference 0: preservation withdrawn; alternative: NBI 4 rate)")
    _m0 = M.MAINT[0]
    val += [dict(r, sweep="lowest_state_maintenance", case=f"{_m0:g} $/m2 (reference)") for r in ref]
    M.MAINT[0] = M.MAINT[1]
    val += by_budget("lowest_state_maintenance", f"{M.MAINT[0]:g} $/m2 (NBI 4 rate)")
    M.MAINT[0] = _m0
    if save:
        vp = C.OUT_DIR / "valuation_sensitivity.csv"
        with open(vp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["sweep", "case", "budget_mult", "excess_pct", "cost_program_capital_M"]); w.writeheader()
            for r in val: w.writerow(r)
        if verbose: print(f"  saved -> {vp}")

    if save:
        path = C.OUT_DIR / "sensitivity_results.csv"
        cols = ["sweep","case","n_cand","total_scour","cost_opt_M","p75","p90","p100","carbon_k"]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow({k: r.get(k,"") for k in cols})
        if verbose: print(f"\n  saved -> {path}")
    return rows


if __name__ == "__main__":
    run()
