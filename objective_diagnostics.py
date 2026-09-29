# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
objective_diagnostics.py  -  analyses that test the objective-screening argument
===============================================================================
Run after run_all.py (it reads the inventory, transition matrix, and deterioration
fit that run_all.py builds):

    python objective_diagnostics.py

Three analyses, each writing CSVs to outputs/:

  1. mechanism()        Why cost and emissions agree, and why the agreement breaks.
                        Splits each bridge's rehabilitate-versus-defer change into
                        cost components (road-user, maintenance and inspection,
                        capital) and emission components (use-phase detour,
                        embodied). A "carbon-negative rehabilitation" is one that
                        lowers life-cycle cost but raises emissions (embodied carbon
                        exceeds the detour emissions it avoids). Counts them in the
                        cost-optimal program across budgets and diversion schedules.
                        -> mechanism_assets.csv, mechanism.csv

  2. diversion_sweep()  Continuous family of diverted-traffic schedules
                        PHI = [1, 0.25 s, 0.05 s, 0, 0], s = 0.0 ... 2.0
                        (s = 1 baseline; s = 0.4 the "mild" schedule; s = 0 lowest
                        state only), at the program budget and 1.5 times it.
                        Locates the s at which the carbon price of redundancy
                        crosses 1% and 5%.                     -> diversion_sweep.csv

  3. rule_holdout()     Held-out test of the decision rule. 40 cases drawn by Latin
                        hypercube (seed differs from the Monte Carlo) over diversion
                        scale, embodied-carbon multiplier, deterioration-rate factor,
                        and discount rate; none was used to set the thresholds
                        (rho >= 0.95, delta <= 5%), which were fixed in advance.
                        For each case the rules decide at one screening point
                        (program budget, no protection requirement); the ground
                        truth is whether omitting carbon stays within 5% over the
                        whole decision domain (budget 0.5, 1.0, 1.5 x program level;
                        protection 0, 50, 90%). Reports wrong omissions and
                        unnecessary retentions for correlation-only, loss-only, and
                        combined rules. Also records the budget left unspent by the
                        cost-optimal program (an explanatory variable identified from
                        the mechanism analysis; not scored as a rule, because it was
                        found after seeing results). -> rule_holdout_cases.csv,
                                                          rule_holdout_summary.csv
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import qmc

import config as C
import engine as M
import economics as E
import adaptation as A
import problem as P
import optimizer as OPT
import screening as S
import build_matrix as BM

CO, CA, CAP, SC = 0, 1, 2, 3
ETA = 5.0                          # tolerated carbon price of redundancy, % (fixed in advance)
RHO_CUT = 0.95                     # alignment cutoff (fixed in advance)
BASE_PHI, BASE_POST = list(M.PHI), list(M.PHI_POST)   # reference closure and posting schedules
BASE_HORIZON = M.HORIZON           # reference analysis period (config.ANALYSIS_YEARS)
EVIDENCE = json.load(open(C.ROOT / "diversion_evidence.json"))
BASE_EMB = E.EMB_REHAB_M2
BASE_DR = M.DR
RATES = np.array(json.load(open(C.OUT_DIR / "deterioration_fit.json"))["rates"])


# ----------------------------------------------------------------- scenario control
def _fmt_phi(phi):
    """'closed c0/c1/... | posted p0/p1/...' for a (closed, posted) pair."""
    c, p = phi
    return "closed " + "/".join(f"{x:.3f}" for x in c) + " | posted " + "/".join(f"{x:.3f}" for x in p)


def set_scenario(s=1.0, emb_mult=1.0, det_factor=1.0, dr=BASE_DR, horizon=None, phi=None):
    """Set diversion (scale s on the reference schedule, capped at 1, or an explicit phi),
    embodied multiplier, deterioration-rate factor, discount rate, and analysis period."""
    if phi is not None:                                   # explicit (closed, posted) pair
        M.set_diversion(*phi)
    else:
        M.set_diversion([min(1.0, s * x) for x in BASE_PHI], [min(1.0, s * x) for x in BASE_POST])
    M.HORIZON = BASE_HORIZON if horizon is None else horizon
    E.EMB_REHAB_M2 = A.EMB_REHAB_M2 = BASE_EMB * emb_mult
    M.REAL_T = expm(BM.generator(RATES * det_factor))
    M.DR = dr


def reset():
    set_scenario()


def screen_point(prob, budget_mult=1.0, tau_frac=0.0):
    """Carbon price of redundancy (%) and agreement for one budget and protection level,
    using the same tie-broken exact programs as screening.py."""
    V, B, tot, n = prob.V, prob.budget_M * budget_mult, prob.total_scour, len(prob.V)
    rc = OPT.min_cost_with_protection(V, tau_frac * tot, B)
    re = OPT.carbon_optimal(V, B, min_protection=tau_frac * tot)
    if rc is None or re is None:
        return None
    (chc, sc), (che, se) = rc, re
    return dict(delta=(sc["carbon"] - se["carbon"]) / se["carbon"] * 100, abs_t=sc["carbon"] - se["carbon"],
                same=float(np.mean([chc[i] == che[i] for i in range(n)]) * 100), cost_choice=chc, carbon_choice=che,
                budget=B, cost_capital=sc["capital"], carbon_capital=se["capital"])


def densities(prob):
    """Rehabilitate-versus-defer changes and value densities, exactly as screening.py."""
    V, n = prob.V, len(prob.V)
    cap = np.array([V[i][1][CAP] for i in range(n)])
    dcost = np.array([V[i][0][CO] - V[i][1][CO] for i in range(n)])
    dcarb = np.array([V[i][0][CA] - V[i][1][CA] for i in range(n)])
    return dcost, dcarb, cap


# ----------------------------------------------------------------- 0. diversion x analysis period
def diversion_horizon_grid(lams=(0.0, 0.25, 0.5, 0.75, 1.0), horizons=tuple(range(10, 41)), budget_mult=1.0):
    """Central result: emissions loss, unspent budget, and alignment over diversion schedules
    (lambda interpolates from the evidence-based reference, 0, to the original assumption, 1;
    the 2024-only evidence is added as its own row) and benefit-accounting periods."""
    print("DIVERSION x ANALYSIS-PERIOD GRID (program budget; NHS condition requirement active)")
    rc, rp = np.array(EVIDENCE[C.REFERENCE_DIVERSION]["closed"]), np.array(EVIDENCE[C.REFERENCE_DIVERSION]["posted"])
    oc = np.array(EVIDENCE["original"]["closed"])
    schedules = [(f"lambda {l:.2f}", (rc + l * (oc - rc), (1 - l) * rp)) for l in lams] + \
                [("evidence 2024 only", (np.array(EVIDENCE["y2024"]["closed"]), np.array(EVIDENCE["y2024"]["posted"])))]
    rows = []
    try:
        for label, phi in schedules:
            line = []
            for h in horizons:
                set_scenario(phi=phi, horizon=h); pb = P.build(); dcost, dcarb, cap = densities(pb)
                r = screen_point(pb, budget_mult)
                if r is None:
                    rows.append(dict(schedule=label, phi=_fmt_phi(phi), analysis_years=h,
                                     feasible=False)); line.append("  infeas."); continue
                rows.append(dict(schedule=label, phi=_fmt_phi(phi), analysis_years=h, feasible=True,
                                 rho=round(S.rank_corr(dcost / cap, dcarb / cap), 4), delta_pct=round(r["delta"], 4),
                                 abs_loss_t=round(r["abs_t"], 1), same_option_pct=round(r["same"], 1),
                                 cost_program_unspent_pct=round((1 - r["cost_capital"] / r["budget"]) * 100, 2)))
                line.append(f"{r['delta']:8.3f}")
            print(f"  {label:20s} loss % by analysis period {list(horizons)}: " + " ".join(line), flush=True)
    finally:
        reset()
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "diversion_horizon_grid.csv", index=False)
    print(f"  saved -> {C.OUT_DIR / 'diversion_horizon_grid.csv'}")
    return rows


def discount_horizon_grid(rates=(C.DR, 0.03, 0.04, 0.05, 0.07), horizons=tuple(range(10, 41)), budget_mult=1.0):
    """Accounting conventions: emissions loss, unspent budget, and alignment over discount rates
    and benefit-accounting periods (reference diversion; NHS condition requirement active)."""
    print("DISCOUNT-RATE x ANALYSIS-PERIOD GRID (reference diversion, program budget)")
    rows = []
    try:
        for dr in rates:
            line = []
            for h in horizons:
                set_scenario(dr=dr, horizon=h); pb = P.build(); dcost, dcarb, cap = densities(pb)
                r = screen_point(pb, budget_mult)
                if r is None:
                    rows.append(dict(discount_rate=dr, analysis_years=h, feasible=False)); line.append("  infeas."); continue
                rows.append(dict(discount_rate=dr, analysis_years=h, feasible=True,
                                 rho=round(S.rank_corr(dcost / cap, dcarb / cap), 4), delta_pct=round(r["delta"], 4),
                                 abs_loss_t=round(r["abs_t"], 1), same_option_pct=round(r["same"], 1),
                                 cost_program_unspent_pct=round((1 - r["cost_capital"] / r["budget"]) * 100, 2)))
                line.append(f"{r['delta']:8.3f}")
            print(f"  discount rate {dr:.2f}  loss % by analysis period {list(horizons)}: " + " ".join(line), flush=True)
    finally:
        reset()
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "discount_horizon_grid.csv", index=False)
    print(f"  saved -> {C.OUT_DIR / 'discount_horizon_grid.csv'}")
    return rows


# ----------------------------------------------------------------- 1. mechanism
def asset_components(prob):
    """Per-bridge components of the rehabilitate-versus-defer change."""
    rows = []
    for b, v in zip(prob.candidates, prob.V):
        dd, _ = M._mc_exact(b, -1, "none"); di, _ = M._mc_exact(b, 1, "major")
        user_d, user_r = M._pv_user_cost(b, dd), M._pv_user_cost(b, di)
        om_d, om_r = E.cost_no_failure_M(b, dd) - user_d, E.cost_no_failure_M(b, di) - user_r
        op_d, op_r = E.operational_carbon_t(b, dd), E.operational_carbon_t(b, di)
        cap, emb = v[1][CAP], b.deck_m2 * E.EMB_REHAB_M2 / 1000.0
        rows.append(dict(bid=str(b.bid).strip(), state0=b.state0, exposure=b.aadt * b.detour_km,
                         user_saving_M=user_d - user_r, om_saving_M=om_d - om_r, capital_M=cap,
                         usephase_saving_t=op_d - op_r, embodied_t=emb))
    d = pd.DataFrame(rows)
    d["cost_net_M"] = d.user_saving_M + d.om_saving_M - d.capital_M
    d["carbon_net_t"] = d.usephase_saving_t - d.embodied_t
    d["user_share_of_saving"] = d.user_saving_M / (d.user_saving_M + d.om_saving_M).replace(0, np.nan)
    d["carbon_negative"] = d.carbon_net_t < 0
    return d


def mechanism(budget_mults=(0.5, 1.0, 1.5, 2.0), scales=(0.5, 1.0, 2.5, 5.0)):
    print("MECHANISM: components of the rehabilitate-versus-defer change")
    reset(); prob = P.build(); d = asset_components(prob)
    d.to_csv(C.OUT_DIR / "mechanism_assets.csv", index=False)
    dcost, dcarb, cap = densities(prob)
    print(f"  median share of gross cost saving from road-user cost: {d.user_share_of_saving.median():.3f}")
    print(f"  rank correlation with detour exposure (AADT x detour km): cost density "
          f"{S.rank_corr(dcost / cap, d.exposure):.3f}, emissions density {S.rank_corr(dcarb / cap, d.exposure):.3f}")
    print(f"  carbon-negative rehabilitations among candidates: {int(d.carbon_negative.sum())} of {len(d)}")
    rows = []
    for s in scales:
        set_scenario(s=s); pb = P.build(); da = asset_components(pb)
        for bm in budget_mults:
            r = screen_point(pb, bm)
            reh_c = {i for i, g in enumerate(r["cost_choice"]) if g in (1, 3)}
            reh_e = {i for i, g in enumerate(r["carbon_choice"]) if g in (1, 3)}
            only_e = sorted(reh_e - reh_c); only_c = sorted(reh_c - reh_e)
            neg_c = int(da.iloc[sorted(reh_c)].carbon_negative.sum()) if reh_c else 0
            cost_neg_e = int((da.iloc[only_e].cost_net_M < 0).sum()) if only_e else 0
            rows.append(dict(diversion_scale=s, budget_mult=bm, budget_M=round(r["budget"], 1),
                             delta_pct=round(r["delta"], 4), abs_loss_t=round(r["abs_t"], 1),
                             same_option_pct=round(r["same"], 1),
                             cost_program_capital_M=round(r["cost_capital"], 1),
                             cost_program_unspent_pct=round((1 - r["cost_capital"] / r["budget"]) * 100, 1),
                             carbon_program_capital_M=round(r["carbon_capital"], 1),
                             rehabs_cost_program=len(reh_c), rehabs_carbon_program=len(reh_e),
                             carbon_negative_rehabs_in_cost_program=neg_c,
                             rehabs_only_in_carbon_program=len(only_e), of_which_cost_negative=cost_neg_e,
                             rehabs_only_in_cost_program=len(only_c),
                             emissions_saved_by_carbon_only_rehabs_t=round(float(da.iloc[only_e].carbon_net_t.sum()), 1) if only_e else 0.0))
            print(f"  s={s:3.1f} B x{bm:3.1f}: delta {r['delta']:8.4f}% | cost program spends "
                  f"{r['cost_capital']:6.1f} of {r['budget']:6.1f} M$ ({len(reh_c)} rehabs) | emissions program spends "
                  f"{r['carbon_capital']:6.1f} ({len(reh_e)} rehabs; {len(only_e)} not in cost program, "
                  f"{cost_neg_e} of them cost-negative)")
    reset()
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "mechanism.csv", index=False)
    print(f"  saved -> {C.OUT_DIR / 'mechanism.csv'} and mechanism_assets.csv")
    return rows


# ----------------------------------------------------------------- 2. diversion sweep
def crossing(x, y, level):
    """First x where y rises through level (linear interpolation between sampled points)."""
    for k in range(1, len(x)):
        if (y[k - 1] - level) * (y[k] - level) <= 0 and y[k - 1] != y[k]:
            return float(x[k - 1] + (level - y[k - 1]) * (x[k] - x[k - 1]) / (y[k] - y[k - 1]))
    return None


def diversion_sweep(scales=np.round(np.arange(0.0, 6.01, 0.25), 2), budget_mults=(1.0, 1.5)):
    print("CONTINUOUS DIVERSION SWEEP  PHI = min(1, s x reference schedule); s = 1 is the evidence-based reference")
    rows = []
    try:
        for s in scales:
            set_scenario(s=float(s)); pb = P.build(); dcost, dcarb, cap = densities(pb)
            rho = S.rank_corr(dcost / cap, dcarb / cap)
            for bm in budget_mults:
                r = screen_point(pb, bm)
                rows.append(dict(diversion_scale=float(s), budget_mult=bm, rho=round(rho, 4),
                                 delta_pct=round(r["delta"], 4), abs_loss_t=round(r["abs_t"], 1),
                                 same_option_pct=round(r["same"], 1)))
            print(f"  s={s:4.2f}  rho {rho:.4f}  delta " + "  ".join(
                f"(B x{r_['budget_mult']}) {r_['delta_pct']:8.4f}%" for r_ in rows[-len(budget_mults):]))
    finally:
        reset()
    df = pd.DataFrame(rows); df.to_csv(C.OUT_DIR / "diversion_sweep.csv", index=False)
    for bm in budget_mults:
        g = df[df.budget_mult == bm].sort_values("diversion_scale", ascending=False)   # from heavy to light diversion
        for lvl in (1.0, 5.0):
            x = crossing(g.diversion_scale.values, g.delta_pct.values, lvl)
            print(f"  budget x{bm}: delta crosses {lvl:.0f}% at diversion scale s = "
                  f"{x:.2f}" if x is not None else f"  budget x{bm}: delta never crosses {lvl:.0f}%")
    print(f"  saved -> {C.OUT_DIR / 'diversion_sweep.csv'}")
    return df


# ----------------------------------------------------------------- 3. held-out rule test
def rule_holdout(n_cases=40, seed=C.SEED + 7, budgets=(0.5, 1.0, 1.5), taus=(0.0, 0.5, 0.9)):
    print(f"HELD-OUT DECISION-RULE TEST ({n_cases} cases; thresholds fixed in advance: rho >= {RHO_CUT}, delta <= {ETA}%)")
    u = qmc.LatinHypercube(d=4, seed=seed).random(n_cases)
    design = np.column_stack([6.0 * u[:, 0],                                   # diversion scale s in [0, 6] (s=1 reference)
                              np.exp(np.log(0.5) + u[:, 1] * (np.log(40) - np.log(0.5))),   # embodied x [0.5, 40], log-uniform
                              0.75 + 1.25 * u[:, 2],                           # deterioration factor [0.75, 2]
                              0.02 + 0.05 * u[:, 3]])                          # discount rate [0.02, 0.07]
    part = C.OUT_DIR / "rule_holdout_cases_partial.csv"      # checkpoint: one row per finished case
    rows = pd.read_csv(part).to_dict("records") if part.exists() else []
    rows = [r for r in rows if r["case"] < n_cases and abs(r["diversion_scale"] - round(design[int(r["case"]), 0], 3)) < 1e-9]
    done = {int(r["case"]) for r in rows}
    if done:
        print(f"  resuming: {len(done)} of {n_cases} cases already finished")
    try:
        for k, (s, m, f, dr) in enumerate(design):
            if k in done:
                continue
            set_scenario(s=s, emb_mult=m, det_factor=f, dr=dr); pb = P.build()
            dcost, dcarb, cap = densities(pb); rho = S.rank_corr(dcost / cap, dcarb / cap)
            cells = {(bm, t): screen_point(pb, bm, t) for bm in budgets for t in taus}
            feas = [c["delta"] for c in cells.values() if c is not None]
            d0, dmax = cells[(1.0, 0.0)]["delta"], max(feas)
            rows.append(dict(case=k, diversion_scale=round(s, 3), embodied_mult=round(m, 3), det_factor=round(f, 3),
                             discount_rate=round(dr, 4), rho=round(rho, 4), delta_screen_pct=round(d0, 4),
                             delta_domain_max_pct=round(dmax, 4), feasible_cells=len(feas),
                             cost_program_unspent_pct_screen=round((1 - cells[(1.0, 0.0)]["cost_capital"] / cells[(1.0, 0.0)]["budget"]) * 100, 2),
                             cost_program_unspent_pct_max=round(max((1 - c['cost_capital'] / c['budget']) * 100 for c in cells.values() if c is not None), 2),
                             truth_safe_to_omit=dmax <= ETA))
            pd.DataFrame(rows).to_csv(part, index=False)       # save after every case
            print(f"  case {k + 1:2d}/{n_cases}: s {s:4.2f} emb x{m:5.1f} det x{f:4.2f} dr {dr:.3f} | "
                  f"rho {rho:.4f} delta screen {d0:7.3f}% domain max {dmax:7.3f}%", flush=True)
    finally:
        reset()
    df = pd.DataFrame(rows).sort_values("case").reset_index(drop=True)
    df["truth_safe_to_omit"] = df.truth_safe_to_omit.astype(bool)
    df.to_csv(C.OUT_DIR / "rule_holdout_cases.csv", index=False)
    if len(df) == n_cases and part.exists():
        part.unlink()                                          # complete: remove the checkpoint
    truth = df.truth_safe_to_omit
    summ = []
    for rc in (0.90, 0.95, 0.975):
        rules = {f"correlation only (rho >= {rc})": df.rho >= rc,
                 "loss only (delta at screening point <= 5%)": df.delta_screen_pct <= ETA,
                 f"combined (rho >= {rc} and delta <= 5%)": (df.rho >= rc) & (df.delta_screen_pct <= ETA)}
        for name, omit in rules.items():
            if rc != 0.95 and name.startswith("loss only"):
                continue
            summ.append(dict(rule=name, cases=len(df), omit_decisions=int(omit.sum()),
                             wrong_omissions=int((omit & ~truth).sum()),
                             unnecessary_retentions=int((~omit & truth).sum()),
                             correct=int((omit == truth).sum())))
    sm = pd.DataFrame(summ); sm.to_csv(C.OUT_DIR / "rule_holdout_summary.csv", index=False)
    print(f"  ground truth: omission safe over the whole domain in {int(truth.sum())} of {len(df)} cases")
    print(sm.to_string(index=False))
    print(f"  saved -> {C.OUT_DIR / 'rule_holdout_cases.csv'} and rule_holdout_summary.csv")
    return df, sm


# ----------------------------------------------------------------- budget use on a fine grid
def budget_use_sweep(mults=tuple(np.round(np.arange(0.50, 2.0001, 0.02), 2))):
    """Excess emissions, option agreement, and capital spent by the cost- and emissions-optimal programs
    on a fine budget grid, plus each program's capital without a budget limit (budget above the capital
    of every intervention). Writes budget_use_sweep.csv and budget_unconstrained.csv."""
    reset(); prob = P.build()
    print("BUDGET USE ON A FINE GRID (reference formulation)")
    rows = []
    for m in mults:
        r = screen_point(prob, float(m))
        rows.append(dict(budget_mult=float(m), budget_M=round(prob.budget_M * float(m), 2), feasible=r is not None,
                         excess_pct=round(r["delta"], 4) if r else "", same_pct=round(r["same"], 2) if r else "",
                         cost_capital_M=round(r["cost_capital"], 2) if r else "", carbon_capital_M=round(r["carbon_capital"], 2) if r else ""))
    df = pd.DataFrame(rows); df.to_csv(C.OUT_DIR / "budget_use_sweep.csv", index=False)
    big = sum(max(prob.V[i][g][CAP] for g in range(len(prob.V[i]))) for i in range(len(prob.V))) * 1.01
    r = screen_point(prob, big / prob.budget_M)
    u = dict(program_budget_M=round(prob.budget_M, 2), cost_capital_unconstrained_M=round(r["cost_capital"], 2),
             carbon_capital_unconstrained_M=round(r["carbon_capital"], 2),
             cost_stopping_multiple=round(r["cost_capital"] / prob.budget_M, 4))
    pd.DataFrame([u]).to_csv(C.OUT_DIR / "budget_unconstrained.csv", index=False)
    ok = df[df.feasible]
    print(f"  {len(ok)} of {len(df)} budgets feasible; excess at 1.0x {ok.set_index('budget_mult').excess_pct.get(1.0)}%, at 2.0x {ok.set_index('budget_mult').excess_pct.get(2.0)}%")
    print(f"  without a budget limit: cost program commits {u['cost_capital_unconstrained_M']} M$ "
          f"({u['cost_stopping_multiple']:.3f} x program budget); emissions program commits {u['carbon_capital_unconstrained_M']} M$")
    print(f"  saved -> {C.OUT_DIR / 'budget_use_sweep.csv'} and budget_unconstrained.csv")
    return df, u


def main():
    import os
    diversion_horizon_grid(); print(); discount_horizon_grid(); print(); budget_use_sweep(); print(); mechanism(); print(); diversion_sweep(); print(); rule_holdout(int(os.environ.get('IAM_HOLDOUT_CASES') or 40))
    print(f"\nSolver status counts: {OPT.SOLVE_LOG}")


if __name__ == "__main__":
    main()
