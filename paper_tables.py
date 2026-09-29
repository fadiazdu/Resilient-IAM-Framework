# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
paper_tables.py  -  results summary table of the paper, computed from existing outputs
=======================================================================================
Excess emissions of the cost-optimal program over the minimum achievable, for the reference
formulation and for the cases in which substitution was tested, with the 5 percent omission
tolerance applied. Thresholds (budget and analysis period at which the excess reaches 5 percent)
are interpolated linearly on the fine budget grid and the annual period grid.
Writes outputs/table_results_summary.csv. Run after run_all.py (it is called by run_all step 8).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config as C

OUT, TOL = C.OUT_DIR, 5.0


def first_crossing(x, y, level=TOL):
    """Smallest x at which y reaches `level`, by linear interpolation between grid points (None if never)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(1, len(x)):
        if y[i - 1] < level <= y[i]:
            return x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
    return None


def last_crossing_down(x, y, level=TOL):
    """Largest x below which y exceeds `level` (x increasing, y decreasing), by linear interpolation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(len(x) - 1, 0, -1):
        if y[i - 1] > level >= y[i]:
            return x[i - 1] + (y[i - 1] - level) * (x[i] - x[i - 1]) / (y[i - 1] - y[i])
    return None


def main():
    scr = pd.read_csv(OUT / "screening_results.csv").set_index("metric")["value"]
    rho_ref = float(scr["rho_vpc_carbon"])
    mech = pd.read_csv(OUT / "mechanism.csv"); mech = mech[mech.diversion_scale == 1.0].set_index("budget_mult")
    bud = pd.read_csv(OUT / "budget_use_sweep.csv"); bud = bud[bud.feasible.astype(str) == "True"].astype({"excess_pct": float})
    unc = pd.read_csv(OUT / "budget_unconstrained.csv").iloc[0]
    div = pd.read_csv(OUT / "diversion_horizon_grid.csv"); ref = div[div.schedule == "lambda 0.00"].set_index("analysis_years")
    dis = pd.read_csv(OUT / "discount_horizon_grid.csv"); d7 = dis[dis.discount_rate.round(3) == 0.07].set_index("analysis_years")
    val = pd.read_csv(OUT / "valuation_sensitivity.csv")
    truck = val[(val.sweep == "time_value") & val.case.str.startswith("trucks") & val.case.str.contains("all-purpose")].set_index("budget_mult")
    mc = pd.read_csv(OUT / "montecarlo_results.csv")
    B0 = float(unc.program_budget_M); NA = np.nan
    row = lambda case, rho, same, unspent, kt, pct: dict(case=case, alignment_rho=rho, same_option_pct=same, budget_unspent_pct=unspent,
                                                         excess_kt=kt, excess_pct=pct, emissions_omitted="Yes" if pct <= TOL else "No")
    rows = [row("Reference: program budget, 35-year period, 2.3% discount rate", rho_ref, mech.loc[1.0, "same_option_pct"],
                mech.loc[1.0, "cost_program_unspent_pct"], mech.loc[1.0, "abs_loss_t"] / 1e3, mech.loc[1.0, "delta_pct"])]
    for m in (1.5, 2.0):                       # alignment compares capital-normalized savings, so it does not depend on the budget
        rows.append(row(f"Budget ratio {m:.1f}", rho_ref, mech.loc[m, "same_option_pct"], mech.loc[m, "cost_program_unspent_pct"],
                        mech.loc[m, "abs_loss_t"] / 1e3, mech.loc[m, "delta_pct"]))
    for y in (20, 15, 10):
        r = ref.loc[y]
        rows.append(row(f"Analysis period {y} years", r.rho, r.same_option_pct, r.cost_program_unspent_pct, r.abs_loss_t / 1e3, r.delta_pct))
    r = d7.loc[35]
    rows.append(row("Discount rate 7%, 35-year period", r.rho, r.same_option_pct, r.cost_program_unspent_pct, r.abs_loss_t / 1e3, r.delta_pct))
    for m in (1.0, 2.0):                       # this analysis records only excess emissions and capital committed
        unspent = max(0.0, 100 * (1 - float(truck.loc[m, "cost_program_capital_M"]) / (B0 * m)))
        rows.append(row(f"Truck-driver time value, budget ratio {m:.1f}", NA, NA, unspent, NA, float(truck.loc[m, "excess_pct"])))
    ex, kt, rh = mc["por_carbon_%"].dropna(), mc["carbon_abs_loss_t"].dropna() / 1e3, mc["rho_vpc_carbon"].dropna()
    for lab, q in (("median", 0.5), ("95th percentile", 0.95), ("maximum", 1.0)):
        rows.append(row(f"Monte Carlo, 512 draws: {lab}", float(rh.quantile(1 - q if q < 1 else 0.0)) if q < 1 else float(rh.min()),
                        NA, NA, float(kt.quantile(q)), float(ex.quantile(q))))
    t = pd.DataFrame(rows).round({"alignment_rho": 4, "same_option_pct": 1, "budget_unspent_pct": 1, "excess_kt": 2, "excess_pct": 4})
    t.to_csv(OUT / "table_results_summary.csv", index=False)
    b5 = first_crossing(bud.budget_mult, bud.excess_pct)
    p5 = last_crossing_down(ref.index.values, ref.delta_pct.values)
    thr = pd.DataFrame([dict(quantity="budget multiple at which the excess reaches 5 percent", value=round(b5, 3) if b5 else ""),
                        dict(quantity="analysis period (years) below which the excess exceeds 5 percent", value=round(p5, 2) if p5 else ""),
                        dict(quantity="cost program capital without a budget limit (M$)", value=unc.cost_capital_unconstrained_M),
                        dict(quantity="cost program stopping point (multiple of the program budget)", value=unc.cost_stopping_multiple)])
    thr.to_csv(OUT / "table_results_thresholds.csv", index=False)
    print("RESULTS SUMMARY TABLE (Table 2): cost-optimal program vs the minimum achievable emissions")
    print(t.to_string(index=False)); print(thr.to_string(index=False))
    print("  Monte Carlo rows pair each excess quantile with the matching lower-tail alignment (median; 5th percentile; minimum).")
    print(f"  saved -> {OUT / 'table_results_summary.csv'} and table_results_thresholds.csv")


if __name__ == "__main__":
    main()
