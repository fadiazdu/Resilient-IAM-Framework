# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
descriptives.py  -  inventory descriptives for Section 5.1
==========================================================
Computes and reports every quantity cited in Section 5.1 directly from the
portfolio CSV, so the case-study descriptives are fully reproducible:
  - governing condition-state distribution (worst-component rule)
  - Poor-or-worse count (the preservation candidates)
  - NBI Item 113 code breakdown: not-over-water, coded-stable, scour-critical
  - bridges with nonzero assessed scour exposure (the protectable set) and total
  - climate-candidate count (state <= 2 OR nonzero exposure)
  - correlation between scour severity and daily traffic (AADT)
Writes outputs/inventory_descriptives.csv.
"""
from __future__ import annotations
import csv
import numpy as np
import pandas as pd
import adaptation as CR
import config as C
import problem as P


def run(save=True, verbose=True):
    df = pd.read_csv(CR.CSV)
    df["scour_113"] = df["scour_113"].astype(str).str.strip().str.upper()
    state = df["state0"].astype(int)
    sev = df["scour_severity"].astype(float)
    aadt = df["aadt"].astype(float)

    cond = {s: int((state == s).sum()) for s in [4, 3, 2, 1, 0]}          # Good .. Failed
    poor_or_worse = int((state <= 2).sum())                              # preservation candidates
    not_over_water = int((df["scour_113"] == "N").sum())                 # NBI 113 = N
    coded_stable = int((df["scour_113"] == "5").sum())                   # NBI 113 = 5
    scour_critical = int(df["scour_113"].isin(["0", "1", "2", "3"]).sum())  # NBI 113 codes 0-3
    nonzero = int((sev > 0).sum())                                       # protectable set
    total_scour = float(sev.sum())                                       # total exposure
    candidates_pre = int(((state <= 2) | (sev > 0)).sum())                # before mega-project exclusion
    corr_sev_aadt = float(np.corrcoef(sev, aadt)[0, 1])

    prob = P.build()                                    # authoritative optimizable set + exclusions
    n_optimizable = len(prob.candidates)
    n_mega = len(prob.megaprojects)
    mega_capital = float(sum(c for _, c in prob.megaprojects))

    rows = [
        ("n_bridges", len(df)),
        ("cond_good_4", cond[4]), ("cond_fair_3", cond[3]), ("cond_poor_2", cond[2]),
        ("cond_serious_1", cond[1]), ("cond_failed_0", cond[0]),
        ("poor_or_worse", poor_or_worse),
        ("not_over_water_N", not_over_water),
        ("coded_stable_5", coded_stable),
        ("scour_critical_0to3", scour_critical),
        ("nonzero_exposure", nonzero),
        ("total_scour", round(total_scour, 2)),
        ("climate_candidates_pre_exclusion", candidates_pre),
        ("mega_projects_excluded", n_mega),
        ("mega_projects_capital_M", round(mega_capital, 1)),
        ("optimizable_candidates", n_optimizable),
        ("corr_severity_aadt", round(corr_sev_aadt, 3)),
    ]
    if verbose:
        print("  INVENTORY DESCRIPTIVES (Section 5.1)")
        print(f"    condition: Good {cond[4]}, Fair {cond[3]}, Poor {cond[2]}, "
              f"Serious {cond[1]}, Failed {cond[0]}")
        print(f"    Poor-or-worse (preservation candidates): {poor_or_worse}")
        print(f"    NBI Item 113: not-over-water {not_over_water}, coded-stable {coded_stable}, "
              f"scour-critical {scour_critical}")
        print(f"    nonzero scour exposure (protectable): {nonzero} bridges, total {total_scour:.1f}")
        print(f"    climate candidates: {candidates_pre} before mega-project exclusion, "
              f"{n_mega} mega-projects excluded (${mega_capital:.0f}M combined), "
              f"{n_optimizable} optimizable")
        print(f"    corr(severity, AADT): {corr_sev_aadt:+.3f}")
    if save:
        path = C.OUT_DIR / "inventory_descriptives.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["quantity", "value"])
            for k, v in rows:
                w.writerow([k, v])
        if verbose:
            print(f"  saved -> {path}")
    return dict(rows)


if __name__ == "__main__":
    run()
