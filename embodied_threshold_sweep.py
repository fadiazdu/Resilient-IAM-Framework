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
embodied_threshold_sweep.py  -  two-sided validation of the objective screen
=============================================================================
Sweeps embodied-carbon intensity (economics.EMB_REHAB_M2 / adaptation.EMB_
REHAB_M2), holding every other input fixed, and re-runs the real screen
(screening.run) at each level to find where the carbon classification
actually flips.

The flip point is found by applying screening.classify() -- the SAME
shared, pre-specified rule (rho_vpc >= 0.95 AND delta <= 5%) used for the
paper's headline result -- to every swept point, not by reading off a
single statistic's own threshold crossing in isolation. Because rho_vpc
falls and the price of redundancy rises together as embodied intensity
increases, the classification switches from "redundant" to "competing"
at the EARLIER of the two crossings; marking only the rho crossing (as
an earlier version of this analysis did) overstates the margin.

Writes outputs/embodied_threshold.csv.
"""
from __future__ import annotations
import contextlib, io, csv
import numpy as np
import config as C
import economics as E, adaptation as A, problem as P, screening as S

BASELINE = E.EMB_REHAB_M2   # kgCO2e/m2, sourced bridge-deck value (340)


def sweep(mults=None):
    mults = mults if mults is not None else [
        0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 65, 90, 120, 160, 210, 280, 360, 460
    ]
    rows = []
    for mult in mults:
        E.EMB_REHAB_M2 = BASELINE * mult
        A.EMB_REHAB_M2 = BASELINE * mult
        try:
            prob = P.build()
            with contextlib.redirect_stdout(io.StringIO()):
                r = S.run(prob, save=False)
            cls = S.classify(r["rho_vpc"], r["por_carbon"])
            rows.append((mult, BASELINE * mult, r["rho_vpc"], r["por_carbon"], cls))
        finally:
            E.EMB_REHAB_M2 = BASELINE
            A.EMB_REHAB_M2 = BASELINE
    return rows


def find_flip(rows):
    """First swept point (by increasing intensity) classified 'competing'."""
    for mult, inten, rho, delta, cls in rows:
        if cls == "competing":
            return mult, inten
    return None, None


def main():
    rows = sweep()
    print(f"{'mult':>6} {'kgCO2e/m2':>10} {'rho_vpc':>8} {'delta%':>8}  classification")
    for mult, inten, rho, delta, cls in rows:
        print(f"{mult:>6.1f} {inten:>10.0f} {rho:>8.4f} {delta:>8.2f}  {cls}")
    flip_mult, flip_inten = find_flip(rows)
    if flip_mult:
        print(f"\nclassification flips to COMPETING at {flip_mult:.0f}x baseline "
              f"({flip_inten:.0f} kgCO2e/m2), decided by the shared classify() rule "
              f"(rho>=0.95 AND delta<=5%), not by either statistic's own crossing alone.")
    out = C.OUT_DIR / "embodied_threshold.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mult", "emb_kgCO2e_m2", "rho_vpc", "por_carbon_pct", "classification"])
        for mult, inten, rho, delta, cls in rows:
            w.writerow([f"{mult:.2f}", f"{inten:.1f}", f"{rho:.4f}", f"{delta:.3f}", cls])
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
