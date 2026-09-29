# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
diversion_evidence.py - diverted-traffic schedule from public NBI records
========================================================================
The share of traffic diverted around a bridge in each condition state drives both
road-user cost and use-phase emissions. This module estimates it from the downloaded
FHWA files instead of assuming it:

  NBI Item 41  operational status: A open, P posted for load, R posted for other
               restriction, K closed (D, E, G: shored, temporary, not yet open)
  NBI Item 109 average daily truck traffic, percent

For condition state s (config.NBI_TO_STATE on the worst of Items 58-60):
  phi_s = P(closed | s) + P(posted | s) x median truck share
A closure diverts all traffic; a load posting diverts at most the heavy vehicles
(the truck share), so the posting term is an upper bound. Condition states 3-4
carry no diversion in the model (closures and postings there are rare and not
condition-driven). Two estimates are written:
  pooled  all Rhode Island records 2019-2024 (reference)
  y2024   the 2024 file only (sensitivity)
Outputs: diversion_evidence.json (package root), outputs/diversion_evidence.csv.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import config as C

RAW = C.ROOT / "data" / "raw"
CONDS = ["DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060"]


def records(year: int) -> pd.DataFrame:
    d = pd.read_csv(RAW / f"RI{year}.csv", dtype=str, quotechar="'", encoding="latin-1", on_bad_lines="error")
    for c in CONDS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=CONDS).copy()
    d["state"] = d[CONDS].min(axis=1).map(C.NBI_TO_STATE)
    d["status"] = d.OPEN_CLOSED_POSTED_041.str.strip()
    d["truck_pct"] = pd.to_numeric(d.PERCENT_ADT_TRUCK_109, errors="coerce")
    d["year"] = year
    return d


def schedule(d: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Closure and load-posting shares by model state. The observation unit is the bridge-year record.
    K (closed) diverts all traffic; P (posted for load) diverts heavy vehicles only. R (restriction
    other than load, e.g. speed or vehicle count) and A, B, D, E (open) divert no traffic; G (new
    structure not yet open) is excluded. States 3-4 (NBI 7 and 8-9) are set to zero by modeling
    choice: their few closures and postings are treated as unrelated to condition."""
    d = d[d.status != "G"]
    truck = float(d.truck_pct.median()) / 100.0
    rows, closed_v, posted_v = [], [], []
    for s in range(5):
        g = d[d.state == s]
        closed = float((g.status == "K").mean()); posted = float((g.status == "P").mean())
        c_mod, p_mod = (closed, posted) if s <= 2 else (0.0, 0.0)
        closed_v.append(round(c_mod, 4)); posted_v.append(round(p_mod, 4))
        rows.append(dict(state=s, records=len(g), closed_share=round(closed, 4), posted_share=round(posted, 4),
                         median_truck_share=round(truck, 4), closed_model=round(c_mod, 4), posted_model=round(p_mod, 4),
                         vehicle_equivalent=round(c_mod + p_mod * truck, 4)))
    return dict(closed=closed_v, posted=posted_v), pd.DataFrame(rows)


def main():
    all_years = pd.concat([records(y) for y in range(2019, 2025)], ignore_index=True)
    phi_pooled, t_pooled = schedule(all_years)
    phi_2024, t_2024 = schedule(all_years[all_years.year == 2024])
    out = dict(pooled=phi_pooled, y2024=phi_2024, original=dict(closed=list(C.ORIGINAL_PHI), posted=[0.0] * 5))
    json.dump(out, open(C.ROOT / "diversion_evidence.json", "w"), indent=1)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([t_pooled.assign(estimate="pooled 2019-2024"), t_2024.assign(estimate="2024")]).to_csv(
        C.OUT_DIR / "diversion_evidence.csv", index=False)
    print("  closure (all traffic) and load-posting (trucks) shares by state 0..4 (NBI 0-3, 4, 5-6, 7, 8-9):")
    for k, v in out.items():
        print(f"    {k:9s} closed " + " / ".join(f"{x:.4f}" for x in v["closed"]) + "   posted " + " / ".join(f"{x:.4f}" for x in v["posted"]))
    print(t_pooled.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
