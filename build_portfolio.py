# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
build_portfolio.py  -  STAGE 1: FHWA NBI file -> Rhode Island bridge inventory
===============================================================================
Reads data/raw/RI2024.csv (downloaded and hash-verified by download_data.py) and
reduces it to the fields the framework uses:

  state0          governing condition state (worst of Items 58-60, config crosswalk)
  deck_m2         deck area, length x width (Items 49, 52), unrounded
  is_highway      NHS membership from Item 104 (audit correction: previously a
                  functional-class proxy that misclassified 124 bridges)
  aadt, truck     Items 29 and 109
  detour_km_019   measured detour length (Item 19)
  scour_113, channel_61, scour_severity   scour code, channel condition, weight

Run by run_all.py; standalone:  python build_portfolio.py
Field layout: FHWA SI&A coding guide (https://www.fhwa.dot.gov/bridge/mtguide.pdf).
Never fabricates: a missing required column is reported, not guessed.
"""
import argparse, os, sys
import pandas as pd

import config as C   # NBI_TO_STATE, SCOUR_SEVERITY, SCOUR_LETTER, SCOUR_BLANK, DETOUR_KM

# NBI delimited headers -> model fields (verify against format.cfm for the year)
COLUMN_MAP = {
    "structure_id": "STRUCTURE_NUMBER_008", "state_fips": "STATE_CODE_001",
    "lat": "LAT_016", "lon": "LONG_017", "year_built": "YEAR_BUILT_027",
    "aadt": "ADT_029", "pct_truck": "PERCENT_ADT_TRUCK_109",
    "lanes_on": "TRAFFIC_LANES_ON_028A", "detour_km": "DETOUR_KILOS_019",
    "length_m": "STRUCTURE_LEN_MT_049", "deck_width_m": "DECK_WIDTH_MT_052",
    "n_spans": "MAIN_UNIT_SPANS_045", "func_class": "FUNCTIONAL_CLASS_026",
    "deck_cond": "DECK_COND_058", "super_cond": "SUPERSTRUCTURE_COND_059",
    "sub_cond": "SUBSTRUCTURE_COND_060", "nhs_104": "HIGHWAY_SYSTEM_104",
    "scour_113": "SCOUR_CRITICAL_113", "channel_61": "CHANNEL_COND_061",
}
HIGHWAY_FUNC_CLASSES = {1, 11, 2, 12}
FIPS_TO_POSTAL = {"44": "RI", "01": "AL", "06": "CA", "36": "NY", "48": "TX"}  # extend as needed


def nbi_to_state(r):
    """Governing NBI rating (0-9) -> 5-state model via the config crosswalk."""
    return C.NBI_TO_STATE.get(int(r), 0)


def scour_to_severity(code):
    """NBI Item 113 -> hazard weight in [0,1] (flagged ordinal map from config)."""
    if code is None: return C.SCOUR_BLANK
    c = str(code).strip().upper()
    if c in ("", "NAN"): return C.SCOUR_BLANK
    if c in C.SCOUR_LETTER: return C.SCOUR_LETTER[c]
    try:
        return C.SCOUR_SEVERITY.get(int(c), 0.0)   # 0-4 mapped; 5-9 -> stable (0)
    except ValueError:
        return C.SCOUR_BLANK


def channel_boost(channel_code, base):
    """NBI Item 61 <= 4 (channel severely undermined) corroborates scour: +0.1, capped 1."""
    try:
        return min(1.0, base + 0.1) if int(str(channel_code).strip()) <= 4 else base
    except (ValueError, TypeError):
        return base


def read_nbi_csv(f):
    return pd.read_csv(f, quotechar="'", dtype=str, encoding="latin-1",
                       engine="python", on_bad_lines="error")   # never drop records silently


def load_dataframe(args):
    """Read the FHWA Rhode Island file downloaded and hash-verified by download_data.py."""
    path = args.local or str(C.ROOT / "data" / "raw" / f"RI{args.year}.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run download_data.py first (step 0 of run_all.py).")
    return read_nbi_csv(path)


def reduce_to_model_fields(df, state_fips=None):
    missing = [v for v in COLUMN_MAP.values() if v not in df.columns]
    if missing:
        print("ERROR: NBI columns not found:", missing)
        print("Available:", list(df.columns)[:60]); sys.exit("Edit COLUMN_MAP (see format.cfm).")
    d = df.rename(columns={v: k for k, v in COLUMN_MAP.items()})
    # numeric coercion EXCLUDES scour_113/channel_61 (single-char codes incl. N/U/T)
    for c in ["lat", "lon", "year_built", "aadt", "pct_truck", "lanes_on",
              "detour_km", "length_m", "deck_width_m", "n_spans", "func_class",
              "deck_cond", "super_cond", "sub_cond"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if state_fips is not None:
        d = d[d["state_fips"].astype(str).str.zfill(2) == str(state_fips).zfill(2)]
    # keep records with the strictly required fields (NOT scour: many valid 'N')
    d = d.dropna(subset=["deck_cond", "super_cond", "sub_cond",
                         "aadt", "length_m", "deck_width_m", "year_built"])
    d = d[(d["length_m"] > 0) & (d["deck_width_m"] > 0) & (d["aadt"] >= 0)]

    scour_raw = d["scour_113"].astype(str).str.strip().str.upper()
    channel_raw = d["channel_61"].astype(str).str.strip()
    severity = [channel_boost(ch, scour_to_severity(s))
                for s, ch in zip(scour_raw, channel_raw)]
    controlling = d[["deck_cond", "super_cond", "sub_cond"]].min(axis=1)
    out = pd.DataFrame({
        "bid": d["structure_id"].astype(str).str.strip(),
        "x_lon": d["lon"], "y_lat": d["lat"],
        "deck_m2": d["length_m"] * d["deck_width_m"],            # unrounded (audit correction)
        "n_lanes": d["lanes_on"].fillna(2).astype("Int64"),
        "aadt": d["aadt"].round(0),
        "truck_frac": d["pct_truck"] / 100.0,       # NBI Item 109; missing stays missing (model uses the statewide median)
        "state0": controlling.map(nbi_to_state).astype("Int64"),
        "year_built": d["year_built"].astype("Int64"),
        "span_m": (d["length_m"] / d["n_spans"].replace(0, 1)).round(1),
        "is_highway": d["nhs_104"].astype(str).str.strip() == "1",   # NHS from Item 104 (audit correction; was functional-class proxy)
        "controlling_nbi_cond": controlling.astype("Int64"),
        "detour_km_019": d["detour_km"].fillna(C.DETOUR_KM).round(2),
        "scour_113": scour_raw,
        "channel_61": channel_raw,
        "scour_severity": [round(s, 3) for s in severity],
    })
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=C.STATE_FIPS)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--local", default=None, help="path to an NBI file (default: data/raw/RI<year>.csv)")
    ap.add_argument("--out", default=str(C.PORTFOLIO_CSV))
    args = ap.parse_args()
    df = load_dataframe(args); print(f"Loaded {len(df):,} raw NBI records.")
    port = reduce_to_model_fields(df, state_fips=args.state)
    port.to_csv(args.out, index=False)
    print(f"{len(port):,} bridges retained -> {args.out}")
    print("condition dist (0 failed .. 4 good):",
          port["state0"].value_counts().sort_index().to_dict())
    print("scour-critical (severity>=0.6):", int((port["scour_severity"] >= 0.6).sum()))


if __name__ == "__main__":
    main()
