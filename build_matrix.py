# A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management
# Code, data, and supplemental materials
# Fredy Díaz-Durán · ORCID 0000-0001-5344-5466 · diazdura@ualberta.ca · fadiazdu@uwaterloo.ca
# Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada
# Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada
# DOI: 10.5281/zenodo.22973335
# Licenses. Code: MIT (LICENSE). Documents, figures, and outputs: CC BY 4.0 (LICENSE-CC-BY-4.0.md). NBI files: public domain (DATA_NOTICE.md).
# SPDX-License-Identifier: MIT
"""
build_matrix.py - deterioration model from inspection intervals
===============================================================
Estimates the annual condition-transition matrix from the six FHWA Rhode Island
NBI files downloaded by download_data.py (data/raw/RI2019.csv ... RI2024.csv).

Audit correction. Consecutive annual NBI submissions are NOT consecutive
inspections: routine inspections occur at intervals of up to 24 months, and
ratings are carried forward between them (1,507 adjacent-year record pairs share
an inspection date). The previous version paired annual files as if each were a
new one-year observation, which overstated persistence. This version:
  1. dates every record by its inspection date (Item 90, MMYY);
  2. deletes (bridge, date) groups whose component ratings disagree across the
     repeated reports, and keeps one record per remaining inspection;
  3. forms intervals between consecutive distinct inspections of each bridge,
     excluding intervals with any component-rating increase (possible unrecorded
     repair) or that cross a deleted conflicting inspection;
  4. fits a continuous-time one-step deterioration process (generator Q with
     rates lambda_1..lambda_4; state 0 absorbing) by maximizing the likelihood of
     the observed interval outcomes, P(dt) = expm(Q dt) (Jackson 2011);
  5. reports the annual matrix P = expm(Q), and bridge-cluster bootstrap
     matrices (bridges resampled with all their intervals) for the Monte Carlo.

States are indexed worst (0) to best (4), matching config.NBI_TO_STATE and
engine.py. Outputs:
  transition_matrix.csv          annual matrix used by the model (package root)
  transition_bootstrap.npy       bootstrap matrices for the Monte Carlo (package root)
  outputs/inspection_intervals.csv, outputs/deterioration_estimates.csv,
  outputs/deterioration_fit.json, outputs/annual_snapshot_matrix.csv
  (the previous annual-pairing estimate, kept only for comparison)
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import exprel
import config as C

RAW = C.ROOT / "data" / "raw"
YEARS = range(2019, 2025)
CONDS = ["DECK_COND_058", "SUPERSTRUCTURE_COND_059", "SUBSTRUCTURE_COND_060"]


# ----------------------------------------------------------------- inspection records
def read_year(year: int) -> pd.DataFrame:
    d = pd.read_csv(RAW / f"RI{year}.csv", dtype=str, quotechar="'", encoding="latin-1", on_bad_lines="error")
    assert d.STATE_CODE_001.eq(C.STATE_FIPS).all(), "unexpected state code"
    d["bid"] = d.STRUCTURE_NUMBER_008.str.strip()
    assert not d.bid.duplicated().any(), "duplicate structure identifier"
    for c in CONDS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["rating"] = d[CONDS].min(axis=1).where(d[CONDS].notna().all(axis=1))
    d["state"] = d.rating.map(C.NBI_TO_STATE)

    def month(code):                      # Item 90 'MMYY' -> integer month index
        s = str(code).strip().zfill(4)
        if not s.isdigit() or len(s) != 4:
            return np.nan
        m, y = int(s[:2]), int(s[2:]); y += 2000 if y < 80 else 1900
        return y * 12 + m - 1 if 1 <= m <= 12 and y <= year else np.nan
    d["inspect_month"] = d.DATE_OF_INSPECT_090.map(month)
    d["snapshot"] = year
    return d


def inspection_intervals(frames):
    allr = pd.concat(frames, ignore_index=True).dropna(subset=["state", "inspect_month"])
    conflict = allr.groupby(["bid", "inspect_month"])[CONDS].nunique().max(axis=1) > 1
    bad = set(conflict[conflict].index)
    allr["conflict"] = [(b, t) in bad for b, t in zip(allr.bid, allr.inspect_month)]
    ev = (allr.loc[~allr.conflict].sort_values(["bid", "inspect_month", "snapshot"])
              .drop_duplicates(["bid", "inspect_month"], keep="last"))
    bad_by_bid = {}
    for b, t in bad:
        bad_by_bid.setdefault(b, []).append(t)
    rows = []
    for bid, g in ev.groupby("bid"):
        r = g.to_dict("records")
        for a, b in zip(r[:-1], r[1:]):
            cross = any(a["inspect_month"] < t < b["inspect_month"] for t in bad_by_bid.get(bid, []))
            improve = any(b[c] > a[c] for c in CONDS)
            rows.append(dict(bid=bid, origin=int(a["state"]), destination=int(b["state"]),
                             years=(b["inspect_month"] - a["inspect_month"]) / 12,
                             component_improvement=improve, crosses_conflict=cross,
                             included=not (improve or cross)))
    pairs = pd.DataFrame(rows)
    assert pairs.years.gt(0).all()
    same = 0                               # adjacent annual records that repeat one inspection
    for a, b in zip(frames[:-1], frames[1:]):
        j = a.dropna(subset=["state"]).merge(b.dropna(subset=["state"]), on="bid", suffixes=("_a", "_b"))
        same += int(j.inspect_month_a.eq(j.inspect_month_b).sum())
    audit = dict(conflicting_inspection_dates=len(bad), distinct_inspections=len(ev), all_intervals=len(pairs),
                 retained_intervals=int(pairs.included.sum()),
                 component_improvement_intervals=int(pairs.component_improvement.sum()),
                 conflict_crossing_intervals=int(pairs.crosses_conflict.sum()),
                 same_date_annual_records=same)
    return pairs, audit


def annual_snapshot_matrix(frames):
    """Previous estimator (annual files treated as annual inspections), for comparison only."""
    n = np.zeros((5, 5))
    for a, b in zip(frames[:-1], frames[1:]):
        j = a.dropna(subset=["state"]).merge(b.dropna(subset=["state"]), on="bid", suffixes=("_a", "_b"))
        for i, k in zip(j.state_a.astype(int), j.state_b.astype(int)):
            if k <= i:
                n[i, k] += 1
    P = n / n.sum(1, keepdims=True); P[0] = [1, 0, 0, 0, 0]
    return P


# ----------------------------------------------------------------- likelihood
def generator(rates):
    q = np.zeros((5, 5))
    for s, r in enumerate(rates, 1):
        q[s, s - 1] = r; q[s, s] = -r
    return q


def interval_probabilities(rates, i, j, dt):
    """P(state j after dt | state i) under the one-step pure-deterioration process."""
    r = np.r_[0.0, rates]; p = np.ones(len(i))
    stay, one, multi = i == j, (i - j) == 1, (i - j) > 1
    p[stay] = np.exp(-r[i[stay]] * dt[stay])
    u, v, t = i[one], j[one], dt[one]                     # closed form, stable for equal rates
    p[one] = r[u] * t * np.exp(-r[u] * t) * exprel((r[u] - r[v]) * t)
    if multi.any():
        q = generator(rates)
        for tt in np.unique(dt[multi]):
            k = multi & (dt == tt); p[k] = expm(q * tt)[i[k], j[k]]
    if np.any(j > i):
        raise ValueError("improvement interval passed to the deterioration likelihood")
    return np.clip(p, 1e-300, 1.0)


FALLBACKS = []   # fits that needed the derivative-free fallback


def fit(pairs, weights=None):
    i = pairs.origin.to_numpy(int); j = pairs.destination.to_numpy(int); dt = pairs.years.to_numpy(float)
    w = np.ones(len(i)) if weights is None else weights
    nll = lambda lr: -np.sum(w * np.log(interval_probabilities(np.exp(lr), i, j, dt)))
    best = min((minimize(nll, np.log(s0), method="L-BFGS-B", bounds=[(-12, 1)] * 4,
                         options={"ftol": 1e-12, "gtol": 1e-6, "maxiter": 250})
                for s0 in ([.025, .015, .06, .10], [.05, .03, .10, .20])), key=lambda o: o.fun)
    if not best.success:
        # L-BFGS-B can end ABNORMAL at an optimum when the numerical gradient is too noisy to
        # progress (seen with some SciPy versions). Continue derivative-free from its best point
        # and accept only a converged result with an equal or better likelihood.
        nm = minimize(nll, best.x, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
        if not (nm.success and nm.fun <= best.fun + 1e-9):
            raise RuntimeError(f"deterioration fit failed: {best.message}; fallback: {nm.message}")
        FALLBACKS.append(float(nm.fun)); best = nm
    return np.exp(best.x), float(best.fun)


def write_matrix(P, path):
    pd.DataFrame(P, index=range(5), columns=range(5)).to_csv(path)


def main(n_boot: int = 512):
    frames = [read_year(y) for y in YEARS]
    pairs, audit = inspection_intervals(frames)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(C.OUT_DIR / "inspection_intervals.csv", index=False)
    keep = pairs.loc[pairs.included].reset_index(drop=True)
    rates, nll = fit(keep)
    P = expm(generator(rates)); write_matrix(P, C.MATRIX_CSV)
    pd.DataFrame(dict(state=range(1, 5), annual_rate=rates,
                      intervals=[int((keep.origin == s).sum()) for s in range(1, 5)],
                      observed_drops=[int(((keep.origin == s) & (keep.destination < s)).sum()) for s in range(1, 5)])
                 ).to_csv(C.OUT_DIR / "deterioration_estimates.csv", index=False)
    write_matrix(annual_snapshot_matrix(frames), C.OUT_DIR / "annual_snapshot_matrix.csv")
    rng = np.random.default_rng(C.SEED); ids = keep.bid.unique()
    index = keep.bid.map({b: k for k, b in enumerate(ids)}).to_numpy(); boots, boot_rates = [], []
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        r_b = fit(keep, mult[index])[0]; boot_rates.append(r_b)
        boots.append(expm(generator(r_b)))
        if (b + 1) % 64 == 0:
            print(f"  bridge bootstrap {b + 1}/{n_boot}", flush=True)
    np.save(C.ROOT / "transition_bootstrap.npy", np.asarray(boots))
    # Rate intervals and implied mean holding times (1/rate) of the pure-deterioration model; the passage
    # from the best state (NBI 8-9) to NBI 4 is the sum of the holding times in NBI 8-9, 7, and 5-6.
    br = np.asarray(boot_rates); labels = {1: "NBI 4", 2: "NBI 5-6", 3: "NBI 7", 4: "NBI 8-9"}; ht = []
    for k, s in enumerate(range(1, 5)):
        lo, hi = np.percentile(br[:, k], [5, 95])
        ht.append(dict(origin=labels[s], annual_rate=rates[k], rate_p05=lo, rate_p95=hi,
                       mean_holding_years=1 / rates[k], holding_p05=1 / hi, holding_p95=1 / lo))
    pas, pas_b = sum(1 / rates[k] for k in (1, 2, 3)), (1 / br[:, 1:4]).sum(axis=1)
    ht.append(dict(origin="passage NBI 8-9 to NBI 4", annual_rate="", rate_p05="", rate_p95="",
                   mean_holding_years=pas, holding_p05=np.percentile(pas_b, 5), holding_p95=np.percentile(pas_b, 95)))
    pd.DataFrame(ht).round(4).to_csv(C.OUT_DIR / "deterioration_holding_times.csv", index=False)
    print("  mean holding times (years): " + ", ".join(f"{h['origin']} {h['mean_holding_years']:.1f}" for h in ht))
    audit.update(rates=rates.tolist(), negative_log_likelihood=nll, bridges=int(keep.bid.nunique()),
                 median_interval_years=float(keep.years.median()), bootstrap_fits=n_boot,
                 fits_needing_derivative_free_fallback=len(FALLBACKS))
    json.dump(audit, open(C.OUT_DIR / "deterioration_fit.json", "w"), indent=1)
    print(json.dumps(audit, indent=1))
    print("annual matrix (states 0..4):\n", np.round(P, 5))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=512, help="bridge-cluster bootstrap fits for the Monte Carlo")
    main(ap.parse_args().boot)
