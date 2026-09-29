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
run_all.py  -  full reproduction from public data, in one command
=================================================================
    python run_all.py                  # download, verify, and run everything
    python run_all.py --offline        # only if FHWA is unreachable: archived files in data/raw (hash-verified)

Data stage (runs first; nothing is analyzed until it passes):
  0  download the six FHWA Rhode Island NBI files (2019-2024) and verify each
     against its reference SHA-256 hash; STOP if any file differs, unless
     --allow-revised is given (FHWA occasionally revises archived files)
  1  build the bridge inventory                    (build_portfolio.py)
  2  estimate the deterioration model from inspection intervals (build_matrix.py)
Analysis stage (uses only the files produced above):
  descriptives, objective screen, adaptation-cost knee, baselines, carbon cap,
  sensitivity, budget sweep, embodied-carbon boundary test, Monte Carlo,
  extended analyses, objective diagnostics, solver study, figures. Results go to ./outputs and ./figures,
  and a readable log to ./results_summary.txt.

Heavier stages can be sized with environment variables (defaults reproduce the paper):
  IAM_MC_DRAWS=512  IAM_GA_SEEDS=8  IAM_SCAL=1  IAM_KNEE_NPTS=41  IAM_HOLDOUT_CASES=40
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, io, contextlib
import config as C
import download_data, build_portfolio, build_matrix, diversion_evidence   # data modules depend only on config

MC_DRAWS = int(os.environ.get("IAM_MC_DRAWS") or 512)
GA_SEEDS = int(os.environ.get("IAM_GA_SEEDS") or 8)
DO_SCALABILITY = os.environ.get("IAM_SCAL", "1") == "1"
KNEE_NPTS = int(os.environ.get("IAM_KNEE_NPTS") or 41)



def data_stage(offline: bool, allow_revised: bool, n_boot: int, verify_only: bool = False):
    """Download (or locate) and verify the public data, then build the model inputs.
    Runs before any model module is imported, because engine.py and montecarlo.py read
    the transition matrix and bootstrap file at import time."""
    bar = "=" * 70
    print(bar); print("DATA 1  PUBLIC FHWA FILES: DOWNLOAD AND VERIFY (before any analysis)"); print(bar)
    if offline:
        ref = {e["year"]: e for e in json.load(open(C.ROOT / "data" / "raw" / "download_manifest.json"))}
        log = []
        for y, e in ref.items():
            p = C.ROOT / "data" / "raw" / f"RI{y}.csv"
            ok = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == e["sha256"]
            print(f"  {y}: local file {'matches reference' if ok else 'MISSING OR DIFFERENT'}")
            log.append(dict(year=y, matches_reference=ok))
    else:
        log = download_data.download()
    bad = [e["year"] for e in log if not e["matches_reference"]]
    if bad and not allow_revised:
        sys.exit(f"STOPPED before analysis: files for {bad} do not match the reference files. "
                 f"Re-run with --allow-revised to analyze the revised FHWA data deliberately.")
    if verify_only:
        return
    print("\n" + bar); print("DATA 2  BRIDGE INVENTORY (build_portfolio.py)"); print(bar)
    port = build_portfolio.reduce_to_model_fields(
        build_portfolio.load_dataframe(argparse.Namespace(local=None, year=2024)), state_fips=C.STATE_FIPS)
    port.to_csv(C.PORTFOLIO_CSV, index=False)
    print(f"  {len(port)} bridges retained; condition states 0..4:",
          port["state0"].value_counts().sort_index().to_dict())
    print("\n" + bar); print("DATA 3  DETERIORATION MODEL FROM INSPECTION INTERVALS (build_matrix.py)"); print(bar)
    build_matrix.main(n_boot)
    print("\n" + bar); print("DATA 4  DIVERTED-TRAFFIC SCHEDULE FROM NBI ITEMS 41 AND 109 (diversion_evidence.py)"); print(bar)
    diversion_evidence.main()


def knee_curve(prob):
    """Adaptation-cost premium on a refined grid, with the knee located by
    BOTH standard definitions (max distance from the chord joining the
    endpoints, and max curvature) rather than read by eye off a handful of
    pre-chosen points; the two are reported so their agreement (or lack of
    it) is visible rather than assumed."""
    import numpy as np
    V, B, tot = prob.V, prob.budget_M, prob.total_scour
    _, co = OPT.cost_optimal(V, B); C0 = co["cost"]
    fracs = np.linspace(0, 1, KNEE_NPTS)
    rows = []
    for f in fracs:
        r = OPT.min_cost_with_protection(V, f * tot, B)
        if r is None:
            continue
        achieved = r[1]["protected"] / tot
        rows.append((achieved, r[1]["cost"], (r[1]["cost"] - C0) / C0 * 100.0, f))
    f_arr = np.array([r[0] for r in rows]); p_arr = np.array([r[2] for r in rows])

    i_chord, i_curv = extended_analyses.knee_points(f_arr, p_arr)   # shared knee math

    print(f"ADAPTATION-COST KNEE  (refined grid, {len(rows)} feasible points of {KNEE_NPTS} requested)")
    print("  protect%   cost_M   premium%")
    for pct in [0, 25, 50, 75, 90, 100]:
        i = int(np.argmin(np.abs(f_arr * 100 - pct)))
        print(f"   {f_arr[i]*100:5.1f}%  ${rows[i][1]:7.0f}M  +{rows[i][2]:5.2f}%")
    print(f"  max-chord-distance knee : protect {f_arr[i_chord]*100:.1f}%  premium +{p_arr[i_chord]:.2f}%")
    print(f"  max-curvature knee      : protect {f_arr[i_curv]*100:.1f}%  premium +{p_arr[i_curv]:.2f}%")

    with open(C.OUT_DIR / "knee_curve.csv", "w") as f:
        f.write("protected_pct,cost_M,premium_pct,required_pct\n")
        for achieved, cost, prem, req in rows:
            f.write(f"{achieved*100:.2f},{cost:.1f},{prem:.2f},{req*100:.2f}\n")
    with open(C.OUT_DIR / "knee_estimate.csv", "w") as f:
        f.write("definition,protected_pct,premium_pct\n")
        f.write(f"max_chord_distance,{f_arr[i_chord]*100:.2f},{p_arr[i_chord]:.2f}\n")
        f.write(f"max_curvature,{f_arr[i_curv]*100:.2f},{p_arr[i_curv]:.2f}\n")
    # Efficiency check of the sampled frontier: duplicates and points dominated by another sampled point
    pts = [(a, c) for a, c, _, _ in rows]
    dup = len(pts) - len(set(pts))
    dom = sum(any(a2 >= a and c2 <= c and (a2 > a or c2 < c) for a2, c2 in pts) for a, c in pts)
    with open(C.OUT_DIR / "knee_frontier_check.csv", "w") as f:
        f.write("sampled_points,duplicate_points,dominated_points\n"); f.write(f"{len(pts)},{dup},{dom}\n")
    print(f"  frontier check: {len(pts)} sampled points, {dup} duplicates, {dom} dominated by another sampled point")
    print(f"  saved -> {C.OUT_DIR / 'knee_curve.csv'}, {C.OUT_DIR / 'knee_estimate.csv'}, and {C.OUT_DIR / 'knee_frontier_check.csv'}")
    return rows, (f_arr[i_chord], p_arr[i_chord]), (f_arr[i_curv], p_arr[i_curv])



class _Tee(io.TextIOBase):
    """Write to the console (live) and to a buffer at the same time."""
    def __init__(self, live, buf):
        self.live, self.buf = live, buf
    def write(self, s):
        self.live.write(s); self.buf.write(s); self.live.flush(); return len(s)
    def flush(self):
        self.live.flush()


PROGRESS = C.OUT_DIR / ".progress.json"
PARTIALS = [C.OUT_DIR / ".mc_partial.pkl", C.OUT_DIR / "rule_holdout_cases_partial.csv"]


def _signature():
    """Fingerprint of the computational code and run settings: a checkpoint is reused only when the
    same code with the same settings produced it. paper_figures.py only draws figures from the outputs
    and changes no computed result, so it is excluded; figures can be redrawn alone with it."""
    h = hashlib.sha256()
    for f in sorted(C.ROOT.glob("*.py")):
        if f.name == "paper_figures.py":
            continue
        h.update(f.name.encode()); h.update(f.read_bytes())
    for k in ("IAM_MC_DRAWS", "IAM_GA_SEEDS", "IAM_SCAL", "IAM_KNEE_NPTS", "IAM_HOLDOUT_CASES", "IAM_MIP_GAP"):
        h.update(f"{k}={os.environ.get(k, '')}".encode())
    return h.hexdigest()


def _load_progress(fresh):
    sig = _signature()
    try:
        p = json.load(open(PROGRESS))
    except (FileNotFoundError, ValueError):
        p = {}
    if fresh or p.get("signature") != sig:
        if p:
            print("Checkpoints from a different code version or settings (or --fresh): starting from the beginning.")
        for f in PARTIALS:
            f.unlink(missing_ok=True)
        p = {"signature": sig, "steps": {}}
    elif p["steps"]:
        print(f"Resuming: {len(p['steps'])} step(s) already completed with this code will be skipped.")
    return p


def _save_progress(p):
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS.with_suffix(".tmp"); json.dump(p, open(tmp, "w")); tmp.replace(PROGRESS)


def _record_environment():
    """Write outputs/environment.json: interpreter, platform, and library versions of this run."""
    import platform, datetime, importlib
    env = dict(run_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               python=sys.version.split()[0], platform=platform.platform(), code_signature=_signature()[:16])
    for m in ("numpy", "pandas", "scipy", "matplotlib", "pulp", "contextily", "pyproj"):
        try:
            env[m] = importlib.import_module(m).__version__
        except Exception:
            env[m] = "not installed"
    try:
        import pulp
        env["cbc_available"] = bool(pulp.PULP_CBC_CMD(msg=False).available())
    except Exception:
        env["cbc_available"] = False
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(env, open(C.OUT_DIR / "environment.json", "w"), indent=1)
    print("Environment recorded in outputs/environment.json:", ", ".join(f"{k} {v}" for k, v in env.items() if k in ("python", "numpy", "pandas", "scipy", "pulp")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="only if FHWA is unreachable: use archived files in data/raw (hash-verified first)")
    ap.add_argument("--allow-revised", action="store_true", help="proceed even if FHWA files differ from the reference")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints and run every step")
    args = ap.parse_args()
    t0 = time.time(); bar = "=" * 70; summary = io.StringIO()
    progress = _load_progress(args.fresh)
    import about
    print(about.banner()); print(); summary.write(about.banner() + "\n\n")
    _record_environment()

    def step(key, title, fn):
        """Run one step with its output teed to the summary; skip it if already completed."""
        print("\n" + bar); print(title); print(bar); summary.write("\n" + title + "\n")
        done = progress["steps"].get(key)
        if done is not None:
            print("  (completed earlier with this code and settings; skipped)")
            summary.write(done + "\n"); return
        buf = io.StringIO()
        with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
            fn()
        summary.write(buf.getvalue() + "\n")
        progress["steps"][key] = buf.getvalue(); _save_progress(progress)

    # ---- data stage (before any model module is imported)
    # Every run downloads the public FHWA files and rebuilds every model input from them; nothing stored
    # from an earlier run is reused (--offline, used only on explicit request, reads archived copies instead).
    dbuf = io.StringIO()                              # the download and verification record goes into the summary
    with contextlib.redirect_stdout(_Tee(sys.stdout, dbuf)):
        data_stage(args.offline, args.allow_revised, MC_DRAWS)
    summary.write(dbuf.getvalue() + "\n")

    # ---- model modules are imported only now, after their input files exist
    global P, OPT, descriptives, screening, baselines, carbon_cap, montecarlo, solver_study, figures, \
        extended_analyses, budget_sweep, embodied_threshold_sweep
    import problem as P, optimizer as OPT
    import descriptives, screening, baselines, carbon_cap, montecarlo, solver_study, figures, extended_analyses
    import budget_sweep, embodied_threshold_sweep, objective_diagnostics, runpy
    print(bar); print("RESILIENT INFRASTRUCTURE ASSET MANAGEMENT  -  full reproduction"); print(bar)
    prob = P.build()
    hdr = (f"Inventory: {len(prob.candidates)} candidates | budget ${prob.budget_M:.0f}M | "
           f"total scour exposure {prob.total_scour:.1f} | analysis period {C.ANALYSIS_YEARS} yr")
    print(hdr); summary.write(hdr + "\n")

    def sensitivity():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runpy.run_module("sensitivity", run_name="__main__")
        print(buf.getvalue(), end="")

    steps = [
        ("0", "0  INVENTORY DESCRIPTIVES", descriptives.run),
        ("1", "1  OBJECTIVE SCREEN", lambda: screening.run(prob)),
        ("2", "2  ADAPTATION-COST KNEE", lambda: knee_curve(prob)),
        ("3", "3  SINGLE-CRITERION BASELINES", lambda: baselines.run(prob)),
        ("4", "4  NET-ZERO / CARBON-CAP TEST", lambda: carbon_cap.run(prob)),
        ("5", "5  SENSITIVITY", sensitivity),
        ("5b", "5b  BUDGET-RANGE REDUNDANCY SWEEP", budget_sweep.main),
        ("5c", "5c  EMBODIED-CARBON THRESHOLD (two-sided screen validation)", embodied_threshold_sweep.main),
        ("6", f"6  PARAMETER MONTE CARLO (N={MC_DRAWS})", lambda: montecarlo.main(MC_DRAWS)),
        ("6b", "6b EXTENDED ANALYSES (protection x budget regret, carbon price, diversion rules, ...)",
         lambda: extended_analyses.run_all(prob)),
        ("6c", "6c OBJECTIVE DIAGNOSTICS (grid, mechanism, diversion sweep, held-out rule test)", objective_diagnostics.main),
        ("7", f"7  SOLVER STUDY ({GA_SEEDS} seeds)", lambda: solver_study.main(GA_SEEDS, DO_SCALABILITY)),
        ("8", "8  FIGURES AND TABLE (diagnostic suite, paper and supplement figures, results summary table)", lambda: (figures.run_all(), __import__("paper_tables").main(), __import__("paper_figures").main())),
    ]
    for key, title, fn in steps:
        step(key, title, fn)

    footer = (f"\nDONE in {time.time()-t0:.0f}s. Outputs in {C.OUT_DIR}, figures in {C.FIG_DIR}, "
              f"summary in {C.ROOT/'results_summary.txt'}.")
    print(footer); summary.write(footer + "\n")
    with open(C.ROOT / "results_summary.txt", "w") as f:
        f.write(summary.getvalue())
    PROGRESS.unlink(missing_ok=True)   # run complete: the next run starts from the beginning


if __name__ == "__main__":
    main()
