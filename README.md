A Multi-Objective Optimization Framework for Resilient Infrastructure Asset Management<br>
Code, data, and supplemental materials<br>
Fredy Díaz-Durán · ORCID [0000-0001-5344-5466](https://orcid.org/0000-0001-5344-5466) · [diazdura@ualberta.ca](mailto:diazdura@ualberta.ca) · [fadiazdu@uwaterloo.ca](mailto:fadiazdu@uwaterloo.ca)<br>
Department of Civil and Environmental Engineering, University of Alberta, Edmonton, AB, Canada<br>
Department of Civil and Environmental Engineering, University of Waterloo, Waterloo, ON, Canada<br>
DOI: [10.5281/zenodo.22973335](https://doi.org/10.5281/zenodo.22973335)<br>
Licenses. Code: MIT (`LICENSE`). Documents, figures, and outputs: CC BY 4.0 (`LICENSE-CC-BY-4.0.md`). NBI files: public domain (`DATA_NOTICE.md`).

This record reproduces every number, table, and figure of the paper and its supplement from the
public FHWA National Bridge Inventory (NBI) files for Rhode Island, 2019-2024.

## Contents

| Path | Contents |
|---|---|
| `run_all.py` | Runs the complete analysis: data verification, model inputs, all analyses, all figures |
| `run_all.ipynb` | The same run as a notebook, for Google Colab or a local Jupyter session |
| `*.py` | Model and analysis modules (see Pipeline below); `about.py` holds the metadata |
| `data/raw/` | SHA-256 manifest of the six NBI files; every run downloads the files from FHWA into this folder (the Zenodo record also archives them for `--offline` use) |
| `outputs/` | Results of the reported run (CSV/JSON), including `environment.json` with the software versions used |
| `figures/` | Figures 1-7 of the paper and Figures S1-S4 of the supplement (PNG, 600 dpi, and PDF) |
| `supplement/` | The supplementary document (PDF) |
| `requirements.txt` | Python dependencies |

## Run

**On a computer** (Python 3.10 or later):

    pip install -r requirements.txt
    python run_all.py                  # download the NBI files from FHWA, verify them, build and run everything
    python run_all.py --offline        # only if FHWA is unreachable: archived files in data/raw (hash-verified)
    python run_all.py --fresh          # ignore checkpoints and rerun every step
    python paper_figures.py            # redraw the figures only, from existing outputs

**In Google Colab:** upload this folder to Google Drive, open `run_all.ipynb`, set `PROJECT_DIR`
in the first cell to the folder's location, and run all cells. The same notebook also runs in a
local Jupyter session, where it uses the folder it is opened from.

A full run takes about 60-90 minutes (the Monte Carlo analysis is the longest step). Every run downloads the NBI files and rebuilds every model input from them. If a run is
interrupted, run the same command again: it downloads and rebuilds the inputs, then continues after the
last completed analysis (checkpointed with a fingerprint of the code and settings); the Monte Carlo and
held-out analyses resume inside their loops. A completed run clears its checkpoints, so the next run
starts from the beginning. A resumed run gives results identical to an uninterrupted one.

**Zenodo archive.** After a complete run, the last notebook cell (or the same code run from a Python session) packages
the code, the six NBI files that run downloaded (checked against the reference hashes), the outputs, the figures and
`results_summary.txt` into `ResilientIAM_Framework_v1.0.0_Zenodo.zip`.

## Reproduction guide

| Result in the paper or supplement | Output file(s) |
|---|---|
| Reference programs, excess emissions, alignment, NHS condition shares | `screening_results.csv` |
| Excess emissions and budget use on a fine budget grid; unconstrained capital (Fig. 4) | `budget_use_sweep.csv`, `budget_unconstrained.csv`, `budget_sweep.csv`, `mechanism.csv` |
| Results summary table (Table 2) and thresholds | `table_results_summary.csv`, `table_results_thresholds.csv` |
| Diversion and discount-rate grids by analysis period (Fig. 5; Table S4) | `diversion_horizon_grid.csv`, `discount_horizon_grid.csv` |
| Continuous diversion sweep (Fig. S2) | `diversion_sweep.csv` |
| Cost-adaptation trade-off and knee (Fig. 6; Fig. S1) | `knee_curve.csv`, `knee_estimate.csv`, `knee_stability.csv`, `knee_composition.csv` |
| Excess emissions over protection targets, budget x protection, carbon price | `frontier_regret.csv`, `carbon_regret_grid.csv`, `carbon_price_sweep.csv` |
| Single-rule comparisons | `baselines_results.csv` |
| Deterministic sensitivity cases | `sensitivity_results.csv`, `cm_effectiveness.csv`, `mapping_alignment.csv`, `deterioration_rates.csv`, `diversion_rules.csv` |
| Time value by vehicle class; lowest-state maintenance (with the cost program's capital by budget) | `valuation_sensitivity.csv` |
| Deterioration rate intervals and implied holding times (Table S1) | `deterioration_holding_times.csv` |
| Efficiency check of the sampled cost-adaptation frontier | `knee_frontier_check.csv` |
| Monte Carlo analysis (Fig. S3; Tables S6-S7) | `montecarlo_results.csv` |
| Embodied-carbon boundary test (Fig. 7) | `embodied_threshold.csv` |
| Held-out decision-rule test | `rule_holdout_cases.csv`, `rule_holdout_summary.csv` |
| Solver benchmark (Fig. S4; Table S9) | `solver_study.csv`, `solver_convergence.csv`, `solver_scalability.csv` |
| Deterioration model (Table S1) | `deterioration_fit.json`, `transition_matrix.csv` |
| Diversion evidence (Table 3) | `diversion_evidence.csv` |
| Inventory counts | `inventory_descriptives.csv` |
| Complete console record of the run | `results_summary.txt` |

## Figure files

`figures/` holds Figures 1-7 of the paper (`fig1_framework`, `fig2_study_area`, `fig3_screen`, `fig4_budget`,
`fig5_conventions`, `fig6_knee`, `fig7_embodied`) and Figures S1-S4 of the supplemental materials
(`figS1_program_map`, `figS2_diversion_sweep`, `figS3_montecarlo`, `figS4_solver`), each as 600-dpi PNG and PDF.
The maps (Figs. 2 and S1) use the Esri World Imagery basemap (Esri and its data providers), fetched at run time;
without internet access they are drawn on a plain background.

## Pipeline

| Step | Module | What it does |
|---|---|---|
| DATA 1 | `download_data.py` | Downloads the Rhode Island NBI files 2019-2024 from fhwa.dot.gov and checks each against its SHA-256 hash in `data/raw/download_manifest.json`. The run stops before any analysis if a file differs (`--allow-revised` proceeds deliberately). |
| DATA 2 | `build_portfolio.py` | 658-bridge inventory from the 2024 file (`nbi_real_portfolio.csv`). |
| DATA 3 | `build_matrix.py` | Deterioration model estimated from inspection intervals (`transition_matrix.csv`) and 512 bridge-cluster bootstrap matrices (`transition_bootstrap.npy`). |
| DATA 4 | `diversion_evidence.py` | Diverted-traffic fraction by condition state from NBI Items 41 (closed/posted) and 109 (truck share) (`diversion_evidence.json`). |
| 0-6b | `descriptives`, `screening`, `run_all.knee_curve`, `baselines`, `carbon_cap`, `sensitivity`, `budget_sweep`, `embodied_threshold_sweep`, `montecarlo`, `extended_analyses` | Objective screen, adaptation-cost frontier, sensitivity, budget sweep, embodied-carbon test, Monte Carlo, extended analyses. |
| 6c | `objective_diagnostics.py` | Diversion-by-analysis-period and discount-rate-by-analysis-period grids (central result); mechanism (budget use and where the programs differ); continuous diversion sweep; held-out test of correlation-only, loss-only, and combined decision rules. |
| 7-8 | `solver_study`, `figures`, `paper_figures` | Solver benchmark; diagnostic figures; the paper's Figures 1-7 and the supplement's Figures S1-S4 (`python paper_figures.py` redraws them alone from existing outputs). |

Results: `outputs/` (CSV), `figures/` (PNG and PDF), `results_summary.txt` (log).

## Reference formulation

| Element | Reference | Basis | Alternatives examined |
|---|---|---|---|
| Diverted traffic by state (NBI 0-3 / 4 / 5-6) | 0.227 / 0.042 / 0.010 | Closed share + posted share x median truck share, pooled NBI records 2019-2024 | 2024 only; the original assumption 1.0 / 0.25 / 0.05; a continuous range |
| Benefit-accounting period | 35 years | FHWA LCCA policy minimum (1996), applied to bridges by analogy | 10-40 years (grid) |
| Truck share of traffic | Each bridge's NBI Item 109 value (median 7%) | Reported by the inventory; one value for road-user cost, emissions, and diversion | x0.8-1.2 in the Monte Carlo |
| Real discount rate | 2.3% | OMB Circular A-94 Appendix C (2025): 30-year real Treasury rate for constant-dollar cost-effectiveness analysis (retained by OMB M-25-23) | 2.3-7% (grid; 7% is the A-94 benefit-cost base case); not sampled in the Monte Carlo |
| NHS minimum condition | Structurally deficient NHS deck area <= 10% in years 1-3 | 23 CFR 490.411; 23 U.S.C. 119(f)(2) three-year period | - |
| Capital budget | $509.9M (10 years of Bridge Formula Program funding) | IIJA apportionment | 0.25-2 x |

`config.py` holds these settings (`REFERENCE_DIVERSION`, `ANALYSIS_YEARS`, `NHS_SD_LIMIT`,
`NHS_WINDOW_YEARS`, `PROGRAM_YEARS`). The budget period and the analysis period are separate.

## Corrections from the technical audit (September 2026)

| Correction | Where |
|---|---|
| NHS membership from NBI Item 104 instead of a functional-class proxy (124 bridges had the wrong unit cost) | `build_portfolio.py` |
| Deterioration estimated from actual inspection intervals (Item 90 dates; repeated, conflicting, and improvement records handled) instead of treating annual files as annual inspections | `build_matrix.py` |
| Two undocumented user-cost terms removed (a truck charge that applied the truck share twice; a slowdown based on average span) | `engine.py` |
| Monte Carlo: fixed candidate population; bridge-cluster bootstrap matrices; active detour multiplier; a failed draw stops the run | `montecarlo.py`, `problem.py` |
| Rank correlations preserve exact ties | `screening.rank_corr` |
| Only certified optima accepted (time limits or unproven solutions stop the run) | `optimizer.py` |
| Deck area unrounded; malformed input lines raise an error | `build_portfolio.py` |
| Diversion split into two streams: closures divert all traffic (bridge car/truck mix); load postings (Item 41 code P only) divert trucks only, at the truck emission factor and truck operating cost | `diversion_evidence.py`, `engine.py`, `economics.py` |
| Truck share from each bridge's NBI Item 109 in road-user cost and emissions, replacing a specified 12% (the 90th percentile of reported values); a missing value uses the statewide median, not zero | `engine.truck_share`, `economics.py`, `build_portfolio.py` |

## Solver settings

CBC through PuLP. Every program is proven optimal within a relative gap of 1e-6 (about
$500 on a $500M program, far below reported precision); ties are broken by a second
stage within max(stated tolerance, 1e-7 x optimum), warm-started from the first stage.
`IAM_MIP_GAP` and `IAM_SOLVER_TIME` (default 600 s per solve) adjust these.
`IAM_MC_DRAWS`, `IAM_GA_SEEDS`, `IAM_SCAL`, `IAM_KNEE_NPTS`, and `IAM_HOLDOUT_CASES` size the
heavier steps; the defaults reproduce the paper.
