# AGENTS.md

## Cursor Cloud specific instructions

This repository is a Python research pipeline (no web app / server) for the paper
*"Extending Model Selection Criteria with Extrapolation and Sensitivity Penalties
for Symbolic Regression"*. See `README.md` for the science and per-file breakdown.

### Environment / how to run things

- All Python work runs inside the virtualenv at `~/.venv-hybrid`. Use
  `~/.venv-hybrid/bin/python` (or activate with `source ~/.venv-hybrid/bin/activate`).
  The system `python3` does NOT have the dependencies.
- Dependencies are installed by the startup update script (`pip install -r requirements.txt`).
  Julia is NOT installed by the update script: PySR pulls in `juliacall`/`juliapkg`,
  which download a private Julia + the `SymbolicRegression.jl` backend into
  `~/.venv-hybrid/julia_env` on the **first** `import pysr`. That first import is slow
  (~1-2 min) and needs network; later imports are fast. If the venv is ever rebuilt
  from scratch, expect the first `import pysr` to re-trigger this Julia install.

### Two flows

1. Experiment pipeline (core): `main.py` -> `kde_analysis` + `m1_evaluation` (uses PySR).
   Produces per-dataset CSVs (KDE splits, PySR Pareto-front equations, AIC/BIC/MDL +
   sensitivity/divergence metrics, hybrid scores) and plots.
2. Hybrid analysis (post-processing): scripts in the `hybrid code/` directory
   (note the space in the folder name) read aggregated per-dataset criteria CSVs and
   compute hybrid scores + Spearman correlations across 1331 lambda combinations.

### Non-obvious gotchas

- `main.py` has hard-coded absolute paths (`DATA_DIR`/`OUT_ROOT` under `/home/fitria_w/...`)
  and an empty `DATASETS` list, so `python main.py` does nothing useful as-is. To run the
  pipeline locally, either edit those constants (the commented-out relative paths
  `../../datasets` and `../../results/...` show the intended values) or call
  `kde_analysis()` + `evaluate_models()` directly. Datasets live in `datasets/*.npy`
  (20 PMLB datasets, shape `(n_samples, n_features + 1)`, target in the last column).
- PySR search size is configurable via env vars without editing code:
  `PY_SR_NITERATIONS`, `PY_SR_POPULATION_SIZE`, `PY_SR_BATCH_SIZE`, `PY_SR_MAXSIZE`,
  `PY_SR_PARALLELISM`, `PY_SR_PROCS` (see `model_trainer.py`). For a quick smoke test use
  small values, e.g. `PY_SR_NITERATIONS=20 PY_SR_POPULATION_SIZE=30`. Defaults
  (niterations=300, population_size=200) are tuned for real experiments and are slow.
- `m1_evaluation.py` does `from interval import interval, inf, imath`, but these symbols
  are never used. The real package (`pyinterval` -> `crlibm`) does not build on
  Python >= 3.12. The environment therefore provides a tiny `interval` shim module in the
  venv's site-packages so the import resolves; the update script recreates it if missing.
  If you ever need genuine interval arithmetic, install real `pyinterval` under Python <= 3.10.
- The `hybrid code/calculate_hybrid_scores.py` discovery logic expects, for each dataset,
  a `gp_model_selection_criteria_combined.csv` located in a folder that is a **sibling** of
  the `results_dir` argument (it reads `<results_dir>/../<dataset>/..._combined.csv` and
  writes `<results_dir>/hybrid_results/`). The committed `results/based_results/<dataset>/`
  folders only contain the non-combined `gp_model_selection_criteria.csv`, so you must
  assemble/rename a combined CSV into the expected layout before running the analysis.
- The analysis scripts are CPU/IO heavy: with default `--save-detailed-results` they emit
  thousands of per-lambda CSVs + PNG histograms per dataset and take many minutes even for a
  single dataset. Use `--no-detailed-results` and/or `--dataset <name>` to scope a smoke test.

### Lint / test / build

- There is no configured linter (no ruff/flake8/black config) and no automated test suite.
  Use `~/.venv-hybrid/bin/python -m py_compile <files>` as a basic compile/syntax check.
- There is no build step (pure Python).
