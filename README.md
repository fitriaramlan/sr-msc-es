# Extending Model Selection Criteria with Extrapolation and Sensitivity Penalties for Symbolic Regression

Code for the paper:

> **Extending Model Selection Criteria with Extrapolation and Sensitivity Penalties for Symbolic Regression**  
> Fitria Wulandari Ramlan, Colm O'Riordan, James McDermott  
> EuroGP 2026, Lecture Notes in Computer Science, vol 16521, pp. 189–204  
> DOI: [10.1007/978-3-032-23005-8_12](https://doi.org/10.1007/978-3-032-23005-8_12)

---

## What this paper is about

Model selection criteria like AIC, BIC, and MDL pick symbolic regression models based on training error and complexity. But they do not say anything about how a model behaves on data it has never seen, particularly in regions with little or no training data (extrapolation).

We extend these criteria by adding three penalty terms:
1. **Extrapolation divergence**: how much the model's prediction range shifts when moving from dense to sparse data regions
2. **Interpolation sensitivity**: how much predictions change when inputs are slightly perturbed in dense regions
3. **Extrapolation sensitivity**: the same, but in sparse regions

We test these extended (hybrid) criteria across 20 regression datasets from the [PMLB repository](https://github.com/EpistasisLab/pmlb), using [PySR](https://github.com/MilesCranmer/PySR) to see whether they pick better-generalising models than the standard criteria alone.

---

## Code structure

```
├── main.py                              # Runs the experiments
├── kde_analysis.py                      # Splits data into interpolation / extrapolation regions
├── data_splitter.py                     # Builds train / test splits from those regions
├── model_trainer.py                     # Sets up PySR for symbolic regression
├── m1_evaluation.py                     # Evaluates models and computes all metrics
│
└── hybrid_code/
    ├── run_all_analyses.py              # Runs all analysis scripts in order
    ├── calculate_hybrid_scores.py       # Computes hybrid scores for all lambda combinations
    ├── generate_summary_statistics.py   # Summarises results across datasets
    ├── compare_correlations_baseline.py # Compares each lambda setting to the (0,0,0) baseline
    ├── calculate_best_per_run.py        # Finds the best metric and lambda per run
    └── aggregate_extrapolation_performance.py  # Per-dataset summary tables
```

---

## What each file does

### Experiment scripts

**`main.py`**  
The entry point. Runs the pipeline for each dataset, with support for parallel execution across multiple CPU cores. You can stop and resume using `--start-run` and `--end-run`.

**`kde_analysis.py`**  
Uses kernel density estimation to label each data point as either inside (high-density, interpolation) or outside (low-density, extrapolation) the training distribution. Tries several bandwidth values and saves plots and CSVs for each.

**`data_splitter.py`**  
Reads the KDE output and creates the actual train/test splits. The outside region becomes the extrapolation test set. Supports holdout splits and K-fold cross-validation.

**`model_trainer.py`**  
Configures PySR with protected versions of division, log, and square root to avoid numerical errors during search.

**`m1_evaluation.py`**  
For each equation PySR finds, this script computes RMSE and MSE on all splits, AIC, BIC, MDL, equation complexity, sensitivity under input perturbations (single-variable and multi-variable), extrapolation divergence, and Spearman correlations between each criterion and test error.

---

### Analysis scripts (`hybrid_code/`)

**`run_all_analyses.py`**  
Runs `calculate_hybrid_scores.py`, then `generate_summary_statistics.py`, and optionally `compare_correlations_baseline.py`, in that order.

**`calculate_hybrid_scores.py`**  
Tries all combinations of (λ₁, λ₂, λ₃) in steps of 0.1, giving 1,331 combinations in total. For each, it computes:

```
Hybrid score = base_metric + λ₁ × extrap_div + λ₂ × sens_interp + λ₃ × sens_extrap
```

then measures how well that score ranks models by Spearman correlation with test MSE.

**`generate_summary_statistics.py`**  
Reads the per-dataset results and produces cross-dataset summaries: clamping thresholds used, normalisation checks, and which hybrid settings beat the standard criteria.

**`compare_correlations_baseline.py`**  
Takes the (0, 0, 0) setting (i.e. no penalty, just the base criterion) as a baseline and measures how much each other setting improves or worsens the Spearman correlation. Ranks all settings and optionally produces plots.

**`calculate_best_per_run.py`**  
For each dataset and each run, picks the (metric, lambda combination) with the highest Spearman correlation. Saves results in wide format, one row per run with separate columns for SVP/MVP and interpolation/extrapolation targets.

**`aggregate_extrapolation_performance.py`**  
Summarises results at the dataset level. Categorises each dataset by how much the best hybrid setting improves over the baseline.

---

## Requirements

- Python ≥ 3.10
- [PySR](https://github.com/MilesCranmer/PySR), requires Julia
- scikit-learn, NumPy, pandas, matplotlib, seaborn
- JAX (CPU only)
- SymPy, SciPy
- `interval`

```bash
pip install pysr scikit-learn numpy pandas matplotlib seaborn jax sympy scipy
```

See the [PySR installation guide](https://astroautomata.com/PySR/api/) for Julia setup.

---

## Running the code

### Run experiments

Edit the top of `main.py` to set:
- `DATASETS`, which `.npy` files to run
- `DATA_DIR` / `OUT_ROOT`, where data lives and where results go
- `NUM_RUNS`, `BANDWIDTHS`, `USE_CV`, experimental settings


---

## Output files

**Per dataset** (in `OUT_ROOT/<dataset_name>/`):

| File | Contents |
|------|----------|
| `inside_points_bw_<bw>.csv` | Interpolation region samples |
| `outside_points_bw_<bw>.csv` | Extrapolation region samples |
| `plot_bw_<bw>.png/pdf` | KDE scatter plots |
| `gp_model_selection_criteria.csv` | AIC, BIC, MDL, sensitivity, all metrics |
| `hybrid_model_scores_sv.csv` | Hybrid scores using single-variable perturbation |
| `hybrid_model_scores_mv.csv` | Hybrid scores using multi-variable perturbation |
| `pysr_equations_all_runs.csv` | Every equation found across all runs |

**Hybrid analysis** (in `hybrid_results/`):

| File | Contents |
|------|----------|
| `correlations_<method>_<weight>.csv` | Spearman ρ per run and lambda combination |
| `best_hybrid_overall*.csv` | Best lambda settings found |
| `dataset_*_performance_summary*.csv` | Per-dataset results |
| `best_per_run/best_per_run_all_datasets.csv` | Best setting per run across all datasets |

---

## Datasets

All 20 datasets come from [PMLB](https://github.com/EpistasisLab/pmlb), stored as `.npy` arrays of shape `(n_samples, n_features + 1)` with the target in the last column:

`1096_FacultySalaries`, `192_vineyard`, `228_elusage`, `542_pollution`, `523_analcatdata_neavote`, `678_visualizing_environmental`, `229_pwLinear`, `561_cpu`, `712_chscase_geyser1`, `605_fri_c2_250_25`, `519_vinnie`, `556_analcatdata_apnea2`, `557_analcatdata_apnea1`, `584_fri_c4_500_25`, `637_fri_c1_500_50`, `589_fri_c2_1000_25`, `1028_SWD`, `522_pm10`, `227_cpu_small`, `529_pollen`

---

## Citation

```bibtex
@inproceedings{ramlan2026extending,
  author    = {Ramlan, Fitria Wulandari and O'Riordan, Colm and McDermott, James},
  title     = {Extending Model Selection Criteria with Extrapolation and Sensitivity Penalties for Symbolic Regression},
  booktitle = {Genetic Programming. EuroGP 2026},
  series    = {Lecture Notes in Computer Science},
  volume    = {16521},
  pages     = {189--204},
  publisher = {Springer, Cham},
  year      = {2026},
  doi       = {10.1007/978-3-032-23005-8_12}
}
```

---

## Acknowledgement

This work was supported by Taighde Éireann, Research Ireland, Grant No. 18/CRT/6223.
