"""
For each dataset, find the best (metric, lambda combination) per run, judged by
Spearman correlation between the hybrid score and the corresponding test MSE.

Whereas calculate_train_metrics_averages.py averages across 30 runs to produce
one row per lambda combination, this script keeps everything at the run level:
each (dataset, run, test target) gets its own best metric and lambda triple.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product


DATASET_FOLDERS = [
    "1096_FacultySalaries",
    "192_vineyard",
    "228_elusage",
    "542_pollution",
    "523_analcatdata_neavote",
    "678_visualizing_environmental",
    "229_pwLinear",
    "561_cpu",
    "712_chscase_geyser1",
    "605_fri_c2_250_25",
    "519_vinnie",
    "556_analcatdata_apnea2",
    "557_analcatdata_apnea1",
    "584_fri_c4_500_25",
    "637_fri_c1_500_50",
    "589_fri_c2_1000_25",
    "1028_SWD",
    "522_pm10",
    "227_cpu_small",
    "529_pollen",
]

BASE_METRICS = [
    "MSE_Train",
    "AIC_Train",
    "BIC_Train",
    "MDL_Train",
    "PSM_Train",
]

LAMBDA_VALUES = [round(i * 0.1, 1) for i in range(11)]


def generate_lambda_combinations():
    """Build all 11 x 11 x 11 = 1331 lambda combinations."""
    return list(product(LAMBDA_VALUES, LAMBDA_VALUES, LAMBDA_VALUES))


def format_weight_string(lambda1, lambda2, lambda3):
    """Lambda values -> 'w<l1>_<l2>_<l3>'."""
    return f"w{lambda1}_{lambda2}_{lambda3}"


def calculate_hybrid_score(base_metric, extrap_div, sens_interp, sens_extrap,
                           lambda1=0.0, lambda2=0.0, lambda3=0.0):
    """Hybrid = base + l1*extrap_div + l2*sens_interp + l3*sens_extrap."""
    return base_metric + lambda1 * extrap_div + lambda2 * sens_interp + lambda3 * sens_extrap


def calculate_spearman_correlations_per_run(df, method, lambda1, lambda2, lambda3):
    """Return a DataFrame of per-run Spearman correlations between hybrid score and test MSE."""
    sens_interp_col = f"Sensitivity_Interpolation_{method}"
    sens_extrap_col = f"Sensitivity_Extrapolation_{method}"
    extrap_div_col = "Extrapolation_Divergence"

    required_test_cols = ['MSE_Test_Interpolation', 'MSE_Test_Extrapolation', 'Run']
    if (sens_interp_col not in df.columns or sens_extrap_col not in df.columns
            or extrap_div_col not in df.columns
            or not all(col in df.columns for col in required_test_cols)):
        return None

    hybrid_scores_dict = {}
    for metric in BASE_METRICS:
        if metric in df.columns:
            hybrid_scores_dict[f"Hybrid_{metric}"] = calculate_hybrid_score(
                df[metric],
                df[extrap_div_col],
                df[sens_interp_col],
                df[sens_extrap_col],
                lambda1, lambda2, lambda3,
            )

    hybrid_df = pd.DataFrame(hybrid_scores_dict)
    hybrid_df['Run'] = df['Run'].values
    hybrid_df['MSE_Test_Interpolation'] = df['MSE_Test_Interpolation'].values
    hybrid_df['MSE_Test_Extrapolation'] = df['MSE_Test_Extrapolation'].values

    correlations_per_run = []
    for run_id in df['Run'].unique():
        run_data = hybrid_df[hybrid_df['Run'] == run_id]
        if len(run_data) < 2:
            continue

        run_corrs = {'Run': run_id}
        for hybrid_col in hybrid_scores_dict.keys():
            corr_interp = run_data[hybrid_col].corr(
                run_data['MSE_Test_Interpolation'], method='spearman'
            )
            run_corrs[f"{hybrid_col}_vs_Interpolation"] = (
                corr_interp if not np.isnan(corr_interp) else np.nan
            )
            corr_extrap = run_data[hybrid_col].corr(
                run_data['MSE_Test_Extrapolation'], method='spearman'
            )
            run_corrs[f"{hybrid_col}_vs_Extrapolation"] = (
                corr_extrap if not np.isnan(corr_extrap) else np.nan
            )
        correlations_per_run.append(run_corrs)

    if not correlations_per_run:
        return None
    return pd.DataFrame(correlations_per_run)


def find_best_per_run_for_dataset(df, dataset_name, method):
    """Per-run best (metric, lambda combination) for interpolation and extrapolation.

    The winner is the entry with the highest Spearman correlation between
    hybrid score and the corresponding test MSE.
    """
    lambda_combinations = generate_lambda_combinations()
    all_run_correlations = {}

    for lambda1, lambda2, lambda3 in lambda_combinations:
        corr_df = calculate_spearman_correlations_per_run(df, method, lambda1, lambda2, lambda3)
        if corr_df is None or corr_df.empty:
            continue

        weight_str = format_weight_string(lambda1, lambda2, lambda3)

        for _, row in corr_df.iterrows():
            run_id = row['Run']
            if run_id not in all_run_correlations:
                all_run_correlations[run_id] = []

            for metric in BASE_METRICS:
                hybrid_col = f"Hybrid_{metric}"
                for test_target, col_suffix in [
                    ('Interpolation', '_vs_Interpolation'),
                    ('Extrapolation', '_vs_Extrapolation'),
                ]:
                    col = f"{hybrid_col}{col_suffix}"
                    if col in row:
                        spearman = row[col]
                        if pd.notna(spearman):
                            all_run_correlations[run_id].append({
                                'lambda1': lambda1,
                                'lambda2': lambda2,
                                'lambda3': lambda3,
                                'weight': weight_str,
                                'metric': metric,
                                'test_target': test_target,
                                'spearman': spearman,
                            })

    # Pick the (run, test_target) winner with the largest Spearman value.
    results = []
    for run_id in sorted(all_run_correlations.keys()):
        entries = all_run_correlations[run_id]
        by_target = {}
        for e in entries:
            key = (run_id, e['test_target'])
            if key not in by_target or e['spearman'] > by_target[key]['spearman']:
                by_target[key] = e

        for (r, t), best in by_target.items():
            results.append({
                'Dataset': dataset_name,
                'Run': r,
                'Method': method,
                'Test_Target': t,
                'Best_Metric': best['metric'],
                'Best_Weight': best['weight'],
                'Lambda1': best['lambda1'],
                'Lambda2': best['lambda2'],
                'Lambda3': best['lambda3'],
                'Best_Spearman': best['spearman'],
            })

    return results


def _parse_weight(weight_str):
    """'w0.0_0.0_0.0' -> (0.0, 0.0, 0.0)."""
    try:
        parts = weight_str.replace("w", "").split("_")
        return tuple(float(x) for x in parts[:3])
    except Exception:
        return (0.0, 0.0, 0.0)


def find_best_per_run_from_correlation_files(dataset_name, hybrid_results_dir):
    """Fast path: reuse the correlation CSVs produced by calculate_hybrid_scores.

    Returns the list of per-run winners, or None when the correlation files are
    missing.
    """
    dataset_dir = hybrid_results_dir / dataset_name
    if not dataset_dir.exists():
        return None

    corr_files = list(dataset_dir.glob("correlations_*_w*.csv"))
    if not corr_files:
        return None

    frames = []
    for f in corr_files:
        try:
            df = pd.read_csv(f)
            if "Weight" not in df.columns:
                continue
            # Extract lambda values from the weight string (e.g. 'w0.0_0.0_0.0').
            w = df["Weight"].iloc[0]
            l1, l2, l3 = _parse_weight(w)
            df["Lambda1"] = l1
            df["Lambda2"] = l2
            df["Lambda3"] = l3
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined["Test_Target"] = combined["Test_Metric"].map({
        "MSE_Test_Interpolation": "Interpolation",
        "MSE_Test_Extrapolation": "Extrapolation",
    })
    combined = combined[combined["Test_Target"].notna()]

    best_rows = (
        combined.sort_values("Spearman", ascending=False)
        .groupby(["Dataset", "Run", "Method", "Test_Target"], as_index=False)
        .first()
    )
    results = []
    for _, row in best_rows.iterrows():
        results.append({
            "Dataset": row["Dataset"],
            "Run": row["Run"],
            "Method": row["Method"],
            "Test_Target": row["Test_Target"],
            "Best_Metric": row["Metric"],
            "Best_Weight": row["Weight"],
            "Lambda1": row["Lambda1"],
            "Lambda2": row["Lambda2"],
            "Lambda3": row["Lambda3"],
            "Best_Spearman": row["Spearman"],
        })
    return results


def process_dataset(dataset_name, base_dir, output_base_dir, hybrid_results_dir=None):
    """Compute per-run best results for a single dataset and write them to disk."""
    print(f"\nProcessing dataset: {dataset_name}")

    dataset_output_dir = output_base_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    if hybrid_results_dir is not None:
        fast_results = find_best_per_run_from_correlation_files(dataset_name, hybrid_results_dir)
        if fast_results:
            print(f"  Using pre-computed correlations from hybrid_results/")
            all_results = fast_results
        else:
            fast_results = None
    else:
        fast_results = None

    if fast_results is None:
        dataset_dir = base_dir / dataset_name
        input_csv = dataset_dir / "gp_model_selection_criteria_combined.csv"

        if not input_csv.exists():
            print(f"  Input file not found: {input_csv}")
            return None

        try:
            df = pd.read_csv(input_csv)
            print(f"  Loaded {len(df)} rows from {df['Run'].nunique()} runs")
        except Exception as e:
            print(f"  Error loading {input_csv.name}: {e}")
            return None

        required_cols = (
            BASE_METRICS
            + ['Run', 'Extrapolation_Divergence']
            + ['Sensitivity_Interpolation_SVP', 'Sensitivity_Extrapolation_SVP']
            + ['Sensitivity_Interpolation_MVP', 'Sensitivity_Extrapolation_MVP']
            + ['MSE_Test_Interpolation', 'MSE_Test_Extrapolation']
        )
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"  Missing required columns: {missing}")
            return None

        all_results = []
        for method in ['SVP', 'MVP']:
            print(f"  Method: {method}")
            run_results = find_best_per_run_for_dataset(df, dataset_name, method)
            all_results.extend(run_results)

    if not all_results:
        print(f"  No results produced")
        return

    # Reshape from long to wide: one row per (Dataset, Run) with explicit
    # SVP/MVP and Interpolation/Extrapolation columns.
    long_df = pd.DataFrame(all_results)
    wide_rows = []
    for (ds, run), group in long_df.groupby(['Dataset', 'Run']):
        row = {'Dataset': ds, 'Run': run}
        for _, r in group.iterrows():
            method = r['Method']
            target = r['Test_Target']
            prefix = f"{method}_{target}"
            row[f"{prefix}_Best_Metric"] = r['Best_Metric']
            row[f"{prefix}_Best_Weight"] = r['Best_Weight']
            row[f"{prefix}_Lambda1"] = r['Lambda1']
            row[f"{prefix}_Lambda2"] = r['Lambda2']
            row[f"{prefix}_Lambda3"] = r['Lambda3']
            row[f"{prefix}_Best_Spearman"] = r['Best_Spearman']
        wide_rows.append(row)

    out_df = pd.DataFrame(wide_rows)
    column_order = (
        ['Dataset', 'Run']
        + [
            f"{m}_{t}_{c}"
            for m in ['MVP', 'SVP']
            for t in ['Extrapolation', 'Interpolation']
            for c in ['Best_Metric', 'Best_Weight', 'Lambda1', 'Lambda2', 'Lambda3', 'Best_Spearman']
        ]
    )
    out_df = out_df[[c for c in column_order if c in out_df.columns]]

    output_file = dataset_output_dir / f"{dataset_name}_best_per_run.csv"
    out_df.to_csv(output_file, index=False)
    n_runs = out_df['Run'].nunique()
    print(f"  Saved {len(out_df)} rows ({n_runs} runs x 2 methods x 2 targets) to {output_file.name}")

    return out_df


def main():
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent
    hybrid_results_dir = script_dir / "hybrid_results"
    output_base_dir = script_dir / "hybrid_results" / "best_per_run"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_base_dir}")
    print(f"Processing {len(DATASET_FOLDERS)} datasets")
    print("Finding best (metric, lambda) per run based on Spearman correlation")
    if any((hybrid_results_dir / d).exists() for d in DATASET_FOLDERS):
        print("(Using pre-computed correlations from hybrid_results when available)")

    all_datasets_results = []
    for dataset_name in DATASET_FOLDERS:
        result_df = process_dataset(
            dataset_name, base_dir, output_base_dir,
            hybrid_results_dir=hybrid_results_dir,
        )
        if result_df is not None:
            all_datasets_results.append(result_df)

    if all_datasets_results:
        combined = pd.concat(all_datasets_results, ignore_index=True)
        combined_file = output_base_dir / "best_per_run_all_datasets.csv"
        combined.to_csv(combined_file, index=False)
        print(f"\nSaved combined results to {combined_file.name} ({len(combined)} rows)")

    print(f"\nDone. Results in {output_base_dir}")


if __name__ == "__main__":
    main()
