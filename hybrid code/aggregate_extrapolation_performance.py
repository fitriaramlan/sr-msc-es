"""
Aggregate dataset-level performance from the hybrid score correlation files,
separately for interpolation and extrapolation test targets.

For each dataset we compute the largest correlation change against the
baseline (w0.0_0.0_0.0) across every weight combination and base metric, and
categorise the dataset on the magnitude / sign of that change.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def parse_weight_string(weight_str):
    """'w0.0_0.0_0.0' -> (0.0, 0.0, 0.0); None on bad input."""
    weight_str = weight_str.replace('w', '')
    parts = weight_str.split('_')
    if len(parts) == 3:
        return tuple(float(x) for x in parts)
    return None


def load_all_correlation_files(results_dir):
    """Load every per-dataset correlations_*.csv from hybrid_results/."""
    results_path = Path(results_dir)
    hybrid_results_dir = results_path / "hybrid_results"

    if not hybrid_results_dir.exists():
        print(f"Error: hybrid_results directory not found at {hybrid_results_dir}")
        return pd.DataFrame()

    all_correlations = []

    for dataset_dir in hybrid_results_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        correlation_files = list(dataset_dir.glob("correlations_*.csv"))

        for corr_file in correlation_files:
            try:
                df = pd.read_csv(corr_file)

                # Files normally already have Dataset/Method/Weight columns;
                # fall back to parsing the filename otherwise.
                if 'Dataset' not in df.columns:
                    df['Dataset'] = dataset_name

                if 'Method' not in df.columns or 'Weight' not in df.columns:
                    filename = corr_file.stem
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        method = parts[1]
                        weight_str = '_'.join(parts[2:])
                        if 'Method' not in df.columns:
                            df['Method'] = method
                        if 'Weight' not in df.columns:
                            df['Weight'] = weight_str

                required_cols = ['Dataset', 'Run', 'Method', 'Weight', 'Metric', 'Test_Metric', 'Spearman']
                missing_cols = [c for c in required_cols if c not in df.columns]
                if missing_cols:
                    print(f"{corr_file.name} missing columns: {missing_cols}")
                    continue

                all_correlations.append(df)
            except Exception as e:
                print(f"Could not load {corr_file}: {e}")

    if not all_correlations:
        print("No correlation files found.")
        return pd.DataFrame()

    return pd.concat(all_correlations, ignore_index=True)


def get_dataset_metadata(results_dir, corr_df=None):
    """Look up (n samples, d features) for every dataset under results_dir."""
    results_path = Path(results_dir)
    metadata = {}

    if corr_df is not None and not corr_df.empty:
        dataset_folders = sorted(corr_df['Dataset'].unique())
    else:
        dataset_folders = [
            "1096_FacultySalaries", "192_vineyard", "228_elusage", "542_pollution",
            "523_analcatdata_neavote", "678_visualizing_environmental", "229_pwLinear",
            "561_cpu", "712_chscase_geyser1", "605_fri_c2_250_25", "519_vinnie",
            "556_analcatdata_apnea2", "557_analcatdata_apnea1", "584_fri_c4_500_25",
            "637_fri_c1_500_50", "589_fri_c2_1000_25", "1028_SWD", "522_pm10",
            "227_cpu_small", "529_pollen",
        ]

    for dataset_name in dataset_folders:
        possible_paths = [
            results_path.parent / dataset_name / "gp_model_selection_criteria_combined.csv",
            results_path / dataset_name / "gp_model_selection_criteria_combined.csv",
            results_path / ".." / dataset_name / "gp_model_selection_criteria_combined.csv",
        ]

        csv_file = None
        for path in possible_paths:
            if path.exists():
                csv_file = path
                break

        if csv_file is None:
            # Fall back to the normalised data file in hybrid_results/.
            hybrid_results_dir = results_path / "hybrid_results" / dataset_name
            if hybrid_results_dir.exists():
                norm_file = hybrid_results_dir / f"{dataset_name}_normalised_data.csv"
                if norm_file.exists():
                    csv_file = norm_file

        if csv_file and csv_file.exists():
            try:
                dataset_dir = csv_file.parent
                datasets_dir = dataset_dir.parent / 'datasets'
                data_file = datasets_dir / f"{dataset_name}.npy"

                n_samples = None
                n_features = None

                if data_file.exists():
                    try:
                        data = np.load(data_file)
                        if data.ndim == 2:
                            n_samples = data.shape[0]
                            # Assume the last column holds y.
                            n_features = data.shape[1] - 1
                        elif data.ndim == 1:
                            n_samples = len(data)
                    except Exception:
                        n_samples = None
                        n_features = None
                else:
                    x_file = datasets_dir / f"{dataset_name}_X.npy"
                    y_file = datasets_dir / f"{dataset_name}_y.npy"

                    if x_file.exists():
                        try:
                            x_data = np.load(x_file)
                            if x_data.ndim == 2:
                                n_samples = x_data.shape[0]
                                n_features = x_data.shape[1]
                        except Exception:
                            n_samples = None
                            n_features = None
                    elif y_file.exists():
                        try:
                            y_data = np.load(y_file)
                            if y_data.ndim == 1:
                                n_samples = len(y_data)
                            elif y_data.ndim == 2:
                                n_samples = y_data.shape[0]
                        except Exception:
                            n_samples = None
                            n_features = None

                metadata[dataset_name] = {
                    'n': n_samples if n_samples is not None and n_samples > 0 else None,
                    'd': n_features if n_features is not None and n_features > 0 else None,
                }
            except Exception as e:
                print(f"Could not get metadata for {dataset_name}: {e}")
                metadata[dataset_name] = {'n': None, 'd': None}
        else:
            metadata[dataset_name] = {'n': None, 'd': None}

    return metadata


def calculate_baseline_correlations(corr_df):
    """Mean / min / max / std baseline correlations per (dataset, method, metric, test)."""
    baseline_mask = corr_df['Weight'] == 'w0.0_0.0_0.0'
    baseline_data = corr_df[baseline_mask].copy()

    baseline_summary = baseline_data.groupby(
        ['Dataset', 'Method', 'Metric', 'Test_Metric']
    )['Spearman'].agg(['mean', 'min', 'max', 'std', 'count']).reset_index()

    dataset_baseline = baseline_summary.groupby(['Dataset', 'Test_Metric'])['mean'].agg(
        ['min', 'max']
    ).reset_index()

    return baseline_summary, dataset_baseline


def calculate_correlation_changes(corr_df, baseline_summary,
                                  test_metric='MSE_Test_Extrapolation'):
    """Compute delta rho = hybrid correlation - baseline correlation."""
    test_data = corr_df[corr_df['Test_Metric'] == test_metric].copy()

    baseline_for_merge = baseline_summary[
        baseline_summary['Test_Metric'] == test_metric
    ][['Dataset', 'Method', 'Metric', 'mean']].rename(columns={'mean': 'Baseline_Spearman'})

    merged = test_data.merge(
        baseline_for_merge, on=['Dataset', 'Method', 'Metric'], how='left'
    )

    merged['Delta_Rho'] = merged['Spearman'] - merged['Baseline_Spearman']
    merged['Abs_Delta_Rho'] = np.abs(merged['Delta_Rho'])
    return merged


def categorize_dataset(max_abs_delta_rho, max_positive_delta=None,
                       max_negative_delta=None, delta_rho_values=None):
    """Bucket a dataset based on the largest |delta rho| seen across weight combinations.

    Categories:
      - Insensitive          : |delta rho| < 0.05
      - Selective Improvement: |delta rho| >= 0.05 and positives dominate
      - Instability          : |delta rho| >= 0.05 and we see both big
                               improvements and big degradations
    """
    if pd.isna(max_abs_delta_rho):
        return "Unknown"
    if max_abs_delta_rho < 0.05:
        return "Insensitive"
    if max_abs_delta_rho >= 0.05:
        # Variance check for instability.
        if delta_rho_values is not None and len(delta_rho_values) > 1:
            significant_improvements = np.sum(delta_rho_values > 0.05)
            significant_degradations = np.sum(delta_rho_values < -0.05)

            mean_abs_delta = np.mean(np.abs(delta_rho_values))
            std_delta = np.std(delta_rho_values)

            has_both = significant_improvements > 0 and significant_degradations > 0
            high_variance = (
                mean_abs_delta > 0
                and std_delta / mean_abs_delta > 1.5
                and std_delta > 0.15
            )

            if has_both or high_variance:
                return "Instability"

        if max_positive_delta is not None and max_negative_delta is not None:
            if abs(max_negative_delta) > max_positive_delta and abs(max_negative_delta) >= 0.05:
                return "Instability"

        return "Selective Improvement"
    return "Unknown"


def generate_latex_table(summary_df, output_file, method_name="", test_type=""):
    """Write a LaTeX table summarising the per-dataset performance."""
    label_suffix = f"_{method_name}" if method_name else ""

    with open(output_file, 'w') as f:
        f.write("\\begin{table*}\n")
        f.write("\\centering\n")
        f.write("\\caption{Dataset-wise summary of hybrid score performance")
        if method_name:
            f.write(f" using {method_name} method")
        if test_type:
            f.write(f" for {test_type.lower()} performance")
        f.write(". Datasets are categorised by the maximum correlation change ")
        f.write("($|\\Delta\\rho|$) seen across all weight combinations and base ")
        if test_type:
            f.write(f"metrics when correlating with {test_type.lower()} test MSE. ")
        else:
            f.write("metrics when correlating with test MSE. ")
        f.write("$n$ = number of samples, $d$ = number of features. ")
        f.write("Positive $\\Delta\\rho$ = improvement, negative = degradation.}\n")
        f.write(f"\\label{{tab:dataset_{test_type.lower()}_performance{label_suffix}}}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Dataset} & \\textbf{n} & \\textbf{d} & ")
        f.write("\\textbf{Baseline $\\rho$} & \\textbf{Category} & ")
        f.write("\\textbf{Max $\\Delta\\rho$} \\\\\n")
        f.write("\\midrule\n")

        for _, row in summary_df.iterrows():
            dataset = row['Dataset']
            n = row['n'] if pd.notna(row['n']) else 'N/A'
            d = row['d'] if pd.notna(row['d']) else 'N/A'
            baseline_rho = row['Baseline_rho']
            category = row['Category']
            max_delta = row['Max_Delta_Rho'] if 'Max_Delta_Rho' in row and pd.notna(row['Max_Delta_Rho']) else 'N/A'

            if isinstance(n, (int, float)):
                n = f"{int(n):,}"
            if isinstance(d, (int, float)):
                d = f"{int(d)}"
            if isinstance(max_delta, (int, float)):
                max_delta = f"{max_delta:+.4f}"

            f.write(f"{dataset} & {n} & {d} & {baseline_rho} & {category} & {max_delta} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

    print(f"  LaTeX table generated with {len(summary_df)} rows")


def create_method_comparison(summary_svp, summary_mvp, dataset_metadata):
    """SVP vs MVP comparison table."""
    comparison_rows = []

    all_datasets = sorted(
        set(summary_svp['Dataset'].unique()) | set(summary_mvp['Dataset'].unique())
    )

    for dataset in all_datasets:
        svp_row = summary_svp[summary_svp['Dataset'] == dataset]
        mvp_row = summary_mvp[summary_mvp['Dataset'] == dataset]

        meta = dataset_metadata.get(dataset, {'n': None, 'd': None})

        svp_max_delta = (
            svp_row['Max_Delta_Rho'].iloc[0] if not svp_row.empty and 'Max_Delta_Rho' in svp_row.columns
            else np.nan
        )
        mvp_max_delta = (
            mvp_row['Max_Delta_Rho'].iloc[0] if not mvp_row.empty and 'Max_Delta_Rho' in mvp_row.columns
            else np.nan
        )

        comparison_rows.append({
            'Dataset': dataset,
            'n': meta.get('n'),
            'd': meta.get('d'),
            'SVP_Baseline_rho': svp_row['Baseline_rho'].iloc[0] if not svp_row.empty else 'N/A',
            'MVP_Baseline_rho': mvp_row['Baseline_rho'].iloc[0] if not mvp_row.empty else 'N/A',
            'SVP_Category': svp_row['Category'].iloc[0] if not svp_row.empty else 'N/A',
            'MVP_Category': mvp_row['Category'].iloc[0] if not mvp_row.empty else 'N/A',
            'SVP_Max_Delta_Rho': svp_max_delta,
            'MVP_Max_Delta_Rho': mvp_max_delta,
            'SVP_Max_Abs_Delta_Rho': abs(svp_max_delta) if pd.notna(svp_max_delta) else np.nan,
            'MVP_Max_Abs_Delta_Rho': abs(mvp_max_delta) if pd.notna(mvp_max_delta) else np.nan,
            'SVP_Best_Weight': svp_row['Best_Weight'].iloc[0] if not svp_row.empty and 'Best_Weight' in svp_row.columns else 'N/A',
            'MVP_Best_Weight': mvp_row['Best_Weight'].iloc[0] if not mvp_row.empty and 'Best_Weight' in mvp_row.columns else 'N/A',
            'SVP_Best_Metric': svp_row['Best_Metric'].iloc[0] if not svp_row.empty and 'Best_Metric' in svp_row.columns else 'N/A',
            'MVP_Best_Metric': mvp_row['Best_Metric'].iloc[0] if not mvp_row.empty and 'Best_Metric' in mvp_row.columns else 'N/A',
            'Difference_Max_Delta_Rho': (
                (mvp_max_delta - svp_max_delta)
                if pd.notna(svp_max_delta) and pd.notna(mvp_max_delta)
                else np.nan
            ),
            'Difference_Max_Abs_Delta_Rho': (
                (abs(mvp_max_delta) - abs(svp_max_delta))
                if pd.notna(svp_max_delta) and pd.notna(mvp_max_delta)
                else np.nan
            ),
        })

    return pd.DataFrame(comparison_rows)


def aggregate_performance_across_datasets(corr_df, dataset_metadata,
                                          method=None,
                                          test_metric='MSE_Test_Extrapolation'):
    """Per-dataset summary for one test target (and optionally one method).

    When method is None, the largest |delta rho| across both SVP and MVP is
    used. Otherwise the data is filtered to that method first.
    """
    test_data = corr_df[corr_df['Test_Metric'] == test_metric].copy()

    if method is not None:
        test_data = test_data[test_data['Method'] == method].copy()
        if test_data.empty:
            print(f"No {test_metric} correlation data for method {method}.")
            return pd.DataFrame()

    if test_data.empty:
        print(f"No {test_metric} correlation data found.")
        return pd.DataFrame()

    baseline_summary, dataset_baseline = calculate_baseline_correlations(corr_df)

    if method is not None:
        baseline_summary = baseline_summary[baseline_summary['Method'] == method].copy()
        dataset_baseline = dataset_baseline[
            dataset_baseline['Dataset'].isin(test_data['Dataset'].unique())
        ].copy()

    merged = calculate_correlation_changes(corr_df, baseline_summary, test_metric=test_metric)
    if method is not None:
        merged = merged[merged['Method'] == method].copy()

    dataset_summary = []

    for dataset in sorted(test_data['Dataset'].unique()):
        dataset_data = merged[merged['Dataset'] == dataset]
        if dataset_data.empty:
            continue

        baseline_for_dataset = baseline_summary[
            (baseline_summary['Dataset'] == dataset)
            & (baseline_summary['Test_Metric'] == test_metric)
        ]

        if not baseline_for_dataset.empty:
            baseline_min = baseline_for_dataset['mean'].min()
            baseline_max = baseline_for_dataset['mean'].max()
            baseline_range_str = f"[{baseline_min:.3f}, {baseline_max:.3f}]"
        else:
            baseline_range = dataset_baseline[
                (dataset_baseline['Dataset'] == dataset)
                & (dataset_baseline['Test_Metric'] == test_metric)
            ]
            if not baseline_range.empty:
                baseline_min = baseline_range['min'].iloc[0]
                baseline_max = baseline_range['max'].iloc[0]
                baseline_range_str = f"[{baseline_min:.3f}, {baseline_max:.3f}]"
            else:
                baseline_range_str = "[N/A, N/A]"

        positive_deltas = dataset_data[dataset_data['Delta_Rho'] > 0]
        if not positive_deltas.empty:
            max_positive_delta = positive_deltas['Delta_Rho'].max()
            max_positive_row = positive_deltas.loc[positive_deltas['Delta_Rho'].idxmax()]
            best_positive_weight = max_positive_row['Weight']
            best_positive_metric = max_positive_row['Metric']
        else:
            max_positive_delta = 0.0
            best_positive_weight = 'N/A'
            best_positive_metric = 'N/A'

        negative_deltas = dataset_data[dataset_data['Delta_Rho'] < 0]
        if not negative_deltas.empty:
            max_negative_delta = negative_deltas['Delta_Rho'].min()
            max_negative_row = negative_deltas.loc[negative_deltas['Delta_Rho'].idxmin()]
            worst_negative_weight = max_negative_row['Weight']
            worst_negative_metric = max_negative_row['Metric']
        else:
            max_negative_delta = 0.0
            worst_negative_weight = 'N/A'
            worst_negative_metric = 'N/A'

        max_abs_delta = dataset_data['Abs_Delta_Rho'].max()
        max_abs_row = dataset_data.loc[dataset_data['Abs_Delta_Rho'].idxmax()]
        best_weight = max_abs_row['Weight']
        best_method = max_abs_row['Method']
        best_metric = max_abs_row['Metric']
        best_delta = max_abs_row['Delta_Rho']

        meta = dataset_metadata.get(dataset, {'n': None, 'd': None})

        delta_rho_values = dataset_data['Delta_Rho'].dropna().values
        category = categorize_dataset(
            max_abs_delta,
            max_positive_delta=max_positive_delta if max_positive_delta > 0 else None,
            max_negative_delta=max_negative_delta if max_negative_delta < 0 else None,
            delta_rho_values=delta_rho_values,
        )

        significant_positive = dataset_data[dataset_data['Delta_Rho'] > 0.05]
        significant_negative = dataset_data[dataset_data['Delta_Rho'] < -0.05]

        dataset_summary.append({
            'Dataset': dataset,
            'Method': method if method is not None else best_method,
            'n': meta.get('n'),
            'd': meta.get('d'),
            'Baseline_rho': baseline_range_str,
            'Category': category,
            'Max_Delta_Rho': best_delta,
            'Max_Positive_Delta_Rho': max_positive_delta if max_positive_delta > 0 else None,
            'Max_Negative_Delta_Rho': max_negative_delta if max_negative_delta < 0 else None,
            'Max_Abs_Delta_Rho': max_abs_delta,
            'Best_Weight': best_weight,
            'Best_Metric': best_metric,
            'Best_Positive_Weight': best_positive_weight if max_positive_delta > 0 else 'N/A',
            'Best_Positive_Metric': best_positive_metric if max_positive_delta > 0 else 'N/A',
            'Worst_Negative_Weight': worst_negative_weight if max_negative_delta < 0 else 'N/A',
            'Worst_Negative_Metric': worst_negative_metric if max_negative_delta < 0 else 'N/A',
            'N_Improvements': len(significant_positive),
            'N_Degradations': len(significant_negative),
            'N_Total_Combinations': len(dataset_data),
        })

    return pd.DataFrame(dataset_summary)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-dataset hybrid score correlation deltas for "
            "interpolation and extrapolation test targets."
        ),
        epilog="""
Datasets are categorised by the maximum correlation change (|delta rho| >= 0.05)
observed across all weight combinations when correlating hybrid scores with
test MSE.

Categories:
  - Insensitive          : |delta rho| < 0.05
  - Selective Improvement: |delta rho| >= 0.05 with mostly positive changes
  - Instability          : both significant improvements and degradations

For each test target the script produces SVP, MVP and combined summaries plus
a LaTeX table and an SVP-vs-MVP comparison table.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'results_dir',
        nargs='?',
        default='.',
        help='Root directory containing hybrid_results/',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: <results_dir>/hybrid_results)',
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "hybrid_results"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AGGREGATING INTERPOLATION AND EXTRAPOLATION PERFORMANCE ACROSS DATASETS")
    print("=" * 80)
    print(f"\nResults directory: {results_dir}")
    print(f"Output directory:  {output_dir}\n")

    print("Step 1: Loading correlation files...")
    corr_df = load_all_correlation_files(results_dir)

    if corr_df.empty:
        print("Error: no correlation data found. Run calculate_hybrid_scores.py first.")
        return

    print(f"  {len(corr_df)} correlation rows loaded")
    print(f"  Datasets:        {corr_df['Dataset'].nunique()}")
    print(f"  Methods:         {sorted(corr_df['Method'].unique())}")
    print(f"  Test metrics:    {sorted(corr_df['Test_Metric'].unique())}")
    print(f"  Weight combos:   {corr_df['Weight'].nunique()}")

    print("\nStep 2: Resolving dataset metadata (n, d)...")
    dataset_metadata = get_dataset_metadata(results_dir, corr_df=corr_df)
    print(f"  Resolved metadata for {len(dataset_metadata)} datasets")

    missing_meta = [
        ds for ds, meta in dataset_metadata.items()
        if meta.get('n') is None or meta.get('d') is None
    ]
    if missing_meta:
        print(f"  {len(missing_meta)} dataset(s) missing metadata: {missing_meta[:5]}")
        if len(missing_meta) > 5:
            print(f"    ... and {len(missing_meta) - 5} more")

    print("\nStep 3: Aggregating performance...")

    summaries = {}
    for test_metric in ['MSE_Test_Interpolation', 'MSE_Test_Extrapolation']:
        test_type = 'Interpolation' if 'Interpolation' in test_metric else 'Extrapolation'
        print(f"\n  --- {test_type} ---")

        print(f"    SVP for {test_type}...")
        summary_svp = aggregate_performance_across_datasets(
            corr_df, dataset_metadata, method='SVP', test_metric=test_metric
        )

        print(f"    MVP for {test_type}...")
        summary_mvp = aggregate_performance_across_datasets(
            corr_df, dataset_metadata, method='MVP', test_metric=test_metric
        )

        print(f"    Combined (SVP + MVP) for {test_type}...")
        summary_combined = aggregate_performance_across_datasets(
            corr_df, dataset_metadata, method=None, test_metric=test_metric
        )

        summaries[test_metric] = {
            'SVP': summary_svp,
            'MVP': summary_mvp,
            'Combined': summary_combined,
        }

    has_data = any(
        not summaries[t][m].empty
        for t in summaries
        for m in summaries[t]
    )

    if not has_data:
        print("Error: could not produce any summary. Check correlation data.")
        return

    print("\nStep 4: Saving results...")

    def create_clean_summary(df):
        if df.empty:
            return pd.DataFrame()
        clean = df[[
            'Dataset', 'n', 'd', 'Baseline_rho', 'Category', 'Max_Delta_Rho'
        ]].copy()
        return clean

    for test_metric in ['MSE_Test_Interpolation', 'MSE_Test_Extrapolation']:
        test_type = 'Interpolation' if 'Interpolation' in test_metric else 'Extrapolation'
        test_prefix = test_type.lower()

        summary_svp = summaries[test_metric]['SVP']
        summary_mvp = summaries[test_metric]['MVP']
        summary_combined = summaries[test_metric]['Combined']

        if not summary_svp.empty:
            clean_svp = create_clean_summary(summary_svp)
            f = output_dir / f"dataset_{test_prefix}_performance_summary_SVP.csv"
            clean_svp.to_csv(f, index=False)
            print(f"  SVP {test_type} summary -> {f.name}")

            full_f = output_dir / f"dataset_{test_prefix}_performance_summary_full_SVP.csv"
            summary_svp.to_csv(full_f, index=False)
            print(f"  SVP {test_type} full summary -> {full_f.name}")

        if not summary_mvp.empty:
            clean_mvp = create_clean_summary(summary_mvp)
            f = output_dir / f"dataset_{test_prefix}_performance_summary_MVP.csv"
            clean_mvp.to_csv(f, index=False)
            print(f"  MVP {test_type} summary -> {f.name}")

            full_f = output_dir / f"dataset_{test_prefix}_performance_summary_full_MVP.csv"
            summary_mvp.to_csv(full_f, index=False)
            print(f"  MVP {test_type} full summary -> {full_f.name}")

        if not summary_combined.empty:
            clean_combined = create_clean_summary(summary_combined)
            f = output_dir / f"dataset_{test_prefix}_performance_summary.csv"
            clean_combined.to_csv(f, index=False)
            print(f"  Combined {test_type} summary -> {f.name}")

            full_f = output_dir / f"dataset_{test_prefix}_performance_summary_full.csv"
            summary_combined.to_csv(full_f, index=False)
            print(f"  Combined {test_type} full summary -> {full_f.name}")

        if not summary_svp.empty and not summary_mvp.empty:
            print(f"\n  Creating SVP vs MVP comparison for {test_type}...")
            comparison = create_method_comparison(summary_svp, summary_mvp, dataset_metadata)
            comparison_file = output_dir / f"dataset_{test_prefix}_performance_SVP_vs_MVP_comparison.csv"
            comparison.to_csv(comparison_file, index=False)
            print(f"  Comparison -> {comparison_file.name}")

        print(f"\n  Detailed {test_type} analysis (all weight combinations)...")
        baseline_summary, _ = calculate_baseline_correlations(corr_df)
        detailed = calculate_correlation_changes(corr_df, baseline_summary, test_metric=test_metric)

        detailed_file = output_dir / f"dataset_{test_prefix}_performance_detailed.csv"
        detailed.to_csv(detailed_file, index=False)
        print(f"  Detailed -> {detailed_file.name}")

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for test_metric in ['MSE_Test_Interpolation', 'MSE_Test_Extrapolation']:
        test_type = 'Interpolation' if 'Interpolation' in test_metric else 'Extrapolation'
        print(f"\n{'=' * 80}")
        print(f"{test_type.upper()} PERFORMANCE")
        print(f"{'=' * 80}")

        summary_svp = summaries[test_metric]['SVP']
        summary_mvp = summaries[test_metric]['MVP']

        if not summary_svp.empty:
            print(f"\n[SVP - {test_type}] Datasets: {len(summary_svp)}")
            print(f"[SVP - {test_type}] Category distribution:")
            category_counts_svp = summary_svp['Category'].value_counts()
            for category, count in category_counts_svp.items():
                print(f"  {category}: {count} ({100 * count / len(summary_svp):.1f}%)")

            print(f"\n[SVP - {test_type}] delta rho across datasets:")
            print(f"  Maximum: {summary_svp['Max_Delta_Rho'].max():+.4f}")
            print(f"  Minimum: {summary_svp['Max_Delta_Rho'].min():+.4f}")
            print(f"  Mean:    {summary_svp['Max_Delta_Rho'].mean():+.4f}")
            print(f"  Median:  {summary_svp['Max_Delta_Rho'].median():+.4f}")

            print(f"\n[SVP - {test_type}] Top 5 positive changes:")
            for _, row in summary_svp.nlargest(5, 'Max_Delta_Rho').iterrows():
                print(f"  {row['Dataset']}: delta rho = {row['Max_Delta_Rho']:+.4f} "
                      f"(Category: {row['Category']}, Weight: {row['Best_Weight']}, "
                      f"Metric: {row['Best_Metric']})")

            print(f"\n[SVP - {test_type}] Top 5 negative changes:")
            for _, row in summary_svp.nsmallest(5, 'Max_Delta_Rho').iterrows():
                print(f"  {row['Dataset']}: delta rho = {row['Max_Delta_Rho']:+.4f} "
                      f"(Category: {row['Category']}, Weight: {row['Best_Weight']}, "
                      f"Metric: {row['Best_Metric']})")

        if not summary_mvp.empty:
            print(f"\n[MVP - {test_type}] Datasets: {len(summary_mvp)}")
            print(f"[MVP - {test_type}] Category distribution:")
            category_counts_mvp = summary_mvp['Category'].value_counts()
            for category, count in category_counts_mvp.items():
                print(f"  {category}: {count} ({100 * count / len(summary_mvp):.1f}%)")

            print(f"\n[MVP - {test_type}] delta rho across datasets:")
            print(f"  Maximum: {summary_mvp['Max_Delta_Rho'].max():+.4f}")
            print(f"  Minimum: {summary_mvp['Max_Delta_Rho'].min():+.4f}")
            print(f"  Mean:    {summary_mvp['Max_Delta_Rho'].mean():+.4f}")
            print(f"  Median:  {summary_mvp['Max_Delta_Rho'].median():+.4f}")

            print(f"\n[MVP - {test_type}] Top 5 positive changes:")
            for _, row in summary_mvp.nlargest(5, 'Max_Delta_Rho').iterrows():
                print(f"  {row['Dataset']}: delta rho = {row['Max_Delta_Rho']:+.4f} "
                      f"(Category: {row['Category']}, Weight: {row['Best_Weight']}, "
                      f"Metric: {row['Best_Metric']})")

            print(f"\n[MVP - {test_type}] Top 5 negative changes:")
            for _, row in summary_mvp.nsmallest(5, 'Max_Delta_Rho').iterrows():
                print(f"  {row['Dataset']}: delta rho = {row['Max_Delta_Rho']:+.4f} "
                      f"(Category: {row['Category']}, Weight: {row['Best_Weight']}, "
                      f"Metric: {row['Best_Metric']})")

        if not summary_svp.empty and not summary_mvp.empty:
            comparison = create_method_comparison(summary_svp, summary_mvp, dataset_metadata)
            if 'Difference_Max_Delta_Rho' in comparison.columns:
                valid_diff = comparison['Difference_Max_Delta_Rho'].dropna()
                if not valid_diff.empty:
                    print(f"\n[SVP vs MVP - {test_type}] Signed delta rho:")
                    print(f"  Mean difference (MVP - SVP): {valid_diff.mean():+.4f}")
                    print(f"  MVP > SVP: {(valid_diff > 0).sum()}")
                    print(f"  SVP > MVP: {(valid_diff < 0).sum()}")
                    print(f"  Equal:     {(valid_diff == 0).sum()}")

            if 'Difference_Max_Abs_Delta_Rho' in comparison.columns:
                valid_diff_abs = comparison['Difference_Max_Abs_Delta_Rho'].dropna()
                if not valid_diff_abs.empty:
                    print(f"\n[SVP vs MVP - {test_type}] Absolute |delta rho|:")
                    print(f"  Mean difference (MVP - SVP): {valid_diff_abs.mean():+.4f}")
                    print(f"  MVP > SVP: {(valid_diff_abs > 0).sum()}")
                    print(f"  SVP > MVP: {(valid_diff_abs < 0).sum()}")
                    print(f"  Equal:     {(valid_diff_abs == 0).sum()}")

    print("\nStep 5: Generating LaTeX tables...")

    for test_metric in ['MSE_Test_Interpolation', 'MSE_Test_Extrapolation']:
        test_type = 'Interpolation' if 'Interpolation' in test_metric else 'Extrapolation'
        test_prefix = test_type.lower()

        summary_svp = summaries[test_metric]['SVP']
        summary_mvp = summaries[test_metric]['MVP']
        summary_combined = summaries[test_metric]['Combined']

        if not summary_svp.empty:
            clean_svp = create_clean_summary(summary_svp)
            latex_file = output_dir / f"dataset_{test_prefix}_performance_summary_SVP.tex"
            generate_latex_table(clean_svp, latex_file, method_name="SVP", test_type=test_type)
            print(f"  SVP {test_type} LaTeX -> {latex_file.name}")

        if not summary_mvp.empty:
            clean_mvp = create_clean_summary(summary_mvp)
            latex_file = output_dir / f"dataset_{test_prefix}_performance_summary_MVP.tex"
            generate_latex_table(clean_mvp, latex_file, method_name="MVP", test_type=test_type)
            print(f"  MVP {test_type} LaTeX -> {latex_file.name}")

        if not summary_combined.empty:
            clean_combined = create_clean_summary(summary_combined)
            latex_file = output_dir / f"dataset_{test_prefix}_performance_summary.tex"
            generate_latex_table(clean_combined, latex_file, method_name="", test_type=test_type)
            print(f"  Combined {test_type} LaTeX -> {latex_file.name}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
