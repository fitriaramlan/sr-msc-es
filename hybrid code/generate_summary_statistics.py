"""
Produce summary statistics for the hybrid metric analysis.

It walks the per-dataset CSV files under hybrid_results/ and aggregates:
  - clamp values per metric / dataset / run,
  - post-normalisation ranges (sanity check that everything sits in [0, 1]),
  - distribution statistics for each normalised metric,
  - which hybrid configurations beat their base metric / the best base metric,
  - correlation deltas vs the baseline (w0.0_0.0_0.0).
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def load_all_clamp_statistics(results_dir):
    """Concatenate every per-dataset clamping-threshold CSV."""
    results_path = Path(results_dir)
    hybrid_results_dir = results_path / "hybrid_results"

    if not hybrid_results_dir.exists():
        print(f"Error: hybrid_results directory not found at {hybrid_results_dir}")
        return pd.DataFrame()

    all_clamp_stats = []
    for dataset_dir in hybrid_results_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        clamp_file = dataset_dir / f"{dataset_dir.name}_clamping_thresholds_per_pareto_front.csv"
        if clamp_file.exists():
            try:
                df = pd.read_csv(clamp_file)
                df['Dataset'] = dataset_dir.name
                all_clamp_stats.append(df)
            except Exception as e:
                print(f"Could not load {clamp_file}: {e}")

    if not all_clamp_stats:
        return pd.DataFrame()

    return pd.concat(all_clamp_stats, ignore_index=True)


def check_clamp_values(clamp_df, results_dir, extreme_threshold=1000.0):
    """Print and persist a summary of the clamp values, flagging extreme cases."""
    if clamp_df.empty:
        print("No clamp statistics found.")
        return

    print("\n" + "=" * 80)
    print("CLAMP VALUE ANALYSIS")
    print("=" * 80)

    print("\nClamp values per metric / dataset / run")
    print("-" * 80)

    available_cols = ['Dataset', 'Run', 'Metric', 'Upper_Clamp']
    if 'Constant_Value' in clamp_df.columns:
        available_cols.append('Constant_Value')
    elif 'Threshold_Value' in clamp_df.columns:
        available_cols.append('Threshold_Value')

    display_df = clamp_df[[col for col in available_cols if col in clamp_df.columns]].copy()
    display_df = display_df.sort_values(['Metric', 'Dataset', 'Run'])

    print("\nFirst 20 rows:")
    print(display_df.head(20).to_string(index=False))
    if len(display_df) > 20:
        print(f"\n... and {len(display_df) - 20} more rows (see CSV).")

    print("\n\nUpper_Clamp summary per metric:")
    print("-" * 80)

    for metric in sorted(clamp_df['Metric'].unique()):
        metric_data = clamp_df[clamp_df['Metric'] == metric]
        print(f"\n{metric}:")
        print(f"  Min clamp value:    {metric_data['Upper_Clamp'].min():.6f}")
        print(f"  Max clamp value:    {metric_data['Upper_Clamp'].max():.6f}")
        print(f"  Mean clamp value:   {metric_data['Upper_Clamp'].mean():.6f}")
        print(f"  Median clamp value: {metric_data['Upper_Clamp'].median():.6f}")
        print(f"  Rows:               {len(metric_data)}")

        extreme_cases = metric_data[metric_data['Upper_Clamp'] > extreme_threshold]
        if len(extreme_cases) > 0:
            print(f"  WARNING: {len(extreme_cases)} cases with clamp > {extreme_threshold}")
            for _, row in extreme_cases.head(10).iterrows():
                print(f"    Dataset: {row['Dataset']}, Run: {row['Run']}, "
                      f"Clamp: {row['Upper_Clamp']:.2f}")
        else:
            print(f"  No extreme clamp values (all < {extreme_threshold})")

    all_extreme = clamp_df[clamp_df['Upper_Clamp'] > extreme_threshold]
    if len(all_extreme) > 0:
        print(f"\nWARNING: {len(all_extreme)} total cases exceed {extreme_threshold}.")
        print("   This can happen when more than ~30% of values in a run are outliers.")
        for (dataset, run, metric), group in all_extreme.groupby(['Dataset', 'Run', 'Metric']):
            print(f"     Dataset: {dataset}, Run: {run}, Metric: {metric}, "
                  f"Clamp: {group['Upper_Clamp'].iloc[0]:.2f}")
    else:
        print(f"\nAll clamp values within reasonable range.")
        print(f"   Max clamp value across all metrics: {clamp_df['Upper_Clamp'].max():.6f}")

    results_path = Path(results_dir) if isinstance(results_dir, str) else results_dir
    output_file = results_path / "hybrid_results" / "clamp_value_summary.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sorted_clamp_df = clamp_df.sort_values(['Metric', 'Dataset', 'Run'])
    sorted_clamp_df.to_csv(output_file, index=False)
    print(f"\nClamp statistics saved to: {output_file}")


def verify_normalization(results_dir):
    """Verify that every metric ends up in the [0, 1] range after normalisation."""
    results_path = Path(results_dir)
    hybrid_results_dir = results_path / "hybrid_results"

    if not hybrid_results_dir.exists():
        print(f"Error: hybrid_results directory not found at {hybrid_results_dir}")
        return

    print("\n" + "=" * 80)
    print("NORMALISATION VERIFICATION")
    print("=" * 80)
    print("\nChecking that all metrics are min-max normalised to [0, 1]")
    print("-" * 80)

    all_norm_stats = []
    for dataset_dir in hybrid_results_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        norm_file = dataset_dir / f"{dataset_dir.name}_normalisation_statistics.csv"
        if norm_file.exists():
            try:
                df = pd.read_csv(norm_file)
                df['Dataset'] = dataset_dir.name
                all_norm_stats.append(df)
            except Exception as e:
                print(f"Could not load {norm_file}: {e}")

    if not all_norm_stats:
        print("No normalisation statistics found.")
        return

    combined = pd.concat(all_norm_stats, ignore_index=True)

    print("\nNormalisation status per metric:")
    print("-" * 80)

    all_in_range = True
    for metric in sorted(combined['Metric'].unique()):
        metric_data = combined[combined['Metric'] == metric]
        all_ranges_valid = True

        for _, row in metric_data.iterrows():
            post_min = row['Post_Norm_Min']
            post_max = row['Post_Norm_Max']
            if not (np.isnan(post_min) and np.isnan(post_max)):
                if post_min < -0.0001 or post_max > 1.0001:
                    all_ranges_valid = False
                    all_in_range = False

        tag = "OK" if all_ranges_valid else "!!"
        min_val = metric_data['Post_Norm_Min'].min()
        max_val = metric_data['Post_Norm_Max'].max()
        mean_min = metric_data['Post_Norm_Min'].mean()
        mean_max = metric_data['Post_Norm_Max'].mean()
        print(f"[{tag}] {metric}:")
        print(f"    Range across datasets:  [{min_val:.6f}, {max_val:.6f}]")
        print(f"    Mean min/max:           [{mean_min:.6f}, {mean_max:.6f}]")
        print(f"    Number of datasets:     {len(metric_data)}")

    base_metrics = ["MSE_Train", "AIC_Train", "BIC_Train", "MDL_Train", "PSM_Train"]
    print("\n\nBase metric normalisation check:")
    print("-" * 80)
    for metric in base_metrics:
        metric_data = combined[combined['Metric'] == metric]
        if metric_data.empty:
            print(f"[!!] {metric}: not found in normalisation statistics")
        else:
            min_val = metric_data['Post_Norm_Min'].min()
            max_val = metric_data['Post_Norm_Max'].max()
            in_range = (min_val >= -0.0001 and max_val <= 1.0001)
            tag = "OK" if in_range else "!!"
            print(f"[{tag}] {metric}: normalised to [{min_val:.6f}, {max_val:.6f}]")

    if all_in_range:
        print("\nAll metrics (including AIC, MDL, BIC, PSM) sit in [0, 1].")
    else:
        print("\nWARNING: some metrics fall outside [0, 1]!")

    results_path = Path(results_dir) if isinstance(results_dir, str) else results_dir
    output_file = results_path / "hybrid_results" / "normalisation_summary.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False)
    print(f"\nNormalisation statistics saved to: {output_file}")

    # Split into SVP / MVP subsets (shared base + method-specific sensitivities).
    base_metrics_list = ["MSE_Train", "AIC_Train", "BIC_Train", "MDL_Train",
                         "PSM_Train", "Extrapolation_Divergence"]
    svp_metrics = base_metrics_list + ["Sensitivity_Interpolation_SVP", "Sensitivity_Extrapolation_SVP"]
    mvp_metrics = base_metrics_list + ["Sensitivity_Interpolation_MVP", "Sensitivity_Extrapolation_MVP"]

    svp_data = combined[combined['Metric'].isin(svp_metrics)]
    if not svp_data.empty:
        svp_file = results_path / "hybrid_results" / "normalisation_summary_SVP.csv"
        svp_data.to_csv(svp_file, index=False)
        print(f"  SVP subset saved to: {svp_file.name}")

    mvp_data = combined[combined['Metric'].isin(mvp_metrics)]
    if not mvp_data.empty:
        mvp_file = results_path / "hybrid_results" / "normalisation_summary_MVP.csv"
        mvp_data.to_csv(mvp_file, index=False)
        print(f"  MVP subset saved to: {mvp_file.name}")


def load_model_selection_results(results_dir):
    """Concatenate every per-dataset model selection analysis CSV."""
    results_path = Path(results_dir)
    hybrid_results_dir = results_path / "hybrid_results"

    if not hybrid_results_dir.exists():
        return pd.DataFrame()

    all_selections = []
    for dataset_dir in hybrid_results_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        selection_file = dataset_dir / f"{dataset_dir.name}_model_selection_analysis.csv"
        if selection_file.exists():
            try:
                df = pd.read_csv(selection_file)
                all_selections.append(df)
            except Exception as e:
                print(f"Could not load {selection_file}: {e}")

    if not all_selections:
        return pd.DataFrame()

    return pd.concat(all_selections, ignore_index=True)


def analyze_metric_distributions(results_dir):
    """Compute distribution statistics for each normalised metric."""
    results_path = Path(results_dir)
    hybrid_results_dir = results_path / "hybrid_results"

    if not hybrid_results_dir.exists():
        print(f"Error: hybrid_results directory not found at {hybrid_results_dir}")
        return

    print("\n" + "=" * 80)
    print("METRIC DISTRIBUTION ANALYSIS (after normalisation)")
    print("=" * 80)
    print("-" * 80)

    all_distributions = []

    for dataset_dir in hybrid_results_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        norm_file = dataset_dir / f"{dataset_dir.name}_normalised_data.csv"
        if norm_file.exists():
            try:
                df = pd.read_csv(norm_file)
                dataset_name = dataset_dir.name

                base_metrics = ["MSE_Train", "AIC_Train", "BIC_Train", "MDL_Train", "PSM_Train"]
                new_metrics = [
                    "Extrapolation_Divergence",
                    "Sensitivity_Interpolation_SVP",
                    "Sensitivity_Interpolation_MVP",
                    "Sensitivity_Extrapolation_SVP",
                    "Sensitivity_Extrapolation_MVP",
                ]
                all_metrics = base_metrics + new_metrics

                for metric in all_metrics:
                    if metric not in df.columns:
                        continue

                    values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(values) == 0:
                        continue

                    all_distributions.append({
                        'Dataset': dataset_name,
                        'Metric': metric,
                        'Count': len(values),
                        'Mean': values.mean(),
                        'Std': values.std(),
                        'Min': values.min(),
                        'Max': values.max(),
                        'Median': values.median(),
                        'Q25': values.quantile(0.25),
                        'Q75': values.quantile(0.75),
                        'Skewness': values.skew() if len(values) > 2 else np.nan,
                        'Kurtosis': values.kurtosis() if len(values) > 2 else np.nan,
                    })
            except Exception as e:
                print(f"Could not load {norm_file}: {e}")

    if not all_distributions:
        print("No normalised data available.")
        return

    dist_df = pd.DataFrame(all_distributions)

    print("\nDistribution summary per metric (across datasets):")
    print("-" * 80)
    for metric in sorted(dist_df['Metric'].unique()):
        metric_data = dist_df[dist_df['Metric'] == metric]
        print(f"\n{metric}:")
        print(f"  Datasets:               {len(metric_data)}")
        print(f"  Mean (across datasets): {metric_data['Mean'].mean():.6f}")
        print(f"  Std  (across datasets): {metric_data['Std'].mean():.6f}")
        print(f"  Min value (all data):   {metric_data['Min'].min():.6f}")
        print(f"  Max value (all data):   {metric_data['Max'].max():.6f}")
        print(f"  Median (across datasets): {metric_data['Median'].mean():.6f}")
        print(f"  Mean skewness:          {metric_data['Skewness'].mean():.6f}")

    output_file = results_path / "hybrid_results" / "metric_distributions_after_normalisation.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dist_df.to_csv(output_file, index=False)
    print(f"\nDistribution statistics saved to: {output_file}")

    base_metrics_list = ["MSE_Train", "AIC_Train", "BIC_Train", "MDL_Train",
                         "PSM_Train", "Extrapolation_Divergence"]
    svp_metrics = base_metrics_list + ["Sensitivity_Interpolation_SVP", "Sensitivity_Extrapolation_SVP"]
    mvp_metrics = base_metrics_list + ["Sensitivity_Interpolation_MVP", "Sensitivity_Extrapolation_MVP"]

    svp_dist = dist_df[dist_df['Metric'].isin(svp_metrics)]
    if not svp_dist.empty:
        svp_file = results_path / "hybrid_results" / "metric_distributions_after_normalisation_SVP.csv"
        svp_dist.to_csv(svp_file, index=False)
        print(f"  SVP subset saved to: {svp_file.name}")

    mvp_dist = dist_df[dist_df['Metric'].isin(mvp_metrics)]
    if not mvp_dist.empty:
        mvp_file = results_path / "hybrid_results" / "metric_distributions_after_normalisation_MVP.csv"
        mvp_dist.to_csv(mvp_file, index=False)
        print(f"  MVP subset saved to: {mvp_file.name}")

    return dist_df


def calculate_best_hybrid_metrics(selection_df, results_dir):
    """Aggregate hybrid vs base/best-base wins and pick top performers."""
    if selection_df.empty:
        print("No model selection results available.")
        return None

    print("\n" + "=" * 80)
    print("BEST HYBRID METRIC ANALYSIS")
    print("=" * 80)
    print("-" * 80)

    results = []

    for dataset in sorted(selection_df['Dataset'].unique()):
        dataset_data = selection_df[selection_df['Dataset'] == dataset]

        for run_id in sorted(dataset_data['Run'].unique()):
            run_data = dataset_data[dataset_data['Run'] == run_id]

            baseline_mask = (
                (run_data['Lambda1'] == 0.0)
                & (run_data['Lambda2'] == 0.0)
                & (run_data['Lambda3'] == 0.0)
            )
            baseline_data = run_data[baseline_mask]
            hybrid_data = run_data[~baseline_mask]

            for _, hybrid_row in hybrid_data.iterrows():
                base_metric = hybrid_row['Metric']
                method = hybrid_row['Method']
                lambda1 = hybrid_row['Lambda1']
                lambda2 = hybrid_row['Lambda2']
                lambda3 = hybrid_row['Lambda3']

                base_baseline = baseline_data[baseline_data['Metric'] == base_metric]

                # Interpolation outcome
                if not pd.isna(hybrid_row['MSE_Test_Interpolation']):
                    hybrid_interp = hybrid_row['MSE_Test_Interpolation']

                    if not base_baseline.empty:
                        base_interp = base_baseline.iloc[0]['MSE_Test_Interpolation']
                        better_than_base = hybrid_interp < base_interp if not pd.isna(base_interp) else False
                    else:
                        better_than_base = False

                    if not baseline_data.empty:
                        best_base_interp = baseline_data['MSE_Test_Interpolation'].min()
                        better_than_best_base = hybrid_interp < best_base_interp if not pd.isna(best_base_interp) else False
                    else:
                        better_than_best_base = False

                    results.append({
                        'Dataset': dataset,
                        'Run': run_id,
                        'Method': method,
                        'Metric': base_metric,
                        'Lambda1': lambda1,
                        'Lambda2': lambda2,
                        'Lambda3': lambda3,
                        'Test_Type': 'Interpolation',
                        'Hybrid_MSE': hybrid_interp,
                        'Base_Metric_MSE': base_baseline.iloc[0]['MSE_Test_Interpolation'] if not base_baseline.empty else np.nan,
                        'Best_Base_MSE': baseline_data['MSE_Test_Interpolation'].min() if not baseline_data.empty else np.nan,
                        'Better_Than_Base': better_than_base,
                        'Better_Than_Best_Base': better_than_best_base,
                    })

                # Extrapolation outcome
                if not pd.isna(hybrid_row['MSE_Test_Extrapolation']):
                    hybrid_extrap = hybrid_row['MSE_Test_Extrapolation']

                    if not base_baseline.empty:
                        base_extrap = base_baseline.iloc[0]['MSE_Test_Extrapolation']
                        better_than_base = hybrid_extrap < base_extrap if not pd.isna(base_extrap) else False
                    else:
                        better_than_base = False

                    if not baseline_data.empty:
                        best_base_extrap = baseline_data['MSE_Test_Extrapolation'].min()
                        better_than_best_base = hybrid_extrap < best_base_extrap if not pd.isna(best_base_extrap) else False
                    else:
                        better_than_best_base = False

                    results.append({
                        'Dataset': dataset,
                        'Run': run_id,
                        'Method': method,
                        'Metric': base_metric,
                        'Lambda1': lambda1,
                        'Lambda2': lambda2,
                        'Lambda3': lambda3,
                        'Test_Type': 'Extrapolation',
                        'Hybrid_MSE': hybrid_extrap,
                        'Base_Metric_MSE': base_baseline.iloc[0]['MSE_Test_Extrapolation'] if not base_baseline.empty else np.nan,
                        'Best_Base_MSE': baseline_data['MSE_Test_Extrapolation'].min() if not baseline_data.empty else np.nan,
                        'Better_Than_Base': better_than_base,
                        'Better_Than_Best_Base': better_than_best_base,
                    })

    if not results:
        print("No results produced.")
        return None

    results_df = pd.DataFrame(results)

    print("\nBest hybrid metric overall")
    print("-" * 80)

    overall_stats = results_df.groupby(['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3']).agg({
        'Hybrid_MSE': ['mean', 'std', 'count'],
        'Better_Than_Base': 'sum',
        'Better_Than_Best_Base': 'sum',
        'Dataset': lambda x: ', '.join(sorted(x.unique())),
    }).reset_index()

    overall_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                             for col in overall_stats.columns.values]

    def count_datasets(x):
        if pd.isna(x):
            return 0
        x_str = str(x).strip()
        if not x_str:
            return 0
        return len(x_str.split(', '))

    overall_stats['N_Unique_Datasets'] = overall_stats['Dataset_<lambda>'].apply(count_datasets)

    better_than_base_by_dataset = results_df[results_df['Better_Than_Base'] == True].groupby(
        ['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3']
    )['Dataset'].nunique().reset_index(name='N_Datasets_Better_Than_Base')

    better_than_best_base_by_dataset = results_df[results_df['Better_Than_Best_Base'] == True].groupby(
        ['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3']
    )['Dataset'].nunique().reset_index(name='N_Datasets_Better_Than_Best_Base')

    overall_stats = overall_stats.merge(better_than_base_by_dataset,
                                        on=['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3'],
                                        how='left')
    overall_stats = overall_stats.merge(better_than_best_base_by_dataset,
                                        on=['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3'],
                                        how='left')

    overall_stats['N_Datasets_Better_Than_Base'] = overall_stats['N_Datasets_Better_Than_Base'].fillna(0).astype(int)
    overall_stats['N_Datasets_Better_Than_Best_Base'] = overall_stats['N_Datasets_Better_Than_Best_Base'].fillna(0).astype(int)

    overall_stats = overall_stats.rename(columns={
        'Hybrid_MSE_mean': 'Mean_MSE',
        'Hybrid_MSE_std': 'Std_MSE',
        'Hybrid_MSE_count': 'N_Combinations',
        'Better_Than_Base_sum': 'N_Better_Than_Base',
        'Better_Than_Best_Base_sum': 'N_Better_Than_Best_Base',
        'Dataset_<lambda>': 'Datasets',
    })

    overall_stats = overall_stats.sort_values('Mean_MSE', ascending=True)

    print("\nTop 10 hybrid metrics by mean MSE:")
    for _, row in overall_stats.head(10).iterrows():
        print(f"  {row['Method']} | {row['Metric']} | "
              f"lambda=({row['Lambda1']:.2f}, {row['Lambda2']:.2f}, {row['Lambda3']:.2f}) | "
              f"Mean MSE: {row['Mean_MSE']:.6f} | "
              f"Better than base: {row['N_Datasets_Better_Than_Base']}/{row['N_Unique_Datasets']} | "
              f"Better than best base: {row['N_Datasets_Better_Than_Best_Base']}/{row['N_Unique_Datasets']}")

    best_overall = overall_stats.iloc[0]
    print(f"\nBest hybrid metric overall:")
    print(f"  Method:         {best_overall['Method']}")
    print(f"  Base metric:    {best_overall['Metric']}")
    print(f"  Lambda values:  ({best_overall['Lambda1']:.2f}, {best_overall['Lambda2']:.2f}, {best_overall['Lambda3']:.2f})")
    print(f"  Mean MSE:       {best_overall['Mean_MSE']:.6f}")
    print(f"  Better than base metric:      {best_overall['N_Datasets_Better_Than_Base']}/{best_overall['N_Unique_Datasets']} datasets")
    print(f"  Better than best base metric: {best_overall['N_Datasets_Better_Than_Best_Base']}/{best_overall['N_Unique_Datasets']} datasets")

    # Interpolation-only ranking
    print("\n\nBest hybrid metric for test-interpolation")
    print("-" * 80)
    interp_data = results_df[results_df['Test_Type'] == 'Interpolation']
    interp_stats = interp_data.groupby(['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3']).agg({
        'Hybrid_MSE': ['mean', 'std', 'count'],
        'Better_Than_Base': 'sum',
        'Better_Than_Best_Base': 'sum',
        'Dataset': lambda x: ', '.join(sorted(x.unique())),
    }).reset_index()

    interp_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                            for col in interp_stats.columns.values]
    interp_stats['N_Unique_Datasets'] = interp_stats['Dataset_<lambda>'].apply(count_datasets)

    better_than_base_interp = interp_data[interp_data['Better_Than_Base'] == True].groupby(
        ['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3']
    )['Dataset'].nunique().reset_index(name='N_Datasets_Better_Than_Base')

    better_than_best_base_interp = interp_data[interp_data['Better_Than_Best_Base'] == True].groupby(
        ['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3']
    )['Dataset'].nunique().reset_index(name='N_Datasets_Better_Than_Best_Base')

    interp_stats = interp_stats.merge(better_than_base_interp,
                                      on=['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3'],
                                      how='left')
    interp_stats = interp_stats.merge(better_than_best_base_interp,
                                      on=['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3'],
                                      how='left')

    interp_stats['N_Datasets_Better_Than_Base'] = interp_stats['N_Datasets_Better_Than_Base'].fillna(0).astype(int)
    interp_stats['N_Datasets_Better_Than_Best_Base'] = interp_stats['N_Datasets_Better_Than_Best_Base'].fillna(0).astype(int)

    interp_stats = interp_stats.rename(columns={
        'Hybrid_MSE_mean': 'Mean_MSE',
        'Hybrid_MSE_std': 'Std_MSE',
        'Hybrid_MSE_count': 'N_Combinations',
        'Better_Than_Base_sum': 'N_Better_Than_Base',
        'Better_Than_Best_Base_sum': 'N_Better_Than_Best_Base',
        'Dataset_<lambda>': 'Datasets',
    })
    interp_stats = interp_stats.sort_values('Mean_MSE', ascending=True)

    print("\nTop 10 hybrid metrics for interpolation:")
    for _, row in interp_stats.head(10).iterrows():
        print(f"  {row['Method']} | {row['Metric']} | "
              f"lambda=({row['Lambda1']:.2f}, {row['Lambda2']:.2f}, {row['Lambda3']:.2f}) | "
              f"Mean MSE: {row['Mean_MSE']:.6f} | "
              f"Better than base: {row['N_Datasets_Better_Than_Base']}/{row['N_Unique_Datasets']}")

    best_interp = interp_stats.iloc[0]
    print(f"\nBest hybrid metric for interpolation:")
    print(f"  Method:        {best_interp['Method']}")
    print(f"  Base metric:   {best_interp['Metric']}")
    print(f"  Lambda values: ({best_interp['Lambda1']:.2f}, {best_interp['Lambda2']:.2f}, {best_interp['Lambda3']:.2f})")
    print(f"  Mean MSE:      {best_interp['Mean_MSE']:.6f}")
    print(f"  Better than base metric: {best_interp['N_Datasets_Better_Than_Base']}/{best_interp['N_Unique_Datasets']} datasets")

    # Extrapolation-only ranking
    print("\n\nBest hybrid metric for test-extrapolation")
    print("-" * 80)
    extrap_data = results_df[results_df['Test_Type'] == 'Extrapolation']
    extrap_stats = extrap_data.groupby(['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3']).agg({
        'Hybrid_MSE': ['mean', 'std', 'count'],
        'Better_Than_Base': 'sum',
        'Better_Than_Best_Base': 'sum',
        'Dataset': lambda x: ', '.join(sorted(x.unique())),
    }).reset_index()

    extrap_stats.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                            for col in extrap_stats.columns.values]
    extrap_stats['N_Unique_Datasets'] = extrap_stats['Dataset_<lambda>'].apply(count_datasets)

    better_than_base_extrap = extrap_data[extrap_data['Better_Than_Base'] == True].groupby(
        ['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3']
    )['Dataset'].nunique().reset_index(name='N_Datasets_Better_Than_Base')

    better_than_best_base_extrap = extrap_data[extrap_data['Better_Than_Best_Base'] == True].groupby(
        ['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3']
    )['Dataset'].nunique().reset_index(name='N_Datasets_Better_Than_Best_Base')

    extrap_stats = extrap_stats.merge(better_than_base_extrap,
                                      on=['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3'],
                                      how='left')
    extrap_stats = extrap_stats.merge(better_than_best_base_extrap,
                                      on=['Method', 'Metric', 'Lambda1', 'Lambda2', 'Lambda3'],
                                      how='left')

    extrap_stats['N_Datasets_Better_Than_Base'] = extrap_stats['N_Datasets_Better_Than_Base'].fillna(0).astype(int)
    extrap_stats['N_Datasets_Better_Than_Best_Base'] = extrap_stats['N_Datasets_Better_Than_Best_Base'].fillna(0).astype(int)

    extrap_stats = extrap_stats.rename(columns={
        'Hybrid_MSE_mean': 'Mean_MSE',
        'Hybrid_MSE_std': 'Std_MSE',
        'Hybrid_MSE_count': 'N_Combinations',
        'Better_Than_Base_sum': 'N_Better_Than_Base',
        'Better_Than_Best_Base_sum': 'N_Better_Than_Best_Base',
        'Dataset_<lambda>': 'Datasets',
    })
    extrap_stats = extrap_stats.sort_values('Mean_MSE', ascending=True)

    print("\nTop 10 hybrid metrics for extrapolation:")
    for _, row in extrap_stats.head(10).iterrows():
        print(f"  {row['Method']} | {row['Metric']} | "
              f"lambda=({row['Lambda1']:.2f}, {row['Lambda2']:.2f}, {row['Lambda3']:.2f}) | "
              f"Mean MSE: {row['Mean_MSE']:.6f} | "
              f"Better than base: {row['N_Datasets_Better_Than_Base']}/{row['N_Unique_Datasets']}")

    best_extrap = extrap_stats.iloc[0]
    print(f"\nBest hybrid metric for extrapolation:")
    print(f"  Method:        {best_extrap['Method']}")
    print(f"  Base metric:   {best_extrap['Metric']}")
    print(f"  Lambda values: ({best_extrap['Lambda1']:.2f}, {best_extrap['Lambda2']:.2f}, {best_extrap['Lambda3']:.2f})")
    print(f"  Mean MSE:      {best_extrap['Mean_MSE']:.6f}")
    print(f"  Better than base metric: {best_extrap['N_Datasets_Better_Than_Base']}/{best_extrap['N_Unique_Datasets']} datasets")

    output_dir = Path(results_dir) / "hybrid_results"
    results_df.to_csv(output_dir / "hybrid_comparison_results.csv", index=False)
    overall_stats.to_csv(output_dir / "best_hybrid_overall.csv", index=False)
    interp_stats.to_csv(output_dir / "best_hybrid_interpolation.csv", index=False)
    extrap_stats.to_csv(output_dir / "best_hybrid_extrapolation.csv", index=False)

    for method in ['SVP', 'MVP']:
        results_method = results_df[results_df['Method'] == method]
        if not results_method.empty:
            results_method.to_csv(output_dir / f"hybrid_comparison_results_{method}.csv", index=False)

        overall_method = overall_stats[overall_stats['Method'] == method]
        if not overall_method.empty:
            overall_method.to_csv(output_dir / f"best_hybrid_overall_{method}.csv", index=False)

        interp_method = interp_stats[interp_stats['Method'] == method]
        if not interp_method.empty:
            interp_method.to_csv(output_dir / f"best_hybrid_interpolation_{method}.csv", index=False)

        extrap_method = extrap_stats[extrap_stats['Method'] == method]
        if not extrap_method.empty:
            extrap_method.to_csv(output_dir / f"best_hybrid_extrapolation_{method}.csv", index=False)

    print(f"\nAll detailed results saved to {output_dir}")

    return {
        'overall': overall_stats,
        'interpolation': interp_stats,
        'extrapolation': extrap_stats,
        'detailed': results_df,
    }


def parse_weight_string(weight_str):
    """'w0.0_0.0_0.0' -> (0.0, 0.0, 0.0); None on bad input."""
    weight_str = weight_str.replace('w', '')
    parts = weight_str.split('_')
    if len(parts) == 3:
        try:
            return tuple(float(x) for x in parts)
        except ValueError:
            return None
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

                if 'Dataset' not in df.columns:
                    df['Dataset'] = dataset_name

                # Fall back to parsing the filename if columns are missing.
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

                all_correlations.append(df)
            except Exception as e:
                print(f"Could not load {corr_file}: {e}")

    if not all_correlations:
        return pd.DataFrame()

    return pd.concat(all_correlations, ignore_index=True)


def calculate_correlation_delta_statistics(results_dir):
    """Compare every weight combination against the baseline (w0.0_0.0_0.0).

    For each (method, test target) we emit a table containing the median
    delta correlation and the (+/=/-) counts across datasets.
    """
    print("\n" + "=" * 80)
    print("CORRELATION DELTA STATISTICS")
    print("=" * 80)

    print("\nLoading correlation files...")
    corr_df = load_all_correlation_files(results_dir)

    if corr_df.empty:
        print("No correlation files found.")
        return

    print(f"  {len(corr_df)} correlation rows loaded")
    print(f"  Datasets:          {corr_df['Dataset'].nunique()}")
    print(f"  Methods:           {sorted(corr_df['Method'].unique())}")
    print(f"  Weight combos:     {corr_df['Weight'].nunique()}")

    base_metrics = ["MSE_Train", "AIC_Train", "BIC_Train", "MDL_Train", "PSM_Train"]
    metric_short_names = {
        "MSE_Train": "MSE",
        "AIC_Train": "AIC",
        "BIC_Train": "BIC",
        "MDL_Train": "MDL",
        "PSM_Train": "PSM",
    }

    test_types = ["MSE_Test_Interpolation", "MSE_Test_Extrapolation"]
    test_type_names = {
        "MSE_Test_Interpolation": "Interpolation",
        "MSE_Test_Extrapolation": "Extrapolation",
    }

    for method in sorted(corr_df['Method'].unique()):
        method_corr = corr_df[corr_df['Method'] == method].copy()
        print(f"\nProcessing method: {method}")

        baseline_mask = method_corr['Weight'] == 'w0.0_0.0_0.0'
        baseline_data = method_corr[baseline_mask].copy()

        if baseline_data.empty:
            print(f"  No baseline data for {method}")
            continue

        baseline_means = baseline_data.groupby(['Dataset', 'Metric', 'Test_Metric'])['Spearman'].mean().reset_index()
        baseline_means = baseline_means.rename(columns={'Spearman': 'Baseline_Spearman'})

        non_baseline = method_corr[~baseline_mask].copy()
        if non_baseline.empty:
            print(f"  No non-baseline data for {method}")
            continue

        non_baseline_means = non_baseline.groupby(
            ['Dataset', 'Weight', 'Metric', 'Test_Metric']
        )['Spearman'].mean().reset_index()
        non_baseline_means = non_baseline_means.rename(columns={'Spearman': 'Hybrid_Spearman'})

        merged = non_baseline_means.merge(
            baseline_means, on=['Dataset', 'Metric', 'Test_Metric'], how='left'
        )
        merged['Delta'] = merged['Hybrid_Spearman'] - merged['Baseline_Spearman']

        method_test_data = method_corr[method_corr['Test_Metric'].isin(test_types)]
        n_datasets = method_test_data['Dataset'].nunique()

        for test_type in test_types:
            test_name = test_type_names[test_type]
            test_data = merged[merged['Test_Metric'] == test_type].copy()
            if test_data.empty:
                continue

            print(f"\n  Processing {test_name}...")

            summary_rows = []

            # Baseline row: deltas are zero by definition.
            baseline_weight_str = 'w0.0_0.0_0.0'
            baseline_row = {'base': '(0,0,0)', 'Weight': baseline_weight_str}
            for base_metric in base_metrics:
                metric_short = metric_short_names[base_metric]
                baseline_row[f'{metric_short}_plus_equal_minus'] = f"0,{n_datasets},0"
                baseline_row[f'{metric_short}_median_delta'] = "0.0000"
            summary_rows.append(baseline_row)

            unique_weights = sorted(
                [w for w in test_data['Weight'].unique() if w != baseline_weight_str]
            )

            for weight_str in unique_weights:
                weight_data = test_data[test_data['Weight'] == weight_str]

                weight_tuple = parse_weight_string(weight_str)
                if weight_tuple is None:
                    continue

                row_data = {'base': str(weight_tuple), 'Weight': weight_str}

                for base_metric in base_metrics:
                    metric_short = metric_short_names[base_metric]
                    metric_data = weight_data[weight_data['Metric'] == base_metric]

                    if metric_data.empty:
                        row_data[f'{metric_short}_plus_equal_minus'] = ''
                        row_data[f'{metric_short}_median_delta'] = ''
                    else:
                        deltas = metric_data['Delta'].dropna()
                        if len(deltas) == 0:
                            row_data[f'{metric_short}_plus_equal_minus'] = ''
                            row_data[f'{metric_short}_median_delta'] = ''
                        else:
                            # Treat |delta| <= 0.001 as "equal".
                            threshold = 0.001
                            n_plus = (deltas > threshold).sum()
                            n_equal = ((deltas >= -threshold) & (deltas <= threshold)).sum()
                            n_minus = (deltas < -threshold).sum()

                            row_data[f'{metric_short}_plus_equal_minus'] = f"{n_plus},{n_equal},{n_minus}"
                            median_delta = deltas.median()
                            row_data[f'{metric_short}_median_delta'] = (
                                f"{median_delta:.4f}" if not pd.isna(median_delta) else ''
                            )

                summary_rows.append(row_data)

            summary_df = pd.DataFrame(summary_rows)

            column_order = ['base', 'Weight']
            for base_metric in base_metrics:
                metric_short = metric_short_names[base_metric]
                column_order.append(f'{metric_short}_plus_equal_minus')
                column_order.append(f'{metric_short}_median_delta')
            column_order = [c for c in column_order if c in summary_df.columns]
            summary_df = summary_df[column_order]

            output_dir = Path(results_dir) / "hybrid_results"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"correlation_delta_summary_{method}_{test_name}.csv"
            summary_df.to_csv(output_file, index=False)
            print(f"    Saved {test_name} summary to: {output_file.name}")
            print(f"      Rows: {len(summary_df)}, Columns: {len(summary_df.columns)}")

            # Pivoted view (one column pair per base metric).
            pivoted_rows = []
            for _, row in summary_df.iterrows():
                pivoted_row = {'base': row['base'], 'Weight': row['Weight']}
                for base_metric in base_metrics:
                    metric_short = metric_short_names[base_metric]
                    plus_equal_minus = row.get(f'{metric_short}_plus_equal_minus', '')
                    median_delta = row.get(f'{metric_short}_median_delta', '')
                    pivoted_row[f'{metric_short}_+/-'] = plus_equal_minus
                    pivoted_row[f'{metric_short}_median_delta'] = median_delta
                pivoted_rows.append(pivoted_row)

            pivoted_df = pd.DataFrame(pivoted_rows)
            pivoted_file = output_dir / f"correlation_delta_summary_{method}_{test_name}_pivoted.csv"
            pivoted_df.to_csv(pivoted_file, index=False)
            print(f"    Saved {test_name} pivoted summary to: {pivoted_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate summary statistics for hybrid metrics")
    parser.add_argument('results_dir', nargs='?', default='.',
                        help='Root directory containing hybrid_results/')
    parser.add_argument('--extreme-threshold', type=float, default=1000.0,
                        help='Threshold for flagging extreme clamp values (default 1000.0)')

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    print("=" * 80)
    print("HYBRID METRIC SUMMARY STATISTICS")
    print("=" * 80)

    clamp_df = load_all_clamp_statistics(results_dir)
    if not clamp_df.empty:
        check_clamp_values(clamp_df, results_dir, extreme_threshold=args.extreme_threshold)

    verify_normalization(results_dir)

    analyze_metric_distributions(results_dir)

    selection_df = load_model_selection_results(results_dir)
    if not selection_df.empty:
        calculate_best_hybrid_metrics(selection_df, results_dir)

    calculate_correlation_delta_statistics(results_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
