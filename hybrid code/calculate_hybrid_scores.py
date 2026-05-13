from pathlib import Path
import argparse
import sys
import tempfile
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from ackp_selection import (
        select_ackp,
        estimate_dataset_difficulty,
        find_traditional_knee_point,
        find_minimal_complexity_knee_point,
        visualise_knee_points,
        visualise_angle_calculation_figure2,
        visualise_figure4_clustering_threshold,
        calculate_angles_theta_L_R,
    )
    ACKP_AVAILABLE = True
except ImportError:
    ACKP_AVAILABLE = False
    print("Warning: ACKP selection module not available. Falling back to minimum hybrid score.")


def _parse_lambda_string(s: str) -> tuple:
    """Parse a string like '(0.1, 0.0, 0.0)' into a tuple of floats."""
    s = s.strip().strip('()')
    return tuple(round(float(x.strip()), 4) for x in s.split(','))


def _load_best_lambda_combos(output_base_dir: Path, dataset_name: str) -> set:
    """Read the best lambda combinations from the per-dataset report CSVs.

    Returns a set of (method, (lambda1, lambda2, lambda3)) tuples; empty if
    no report files are present.
    """
    report_dir = output_base_dir / 'train_metrics_averages'
    best_combos = set()

    for method in ['SVP', 'MVP']:
        report_file = report_dir / f'report_best_per_dataset_{method}.csv'
        if not report_file.exists():
            continue
        try:
            report_df = pd.read_csv(report_file)
            rows = report_df[report_df['Dataset'] == dataset_name]
            for _, row in rows.iterrows():
                for col in ['Best_Interp_Lambda', 'Best_Extrap_Lambda']:
                    if col in row and pd.notna(row[col]):
                        lam = _parse_lambda_string(str(row[col]))
                        best_combos.add((method, lam))
        except Exception as e:
            print(f"  Could not read {report_file.name}: {e}")

    return best_combos


def calculate_hybrid_score(base_metric, extrap_div, sens_interp, sens_extrap,
                           lambda1=0.0, lambda2=0.0, lambda3=0.0):
    """Hybrid = base + l1*extrap_div + l2*sens_interp + l3*sens_extrap."""
    return base_metric + lambda1 * extrap_div + lambda2 * sens_interp + lambda3 * sens_extrap


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load a CSV, skipping rows whose field count does not match the header."""
    clean_lines: list[str] = []
    corrupted_count = 0

    with open(filepath, 'r') as f:
        header = f.readline().strip()
        clean_lines.append(header)
        expected_fields = header.count(',') + 1

        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.count(',') + 1 == expected_fields:
                clean_lines.append(line)
            else:
                corrupted_count += 1

    if corrupted_count > 0:
        print(f"    {filepath.name}: skipped {corrupted_count}")

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
        tmp_path = tmp.name
        for line in clean_lines:
            tmp.write(line + '\n')

    df = pd.read_csv(tmp_path)
    Path(tmp_path).unlink()

    initial_rows = len(df)
    df = df.dropna()
    nan_rows_removed = initial_rows - len(df)
    if nan_rows_removed > 0:
        print(f"    {filepath.name}: dropped {nan_rows_removed} row(s) with missing values")

    return df


def get_y_range_from_dataset(csv_file: Path) -> float:
    """Return the range of y values for the corresponding .npy dataset, or None."""
    dataset_name = csv_file.parent.name
    datasets_dir = csv_file.parent.parent / 'datasets'
    y_file = datasets_dir / f"{dataset_name}.npy"

    if y_file.exists():
        try:
            y_values = np.load(y_file)

            if y_values.ndim > 1:
                if y_values.shape[1] == 1:
                    y_values = y_values.flatten()
                else:
                    y_values = y_values[:, -1]

            y_values = np.array(y_values).flatten()
            y_values = y_values[np.isfinite(y_values)]

            if len(y_values) > 0:
                y_range = np.max(y_values) - np.min(y_values)
                print(f"    found y values in {y_file}: y_range = {y_range:.6f}")
                return float(y_range)
            else:
                print(f"    warning: {y_file.name} contains no valid y values")
        except Exception as e:
            print(f"    warning: failed to load y values from {y_file}: {e}")
    else:
        print(f"    warning: could not find y values file at {y_file}")

    return None


def simple_constant_clamp(constant_value: float = 10.0) -> float:
    """Return the fixed upper clamp value used for outlier clamping."""
    return constant_value


def clamp_metrics_per_pareto_front(df: pd.DataFrame, csv_file: Path = None,
                                   percentile_threshold: float = 70.0,
                                   absolute_max_cap: float = 1000.0,
                                   constant_clamp_value: float = 10.0) -> tuple:
    """Clamp the new metrics with a single constant upper bound, per run.

    Base metrics are not clamped; only the divergence/sensitivity columns are.
    `percentile_threshold` and `absolute_max_cap` are kept for backward
    compatibility with older callers and are not used here.
    """
    if 'Run' not in df.columns:
        raise ValueError("DataFrame must have a 'Run' column")

    df_clamped = df.copy()

    new_metrics = [
        "Extrapolation_Divergence",
        "Sensitivity_Interpolation_SVP",
        "Sensitivity_Interpolation_MVP",
        "Sensitivity_Extrapolation_SVP",
        "Sensitivity_Extrapolation_MVP",
    ]

    clamp_stats = []
    upper_clamp = simple_constant_clamp(constant_value=constant_clamp_value)
    print(f"    Constant clamping: upper bound = {upper_clamp}")

    for run_id in sorted(df['Run'].unique()):
        run_mask = df['Run'] == run_id
        run_data = df_clamped.loc[run_mask].copy()

        for metric in new_metrics:
            if metric not in run_data.columns:
                continue

            valid_values = run_data[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_values) < 2:
                continue

            original_min = valid_values.min()
            original_max = valid_values.max()

            run_data[metric] = run_data[metric].replace([np.inf, -np.inf], np.nan)
            run_data[metric] = run_data[metric].clip(upper=upper_clamp)

            df_clamped.loc[run_mask, metric] = run_data[metric]

            clamped_values = run_data[metric].dropna()
            clamped_min = clamped_values.min() if len(clamped_values) > 0 else np.nan
            clamped_max = clamped_values.max() if len(clamped_values) > 0 else np.nan

            valid_mask = run_data[metric].notna()
            n_clamped_upper = ((run_data[metric] == upper_clamp) & valid_mask).sum()

            clamp_stats.append({
                'Run': run_id,
                'Metric': metric,
                'Clamp_Method': 'simple_constant',
                'Constant_Value': constant_clamp_value,
                'Threshold_Value': upper_clamp,
                'Upper_Clamp': upper_clamp,
                'Original_Min': original_min,
                'Original_Max': original_max,
                'Clamped_Min': clamped_min,
                'Clamped_Max': clamped_max,
                'N_Clamped_Upper': n_clamped_upper,
                'N_Total': len(run_data),
                'N_Valid': len(valid_values),
            })

    if clamp_stats:
        print(f"    Per-Pareto-front clamping summary (constant={constant_clamp_value}):")
        threshold_df = pd.DataFrame(clamp_stats)
        for metric in new_metrics:
            metric_stats = threshold_df[threshold_df['Metric'] == metric]
            if not metric_stats.empty:
                n_clamped = metric_stats['N_Clamped_Upper'].sum()
                n_total = metric_stats['N_Total'].sum()
                print(f"      {metric}: clamped {n_clamped}/{n_total} values to {upper_clamp}")

    return df_clamped, clamp_stats


def create_histograms(df: pd.DataFrame, output_dir: Path, dataset_name: str,
                      stage: str = 'final') -> None:
    """Save histograms of every metric for the given processing stage."""
    base_metrics = BASE_METRICS.copy()
    new_metrics = [
        "Extrapolation_Divergence",
        "Sensitivity_Interpolation_SVP",
        "Sensitivity_Interpolation_MVP",
        "Sensitivity_Extrapolation_SVP",
        "Sensitivity_Extrapolation_MVP",
    ]

    all_metrics = base_metrics + new_metrics

    hist_dir = output_dir / 'histograms'
    hist_dir.mkdir(exist_ok=True)

    for metric in all_metrics:
        if metric not in df.columns:
            continue

        valid_values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_values) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(valid_values, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{metric} Distribution - {stage.replace("_", " ").title()}', fontsize=14)
        ax.grid(True, alpha=0.3)

        stats_text = (
            f"Mean: {valid_values.mean():.4f}\n"
            f"Std:  {valid_values.std():.4f}\n"
            f"Min:  {valid_values.min():.4f}\n"
            f"Max:  {valid_values.max():.4f}\n"
            f"N:    {len(valid_values)}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        metric_safe = metric.replace('/', '_').replace('\\', '_')
        filename = hist_dir / f"{dataset_name}_{metric_safe}_{stage}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"    saved histograms to {hist_dir.name}/")


def normalise_metrics_per_problem(df: pd.DataFrame) -> tuple:
    """Min-max normalise every metric to the [0, 1] range."""
    df_norm = df.copy()

    metrics_to_normalise = BASE_METRICS + [
        "Extrapolation_Divergence",
        "Sensitivity_Interpolation_SVP",
        "Sensitivity_Interpolation_MVP",
        "Sensitivity_Extrapolation_SVP",
        "Sensitivity_Extrapolation_MVP",
    ]

    range_stats = []

    for metric in metrics_to_normalise:
        if metric not in df_norm.columns:
            continue

        valid_values = df_norm[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_values) == 0:
            continue

        pre_mean = valid_values.mean()
        pre_std = valid_values.std()
        pre_min = valid_values.min()
        pre_max = valid_values.max()

        min_val = valid_values.min()
        max_val = valid_values.max()
        range_val = max_val - min_val

        if range_val > 0:
            mask = np.isfinite(df_norm[metric])
            df_norm.loc[mask, metric] = (df_norm.loc[mask, metric] - min_val) / range_val
        else:
            # Constant metric — collapse to the midpoint of [0, 1].
            mask = np.isfinite(df_norm[metric])
            df_norm.loc[mask, metric] = 0.5
            print(f"    Note: '{metric}' is constant, set to 0.5 after normalisation")

        normalised_values = df_norm[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if len(normalised_values) > 0:
            post_min = normalised_values.min()
            post_max = normalised_values.max()
            post_mean = normalised_values.mean()
            post_std = normalised_values.std()
            post_range = post_max - post_min
        else:
            post_min = post_max = post_mean = post_std = post_range = np.nan

        range_stats.append({
            'Metric': metric,
            'Pre_Norm_Mean': pre_mean,
            'Pre_Norm_Std': pre_std,
            'Pre_Norm_Min': pre_min,
            'Pre_Norm_Max': pre_max,
            'Post_Norm_Mean': post_mean,
            'Post_Norm_Std': post_std,
            'Post_Norm_Min': post_min,
            'Post_Norm_Max': post_max,
            'Post_Norm_Range': post_range,
        })

    return df_norm, range_stats


def analyze_model_selection(df: pd.DataFrame, results: pd.DataFrame, method: str,
                            lambda1: float, lambda2: float, lambda3: float,
                            dataset_name: str,
                            use_ackp: bool = False,
                            difficulty_threshold: float = 0.6,
                            save_visualisations: bool = False,
                            output_dir: Path = None) -> pd.DataFrame:
    """Choose the best equation per (run, metric) using either ACKP or min-hybrid."""
    selection_rows = []

    merge_cols = ['Run', 'Equation_Index']
    optional_cols = ['Complexity', 'MSE_Test_Interpolation', 'MSE_Test_Extrapolation'] + BASE_METRICS
    available_cols = merge_cols + [col for col in optional_cols if col in df.columns]

    cols_to_merge = [col for col in available_cols if col not in results.columns]
    if cols_to_merge:
        merged = results.merge(df[merge_cols + cols_to_merge], on=['Run', 'Equation_Index'], how='left')
    else:
        merged = results.copy()

    for run_id in sorted(merged['Run'].unique()):
        run_mask = merged['Run'] == run_id
        run_data = merged[run_mask].copy()

        if len(run_data) == 0:
            continue

        for base_metric in BASE_METRICS:
            hybrid_col = f"Hybrid_{base_metric}"

            if hybrid_col not in run_data.columns:
                continue

            valid_mask = run_data[hybrid_col].notna()
            if valid_mask.sum() == 0:
                continue

            valid_run_data = run_data.loc[valid_mask].copy()

            if use_ackp and ACKP_AVAILABLE and 'Complexity' in valid_run_data.columns:
                ackp_mask = (valid_run_data['Complexity'].notna() & valid_run_data[hybrid_col].notna())
                if ackp_mask.sum() < 3:
                    # Too few points — fall back to the minimum hybrid score.
                    best_idx = valid_run_data[hybrid_col].idxmin()
                    selection_method = 'minimum_hybrid_score_fallback'
                    difficulty = np.nan
                else:
                    ackp_data = valid_run_data.loc[ackp_mask]
                    complexity_values = ackp_data['Complexity'].values
                    hybrid_values = ackp_data[hybrid_col].values

                    difficulty = estimate_dataset_difficulty(
                        df, ackp_data,
                        test_interp_col='MSE_Test_Interpolation',
                        test_extrap_col='MSE_Test_Extrapolation',
                        train_col=base_metric if base_metric in ackp_data.columns else 'MSE_Train',
                    )

                    selected_local_idx, selection_method = select_ackp(
                        complexity_values,
                        hybrid_values,
                        difficulty,
                        difficulty_threshold=difficulty_threshold,
                    )

                    ackp_indices = ackp_data.index.values
                    best_idx = ackp_indices[selected_local_idx]

                    if save_visualisations and output_dir is not None:
                        vis_dir = output_dir / 'knee_point_visualisations'
                        vis_dir.mkdir(parents=True, exist_ok=True)
                        lambda_str = f"l{lambda1:.1f}_{lambda2:.1f}_{lambda3:.1f}"
                        base_name = f"{dataset_name}_run{run_id}_{method}_{hybrid_col}_{lambda_str}_ackp"
                        plot_title = f"{dataset_name} - Run {run_id} - {method} - {hybrid_col.replace('_', ' ')}"
                        subtitle = (
                            f"$\\lambda_1$={lambda1:.2f}, "
                            f"$\\lambda_2$={lambda2:.2f}, "
                            f"$\\lambda_3$={lambda3:.2f} | Difficulty: {difficulty:.2f}"
                        )

                        trad_knee = find_traditional_knee_point(complexity_values, hybrid_values)
                        mckp = find_minimal_complexity_knee_point(complexity_values, hybrid_values)

                        highlight_indices = []
                        if trad_knee is not None:
                            highlight_indices.append(ackp_data.index[trad_knee])
                        if mckp is not None:
                            highlight_indices.append(ackp_data.index[mckp])
                        if selected_local_idx is not None:
                            highlight_indices.append(ackp_data.index[selected_local_idx])

                        full_title = f'{plot_title}\n{subtitle} | Method: {selection_method}'
                        y_axis_label = hybrid_col.replace('_', ' ')
                        png_file = vis_dir / f"{base_name}_figure2_angle_calculation.png"
                        pdf_file = vis_dir / f"{base_name}_figure2_angle_calculation.pdf"

                        visualise_angle_calculation_figure2(
                            complexity_values, hybrid_values,
                            output_path=png_file, title=full_title,
                            highlight_indices=highlight_indices if highlight_indices else None,
                            y_label=y_axis_label,
                        )
                        visualise_angle_calculation_figure2(
                            complexity_values, hybrid_values,
                            output_path=pdf_file, title=full_title,
                            highlight_indices=highlight_indices if highlight_indices else None,
                            y_label=y_axis_label,
                        )

                        mckp_idx = find_minimal_complexity_knee_point(
                            complexity_values, hybrid_values, use_clustering=True
                        )

                        fig4_png_file = vis_dir / f"{base_name}_figure4_clustering_threshold.png"
                        fig4_pdf_file = vis_dir / f"{base_name}_figure4_clustering_threshold.pdf"

                        visualise_figure4_clustering_threshold(
                            complexity_values, hybrid_values,
                            selected_idx=mckp_idx,
                            output_path=fig4_png_file,
                            title=f'{plot_title} - Clustering Threshold\n{subtitle}',
                            y_label=y_axis_label,
                        )
                        visualise_figure4_clustering_threshold(
                            complexity_values, hybrid_values,
                            selected_idx=mckp_idx,
                            output_path=fig4_pdf_file,
                            title=f'{plot_title} - Clustering Threshold\n{subtitle}',
                            y_label=y_axis_label,
                        )

                        if len([f for f in vis_dir.glob("*figure2*.png")]) <= 5:
                            print(f"      Saved Figure 2 plot: {base_name}_figure2_angle_calculation.png/pdf")
                            print(f"      Saved Figure 4 plot: {base_name}_figure4_clustering_threshold.png/pdf")
            else:
                best_idx = valid_run_data[hybrid_col].idxmin()
                selection_method = 'minimum_hybrid_score'
                difficulty = np.nan

            best_row = run_data.loc[best_idx]

            def safe_get(row, key, default=np.nan):
                return row[key] if key in row.index else default

            selection_rows.append({
                'Dataset': dataset_name,
                'Run': run_id,
                'Method': method,
                'Lambda1': lambda1,
                'Lambda2': lambda2,
                'Lambda3': lambda3,
                'Metric': base_metric,
                'Selection_Method': selection_method,
                'Difficulty_Score': difficulty,
                'Selected_Equation_Index': safe_get(best_row, 'Equation_Index'),
                'Selected_Complexity': safe_get(best_row, 'Complexity'),
                'Hybrid_Score': safe_get(best_row, hybrid_col),
                'MSE_Test_Interpolation': safe_get(best_row, 'MSE_Test_Interpolation'),
                'MSE_Test_Extrapolation': safe_get(best_row, 'MSE_Test_Extrapolation'),
            })

    if selection_rows:
        return pd.DataFrame(selection_rows)
    return pd.DataFrame()


def process_csv_file(csv_file: Path, output_base_dir: Path,
                     dataset_name: str = None,
                     use_systematic_exploration: bool = False,
                     lambda1_step: float = 0.1,
                     lambda2_step: float = 0.1,
                     lambda3_step: float = 0.1,
                     num_points: int = None,
                     save_detailed_results: bool = True,
                     use_ackp: bool = False,
                     difficulty_threshold: float = 0.6,
                     save_ackp_visualisations: bool = False,
                     clamp_percentile: float = 70.0,
                     absolute_max_cap: float = 1000.0,
                     constant_clamp_value: float = 10.0) -> bool:
    print(f"\nProcessing {csv_file.parent.name}/{csv_file.name}")

    if dataset_name is None:
        dataset_name = csv_file.parent.name
    output_dir = output_base_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_csv(csv_file)
        print(f"  loaded {len(df)} rows from {df['Run'].nunique()} runs")
    except Exception as e:
        print(f"  failed to read {csv_file.name}: {e}")
        return False

    required_cols = BASE_METRICS + [
        "Extrapolation_Divergence",
        "Sensitivity_Interpolation_SVP",
        "Sensitivity_Extrapolation_SVP",
        "Sensitivity_Interpolation_MVP",
        "Sensitivity_Extrapolation_MVP",
        "Run",
        "Equation_Index",
        "MSE_Test_Interpolation",
        "MSE_Test_Extrapolation",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  Missing required columns: {missing_cols}")
        return False

    # Complexity is optional; without it ACKP cannot run.
    if 'Complexity' not in df.columns:
        print(f"  Warning: 'Complexity' column not found in {csv_file.name}")
        print(f"    Model selection analysis will have NaN values for complexity")
    else:
        print(f"  Found 'Complexity' column: {df['Complexity'].notna().sum()} non-null values")

    print(f"  Creating histograms (before clamping)...")
    create_histograms(df, output_dir, dataset_name, stage='before_clamping')

    print(f"  Clamping new metrics with constant={constant_clamp_value} per Pareto front...")
    df_clamped, clamp_stats = clamp_metrics_per_pareto_front(
        df,
        csv_file=csv_file,
        percentile_threshold=clamp_percentile,
        absolute_max_cap=absolute_max_cap,
        constant_clamp_value=constant_clamp_value,
    )

    if clamp_stats:
        clamp_stats_df = pd.DataFrame(clamp_stats)
        clamp_stats_file = output_dir / f"{dataset_name}_clamping_thresholds_per_pareto_front.csv"
        clamp_stats_df.to_csv(clamp_stats_file, index=False)
        print(f"  Saved clamping thresholds to {clamp_stats_file.name}")

        total_clamped = clamp_stats_df['N_Clamped_Upper'].sum()
        total_values = clamp_stats_df['N_Valid'].sum()
        if total_clamped > 0:
            pct_clamped = (total_clamped / total_values * 100) if total_values > 0 else 0
            print(f"  Clamped {total_clamped}/{total_values} values ({pct_clamped:.1f}%) to {constant_clamp_value}")
        else:
            print(f"  All values within the clamp threshold (max clamp = {constant_clamp_value})")

    print(f"  Creating histograms (after clamping)...")
    create_histograms(df_clamped, output_dir, dataset_name, stage='after_clamping')

    print(f"  Normalising all metrics to [0, 1] (min-max)...")
    df_normalised, range_stats = normalise_metrics_per_problem(df_clamped)

    if range_stats:
        range_stats_df = pd.DataFrame(range_stats)
        range_stats_file = output_dir / f"{dataset_name}_normalisation_statistics.csv"
        range_stats_df.to_csv(range_stats_file, index=False)
        print(f"  Saved normalisation statistics to {range_stats_file.name}")

        print(f"  Post-normalisation ranges:")
        for _, row in range_stats_df.iterrows():
            metric = row['Metric']
            post_min = row['Post_Norm_Min']
            post_max = row['Post_Norm_Max']
            in_range = (post_min >= 0.0 and post_max <= 1.0) or (np.isnan(post_min) and np.isnan(post_max))
            tag = "OK" if in_range else "!!"
            print(f"    [{tag}] {metric}: [{post_min:.6f}, {post_max:.6f}]")

        base_metrics_normalised = all(
            metric in range_stats_df['Metric'].values
            for metric in BASE_METRICS
        )
        if base_metrics_normalised:
            print(f"  All base metrics (AIC, MDL, BIC, PSM, MSE) normalised to [0, 1]")
        else:
            missing = [m for m in BASE_METRICS if m not in range_stats_df['Metric'].values]
            print(f"  WARNING: some base metrics not normalised: {missing}")

    print(f"  Creating histograms (after normalisation)...")
    create_histograms(df_normalised, output_dir, dataset_name, stage='after_normalisation')

    clamped_file = output_dir / f"{dataset_name}_clamped_data_{int(clamp_percentile)}th_percentile.csv"
    df_clamped.to_csv(clamped_file, index=False)

    normalised_file = output_dir / f"{dataset_name}_normalised_data.csv"
    df_normalised.to_csv(normalised_file, index=False)
    print(f"  Saved clamped and normalised data")

    key_metrics = ['MSE_Train', 'Extrapolation_Divergence',
                   'Sensitivity_Interpolation_SVP', 'Sensitivity_Extrapolation_SVP']
    for metric in key_metrics:
        metric_stats = [s for s in range_stats if s['Metric'] == metric]
        if metric_stats:
            stat = metric_stats[0]
            print(f"    {metric}: range=[{stat['Post_Norm_Min']:.3f}, {stat['Post_Norm_Max']:.3f}], "
                  f"span={stat['Post_Norm_Range']:.3f}")

    df = df_normalised

    if use_systematic_exploration:
        expected_count = calculate_combination_count(
            lambda1_start=0.0, lambda1_end=1.0, lambda1_step=lambda1_step,
            lambda2_start=0.0, lambda2_end=1.0, lambda2_step=lambda2_step,
            lambda3_start=0.0, lambda3_end=1.0, lambda3_step=lambda3_step,
            num_points=num_points,
        )
        if num_points is not None:
            print(f"  systematic exploration: ~{expected_count:,} combinations "
                  f"(linspace 0..1.0, {num_points} per lambda)")
        else:
            print(f"  systematic exploration: ~{expected_count:,} combinations "
                  f"(steps {lambda1_step}/{lambda2_step}/{lambda3_step})")
        if expected_count > 10000:
            print(f"    Note: lots of combinations — consider --no-detailed-results.")

        weight_combinations = generate_systematic_lambda_combinations(
            lambda1_start=0.0, lambda1_end=1.0, lambda1_step=lambda1_step,
            lambda2_start=0.0, lambda2_end=1.0, lambda2_step=lambda2_step,
            lambda3_start=0.0, lambda3_end=1.0, lambda3_step=lambda3_step,
            num_points=num_points,
        )
        total_combinations = len(weight_combinations)
        print(f"  generated {total_combinations:,} lambda combinations")
    else:
        weight_combinations = WEIGHT_COMBINATIONS
        print(f"  using fixed weight combinations: {len(weight_combinations)}")

    files_created = []
    all_selection_analyses = []

    best_lambda_combos = set()
    if save_ackp_visualisations:
        best_lambda_combos = _load_best_lambda_combos(output_base_dir, dataset_name)
        if best_lambda_combos:
            print(f"  ACKP visualisations: only plotting {len(best_lambda_combos)} best-lambda combo(s) from reports")
            for method_name, lam in sorted(best_lambda_combos):
                lambda_str = f"l{lam[0]:.1f}_{lam[1]:.1f}_{lam[2]:.1f}"
                print(f"    {method_name}: lambda={lam}  ->  {dataset_name}_run*_{method_name}_Hybrid_*_{lambda_str}_ackp_figure{{2,4}}_*.{{png,pdf}}")
        else:
            print(f"  ACKP visualisations: no report files found — skipping all plots")
            print(f"    (run the analysis first to generate report_best_per_dataset_*.csv)")
            save_ackp_visualisations = False

    for method in ['SVP', 'MVP']:
        print(f"  method: {method}")

        extrap_div_col = "Extrapolation_Divergence"
        sens_interp_col = f"Sensitivity_Interpolation_{method}"
        sens_extrap_col = f"Sensitivity_Extrapolation_{method}"

        for idx, (lambda1, lambda2, lambda3) in enumerate(weight_combinations, 1):
            results = pd.DataFrame()
            results['Run'] = df['Run']
            results['Equation_Index'] = df['Equation_Index']

            for base_metric in BASE_METRICS:
                hybrid_col_name = f"Hybrid_{base_metric}"
                results[hybrid_col_name] = calculate_hybrid_score(
                    df[base_metric],
                    df[extrap_div_col],
                    df[sens_interp_col],
                    df[sens_extrap_col],
                    lambda1, lambda2, lambda3,
                )

            results['Extrapolation_Divergence'] = df[extrap_div_col]
            results['Sensitivity_Interpolation'] = df[sens_interp_col]
            results['Sensitivity_Extrapolation'] = df[sens_extrap_col]
            results['Lambda1'] = lambda1
            results['Lambda2'] = lambda2
            results['Lambda3'] = lambda3

            if 'Complexity' in df.columns:
                results['Complexity'] = df['Complexity'].values
            if 'MSE_Test_Interpolation' in df.columns:
                results['MSE_Test_Interpolation'] = df['MSE_Test_Interpolation'].values
            if 'MSE_Test_Extrapolation' in df.columns:
                results['MSE_Test_Extrapolation'] = df['MSE_Test_Extrapolation'].values

            weight_str = f"w{lambda1}_{lambda2}_{lambda3}"

            should_save_vis = False
            if save_ackp_visualisations and best_lambda_combos:
                current_lambda = (round(lambda1, 4), round(lambda2, 4), round(lambda3, 4))
                should_save_vis = (method, current_lambda) in best_lambda_combos

            selection_analysis = analyze_model_selection(
                df, results, method, lambda1, lambda2, lambda3, dataset_name,
                use_ackp=use_ackp,
                difficulty_threshold=difficulty_threshold,
                save_visualisations=should_save_vis,
                output_dir=output_dir,
            )
            if not selection_analysis.empty:
                all_selection_analyses.append(selection_analysis)

            if save_detailed_results:
                output_file = output_dir / f"hybrid_scores_{method}_{weight_str}.csv"
                results.to_csv(output_file, index=False)
                files_created.append(output_file.name)

            # Spearman correlation between hybrid scores and test MSE, per run
            correlation_rows = []
            for run_id in sorted(results['Run'].unique()):
                mask = results['Run'] == run_id
                if mask.sum() < 2:
                    continue

                test_interp = results.loc[mask, 'MSE_Test_Interpolation']
                for base_metric in BASE_METRICS:
                    hybrid_col_name = f"Hybrid_{base_metric}"
                    if hybrid_col_name not in results.columns:
                        continue
                    hybrid_values = results.loc[mask, hybrid_col_name]
                    if hybrid_values.nunique() <= 1 or test_interp.nunique() <= 1:
                        corr_val = float('nan')
                    else:
                        corr_val = hybrid_values.corr(test_interp, method='spearman')
                    correlation_rows.append({
                        'Dataset': dataset_name,
                        'Run': run_id,
                        'Method': method,
                        'Weight': weight_str,
                        'Metric': base_metric,
                        'Test_Metric': 'MSE_Test_Interpolation',
                        'Spearman': corr_val,
                    })

                test_extrap = results.loc[mask, 'MSE_Test_Extrapolation']
                for base_metric in BASE_METRICS:
                    hybrid_col_name = f"Hybrid_{base_metric}"
                    if hybrid_col_name not in results.columns:
                        continue
                    hybrid_values = results.loc[mask, hybrid_col_name]
                    if hybrid_values.nunique() <= 1 or test_extrap.nunique() <= 1:
                        corr_val = float('nan')
                    else:
                        corr_val = hybrid_values.corr(test_extrap, method='spearman')
                    correlation_rows.append({
                        'Dataset': dataset_name,
                        'Run': run_id,
                        'Method': method,
                        'Weight': weight_str,
                        'Metric': base_metric,
                        'Test_Metric': 'MSE_Test_Extrapolation',
                        'Spearman': corr_val,
                    })

            if correlation_rows and save_detailed_results:
                corr_df = pd.DataFrame(correlation_rows)
                corr_file = output_dir / f"correlations_{method}_{weight_str}.csv"
                corr_df.to_csv(corr_file, index=False)

            if save_detailed_results or idx % max(1, len(weight_combinations) // 10) == 0:
                desc = ""
                if (lambda1, lambda2, lambda3) == (0.0, 0.0, 0.0):
                    desc = " (baseline)"
                elif (lambda1, lambda2, lambda3) == (1.0, 1.0, 1.0):
                    desc = " (all penalties)"
                if save_detailed_results:
                    print(f"    wrote hybrid_scores_{method}_{weight_str}.csv{desc}")
                else:
                    print(f"    processed lambda=({lambda1:.2f}, {lambda2:.2f}, {lambda3:.2f}) "
                          f"[{idx}/{len(weight_combinations)}]{desc}")

    if all_selection_analyses:
        selection_summary = pd.concat(all_selection_analyses, ignore_index=True)
        selection_file = output_dir / f"{dataset_name}_model_selection_analysis.csv"
        selection_summary.to_csv(selection_file, index=False)
        print(f"  Saved model selection analysis to {selection_file.name}")

        if use_ackp:
            ackp_selections = selection_summary[
                selection_summary['Selection_Method'].str.contains('knee|mckp', case=False, na=False)
            ]
            baseline_selections = selection_summary[
                selection_summary['Selection_Method'].str.contains('minimum_hybrid', case=False, na=False)
            ]
            print(f"  ACKP selections: {len(ackp_selections)}, baseline fallbacks: {len(baseline_selections)}")

    if save_ackp_visualisations and use_ackp:
        vis_dir = output_dir / 'knee_point_visualisations'
        if vis_dir.exists():
            num_plots = len(list(vis_dir.glob("*.png")))
            if num_plots > 0:
                print(f"  saved {num_plots} knee point visualisation(s) to {vis_dir.name}/")
            else:
                print(f"  warning: visualisation directory exists but no plots were written")
        else:
            print(f"  warning: visualisation directory not created (ACKP may not have run)")

    print(f"  generated {len(files_created)} CSV files")
    return True


# Base metrics used for hybrid score construction.
BASE_METRICS = [
    "MSE_Train",
    "AIC_Train",
    "BIC_Train",
    "MDL_Train",
    "PSM_Train",
]

INTERPOLATION_HYBRID_COLUMNS = [f"Hybrid_{m}" for m in BASE_METRICS if not m.endswith("_Extrapolation")]
EXTRAPOLATION_HYBRID_COLUMNS = [f"Hybrid_{m}" for m in BASE_METRICS if m.endswith("_Extrapolation")]


def calculate_combination_count(lambda1_start=0.0, lambda1_end=1.0, lambda1_step=0.1,
                                lambda2_start=0.0, lambda2_end=1.0, lambda2_step=0.1,
                                lambda3_start=0.0, lambda3_end=1.0, lambda3_step=0.1,
                                num_points=None):
    """Count how many lambda combinations the current parameters will produce."""
    if num_points is not None:
        return num_points ** 3

    lambda1_count = int((lambda1_end - lambda1_start) / lambda1_step) + 1
    lambda2_count = int((lambda2_end - lambda2_start) / lambda2_step) + 1
    lambda3_count = int((lambda3_end - lambda3_start) / lambda3_step) + 1
    return lambda1_count * lambda2_count * lambda3_count


def generate_systematic_lambda_combinations(lambda1_start=0.0, lambda1_end=1.0, lambda1_step=0.1,
                                            lambda2_start=0.0, lambda2_end=1.0, lambda2_step=0.1,
                                            lambda3_start=0.0, lambda3_end=1.0, lambda3_step=0.1,
                                            num_points=None):
    """Enumerate every lambda triple from the grid."""
    combinations = []

    if num_points is not None:
        lambda1_values = np.round(np.linspace(lambda1_start, lambda1_end, num_points), decimals=2)
        lambda2_values = np.round(np.linspace(lambda2_start, lambda2_end, num_points), decimals=2)
        lambda3_values = np.round(np.linspace(lambda3_start, lambda3_end, num_points), decimals=2)
    else:
        lambda1_values = np.round(np.arange(lambda1_start, lambda1_end + lambda1_step / 2, lambda1_step), decimals=2)
        lambda2_values = np.round(np.arange(lambda2_start, lambda2_end + lambda2_step / 2, lambda2_step), decimals=2)
        lambda3_values = np.round(np.arange(lambda3_start, lambda3_end + lambda3_step / 2, lambda3_step), decimals=2)

    for l1 in lambda1_values:
        for l2 in lambda2_values:
            for l3 in lambda3_values:
                combinations.append((
                    float(np.round(l1, decimals=2)),
                    float(np.round(l2, decimals=2)),
                    float(np.round(l3, decimals=2)),
                ))

    return combinations


# Default lambda grid: 11 evenly spaced values per axis on [0, 1].
# lambda1 -> Extrapolation_Divergence penalty
# lambda2 -> Sensitivity_Interpolation penalty
# lambda3 -> Sensitivity_Extrapolation penalty
lambda1_values = np.round(np.linspace(0.0, 1.0, 11), decimals=2)
lambda2_values = np.round(np.linspace(0.0, 1.0, 11), decimals=2)
lambda3_values = np.round(np.linspace(0.0, 1.0, 11), decimals=2)

WEIGHT_COMBINATIONS = [
    (
        float(np.round(l1, decimals=2)),
        float(np.round(l2, decimals=2)),
        float(np.round(l3, decimals=2)),
    )
    for l1, l2, l3 in product(lambda1_values, lambda2_values, lambda3_values)
]


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


def _process_csv_file_wrapper(args_tuple):
    """Unpack the argument tuple from the worker pool and call process_csv_file."""
    (csv_file, output_base_dir, dataset_name, use_systematic_exploration,
     lambda1_step, lambda2_step, lambda3_step, num_points, save_detailed_results,
     use_ackp, difficulty_threshold, save_ackp_visualisations,
     clamp_percentile, absolute_max_cap, constant_clamp_value) = args_tuple
    return process_csv_file(
        csv_file, output_base_dir, dataset_name=dataset_name,
        use_systematic_exploration=use_systematic_exploration,
        lambda1_step=lambda1_step, lambda2_step=lambda2_step, lambda3_step=lambda3_step,
        num_points=num_points, save_detailed_results=save_detailed_results,
        use_ackp=use_ackp,
        difficulty_threshold=difficulty_threshold,
        save_ackp_visualisations=save_ackp_visualisations,
        clamp_percentile=clamp_percentile,
        absolute_max_cap=absolute_max_cap,
        constant_clamp_value=constant_clamp_value,
    )


def find_and_process_all(root_dir, dataset_name=None, use_systematic_exploration=False,
                         lambda1_step=0.1, lambda2_step=0.1, lambda3_step=0.1,
                         num_points=None, save_detailed_results=True, num_workers=None,
                         use_ackp=False, difficulty_threshold=0.6, save_ackp_visualisations=False,
                         clamp_percentile=70.0, absolute_max_cap=1000.0, constant_clamp_value=10.0):
    """Discover dataset CSVs under root_dir and compute hybrid scores for each."""
    root_path = Path(root_dir).resolve()

    if not root_path.exists():
        print(f"directory not found: {root_dir} (resolved to {root_path})")
        return

    if not root_path.is_dir():
        print(f"path is not a directory: {root_dir}")
        return

    datasets_to_process = [dataset_name] if dataset_name else DATASET_FOLDERS

    if dataset_name and dataset_name not in DATASET_FOLDERS:
        print(f"Error: dataset '{dataset_name}' not found in DATASET_FOLDERS")
        print(f"Available datasets: {', '.join(DATASET_FOLDERS[:5])}...")
        return

    csv_files_with_names = []
    missing_folders = []
    for folder_name in datasets_to_process:
        folder_path = root_path / folder_name
        csv_file = root_path.parent / folder_name / "gp_model_selection_criteria_combined.csv"

        if csv_file.exists():
            csv_files_with_names.append((csv_file, folder_name))
        elif folder_path.exists():
            print(f"  warning: '{folder_name}' exists but is missing gp_model_selection_criteria_combined.csv")
        else:
            missing_folders.append(folder_name)

    if missing_folders:
        summary = ', '.join(missing_folders[:5])
        print(f"  missing {len(missing_folders)} folder(s): {summary}")
        if len(missing_folders) > 5:
            print(f"    ... plus {len(missing_folders) - 5} more")

    if not csv_files_with_names:
        print("no gp_model_selection_criteria_combined.csv files found")
        return

    hybrid_results_dir = root_path / "hybrid_results"
    hybrid_results_dir.mkdir(exist_ok=True)
    print(f"Output directory: {hybrid_results_dir}")

    if num_workers is None:
        import os
        num_workers = min(len(csv_files_with_names), os.cpu_count() or 1)

    if dataset_name:
        print(f"Processing dataset: {dataset_name}")
    else:
        print(
            f"Starting batch: {len(datasets_to_process)} datasets, "
            f"{len(csv_files_with_names)} files queued, "
            f"{len(BASE_METRICS)} metrics, "
            f"{len(WEIGHT_COMBINATIONS)} weight combinations"
        )
        if num_workers > 1:
            print(f"Processing {num_workers} datasets in parallel")

    process_args = [
        (csv_file, hybrid_results_dir, exact_dataset_name, use_systematic_exploration,
         lambda1_step, lambda2_step, lambda3_step,
         num_points, save_detailed_results,
         use_ackp, difficulty_threshold, save_ackp_visualisations,
         clamp_percentile, absolute_max_cap, constant_clamp_value)
        for csv_file, exact_dataset_name in sorted(csv_files_with_names)
    ]

    successful = 0
    failed = 0

    if num_workers > 1 and len(csv_files_with_names) > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(_process_csv_file_wrapper, args): (args[0], args[2])
                for args in process_args
            }

            for future in as_completed(future_to_file):
                csv_file, exact_dataset_name = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                        print(f"Completed: {exact_dataset_name}")
                    else:
                        failed += 1
                        print(f"Failed: {exact_dataset_name}")
                except Exception as e:
                    failed += 1
                    print(f"Error processing {exact_dataset_name}: {e}")
    else:
        for args in process_args:
            csv_file, exact_dataset_name = args[0], args[2]
            result = _process_csv_file_wrapper(args)
            if result:
                successful += 1
            else:
                failed += 1

    print(
        "\nBatch complete: "
        f"{successful}/{len(csv_files_with_names)} succeeded, "
        f"{failed} failed"
    )


def main():
    script_dir = Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser(description="Calculate hybrid scores for symbolic regression results")
    parser.add_argument('results_dir', nargs='?', default=str(script_dir),
                        help='Root directory containing dataset result folders')
    parser.add_argument('--systematic-exploration', action='store_true',
                        help='Use systematic lambda exploration')
    parser.add_argument('--lambda1-step', type=float, default=0.1, help='Step size for lambda1')
    parser.add_argument('--lambda2-step', type=float, default=0.1, help='Step size for lambda2')
    parser.add_argument('--lambda3-step', type=float, default=0.1, help='Step size for lambda3')
    parser.add_argument('--num-points', type=int, default=None,
                        help='Number of evenly spaced points per lambda')
    parser.add_argument('--no-detailed-results', action='store_true',
                        help='Skip saving detailed CSV files')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Process a single dataset only')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Parallel workers (default: CPU count)')
    parser.add_argument('--use-ackp', action='store_true',
                        help='Use Adaptive Complexity Knee Point (ACKP) selection')
    parser.add_argument('--difficulty-threshold', type=float, default=0.6,
                        help='Difficulty threshold for ACKP (default 0.6)')
    parser.add_argument('--save-ackp-visualisations', action='store_true',
                        help='Save knee point plots when using ACKP')
    parser.add_argument('--clamp-percentile', type=float, default=70.0,
                        help='(Unused, kept for compatibility)')
    parser.add_argument('--absolute-max-cap', type=float, default=1000.0,
                        help='(Unused, kept for compatibility)')
    parser.add_argument('--constant-clamp-value', type=float, default=10.0,
                        help='Constant upper bound for clamping (default 10.0)')

    args = parser.parse_args()

    if args.use_ackp:
        print(f"Using ACKP selection with difficulty threshold: {args.difficulty_threshold}")
        if args.save_ackp_visualisations:
            print("  Will save knee point visualisations")

    find_and_process_all(
        args.results_dir,
        dataset_name=args.dataset,
        use_systematic_exploration=args.systematic_exploration,
        lambda1_step=args.lambda1_step,
        lambda2_step=args.lambda2_step,
        lambda3_step=args.lambda3_step,
        num_points=args.num_points,
        save_detailed_results=not args.no_detailed_results,
        num_workers=args.num_workers,
        use_ackp=args.use_ackp,
        difficulty_threshold=args.difficulty_threshold,
        save_ackp_visualisations=args.save_ackp_visualisations,
        clamp_percentile=args.clamp_percentile,
        absolute_max_cap=args.absolute_max_cap,
        constant_clamp_value=args.constant_clamp_value,
    )


if __name__ == "__main__":
    main()
