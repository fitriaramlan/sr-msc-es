"""Compare correlation results against the baseline lambda setting (0, 0, 0)."""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def parse_weight_string(weight_str):
    """Extract lambda values from a string such as 'w0.0_0.0_0.0'."""
    parts = weight_str.replace('w', '').split('_')
    if len(parts) == 3:
        try:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            return None
    return None


def is_baseline(weight_str):
    """Return True for the baseline weight string."""
    lambdas = parse_weight_string(weight_str)
    return lambdas == (0.0, 0.0, 0.0) if lambdas else False


def load_correlations_for_dataset(results_dir, dataset_name, method):
    """Load all correlation files for one dataset and method."""
    dataset_dir = results_dir / "hybrid_results" / dataset_name
    if not dataset_dir.exists():
        return pd.DataFrame()
    
    correlation_files = list(dataset_dir.glob(f"correlations_{method}_*.csv"))
    if not correlation_files:
        return pd.DataFrame()
    
    all_correlations = []
    for file_path in correlation_files:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                all_correlations.append(df)
        except Exception:
            pass
    
    if not all_correlations:
        return pd.DataFrame()
    
    return pd.concat(all_correlations, ignore_index=True)


def compare_against_baseline(corr_df, dataset_name, method):
    """Compare each lambda combination with the baseline."""
    if corr_df.empty:
        return pd.DataFrame()
    
    # Split the data into baseline and non-baseline rows.
    baseline_mask = corr_df['Weight'].apply(is_baseline)
    baseline_df = corr_df[baseline_mask].copy()
    comparison_df = corr_df[~baseline_mask].copy()
    
    if baseline_df.empty or comparison_df.empty:
        return pd.DataFrame()
    
    # Work with mean correlations per metric and test target.
    baseline_means = baseline_df.groupby(['Metric', 'Test_Metric'])['Spearman'].mean().reset_index()
    baseline_means.rename(columns={'Spearman': 'Baseline_Correlation'}, inplace=True)
    
    comparison_means = comparison_df.groupby(['Weight', 'Metric', 'Test_Metric'])['Spearman'].mean().reset_index()
    comparison_means.rename(columns={'Spearman': 'Comparison_Correlation'}, inplace=True)
    
    comparison_results = comparison_means.merge(baseline_means, on=['Metric', 'Test_Metric'], how='left')
    comparison_results['Improvement'] = comparison_results['Comparison_Correlation'] - comparison_results['Baseline_Correlation']
    
    # Avoid dividing by zero when the baseline correlation is exactly zero.
    baseline_abs = comparison_results['Baseline_Correlation'].abs()
    comparison_results['Improvement_Percent'] = np.where(
        baseline_abs > 0,
        (comparison_results['Improvement'] / baseline_abs) * 100,
        np.where(comparison_results['Improvement'] > 0, np.inf, -np.inf)
    )
    
    comparison_results['Is_Better'] = comparison_results['Improvement'] > 0
    
    # Keep lambda values explicit for downstream grouping and plotting.
    lambda_values = comparison_results['Weight'].apply(parse_weight_string)
    comparison_results['Lambda1'] = [l[0] if l else np.nan for l in lambda_values]
    comparison_results['Lambda2'] = [l[1] if l else np.nan for l in lambda_values]
    comparison_results['Lambda3'] = [l[2] if l else np.nan for l in lambda_values]
    
    comparison_results['Dataset'] = dataset_name
    comparison_results['Method'] = method
    
    cols = ['Dataset', 'Method', 'Lambda1', 'Lambda2', 'Lambda3', 'Weight',
            'Metric', 'Test_Metric', 'Baseline_Correlation', 'Comparison_Correlation',
            'Improvement', 'Improvement_Percent', 'Is_Better']
    comparison_results = comparison_results[[c for c in cols if c in comparison_results.columns]]
    
    return comparison_results


def create_summary_statistics(comparison_df):
    """Aggregate statistics across datasets and metrics."""
    if comparison_df.empty:
        return pd.DataFrame()
    
    summary = comparison_df.groupby(['Lambda1', 'Lambda2', 'Lambda3']).agg({
        'Improvement': ['mean', 'std', 'count'],
        'Improvement_Percent': ['mean', 'std'],
        'Is_Better': 'sum',
        'Comparison_Correlation': 'mean',
        'Baseline_Correlation': 'mean'
    }).reset_index()
    
    # Flatten column names after aggregation.
    summary.columns = [
        'Lambda1', 'Lambda2', 'Lambda3',
        'Mean_Improvement', 'Std_Improvement', 'N_Comparisons',
        'Mean_Improvement_Percent', 'Std_Improvement_Percent',
        'N_Better_Than_Baseline',
        'Mean_Comparison_Correlation', 'Mean_Baseline_Correlation'
    ]
    
    summary['Percent_Better'] = (summary['N_Better_Than_Baseline'] / summary['N_Comparisons']) * 100
    summary['Overall_Score'] = summary['Mean_Improvement'] * summary['N_Comparisons']
    
    summary = summary.sort_values('Overall_Score', ascending=False).reset_index(drop=True)
    
    return summary


def plot_correlation_improvements(comparison_df, output_dir, dataset_name=None):
    """Create plots for correlation improvements over the baseline."""
    if comparison_df.empty:
        return
    
    if dataset_name:
        plot_df = comparison_df[comparison_df['Dataset'] == dataset_name].copy()
        title_suffix = f" - {dataset_name}"
    else:
        plot_df = comparison_df.copy()
        title_suffix = " - All Datasets"
    
    if plot_df.empty:
        return
    
    # Aggregate by lambda triple before plotting.
    summary = plot_df.groupby(['Lambda1', 'Lambda2', 'Lambda3']).agg({
        'Improvement': 'mean',
        'Is_Better': 'sum',
        'Comparison_Correlation': 'mean'
    }).reset_index()
    summary['N_Better'] = summary['Is_Better']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    top_lambdas = summary.nlargest(15, 'Improvement')
    top_lambdas['Lambda_Str'] = top_lambdas.apply(
        lambda row: f"({row['Lambda1']:.2f},{row['Lambda2']:.2f},{row['Lambda3']:.2f})",
        axis=1
    )
    y_pos = np.arange(len(top_lambdas))
    ax1.barh(y_pos, top_lambdas['Improvement'], color='steelblue', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_lambdas['Lambda_Str'], fontsize=8)
    ax1.set_xlabel('Mean Correlation Improvement')
    ax1.set_title(f'Top 15 Lambda Combinations by Improvement{title_suffix}', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Baseline')
    ax1.legend()
    
    ax2 = axes[0, 1]
    top_better = summary.nlargest(15, 'N_Better')
    top_better['Lambda_Str'] = top_better.apply(
        lambda row: f"({row['Lambda1']:.2f},{row['Lambda2']:.2f},{row['Lambda3']:.2f})",
        axis=1
    )
    y_pos = np.arange(len(top_better))
    ax2.barh(y_pos, top_better['N_Better'], color='coral', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_better['Lambda_Str'], fontsize=8)
    ax2.set_xlabel('Number of Comparisons Better Than Baseline')
    ax2.set_title(f'Top 15 by Number of Better Comparisons{title_suffix}', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.hist(plot_df['Improvement'].dropna(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Baseline (0)')
    ax3.axvline(plot_df['Improvement'].mean(), color='green', linestyle='--', linewidth=2, label='Mean')
    ax3.set_xlabel('Correlation Improvement')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Distribution of Correlation Improvements{title_suffix}', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    ax4 = axes[1, 1]
    scatter = ax4.scatter(summary['Lambda1'], summary['Lambda2'], 
                         c=summary['Improvement'], s=100, alpha=0.6,
                         cmap='RdYlGn', vmin=-0.1, vmax=0.1)
    ax4.set_xlabel('Lambda1')
    ax4.set_ylabel('Lambda2')
    ax4.set_title(f'Lambda1 vs Lambda2 (coloured by Improvement){title_suffix}', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax4, label='Mean Improvement')
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if dataset_name:
        output_file = output_dir / f"correlation_improvements_{dataset_name}.png"
    else:
        output_file = output_dir / "correlation_improvements_all_datasets.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file.name}")


def main():
    script_dir = Path(__file__).parent.absolute()
    
    parser = argparse.ArgumentParser(description="Compare correlations vs baseline (0,0,0)")
    parser.add_argument('results_dir', nargs='?', default=str(script_dir),
                       help='Root directory with hybrid_results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: results_dir/correlation_comparisons)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset (default: all)')
    parser.add_argument('--method', type=str, choices=['SVP', 'MVP', 'both'],
                       default='both', help='SVP, MVP, or both')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create plots')
    
    args = parser.parse_args()
    
    results_path = Path(args.results_dir).resolve()
    hybrid_results_dir = results_path / "hybrid_results"
    
    if not hybrid_results_dir.exists():
        print(f"Error: hybrid_results directory not found at {hybrid_results_dir}")
        return
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path / "correlation_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Comparing Correlations Against Baseline (0,0,0)")
    print("="*80)
    print(f"\nResults directory: {results_path}")
    print(f"Output directory: {output_dir}\n")
    
    if args.dataset:
        dataset_names = [args.dataset]
    else:
        dataset_names = sorted([d.name for d in hybrid_results_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(dataset_names)} dataset(s) to process\n")
    
    methods_to_process = ['SVP', 'MVP'] if args.method == 'both' else [args.method]
    all_comparisons = []
    
    for dataset_name in dataset_names:
        print(f"Processing {dataset_name}...")
        
        for method in methods_to_process:
            corr_df = load_correlations_for_dataset(results_path, dataset_name, method)
            
            if corr_df.empty:
                print(f"  No correlation data found for {dataset_name} - {method}")
                continue
            
            comparison_df = compare_against_baseline(corr_df, dataset_name, method)
            
            if comparison_df.empty:
                print(f"  No comparison results for {dataset_name} - {method}")
                continue
            
            output_file = output_dir / f"correlation_comparison_{dataset_name}_{method}.csv"
            comparison_df.to_csv(output_file, index=False)
            print(f"  Saved: {output_file.name}")
            
            all_comparisons.append(comparison_df)
    
    if all_comparisons:
        print("\n" + "="*80)
        print("Creating Summary Statistics")
        print("="*80 + "\n")
        
        combined_df = pd.concat(all_comparisons, ignore_index=True)
        summary_df = create_summary_statistics(combined_df)
        
        summary_file = output_dir / "correlation_comparison_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary: {summary_file.name}")
        print(f"  Total lambda combinations: {len(summary_df)}")
        best = summary_df.iloc[0]
        print(f"  Best: ({best['Lambda1']:.3f}, {best['Lambda2']:.3f}, {best['Lambda3']:.3f})")
        print(f"    Mean improvement: {best['Mean_Improvement']:.6f}")
        print(f"    Percent better: {best['Percent_Better']:.1f}%")
        
        combined_file = output_dir / "correlation_comparison_all_results.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\nSaved detailed results: {combined_file.name}")
        
        if args.create_plots:
            print("\n" + "="*80)
            print("Creating Plots")
            print("="*80 + "\n")
            plot_correlation_improvements(combined_df, output_dir)
            for dataset_name in dataset_names:
                plot_correlation_improvements(combined_df, output_dir, dataset_name=dataset_name)
    
    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

