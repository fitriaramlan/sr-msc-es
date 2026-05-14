import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

# Legend size aligned with m1_evaluation figures
PAPER_LEGEND_FONTSIZE = 14


def report_nan_statistics(df, columns, output_dir):
    # Writes nan_statistics_report.txt and prints a short summary
    nan_stats = {}
    total_rows = len(df)
    
    print("\n" + "="*60)
    print("NaN STATISTICS REPORT")
    print("="*60)
    print(f"Total rows in dataset: {total_rows}")
    
    for col in columns:
        n_nan = df[col].isna().sum()
        pct_nan = (n_nan / total_rows) * 100 if total_rows > 0 else 0
        nan_stats[col] = {'count': n_nan, 'percentage': pct_nan}
        
        if n_nan > 0:
            print(f"  {col}: {n_nan} NaN values ({pct_nan:.2f}%)")
    
    # rows with any NaN in selected columns
    rows_with_nan = df[columns].isna().any(axis=1).sum()
    pct_rows_nan = (rows_with_nan / total_rows) * 100 if total_rows > 0 else 0
    
    print(f"\nRows with ANY NaN: {rows_with_nan} ({pct_rows_nan:.2f}%)")
    print(f"Rows without NaN: {total_rows - rows_with_nan} ({100 - pct_rows_nan:.2f}%)")
    print("="*60 + "\n")
    
    report_file = os.path.join(output_dir, "nan_statistics_report.txt")
    with open(report_file, 'w') as f:
        f.write("NaN STATISTICS REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Total rows in dataset: {total_rows}\n\n")
        for col, stats in nan_stats.items():
            if stats['count'] > 0:
                f.write(f"{col}: {stats['count']} NaN values ({stats['percentage']:.2f}%)\n")
        f.write(f"\nRows with ANY NaN: {rows_with_nan} ({pct_rows_nan:.2f}%)\n")
        f.write(f"Rows without NaN: {total_rows - rows_with_nan} ({100 - pct_rows_nan:.2f}%)\n")
    
    return nan_stats


def handle_nan_values(df, columns, method=None, output_dir=None):
    # Drop rows with NaN or inf
    if output_dir:
        report_nan_statistics(df, columns, output_dir)

    rows_before = len(df)

    df = df.dropna()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    rows_after = len(df)
    rows_removed = rows_before - rows_after
    print(f"  Rows removed (NaN and inf): {rows_removed}")
    if rows_removed > 0:
        print(f"  Removed {rows_removed} samples ({rows_removed/rows_before*100:.2f}% of dataset)")
    
    print(f"Final dataset size: {len(df)} rows\n")
    
    return df


def compute_bandwidths(data):
    
    n, d = data.shape
    sigma = np.std(data, axis=0).mean()
    
    scott_bandwidth = n ** (-1. / (d + 4)) * sigma
    silverman_bandwidth = (n * (d + 2) / 4) ** (-1. / (d + 4)) * sigma
    
    return scott_bandwidth, silverman_bandwidth


def kde_analysis(input_file, output_dir, shrink=10, nan_method=None):
    # KDE on standardised X; percentile threshold splits inside vs outside; writes CSVs and plots per bandwidth
    os.makedirs(output_dir, exist_ok=True)

    df = np.load(input_file)
    original_size = df.shape[0]
    n_features = df.shape[1] - 1
    columns = [f"feature_{i+1}" for i in range(n_features)] + ["target"]
    df = pd.DataFrame(df, columns=columns)
    
    print("\n" + "="*60)
    print("DATASET LOADING SUMMARY")
    print("="*60)
    print(f"Original dataset size: {original_size} samples")
    print(f"Number of features: {n_features}")
    
    df = handle_nan_values(df, columns, method=nan_method, output_dir=output_dir)
    
    final_size = df.shape[0]
    print(f"Final dataset size after NaN handling: {final_size} samples")
    if final_size < original_size:
        print(f"Samples removed: {original_size - final_size} ({((original_size - final_size) / original_size * 100):.2f}%)")
    print("="*60 + "\n")

    data_original = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
    feature_names = df.columns[:-1]

    # KDE needs scaled features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_original)

    # quick look: dim 0 vs y (full sample)
    if n_features >= 1:
        fig0, ax0 = plt.subplots(figsize=(7, 5))
        ax0.scatter(
            data_scaled[:, 0],
            target,
            c="0.35",
            s=12,
            alpha=0.65,
            edgecolors="none",
        )
        ax0.set_xlabel(f"{feature_names[0]} (standardised)")
        ax0.set_ylabel("target")
        ax0.set_title("Full data: feature 0 vs target (before KDE inside/outside split)")
        ax0.grid(True, alpha=0.3)
        fig0.tight_layout()
        fig0.savefig(os.path.join(output_dir, "plot_feature1_vs_target_before_kde_split.png"))
        fig0.savefig(os.path.join(output_dir, "plot_feature1_vs_target_before_kde_split.pdf"))
        plt.close(fig0)

    # fixed grid plus Scott/Silverman
    scott_bandwidth, silverman_bandwidth = compute_bandwidths(data_scaled)
    bandwidths = [0.1, 0.3, 0.5, 1.0, scott_bandwidth, silverman_bandwidth]

    for bandwidth in bandwidths:
        print(f"\n--- Processing Bandwidth: {bandwidth:.2f} ---")
        plt.figure(figsize=(7, 5))
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data_scaled)
        densities_data = np.exp(kde.score_samples(data_scaled))

        threshold = np.percentile(densities_data, shrink)
        inside_mask = densities_data >= threshold
        outside_mask = densities_data < threshold

        # empty outside band (tiny h or nearly constant density)
        if np.sum(outside_mask) == 0:
            print(f"  Warning: no outside points; taking lowest-density tail")
            bottom_n = max(10, len(data_scaled) // 10)
            forced_indices = np.argsort(densities_data)[:bottom_n]
            outside_mask[:] = False
            outside_mask[forced_indices] = True
            inside_mask = ~outside_mask

        inside_points = pd.DataFrame(data_scaled[inside_mask], columns=feature_names)
        inside_points["target"] = target[inside_mask]
        outside_points = pd.DataFrame(data_scaled[outside_mask], columns=feature_names)
        outside_points["target"] = target[outside_mask]
        
        inside_has_nan = inside_points.isna().any().any()
        outside_has_nan = outside_points.isna().any().any()
        
        if inside_has_nan:
            print(f"  [WARNING] NaN detected in inside_points!")
            nan_cols = inside_points.columns[inside_points.isna().any()].tolist()
            print(f"  Columns with NaN: {nan_cols}")
            
            # Save NaN details to CSV
            nan_csv_rows = []
            for col in nan_cols:
                nan_rows = inside_points[inside_points[col].isna()].index.tolist()
                print(f"    {col}: NaN at rows {nan_rows}")
                for row_idx in nan_rows:
                    nan_csv_rows.append({
                        'data_name': f'inside_points_bw_{bandwidth:.2f}',
                        'row_index': int(row_idx),
                        'column_name': col,
                        'column_index': None,
                        'value': None
                    })
            
            if nan_csv_rows:
                csv_file = os.path.join(output_dir, f"nan_detection_inside_points_bw_{bandwidth:.2f}.csv")
                pd.DataFrame(nan_csv_rows).to_csv(csv_file, index=False)
                print(f"  Saved NaN details to: {csv_file}")
        
        if outside_has_nan:
            print(f"  [WARNING] NaN detected in outside_points!")
            nan_cols = outside_points.columns[outside_points.isna().any()].tolist()
            print(f"  Columns with NaN: {nan_cols}")
            
            # Save NaN details to CSV
            nan_csv_rows = []
            for col in nan_cols:
                nan_rows = outside_points[outside_points[col].isna()].index.tolist()
                print(f"    {col}: NaN at rows {nan_rows}")
                for row_idx in nan_rows:
                    nan_csv_rows.append({
                        'data_name': f'outside_points_bw_{bandwidth:.2f}',
                        'row_index': int(row_idx),
                        'column_name': col,
                        'column_index': None,
                        'value': None
                    })
            
            if nan_csv_rows:
                csv_file = os.path.join(output_dir, f"nan_detection_outside_points_bw_{bandwidth:.2f}.csv")
                pd.DataFrame(nan_csv_rows).to_csv(csv_file, index=False)
                print(f"  Saved NaN details to: {csv_file}")
        
        if not inside_has_nan and not outside_has_nan:
            print(f"  No NaN detected in inside_points or outside_points")

        # Report region sizes
        n_inside = len(inside_points)
        n_outside = len(outside_points)
        total = n_inside + n_outside
        print(f"  Inside region (high density): {n_inside} samples ({n_inside/total*100:.1f}%)")
        print(f"  Outside region (low density): {n_outside} samples ({n_outside/total*100:.1f}%)")

        inside_points.to_csv(os.path.join(output_dir, f"inside_points_bw_{bandwidth:.2f}.csv"), index=False)
        outside_points.to_csv(os.path.join(output_dir, f"outside_points_bw_{bandwidth:.2f}.csv"), index=False)
        print(f"  Saved: inside_points_bw_{bandwidth:.2f}.csv")
        print(f"  Saved: outside_points_bw_{bandwidth:.2f}.csv")

        # (x1, x2) coloured by region
        x1, x2 = data_scaled[:, 0], data_scaled[:, 1]
        plt.scatter(x1[inside_mask], x2[inside_mask], c='blue', s=5, label='Inside', alpha=0.6)
        plt.scatter(x1[outside_mask], x2[outside_mask], c='red', s=5, label='Outside', alpha=0.6)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f"Plot with Bandwidth: {bandwidth:.2f}", fontsize=16)
        plt.legend(fontsize=PAPER_LEGEND_FONTSIZE)
        plt.savefig(os.path.join(output_dir, f"plot_bw_{bandwidth:.2f}.pdf"))
        plt.savefig(os.path.join(output_dir, f"plot_bw_{bandwidth:.2f}.png"))
        plt.show()
