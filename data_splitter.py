import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold


def check_and_print_nan(data, data_name, feature_names=None, output_dir=None):
    # Prints NaN locations; optional CSV/txt under output_dir for debugging splits
    if isinstance(data, pd.DataFrame):
        data_array = data.values
        if feature_names is None:
            feature_names = list(data.columns)
    else:
        data_array = np.asarray(data)
    
    nan_mask = np.isnan(data_array)
    has_nan = np.any(nan_mask)
    
    if has_nan:
        print(f"\n{'='*80}")
        print(f"NaN DETECTED in: {data_name}")
        print(f"{'='*80}")
        print(f"Data shape: {data_array.shape}")
        
        nan_report_rows = []
        nan_csv_rows = []
        
        if len(data_array.shape) == 1:
            # 1D (target)
            nan_indices = np.where(nan_mask)[0]
            nan_count = len(nan_indices)
            nan_pct = (nan_count / len(data_array)) * 100
            
            print(f"NaN count: {nan_count} out of {len(data_array)} ({nan_pct:.2f}%)")
            print(f"NaN at indices: {nan_indices}")
            
            nan_report_rows.append(f"NaN DETECTED in: {data_name}")
            nan_report_rows.append(f"Data shape: {data_array.shape}")
            nan_report_rows.append(f"NaN count: {nan_count} out of {len(data_array)} ({nan_pct:.2f}%)")
            nan_report_rows.append(f"NaN at indices: {list(nan_indices)}")
            
            # one CSV row per NaN
            for idx in nan_indices:
                nan_csv_rows.append({
                    'data_name': data_name,
                    'row_index': idx,
                    'column_name': 'target',
                    'column_index': None,
                    'value': None
                })
        else:
            # 2D (features)
            nan_count_per_col = np.sum(nan_mask, axis=0)
            nan_count_per_row = np.sum(nan_mask, axis=1)
            total_nan = np.sum(nan_mask)
            
            print(f"Total NaN values: {total_nan}")
            print(f"\nNaN per column:")
            
            nan_report_rows.append(f"NaN DETECTED in: {data_name}")
            nan_report_rows.append(f"Data shape: {data_array.shape}")
            nan_report_rows.append(f"Total NaN values: {total_nan}")
            nan_report_rows.append("\nNaN per column:")
            
            for col_idx in range(data_array.shape[1]):
                if nan_count_per_col[col_idx] > 0:
                    col_name = feature_names[col_idx] if feature_names and col_idx < len(feature_names) else f"col_{col_idx}"
                    row_indices = np.where(nan_mask[:, col_idx])[0]
                    print(f"  {col_name}: {nan_count_per_col[col_idx]} NaN values")
                    print(f"    Rows with NaN: {row_indices}")
                    
                    nan_report_rows.append(f"  {col_name}: {nan_count_per_col[col_idx]} NaN values")
                    nan_report_rows.append(f"    Rows with NaN: {list(row_indices)}")
                    
                    # one CSV row per NaN
                    for row_idx in row_indices:
                        nan_csv_rows.append({
                            'data_name': data_name,
                            'row_index': int(row_idx),
                            'column_name': col_name,
                            'column_index': int(col_idx),
                            'value': None
                        })
            
            print(f"\nNaN per row:")
            rows_with_nan = np.where(nan_count_per_row > 0)[0]
            pct_rows_nan = (len(rows_with_nan) / data_array.shape[0]) * 100
            print(f"Rows with ANY NaN: {len(rows_with_nan)} out of {data_array.shape[0]} ({pct_rows_nan:.2f}%)")
            
            nan_report_rows.append(f"\nNaN per row:")
            nan_report_rows.append(f"Rows with ANY NaN: {len(rows_with_nan)} out of {data_array.shape[0]} ({pct_rows_nan:.2f}%)")
            
            for row_idx in rows_with_nan:
                nan_cols = np.where(nan_mask[row_idx, :])[0]
                col_names = [feature_names[c] if feature_names and c < len(feature_names) else f"col_{c}" for c in nan_cols]
                print(f"  Row {row_idx}: NaN in columns {col_names}")
                
                nan_report_rows.append(f"  Row {row_idx}: NaN in columns {col_names}")
        
        print(f"{'='*80}\n")
        
        if output_dir:
            # safe filename from data_name
            safe_name = data_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c in ('_', '-'))
            
            txt_file = os.path.join(output_dir, f"nan_detection_{safe_name}.txt")
            with open(txt_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("NaN DETECTION REPORT\n")
                f.write("="*80 + "\n\n")
                f.write("\n".join(nan_report_rows))
                f.write("\n" + "="*80 + "\n")
            print(f"  Saved NaN report to: {txt_file}")
            
            if nan_csv_rows:
                csv_file = os.path.join(output_dir, f"nan_detection_{safe_name}.csv")
                df_nan = pd.DataFrame(nan_csv_rows)
                df_nan.to_csv(csv_file, index=False)
                print(f"  Saved NaN details to: {csv_file}")
        
        return True
    else:
        print(f"No NaN detected in: {data_name}")
        return False

def _paths(bandwidth, output_dir):
    # inside/outside CSV paths for this bandwidth
    inside_file = os.path.join(output_dir, f"inside_points_bw_{bandwidth:.2f}.csv")
    outside_file = os.path.join(output_dir, f"outside_points_bw_{bandwidth:.2f}.csv")
    return inside_file, outside_file

def split_data(bandwidth, feature_names, output_dir, random_state=42):
    # Train/test inside KDE hull; extrapolation set = outside CSV
    inside_file, outside_file = _paths(bandwidth, output_dir)

    if not os.path.exists(outside_file):
        return None

    outside_df = pd.read_csv(outside_file)
    X_test_extrapolation = outside_df[feature_names].values
    y_test_extrapolation = outside_df['target'].values
    
    check_and_print_nan(X_test_extrapolation, "X_test_extrapolation (outside_df)", feature_names, output_dir)
    check_and_print_nan(y_test_extrapolation, "y_test_extrapolation (outside_df)", None, output_dir)

    if os.path.exists(inside_file):
        inside_df = pd.read_csv(inside_file)
        X_inside = inside_df[feature_names].values
        y_inside = inside_df['target'].values

        check_and_print_nan(X_inside, "X_inside (before split)", feature_names, output_dir)
        check_and_print_nan(y_inside, "y_inside (before split)", None, output_dir)

        X_train, X_test_interpolation, y_train, y_test_interpolation = train_test_split(
            X_inside, y_inside, test_size=0.2, random_state=random_state
        )

        check_and_print_nan(X_train, "X_train (after split)", feature_names, output_dir)
        check_and_print_nan(y_train, "y_train (after split)", None, output_dir)
        check_and_print_nan(X_test_interpolation, "X_test_interpolation (after split)", feature_names, output_dir)
        check_and_print_nan(y_test_interpolation, "y_test_interpolation (after split)", None, output_dir)

        print(f"\nDATA SPLIT SUMMARY (Bandwidth: {bandwidth:.2f}):")
        print(f"  Train set:              {len(X_train)} samples (80% of inside)")
        print(f"  Test Interpolation set: {len(X_test_interpolation)} samples (20% of inside)")
        print(f"  Test Extrapolation set: {len(X_test_extrapolation)} samples (outside region)")
        total = len(X_train) + len(X_test_interpolation) + len(X_test_extrapolation)
        print(f"  Total:                  {total} samples")

        return X_train, y_train, X_test_interpolation, y_test_interpolation, X_test_extrapolation, y_test_extrapolation

    return None

def _load_inside_outside(bandwidth, feature_names, output_dir):
    inside_file, outside_file = _paths(bandwidth, output_dir)
    inside_df = pd.read_csv(inside_file) if os.path.exists(inside_file) else None
    outside_df = pd.read_csv(outside_file) if os.path.exists(outside_file) else None

    # post-load sanity on CSVs
    if inside_df is not None:
        print(f"\nChecking inside_df after loading from CSV (bandwidth={bandwidth:.2f}):")
        if inside_df.isna().any().any():
            print(f"  [WARNING] NaN detected in inside_df!")
            nan_cols = inside_df.columns[inside_df.isna().any()].tolist()
            print(f"  Columns with NaN: {nan_cols}")
            
            # Save NaN details to CSV
            nan_csv_rows = []
            for col in nan_cols:
                nan_rows = inside_df[inside_df[col].isna()].index.tolist()
                print(f"    {col}: NaN at rows {nan_rows}")
                for row_idx in nan_rows:
                    nan_csv_rows.append({
                        'data_name': f'inside_df_loaded_bw_{bandwidth:.2f}',
                        'row_index': int(row_idx),
                        'column_name': col,
                        'column_index': None,
                        'value': None
                    })
            
            if nan_csv_rows and output_dir:
                csv_file = os.path.join(output_dir, f"nan_detection_inside_df_loaded_bw_{bandwidth:.2f}.csv")
                pd.DataFrame(nan_csv_rows).to_csv(csv_file, index=False)
                print(f"  Saved NaN details to: {csv_file}")
        else:
            print(f"  No NaN detected in inside_df")
    
    if outside_df is not None:
        print(f"\nChecking outside_df after loading from CSV (bandwidth={bandwidth:.2f}):")
        if outside_df.isna().any().any():
            print(f"  [WARNING] NaN detected in outside_df!")
            nan_cols = outside_df.columns[outside_df.isna().any()].tolist()
            print(f"  Columns with NaN: {nan_cols}")
            
            # Save NaN details to CSV
            nan_csv_rows = []
            for col in nan_cols:
                nan_rows = outside_df[outside_df[col].isna()].index.tolist()
                print(f"    {col}: NaN at rows {nan_rows}")
                for row_idx in nan_rows:
                    nan_csv_rows.append({
                        'data_name': f'outside_df_loaded_bw_{bandwidth:.2f}',
                        'row_index': int(row_idx),
                        'column_name': col,
                        'column_index': None,
                        'value': None
                    })
            
            if nan_csv_rows and output_dir:
                csv_file = os.path.join(output_dir, f"nan_detection_outside_df_loaded_bw_{bandwidth:.2f}.csv")
                pd.DataFrame(nan_csv_rows).to_csv(csv_file, index=False)
                print(f"  Saved NaN details to: {csv_file}")
        else:
            print(f"  No NaN detected in outside_df")

    X_out = y_out = None
    if outside_df is not None:
        X_out = outside_df[feature_names].values
        y_out = outside_df['target'].values
    return inside_df, X_out, y_out

def split_data_kfold(bandwidth, feature_names, output_dir, n_splits=3, shuffle=True, random_state=42, fold_number=None):
    # 20% inside held out for interpolation test; K-fold on the rest. fold_number: yield one fold only (1-based).
    inside_df, X_out, y_out = _load_inside_outside(bandwidth, feature_names, output_dir)
    if inside_df is None:
        return

    X_in = inside_df[feature_names].values
    y_in = inside_df['target'].values

    check_and_print_nan(X_in, "X_in (before CV split)", feature_names, output_dir)
    check_and_print_nan(y_in, "y_in (before CV split)", None, output_dir)

    # Hold out 20% of inside region for interpolation testing
    X_train_base, X_test_interpolation, y_train_base, y_test_interpolation = train_test_split(
        X_in, y_in, test_size=0.2, random_state=random_state
    )

    check_and_print_nan(X_train_base, "X_train_base (after initial split)", feature_names, output_dir)
    check_and_print_nan(y_train_base, "y_train_base (after initial split)", None, output_dir)
    check_and_print_nan(X_test_interpolation, "X_test_interpolation (after initial split)", feature_names, output_dir)
    check_and_print_nan(y_test_interpolation, "y_test_interpolation (after initial split)", None, output_dir)
    
    # Check outside data if available
    if X_out is not None:
        check_and_print_nan(X_out, "X_out (extrapolation)", feature_names, output_dir)
    if y_out is not None:
        check_and_print_nan(y_out, "y_out (extrapolation)", None, output_dir)

    if fold_number is not None:
        # single fold
        print(f"\nK-FOLD CV DATA SPLIT (Bandwidth: {bandwidth:.2f}, Fold {fold_number}/{n_splits}, seed={random_state}):")
        print(f"  Inside region total:    {len(X_in)} samples")
        print(f"  Test Interpolation set: {len(X_test_interpolation)} samples (20% of inside, held out)")
        print(f"  Train+Val base:         {len(X_train_base)} samples (80% of inside, used for CV)")
        print(f"  Test Extrapolation set: {len(X_out) if X_out is not None else 0} samples (outside region)")
        
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        folds = list(kf.split(X_train_base))
        if fold_number < 1 or fold_number > len(folds):
            return
        
        tr, va = folds[fold_number - 1]
        print(f"\n  Fold {fold_number}/{n_splits}:")
        print(f"    Train: {len(tr)} samples")
        print(f"    Val:   {len(va)} samples")
        
        X_train_fold = X_train_base[tr]
        y_train_fold = y_train_base[tr]
        X_val_fold = X_train_base[va]
        y_val_fold = y_train_base[va]

        check_and_print_nan(X_train_fold, f"X_train_fold (fold {fold_number})", feature_names, output_dir)
        check_and_print_nan(y_train_fold, f"y_train_fold (fold {fold_number})", None, output_dir)
        check_and_print_nan(X_val_fold, f"X_val_fold (fold {fold_number})", feature_names, output_dir)
        check_and_print_nan(y_val_fold, f"y_val_fold (fold {fold_number})", None, output_dir)
        
        yield fold_number, X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_interpolation, y_test_interpolation, X_out, y_out
    else:
        # all folds
        print(f"\nK-FOLD CV DATA SPLIT SUMMARY (Bandwidth: {bandwidth:.2f}, {n_splits} folds):")
        print(f"  Inside region total:    {len(X_in)} samples")
        print(f"  Test Interpolation set: {len(X_test_interpolation)} samples (20% of inside, held out)")
        print(f"  Train+Val base:         {len(X_train_base)} samples (80% of inside, used for CV)")
        print(f"  Test Extrapolation set: {len(X_out) if X_out is not None else 0} samples (outside region)")
        print(f"  Total:                  {len(X_in) + (len(X_out) if X_out is not None else 0)} samples")

        # Apply K-fold to remaining 80%
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for idx, (tr, va) in enumerate(kf.split(X_train_base), start=1):
            print(f"\n  Fold {idx}/{n_splits}:")
            print(f"    Train: {len(tr)} samples")
            print(f"    Val:   {len(va)} samples")
            
            X_train_fold = X_train_base[tr]
            y_train_fold = y_train_base[tr]
            X_val_fold = X_train_base[va]
            y_val_fold = y_train_base[va]

            check_and_print_nan(X_train_fold, f"X_train_fold (fold {idx})", feature_names, output_dir)
            check_and_print_nan(y_train_fold, f"y_train_fold (fold {idx})", None, output_dir)
            check_and_print_nan(X_val_fold, f"X_val_fold (fold {idx})", feature_names, output_dir)
            check_and_print_nan(y_val_fold, f"y_val_fold (fold {idx})", None, output_dir)
            
            yield idx, X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test_interpolation, y_test_interpolation, X_out, y_out
